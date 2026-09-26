"""GPU clocks, power and throttle reasons for the whole run, per phase.

Samples are stored column-wise as integers (one list per field and GPU) so
the raw series compresses to a few hundred kB and later statistics can
re-analyze them. ``analyze`` turns them into per-phase verdicts.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from statistics import median

FIELDS = ("sm_mhz", "mem_mhz", "gr_mhz", "power_dw", "temp_c", "util_pct", "mem_util_pct",
          "reasons", "pcie_gen", "pcie_width", "pstate", "energy_j", "mem_used_mib", "fan_pct")
# NVML clocks event (throttle) reason bits.
REASONS = {
    "gpu_idle": 0x1,
    "applications_clocks": 0x2,
    "sw_power_cap": 0x4,
    "hw_slowdown": 0x8,
    "sync_boost": 0x10,
    "sw_thermal": 0x20,
    "hw_thermal": 0x40,
    "hw_power_brake": 0x80,
    "display_clocks": 0x100,
}
HW_REASONS = REASONS["hw_slowdown"] | REASONS["hw_thermal"] | REASONS["hw_power_brake"]
SMI_QUERY = ("index,clocks.sm,clocks.mem,clocks.gr,power.draw,temperature.gpu,utilization.gpu,"
             "utilization.memory,clocks_event_reasons.active,pcie.link.gen.current,"
             "pcie.link.width.current,pstate,memory.used,fan.speed")


class NvmlSource:
    name = "nvml"

    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self.nvml = pynvml
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        reasons = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
        self.reasons = reasons or pynvml.nvmlDeviceGetCurrentClocksThrottleReasons

    def _get(self, fn, *args):
        try:
            return fn(*args)
        except Exception:  # noqa: BLE001 - unsupported fields stay empty
            return None

    def sample(self) -> list[dict]:
        n = self.nvml
        rows = []
        for handle in self.handles:
            util = self._get(n.nvmlDeviceGetUtilizationRates, handle)
            memory = self._get(n.nvmlDeviceGetMemoryInfo, handle)
            power = self._get(n.nvmlDeviceGetPowerUsage, handle)
            energy = self._get(n.nvmlDeviceGetTotalEnergyConsumption, handle)
            rows.append({
                "sm_mhz": self._get(n.nvmlDeviceGetClockInfo, handle, n.NVML_CLOCK_SM),
                "mem_mhz": self._get(n.nvmlDeviceGetClockInfo, handle, n.NVML_CLOCK_MEM),
                "gr_mhz": self._get(n.nvmlDeviceGetClockInfo, handle, n.NVML_CLOCK_GRAPHICS),
                "power_dw": round(power / 100) if power is not None else None,
                "temp_c": self._get(n.nvmlDeviceGetTemperature, handle, n.NVML_TEMPERATURE_GPU),
                "util_pct": util.gpu if util is not None else None,
                "mem_util_pct": util.memory if util is not None else None,
                "reasons": self._get(self.reasons, handle),
                "pcie_gen": self._get(n.nvmlDeviceGetCurrPcieLinkGeneration, handle),
                "pcie_width": self._get(n.nvmlDeviceGetCurrPcieLinkWidth, handle),
                "pstate": self._get(n.nvmlDeviceGetPerformanceState, handle),
                "energy_j": round(energy / 1000) if energy is not None else None,
                "mem_used_mib": memory.used // 2**20 if memory is not None else None,
                "fan_pct": self._get(n.nvmlDeviceGetFanSpeed, handle),
            })
        return rows


class SmiSource:
    name = "nvidia-smi"

    def __init__(self):
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("neither NVML nor nvidia-smi is available")

    def sample(self) -> list[dict]:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={SMI_QUERY}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=False).stdout
        rows = []
        for line in out.strip().splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < 14:
                continue
            def num(value, scale=1.0):
                try:
                    return round(float(value) * scale)
                except ValueError:
                    return None
            reasons = cells[8]
            rows.append({
                "sm_mhz": num(cells[1]), "mem_mhz": num(cells[2]), "gr_mhz": num(cells[3]),
                "power_dw": num(cells[4], 10), "temp_c": num(cells[5]), "util_pct": num(cells[6]),
                "mem_util_pct": num(cells[7]),
                "reasons": int(reasons, 16) if reasons.startswith("0x") else None,
                "pcie_gen": num(cells[9]), "pcie_width": num(cells[10]),
                "pstate": num(cells[11].lstrip("P")) if cells[11].startswith("P") else None,
                "energy_j": None, "mem_used_mib": num(cells[12]), "fan_pct": num(cells[13]),
            })
        return rows


def open_source():
    try:
        return NvmlSource()
    except Exception:  # noqa: BLE001 - fall back to the CLI
        return SmiSource()


def read_links(bdfs: list[str], sys_root: str = "/sys") -> dict:
    """Current speed and width of PCI links (idle links train down, so sample under load)."""
    links = {}
    for bdf in bdfs:
        base = f"{sys_root}/bus/pci/devices/{bdf}"
        try:
            with open(f"{base}/current_link_speed") as speed, open(f"{base}/current_link_width") as width:
                links[bdf] = f"{speed.read().split()[0]}x{width.read().strip()}"
        except (OSError, IndexError):
            continue
    return links


class Recorder:
    """Background sampler; ``series()`` returns columns per GPU.

    Every ``link_every`` samples it also records the link state of every PCI
    hop in ``link_bdfs`` (root ports and switch ports on the GPU paths).
    """

    def __init__(self, interval: float = 0.5, source=None, link_bdfs: list[str] | None = None,
                 link_every: int = 20, sys_root: str = "/sys"):
        self.interval = interval
        self.source = source or open_source()
        self.t0 = time.time()
        self.times: list[int] = []
        self.columns: list[dict[str, list]] = []
        self.errors = 0
        self.link_bdfs = list(link_bdfs or [])
        self.link_every = link_every
        self.sys_root = sys_root
        self.links: list[dict] = []
        self._count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lil-bench-telemetry", daemon=True)

    def start(self) -> "Recorder":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        next_at = time.monotonic()
        while not self._stop.is_set():
            self.record_once()
            next_at += self.interval
            self._stop.wait(max(0.0, next_at - time.monotonic()))

    def record_once(self) -> None:
        now = time.time()
        try:
            rows = self.source.sample()
        except Exception:  # noqa: BLE001 - a missed sample is not fatal
            self.errors += 1
            return
        if not self.columns:
            self.columns = [{f: [] for f in FIELDS} for _ in rows]
        if len(rows) != len(self.columns):
            self.errors += 1
            return
        self.times.append(round((now - self.t0) * 1000))
        for columns, row in zip(self.columns, rows):
            for field in FIELDS:
                columns[field].append(row.get(field))
        if self.link_bdfs and self._count % self.link_every == 0:
            self.links.append({"t_ms": self.times[-1], "links": read_links(self.link_bdfs, self.sys_root)})
        self._count += 1

    def series(self) -> dict:
        return {
            "source": self.source.name,
            "interval_s": self.interval,
            "t0": self.t0,
            "t_ms": self.times,
            "gpus": self.columns,
            "fields": {"power_dw": "deciwatts", "energy_j": "cumulative joules", "reasons": "NVML clocks event reasons bitmask"},
            "errors": self.errors,
            "pcie_links": self.links,
        }


def window(series: dict, start: float, end: float) -> list[int]:
    """Sample indexes whose wall-clock time falls in [start, end]."""
    t0 = series["t0"]
    return [i for i, t in enumerate(series["t_ms"]) if start <= t0 + t / 1000 <= end]


def _share(values: list, mask: int) -> float:
    known = [v for v in values if v is not None]
    return round(sum(1 for v in known if v & mask) / len(known), 3) if known else 0.0


def analyze_phase(series: dict, start: float, end: float, limits: list[dict]) -> dict:
    """Clock, power and throttle summary of every GPU inside one phase."""
    indexes = window(series, start, end)
    result = {"samples": len(indexes), "gpus": []}
    verdicts = []
    for gpu_index, columns in enumerate(series["gpus"]):
        pick = lambda f: [columns[f][i] for i in indexes if columns[f][i] is not None]  # noqa: E731
        sm = pick("sm_mhz")
        if not sm:
            result["gpus"].append({"gpu": gpu_index, "samples": 0})
            continue
        head = sm[: max(1, len(sm) // 10)]
        tail = sm[len(sm) // 2:]
        reasons = pick("reasons")
        power = [p / 10 for p in pick("power_dw")]
        limit = (limits[gpu_index] if gpu_index < len(limits) else {}).get("enforced_limit_w")
        energy = pick("energy_j")
        busy = [r for r in reasons if not r & REASONS["gpu_idle"]]
        shares = {name: _share(reasons, bit) for name, bit in REASONS.items()}
        known = sum(REASONS.values())
        for bit in (1 << i for i in range(32)):
            if not bit & known and any(r & bit for r in reasons):
                shares[f"bit_0x{bit:x}"] = _share(reasons, bit)  # newer drivers add reasons
        drop = (median(head) - median(tail)) / median(head) if median(head) else 0.0
        if busy and _share(busy, HW_REASONS) > 0.05:
            verdict = "hw_slowdown"
        elif busy and _share(busy, REASONS["sw_thermal"]) > 0.05:
            verdict = "thermal"
        elif busy and _share(busy, REASONS["sw_power_cap"]) > 0.2:
            verdict = "power_capped"
        else:
            verdict = "ok"
        verdicts.append(verdict)
        result["gpus"].append({
            "gpu": gpu_index,
            "samples": len(sm),
            "sm_mhz": {"median": median(sm), "min": min(sm), "max": max(sm)},
            "mem_mhz_median": median(pick("mem_mhz")) if pick("mem_mhz") else None,
            "sm_clock_drop_pct": round(100 * drop, 1),
            "temp_c_max": max(pick("temp_c"), default=None),
            "power_w": {"median": round(median(power), 1), "max": round(max(power), 1)} if power else None,
            "power_to_limit_max": round(max(power) / limit, 3) if power and limit else None,
            "util_pct_median": median(pick("util_pct")) if pick("util_pct") else None,
            "pcie": {"gen_min": min(pick("pcie_gen"), default=None), "width_min": min(pick("pcie_width"), default=None)},
            "energy_j": energy[-1] - energy[0] if len(energy) > 1 else None,
            "reason_share": {k: v for k, v in shares.items() if v},
            "verdict": verdict,
        })
    order = ["ok", "power_capped", "thermal", "hw_slowdown"]
    result["verdict"] = max(verdicts, key=order.index) if verdicts else "no_data"
    return result


def overclock_signals(series: dict, gpus: list[dict]) -> list[dict]:
    """Observed clocks above the card's rated maximum (factory or user OC)."""
    findings = []
    for gpu_index, columns in enumerate(series["gpus"]):
        info = gpus[gpu_index] if gpu_index < len(gpus) else {}
        clocks = info.get("clocks") or {}
        rated = clocks.get("max_customer_boost_sm_mhz") or clocks.get("max_sm_mhz")
        observed = max((v for v in columns["sm_mhz"] if v is not None), default=None)
        if rated and observed and observed > rated * 1.01:
            findings.append({"gpu": gpu_index, "kind": "sm_clock_above_rated",
                             "detail": f"observed {observed} MHz > rated {rated} MHz"})
        mem_rated = clocks.get("max_mem_mhz")
        mem_observed = max((v for v in columns["mem_mhz"] if v is not None), default=None)
        if mem_rated and mem_observed and mem_observed > mem_rated * 1.01:
            findings.append({"gpu": gpu_index, "kind": "mem_clock_above_rated",
                             "detail": f"observed {mem_observed} MHz > rated {mem_rated} MHz"})
    return findings
