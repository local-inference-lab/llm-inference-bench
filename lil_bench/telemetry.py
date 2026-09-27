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
          "reasons", "pcie_gen", "pcie_width", "pstate", "energy_j", "mem_used_mib", "fan_pct",
          "pcie_replay", "pcie_corr_err", "pcie_recovery")
# Cumulative NVML link error counters. (The PCIe byte counters are 32-bit
# and wrap within one sample at GB/s, so traffic comes from nvidia-smi dmon.)
PCIE_FIELDS = (
    ("pcie_replay", "NVML_FI_DEV_PCIE_REPLAY_COUNTER"),
    ("pcie_corr_err", "NVML_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS"),
    ("pcie_recovery", "NVML_FI_DEV_PCIE_L0_TO_RECOVERY_COUNTER"),
)
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
        self.pcie_fields = [(name, getattr(pynvml, const)) for name, const in PCIE_FIELDS if hasattr(pynvml, const)]

    def _counters(self, handle) -> dict:
        if not self.pcie_fields:
            return {}
        values = self._get(self.nvml.nvmlDeviceGetFieldValues, handle, [fid for _, fid in self.pcie_fields])
        if not values:
            return {}
        return {name: (value.value.ullVal if value.nvmlReturn == 0 else None)
                for (name, _), value in zip(self.pcie_fields, values)}

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
                **self._counters(handle),
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
            "counters": {"pcie_replay": "cumulative", "pcie_corr_err": "cumulative", "pcie_recovery": "cumulative"},
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
        def delta(field):
            values = [v for v in (columns.get(field) or [])[indexes[0]: indexes[-1] + 1] if v is not None]
            return values[-1] - values[0] if len(values) > 1 else None
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
            "pcie_errors": {k: delta(k) for k in ("pcie_replay", "pcie_corr_err", "pcie_recovery")
                            if delta(k)},
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


# ---------------------------------------------------------------------------
# Server and host: vLLM Prometheus counters and CPU load, once per second
# ---------------------------------------------------------------------------

SERVER_METRICS = (
    "vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:prompt_tokens_cached_total",
    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total", "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_drafts_total",
    "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total", "vllm:request_success_total",
)


def cpu_times(proc_root: str = "/proc") -> tuple[int, int] | None:
    try:
        with open(f"{proc_root}/stat") as handle:
            fields = [int(v) for v in handle.readline().split()[1:]]
    except (OSError, ValueError):
        return None
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return sum(fields), idle


def cpu_temperature(sys_root: str = "/sys") -> float | None:
    """Hottest CPU package/die sensor (k10temp, coretemp, zenpower)."""
    import glob
    import os
    hottest = None
    for hwmon in glob.glob(f"{sys_root}/class/hwmon/hwmon*"):
        try:
            with open(os.path.join(hwmon, "name")) as handle:
                name = handle.read().strip().lower()
        except OSError:
            continue
        if not any(chip in name for chip in ("k10temp", "coretemp", "zenpower")):
            continue
        for path in glob.glob(os.path.join(hwmon, "temp*_input")):
            try:
                with open(path) as handle:
                    value = int(handle.read().strip()) / 1000
            except (OSError, ValueError):
                continue
            hottest = value if hottest is None else max(hottest, value)
    return hottest


class ServerRecorder:
    """Samples vLLM /metrics counters and host CPU load in the background."""

    def __init__(self, base_url: str, t0: float, interval: float = 1.0, fetch=None,
                 proc_root: str = "/proc", sys_root: str = "/sys"):
        from . import server
        self.url = f"{base_url}/metrics"
        self.t0 = t0
        self.interval = interval
        self.fetch = fetch or (lambda: server.http_text(self.url, timeout=5))
        self.parse = server.parse_metrics
        self.proc_root, self.sys_root = proc_root, sys_root
        self.times: list[int] = []
        self.metrics: dict[str, list] = {name: [] for name in SERVER_METRICS}
        self.cpu_pct: list = []
        self.cpu_temp_c: list = []
        self.errors = 0
        self._last_cpu = cpu_times(proc_root)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lil-bench-server", daemon=True)

    def start(self) -> "ServerRecorder":
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
            samples = self.parse(self.fetch())
        except Exception:  # noqa: BLE001 - a missed scrape is not fatal
            self.errors += 1
            samples = None
        totals: dict[str, float] = {}
        for name, _, value in samples or []:
            if name in self.metrics:
                totals[name] = totals.get(name, 0.0) + value
        self.times.append(round((now - self.t0) * 1000))
        for name, values in self.metrics.items():
            value = totals.get(name) if samples is not None else None
            values.append(round(value, 4) if value is not None else None)
        current = cpu_times(self.proc_root)
        if current and self._last_cpu and current[0] > self._last_cpu[0]:
            total, idle = current[0] - self._last_cpu[0], current[1] - self._last_cpu[1]
            self.cpu_pct.append(round(100 * (total - idle) / total, 1))
        else:
            self.cpu_pct.append(None)
        self._last_cpu = current
        temperature = cpu_temperature(self.sys_root)
        self.cpu_temp_c.append(round(temperature, 1) if temperature is not None else None)

    def series(self) -> dict:
        return {"interval_s": self.interval, "t0": self.t0, "t_ms": self.times, "metrics": self.metrics,
                "host": {"cpu_pct": self.cpu_pct, "cpu_temp_c": self.cpu_temp_c}, "errors": self.errors,
                "fields": {"metrics": "vLLM Prometheus values summed over label sets; *_total are cumulative"}}


# ---------------------------------------------------------------------------
# PCIe traffic: nvidia-smi dmon, MB/s per GPU averaged over each second
# ---------------------------------------------------------------------------

class PcieRecorder:
    """Streams ``nvidia-smi dmon -s t`` (rx/tx MB/s per GPU, 1 s windows)."""

    def __init__(self, t0: float, command=None):
        self.t0 = t0
        self.command = command or ["nvidia-smi", "dmon", "-s", "t", "-d", "1"]
        self.times: list[int] = []
        self.gpus: dict[int, dict[str, list]] = {}
        self.error = None
        self.proc = None
        self._last_index = None
        self._thread = threading.Thread(target=self._run, name="lil-bench-pcie", daemon=True)

    def start(self) -> "PcieRecorder":
        if shutil.which(self.command[0]) is None:
            self.error = f"{self.command[0]} not found"
            return self
        try:
            self.proc = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except OSError as error:
            self.error = str(error)
            return self
        self._thread.start()
        return self

    def feed(self, line: str, now: float) -> None:
        parts = line.split()
        if len(parts) < 3 or line.lstrip().startswith("#"):
            return
        try:
            index = int(parts[0])
        except ValueError:
            return

        def number(text):
            try:
                return float(text)
            except ValueError:
                return None  # "-" when the driver has no sample
        # dmon prints one line per GPU per second; a new second starts when the index does not increase.
        if self._last_index is None or index <= self._last_index:
            self.times.append(round((now - self.t0) * 1000))
        self._last_index = index
        gpu = self.gpus.setdefault(index, {"rx_mbs": [], "tx_mbs": []})
        missing = len(self.times) - 1 - len(gpu["rx_mbs"])
        gpu["rx_mbs"].extend([None] * missing)
        gpu["tx_mbs"].extend([None] * missing)
        gpu["rx_mbs"].append(number(parts[1]))
        gpu["tx_mbs"].append(number(parts[2]))

    def _run(self) -> None:
        for line in self.proc.stdout:
            self.feed(line, time.time())

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def series(self) -> dict:
        n = len(self.times)
        gpus = []
        for index in sorted(self.gpus):
            gpu = self.gpus[index]
            gpus.append({key: (values + [None] * (n - len(values)))[:n] for key, values in gpu.items()})
        return {"source": "nvidia-smi dmon -s t", "interval_s": 1.0, "t0": self.t0, "t_ms": self.times,
                "gpus": gpus, "units": "MB/s", "error": self.error}


def pcie_rates(series: dict, start: float, end: float) -> list[dict]:
    """Mean and peak rx/tx GB/s of every GPU inside a phase."""
    t0 = series.get("t0") or 0
    idx = [i for i, t in enumerate(series.get("t_ms") or []) if start <= t0 + t / 1000 <= end]
    out = []
    for gpu in series.get("gpus") or []:
        row = {}
        for key in ("rx_mbs", "tx_mbs"):
            values = [gpu[key][i] for i in idx if gpu[key][i] is not None]
            name = key[:2]
            row[name] = round(sum(values) / len(values) / 1000, 3) if values else None
            row[f"{name}_peak"] = round(max(values) / 1000, 3) if values else None
        out.append(row)
    return out
