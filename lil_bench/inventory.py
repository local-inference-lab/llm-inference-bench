"""Exact hardware and PCIe topology, readable inside an unprivileged container.

Nothing here needs root-only tools (dmidecode) or firmware tables. Serial
numbers, board serials and system UUIDs are never collected; a GPU is
recognizable across runs only through a salted hash of its UUID.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
PCI_ATTRS = ("vendor", "device", "subsystem_vendor", "subsystem_device", "class", "revision",
             "current_link_speed", "current_link_width", "max_link_speed", "max_link_width", "numa_node")
DMI_FIELDS = ("sys_vendor", "product_name", "product_version", "board_vendor", "board_name",
              "board_version", "bios_vendor", "bios_version", "bios_date", "chassis_type")
EDAC_FIELDS = ("dimm_mem_type", "size", "dimm_label", "dimm_location", "dimm_dev_type", "dimm_edac_mode")
LSPCI_KEEP = re.compile(r"^\s*(LnkCap|LnkSta|LnkCtl2|LnkSta2|ACSCap|ACSCtl|ATSCap|ATSCtl|DevCap2|DevCtl2|Capabilities: \[[0-9a-f]+\] (Access Control|Address Translation|Resizable BAR))")
KNOWN_SWITCH_VENDORS = {"0x10b5": "Broadcom/PLX", "0x1000": "Broadcom", "0x11f8": "Microchip", "0x1d9b": "Microchip"}
SALT = "lil-bench/1:"
PCIE_KEYS = ("max_gen", "gpu_max_gen", "max_width", "current_gen", "current_width", "replay_counter")


def identity_hash(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256((SALT + value).encode()).hexdigest()[:24]


def read(path: Path | str, limit: int = 4096) -> str | None:
    try:
        return Path(path).read_text(errors="replace")[:limit].strip()
    except OSError:
        return None


def run(cmd: list[str], timeout: float = 15.0, limit: int = 200_000) -> dict:
    if shutil.which(cmd[0]) is None:
        return {"cmd": cmd, "error": "not installed"}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"cmd": cmd, "error": f"{type(error).__name__}: {error}"}
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout[:limit],
            "stderr": proc.stderr[:2000]}


def normalize_bdf(bus_id: str) -> str:
    """NVML's 00000000:01:00.0 → sysfs 0000:01:00.0."""
    bus_id = bus_id.strip().lower()
    domain, _, rest = bus_id.partition(":")
    return f"{int(domain, 16):04x}:{rest}" if rest else bus_id


def pci_device(bdf: str, sys_root: str = "/sys") -> dict:
    base = Path(sys_root, "bus/pci/devices", bdf)
    item = {"bdf": bdf}
    for attr in PCI_ATTRS:
        value = read(base / attr)
        if value is not None:
            item[attr] = value
    driver = base / "driver"
    if driver.exists():
        item["driver"] = os.path.basename(os.path.realpath(driver))
    klass = item.get("class", "")
    if klass.startswith("0x0604"):
        item["role"] = "bridge"
        vendor = item.get("vendor", "")
        if vendor in KNOWN_SWITCH_VENDORS:
            item["switch_vendor"] = KNOWN_SWITCH_VENDORS[vendor]
    elif klass.startswith("0x0300") or klass.startswith("0x0302"):
        item["role"] = "gpu"
    return item


def pci_chain(bdf: str, sys_root: str = "/sys") -> list[dict]:
    """Every PCI hop from the root port down to the device, root first."""
    real = os.path.realpath(Path(sys_root, "bus/pci/devices", bdf))
    hops = [part for part in real.split(os.sep) if BDF.match(part)]
    root = next((part for part in real.split(os.sep) if part.startswith("pci")), None)
    chain = [pci_device(hop, sys_root) for hop in hops]
    if chain and root:
        chain[0]["host_bridge"] = root
    return chain


def link_summary(chain: list[dict]) -> dict:
    """The narrowest link on the path and whether a switch sits in it."""
    def speed(value):
        match = re.match(r"([\d.]+)", value or "")
        return float(match.group(1)) if match else None
    speeds = [speed(h.get("current_link_speed")) for h in chain if h.get("current_link_speed")]
    widths = [int(h["current_link_width"]) for h in chain
              if str(h.get("current_link_width", "")).isdigit() and int(h["current_link_width"]) > 0]
    bridges = [h for h in chain if h.get("role") == "bridge"]
    vendors = sorted({h["switch_vendor"] for h in bridges if h.get("switch_vendor")})
    return {
        "hops": len(chain),
        "bridges": len(bridges),
        # A root port plus the switch upstream and downstream ports.
        "behind_switch": len(bridges) >= 3 or bool(vendors),
        "switches": " + ".join(f"{v} switch" for v in vendors),
        # Idle links train down and switch-internal ports report nominal
        # speeds, so these are a snapshot; telemetry samples links under load.
        "min_current_gt_s": min(speeds) if speeds else None,
        "min_current_width": min(widths) if widths else None,
    }


ACS_BITS = ("SrcValid", "TransBlk", "ReqRedir", "CmpltRedir", "UpstreamFwd", "EgressCtrl", "DirectTrans")
# With these ACS controls on, peer-to-peer requests are redirected up to the
# root complex instead of being routed inside the switch.
ACS_REDIRECT = ("ReqRedir", "CmpltRedir", "EgressCtrl")


def parse_acs(line: str) -> dict:
    """``ACSCtl: SrcValid- TransBlk+ …`` → {"SrcValid": False, "TransBlk": True, …}."""
    return {name: sign == "+" for name, sign in re.findall(r"(\w+)([+-])", line.split(":", 1)[1] if ":" in line else line)
            if name in ACS_BITS}


def lspci_detail(bdfs: list[str]) -> dict:
    """Link, ACS and ATS lines of ``lspci -vvv`` for every hop.

    Extended capabilities are readable only with CAP_SYS_ADMIN, which
    ``docker exec --privileged`` grants; otherwise lspci reports
    "<access denied>" and ACS stays unknown.
    """
    detail = {}
    if shutil.which("lspci") is None:
        return detail
    for bdf in bdfs:
        result = run(["lspci", "-vvv", "-s", bdf], timeout=10)
        output = result.get("stdout", "")
        lines = [line.strip() for line in output.splitlines() if LSPCI_KEEP.match(line)]
        head = output.splitlines()[:1]
        item = {"name": head[0] if head else "", "lines": lines,
                "readable": bool(output) and "<access denied>" not in output}
        for line in lines:
            if line.startswith("ACSCap:"):
                item["acs_cap"] = parse_acs(line)
            elif line.startswith("ACSCtl:"):
                item["acs_ctl"] = parse_acs(line)
        if "acs_ctl" in item:
            item["acs_redirect"] = [bit for bit in ACS_REDIRECT if item["acs_ctl"].get(bit)]
        detail[bdf] = item
    return detail


def acs_summary(detail: dict) -> dict:
    """Whether ACS could be read and which hops redirect peer-to-peer traffic."""
    readable = [bdf for bdf, item in detail.items() if item.get("readable")]
    return {
        "readable": bool(detail) and len(readable) == len(detail),
        "unreadable": sorted(set(detail) - set(readable)),
        "hops_with_acs": sorted(bdf for bdf, item in detail.items() if "acs_ctl" in item),
        "redirect": {bdf: item["acs_redirect"] for bdf, item in detail.items() if item.get("acs_redirect")},
    }


def iommu_group(bdf: str, sys_root: str = "/sys") -> str | None:
    link = Path(sys_root, "bus/pci/devices", bdf, "iommu_group")
    return os.path.basename(os.path.realpath(link)) if link.exists() else None


def integrated_gpu(name: str | None) -> bool:
    """GB10 (DGX Spark) sits on the CPU package and shares the host's memory."""
    return "GB10" in (name or "")


def host_memory_mib(proc_root: str = "/proc") -> int | None:
    meminfo = read(Path(proc_root, "meminfo"), 20000) or ""
    match = re.search(r"^MemTotal:\s+(\d+) kB", meminfo, re.M)
    return int(match.group(1)) // 1024 if match else None


def nvml_gpus(proc_root: str = "/proc") -> dict:
    try:
        import pynvml
    except ImportError:
        return {"error": "pynvml not installed", "gpus": []}
    try:
        pynvml.nvmlInit()
    except Exception as error:  # noqa: BLE001 - every NVML failure is reported, not raised
        return {"error": f"nvmlInit: {error}", "gpus": []}

    def call(fn, *args):
        try:
            value = getattr(pynvml, fn)(*args)
        except Exception:  # noqa: BLE001
            return None
        return value.decode() if isinstance(value, bytes) else value

    system = {
        "driver_version": call("nvmlSystemGetDriverVersion"),
        "cuda_driver_version": call("nvmlSystemGetCudaDriverVersion_v2") or call("nvmlSystemGetCudaDriverVersion"),
        "nvml_version": call("nvmlSystemGetNVMLVersion"),
    }
    gpus = []
    for index in range(call("nvmlDeviceGetCount") or 0):
        handle = call("nvmlDeviceGetHandleByIndex", index)
        if handle is None:
            continue
        pci = call("nvmlDeviceGetPciInfo_v3", handle) or call("nvmlDeviceGetPciInfo", handle)
        bus_id = pci.busId.decode() if pci is not None and isinstance(pci.busId, bytes) else getattr(pci, "busId", "")
        memory = call("nvmlDeviceGetMemoryInfo", handle)
        ecc = call("nvmlDeviceGetEccMode", handle)
        mig = call("nvmlDeviceGetMigMode", handle)
        capability = call("nvmlDeviceGetCudaComputeCapability", handle)
        name = call("nvmlDeviceGetName", handle)
        integrated = integrated_gpu(name)
        power_constraints = call("nvmlDeviceGetPowerManagementLimitConstraints", handle)
        clock = pynvml.NVML_CLOCK_SM, pynvml.NVML_CLOCK_MEM, pynvml.NVML_CLOCK_GRAPHICS
        gpu = {
            "index": index,
            "name": name,
            "uuid_hash": identity_hash(call("nvmlDeviceGetUUID", handle)),
            "bdf": normalize_bdf(bus_id) if bus_id else None,
            "pci_device_id": f"0x{pci.pciDeviceId:08x}" if pci is not None else None,
            "pci_subsystem_id": f"0x{pci.pciSubSystemId:08x}" if pci is not None else None,
            "board_part_number": call("nvmlDeviceGetBoardPartNumber", handle),
            "vbios": call("nvmlDeviceGetVbiosVersion", handle),
            "inforom_image": call("nvmlDeviceGetInforomImageVersion", handle),
            "architecture": call("nvmlDeviceGetArchitecture", handle),
            "compute_capability": list(capability) if capability else None,
            "memory_total_mib": memory.total // 2**20 if memory is not None else None,
            "bar1_total_mib": (lambda b: b.bar1Total // 2**20 if b is not None else None)(call("nvmlDeviceGetBAR1MemoryInfo", handle)),
            "ecc_mode": list(ecc) if ecc else None,
            "mig_mode": list(mig) if mig else None,
            "persistence_mode": call("nvmlDeviceGetPersistenceMode", handle),
            "compute_mode": call("nvmlDeviceGetComputeMode", handle),
            "numa_node": call("nvmlDeviceGetNumaNodeId", handle),
            # An integrated GPU has no PCIe link of its own: NVML's link values are not a slot.
            "pcie": dict.fromkeys(PCIE_KEYS) if integrated else {
                "max_gen": call("nvmlDeviceGetMaxPcieLinkGeneration", handle),
                "gpu_max_gen": call("nvmlDeviceGetGpuMaxPcieLinkGeneration", handle),
                "max_width": call("nvmlDeviceGetMaxPcieLinkWidth", handle),
                "current_gen": call("nvmlDeviceGetCurrPcieLinkGeneration", handle),
                "current_width": call("nvmlDeviceGetCurrPcieLinkWidth", handle),
                "replay_counter": call("nvmlDeviceGetPcieReplayCounter", handle),
            },
            "power": {
                "enforced_limit_w": _milli(call("nvmlDeviceGetEnforcedPowerLimit", handle)),
                "limit_w": _milli(call("nvmlDeviceGetPowerManagementLimit", handle)),
                "default_limit_w": _milli(call("nvmlDeviceGetPowerManagementDefaultLimit", handle)),
                "min_limit_w": _milli(power_constraints[0]) if power_constraints else None,
                "max_limit_w": _milli(power_constraints[1]) if power_constraints else None,
            },
            "clocks": {
                "max_sm_mhz": call("nvmlDeviceGetMaxClockInfo", handle, clock[0]),
                "max_mem_mhz": call("nvmlDeviceGetMaxClockInfo", handle, clock[1]),
                "max_graphics_mhz": call("nvmlDeviceGetMaxClockInfo", handle, clock[2]),
                "max_customer_boost_sm_mhz": call("nvmlDeviceGetMaxCustomerBoostClock", handle, clock[0]),
                "default_app_sm_mhz": call("nvmlDeviceGetDefaultApplicationsClock", handle, clock[0]),
                "default_app_mem_mhz": call("nvmlDeviceGetDefaultApplicationsClock", handle, clock[1]),
                "app_sm_mhz": call("nvmlDeviceGetApplicationsClock", handle, clock[0]),
                "app_mem_mhz": call("nvmlDeviceGetApplicationsClock", handle, clock[1]),
                "gpc_vf_offset_mhz": call("nvmlDeviceGetGpcClkVfOffset", handle),
                "mem_vf_offset_mhz": call("nvmlDeviceGetMemClkVfOffset", handle),
                "auto_boost": call("nvmlDeviceGetAutoBoostedClocksEnabled", handle),
            },
            "temperature_thresholds": {
                name: call("nvmlDeviceGetTemperatureThreshold", handle, getattr(pynvml, const))
                for name, const in (
                    ("shutdown_c", "NVML_TEMPERATURE_THRESHOLD_SHUTDOWN"),
                    ("slowdown_c", "NVML_TEMPERATURE_THRESHOLD_SLOWDOWN"),
                    ("gpu_max_c", "NVML_TEMPERATURE_THRESHOLD_GPU_MAX"),
                )
                if hasattr(pynvml, const)
            },
        }
        if integrated:
            gpu["integrated"] = True
            # The GPU allocates from the host's memory; NVML reports no
            # framebuffer of its own (Not Supported or 0).
            gpu["unified_memory"] = True
            if not gpu["memory_total_mib"]:
                gpu["memory_total_mib"] = host_memory_mib(proc_root)
        gpus.append(gpu)
    return {"system": system, "gpus": gpus}


def _milli(value):
    return round(value / 1000, 1) if isinstance(value, (int, float)) else None


def tuning_indicators(gpus: list[dict]) -> list[dict]:
    """Settings that make a GPU faster or slower than stock."""
    findings = []
    for gpu in gpus:
        clocks, power = gpu.get("clocks") or {}, gpu.get("power") or {}
        def add(kind, detail):
            findings.append({"gpu": gpu["index"], "kind": kind, "detail": detail})
        if clocks.get("gpc_vf_offset_mhz"):
            add("gpc_clock_offset", f"{clocks['gpc_vf_offset_mhz']:+d} MHz core clock offset")
        if clocks.get("mem_vf_offset_mhz"):
            add("mem_clock_offset", f"{clocks['mem_vf_offset_mhz']:+d} MHz memory clock offset")
        if power.get("enforced_limit_w") and power.get("default_limit_w") and \
                abs(power["enforced_limit_w"] - power["default_limit_w"]) >= 1:
            add("power_limit", f"power limit {power['enforced_limit_w']:g} W (default {power['default_limit_w']:g} W)")
        if clocks.get("app_sm_mhz") and clocks.get("default_app_sm_mhz") and \
                clocks["app_sm_mhz"] != clocks["default_app_sm_mhz"]:
            add("application_clocks", f"application SM clock {clocks['app_sm_mhz']} MHz (default {clocks['default_app_sm_mhz']} MHz)")
    return findings


def cpu_inventory(sys_root: str = "/sys") -> dict:
    info: dict = {}
    result = run(["lscpu", "-J"])
    try:
        rows = json.loads(result.get("stdout") or "{}").get("lscpu", [])
        info["lscpu"] = {row["field"].rstrip(":"): row.get("data") for row in rows if "field" in row}
    except ValueError:
        info["lscpu_error"] = result.get("error") or result.get("stderr")
    nodes = {}
    for node in sorted(Path(sys_root, "devices/system/node").glob("node[0-9]*")):
        meminfo = read(node / "meminfo") or ""
        total = re.search(r"MemTotal:\s+(\d+) kB", meminfo)
        nodes[node.name] = {
            "cpulist": read(node / "cpulist"),
            "mem_total_mib": int(total.group(1)) // 1024 if total else None,
        }
    info["numa_nodes"] = nodes
    info["governor"] = read(Path(sys_root, "devices/system/cpu/cpu0/cpufreq/scaling_governor"))
    info["max_freq_khz"] = read(Path(sys_root, "devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"))
    info["boost"] = read(Path(sys_root, "devices/system/cpu/cpufreq/boost"))
    info["smt_active"] = read(Path(sys_root, "devices/system/cpu/smt/active"))
    return info


def memory_inventory(sys_root: str = "/sys", proc_root: str = "/proc") -> dict:
    meminfo = read(Path(proc_root, "meminfo"), 20000) or ""
    def kb(name):
        match = re.search(rf"^{name}:\s+(\d+) kB", meminfo, re.M)
        return int(match.group(1)) if match else None
    total = kb("MemTotal")
    dimms = []
    for dimm in sorted(Path(sys_root, "devices/system/edac/mc").glob("mc*/dimm*")) + \
            sorted(Path(sys_root, "devices/system/edac/mc").glob("mc*/rank*")):
        item = {"id": f"{dimm.parent.name}/{dimm.name}"}
        for field in EDAC_FIELDS:
            value = read(dimm / field)
            if value is not None:
                item[field] = value
        if item.get("size") not in (None, "0"):
            dimms.append(item)
    types = sorted({d.get("dimm_mem_type") for d in dimms if d.get("dimm_mem_type")})
    return {
        "mem_total_mib": total // 1024 if total else None,
        "hugepages_total": kb("HugePages_Total"),
        "edac_dimms": dimms,
        "dimm_count": len(dimms),
        "memory_types": types,
        "note": None if dimms else "EDAC reports no DIMMs (memory type and speed need dmidecode on the host)",
    }


def dmi_inventory(sys_root: str = "/sys") -> dict:
    base = Path(sys_root, "class/dmi/id")
    return {field: read(base / field) for field in DMI_FIELDS if read(base / field) is not None}


def host_inventory(proc_root: str = "/proc", sys_root: str = "/sys") -> dict:
    cmdline = read(Path(proc_root, "cmdline")) or ""
    # Only boot switches that matter for P2P and performance; never root= or UUIDs.
    relevant = [arg for arg in cmdline.split() if re.match(
        r"(amd_iommu|intel_iommu|iommu|pci|pcie_\w+|isolcpus|nohz\w*|mitigations|hugepages|default_hugepagesz|processor\.\w+|idle|intel_pstate|amd_pstate)\b", arg)]
    groups = Path(sys_root, "kernel/iommu_groups")
    return {
        "kernel": platform.release(),
        "machine": platform.machine(),
        "kernel_cmdline_relevant": relevant,
        "iommu_groups": len(list(groups.iterdir())) if groups.is_dir() else 0,
        "nvidia_driver": read(Path(proc_root, "driver/nvidia/version")),
        "nvidia_params": read(Path(proc_root, "driver/nvidia/params"), 20000),
        "container_os": (read("/etc/os-release") or "").split("\n")[0:2],
    }


def collect(sys_root: str = "/sys", proc_root: str = "/proc") -> dict:
    nvml = nvml_gpus(proc_root)
    gpus = nvml.get("gpus", [])
    topology = {}
    all_hops = []
    for gpu in gpus:
        if not gpu.get("bdf") or gpu.get("integrated"):
            continue
        chain = pci_chain(gpu["bdf"], sys_root)
        topology[gpu["bdf"]] = {"gpu": gpu["index"], "chain": chain, "summary": link_summary(chain)}
        all_hops.extend(h["bdf"] for h in chain)
    unique_hops = list(dict.fromkeys(all_hops))
    detail = lspci_detail(unique_hops)
    for path in topology.values():
        path["iommu_groups"] = {hop["bdf"]: iommu_group(hop["bdf"], sys_root) for hop in path["chain"]}
    return {
        "gpus": gpus,
        "nvml_system": nvml.get("system"),
        "nvml_error": nvml.get("error"),
        "tuning": tuning_indicators(gpus),
        "pcie": {
            "gpu_paths": topology,
            "lspci_detail": detail,
            "acs": acs_summary(detail),
            "lspci_tree": run(["lspci", "-tv"]).get("stdout"),
            "nvidia_smi_topo": run(["nvidia-smi", "topo", "-m"]).get("stdout"),
            "nvidia_smi_p2p_read": run(["nvidia-smi", "topo", "-p2p", "r"]).get("stdout"),
            "nvidia_smi_p2p_write": run(["nvidia-smi", "topo", "-p2p", "w"]).get("stdout"),
            "nvidia_smi_nvlink": run(["nvidia-smi", "nvlink", "-s"]).get("stdout"),
        },
        "cpu": cpu_inventory(sys_root),
        "memory": memory_inventory(sys_root, proc_root),
        "board": dmi_inventory(sys_root),
        "host": host_inventory(proc_root, sys_root),
    }
