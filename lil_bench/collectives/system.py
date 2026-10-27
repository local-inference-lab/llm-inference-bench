"""The GPUs under test: selection, peer paths through the PCI tree, IOMMU, driver P2P settings, software."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import time
from pathlib import Path

from ..inventory import read

# Peer path between two GPUs, from the PCI tree: a shared PCIe switch, the same CPU root complex
# through different root ports, different host bridges (root complexes) of one CPU, or other sockets.
PATHS = {
    "SW": "through a PCIe switch",
    "RC": "through one CPU root complex (different root ports)",
    "HB": "between CPU host bridges",
    "SYS": "between CPU sockets / NUMA nodes",
}


def select_gpus(hardware: dict, tp: int, requested: str | None) -> list[int]:
    """nvidia-smi indices of the ranks: ``--gpus`` if given, else the first ``tp`` GPUs."""
    available = [gpu["index"] for gpu in hardware.get("gpus") or [] if not gpu.get("integrated")]
    if requested:
        gpus = [int(value) for value in requested.split(",") if value.strip()]
        missing = [gpu for gpu in gpus if gpu not in available]
        if missing:
            raise ValueError(f"GPU(s) {missing} not found; this machine has {available}")
        if len(set(gpus)) != len(gpus):
            raise ValueError("--gpus lists a GPU twice")
        return gpus
    if len(available) < tp:
        raise ValueError(f"TP{tp} needs {tp} GPUs; this machine has {len(available)}")
    return available[:tp]


def _gpu(hardware: dict, index: int) -> dict:
    return next(gpu for gpu in hardware.get("gpus") or [] if gpu["index"] == index)


def _chain(hardware: dict, index: int) -> list[dict]:
    gpu = _gpu(hardware, index)
    return (hardware.get("pcie", {}).get("gpu_paths", {}).get(gpu.get("bdf") or "", {}) or {}).get("chain") or []


def socket_of_node(node, sys_root: str = "/sys") -> int | None:
    """CPU package of a NUMA node (several NUMA nodes per socket are common: NPS2/NPS4, SNC)."""
    if node is None or int(node) < 0:
        return None
    cpus = read(Path(sys_root, f"devices/system/node/node{int(node)}/cpulist")) or ""
    match = re.match(r"(\d+)", cpus)
    if not match:
        return None
    package = read(Path(sys_root, f"devices/system/cpu/cpu{match.group(1)}/topology/physical_package_id"))
    return int(package) if package and package.isdigit() else None


def pair_path(hardware: dict, a: int, b: int, sys_root: str = "/sys") -> str:
    """SW / RC / HB / SYS for two GPUs (see ``PATHS``)."""
    sockets = (socket_of_node(_gpu(hardware, a).get("numa_node"), sys_root),
               socket_of_node(_gpu(hardware, b).get("numa_node"), sys_root))
    if None not in sockets and sockets[0] != sockets[1]:
        return "SYS"
    first, second = _chain(hardware, a), _chain(hardware, b)
    if not first or not second:
        return "?"
    if first[0].get("host_bridge") != second[0].get("host_bridge"):
        return "HB"
    shared = 0
    for x, y in zip(first, second):
        if x["bdf"] != y["bdf"]:
            break
        shared += 1
    # The first hop is the root port; sharing more than it means a switch below it.
    return "SW" if shared >= 2 else "RC"


def pair_paths(hardware: dict, gpus: list[int], sys_root: str = "/sys") -> dict[str, str]:
    return {f"{a}-{b}": pair_path(hardware, a, b, sys_root) for a in gpus for b in gpus if a != b}


def topology_class(paths: dict[str, str]) -> str:
    """One label for the selected set: the worst path any pair of ranks takes."""
    kinds = set(paths.values())
    if not kinds:
        return "single"
    if kinds == {"SW"}:
        return "switch"
    for kind, label in (("SYS", "cross-socket"), ("HB", "host-bridges"), ("RC", "root-ports")):
        if kind in kinds:
            return label if kinds == {kind} else f"mixed-{label}"
    return "unknown"


def iommu(hardware: dict, gpus: list[int], sys_root: str = "/sys") -> dict:
    """IOMMU state: off, on with identity (passthrough) domains for the GPUs, or translating their DMA."""
    units = sorted(p.name for p in Path(sys_root, "class/iommu").glob("*")) if Path(sys_root, "class/iommu").exists() else []
    types = {}
    for index in gpus:
        bdf = _gpu(hardware, index).get("bdf")
        link = Path(sys_root, "bus/pci/devices", bdf or "", "iommu_group")
        group = os.path.basename(os.path.realpath(link)) if bdf and link.exists() else None
        types[str(index)] = read(Path(sys_root, "kernel/iommu_groups", group, "type")) if group else None
    if not units:
        mode = "off"
    elif all(value == "identity" for value in types.values()):
        mode = "passthrough"
    elif any(value and value.startswith("DMA") for value in types.values()):
        mode = "translated"
    else:
        mode = "on"
    cmdline = (hardware.get("host") or {}).get("kernel_cmdline_relevant") or []
    return {"mode": mode, "units": len(units), "gpu_group_types": types,
            "cmdline": [arg for arg in cmdline if "iommu" in arg]}


def driver_params(hardware: dict) -> dict:
    """``/proc/driver/nvidia/params`` as a dict, with the P2P-related registry keys parsed out."""
    text = (hardware.get("host") or {}).get("nvidia_params") or ""
    params = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip():
            params[key.strip()] = value.strip().strip('"')
    registry = {}
    for item in (params.get("RegistryDwords") or "").split(";"):
        key, _, value = item.partition("=")
        if key.strip():
            registry[key.strip()] = value.strip()
    p2p_keys = ("ForceP2P", "RMForceP2PType", "RMPcieP2PType", "GrdmaPciTopoCheckOverride", "RMP2PReadsEnabled",
                "RMP2PWritesEnabled")
    override = {key: registry[key] for key in p2p_keys if key in registry}
    return {"registry_dwords": registry, "p2p_override": override, "p2p_override_present": bool(override),
            "resizable_bar": params.get("EnableResizableBar"), "dma_remap_peer_mmio": params.get("DmaRemapPeerMmio"),
            "driver": (hardware.get("nvml_system") or {}).get("driver_version")}


def busy(gpus: list[int], samples: int = 3, interval: float = 1.0) -> list[str]:
    """GPUs that compute or hold more than 1 GiB (another process would distort the measurement)."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - without NVML the check is skipped
        return []
    found: dict[int, str] = {}
    for sample in range(samples):
        for index in gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            used = pynvml.nvmlDeviceGetMemoryInfo(handle).used // 2**20
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            if util > 5 or used > 1024:
                found[index] = f"GPU {index}: {util}% busy, {used:,} MiB used"
        if sample + 1 < samples:
            time.sleep(interval)
    return [found[index] for index in sorted(found)]


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def flashinfer_source() -> dict:
    """The FlashInfer that the workers will import: installed wheel or a source tree on PYTHONPATH."""
    info = {"version": _version("flashinfer-python"), "commit": os.environ.get("LIL_FLASHINFER_COMMIT"),
            "patches": [p for p in os.environ.get("LIL_FLASHINFER_PATCHES", "").split(",") if p]}
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and Path(entry, "flashinfer", "experimental", "b12x").is_dir():
            info["source_tree"] = entry
            marker = read(Path(entry, ".lil-flashinfer-commit"))
            if marker and not info["commit"]:
                info["commit"] = marker
            break
    if not info["commit"]:
        try:
            import json
            direct = importlib.metadata.distribution("flashinfer-python").read_text("direct_url.json") or "{}"
            info["commit"] = json.loads(direct).get("vcs_info", {}).get("commit_id")
        except Exception:  # noqa: BLE001 - provenance is optional
            pass
    return info


def software() -> dict:
    return {
        "python": platform.python_version(),
        "torch": _version("torch"),
        "cuda_python": _version("cuda-python"),
        "cutlass_dsl": _version("nvidia-cutlass-dsl"),
        "flashinfer": flashinfer_source(),
        "image": os.environ.get("LIL_COLLECTIVES_IMAGE"),
    }
