"""The serving process and image this container runs.

``lil-bench`` runs through ``docker exec`` in the server's own container, so
the exact vLLM command line and environment are read from ``/proc`` instead
of being reconstructed, and the image identifies itself by its manifest.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

# Whole name segments only: HF_TOKEN is a secret, MAX_TOKENS is not. The same
# rule as the configurator's compose importer.
SECRET = re.compile(r"(^|_)(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|KEY|CREDENTIALS?)($|_)", re.I)
ENV_PREFIXES = (
    "VLLM_", "LIL_", "NCCL_", "CUDA_", "B12X_", "TORCH_", "PYTORCH_", "OMP_", "NVIDIA_",
    "FLASHINFER_", "TRITON_", "UCX_", "GLOO_", "HF_HUB_", "SAFETENSORS_", "LMCACHE_",
)
ENV_NAMES = {"MODEL", "TP", "DCP", "SPECULATOR", "HOST", "PORT"}
MANIFEST = "/opt/venv/share/lil-runtime/manifest.json"
CONTRACT = "/opt/lil/runtime/image-contract.json"
# vLLM's own default when --max-num-seqs is not given (V1 engine, GPU).
VLLM_DEFAULT_MAX_NUM_SEQS = 256


def read_cmdline(pid: str, proc_root: str = "/proc") -> list[str]:
    try:
        raw = Path(proc_root, pid, "cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def is_serve_command(argv: list[str]) -> bool:
    if "serve" not in argv:
        return False
    head = " ".join(os.path.basename(a) for a in argv[: argv.index("serve")])
    return bool(re.search(r"\bvllm\b|vllm\.entrypoints|lil-serve|runtime\.launcher", head))


def find_server(proc_root: str = "/proc") -> dict | None:
    """The API server process: the lowest PID whose argv is ``… vllm serve …``."""
    pids = sorted((p for p in os.listdir(proc_root) if p.isdigit()), key=int)
    for pid in pids:
        argv = read_cmdline(pid, proc_root)
        if argv and is_serve_command(argv):
            return {"pid": int(pid), "argv": argv}
    return None


def filtered_environment(pid: int | str, proc_root: str = "/proc") -> dict:
    """Serving-relevant variables of a process, never secrets."""
    try:
        raw = Path(proc_root, str(pid), "environ").read_bytes()
    except OSError:
        return {}
    env = {}
    for item in raw.split(b"\0"):
        key, sep, value = item.decode(errors="replace").partition("=")
        if not sep or SECRET.search(key):
            continue
        if key in ENV_NAMES or key.startswith(ENV_PREFIXES):
            env[key] = value
    return dict(sorted(env.items()))


def parse_serve_args(argv: list[str]) -> dict:
    """Model and options of ``vllm serve`` (``--a b``, ``--a=b`` and flags)."""
    args = argv[argv.index("serve") + 1:] if "serve" in argv else list(argv)
    model = None
    options: dict = {}
    index = 0
    while index < len(args):
        token = args[index]
        if token.startswith("--"):
            name, sep, value = token[2:].partition("=")
            name = name.replace("_", "-")
            if sep:
                options[name] = value
            elif index + 1 < len(args) and not args[index + 1].startswith("--"):
                options[name] = args[index + 1]
                index += 1
            else:
                options[name] = True
        elif model is None:
            model = token
        index += 1
    if model is None and isinstance(options.get("model"), str):
        model = options["model"]
    return {"model": model, "options": options}


def serve_limits(options: dict) -> dict:
    def number(name):
        value = options.get(name)
        try:
            return int(value) if value not in (None, True) else None
        except (TypeError, ValueError):
            return None
    max_num_seqs = number("max-num-seqs")
    return {
        "port": number("port") or 8000,
        "max_num_seqs": max_num_seqs or VLLM_DEFAULT_MAX_NUM_SEQS,
        "max_num_seqs_source": "argv" if max_num_seqs else "vllm_default",
        "max_model_len_arg": number("max-model-len"),
        "tensor_parallel_size": number("tensor-parallel-size") or 1,
        "served_model_name": options.get("served-model-name") if isinstance(options.get("served-model-name"), str) else None,
    }


def read_json(path: str) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def image_identity(manifest_path: str = MANIFEST, contract_path: str = CONTRACT) -> dict:
    manifest = read_json(manifest_path)
    contract = read_json(contract_path)
    identity = {"manifest": manifest, "contract": contract}
    assembly = manifest.get("assembly") if isinstance(manifest, dict) else None
    if isinstance(assembly, dict):
        identity["assembly_sha256"] = assembly.get("assembly_sha256")
        identity["alias"] = assembly.get("alias")
        identity["channel"] = assembly.get("channel")
    return identity


def http_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def http_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode(errors="replace")


METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")


def parse_metrics(text: str) -> list[tuple[str, dict, float]]:
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE.match(line)
        if not match:
            continue
        labels = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', match.group(2) or ""))
        try:
            samples.append((match.group(1), labels, float(match.group(3))))
        except ValueError:
            continue
    return samples


def metric_sum(samples, name: str) -> float:
    return sum(value for metric, _, value in samples if metric == name)


def kv_capacity(samples) -> dict:
    for metric, labels, _ in samples:
        if metric == "vllm:cache_config_info":
            try:
                blocks = int(labels.get("num_gpu_blocks") or 0)
                size = int(labels.get("block_size") or 0)
            except ValueError:
                continue
            return {"num_gpu_blocks": blocks, "block_size": size, "kv_tokens": blocks * size,
                    "cache_config": labels}
    return {}


def server_state(base_url: str) -> dict:
    """What the API reports: served model, context limit and KV capacity."""
    models = http_json(f"{base_url}/v1/models")
    entry = (models.get("data") or [{}])[0]
    state = {
        "model_id": entry.get("id"),
        "max_model_len": entry.get("max_model_len"),
        "root": entry.get("root"),
    }
    try:
        state["version"] = http_json(f"{base_url}/version").get("version")
    except (OSError, ValueError, urllib.error.URLError):
        state["version"] = None
    try:
        samples = parse_metrics(http_text(f"{base_url}/metrics"))
        state.update(kv_capacity(samples))
        state["metrics"] = True
    except (OSError, urllib.error.URLError):
        state["metrics"] = False
    return state


def busy_requests(base_url: str) -> float:
    samples = parse_metrics(http_text(f"{base_url}/metrics", timeout=5))
    return metric_sum(samples, "vllm:num_requests_running") + metric_sum(samples, "vllm:num_requests_waiting")
