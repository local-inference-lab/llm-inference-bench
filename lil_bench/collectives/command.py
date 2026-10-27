"""``lil-bench collectives``: measure, save and upload one collectives diagnostic.

Typical use, in the dedicated image (GPUs 0-3 idle)::

    docker run --rm --gpus all --privileged --ipc=host --shm-size 16g \\
        -e LIL_BENCH_TOKEN=lilb_... ghcr.io/local-inference-lab/lil-collectives

``--tp 2`` or ``--gpus 4,5,6,7`` choose other ranks; ``--quick`` takes about five minutes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .. import DEFAULT_SITE, TOKEN_URL_PATH, VERSION
from .. import inventory, p2p, telemetry, upload
from . import HIDDEN, NCCL_REFERENCE, PROBE_VERSION, QUICK_ROWS, ROWS, SCHEMA, render, summary, system

ROOT = Path(__file__).resolve().parent.parent.parent
EXIT_NO_TOKEN, EXIT_SETUP, EXIT_MEASURE, EXIT_UPLOAD = 2, 3, 4, 5
UPLOAD_PATH = "/api/bench/collectives"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lil-bench collectives", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tp", type=int, default=4, choices=(2, 3, 4, 6, 8), help="ranks (default 4)")
    parser.add_argument("--gpus", default="", help="nvidia-smi indices of the ranks (default: the first --tp GPUs)")
    parser.add_argument("--hidden", type=int, default=HIDDEN, help=f"row width in BF16 values (default {HIDDEN})")
    parser.add_argument("--rows", default="", help="message rows (default 1,2,4,...,192)")
    parser.add_argument("--quick", action="store_true", help="six sizes and the default tuner effort only (~5 min)")
    parser.add_argument("--efforts", default="", help="tuner efforts to report (default: default,audit)")
    parser.add_argument("--skip-fabric", action="store_true")
    parser.add_argument("--skip-p2pmark", action="store_true")
    parser.add_argument("--skip-transports", action="store_true")
    parser.add_argument("--skip-nccl-ring", action="store_true")
    parser.add_argument("--allow-busy", action="store_true", help="measure even if the GPUs are in use (flagged)")
    parser.add_argument("--token", default=os.environ.get("LIL_BENCH_TOKEN", ""),
                        help="uploader identifier (default: $LIL_BENCH_TOKEN)")
    parser.add_argument("--site", default=os.environ.get("LIL_BENCH_SITE", DEFAULT_SITE))
    parser.add_argument("--no-upload", action="store_true", help="measure and save locally only")
    parser.add_argument("--note", default="", help="free text stored with the result (max 500 characters)")
    parser.add_argument("--report", nargs="+", metavar="FILE", help="print saved results instead of measuring")
    args = parser.parse_args(argv)
    if args.gpus:
        args.tp = len([value for value in args.gpus.split(",") if value.strip()])
    args.rows = args.rows or ",".join(map(str, QUICK_ROWS if args.quick else ROWS))
    args.efforts = [e for e in (args.efforts or ("default" if args.quick else "default,audit")).split(",") if e]
    return args


def load(path: str) -> dict:
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)


def _env(gpus: list[int]) -> dict:
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
               OMP_NUM_THREADS="1", PYTHONPATH=os.pathsep.join(
                   [str(ROOT)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    env.pop("LIL_BENCH_TOKEN", None)
    if env.get("NCCL_GRAPH_FILE") == "":
        env.pop("NCCL_GRAPH_FILE")
    for key, value in NCCL_REFERENCE.items():
        env.setdefault(key, value)
    return env


def _run(out, command: list[str], env: dict, label: str) -> int:
    """Run a worker, forwarding its progress lines (rank 0 prints; torchrun noise is dropped)."""
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               stdin=subprocess.DEVNULL)
    tail: list[str] = []
    for line in process.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-60:]
        if line.startswith("  ") and "rows" in line:
            out.info(line.strip())
    code = process.wait()
    if code:
        out.warn(f"{label} exited with {code}; last lines:")
        for line in tail[-25:]:
            out.info(f"  {line}")
    return code


def flashinfer_check(env: dict) -> str | None:
    """Why this Python cannot run FlashInfer's tuned collectives (checked in a child, no CUDA here)."""
    probe = ("from b12x.comm.pcie._collective_tuning import CollectiveQuery, _configs\n"
             "from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce\n")
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=300)
    return None if result.returncode == 0 else (result.stderr.strip().splitlines() or ["import failed"])[-1]


def command_report(out, args) -> int:
    for path in args.report:
        document = load(path)
        document.setdefault("summary", summary.summarize(document))
        out.line(f"{path}", "1")
        for line in render.lines(document):
            out.line(line)
        out.line()
    return 0


def main(argv: list[str]) -> int:
    from ..standard import Output, check_identity, load_bench_module, result_dir, _safe_argv

    args = parse_args(argv)
    out = Output()
    if args.report:
        return command_report(out, args)
    out.line(f"lil-bench {VERSION} · collectives probe v{PROBE_VERSION}", "1")
    identity = None
    if not args.no_upload:
        code, identity = check_identity(out, args)
        if code:
            return code

    out.step("Hardware and PCIe topology")
    hardware = inventory.collect()
    try:
        gpus = system.select_gpus(hardware, args.tp, args.gpus)
    except ValueError as error:
        out.error_panel("Cannot select the GPUs", [str(error)])
        return EXIT_SETUP
    paths = system.pair_paths(hardware, gpus)
    selection = {"gpus": gpus, "tp": len(gpus), "hidden": args.hidden, "rows": [int(r) for r in args.rows.split(",")],
                 "dtype": "bfloat16", "quick": args.quick, "efforts": args.efforts, "pair_paths": paths,
                 "topology_class": system.topology_class(paths)}
    environment = {"iommu": system.iommu(hardware, gpus), "driver": system.driver_params(hardware),
                   "software": system.software(), "acs": (hardware.get("pcie") or {}).get("acs")}
    by_index = {gpu["index"]: gpu for gpu in hardware.get("gpus") or []}
    out.ok(f"TP{len(gpus)} on GPUs {','.join(map(str, gpus))}: {by_index[gpus[0]].get('name')} · "
           f"{selection['topology_class']} · IOMMU {environment['iommu']['mode']} · NVIDIA P2P override "
           f"{'yes' if environment['driver']['p2p_override_present'] else 'none'}")
    for line in render.lines({"selection": selection, "environment": environment, "hardware": hardware})[1:6]:
        out.info(line)
    busy = [] if args.allow_busy else system.busy(gpus)
    if busy:
        out.error_panel("The GPUs are in use", busy + ["Other work distorts the measurement and may be disturbed by it.",
                                                       "Stop it, choose idle GPUs with --gpus, or pass --allow-busy."])
        return EXIT_SETUP

    env = _env(gpus)
    workdir = result_dir()
    run_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc)
    recorder = telemetry.Recorder(interval=0.5).start()
    recorder.link_bdfs = list(dict.fromkeys(
        hop["bdf"] for path in (hardware.get("pcie") or {}).get("gpu_paths", {}).values() for hop in path["chain"]))
    results: dict = {}
    phases: list[dict] = []
    status = "complete"
    scratch = Path(tempfile.mkdtemp(prefix="lil-collectives-"))
    try:
        if not args.skip_fabric:
            out.step("Peer bandwidth: pairs, all-to-all, ring, fan-out, incast")
            start = time.time()
            target = scratch / "fabric.json"
            code = _run(out, [sys.executable, "-m", "lil_bench.collectives.fabric", "--result", str(target)], env,
                        "fabric test")
            results["fabric"] = json.loads(target.read_text()) if code == 0 and target.exists() else \
                {"status": "failed", "returncode": code}
            phases.append({"kind": "fabric", "start": start, "end": time.time()})
            fsum = summary.fabric_summary(results["fabric"])
            if fsum:
                out.ok(f"fabric {fsum['class']}: pair read/write {fsum['pair_gbps']['read']:.1f}/"
                       f"{fsum['pair_gbps']['write']:.1f} GB/s, all-to-all "
                       f"{fsum['pattern_gbps']['all_to_all']['read']:.1f}/{fsum['pattern_gbps']['all_to_all']['write']:.1f}, "
                       f"ring {fsum['pattern_gbps']['ring']['read']:.1f}/{fsum['pattern_gbps']['ring']['write']:.1f}")

        if not args.skip_p2pmark:
            out.step("p2pmark (copy-engine bandwidth, latency, NCCL comparison)")
            start = time.time()
            saved = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=env["CUDA_VISIBLE_DEVICES"])
            try:
                free = lambda: [f for i, f in enumerate(_free_mib()) if i in gpus]  # noqa: E731
                from rich.console import Console
                results["p2pmark"] = p2p.measure(out, hardware, load_bench_module(), free_fn=free, busy_fn=None,
                                                 directory=None, console=Console(force_terminal=out.color, width=120))
                results["p2pmark"]["with_model_loaded"] = False
                if results["p2pmark"].pop("ran", False):
                    phases.append({"kind": "p2pmark", "start": start, "end": time.time()})
            except Exception as error:  # noqa: BLE001 - p2pmark is supplementary
                results["p2pmark"] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
                out.warn(f"p2pmark: {results['p2pmark']['error']}")
            finally:
                if saved is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = saved

        if not args.skip_transports:
            out.step(f"All-reduce transports, {len(selection['rows'])} sizes (kernels compile on first use)")
            reason = flashinfer_check(env)
            if reason:
                results["transports"] = {"status": "unavailable", "reason": reason}
                out.warn(f"FlashInfer collectives unavailable here: {reason}")
                status = "failed"
            else:
                start = time.time()
                target = scratch / "transports.json"
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                           f"--nproc-per-node={len(gpus)}", "--module", "lil_bench.collectives.transports",
                           "--result", str(target), "--rows", args.rows, "--hidden", str(args.hidden),
                           "--efforts", ",".join(args.efforts)]
                if args.quick:
                    command += ["--samples", "4", "--race-samples", "9"]
                code = _run(out, command, env, "transport measurement")
                results["transports"] = json.loads(target.read_text()) if target.exists() else {}
                if code:
                    results["transports"].update(status="failed", returncode=code)
                    status = "failed"
                phases.append({"kind": "transports", "start": start, "end": time.time()})
                if not args.skip_nccl_ring and results["transports"].get("sizes"):
                    out.step("NCCL with the ring algorithm forced")
                    start = time.time()
                    target = scratch / "ring.json"
                    ring_env = dict(env, NCCL_ALGO="Ring")
                    code = _run(out, command[:7] + ["--result", str(target), "--rows", args.rows, "--hidden",
                                                    str(args.hidden), "--nccl-only", "--nccl-label", "nccl_ring"],
                                ring_env, "NCCL ring measurement")
                    if code == 0 and target.exists():
                        ring = {size["rows"]: size for size in json.loads(target.read_text())["sizes"]}
                        for size in results["transports"]["sizes"]:
                            size["arms"] += (ring.get(size["rows"]) or {}).get("arms", [])
                    phases.append({"kind": "nccl_ring", "start": start, "end": time.time()})
    except KeyboardInterrupt:
        status = "interrupted"
        out.warn("interrupted; the partial result is saved but not uploaded")
    finally:
        recorder.stop()

    series = recorder.series()
    events = telemetry.link_events(series, telemetry.link_limits(hardware))
    document = {
        "schema": SCHEMA,
        "run_id": run_id,
        "status": status,
        "started": started.isoformat(),
        "finished": datetime.now(timezone.utc).isoformat(),
        "client": {"lil_bench": VERSION, "probe_version": PROBE_VERSION, "argv": _safe_argv(),
                   "uploader": (identity or {}).get("login"), "note": args.note[:500],
                   "allow_busy": bool(args.allow_busy), "image": os.environ.get("LIL_COLLECTIVES_IMAGE")},
        "hardware": hardware,
        "environment": environment,
        "selection": selection,
        "results": results,
        "phases": phases,
        "telemetry": series,
        "analysis": {"pcie": telemetry.link_health(events)},
    }
    document["summary"] = summary.summarize(document)
    path = workdir / f"collectives-{run_id}.json.gz"
    path.write_bytes(upload.compress(document))
    out.line()
    for line in render.lines(document):
        out.line(line)
    out.line()
    out.ok(f"saved {path} ({path.stat().st_size / 1024:.0f} kB)")
    if args.no_upload or status == "interrupted":
        return 0 if status == "complete" else EXIT_MEASURE
    try:
        reply = upload.upload_bytes(args.site, args.token, path.read_bytes(), path=UPLOAD_PATH)
    except upload.UploadError as error:
        out.error_panel("Upload failed", [str(error), "The result is saved; upload it later with:",
                                          f"  lil-bench upload {path}"])
        return EXIT_UPLOAD
    out.ok(f"uploaded: {reply.get('url')}")
    out.info(f"share this link; all collectives results: {args.site.rstrip('/')}/bench/collectives")
    return 0 if status == "complete" else EXIT_MEASURE


def _free_mib() -> list[int]:
    try:
        import pynvml
        pynvml.nvmlInit()
        return [pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).free // 2**20
                for i in range(pynvml.nvmlDeviceGetCount())]
    except Exception:  # noqa: BLE001
        return []


__all__ = ["main", "TOKEN_URL_PATH"]
