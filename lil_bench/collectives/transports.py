"""Every FlashInfer/b12x all-reduce transport, the joint tuner's choice, and NCCL, on the selected ranks.

Run under torchrun with CUDA_VISIBLE_DEVICES set to the ranks::

    python -m torch.distributed.run --standalone --nproc-per-node=N -m lil_bench.collectives.transports --result FILE

For each message size (rows x hidden BF16 values):

1. screening: every launch configuration of every transport that FlashInfer's joint tuner may
   consider (``_configs`` of the collective query), pinned one at a time and timed briefly;
2. the fastest configuration of each transport, the tuner's own selection at each requested effort
   (fresh selection caches) and NCCL are captured as CUDA graphs and measured:
   * steady state: graphs of ``chain`` back-to-back calls, replayed ``replays`` times, in two
     independent passes with alternating order (the second pass exposes unstable cells);
   * race style: one call per graph after an L2 flush, like the tuner's own race samples;
   every value is the slowest rank's time per call, median over samples;
3. correctness: every b12x result is compared with the FP32 sum rounded to BF16 (one BF16 ulp).

``--nccl-only`` measures only NCCL (used for a second run with NCCL_ALGO=Ring).
Rank 0 writes the result after every size, so a failure keeps the sizes measured before it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from datetime import timedelta
from pathlib import Path

from . import TRANSPORTS, family

FLUSH_MIB = 256  # larger than any current GPU's L2


def prepare_lockstep(session, requests, group, world):
    """Advance one b12x preparation job on every rank, relaying collective readiness and tuning.

    Every rank exchanges its state each step, including ranks that already finished, the same
    protocol Flare's collective preparation uses.
    """
    import torch.distributed as dist

    job = state = error = None
    authorization = {"collective_key": None}
    try:
        while True:
            try:
                if job is None:
                    job = session.begin(requests)
                if state is None or not state.done:
                    state = job.advance(**authorization)
            except Exception as exc:  # noqa: BLE001 - every rank must learn about the failure
                error = exc
            local = (
                state is not None and state.done,
                () if state is None else tuple((item.key, tuple(item.ranks)) for item in state.ready_collectives),
                None if error is None else f"{type(error).__name__}: {error}",
                None if state is None else state.ready_collective_tuning,
                bool(state is not None and (state.ready_cache is not None or state.ready_tuning)),
            )
            states = [None] * world
            dist.all_gather_object(states, local, group=group)
            failures = [item[2] for item in states if item[2] is not None]
            if failures:
                raise RuntimeError(f"preparation failed: {failures}") from error
            if any(item[4] for item in states):
                raise RuntimeError("sharded candidate tuning is not supported here")
            if all(item[0] for item in states):
                return job.result()
            authorization = {"collective_key": None}
            if all(item[3] is not None for item in states):
                authorization = {"collective_tuning": tuple(item[3] for item in states)}
                continue
            ready = [item[1] for item in states]
            if all(ready):
                if any(item != ready[0] for item in ready[1:]):
                    raise RuntimeError(f"ranks declared different collectives: {ready}")
                authorization = {"collective_key": ready[0][0][0]}
    except BaseException:
        if job is not None:
            try:
                if state is not None and state.done:
                    job.result().close()
                else:
                    job.close()
            except BaseException:  # noqa: BLE001 - keep the original failure
                pass
        raise


class Worker:
    def __init__(self, args):
        import torch
        import torch.distributed as dist

        self.args, self.torch, self.dist = args, torch, dist
        self.rank, self.world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.set_num_threads(1)
        dist.init_process_group("gloo", timeout=timedelta(minutes=20))
        self.group = dist.group.WORLD
        self.nccl = dist.new_group(backend="nccl")
        self.rows = [int(value) for value in args.rows.split(",")]
        self.capacity = max(self.rows) * args.hidden * 2
        self.flush = torch.empty(FLUSH_MIB << 20, dtype=torch.uint8, device=self.device)
        self.nccl_buffer = torch.zeros(self.capacity // 2, dtype=torch.bfloat16, device=self.device)
        dist.all_reduce(self.nccl_buffer, group=self.nccl)
        torch.cuda.synchronize()
        self.manager = None
        if not args.nccl_only:
            from b12x.comm.pcie import _oneshot_preparation as oneshot
            from b12x.comm.pcie import _twoshot_preparation as twoshot
            from b12x.comm.pcie._collective_tuning import CollectiveQuery, _configs
            from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
            from b12x.preparation import CollectiveRequirement, PreparationSession, autotune_effort
            from b12x.preparation.types import require_prepared

            self.b12x = dict(oneshot=oneshot, twoshot=twoshot, CollectiveQuery=CollectiveQuery, configs=_configs,
                             CollectiveRequirement=CollectiveRequirement, effort=autotune_effort,
                             require_prepared=require_prepared)
            self.manager = PCIeAllReduce.from_exchange_group(exchange_group=self.group, device=self.device,
                                                             single_channel=True, max_size=self.capacity,
                                                             eager_buffer_bytes=self.capacity)
            self.manager.prepare_channels(("lil-bench:collectives",))
            barrier = lambda key, ranks: dist.barrier(group=self.group)  # noqa: E731
            caches = Path(tempfile.mkdtemp(prefix=f"lil-collectives-rank{self.rank}-"))
            with autotune_effort("default"):
                self.pinned = PreparationSession(device=self.device, autotune=False, collective_barrier=barrier)
            self.tuned = {}
            for effort in args.efforts:
                with autotune_effort(effort):
                    self.tuned[effort] = PreparationSession(device=self.device, autotune=True,
                                                            collective_barrier=barrier, cache_dir=caches / effort)

    def log(self, text: str) -> None:
        if self.rank == 0:
            print(text, flush=True)

    # -- helpers ---------------------------------------------------------------------------------

    def slowest(self, values: list[float]) -> list[float]:
        gathered = [None] * self.world
        self.dist.all_gather_object(gathered, values, group=self.group)
        return [max(rank_values[i] for rank_values in gathered) for i in range(len(values))]

    def prepare(self, session, effort, override, name, inp, out):
        b = self.b12x

        def prime(state):
            if state.query.surface.startswith("PCIeTwoShotBF16."):
                return b["twoshot"].prepared_call(state, payload=inp, out=out)
            return b["oneshot"]._prepare_plain_call(state, inp=inp, out=out)

        with b["effort"](effort):
            plan = self.manager.plan(inp, out=out, override=override)
            request = plan.request(name=name, prepare_call=prime,
                                   collective=b["CollectiveRequirement"](key=name, ranks=tuple(range(self.world))))
            prepare_lockstep(session, (request,), self.group, self.world).close()
        return plan

    def capture(self, session, call, chain):
        torch = self.torch
        graph = torch.cuda.CUDAGraph()
        self.dist.barrier(group=self.group)
        if session is None:
            with torch.cuda.graph(graph):
                for _ in range(chain):
                    call()
        else:
            with session.capture(), self.manager.capture(), torch.cuda.graph(graph):
                for _ in range(chain):
                    call()
        return graph

    def time_graph(self, graph, replays: int, calls: int) -> float:
        torch = self.torch
        self.dist.barrier(group=self.group)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / (replays * calls)

    def race_sample(self, graph) -> float:
        """One call after an L2 flush, timed alone (what the joint tuner's race measures)."""
        torch = self.torch
        self.flush.zero_()
        torch.cuda.synchronize()
        self.dist.barrier(group=self.group)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000

    # -- one message size ------------------------------------------------------------------------

    def screen(self, rows, inp, out, query):
        b, args = self.b12x, self.args
        configs = [c for c in dict.fromkeys(b["configs"](query))
                   if c.implementation in TRANSPORTS and family(c.implementation) != "nccl"]
        screened, started = [], time.monotonic()
        for index, config in enumerate(configs):
            entry = {"implementation": config.implementation, "threads": config.threads, "blocks": config.blocks}
            try:
                plan = self.prepare(self.pinned, "default", config, f"screen-{rows}-{index}", inp, out)
            except RuntimeError as error:
                entry["error"] = str(error)[:300]
                screened.append(entry)
                continue
            graph = self.capture(self.pinned, lambda plan=plan: self.manager.all_reduce(inp, out=out, plan=plan),
                                 args.screen_chain)
            for _ in range(2):
                graph.replay()
            self.torch.cuda.synchronize()
            values = [self.time_graph(graph, args.screen_replays, args.screen_chain) for _ in range(3)]
            entry["us"] = statistics.median(self.slowest(values))
            graph.reset()
            del graph, plan
            screened.append(entry)
        return configs, screened, time.monotonic() - started

    def measure_size(self, rows: int) -> dict:
        torch, args = self.torch, self.args
        nbytes = rows * args.hidden * 2
        record = {"rows": rows, "bytes": nbytes}
        inp = torch.empty(nbytes // 2, dtype=torch.bfloat16, device=self.device)
        out = torch.empty_like(inp)
        inp.fill_(self.rank + 1)
        arms = []
        if self.manager is not None:
            query = self.manager.plan(inp, out=out).query
            if not isinstance(query, self.b12x["CollectiveQuery"]):
                raise SystemExit("this FlashInfer does not plan all-reduces as tuned collectives")
            record["default"] = {"implementation": query.default.implementation, "threads": query.default.threads,
                                 "blocks": query.default.blocks}
            configs, screened, seconds = self.screen(rows, inp, out, query)
            record.update(screen=screened, screen_seconds=seconds)
            best = {}
            for entry, config in zip(screened, configs):
                if "us" in entry and (config.implementation not in best or entry["us"] < best[config.implementation][0]):
                    best[config.implementation] = (entry["us"], config)
            arms += [{"label": f"{c.implementation}/{c.threads}x{c.blocks}", "kind": "best",
                      "implementation": c.implementation, "threads": c.threads, "blocks": c.blocks, "override": c}
                     for _, c in best.values()]
            arms += [{"label": f"tuned:{effort}", "kind": "tuned", "effort": effort} for effort in args.efforts]
        arms.append({"label": args.nccl_label, "kind": "nccl", "implementation": args.nccl_label})

        steady, race = {}, {}
        for index, arm in enumerate(arms):
            if arm["kind"] == "nccl":
                buffer = self.nccl_buffer[: nbytes // 2]
                call = lambda buffer=buffer: self.dist.all_reduce(buffer, group=self.nccl)  # noqa: E731
                session = None
            else:
                session = self.pinned if arm["kind"] == "best" else self.tuned[arm["effort"]]
                effort = "default" if arm["kind"] == "best" else arm["effort"]
                started = time.monotonic()
                plan = self.prepare(session, effort, arm.pop("override", None), f"arm-{rows}-{index}", inp, out)
                arm["prepare_seconds"] = time.monotonic() - started
                if arm["kind"] == "tuned":
                    config = plan.selection.config
                    arm.update(implementation=config.implementation, threads=config.threads, blocks=config.blocks,
                               selection_source=str(plan.selection.source))
                arm["surface"] = self.b12x["require_prepared"](plan, "comm.pcie", self.device).query.surface
                call = lambda plan=plan: self.manager.all_reduce(inp, out=out, plan=plan)  # noqa: E731
            arm["family"] = family(arm["implementation"])
            steady[arm["label"]] = self.capture(session, call, args.chain)
            race[arm["label"]] = self.capture(session, call, 1)

        for step in range(2):
            generator = torch.Generator().manual_seed(args.seed + 1000 * rows + step)
            operands = torch.randn((self.world, inp.numel()), generator=generator).bfloat16()
            reference = operands.float().sum(0)
            inp.copy_(operands[self.rank])
            for arm in arms:
                if arm["kind"] == "nccl":
                    continue
                out.fill_(float("nan"))
                torch.cuda.synchronize()
                self.dist.barrier(group=self.group)
                steady[arm["label"]].replay()
                torch.cuda.synchronize()
                error = (out.float().cpu() - reference).abs()
                ok = bool(torch.isfinite(out).all().item()) and bool(
                    (error <= reference.abs() * 2.0 ** -7 + 1e-5).all().item())
                arm["correct"] = arm.get("correct", True) and ok

        labels = [arm["label"] for arm in arms]
        passes = []
        for first_reversed in (False, True):
            for graph in steady.values():  # every pass starts warm (NCCL settles over its first replays)
                for _ in range(args.warmup):
                    graph.replay()
            torch.cuda.synchronize()
            samples = {label: [] for label in labels}
            for sample in range(args.samples):
                order = labels if (sample % 2 == 0) != first_reversed else list(reversed(labels))
                for label in order:
                    samples[label].append(self.time_graph(steady[label], args.replays, args.chain))
            passes.append({label: self.slowest(values) for label, values in samples.items()})
        races = {label: [] for label in labels}
        for _ in range(args.race_samples):
            for label in labels:
                races[label].append(self.race_sample(race[label]))
        races = {label: self.slowest(values) for label, values in races.items()}
        flags = [None] * self.world
        self.dist.all_gather_object(flags, {arm["label"]: arm.get("correct") for arm in arms}, group=self.group)
        for arm in arms:
            label = arm["label"]
            arm.update(steady_us=statistics.median(passes[0][label]), steady_samples=passes[0][label],
                       pass2_us=statistics.median(passes[1][label]), pass2_samples=passes[1][label],
                       race_us=statistics.median(races[label]), race_samples=races[label],
                       correct=all(item[label] in (True, None) for item in flags))
            if arm["kind"] == "nccl":
                arm.pop("correct")
        for graph in (*steady.values(), *race.values()):
            graph.reset()
        record["arms"] = arms
        return record

    def run(self) -> None:
        import torch

        result = {"world_size": self.world, "hidden": self.args.hidden, "chain": self.args.chain,
                  "replays": self.args.replays, "samples": self.args.samples, "race_samples": self.args.race_samples,
                  "efforts": self.args.efforts, "nccl_label": self.args.nccl_label,
                  "nccl_env": {key: value for key, value in sorted(os.environ.items()) if key.startswith("NCCL_")},
                  "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                  "device": torch.cuda.get_device_name(self.device), "sizes": []}
        try:
            result["nccl_version"] = ".".join(map(str, torch.cuda.nccl.version()))
        except Exception:  # noqa: BLE001 - provenance only
            pass
        started = time.monotonic()
        try:
            for rows in self.rows:
                record = self.measure_size(rows)
                result["sizes"].append(record)
                if self.rank == 0:
                    Path(self.args.result).write_text(json.dumps(result))
                best = [arm for arm in record["arms"] if arm["kind"] == "best"]
                fastest = min(best or record["arms"], key=lambda arm: arm["steady_us"])
                tuned = "  ".join(f"tuner {arm['effort']} {arm['implementation']}/{arm['threads']}x{arm['blocks']} "
                                  f"{arm['steady_us']:.2f}" for arm in record["arms"] if arm["kind"] == "tuned")
                screened = len(record.get("screen") or [])
                self.log(f"  {rows:4d} rows ({rows * self.args.hidden * 2 // 1024:5d} KiB): "
                         + (f"{screened} configurations, " if screened else "")
                         + f"fastest {fastest['label']} {fastest['steady_us']:.2f} us"
                         + (f"; {tuned}" if tuned else "") + f"  [{time.monotonic() - started:.0f} s]")
        finally:
            torch.cuda.synchronize()
            if self.manager is not None:
                for session in (self.pinned, *self.tuned.values()):
                    session.close()
                self.manager.close()
        self.dist.barrier(group=self.group)
        self.dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--efforts", default="default,audit")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--chain", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--race-samples", type=int, default=15)
    parser.add_argument("--screen-chain", type=int, default=8)
    parser.add_argument("--screen-replays", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nccl-only", action="store_true")
    parser.add_argument("--nccl-label", default="nccl")
    args = parser.parse_args()
    args.efforts = [] if args.nccl_only else [effort for effort in args.efforts.split(",") if effort]
    Worker(args).run()


if __name__ == "__main__":
    main()
