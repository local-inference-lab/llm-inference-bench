"""Peer read stalls: does a GPU reading from its peers ever wait for milliseconds?

Run as ``python -m lil_bench.collectives.stalls --result FILE`` with CUDA_VISIBLE_DEVICES set to the
ranks under test. One process drives every GPU (peer access enabled). Each GPU runs one persistent
NVRTC kernel for a fixed time: every iteration moves one all-reduce-sized message (default 80 KiB, a
DeepSeek V4.1 TP4 decode step) and stamps ``%globaltimer`` after a grid-wide barrier, so every single
operation's duration is recorded. All GPUs run at once, like the ranks of an all-reduce.

Patterns (each GPU at the same time):

* ``read_ring``: GPU r reads the message from GPU r+1. A stall belongs to that target GPU.
* ``read_all``: GPU r reads the message from every peer (the traffic of one-shot "pull" all-reduce).
* ``write_ring``: GPU r writes the message to GPU r+1 and fences it (posted writes; the control).

On some CPU fabrics (GPUs behind different IO-die quadrants of one CPU), peer reads stall for 1-3 ms
in periodic windows while posted writes never do. Pull transports then pay those stalls on average
even though their median looks best, and a short tuning race cannot see them. Each read pattern runs
for several seconds so that periodic windows (every ~2.3 s on AMD EPYC Turin) are always covered.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np

KERNEL = r"""
struct alignas(16) V { unsigned int x, y, z, w; };
__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ V load_cv(const V* p) {
  V v;
  asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
extern "C" __global__ void stamp(unsigned long long* out) { *out = now_ns(); }
extern "C" __global__ void stall_probe(
    const V* s0, const V* s1, const V* s2, const V* s3, const V* s4, const V* s5, const V* s6,
    V* d0, V* d1, V* d2, V* d3, V* d4, V* d5, V* d6, unsigned long long packs,
    unsigned long long duration_ns, unsigned long long max_iters, unsigned long long* stamps,
    unsigned int* sync, int writes) {
  const V* s; V* d;
  switch (blockIdx.y) {
    case 0: s = s0; d = d0; break; case 1: s = s1; d = d1; break; case 2: s = s2; d = d2; break;
    case 3: s = s3; d = d3; break; case 4: s = s4; d = d4; break; case 5: s = s5; d = d5; break;
    default: s = s6; d = d6; break;
  }
  const unsigned int blocks = gridDim.x * gridDim.y;
  volatile unsigned int* vsync = sync;
  const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (unsigned long long it = 0; it < max_iters; ++it) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < packs; i += stride)
      d[i] = load_cv(s + i);
    if (writes) __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned int generation = vsync[1];
      __threadfence();
      if (atomicAdd(&sync[0], 1u) == blocks - 1) {
        sync[0] = 0;
        const unsigned long long t = now_ns();
        stamps[it] = t;
        if (it + 1 >= max_iters || t - stamps[0] >= duration_ns) vsync[2] = 1;
        vsync[3] = (unsigned int)(it + 1);
        __threadfence();
        atomicAdd(&sync[1], 1u);
      } else {
        while (vsync[1] == generation) {}
      }
      __threadfence();
    }
    __syncthreads();
    if (vsync[2]) break;
  }
}
"""
MAX_SOURCES = 7
WARMUP = 50  # first operations of every GPU are excluded (cold paths, launch skew)
BUCKET_MS = 50  # timeline resolution
EPISODE_GAP_MS = 50  # stalls closer than this belong to one episode
STALL_FLOOR_US = 100.0  # an operation is a stall above max(this, STALL_FACTOR x median)
STALL_FACTOR = 10.0
PATTERNS = (("read_ring", 1.0), ("read_all", 1.0), ("write_ring", 0.5))  # (name, share of --seconds)


def _check(result, what):
    if int(result[0]) != 0:
        raise RuntimeError(f"{what} failed: {result[0]}")
    return result[1] if len(result) == 2 else None


class Probe:
    def __init__(self, kib: int, ctas: int, threads: int, max_iters: int):
        import torch
        from cuda.bindings import driver as cu
        from cuda.bindings import nvrtc
        from cuda.bindings import runtime as rt

        self.torch, self.cu = torch, cu
        self.count = torch.cuda.device_count()
        if not 2 <= self.count <= MAX_SOURCES + 1:
            raise RuntimeError(f"the stall probe needs 2-{MAX_SOURCES + 1} GPUs, got {self.count}")
        for device in range(self.count):
            torch.cuda.set_device(device)
            torch.empty(1, device=device)
        for a in range(self.count):
            torch.cuda.set_device(a)
            for b in range(self.count):
                if a == b:
                    continue
                if not torch.cuda.can_device_access_peer(a, b):
                    raise RuntimeError(f"GPU {a} cannot access GPU {b} (no peer access)")
                err = rt.cudaDeviceEnablePeerAccess(b, 0)[0]
                if int(err) not in (0, int(rt.cudaError_t.cudaErrorPeerAccessAlreadyEnabled)):
                    raise RuntimeError(f"enabling peer access {a}->{b} failed: {err}")
        major, minor = torch.cuda.get_device_capability(0)
        program = _check(nvrtc.nvrtcCreateProgram(KERNEL.encode(), b"stall_probe.cu", 0, [], []), "nvrtcCreateProgram")
        options = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17"]
        if int(nvrtc.nvrtcCompileProgram(program, len(options), options)[0]) != 0:
            size = _check(nvrtc.nvrtcGetProgramLogSize(program), "nvrtcGetProgramLogSize")
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(f"stall probe compilation failed: {log.decode(errors='replace')}")
        size = _check(nvrtc.nvrtcGetCUBINSize(program), "nvrtcGetCUBINSize")
        cubin = b" " * size
        _check((nvrtc.nvrtcGetCUBIN(program, cubin)[0], None), "nvrtcGetCUBIN")
        self.probe, self.stamp_fn = {}, {}
        for device in range(self.count):
            torch.cuda.set_device(device)
            torch.cuda.synchronize()
            module = _check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
            self.probe[device] = _check(cu.cuModuleGetFunction(module, b"stall_probe"), "cuModuleGetFunction")
            self.stamp_fn[device] = _check(cu.cuModuleGetFunction(module, b"stamp"), "cuModuleGetFunction")
        self.nbytes = kib << 10
        self.ctas, self.threads, self.max_iters = ctas, threads, max_iters
        words = self.nbytes // 4
        self.src = {d: torch.full((words,), d + 1, dtype=torch.int32, device=d) for d in range(self.count)}
        self.land = {d: torch.zeros((self.count, words), dtype=torch.int32, device=d) for d in range(self.count)}
        self.stamps = {d: torch.zeros(max_iters, dtype=torch.int64, device=d) for d in range(self.count)}
        self.sync = {d: torch.zeros(4, dtype=torch.int32, device=d) for d in range(self.count)}
        self.types = (ctypes.c_void_p,) * (2 * MAX_SOURCES) + (ctypes.c_ulonglong,) * 3 + \
            (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)

    def _stream(self, device):
        return self.cu.CUstream(self.torch.cuda.current_stream(device).cuda_stream)

    def clock_offsets(self) -> dict[int, int]:
        """GPU %globaltimer minus host monotonic ns, from the tightest of several stamp round trips."""
        torch, offsets = self.torch, {}
        for device in range(self.count):
            out = torch.zeros(1, dtype=torch.int64, device=device)
            best = None
            with torch.cuda.device(device):
                for _ in range(8):
                    before = time.perf_counter_ns()
                    _check((self.cu.cuLaunchKernel(self.stamp_fn[device], 1, 1, 1, 1, 1, 1, 0, self._stream(device),
                                                   ((out.data_ptr(),), (ctypes.c_void_p,)), 0)[0], None), "stamp")
                    torch.cuda.synchronize(device)
                    after = time.perf_counter_ns()
                    if best is None or after - before < best[0]:
                        best = (after - before, int(out.item()) - (before + after) // 2)
            offsets[device] = best[1]
        return offsets

    def plan(self, name: str) -> dict[int, tuple[list[tuple[int, int]], bool]]:
        """Per GPU: (source pointer, destination pointer) pairs and whether the destinations are remote."""
        n, plan = self.count, {}
        for r in range(n):
            if name == "read_ring":
                p = (r + 1) % n
                plan[r] = ([(self.src[p].data_ptr(), self.land[r][p].data_ptr())], False)
            elif name == "read_all":
                plan[r] = ([(self.src[p].data_ptr(), self.land[r][p].data_ptr()) for p in range(n) if p != r], False)
            elif name == "write_ring":
                p = (r + 1) % n
                plan[r] = ([(self.src[r].data_ptr(), self.land[p][r].data_ptr())], True)
            else:
                raise ValueError(name)
        return plan

    def run(self, name: str, seconds: float) -> dict[int, np.ndarray]:
        torch, plan = self.torch, self.plan(name)
        for device in range(self.count):
            self.sync[device].zero_()
            torch.cuda.synchronize(device)
        for device, (pairs, writes) in plan.items():
            sources = [p[0] for p in pairs] + [pairs[0][0]] * (MAX_SOURCES - len(pairs))
            targets = [p[1] for p in pairs] + [pairs[0][1]] * (MAX_SOURCES - len(pairs))
            grid_x = max(1, self.ctas // len(pairs))
            args = tuple(sources + targets) + (self.nbytes // 16, int(seconds * 1e9), self.max_iters,
                                               self.stamps[device].data_ptr(), self.sync[device].data_ptr(),
                                               1 if writes else 0)
            with torch.cuda.device(device):
                _check((self.cu.cuLaunchKernel(self.probe[device], grid_x, len(pairs), 1, self.threads, 1, 1, 0,
                                               self._stream(device), (args, self.types), 0)[0], None), "stall_probe")
        for device in range(self.count):
            torch.cuda.synchronize(device)
        out = {}
        for device in range(self.count):
            done = int(self.sync[device][3].item())
            out[device] = self.stamps[device][:done].cpu().numpy().astype(np.int64)
        return out


def periodicity(stalled: np.ndarray) -> tuple[float | None, float | None, list[float]]:
    """Dominant repeat period of stalled timeline buckets (autocorrelation peak at lags of 0.4 s up to half
    the run) and the lengths of stall windows (stalled buckets at most two buckets apart)."""
    x = stalled.astype(np.float64)
    windows: list[float] = []
    run_start = last = None
    for index in np.flatnonzero(x):
        if last is not None and index - last > 2:
            windows.append((last - run_start + 1) * BUCKET_MS)
            run_start = index
        run_start = index if run_start is None else run_start
        last = index
    if last is not None:
        windows.append((last - run_start + 1) * BUCKET_MS)
    low, high = int(np.ceil(400 / BUCKET_MS)), len(x) // 2
    if x.sum() < 2 or high <= low:
        return None, None, windows
    best_lag, best = None, 0.0
    for lag in range(low, high + 1):
        a, b = x[:-lag], x[lag:]
        norm = np.sqrt((a * a).sum() * (b * b).sum())
        value = float((a * b).sum() / norm) if norm else 0.0
        if value > best:
            best_lag, best = lag, value
    if best_lag is None or best < 0.5:
        return None, best or None, windows
    return best_lag * BUCKET_MS / 1000, best, windows


def gpu_stats(stamps: np.ndarray, offset: int, origin: int) -> dict:
    """Per-operation statistics of one GPU; times are ns of the host clock relative to ``origin``."""
    if len(stamps) <= WARMUP + 2:
        return {"ops": 0}
    ops_us = np.diff(stamps)[WARMUP:] / 1e3
    ends = (stamps[WARMUP + 1:] - offset - origin)
    p50 = float(np.median(ops_us))
    threshold = max(STALL_FLOOR_US, STALL_FACTOR * p50)
    stalled = ops_us > threshold
    total = float(ops_us.sum())
    episodes: list[dict] = []
    for index in np.flatnonzero(stalled):
        start, end = int(ends[index] - ops_us[index] * 1e3), int(ends[index])
        if episodes and start - episodes[-1]["end_ns"] < EPISODE_GAP_MS * 1e6:
            episode = episodes[-1]
            episode["end_ns"], episode["ops"] = end, episode["ops"] + 1
            episode["max_us"] = max(episode["max_us"], float(ops_us[index]))
        else:
            episodes.append({"start_ns": start, "end_ns": end, "ops": 1, "max_us": float(ops_us[index])})
    span = int(ends[-1]) if len(ends) else 0
    buckets = np.zeros(max(1, span // (BUCKET_MS * 1_000_000) + 1))
    np.maximum.at(buckets, np.clip(ends // (BUCKET_MS * 1_000_000), 0, len(buckets) - 1).astype(np.int64), ops_us)
    period, strength, windows = periodicity(buckets > threshold)
    return {
        "ops": int(len(ops_us)), "seconds": total / 1e6, "p50_us": p50,
        "p99_us": float(np.percentile(ops_us, 99)), "p999_us": float(np.percentile(ops_us, 99.9)),
        "max_us": float(ops_us.max()), "mean_us": float(ops_us.mean()),
        "stall_threshold_us": threshold, "stalls": int(stalled.sum()),
        "over_100us": int((ops_us > 100).sum()), "over_1ms": int((ops_us > 1000).sum()),
        "lost_fraction": float((ops_us[stalled] - p50).sum() / total) if total else 0.0,
        "episodes": len(episodes), "stalled_fraction": float((buckets > threshold).mean()),
        "windows": len(windows), "window_ms": float(np.median(windows)) if windows else None,
        "period_s": period, "period_strength": strength,
        "episode_starts_s": [round(e["start_ns"] / 1e9, 3) for e in episodes[:64]],
        "timeline_max_us": [round(float(v), 1) for v in buckets],
    }


def measure(kib: int, seconds: float, ctas: int, threads: int) -> dict:
    probe = Probe(kib, ctas, threads, max_iters=4_000_000)
    offsets = probe.clock_offsets()
    patterns = {}
    for name, share in PATTERNS:
        if name == "read_all" and probe.count == 2:
            continue  # identical to read_ring with two GPUs
        duration = max(2.0, seconds * share)
        started = time.perf_counter_ns()
        stamps = probe.run(name, duration)
        origin = min(int(s[0]) - offsets[d] for d, s in stamps.items() if len(s))
        patterns[name] = {"seconds": duration, "host_start_ns": started,
                          "gpus": [gpu_stats(stamps[d], offsets[d], origin) for d in range(probe.count)]}
    return {"bytes": probe.nbytes, "ctas": ctas, "threads": threads, "bucket_ms": BUCKET_MS,
            "stall_rule": {"floor_us": STALL_FLOOR_US, "factor": STALL_FACTOR, "episode_gap_ms": EPISODE_GAP_MS},
            "patterns": patterns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", required=True)
    parser.add_argument("--kib", type=int, default=80, help="message size per operation (default 80 KiB)")
    parser.add_argument("--seconds", type=float, default=6.0, help="duration of each read pattern (default 6 s)")
    parser.add_argument("--ctas", type=int, default=16)
    parser.add_argument("--threads", type=int, default=256)
    args = parser.parse_args()
    Path(args.result).write_text(json.dumps(measure(args.kib, args.seconds, args.ctas, args.threads)))


if __name__ == "__main__":
    main()
