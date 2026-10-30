#!/usr/bin/env python3
"""lil_read_stalls.py: do peer reads between these GPUs stall?

One file: Python 3.8+ standard library and the NVIDIA driver (libcuda) only. No CUDA toolkit,
PyTorch or pip packages; the probe kernel ships as PTX and the driver compiles it.

    python3 lil_read_stalls.py                    # GPUs 0,1,2,3
    python3 lil_read_stalls.py --gpus 0,4,8,12    # nvidia-smi indices (PCI bus order), 2-8 GPUs
    python3 lil_read_stalls.py --json stalls.json # also save every number

Every GPU runs one persistent kernel at the same time. Each iteration moves one all-reduce-sized
message (default 80 KiB, one DeepSeek V4.1 TP4 decode step) and stamps %globaltimer after a
grid-wide barrier, so every single operation is timed:

  read_ring   GPU r reads the message from GPU r+1
  read_all    GPU r reads the message from every peer (one-shot "pull" all-reduce traffic)
  write_ring  GPU r writes the message to GPU r+1 and fences it (posted writes; the control)

On some CPU fabrics (for example GPUs behind different IO-die quadrants of an AMD EPYC), peer reads
stall for 1-3 ms in periodic windows while posted writes never do. Pull all-reduce transports then
pay those stalls on average although their median looks best. The read patterns run long enough
(6 s each) to cover several such windows.

Exit code: 0 no read stalls, 1 read stalls, 2 the test could not run.
Part of lil-bench (https://github.com/local-inference-lab/llm-inference-bench);
`lil-bench collectives --stalls-only` runs the same test and uploads the result.
"""

from __future__ import annotations

import argparse
import array
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time

# The probe kernel; PTX below is nvcc -arch=compute_80 -ptx of this source (PTX ISA lowered to 7.0).
CUDA_SOURCE = r"""
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
PTX = r""".version 7.0
.target sm_80
.address_size 64

	// .globl	stamp

.visible .entry stamp(
	.param .u64 stamp_param_0
)
{
	.reg .b64 	%rd<4>;


	ld.param.u64 	%rd2, [stamp_param_0];
	cvta.to.global.u64 	%rd3, %rd2;
	// begin inline asm
	mov.u64 %rd1, %globaltimer;
	// end inline asm
	st.global.u64 	[%rd3], %rd1;
	ret;

}
	// .globl	stall_probe
.visible .entry stall_probe(
	.param .u64 stall_probe_param_0,
	.param .u64 stall_probe_param_1,
	.param .u64 stall_probe_param_2,
	.param .u64 stall_probe_param_3,
	.param .u64 stall_probe_param_4,
	.param .u64 stall_probe_param_5,
	.param .u64 stall_probe_param_6,
	.param .u64 stall_probe_param_7,
	.param .u64 stall_probe_param_8,
	.param .u64 stall_probe_param_9,
	.param .u64 stall_probe_param_10,
	.param .u64 stall_probe_param_11,
	.param .u64 stall_probe_param_12,
	.param .u64 stall_probe_param_13,
	.param .u64 stall_probe_param_14,
	.param .u64 stall_probe_param_15,
	.param .u64 stall_probe_param_16,
	.param .u64 stall_probe_param_17,
	.param .u64 stall_probe_param_18,
	.param .u32 stall_probe_param_19
)
{
	.reg .pred 	%p<20>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<49>;


	ld.param.u64 	%rd45, [stall_probe_param_0];
	ld.param.u64 	%rd14, [stall_probe_param_1];
	ld.param.u64 	%rd15, [stall_probe_param_2];
	ld.param.u64 	%rd16, [stall_probe_param_3];
	ld.param.u64 	%rd17, [stall_probe_param_4];
	ld.param.u64 	%rd18, [stall_probe_param_5];
	ld.param.u64 	%rd19, [stall_probe_param_6];
	ld.param.u64 	%rd46, [stall_probe_param_7];
	ld.param.u64 	%rd21, [stall_probe_param_8];
	ld.param.u64 	%rd22, [stall_probe_param_9];
	ld.param.u64 	%rd23, [stall_probe_param_10];
	ld.param.u64 	%rd24, [stall_probe_param_11];
	ld.param.u64 	%rd25, [stall_probe_param_12];
	ld.param.u64 	%rd26, [stall_probe_param_13];
	ld.param.u64 	%rd27, [stall_probe_param_14];
	ld.param.u64 	%rd28, [stall_probe_param_15];
	ld.param.u64 	%rd29, [stall_probe_param_16];
	ld.param.u64 	%rd30, [stall_probe_param_17];
	ld.param.u64 	%rd31, [stall_probe_param_18];
	ld.param.u32 	%r4, [stall_probe_param_19];
	cvta.to.global.u64 	%rd1, %rd30;
	cvta.to.global.u64 	%rd2, %rd31;
	mov.u32 	%r5, %ctaid.y;
	setp.gt.s32 	%p1, %r5, 2;
	@%p1 bra 	$L__BB1_4;

	setp.eq.s32 	%p5, %r5, 0;
	@%p5 bra 	$L__BB1_8;

	setp.eq.s32 	%p6, %r5, 1;
	mov.u64 	%rd45, %rd14;
	mov.u64 	%rd46, %rd21;
	@%p6 bra 	$L__BB1_8;

	setp.eq.s32 	%p7, %r5, 2;
	mov.u64 	%rd45, %rd15;
	mov.u64 	%rd46, %rd22;
	@%p7 bra 	$L__BB1_8;
	bra.uni 	$L__BB1_7;

$L__BB1_4:
	setp.eq.s32 	%p2, %r5, 3;
	mov.u64 	%rd45, %rd16;
	mov.u64 	%rd46, %rd23;
	@%p2 bra 	$L__BB1_8;

	setp.eq.s32 	%p3, %r5, 4;
	mov.u64 	%rd45, %rd17;
	mov.u64 	%rd46, %rd24;
	@%p3 bra 	$L__BB1_8;

	setp.ne.s32 	%p4, %r5, 5;
	mov.u64 	%rd45, %rd18;
	mov.u64 	%rd46, %rd25;
	@%p4 bra 	$L__BB1_7;
	bra.uni 	$L__BB1_8;

$L__BB1_7:
	mov.u64 	%rd45, %rd19;
	mov.u64 	%rd46, %rd26;

$L__BB1_8:
	setp.eq.s64 	%p8, %rd29, 0;
	@%p8 bra 	$L__BB1_24;

	mov.u32 	%r6, %nctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %ctaid.x;
	mul.wide.u32 	%rd33, %r8, %r7;
	mov.u32 	%r1, %tid.x;
	cvt.u64.u32 	%rd34, %r1;
	add.s64 	%rd5, %rd33, %rd34;
	mov.u32 	%r9, %nctaid.y;
	mad.lo.s32 	%r2, %r6, %r9, -1;
	mul.wide.u32 	%rd6, %r6, %r7;
	cvta.to.global.u64 	%rd7, %rd46;
	mov.u64 	%rd47, 0;

$L__BB1_10:
	setp.ge.u64 	%p9, %rd5, %rd27;
	@%p9 bra 	$L__BB1_13;

	mov.u64 	%rd48, %rd5;

$L__BB1_12:
	shl.b64 	%rd36, %rd48, 4;
	add.s64 	%rd35, %rd45, %rd36;
	// begin inline asm
	ld.global.cv.v4.u32 {%r10,%r11,%r12,%r13}, [%rd35];
	// end inline asm
	add.s64 	%rd37, %rd7, %rd36;
	st.global.v4.u32 	[%rd37], {%r10, %r11, %r12, %r13};
	add.s64 	%rd48, %rd48, %rd6;
	setp.lt.u64 	%p10, %rd48, %rd27;
	@%p10 bra 	$L__BB1_12;

$L__BB1_13:
	setp.eq.s32 	%p11, %r4, 0;
	@%p11 bra 	$L__BB1_15;

	membar.sys;

$L__BB1_15:
	setp.ne.s32 	%p12, %r1, 0;
	bar.sync 	0;
	@%p12 bra 	$L__BB1_23;

	ld.volatile.global.u32 	%r3, [%rd2+4];
	membar.gl;
	atom.global.add.u32 	%r14, [%rd2], 1;
	setp.eq.s32 	%p13, %r14, %r2;
	@%p13 bra 	$L__BB1_18;
	bra.uni 	$L__BB1_17;

$L__BB1_18:
	mov.u32 	%r16, 0;
	st.global.u32 	[%rd2], %r16;
	// begin inline asm
	mov.u64 %rd38, %globaltimer;
	// end inline asm
	shl.b64 	%rd39, %rd47, 3;
	add.s64 	%rd40, %rd1, %rd39;
	st.global.u64 	[%rd40], %rd38;
	add.s64 	%rd41, %rd47, 1;
	setp.ge.u64 	%p15, %rd41, %rd29;
	@%p15 bra 	$L__BB1_20;

	ld.global.u64 	%rd42, [%rd1];
	sub.s64 	%rd43, %rd38, %rd42;
	setp.lt.u64 	%p16, %rd43, %rd28;
	@%p16 bra 	$L__BB1_21;

$L__BB1_20:
	mov.u32 	%r17, 1;
	st.volatile.global.u32 	[%rd2+8], %r17;

$L__BB1_21:
	cvt.u32.u64 	%r18, %rd47;
	add.s32 	%r19, %r18, 1;
	st.volatile.global.u32 	[%rd2+12], %r19;
	membar.gl;
	add.s64 	%rd44, %rd2, 4;
	atom.global.add.u32 	%r20, [%rd44], 1;
	bra.uni 	$L__BB1_22;

$L__BB1_17:
	ld.volatile.global.u32 	%r15, [%rd2+4];
	setp.eq.s32 	%p14, %r15, %r3;
	@%p14 bra 	$L__BB1_17;

$L__BB1_22:
	membar.gl;

$L__BB1_23:
	bar.sync 	0;
	ld.volatile.global.u32 	%r21, [%rd2+8];
	setp.eq.s32 	%p17, %r21, 0;
	add.s64 	%rd47, %rd47, 1;
	setp.lt.u64 	%p18, %rd47, %rd29;
	and.pred  	%p19, %p17, %p18;
	@%p19 bra 	$L__BB1_10;

$L__BB1_24:
	ret;

}
"""

WARMUP = 50  # first operations of every GPU are excluded (cold paths, launch skew)
BUCKET_MS = 50  # timeline resolution
EPISODE_GAP_MS = 50  # stalls closer than this belong to one episode
STALL_FLOOR_US = 100.0  # an operation is a stall above max(this, STALL_FACTOR x median)
STALL_FACTOR = 10.0
MAX_SOURCES = 7
MAX_ITERS = 4_000_000
SEVERE_OPS, SEVERE_LOST, MILD_OPS = 3, 0.05, 10
ORDER = ("none", "mild", "severe")


class CudaError(RuntimeError):
    pass


class Driver:
    def __init__(self):
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise CudaError("the NVIDIA driver library libcuda.so.1 was not found")

    def __call__(self, name: str, *args, ok=(0,)):
        code = getattr(self.lib, name)(*args)
        if code not in ok:
            text = ctypes.c_char_p()
            self.lib.cuGetErrorString(code, ctypes.byref(text))
            raise CudaError(f"{name} failed: {code} {(text.value or b'').decode()}")
        return code


class Gpu:
    def __init__(self, cu: Driver, ordinal: int, nbytes: int, count: int):
        self.cu, self.ordinal = cu, ordinal
        self.device = ctypes.c_int()
        cu("cuDeviceGet", ctypes.byref(self.device), ordinal)
        name, bus = ctypes.create_string_buffer(128), ctypes.create_string_buffer(32)
        cu("cuDeviceGetName", name, 128, self.device)
        cu("cuDeviceGetPCIBusId", bus, 32, self.device)
        self.name, self.bus = name.value.decode(), bus.value.decode().lower()
        self.context = ctypes.c_void_p()
        cu("cuDevicePrimaryCtxRetain", ctypes.byref(self.context), self.device)
        self.use()
        module = ctypes.c_void_p()
        cu("cuModuleLoadData", ctypes.byref(module), ctypes.c_char_p(PTX.encode()))
        self.probe, self.stamp = ctypes.c_void_p(), ctypes.c_void_p()
        cu("cuModuleGetFunction", ctypes.byref(self.probe), module, b"stall_probe")
        cu("cuModuleGetFunction", ctypes.byref(self.stamp), module, b"stamp")
        self.src = self.alloc(nbytes, self.ordinal + 1)
        self.land = self.alloc(nbytes * count, 0)  # one landing slot per source / writer GPU
        self.stamps = self.alloc(MAX_ITERS * 8, 0)
        self.sync = self.alloc(16, 0)
        self.clock = self.alloc(8, 0)

    def use(self):
        self.cu("cuCtxSetCurrent", self.context)

    def alloc(self, nbytes: int, fill: int) -> int:
        pointer = ctypes.c_uint64()
        self.cu("cuMemAlloc_v2", ctypes.byref(pointer), ctypes.c_size_t(nbytes))
        self.cu("cuMemsetD32_v2", pointer, ctypes.c_uint(fill), ctypes.c_size_t(nbytes // 4))
        return pointer.value

    def launch(self, function, grid, block, values):
        holders = [ctypes.c_int(v) if isinstance(v, bool) else ctypes.c_uint64(v) for v in values]
        params = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.pointer(h), ctypes.c_void_p) for h in holders])
        self.use()
        self.cu("cuLaunchKernel", function, grid[0], grid[1], 1, block, 1, 1, 0, None, params, None)

    def read_u64(self, pointer: int, count: int) -> array.array:
        out = (ctypes.c_uint64 * max(1, count))()
        self.use()
        self.cu("cuMemcpyDtoH_v2", out, ctypes.c_uint64(pointer), ctypes.c_size_t(8 * count))
        return array.array("Q", bytes(out)[: 8 * count])

    def clock_offset(self) -> int:
        """GPU %globaltimer minus host monotonic ns, from the tightest of several round trips."""
        best = None
        for _ in range(8):
            before = time.perf_counter_ns()
            self.launch(self.stamp, (1, 1), 1, [self.clock])
            self.cu("cuCtxSynchronize")
            after = time.perf_counter_ns()
            if best is None or after - before < best[0]:
                best = (after - before, self.read_u64(self.clock, 1)[0] - (before + after) // 2)
        return best[1]


def run_pattern(gpus: list[Gpu], name: str, nbytes: int, seconds: float, ctas: int, threads: int) -> dict:
    n = len(gpus)
    for gpu in gpus:
        gpu.use()
        gpu.cu("cuMemsetD32_v2", ctypes.c_uint64(gpu.sync), ctypes.c_uint(0), ctypes.c_size_t(4))
        gpu.cu("cuCtxSynchronize")
    for r, gpu in enumerate(gpus):
        nxt = gpus[(r + 1) % n]
        if name == "read_ring":
            pairs, writes = [(nxt.src, gpu.land + nxt.ordinal * nbytes)], False
        elif name == "read_all":
            pairs = [(peer.src, gpu.land + peer.ordinal * nbytes) for peer in gpus if peer is not gpu]
            writes = False
        else:
            pairs, writes = [(gpu.src, nxt.land + gpu.ordinal * nbytes)], True
        sources = [p[0] for p in pairs] + [pairs[0][0]] * (MAX_SOURCES - len(pairs))
        targets = [p[1] for p in pairs] + [pairs[0][1]] * (MAX_SOURCES - len(pairs))
        gpu.launch(gpu.probe, (max(1, ctas // len(pairs)), len(pairs)), threads,
                   sources + targets + [nbytes // 16, int(seconds * 1e9), MAX_ITERS, gpu.stamps, gpu.sync, writes])
    stamps = {}
    for gpu in gpus:
        gpu.use()
        gpu.cu("cuCtxSynchronize")
    for gpu in gpus:
        done = gpu.read_u64(gpu.sync, 2)[1] >> 32  # u32 sync[3] = iterations completed
        stamps[gpu.ordinal] = gpu.read_u64(gpu.stamps, done)
    return stamps


def periodicity(stalled: list[bool]):
    """Repeat period of stalled timeline buckets (autocorrelation peak between 0.4 s and half the run)
    and the lengths of stall windows (stalled buckets at most two buckets apart)."""
    windows, start, last = [], None, None
    for index, value in enumerate(stalled):
        if not value:
            continue
        if last is not None and index - last > 2:
            windows.append((last - start + 1) * BUCKET_MS)
            start = index
        start = index if start is None else start
        last = index
    if last is not None:
        windows.append((last - start + 1) * BUCKET_MS)
    x = [1.0 if v else 0.0 for v in stalled]
    low, high = -(-400 // BUCKET_MS), len(x) // 2
    best_lag, best = None, 0.0
    if sum(x) >= 2:
        for lag in range(low, high + 1):
            a, b = x[:-lag], x[lag:]
            norm = (sum(v * v for v in a) * sum(v * v for v in b)) ** 0.5
            value = sum(p * q for p, q in zip(a, b)) / norm if norm else 0.0
            if value > best:
                best_lag, best = lag, value
    period = best_lag * BUCKET_MS / 1000 if best_lag and best >= 0.5 else None
    return period, windows


def gpu_stats(stamps: array.array, offset: int, origin: int) -> dict:
    if len(stamps) <= WARMUP + 2:
        return {"ops": 0}
    ops = [(stamps[i + 1] - stamps[i]) / 1e3 for i in range(WARMUP, len(stamps) - 1)]
    ends = [stamps[i] - offset - origin for i in range(WARMUP + 1, len(stamps))]
    ordered = sorted(ops)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]  # noqa: E731
    p50 = pick(0.5)
    threshold = max(STALL_FLOOR_US, STALL_FACTOR * p50)
    total = sum(ops)
    buckets = [0.0] * (max(ends) // (BUCKET_MS * 1_000_000) + 1)
    for end, op in zip(ends, ops):
        index = max(0, end // (BUCKET_MS * 1_000_000))
        if op > buckets[index]:
            buckets[index] = op
    period, windows = periodicity([v > threshold for v in buckets])
    stalls = [op for op in ops if op > threshold]
    return {"ops": len(ops), "p50_us": p50, "p99_us": pick(0.99), "p999_us": pick(0.999), "max_us": ordered[-1],
            "mean_us": total / len(ops), "stall_threshold_us": threshold, "stalls": len(stalls),
            "over_100us": sum(1 for op in ops if op > 100), "over_1ms": sum(1 for op in ops if op > 1000),
            "lost_fraction": sum(op - p50 for op in stalls) / total if total else 0.0,
            "period_s": period, "window_ms": statistics.median(windows) if windows else None,
            "timeline_max_us": [round(v, 1) for v in buckets]}


def pattern_summary(gpus: list[dict], indices: list[int]) -> dict:
    gpus = [dict(g, gpu=indices[i]) for i, g in enumerate(gpus) if g.get("ops")]
    if not gpus:
        return {}
    out = {"ops": sum(g["ops"] for g in gpus), "p50_us": statistics.median(g["p50_us"] for g in gpus),
           "p99_us": max(g["p99_us"] for g in gpus), "p999_us": max(g["p999_us"] for g in gpus),
           "max_us": max(g["max_us"] for g in gpus), "stalls": sum(g["stalls"] for g in gpus),
           "over_1ms": sum(g["over_1ms"] for g in gpus), "lost_fraction": max(g["lost_fraction"] for g in gpus),
           "mean_over_p50": statistics.median(g["mean_us"] / g["p50_us"] for g in gpus),
           "period_s": statistics.median([g["period_s"] for g in gpus if g["period_s"]] or [0]) or None,
           "window_ms": statistics.median([g["window_ms"] for g in gpus if g["window_ms"]] or [0]) or None,
           "stalling_gpus": [g["gpu"] for g in gpus if g["over_1ms"] or g["stalls"] >= MILD_OPS]}
    if out["over_1ms"] >= SEVERE_OPS or out["lost_fraction"] >= SEVERE_LOST:
        out["severity"] = "severe"
    elif out["stalls"] >= MILD_OPS or (out["max_us"] >= 500 and out["stalls"] >= 3):
        out["severity"] = "mild"
    else:
        out["severity"] = "none"
    return out


def busy(bus_ids: list[str]) -> list[str]:
    try:
        rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_bus_id",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for row in rows.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) == 3 and parts[0] != str(os.getpid()) and parts[2].lower()[-12:] in [b[-12:] for b in bus_ids]:
            found.append(f"PID {parts[0]} {parts[1]} on {parts[2]}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="0,1,2,3", help="nvidia-smi indices, 2-8 GPUs (default 0,1,2,3)")
    parser.add_argument("--kib", type=int, default=80, help="message per operation (default 80 KiB)")
    parser.add_argument("--seconds", type=float, default=6.0, help="duration of each read pattern (default 6 s)")
    parser.add_argument("--ctas", type=int, default=16)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--allow-busy", action="store_true", help="run even if the GPUs are in use")
    parser.add_argument("--json", help="save all numbers, including the timeline, to this file")
    args = parser.parse_args()
    indices = [int(v) for v in args.gpus.split(",") if v.strip()]
    if not 2 <= len(indices) <= MAX_SOURCES + 1 or len(set(indices)) != len(indices):
        parser.error("--gpus needs 2-8 different GPU indices")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, indices))
    nbytes = args.kib << 10
    try:
        cu = Driver()
        cu("cuInit", 0)
        count = ctypes.c_int()
        cu("cuDeviceGetCount", ctypes.byref(count))
        if count.value != len(indices):
            raise CudaError(f"asked for GPUs {args.gpus} but the driver exposes {count.value}")
        gpus = [Gpu(cu, i, nbytes, len(indices)) for i in range(len(indices))]
        print(f"GPUs {args.gpus}: " + ", ".join(f"{g.name.replace('NVIDIA ', '')} {g.bus}" for g in gpus))
        users = [] if args.allow_busy else busy([g.bus for g in gpus])
        if users:
            print("The GPUs are in use (other work distorts the test); stop it or pass --allow-busy:")
            print("\n".join("  " + u for u in users))
            return 2
        for a in gpus:
            a.use()
            for b in gpus:
                if a is b:
                    continue
                can = ctypes.c_int()
                cu("cuDeviceCanAccessPeer", ctypes.byref(can), a.device, b.device)
                if not can.value:
                    raise CudaError(f"GPU {indices[a.ordinal]} cannot access GPU {indices[b.ordinal]} (no P2P)")
                cu("cuCtxEnablePeerAccess", b.context, 0, ok=(0, 704))  # 704: already enabled
        offsets = {g.ordinal: g.clock_offset() for g in gpus}
        patterns = {}
        for name, share in (("read_ring", 1.0), ("read_all", 1.0), ("write_ring", 0.5)):
            if name == "read_all" and len(gpus) == 2:
                continue
            seconds = max(2.0, args.seconds * share)
            print(f"  {name}: {seconds:.0f} s ...", flush=True)
            stamps = run_pattern(gpus, name, nbytes, seconds, args.ctas, args.threads)
            origin = min(s[0] - offsets[o] for o, s in stamps.items() if len(s))
            patterns[name] = {"seconds": seconds, "gpus": [gpu_stats(stamps[g.ordinal], offsets[g.ordinal], origin)
                                                           for g in gpus]}
    except CudaError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    per = {name: pattern_summary(p["gpus"], indices) for name, p in patterns.items()}
    reads = [per[n] for n in ("read_ring", "read_all") if per.get(n)]
    worst = max(reads, key=lambda p: (ORDER.index(p["severity"]), p["over_1ms"], p["max_us"]))
    write = per.get("write_ring") or {}
    verdict = worst["severity"]
    print()
    print(f"Peer read stalls: {args.kib} KiB per operation, every GPU at once, {args.seconds:.0f} s per read pattern "
          f"(times in µs)")
    print(f"  {'pattern':<28}{'ops':>9}{'p50':>8}{'p99':>8}{'p99.9':>9}{'max':>10}{'>1 ms':>7}{'lost':>7}")
    for name, label in (("read_ring", "read ring (GPU r <- r+1)"), ("read_all", "read all peers (pull)"),
                        ("write_ring", "write ring (control)")):
        p = per.get(name)
        if p:
            print(f"  {label:<28}{p['ops']:>9}{p['p50_us']:>8.1f}{p['p99_us']:>8.1f}{p['p999_us']:>9.1f}"
                  f"{p['max_us']:>10.1f}{p['over_1ms']:>7}{100 * p['lost_fraction']:>6.1f}%")
    print()
    if verdict == "none":
        clean = per.get("read_all") or per["read_ring"]
        print(f"READ STALLS: NONE — no peer read of {args.kib} KiB took longer than {clean['max_us']:.0f} µs "
              f"(median {clean['p50_us']:.1f} µs). Pull all-reduce transports run at their median speed here.")
    else:
        ring = per.get("read_ring") or {}
        pull = per.get("read_all") or ring
        stalling = sorted({g for p in reads for g in p["stalling_gpus"]})
        print(f"READ STALLS: YES ({verdict}) — peer reads of {args.kib} KiB stall up to {worst['max_us'] / 1000:.2f} ms; "
              f"{sum(p['over_1ms'] for p in reads)} reads over 1 ms, up to "
              f"{100 * max(p['lost_fraction'] for p in reads):.1f}% of the time lost")
        detail = []
        if ring.get("period_s"):
            detail.append(f"periodic: every {ring['period_s']:.2f} s"
                          + (f", windows of ~{ring['window_ms']:.0f} ms" if ring.get("window_ms") else ""))
        if stalling:
            detail.append("seen by GPU " + ", ".join(map(str, stalling)))
        if detail:
            print("  " + "; ".join(detail))
        if write.get("severity") == "none":
            print("  writes stay fast (control): the stalls are specific to peer reads")
        else:
            print("  writes stall too: something disturbs the GPUs or the fabric in general (other load?)")
        print(f"  -> pull transports (one-shot/two-shot pull, NCCL with P2P reads) cost {pull['mean_over_p50']:.2f}x "
              "their median on average here; use write-only transports (push, phased, ring)")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"gpus": indices, "bytes": nbytes, "bucket_ms": BUCKET_MS, "verdict": verdict,
                       "write_verdict": write.get("severity"), "summary": per, "patterns": patterns}, handle)
        print(f"saved {args.json}")
    return 0 if verdict == "none" else 1


if __name__ == "__main__":
    sys.exit(main())
