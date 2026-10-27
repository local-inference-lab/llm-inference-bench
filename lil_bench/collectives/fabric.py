"""Raw peer bandwidth between the selected GPUs: who can read and write whom how fast, alone and under load.

Run as ``python -m lil_bench.collectives.fabric --result FILE`` with CUDA_VISIBLE_DEVICES set to the
ranks under test. One process drives every GPU: an NVRTC copy kernel moves 16-byte vectors from a
source to a destination buffer, where either buffer may live on a peer GPU (peer access enabled).

* ``pairs``: one GPU copies from (read, pull direction) or to (write, push direction) one peer
  while every other GPU is idle; 64 MiB for bandwidth and 256 KiB for small transfers.
* ``patterns``, every GPU's rate in GB/s:
  - ``all_to_all``: every GPU reads from / writes to every peer at once (the pattern of one-shot
    and two-shot all-reduce transports);
  - ``ring``: every GPU reads from its predecessor / writes to its successor at once (NCCL's ring);
  - ``fan_out``: one GPU at a time reads from / writes to all peers (fan-out without contention);
  - ``incast``: all other GPUs read from / write into one GPU at once (aggregate rate per target).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path

KERNEL = r"""
struct alignas(16) V { unsigned int x, y, z, w; };
extern "C" __global__ void multi_copy(
    const V* s0, const V* s1, const V* s2, const V* s3, const V* s4, const V* s5, const V* s6, const V* s7,
    V* d0, V* d1, V* d2, V* d3, V* d4, V* d5, V* d6, V* d7, unsigned long long n) {
  const V* s; V* d;
  switch (blockIdx.y) {
    case 0: s = s0; d = d0; break; case 1: s = s1; d = d1; break;
    case 2: s = s2; d = d2; break; case 3: s = s3; d = d3; break;
    case 4: s = s4; d = d4; break; case 5: s = s5; d = d5; break;
    case 6: s = s6; d = d6; break; default: s = s7; d = d7; break;
  }
  unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    d[i] = s[i];
}
"""
MAX_PEERS = 8


def _check(result, what):
    if int(result[0]) != 0:
        raise RuntimeError(f"{what} failed: {result[0]}")
    return result[1] if len(result) == 2 else None


class Fabric:
    def __init__(self, mib: int):
        import torch
        from cuda.bindings import driver as cu
        from cuda.bindings import nvrtc
        from cuda.bindings import runtime as rt

        self.torch, self.cu = torch, cu
        self.count = torch.cuda.device_count()
        if self.count < 2:
            raise RuntimeError("the fabric test needs at least two GPUs")
        if self.count > MAX_PEERS:
            raise RuntimeError(f"the fabric test supports at most {MAX_PEERS} GPUs")
        for device in range(self.count):
            torch.cuda.set_device(device)
            torch.empty(1, device=device)
        self.peer_access = {}
        for a in range(self.count):
            torch.cuda.set_device(a)
            for b in range(self.count):
                if a == b:
                    continue
                ok = bool(torch.cuda.can_device_access_peer(a, b))
                if ok:
                    err = rt.cudaDeviceEnablePeerAccess(b, 0)[0]
                    ok = int(err) in (0, int(rt.cudaError_t.cudaErrorPeerAccessAlreadyEnabled))
                self.peer_access[(a, b)] = ok

        major, minor = torch.cuda.get_device_capability(0)
        program = _check(nvrtc.nvrtcCreateProgram(KERNEL.encode(), b"multi_copy.cu", 0, [], []), "nvrtcCreateProgram")
        options = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17"]
        if int(nvrtc.nvrtcCompileProgram(program, len(options), options)[0]) != 0:
            size = _check(nvrtc.nvrtcGetProgramLogSize(program), "nvrtcGetProgramLogSize")
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(f"copy kernel compilation failed: {log.decode(errors='replace')}")
        size = _check(nvrtc.nvrtcGetCUBINSize(program), "nvrtcGetCUBINSize")
        cubin = b" " * size
        _check((nvrtc.nvrtcGetCUBIN(program, cubin)[0], None), "nvrtcGetCUBIN")
        self.functions, self.sms = {}, {}
        for device in range(self.count):
            torch.cuda.set_device(device)
            torch.cuda.synchronize()
            module = _check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
            self.functions[device] = _check(cu.cuModuleGetFunction(module, b"multi_copy"), "cuModuleGetFunction")
            self.sms[device] = torch.cuda.get_device_properties(device).multi_processor_count
        self.big = mib << 20
        # Every GPU owns one source buffer and one landing slot per peer, so concurrent copies never overlap.
        self.src = {d: torch.full((self.big // 2,), float(d + 1), dtype=torch.bfloat16, device=d)
                    for d in range(self.count)}
        self.land = {d: torch.empty((self.count, self.big // 2), dtype=torch.bfloat16, device=d)
                     for d in range(self.count)}
        self.types = (ctypes.c_void_p,) * (2 * MAX_PEERS) + (ctypes.c_ulonglong,)

    def read_pair(self, executor: int, peer: int):
        """``executor`` pulls ``peer``'s source into its own landing slot."""
        return self.src[peer].data_ptr(), self.land[executor][peer].data_ptr()

    def write_pair(self, executor: int, peer: int):
        """``executor`` pushes its source into its slot on ``peer``."""
        return self.src[executor].data_ptr(), self.land[peer][executor].data_ptr()

    def launch(self, device: int, pairs: list, nbytes: int) -> None:
        sources = [p[0] for p in pairs] + [pairs[0][0]] * (MAX_PEERS - len(pairs))
        targets = [p[1] for p in pairs] + [pairs[0][1]] * (MAX_PEERS - len(pairs))
        stream = self.torch.cuda.current_stream(device).cuda_stream
        blocks = max(1, (2 * self.sms[device]) // len(pairs))
        result = self.cu.cuLaunchKernel(self.functions[device], blocks, len(pairs), 1, 512, 1, 1, 0,
                                        self.cu.CUstream(stream),
                                        (tuple(sources + targets) + (nbytes // 16,), self.types), 0)
        _check((result[0], None), "cuLaunchKernel")

    def concurrent(self, assignments: dict[int, list], nbytes: int, iterations: int) -> dict[int, float]:
        """Start every device's copies together; GB/s per executing device (bytes it moved / its time)."""
        torch = self.torch
        events = {}
        for round_ in range(2):  # warm-up round, then the timed round
            for device, pairs in assignments.items():
                with torch.cuda.device(device):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(iterations if round_ else 1):
                        self.launch(device, pairs, nbytes)
                    end.record()
                    events[device] = (start, end)
            for device in assignments:
                torch.cuda.synchronize(device)
        return {device: len(assignments[device]) * nbytes * iterations / (start.elapsed_time(end) * 1e-3) / 1e9
                for device, (start, end) in events.items()}

    def measure(self) -> dict:
        n, big = self.count, self.big
        peers = lambda d: [p for p in range(n) if p != d]  # noqa: E731
        pairs = []
        for a in range(n):
            for b in peers(a):
                if not self.peer_access[(a, b)]:
                    continue
                entry = {"executor": a, "peer": b}
                for label, nbytes, iterations in (("64m", big, 10), ("256k", 256 << 10, 200)):
                    for mode, pair in (("read", self.read_pair), ("write", self.write_pair)):
                        rate = self.concurrent({a: [pair(a, b)]}, nbytes, iterations)[a]
                        entry[f"{mode}_gbps_{label}"] = rate
                pairs.append(entry)
        patterns = {}
        if all(self.peer_access.values()):
            for mode, pair in (("read", self.read_pair), ("write", self.write_pair)):
                every = self.concurrent({d: [pair(d, p) for p in peers(d)] for d in range(n)}, big, 5)
                ring_peer = (lambda d: (d - 1) % n) if mode == "read" else (lambda d: (d + 1) % n)
                ring = self.concurrent({d: [pair(d, ring_peer(d))] for d in range(n)}, big, 5)
                fan_out = {d: self.concurrent({d: [pair(d, p) for p in peers(d)]}, big, 5)[d] for d in range(n)}
                incast = {}
                for target in range(n):
                    rates = self.concurrent({d: [pair(d, target)] for d in peers(target)}, big, 5)
                    incast[target] = sum(rates.values())
                for name, values in (("all_to_all", every), ("ring", ring), ("fan_out", fan_out), ("incast", incast)):
                    patterns.setdefault(name, {})[mode] = [values[d] for d in range(n)]
        return {"peer_access": {f"{a}-{b}": ok for (a, b), ok in self.peer_access.items()},
                "pairs": pairs, "patterns": patterns, "buffer_mib": big >> 20}


def median_pair(fabric: dict, key: str) -> float | None:
    values = [pair[key] for pair in fabric.get("pairs") or [] if pair.get(key)]
    return statistics.median(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", required=True)
    parser.add_argument("--mib", type=int, default=64)
    args = parser.parse_args()
    Path(args.result).write_text(json.dumps(Fabric(args.mib).measure()))


if __name__ == "__main__":
    main()
