"""Terminal report of a collectives run (also used by ``lil-bench collectives --report FILE``)."""

from __future__ import annotations

from . import TRANSPORTS, describe, family, short
from .summary import best_by_implementation, crossover_text, stall_text

FABRIC_HELP = {
    "push-friendly": "writes to peers stay fast: push transports should win",
    "pull-friendly": "writes to all peers at once collapse, reads do not: pull transports should win",
    "ring-only": "reads and fanned-out writes collapse; only one-neighbour traffic survives: ring (NCCL) should win",
    "collapsing": "every concurrent pattern collapses",
    "neutral": "reads and writes perform alike",
}


def _fmt(value, width=8, digits=2):
    return f"{value:>{width}.{digits}f}" if isinstance(value, (int, float)) else f"{'n/a':>{width}}"


def lines(document: dict) -> list[str]:
    out: list[str] = []
    selection = document.get("selection") or {}
    environment = document.get("environment") or {}
    summary = document.get("summary") or {}
    hardware = document.get("hardware") or {}
    gpus = {gpu["index"]: gpu for gpu in hardware.get("gpus") or []}
    chosen = selection.get("gpus") or []
    out.append(f"TP{len(chosen)} on GPUs {','.join(map(str, chosen))}: "
               f"{gpus.get(chosen[0], {}).get('name', '?') if chosen else '?'} · "
               f"{(hardware.get('cpu') or {}).get('lscpu', {}).get('Model name', '?')}")
    paths = selection.get("pair_paths") or {}
    if chosen:
        out.append("peer paths: SW shared PCIe switch, RC one CPU root complex, HB between CPU host bridges, "
                   "SYS between sockets")
        out.append("      " + "".join(f"{i:>6}" for i in chosen))
        for a in chosen:
            out.append(f"{a:>6}" + "".join(f"{'-' if a == b else paths.get(f'{a}-{b}', '?'):>6}" for b in chosen))
    iommu, driver = environment.get("iommu") or {}, environment.get("driver") or {}
    override = driver.get("p2p_override") or {}
    out.append(f"topology {selection.get('topology_class')} · IOMMU {iommu.get('mode')} · NVIDIA P2P override "
               + (";".join(f"{k}={v}" for k, v in override.items()) if override else "none")
               + f" · driver {driver.get('driver')}")
    flashinfer = (environment.get("software") or {}).get("flashinfer") or {}
    out.append(f"FlashInfer {flashinfer.get('version') or ''} {(flashinfer.get('commit') or '')[:12]}"
               + (f" + {', '.join(flashinfer['patches'])}" if flashinfer.get("patches") else ""))

    ssum = summary.get("stalls") or {}
    if ssum:
        out.append("")
        out.append(f"Peer read stalls: {(ssum.get('bytes') or 0) / 1024:.0f} KiB per operation, every GPU at once, "
                   f"{ssum['seconds']:.0f} s per read pattern")
        out.append(f"  {'pattern':<28}{'ops':>9}{'p50 µs':>9}{'p99 µs':>9}{'p99.9 µs':>10}{'max µs':>10}"
                   f"{'>1 ms':>7}{'lost':>7}")
        for name, label in (("read_ring", "read ring (GPU r <- r+1)"), ("read_all", "read all peers (pull)"),
                            ("write_ring", "write ring (control)")):
            p = (ssum.get("patterns") or {}).get(name)
            if p:
                out.append(f"  {label:<28}{p['ops']:>9}{p['p50_us']:>9.1f}{p['p99_us']:>9.1f}{p['p999_us']:>10.1f}"
                           f"{p['max_us']:>10.1f}{p['over_1ms']:>7}{100 * p['lost_fraction']:>6.1f}%")
        out.extend("  " + line for line in stall_text(ssum))

    fabric = (document.get("results") or {}).get("fabric") or {}
    fsum = summary.get("fabric") or {}
    if fsum:
        out.append("")
        out.append(f"Peer bandwidth, SM copies of {fabric.get('buffer_mib')} MiB, GB/s per GPU (median over GPUs)")
        pair, rates = fsum["pair_gbps"], fsum["pattern_gbps"]
        out.append(f"  {'pattern':<34}{'read (pull)':>12}{'write (push)':>14}")
        rows = [("one pair, others idle", pair),
                ("all peers at once (all-to-all)", rates["all_to_all"]),
                ("ring, one neighbour each", rates["ring"]),
                ("one GPU to all peers, alone", rates["fan_out"]),
                ("all peers into one GPU (aggregate)", rates["incast"])]
        for label, values in rows:
            out.append(f"  {label:<34}{_fmt(values.get('read'), 12, 1)}{_fmt(values.get('write'), 14, 1)}")
        out.append(f"  fabric: {fsum['class']} — {FABRIC_HELP.get(fsum['class'], '')}")

    transports = (document.get("results") or {}).get("transports") or {}
    tsum = summary.get("transports") or {}
    sizes = transports.get("sizes") or []
    if sizes:
        out.append("")
        present = [name for name in TRANSPORTS if any(
            arm.get("implementation") == name for size in sizes for arm in size.get("arms") or []
            if arm.get("kind") != "tuned")]
        efforts = transports.get("efforts") or []
        out.append(f"All-reduce, TP{transports.get('world_size')} BF16 x {transports.get('hidden')}: µs per call, "
                   f"slowest rank, steady state; each transport at its fastest launch configuration")
        header = f"{'rows':>5}{'KiB':>6}" + "".join(f"{short(name):>8}" for name in present) + "   fastest b12x    "
        header += "".join(f"{'tuner ' + effort:>24}" for effort in efforts)
        out.append(header)
        for size in sizes:
            best = best_by_implementation(size)
            b12x = {n: a for n, a in best.items() if family(n) != "nccl"}
            fastest = min(b12x.values(), key=lambda arm: arm["steady_us"]) if b12x else None
            row = f"{size['rows']:>5}{size['bytes'] / 1024:>6.0f}"
            for name in present:
                arm = best.get(name)
                mark = "*" if arm is not None and arm is fastest else " "
                row += f"{arm['steady_us']:>7.2f}{mark}" if arm else f"{'n/a':>8}"
            row += f"   {describe(fastest['implementation']) if fastest else '-':<15}"
            for effort in efforts:
                arm = next((a for a in size.get("arms") or [] if a.get("label") == f"tuner:{effort}"
                            or a.get("label") == f"tuned:{effort}"), None)
                if arm and fastest:
                    loss = arm["steady_us"] / fastest["steady_us"] - 1
                    row += f"{short(arm['implementation']):>8}/{arm.get('threads')}x{arm.get('blocks'):<4}{loss:+5.0%}"
            out.append(row)
        out.append("  (* fastest b12x transport; NCCL columns use vLLM's NCCL settings, 'ring' forces NCCL_ALGO=Ring)")
        out.append("")
        out.append("Summary")
        if tsum.get("crossovers"):
            out.append(f"  fastest b12x transport: {crossover_text(tsum['crossovers'])}")
        if tsum.get("overall_crossovers") and tsum["overall_crossovers"] != tsum.get("crossovers"):
            out.append(f"  fastest including NCCL: {crossover_text(tsum['overall_crossovers'])}")
        pp = tsum.get("push_pull") or []
        if pp:
            push = [item["rows"] for item in pp if item["pull_over_push"] >= 1]
            pull = [item["rows"] for item in pp if item["pull_over_push"] < 1]
            out.append(f"  push wins at rows {push or '-'}; pull wins at rows {pull or '-'}")
        if tsum.get("nccl_vs_best"):
            ratios = [item["nccl_over_b12x"] for item in tsum["nccl_vs_best"]]
            out.append(f"  NCCL takes {min(ratios):.2f}x-{max(ratios):.2f}x the fastest b12x transport's time"
                       + (f"; NCCL is faster at rows {tsum['nccl_wins_rows']}" if tsum.get("nccl_wins_rows") else ""))
        for effort, entry in (tsum.get("tuner") or {}).items():
            bad = [item for item in entry["by_rows"] if item["loss"] > 0.05]
            out.append(f"  joint tuner ({effort}): " + ("within 5% of the fastest transport everywhere" if not bad else
                       "loses " + ", ".join(f"{item['loss']:+.0%} at {item['rows']} rows ({short(item['pick'])})"
                                            for item in bad)))
        for item in tsum.get("race_mismatch") or []:
            out.append(f"  race-style timing would pick {describe(item['race_winner'])} at {item['rows']} rows "
                       f"({item['race_winner_steady_loss']:+.0%} in steady state)")
        rec = tsum.get("recommendation") or {}
        if rec:
            out.append("  recommended: " + "; ".join(f"{name} messages {describe(item['transport'])}"
                                                     for name, item in rec.items()))
        for item in tsum.get("unstable_cells") or []:
            out.append(f"  unstable: {item['arm']} at {item['rows']} rows ({item['pass1_us']:.1f} vs "
                       f"{item['pass2_us']:.1f} µs in two passes)")
        for item in tsum.get("wrong_results") or []:
            out.append(f"  WRONG RESULT: {item['arm']} at {item['rows']} rows")
    return out
