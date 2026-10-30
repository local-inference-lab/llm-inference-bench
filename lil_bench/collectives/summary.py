"""Verdicts of a collectives run: read stalls, fabric class, winners and crossovers, tuner loss, recommendation."""

from __future__ import annotations

import statistics

from . import describe, family, short

# A concurrent pattern "collapses" when it delivers less than this share of a lone pair's rate.
COLLAPSE = 0.25
UNSTABLE = 0.15  # two passes disagreeing by more than this flag a cell
SIZE_CLASSES = (("small", 1, 8), ("medium", 9, 64), ("large", 65, 1 << 30))


def _median(values):
    values = [v for v in values or [] if isinstance(v, (int, float)) and v > 0]
    return statistics.median(values) if values else None


def fabric_summary(fabric: dict | None) -> dict:
    """Pair, all-to-all, ring, fan-out and incast rates; which directions collapse under concurrency."""
    if not fabric or not fabric.get("pairs"):
        return {}
    pair = {mode: _median(p.get(f"{mode}_gbps_64m") for p in fabric["pairs"]) for mode in ("read", "write")}
    patterns = fabric.get("patterns") or {}
    rates = {name: {mode: _median((patterns.get(name) or {}).get(mode)) for mode in ("read", "write")}
             for name in ("all_to_all", "ring", "fan_out", "incast")}
    collapse = {}
    for mode in ("read", "write"):
        base = pair[mode]
        every = rates["all_to_all"][mode]
        collapse[mode] = bool(base and every is not None and every < COLLAPSE * base)
    ring_ok = {mode: bool(pair[mode] and rates["ring"][mode] and rates["ring"][mode] >= COLLAPSE * pair[mode])
               for mode in ("read", "write")}
    if collapse["read"] and collapse["write"]:
        verdict = "ring-only" if ring_ok["write"] or ring_ok["read"] else "collapsing"
    elif collapse["write"]:
        verdict = "pull-friendly"
    elif collapse["read"]:
        verdict = "push-friendly"
    else:
        every = rates["all_to_all"]
        ratio = every["write"] / every["read"] if every["write"] and every["read"] else None
        verdict = ("push-friendly" if ratio and ratio >= 1.15 else
                   "pull-friendly" if ratio and ratio <= 1 / 1.15 else "neutral")
    def ratio(entry):
        return entry["write"] / entry["read"] if entry.get("write") and entry.get("read") else None
    return {"pair_gbps": pair, "pattern_gbps": rates, "collapse": collapse, "ring_ok": ring_ok,
            "write_over_read": {"pair": ratio(pair), "all_to_all": ratio(rates["all_to_all"]),
                                "ring": ratio(rates["ring"])},
            "class": verdict}


# Peer read stalls: a pattern is "severe" with this many operations over 1 ms (all GPUs) or this share of
# its time lost to stalls, "mild" with this many stalled operations (> max(100 us, 10x median)).
STALL_SEVERE_OPS = 3
STALL_SEVERE_LOST = 0.05
STALL_MILD_OPS = 10
STALL_ORDER = ("none", "mild", "severe")


def _stall_pattern(pattern: dict | None) -> dict:
    gpus = [g for g in (pattern or {}).get("gpus") or [] if g.get("ops")]
    if not gpus:
        return {}
    periods = [g["period_s"] for g in gpus if g.get("period_s")]
    windows = [g["window_ms"] for g in gpus if g.get("window_ms")]
    means = [g["mean_us"] / g["p50_us"] for g in gpus if g.get("p50_us")]
    result = {
        "ops": sum(g["ops"] for g in gpus),
        "p50_us": statistics.median(g["p50_us"] for g in gpus),
        "p99_us": max(g["p99_us"] for g in gpus),
        "p999_us": max(g["p999_us"] for g in gpus),
        "max_us": max(g["max_us"] for g in gpus),
        "mean_over_p50": statistics.median(means) if means else None,
        "stalls": sum(g["stalls"] for g in gpus),
        "over_1ms": sum(g["over_1ms"] for g in gpus),
        "lost_fraction": max(g["lost_fraction"] for g in gpus),
        "period_s": statistics.median(periods) if periods else None,
        "window_ms": statistics.median(windows) if windows else None,
        "stalled_fraction": max(g.get("stalled_fraction") or 0 for g in gpus),
        "stalling_gpus": [i for i, g in enumerate((pattern or {}).get("gpus") or [])
                          if g.get("ops") and (g["over_1ms"] or g["stalls"] >= STALL_MILD_OPS)],
    }
    if result["over_1ms"] >= STALL_SEVERE_OPS or result["lost_fraction"] >= STALL_SEVERE_LOST:
        result["severity"] = "severe"
    elif result["stalls"] >= STALL_MILD_OPS or (result["max_us"] >= 500 and result["stalls"] >= 3):
        result["severity"] = "mild"
    else:
        result["severity"] = "none"
    return result


def stall_summary(stalls: dict | None, gpus: list[int] | None = None) -> dict:
    """Do peer reads stall? Verdict over the read patterns, with the write ring as the control."""
    patterns = (stalls or {}).get("patterns") or {}
    if not patterns:
        return {}
    per = {name: _stall_pattern(pattern) for name, pattern in patterns.items()}
    per = {name: value for name, value in per.items() if value}
    reads = [per[name] for name in ("read_ring", "read_all") if name in per]
    if not reads:
        return {}
    worst = max(reads, key=lambda p: (STALL_ORDER.index(p["severity"]), p["over_1ms"], p["max_us"]))
    write = per.get("write_ring") or {}
    gpus = gpus or list(range(len((patterns.get("read_ring") or {}).get("gpus") or [])))
    ring = per.get("read_ring") or {}
    stalling = sorted({gpus[r] for p in reads for r in p.get("stalling_gpus") or [] if r < len(gpus)})
    pull = per.get("read_all") or ring
    return {
        "verdict": worst["severity"],
        "write_verdict": write.get("severity"),
        "read_specific": worst["severity"] != "none" and write.get("severity") == "none",
        "bytes": (stalls or {}).get("bytes"),
        "seconds": max((p.get("seconds") or 0) for p in patterns.values()),
        "max_us": worst["max_us"],
        "over_1ms": sum(p["over_1ms"] for p in reads),
        "lost_fraction": max(p["lost_fraction"] for p in reads),
        "period_s": ring.get("period_s"),
        "window_ms": ring.get("window_ms"),
        "stalled_fraction": ring.get("stalled_fraction"),
        "stalling_gpus": stalling,
        "pull_penalty": pull.get("mean_over_p50"),
        "patterns": per,
    }


def stall_text(stalls: dict) -> list[str]:
    """Plain-language verdict lines shared by the terminal report."""
    if not stalls:
        return []
    verdict, kib = stalls["verdict"], (stalls.get("bytes") or 0) / 1024
    if verdict == "none":
        clean = stalls["patterns"].get("read_all") or stalls["patterns"].get("read_ring") or {}
        return [f"READ STALLS: NONE — no peer read of {kib:.0f} KiB took longer than {clean.get('max_us', 0):.0f} µs "
                f"(median {clean.get('p50_us', 0):.1f} µs) in {stalls['seconds']:.0f} s per pattern"]
    lines = [f"READ STALLS: YES ({verdict}) — peer reads of {kib:.0f} KiB stall up to {stalls['max_us'] / 1000:.2f} ms; "
             f"{stalls['over_1ms']} reads over 1 ms, up to {100 * stalls['lost_fraction']:.1f}% of the time lost"]
    detail = []
    if stalls.get("period_s"):
        detail.append(f"periodic: every {stalls['period_s']:.2f} s"
                      + (f", windows of ~{stalls['window_ms']:.0f} ms" if stalls.get("window_ms") else ""))
    if stalls.get("stalling_gpus"):
        detail.append("seen by GPU " + ", ".join(map(str, stalls["stalling_gpus"])))
    if detail:
        lines.append("  " + "; ".join(detail))
    if stalls.get("read_specific"):
        lines.append("  writes stay fast (control): the stalls are specific to peer reads")
    elif stalls.get("write_verdict") and stalls["write_verdict"] != "none":
        lines.append("  writes stall too: something disturbs the GPUs or the fabric in general (other load?)")
    if stalls.get("pull_penalty"):
        lines.append(f"  -> pull transports (one-shot/two-shot pull, NCCL with P2P reads) cost {stalls['pull_penalty']:.2f}x "
                     "their median on average here; use write-only transports (push, phased, ring)")
    return lines


def best_by_implementation(size: dict) -> dict[str, dict]:
    """Fastest correct arm per transport (tuner arms excluded), keyed by implementation name."""
    best: dict[str, dict] = {}
    for arm in size.get("arms") or []:
        if arm.get("kind") == "tuned" or arm.get("correct") is False or not arm.get("steady_us"):
            continue
        name = arm["implementation"]
        if name not in best or arm["steady_us"] < best[name]["steady_us"]:
            best[name] = arm
    return best


def _ranges(items: list[tuple[int, str]]) -> list[dict]:
    segments: list[dict] = []
    for rows, name in items:
        if segments and segments[-1]["transport"] == name:
            segments[-1]["to_rows"] = rows
        else:
            segments.append({"from_rows": rows, "to_rows": rows, "transport": name})
    return segments


def transport_summary(transports: dict | None) -> dict:
    if not transports or not transports.get("sizes"):
        return {}
    winners, overall, push_pull, nccl_vs_best, unstable, wrong = [], [], [], [], [], []
    race_mismatch = []
    tuner: dict[str, dict] = {}
    for size in transports["sizes"]:
        rows = size["rows"]
        best = best_by_implementation(size)
        b12x = {name: arm for name, arm in best.items() if family(name) != "nccl"}
        ncc = {name: arm for name, arm in best.items() if family(name) == "nccl"}
        for arm in size.get("arms") or []:
            if arm.get("correct") is False:
                wrong.append({"rows": rows, "arm": arm["label"]})
            first, second = arm.get("steady_us"), arm.get("pass2_us")
            if first and second and abs(first - second) / min(first, second) > UNSTABLE:
                unstable.append({"rows": rows, "arm": arm["label"], "pass1_us": first, "pass2_us": second})
        if not b12x and not ncc:
            continue
        fastest = min(b12x.values(), key=lambda arm: arm["steady_us"]) if b12x else None
        everything = min(best.values(), key=lambda arm: arm["steady_us"])
        if fastest:
            winners.append((rows, fastest["implementation"]))
        overall.append((rows, everything["implementation"]))
        push = [arm["steady_us"] for name, arm in b12x.items() if family(name) == "push"]
        pull = [arm["steady_us"] for name, arm in b12x.items() if family(name) == "pull"]
        if push and pull:
            push_pull.append({"rows": rows, "pull_over_push": min(pull) / min(push)})
        if fastest and ncc:
            nccl = min(ncc.values(), key=lambda arm: arm["steady_us"])
            nccl_vs_best.append({"rows": rows, "nccl": nccl["implementation"],
                                 "nccl_over_b12x": nccl["steady_us"] / fastest["steady_us"]})
        if len(b12x) > 1 and all(arm.get("race_us") for arm in b12x.values()):
            race_best = min(b12x.values(), key=lambda arm: arm["race_us"])
            if race_best is not fastest:
                race_mismatch.append({"rows": rows, "steady_winner": fastest["implementation"],
                                      "race_winner": race_best["implementation"],
                                      "race_winner_steady_loss": race_best["steady_us"] / fastest["steady_us"] - 1})
        for arm in size.get("arms") or []:
            if arm.get("kind") != "tuned" or not fastest or not arm.get("steady_us"):
                continue
            entry = tuner.setdefault(arm["effort"], {"by_rows": []})
            entry["by_rows"].append({"rows": rows, "pick": arm["implementation"], "threads": arm.get("threads"),
                                     "blocks": arm.get("blocks"), "us": arm["steady_us"],
                                     "loss": arm["steady_us"] / fastest["steady_us"] - 1})
    for entry in tuner.values():
        losses = [item["loss"] for item in entry["by_rows"]]
        entry["max_loss"] = max(losses) if losses else None
        entry["sizes_over_5pct"] = sum(1 for loss in losses if loss > 0.05)
    recommendation = {}
    for name, low, high in SIZE_CLASSES:
        picks = [impl for rows, impl in overall if low <= rows <= high]
        if picks:
            top = max(set(picks), key=picks.count)
            recommendation[name] = {"transport": top, "family": family(top), "sizes": len(picks)}
    nccl_wins = [item for item in nccl_vs_best if item["nccl_over_b12x"] < 0.9]
    families = [family(impl) for _, impl in winners]
    collective = ("push" if families and families.count("push") >= 0.75 * len(families) else
                  "pull" if families and families.count("pull") >= 0.75 * len(families) else "mixed")
    if len(nccl_wins) >= max(1, len(nccl_vs_best) // 2):
        collective = "ring"
    return {
        "winners": [{"rows": rows, "transport": impl} for rows, impl in winners],
        "overall_winners": [{"rows": rows, "transport": impl} for rows, impl in overall],
        "crossovers": _ranges(winners),
        "overall_crossovers": _ranges(overall),
        "push_pull": push_pull,
        "nccl_vs_best": nccl_vs_best,
        "nccl_wins_rows": [item["rows"] for item in nccl_wins],
        "tuner": tuner,
        "race_mismatch": race_mismatch,
        "unstable_cells": unstable,
        "wrong_results": wrong,
        "recommendation": recommendation,
        "collective_class": collective,
    }


def summarize(document: dict) -> dict:
    results = document.get("results") or {}
    return {"fabric": fabric_summary(results.get("fabric")),
            "transports": transport_summary(results.get("transports")),
            "stalls": stall_summary(results.get("stalls"), (document.get("selection") or {}).get("gpus")),
            "topology_class": (document.get("selection") or {}).get("topology_class"),
            "iommu": ((document.get("environment") or {}).get("iommu") or {}).get("mode"),
            "p2p_override": ((document.get("environment") or {}).get("driver") or {}).get("p2p_override_present")}


def crossover_text(segments: list[dict]) -> str:
    return " | ".join(f"{s['from_rows']}{'' if s['from_rows'] == s['to_rows'] else '-' + str(s['to_rows'])} rows: "
                      f"{describe(s['transport'])}" for s in segments)


__all__ = ["summarize", "fabric_summary", "transport_summary", "stall_summary", "stall_text", "best_by_implementation",
           "crossover_text", "short"]
