"""Verdicts of a collectives run: fabric class, winners and crossovers, tuner loss, recommendation."""

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
            "topology_class": (document.get("selection") or {}).get("topology_class"),
            "iommu": ((document.get("environment") or {}).get("iommu") or {}).get("mode"),
            "p2p_override": ((document.get("environment") or {}).get("driver") or {}).get("p2p_override_present")}


def crossover_text(segments: list[dict]) -> str:
    return " | ".join(f"{s['from_rows']}{'' if s['from_rows'] == s['to_rows'] else '-' + str(s['to_rows'])} rows: "
                      f"{describe(s['transport'])}" for s in segments)


__all__ = ["summarize", "fabric_summary", "transport_summary", "best_by_implementation", "crossover_text", "short"]
