"""lil-bench collectives: GPU selection, peer paths, driver settings, verdicts and the report."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lil_bench.collectives import command, render, summary, system  # noqa: E402

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "collectives_run.json").read_text())


def hop(bdf, host=None):
    item = {"bdf": bdf, "role": "bridge"}
    if host:
        item["host_bridge"] = host
    return item


def hardware():
    """Four GPUs: 0/1 behind one switch under root port 00:01.1, 2 on another root port, 3 on another host bridge."""
    chains = {
        0: [hop("0000:00:01.1", "pci0000:00"), hop("0000:01:00.0"), hop("0000:02:00.0"), {"bdf": "0000:03:00.0"}],
        1: [hop("0000:00:01.1", "pci0000:00"), hop("0000:01:00.0"), hop("0000:02:01.0"), {"bdf": "0000:04:00.0"}],
        2: [hop("0000:00:03.1", "pci0000:00"), {"bdf": "0000:05:00.0"}],
        3: [hop("0000:40:01.1", "pci0000:40"), {"bdf": "0000:41:00.0"}],
    }
    gpus = [{"index": i, "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "bdf": chain[-1]["bdf"],
             "numa_node": None} for i, chain in chains.items()]
    paths = {chain[-1]["bdf"]: {"gpu": i, "chain": chain} for i, chain in chains.items()}
    return {"gpus": gpus, "pcie": {"gpu_paths": paths},
            "host": {"nvidia_params": 'EnableResizableBar: 1\nRegistryDwords: "ForceP2P=0x11;RMForceP2PType=1;Other=3"\n',
                     "kernel_cmdline_relevant": ["amd_iommu=on", "iommu=pt"]},
            "nvml_system": {"driver_version": "615.71.10"}}


def test_select_gpus_defaults_to_the_first_tp_gpus():
    hw = hardware()
    assert system.select_gpus(hw, 4, "") == [0, 1, 2, 3]
    assert system.select_gpus(hw, 2, None) == [0, 1]
    assert system.select_gpus(hw, 2, "3,1") == [3, 1]
    with pytest.raises(ValueError):
        system.select_gpus(hw, 2, "1,7")
    with pytest.raises(ValueError):
        system.select_gpus(hw, 8, "")


def test_pair_paths_and_topology_class(tmp_path):
    hw = hardware()
    assert system.pair_path(hw, 0, 1, str(tmp_path)) == "SW"
    assert system.pair_path(hw, 0, 2, str(tmp_path)) == "RC"
    assert system.pair_path(hw, 0, 3, str(tmp_path)) == "HB"
    assert system.topology_class(system.pair_paths(hw, [0, 1], str(tmp_path))) == "switch"
    assert system.topology_class(system.pair_paths(hw, [0, 1, 2], str(tmp_path))) == "mixed-root-ports"
    assert system.topology_class(system.pair_paths(hw, [2, 3], str(tmp_path))) == "host-bridges"


def test_driver_params_parse_the_p2p_override():
    params = system.driver_params(hardware())
    assert params["p2p_override"] == {"ForceP2P": "0x11", "RMForceP2PType": "1"}
    assert params["p2p_override_present"] and params["resizable_bar"] == "1"
    assert params["driver"] == "615.71.10"


def test_iommu_mode_without_units_is_off(tmp_path):
    info = system.iommu(hardware(), [0, 1], str(tmp_path))
    assert info["mode"] == "off" and info["cmdline"] == ["amd_iommu=on", "iommu=pt"]


def test_fabric_summary_detects_write_collapse():
    fabric = summary.fabric_summary(FIXTURE["results"]["fabric"])
    assert fabric["class"] == "pull-friendly"
    assert fabric["collapse"] == {"read": False, "write": True}
    assert fabric["pattern_gbps"]["ring"]["write"] > 40


def test_fabric_summary_classes():
    def fabric(pair, every, ring):
        pairs = [{"read_gbps_64m": pair[0], "write_gbps_64m": pair[1]}]
        patterns = {"all_to_all": {"read": [every[0]] * 4, "write": [every[1]] * 4},
                    "ring": {"read": [ring[0]] * 4, "write": [ring[1]] * 4}}
        return summary.fabric_summary({"pairs": pairs, "patterns": patterns})["class"]
    assert fabric((50, 50), (31, 48), (42, 50)) == "push-friendly"
    assert fabric((50, 50), (31, 2), (40, 50)) == "pull-friendly"
    assert fabric((51, 42), (1, 2), (0.6, 41)) == "ring-only"


def test_transport_summary_of_the_fixture():
    result = summary.transport_summary(FIXTURE["results"]["transports"])
    assert result["recommendation"]["large"]["transport"] == "twoshot_pull"
    assert [s["transport"] for s in result["crossovers"]][-1] == "twoshot_pull"
    assert result["tuner"]["default"]["max_loss"] > 0.2
    assert not result["wrong_results"]


def test_transport_summary_flags_wrong_and_unstable_cells():
    transports = copy.deepcopy(FIXTURE["results"]["transports"])
    arm = next(a for a in transports["sizes"][0]["arms"] if a["kind"] == "best")
    arm["correct"] = False
    arm["pass2_us"] = arm["steady_us"] * 2
    result = summary.transport_summary(transports)
    assert {"rows": 1, "arm": arm["label"]} in result["wrong_results"]
    assert any(item["arm"] == arm["label"] for item in result["unstable_cells"])


def test_report_renders_and_reads_saved_results(tmp_path, capsys):
    lines = render.lines(FIXTURE)
    text = "\n".join(lines)
    assert "pull-friendly" in text and "fastest b12x transport" in text and "pull2" in text
    path = tmp_path / "run.json"
    path.write_text(json.dumps({k: v for k, v in FIXTURE.items() if k != "summary"}))
    assert command.main(["--report", str(path)]) == 0
    assert "All-reduce, TP4" in capsys.readouterr().out


def test_parse_args_defaults_and_quick():
    args = command.parse_args([])
    assert args.tp == 4 and args.rows == "1,2,4,8,16,32,64,128,192" and args.efforts == ["default", "audit"]
    quick = command.parse_args(["--quick", "--gpus", "4,5"])
    assert quick.tp == 2 and quick.efforts == ["default"] and quick.rows.count(",") == 5


def stall_gpu(p50, maximum, over_1ms=0, stalls=0, lost=0.0, period=None, window=None):
    return {"ops": 100000, "p50_us": p50, "p99_us": p50 * 1.1, "p999_us": maximum / 2, "max_us": maximum,
            "mean_us": p50 * (1 + lost * 2), "stalls": stalls, "over_100us": stalls, "over_1ms": over_1ms,
            "lost_fraction": lost, "period_s": period, "window_ms": window, "stalled_fraction": 0.1 if stalls else 0.0}


def stall_result(read, write, gpus=4):
    return {"bytes": 81920, "patterns": {
        "read_ring": {"seconds": 6.0, "gpus": [read] * gpus},
        "read_all": {"seconds": 6.0, "gpus": [read] * gpus},
        "write_ring": {"seconds": 3.0, "gpus": [write] * gpus}}}


def test_stall_summary_reports_periodic_read_stalls():
    read = stall_gpu(5.6, 1640.0, over_1ms=5, stalls=1500, lost=0.06, period=2.3, window=150.0)
    result = summary.stall_summary(stall_result(read, stall_gpu(3.3, 22.0)), [0, 4, 8, 12])
    assert result["verdict"] == "severe" and result["read_specific"]
    assert result["stalling_gpus"] == [0, 4, 8, 12]
    assert result["period_s"] == 2.3 and result["over_1ms"] == 40
    text = summary.stall_text(result)
    assert text[0].startswith("READ STALLS: YES (severe)") and "every 2.30 s" in text[1]


def test_stall_summary_clean_and_mild():
    clean = summary.stall_summary(stall_result(stall_gpu(7.0, 12.0), stall_gpu(3.3, 4.0)), [0, 1, 2, 3])
    assert clean["verdict"] == "none" and not clean["read_specific"] and clean["stalling_gpus"] == []
    assert summary.stall_text(clean)[0].startswith("READ STALLS: NONE")
    mild = summary.stall_summary(stall_result(stall_gpu(7.0, 300.0, stalls=12), stall_gpu(3.3, 4.0)))
    assert mild["verdict"] == "mild"
    disturbed = summary.stall_summary(stall_result(stall_gpu(7.0, 1500.0, over_1ms=4, stalls=40),
                                                   stall_gpu(3.3, 1500.0, over_1ms=4, stalls=40)))
    assert disturbed["verdict"] == "severe" and not disturbed["read_specific"]
    assert any("writes stall too" in line for line in summary.stall_text(disturbed))


def test_stalls_only_skips_the_other_phases():
    args = command.parse_args(["--stalls-only"])
    assert args.skip_fabric and args.skip_p2pmark and args.skip_transports and not args.skip_stalls
    with pytest.raises(SystemExit):
        command.parse_args(["--stalls-only", "--skip-stalls"])
