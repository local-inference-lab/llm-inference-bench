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
