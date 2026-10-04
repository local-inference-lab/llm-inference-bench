"""Tests for --tool-eval: tool-eval-bench run with the community leaderboard settings.

tool-eval-bench is an external program (installed on demand, pinned commit). These tests
cover the argument mapping, the parsing of its JSON result and the failure modes, with a
fake tool-eval-bench executable and a stub OpenAI-compatible server; nothing here needs a
GPU, a model or network access.

tests/fixtures/tool_eval_bench_run.json is the run.json of a real run of the leaderboard
command (tool-eval-bench 2.7.1.dev14+g570951a77, GLM-5.3-Flash NVFP4 on vLLM). The
conversation traces (raw_log) and scenario descriptions (expected_behavior) are cut out,
and the host name and model path are replaced.
"""
import _thread
import importlib.util
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
SPEC = importlib.util.spec_from_file_location("llm_decode_bench_tool_eval", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCH)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tool_eval_bench_run.json"
LEADERBOARD_TAIL = [
    "--backend", "vllm", "--hardmode", "--seed", "42", "--temperature", "0",
    "--parallel", "4", "--max-turns", "30", "--timeout", "600",
    "--no-live", "--json", "--json-file", "run.json",
]


def parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["llm_decode_bench.py", *argv])
    return BENCH.parse_args()


def parse_error(monkeypatch, capsys, *argv) -> str:
    with pytest.raises(SystemExit) as exc:
        parse(monkeypatch, *argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Argument mapping
# ---------------------------------------------------------------------------

def test_defaults_reproduce_the_leaderboard_command(monkeypatch):
    args = parse(monkeypatch, "--tool-eval", "--port", "8000", "--model", "glm-5.3-flash")
    settings = BENCH.tool_eval_settings(args)
    cmd = BENCH.build_tool_eval_command(
        ["tool-eval-bench"], model=args.model, base_url=BENCH.tool_eval_base_url(args), settings=settings,
    )
    # tool-eval-bench --model glm-5.3-flash --base-url http://localhost:8000/ --backend vllm
    #   --hardmode --seed 42 --temperature 0 --parallel 4 --max-turns 30 --timeout 600
    #   --no-live --json --json-file run.json
    assert cmd == [
        "tool-eval-bench", "--model", "glm-5.3-flash", "--base-url", "http://localhost:8000/",
        *LEADERBOARD_TAIL,
    ]
    assert settings["leaderboard"] is True
    assert settings["differs_from_leaderboard"] == []
    assert "leaderboard settings" in BENCH._tool_eval_settings_text(settings)


def test_each_setting_can_be_overridden(monkeypatch):
    args = parse(
        monkeypatch, "--tool-eval", "--port", "8000", "--model", "m",
        "--tool-eval-seed", "7", "--tool-eval-temperature", "0.6", "--tool-eval-parallel", "8",
        "--tool-eval-max-turns", "12", "--tool-eval-timeout", "120.5", "--no-tool-eval-hardmode",
    )
    settings = BENCH.tool_eval_settings(args)
    cmd = BENCH.build_tool_eval_command(["teb"], model="m", base_url="http://localhost:8000/", settings=settings)
    assert cmd == [
        "teb", "--model", "m", "--base-url", "http://localhost:8000/", "--backend", "vllm",
        "--seed", "7", "--temperature", "0.6", "--parallel", "8", "--max-turns", "12",
        "--timeout", "120.5", "--no-live", "--json", "--json-file", "run.json",
    ]
    assert settings["leaderboard"] is False
    assert settings["differs_from_leaderboard"] == [
        "hardmode", "seed", "temperature", "parallel", "max_turns", "timeout",
    ]
    assert "custom" in BENCH._tool_eval_settings_text(settings)


def test_extra_args_go_last_and_may_change_the_backend(monkeypatch):
    args = parse(
        monkeypatch, "--tool-eval", "--port", "30000", "--model", "m",
        "--tool-eval-args=--scenarios TC-01 TC-02 --backend sglang --no-think",
    )
    settings = BENCH.tool_eval_settings(args)
    cmd = BENCH.build_tool_eval_command(["teb"], model="m", base_url="http://localhost:30000/", settings=settings)
    assert cmd[-6:] == ["--scenarios", "TC-01", "TC-02", "--backend", "sglang", "--no-think"]
    assert cmd.count("--backend") == 1
    assert settings["backend"] == "sglang"
    assert settings["leaderboard"] is False
    assert settings["differs_from_leaderboard"] == ["backend"]


@pytest.mark.parametrize("extra", [
    "--seed 1", "--temperature=0.5", "--json-file other.json", "--base-url http://x:1/",
    "--model other", "--api-key sk-x", "--provider openai", "--no-live", "--hardmode",
])
def test_options_the_bench_sets_are_refused_in_extra_args(monkeypatch, capsys, extra):
    err = parse_error(monkeypatch, capsys, "--tool-eval", f"--tool-eval-args={extra}")
    assert "--tool-eval-args may not contain" in err


@pytest.mark.parametrize("extra", ["--temp 0.7", "--json-f x.json", "--max-t 3", "--hard", "--base x"])
def test_abbreviations_of_managed_options_are_refused(monkeypatch, capsys, extra):
    """tool-eval-bench's argparse would expand them to the options the bench sets."""
    err = parse_error(monkeypatch, capsys, "--tool-eval", f"--tool-eval-args={extra}")
    assert "--tool-eval-args may not contain" in err and "short for" in err


def test_other_tool_eval_bench_options_are_passed_on(monkeypatch):
    extra = "--no-think --hardmode-only --short --scenarios TC-01 --label 'a b' --fail-on-safety"
    args = parse(monkeypatch, "--tool-eval", f"--tool-eval-args={extra}")
    assert args.tool_eval_extra_args == [
        "--no-think", "--hardmode-only", "--short", "--scenarios", "TC-01", "--label", "a b", "--fail-on-safety",
    ]


def test_unbalanced_quotes_in_extra_args_are_refused(monkeypatch, capsys):
    err = parse_error(monkeypatch, capsys, "--tool-eval", "--tool-eval-args=--label 'open")
    assert "not valid shell syntax" in err


def test_tool_eval_options_need_tool_eval(monkeypatch, capsys):
    err = parse_error(monkeypatch, capsys, "--tool-eval-seed", "1")
    assert "only apply to --tool-eval" in err
    err = parse_error(monkeypatch, capsys, "--no-tool-eval-hardmode")
    assert "only apply to --tool-eval" in err


@pytest.mark.parametrize("other", [
    ["--test-profile", "gsm8k"], ["--completion-stats"], ["--p2pmark-only"], ["--amd-fabric"],
    ["--prefill-only"], ["--coding-peak"], ["--prompt", "hi"],
])
def test_tool_eval_is_a_mode_of_its_own(monkeypatch, capsys, other):
    err = parse_error(monkeypatch, capsys, "--tool-eval", *other)
    assert "--tool-eval runs only tool-eval-bench" in err


@pytest.mark.parametrize("bad", [
    ["--tool-eval-parallel", "0"], ["--tool-eval-max-turns", "0"], ["--tool-eval-timeout", "0"],
    ["--tool-eval-temperature", "-1"],
])
def test_out_of_range_settings_are_refused(monkeypatch, capsys, bad):
    err = parse_error(monkeypatch, capsys, "--tool-eval", *bad)
    assert bad[0] in err


def test_ref_and_bin_are_exclusive(monkeypatch, capsys):
    err = parse_error(monkeypatch, capsys, "--tool-eval", "--tool-eval-bin", "/x", "--tool-eval-ref", "main")
    assert "--tool-eval-bin" in err


def test_ref_defaults_to_the_pin_and_the_environment_can_override_it(monkeypatch):
    monkeypatch.delenv(BENCH.TOOL_EVAL_REF_ENV, raising=False)
    assert parse(monkeypatch, "--tool-eval").tool_eval_ref == BENCH.TOOL_EVAL_PINNED_REF
    assert len(BENCH.TOOL_EVAL_PINNED_REF) == 40
    monkeypatch.setenv(BENCH.TOOL_EVAL_REF_ENV, "v2.7.0")
    assert parse(monkeypatch, "--tool-eval").tool_eval_ref == "v2.7.0"
    assert parse(monkeypatch, "--tool-eval", "--tool-eval-ref", "main").tool_eval_ref == "main"


def test_other_modes_are_unchanged(monkeypatch):
    args = parse(monkeypatch, "--test-profile", "needle-checksum")
    assert args.tool_eval is False
    assert args.completion_stats is True
    assert args.tool_eval_extra_args == []


@pytest.mark.parametrize("argv, expected", [
    ([], "http://localhost:5000/"),
    (["--port", "8000"], "http://localhost:8000/"),
    (["--host", "10.0.0.5", "--port", "8000"], "http://10.0.0.5:8000/"),
    (["--host", "https://ai.example.com"], "https://ai.example.com/"),
    (["--host", "https://ai.example.com", "--port", "8443"], "https://ai.example.com:8443/"),
    (["--host", "https://ai.example.com/v1"], "https://ai.example.com/v1"),
])
def test_base_url_is_the_bench_target(monkeypatch, argv, expected):
    assert BENCH.tool_eval_base_url(parse(monkeypatch, "--tool-eval", *argv)) == expected


def test_api_key_travels_in_the_environment_not_argv(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/opt/serving/site-packages")
    monkeypatch.setenv("TOOL_EVAL_PROVIDER", "openai")
    env = BENCH.tool_eval_child_env("sk-secret", tmp_path / "venv")
    assert env["TOOL_EVAL_API_KEY"] == "sk-secret"
    assert env["TOOL_EVAL_PROVIDER"] == ""
    assert "PYTHONPATH" not in env
    assert env["VIRTUAL_ENV"] == str(tmp_path / "venv")
    assert env["PATH"].split(os.pathsep)[0] == str(tmp_path / "venv" / "bin")
    assert BENCH.tool_eval_child_env("")["TOOL_EVAL_API_KEY"] == ""


def test_pip_environment_drops_global_pip_settings(monkeypatch):
    monkeypatch.setenv("PIP_CONSTRAINT", "/etc/pip/constraint.txt")
    monkeypatch.setenv("PIP_TARGET", "/opt/venv/lib")
    env = BENCH._tool_eval_pip_env()
    assert "PIP_CONSTRAINT" not in env and "PIP_TARGET" not in env
    assert env["GIT_TERMINAL_PROMPT"] == "0"


@pytest.mark.parametrize("output, expected", [
    ("benchmark_results.json", "benchmark_results.tool-eval"),
    ("/data/runs/glm.JSON", "/data/runs/glm.tool-eval"),
    ("out", "out.tool-eval"),
])
def test_tool_eval_files_go_next_to_the_output(output, expected):
    assert str(BENCH.tool_eval_artifacts_dir(output)) == expected


# ---------------------------------------------------------------------------
# Parsing tool-eval-bench's JSON
# ---------------------------------------------------------------------------

def test_summary_of_a_real_leaderboard_run():
    raw = load_fixture()
    scores = raw["scores"]
    summary = BENCH.summarize_tool_eval_result(raw, {"TC-01": "Direct Specialist Match"})
    assert summary["final_score"] == raw["final_score"] == scores["final_score"]
    assert summary["rating"] == raw["rating"]
    assert (summary["total_points"], summary["max_points"]) == (scores["total_points"], scores["max_points"])
    results = scores["scenario_results"]
    assert summary["scenarios"] == len(results) == raw["total_scenarios"]
    assert summary["pass"] + summary["partial"] + summary["fail"] == len(results)
    assert summary["pass"] == sum(r["status"] == "pass" for r in results)
    assert summary["graded"] == len(results) - len(summary["excluded_scenarios"])
    assert summary["pass_rate"] == round(100.0 * summary["pass"] / summary["graded"], 1)
    # Every category and Hard Mode capability tool-eval-bench reported, in its order.
    assert [c["category"] for c in summary["categories"]] == [c["category"] for c in scores["category_scores"]]
    assert [c["percent"] for c in summary["categories"]] == [c["percent"] for c in scores["category_scores"]]
    assert sum(c["earned"] for c in summary["categories"]) == scores["total_points"]
    assert "P" in [c["category"] for c in summary["categories"]]
    assert [c["capability"] for c in summary["capabilities"]] == [
        c["capability"] for c in scores.get("capability_scores", [])
    ]
    assert summary["capabilities"], "a --hardmode run reports Hard Mode capabilities"
    not_passed = {entry["scenario_id"] for entry in summary["not_passed"]}
    assert not_passed == {r["scenario_id"] for r in results if r["status"] != "pass"}
    assert summary["config_fingerprint"] == raw["config"]["config_fingerprint"]
    assert summary["config"]["seed"] == 42 and summary["config"]["max_turns"] == 30
    assert summary["config"]["concurrency"] == 4 and summary["config"]["temperature"] == 0.0
    assert summary["tool_eval_bench_version"] == raw["tool_eval_bench_version"]
    assert summary["run_status"] == "completed"
    assert summary["safety_gate_passed"] == raw["safety_gate"]["passed"]
    assert summary["engine"]["engine_name"] == "vLLM"
    json.dumps(summary)  # the summary goes into --output as is


def test_summary_without_hard_mode_or_exclusions():
    raw = load_fixture()
    scores = raw["scores"]
    scores.pop("capability_scores", None)
    scores.pop("excluded_scenarios", None)
    scores.pop("completion_rate", None)
    summary = BENCH.summarize_tool_eval_result(raw)
    assert summary["capabilities"] == []
    assert summary["excluded_scenarios"] == []
    assert summary["completion_rate"] == 100.0
    assert summary["graded"] == summary["scenarios"]


def test_excluded_scenarios_are_not_graded():
    raw = load_fixture()
    failed = [r for r in raw["scores"]["scenario_results"] if r["status"] == "fail"]
    victim = failed[0] if failed else raw["scores"]["scenario_results"][0]
    victim["status"], victim["points"], victim["failure_kind"] = "fail", 0, "timeout"
    raw["scores"]["excluded_scenarios"] = [victim["scenario_id"]]
    raw["scores"]["completion_rate"] = 98.9
    summary = BENCH.summarize_tool_eval_result(raw)
    assert summary["graded"] == summary["scenarios"] - 1
    entry = next(e for e in summary["not_passed"] if e["scenario_id"] == victim["scenario_id"])
    assert entry["excluded"] is True and entry["failure_kind"] == "timeout"
    assert summary["completion_rate"] == 98.9


@pytest.mark.parametrize("raw, kind", [
    ([], "json"),
    ({"schema_version": "1"}, "json"),
    ({"schema_version": "2", "scores": {"final_score": 50}}, "json"),
    ({"error": "Model not found"}, "tool"),
    ({"scores": {"scenario_results": []}, "final_score": None}, "json"),
])
def test_unexpected_json_is_an_error(raw, kind):
    with pytest.raises(BENCH.ToolEvalError) as exc:
        BENCH.summarize_tool_eval_result(raw)
    assert exc.value.kind == kind


def test_partial_counts_from_progress_events():
    events = {
        "TC-01": {"status": "pass", "points": 2},
        "TC-02": {"status": "partial", "points": 1},
        "TC-03": {"status": "fail", "points": 0},
    }
    assert BENCH.tool_eval_partial_summary(events) == {
        "scenarios_finished": 3, "points": 3, "pass": 1, "partial": 1, "fail": 1,
    }


# ---------------------------------------------------------------------------
# Install (pinned commit, cached venv) and its failure modes
# ---------------------------------------------------------------------------

def completed(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


class FakeInstaller:
    """Stands in for git/venv/pip; records the commands it was given."""

    def __init__(self, pip_returncode=0, pip_output="", ls_remote=""):
        self.calls = []
        self.pip_returncode = pip_returncode
        self.pip_output = pip_output
        self.ls_remote = ls_remote

    def __call__(self, cmd, timeout=None, env=None):
        self.calls.append(list(cmd))
        if cmd[:2] == ["git", "ls-remote"]:
            return completed(cmd, stdout=self.ls_remote)
        if "venv" in cmd:
            venv = Path(cmd[-1])
            (venv / "bin").mkdir(parents=True, exist_ok=True)
            for name in ("python", "pip"):
                path = venv / "bin" / name
                path.write_text("#!/bin/sh\n")
                path.chmod(0o755)
            return completed(cmd)
        if "pip" in cmd and "install" in cmd:
            return completed(cmd, self.pip_returncode, stderr=self.pip_output)
        if "-c" in cmd:  # version probe
            return completed(cmd, stdout=json.dumps({
                "version": "2.7.1.dev14+g570951a77", "commit": BENCH.TOOL_EVAL_PINNED_REF,
                "python_version": "3.12.3",
            }) + "\n")
        raise AssertionError(f"unexpected command {cmd}")


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv(BENCH.TOOL_EVAL_REPO_ENV, raising=False)
    return tmp_path / "cache" / "tool-eval-bench"


def test_pinned_install_uses_its_own_venv_and_constraints(monkeypatch, cache):
    fake = FakeInstaller()
    monkeypatch.setattr(BENCH, "_tool_eval_run", fake)
    info = BENCH.ensure_tool_eval_install(BENCH.TOOL_EVAL_PINNED_REF)
    env_dir = cache / BENCH.TOOL_EVAL_PINNED_REF
    assert info["python"] == str(env_dir / "venv" / "bin" / "python")
    assert info["installed_now"] is True
    assert info["version"] == "2.7.1.dev14+g570951a77"
    venv_cmd = next(c for c in fake.calls if "venv" in c)
    assert venv_cmd[0] == sys.executable and venv_cmd[-1] == str(env_dir / "venv")
    pip_cmd = next(c for c in fake.calls if "install" in c)
    assert pip_cmd[0] == str(env_dir / "venv" / "bin" / "python")
    assert pip_cmd[-1] == (
        f"tool-eval-bench @ git+{BENCH.TOOL_EVAL_REPO_URL}@{BENCH.TOOL_EVAL_PINNED_REF}"
    )
    constraints = Path(pip_cmd[pip_cmd.index("-c") + 1]).read_text().split()
    assert constraints == list(BENCH.TOOL_EVAL_PINNED_CONSTRAINTS)
    marker = json.loads((env_dir / "install.json").read_text())
    assert marker["commit"] == BENCH.TOOL_EVAL_PINNED_REF

    # Second run: cached, no git/venv/pip at all.
    monkeypatch.setattr(BENCH, "_tool_eval_run", lambda *a, **k: pytest.fail("should be cached"))
    again = BENCH.ensure_tool_eval_install(BENCH.TOOL_EVAL_PINNED_REF)
    assert again["installed_now"] is False and again["python"] == info["python"]


def test_python_without_ensurepip_gets_a_venv_without_pip(monkeypatch, cache):
    """Debian/Ubuntu without python3-venv: venv --without-pip, then this Python's pip --python."""
    fake = FakeInstaller()
    original = fake.__call__

    def no_ensurepip(cmd, timeout=None, env=None):
        if "venv" in cmd and "--without-pip" not in cmd:
            fake.calls.append(list(cmd))
            return completed(cmd, 1, stderr="The virtual environment was not created successfully "
                                            "because ensurepip is not available.")
        if "--without-pip" in cmd:
            fake.calls.append(list(cmd))
            (Path(cmd[-1]) / "bin").mkdir(parents=True, exist_ok=True)
            python = Path(cmd[-1]) / "bin" / "python"
            python.write_text("#!/bin/sh\n")
            python.chmod(0o755)
            return completed(cmd)
        if cmd[1:] == ["-m", "pip", "--version"]:
            fake.calls.append(list(cmd))
            return completed(cmd, stdout="pip 24.0")
        return original(cmd, timeout=timeout, env=env)

    monkeypatch.setattr(BENCH, "_tool_eval_run", no_ensurepip)
    info = BENCH.ensure_tool_eval_install(BENCH.TOOL_EVAL_PINNED_REF)
    venv_python = str(cache / BENCH.TOOL_EVAL_PINNED_REF / "venv" / "bin" / "python")
    pip_cmd = next(c for c in fake.calls if "install" in c)
    assert pip_cmd[:5] == [sys.executable, "-m", "pip", "--python", venv_python]
    assert info["python"] == venv_python


def test_branch_ref_resolves_to_a_commit_without_pinned_constraints(monkeypatch, cache):
    commit = "8ca15b9" + "0" * 33
    fake = FakeInstaller(ls_remote=f"{commit}\trefs/heads/main\n")
    monkeypatch.setattr(BENCH, "_tool_eval_run", fake)
    monkeypatch.setattr(BENCH.shutil, "which", lambda name: "/usr/bin/git")
    BENCH.ensure_tool_eval_install("main")
    pip_cmd = next(c for c in fake.calls if "install" in c)
    assert pip_cmd[-1].endswith(f"@{commit}")
    assert "-c" not in pip_cmd
    assert (cache / commit / "install.json").exists()


def test_tag_resolves_to_the_peeled_commit(monkeypatch):
    tag_object, commit = "a" * 40, "b" * 40
    out = f"{tag_object}\trefs/tags/v9.0\n{commit}\trefs/tags/v9.0^{{}}\n"
    monkeypatch.setattr(BENCH, "_tool_eval_run", lambda cmd, **kw: completed(cmd, stdout=out))
    monkeypatch.setattr(BENCH.shutil, "which", lambda name: "/usr/bin/git")
    assert BENCH.resolve_tool_eval_ref("v9.0", BENCH.TOOL_EVAL_REPO_URL) == commit
    assert BENCH.resolve_tool_eval_ref("ABCDEF1", BENCH.TOOL_EVAL_REPO_URL) == "abcdef1"
    with pytest.raises(BENCH.ToolEvalError, match="no branch or tag"):
        BENCH.resolve_tool_eval_ref("v0.0-missing", BENCH.TOOL_EVAL_REPO_URL)


def test_no_network_is_a_clear_install_error(monkeypatch, cache):
    fake = FakeInstaller(
        pip_returncode=1,
        pip_output="fatal: unable to access 'https://github.com/SeraphimSerapis/tool-eval-bench.git/': "
                   "Could not resolve host: github.com\nerror: subprocess-exited-with-error\n",
    )
    monkeypatch.setattr(BENCH, "_tool_eval_run", fake)
    monkeypatch.setattr(BENCH, "tool_eval_network_problem",
                        lambda repo: "github.com: ConnectError: [Errno -3] Temporary failure in name resolution")
    with pytest.raises(BENCH.ToolEvalError) as exc:
        BENCH.ensure_tool_eval_install(BENCH.TOOL_EVAL_PINNED_REF)
    error = exc.value
    assert error.kind == "network"
    assert "no network access" in error.message and "github.com" in error.message
    assert "--tool-eval-bin" in error.hint and "LLM_BENCH_CACHE_DIR" in error.hint
    assert "Could not resolve host" in error.detail
    assert not (cache / BENCH.TOOL_EVAL_PINNED_REF / "install.json").exists()


def test_pip_failure_with_network_is_an_install_error(monkeypatch, cache):
    fake = FakeInstaller(pip_returncode=1, pip_output="ERROR: Could not find a version that satisfies rich>=14\n")
    monkeypatch.setattr(BENCH, "_tool_eval_run", fake)
    monkeypatch.setattr(BENCH, "tool_eval_network_problem", lambda repo: "")
    with pytest.raises(BENCH.ToolEvalError) as exc:
        BENCH.ensure_tool_eval_install(BENCH.TOOL_EVAL_PINNED_REF)
    assert exc.value.kind == "install"
    assert "exit 1" in exc.value.message and "--tool-eval-ref" in exc.value.hint
    assert "Could not find a version" in exc.value.detail


def test_old_pip_cannot_fill_a_venv_without_pip(monkeypatch, cache):
    def run(cmd, timeout=None, env=None):
        if "--without-pip" in cmd:
            return completed(cmd)
        if "venv" in cmd:
            return completed(cmd, 1, stderr="ensurepip is not available")
        if cmd[1:] == ["-m", "pip", "--version"]:
            return completed(cmd, stdout="pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.12)")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(BENCH, "_tool_eval_run", run)
    with pytest.raises(BENCH.ToolEvalError) as exc:
        BENCH.ensure_tool_eval_install(BENCH.TOOL_EVAL_PINNED_REF)
    assert exc.value.kind == "install"
    assert "pip 22.0" in exc.value.message and "22.3" in exc.value.message
    assert "python3-venv" in exc.value.hint and "ensurepip is not available" in exc.value.detail


def test_short_id_of_the_pinned_commit_is_the_pinned_commit():
    repo = BENCH.TOOL_EVAL_REPO_URL
    assert BENCH.resolve_tool_eval_ref(BENCH.TOOL_EVAL_PINNED_REF[:12], repo) == BENCH.TOOL_EVAL_PINNED_REF
    assert BENCH.resolve_tool_eval_ref(BENCH.TOOL_EVAL_PINNED_REF[:7].upper(), repo) == BENCH.TOOL_EVAL_PINNED_REF
    assert BENCH.resolve_tool_eval_ref("c483eef", repo) == "c483eef"


def test_tag_resolved_online_is_reused_offline(monkeypatch, cache):
    commit = "c" * 40
    monkeypatch.setattr(BENCH.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(BENCH, "_tool_eval_run", FakeInstaller(ls_remote=f"{commit}\trefs/tags/v9.1\n"))
    BENCH.ensure_tool_eval_install("v9.1")

    def offline(cmd, timeout=None, env=None):
        if cmd[:2] == ["git", "ls-remote"]:
            return completed(cmd, 128, stderr="fatal: unable to access: Could not resolve host: github.com")
        raise AssertionError(f"nothing but git ls-remote may run offline: {cmd}")

    monkeypatch.setattr(BENCH, "_tool_eval_run", offline)
    monkeypatch.setattr(BENCH, "tool_eval_network_problem", lambda repo: "github.com: ConnectError")
    console = BENCH.Console(file=io.StringIO(), width=200)
    info = BENCH.ensure_tool_eval_install("v9.1", console)
    assert info["installed_now"] is False
    assert Path(info["python"]).parents[2].name == commit
    assert "No network" in console.file.getvalue() and commit[:12] in console.file.getvalue()
    with pytest.raises(BENCH.ToolEvalError) as exc:
        BENCH.ensure_tool_eval_install("v9.2")  # never resolved: needs the network
    assert exc.value.kind == "network"


@pytest.mark.parametrize("repo, url", [
    ("https://github.com/SeraphimSerapis/tool-eval-bench.git", "https://github.com/SeraphimSerapis/tool-eval-bench.git"),
    ("git@github.com:org/tool-eval-bench.git", "ssh://git@github.com/org/tool-eval-bench.git"),
    ("ssh://git@git.example.com/mirrors/tool-eval-bench.git", "ssh://git@git.example.com/mirrors/tool-eval-bench.git"),
])
def test_mirror_urls_become_pip_urls(repo, url):
    assert BENCH.tool_eval_pip_url(repo) == url


def test_local_mirror_is_installed_through_a_file_url(monkeypatch, cache, tmp_path):
    mirror = tmp_path / "mirrors" / "tool-eval-bench.git"
    mirror.mkdir(parents=True)
    monkeypatch.setenv(BENCH.TOOL_EVAL_REPO_ENV, str(mirror))
    fake = FakeInstaller()
    monkeypatch.setattr(BENCH, "_tool_eval_run", fake)
    BENCH.ensure_tool_eval_install(BENCH.TOOL_EVAL_PINNED_REF)
    pip_cmd = next(c for c in fake.calls if "install" in c)
    assert pip_cmd[-1] == f"tool-eval-bench @ git+{mirror.resolve().as_uri()}@{BENCH.TOOL_EVAL_PINNED_REF}"


def test_unwritable_cache_is_an_install_error(monkeypatch, tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    monkeypatch.setenv("LLM_BENCH_CACHE_DIR", str(blocker))
    monkeypatch.setattr(BENCH, "_tool_eval_run", FakeInstaller())
    with pytest.raises(BENCH.ToolEvalError) as exc:
        BENCH.ensure_tool_eval_install(BENCH.TOOL_EVAL_PINNED_REF)
    assert exc.value.kind == "install" and "LLM_BENCH_CACHE_DIR" in exc.value.hint


def test_helper_commands_that_hang_or_cannot_start_are_install_errors(monkeypatch):
    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python3", timeout=600)

    monkeypatch.setattr(BENCH.subprocess, "run", hang)
    with pytest.raises(BENCH.ToolEvalError) as exc:
        BENCH._tool_eval_run(["python3", "-m", "venv", "/x"], timeout=600)
    assert exc.value.kind == "install" and "600 s" in exc.value.message
    monkeypatch.undo()
    with pytest.raises(BENCH.ToolEvalError) as exc:
        BENCH._tool_eval_run(["/nonexistent/python3", "-m", "venv", "/x"], timeout=10)
    assert exc.value.kind == "install" and "Cannot run" in exc.value.message


def closed_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_unreachable_repository_fails_fast_for_real(monkeypatch, cache):
    """Real venv + pip + git against a repository and index on a closed local port.

    Hermetic: no proxy, pip or git configuration of the host applies, and every helper
    command is capped at 120 s, so it stays a quick refused connection (about 2 s).
    """
    if subprocess.run([sys.executable, "-m", "ensurepip", "--version"], capture_output=True).returncode:
        pytest.skip("this Python has no ensurepip")
    port = closed_local_port()
    monkeypatch.setenv(BENCH.TOOL_EVAL_REPO_ENV, f"http://127.0.0.1:{port}/tool-eval-bench.git")
    monkeypatch.setenv("PIP_INDEX_URL", f"http://127.0.0.1:{port}/simple/")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                 "PIP_PROXY", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    real_run = BENCH._tool_eval_run
    monkeypatch.setattr(BENCH, "_tool_eval_run",
                        lambda cmd, timeout=None, env=None: real_run(cmd, timeout=min(timeout or 120, 120), env=env))
    started = time.monotonic()
    with pytest.raises(BENCH.ToolEvalError) as exc:
        BENCH.ensure_tool_eval_install(BENCH.TOOL_EVAL_PINNED_REF)
    assert exc.value.kind == "network"
    assert f"127.0.0.1:{port}" in exc.value.message
    assert time.monotonic() - started < 120


# ---------------------------------------------------------------------------
# End to end: main() with a fake tool-eval-bench and a stub server
# ---------------------------------------------------------------------------

FAKE_TOOL = r'''
import json, os, sys
argv = sys.argv[1:]
if argv == ["--version"]:
    print("tool-eval-bench 2.7.1.dev14+g570951a77")
    sys.exit(0)
record = {"argv": argv, "cwd": os.getcwd(),
          "env": {k: os.environ.get(k) for k in ("TOOL_EVAL_API_KEY", "TOOL_EVAL_PROVIDER", "PYTHONPATH")}}
with open(os.environ["FAKE_TOOL_EVAL_RECORD"], "w") as fh:
    json.dump(record, fh)
mode = os.environ.get("FAKE_TOOL_EVAL_MODE", "ok")
json_file = argv[argv.index("--json-file") + 1]
raw = json.load(open(os.environ["FAKE_TOOL_EVAL_FIXTURE"]))
def event(**kw):
    sys.stderr.write(json.dumps(kw) + "\n"); sys.stderr.flush()
if mode == "error_event":
    event(event="error", error="connection_failed", message="Connection refused")
    sys.exit(2)
if mode == "weird_events":
    event(event="scenario_result", scenario_id="TC-00", status="odd", points="two", total="x",
          duration_seconds=None)
    sys.stderr.write('{"event": "scenario_start", "total": [1]}\n[1, 2]\n"text"\n')
results = raw["scores"]["scenario_results"]
for idx, result in enumerate(results[:3]):
    event(event="scenario_start", scenario_id=result["scenario_id"], title="T " + result["scenario_id"],
          category="A", index=idx, total=len(results))
    event(event="scenario_result", scenario_id=result["scenario_id"], status=result["status"],
          points=result["points"], index=idx, total=len(results), duration_seconds=1.5)
sys.stderr.write("a plain warning line [not markup]\n")
sys.stderr.flush()
if mode == "hang":
    import time
    time.sleep(60)
if mode == "bad_json":
    open(json_file, "w").write("{not json")
elif mode == "error_envelope":
    json.dump({"schema_version": "1", "error": "boom: model went away"}, open(json_file, "w"))
    sys.exit(1)
elif mode == "no_json":
    sys.exit(0)
else:
    json.dump(raw, open(json_file, "w"))
event(event="benchmark_complete", json_file=json_file, final_score=raw.get("final_score"))
sys.exit(int(os.environ.get("FAKE_TOOL_EVAL_EXIT", "0")))
'''


class StubServer:
    def __init__(self, model="glm-5.3-flash"):
        payload = json.dumps({"object": "list", "data": [{"id": model, "max_model_len": 131072}]}).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/v1/models":
                    body = payload
                elif self.path == "/version":
                    body = b'{"version": "0.0.0-stub"}'
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    stub = StubServer()
    yield stub
    stub.close()


@pytest.fixture
def fake_tool(tmp_path, monkeypatch):
    path = tmp_path / "tool-eval-bench"
    path.write_text(f"#!{sys.executable}\n{FAKE_TOOL}")
    path.chmod(0o755)
    monkeypatch.setenv("FAKE_TOOL_EVAL_RECORD", str(tmp_path / "record.json"))
    monkeypatch.setenv("FAKE_TOOL_EVAL_FIXTURE", str(FIXTURE))
    monkeypatch.setenv("LLM_BENCH_NO_UPDATE_CHECK", "1")
    return path


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["llm_decode_bench.py", *argv])
    try:
        BENCH.main()
    except SystemExit as exc:
        return exc.code
    return 0


def test_end_to_end_with_leaderboard_settings(monkeypatch, tmp_path, server, fake_tool, capsys):
    output = tmp_path / "glm.json"
    monkeypatch.setenv("PYTHONPATH", "/opt/serving")
    code = run_main(
        monkeypatch, "--tool-eval", "--host", "127.0.0.1", "--port", str(server.port),
        "--api-key", "sk-test", "--tool-eval-bin", str(fake_tool), "--output", str(output),
    )
    assert code == 0
    report = json.loads(output.read_text())
    raw = load_fixture()
    assert report["metadata"]["mode"] == "tool_eval"
    assert report["metadata"]["version"] == BENCH.VERSION
    assert report["metadata"]["model"] == "glm-5.3-flash"  # auto-detected like the other modes
    assert report["metadata"]["engine_version"] == "0.0.0-stub"
    tool_eval = report["tool_eval"]
    assert tool_eval["status"] == "completed" and tool_eval["exit_code"] == 0
    assert tool_eval["summary"]["final_score"] == raw["final_score"]
    assert tool_eval["settings"]["leaderboard"] is True
    assert report["tool_eval_raw"] == raw
    record = json.loads((tmp_path / "record.json").read_text())
    assert record["argv"] == [
        "--model", "glm-5.3-flash", "--base-url", f"http://127.0.0.1:{server.port}/", *LEADERBOARD_TAIL,
    ]
    assert tool_eval["command"] == [str(fake_tool), *record["argv"]]
    artifacts = tmp_path / "glm.tool-eval"
    assert record["cwd"] == str(artifacts)
    assert record["env"] == {"TOOL_EVAL_API_KEY": "sk-test", "TOOL_EVAL_PROVIDER": "", "PYTHONPATH": None}
    assert "sk-test" not in json.dumps(report)
    assert (artifacts / "run.json").exists()
    assert tool_eval["report_path"] == str(artifacts / raw["report_path"])  # relative to its directory
    progress = (artifacts / "progress.jsonl").read_text().splitlines()
    assert json.loads(progress[0])["event"] == "scenario_start"
    out = capsys.readouterr().out
    assert "Tool-Call Benchmark" in out and "leaderboard settings" in out
    assert f"{raw['final_score']}/100" in out
    assert "a plain warning line [not markup]" in out


@pytest.mark.parametrize("mode, kind, text", [
    ("error_event", "tool", "connection_failed: Connection refused"),
    ("bad_json", "json", "invalid JSON"),
    ("error_envelope", "tool", "boom: model went away"),
    ("no_json", "tool", "without writing run.json"),
])
def test_tool_failures_are_reported(monkeypatch, tmp_path, server, fake_tool, mode, kind, text):
    monkeypatch.setenv("FAKE_TOOL_EVAL_MODE", mode)
    output = tmp_path / "out.json"
    code = run_main(
        monkeypatch, "--tool-eval", "--host", "127.0.0.1", "--port", str(server.port),
        "--model", "m", "--tool-eval-bin", str(fake_tool), "--output", str(output),
    )
    assert code == 1
    tool_eval = json.loads(output.read_text())["tool_eval"]
    assert tool_eval["status"] == "failed"
    assert tool_eval["error"]["kind"] == kind
    assert text in tool_eval["error"]["message"]
    assert "summary" not in tool_eval


def test_unreachable_server_fails_before_anything_runs(monkeypatch, tmp_path, fake_tool):
    stub = StubServer()
    port = stub.port
    stub.close()
    output = tmp_path / "out.json"
    code = run_main(
        monkeypatch, "--tool-eval", "--host", "127.0.0.1", "--port", str(port),
        "--tool-eval-bin", str(fake_tool), "--output", str(output),
    )
    assert code == 1
    tool_eval = json.loads(output.read_text())["tool_eval"]
    assert tool_eval["error"]["kind"] == "server"
    assert "Cannot connect" in tool_eval["error"]["message"]
    assert not (tmp_path / "record.json").exists()


def test_failed_safety_gate_passes_its_exit_code_on(monkeypatch, tmp_path, server, fake_tool):
    monkeypatch.setenv("FAKE_TOOL_EVAL_EXIT", "2")
    output = tmp_path / "out.json"
    code = run_main(
        monkeypatch, "--tool-eval", "--host", "127.0.0.1", "--port", str(server.port),
        "--tool-eval-bin", str(fake_tool), "--output", str(output), "--tool-eval-args=--fail-on-safety",
    )
    assert code == 2
    tool_eval = json.loads(output.read_text())["tool_eval"]
    assert tool_eval["status"] == "completed" and tool_eval["safety_gate_failed"] is True
    assert tool_eval["summary"]["final_score"] == load_fixture()["final_score"]


def test_other_nonzero_exit_after_a_result_is_a_warning(monkeypatch, tmp_path, server, fake_tool):
    monkeypatch.setenv("FAKE_TOOL_EVAL_EXIT", "1")
    output = tmp_path / "out.json"
    code = run_main(
        monkeypatch, "--tool-eval", "--host", "127.0.0.1", "--port", str(server.port),
        "--tool-eval-bin", str(fake_tool), "--output", str(output),
    )
    assert code == 1
    tool_eval = json.loads(output.read_text())["tool_eval"]
    assert tool_eval["status"] == "completed" and "safety_gate_failed" not in tool_eval
    assert "exited with code 1" in tool_eval["warning"]


def test_malformed_progress_events_do_not_break_the_run(monkeypatch, tmp_path, server, fake_tool):
    monkeypatch.setenv("FAKE_TOOL_EVAL_MODE", "weird_events")
    output = tmp_path / "out.json"
    code = run_main(
        monkeypatch, "--tool-eval", "--host", "127.0.0.1", "--port", str(server.port),
        "--tool-eval-bin", str(fake_tool), "--output", str(output),
    )
    assert code == 0
    assert json.loads(output.read_text())["tool_eval"]["status"] == "completed"


def test_unexpected_errors_still_write_the_output(monkeypatch, tmp_path, server, fake_tool):
    def broken(raw, titles=None):
        raise RuntimeError("summary bug")

    monkeypatch.setattr(BENCH, "summarize_tool_eval_result", broken)
    output = tmp_path / "out.json"
    code = run_main(
        monkeypatch, "--tool-eval", "--host", "127.0.0.1", "--port", str(server.port),
        "--tool-eval-bin", str(fake_tool), "--output", str(output),
    )
    assert code == 1
    error = json.loads(output.read_text())["tool_eval"]["error"]
    assert error["kind"] == "internal" and "RuntimeError: summary bug" in error["message"]


def test_interrupt_stops_tool_eval_bench(monkeypatch, tmp_path, fake_tool):
    """Ctrl-C that reached only the bench: tool-eval-bench gets SIGINT and exits, events are kept."""
    env = BENCH.tool_eval_child_env()
    env["FAKE_TOOL_EVAL_MODE"] = "hang"
    console = BENCH.Console(file=io.StringIO(), width=200)
    timer = threading.Timer(1.0, _thread.interrupt_main)
    started = time.monotonic()
    timer.start()
    try:
        run = BENCH.run_tool_eval_process(
            [str(fake_tool), "--json-file", "run.json"], cwd=tmp_path, env=env, console=console,
        )
    finally:
        timer.cancel()
    assert run["interrupted"] is True
    assert run["returncode"] != 0
    assert time.monotonic() - started < 20
    assert len(run["results"]) == 3  # the scenarios finished before the interrupt
    assert "KeyboardInterrupt" in (tmp_path / "progress.jsonl").read_text()

