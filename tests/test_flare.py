"""lil-bench beside a Flare server: container, startup log, package, GPUs and the uploaded run."""

from __future__ import annotations

import gzip
import hashlib
import json
import socketserver
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lil_bench import flare, inventory, server, standard, telemetry  # noqa: E402
from test_lil_bench import GOOD_TOKEN, STUB_BENCH, FakeSource, row, run_main, site  # noqa: E402,F401

FINGERPRINT = "ab" * 32
# A synthetic startup log in Flare's plain (non-terminal) format.
STARTUP = f"""[ PENDING  ] Verify package.  0.0s
[    OK    ] Verify package.  signatures  0.5s

Signing key acceptance
Publisher:  Example Publisher, Inc. (example-org)
Identity:   Verified via huggingface.co/example-org
Signature:  Valid
Source:     https://huggingface.co/example-org/Example-Model-FLARE
License:    mit
            https://huggingface.co/example-base/Example-Model/blob/0123/LICENSE
Stored data: Not checked (--verify-weights to check)

Signing key:
{FINGERPRINT}

Executable GPU code can access your process and data.
Do not proceed if this package came from another source.
Only approve a signing key you independently trust.
[ PENDING  ] Verify stored data.  0.0s
[    OK    ] Verify stored data.  0.0 GiB  0.0s
Signing key accepted for example-org.

[ PENDING  ] Prepare model.  0.0s
[    OK    ] Prepare model.  2 ranks  0.5s
[ PENDING  ] Initialize GPUs.  0.0s
[    OK    ] Initialize GPUs.  GPU 0 · SM120 / GPU 1 · SM120  1.1s
[ PENDING  ] Load weights.  12.0 / 40.0 GiB  10.0s
[    OK    ] Load weights.  40.0 GiB  21.6s
[    OK    ] Prepare execution.  concurrency 8  4.1s
[    OK    ] Allocate cache.  9.0 GiB · 1234567 KV tokens · 96 shared snapshot slots · limited by memory  1.2s
[    OK    ] Read model configuration.  example-base/Example-Model  0.0s
[ PENDING  ] Start server.  0.0s
Startup tuning: 6 graphs completed in 1.25s
  Collective choices · device 0 · diagnostic
  0x00000001: 2 groups · 0x00010024: 1 group · 0x00020040: 2 groups · 0x00040010: 1 group
  init1 · Init b=1 t=1 rows=1 drafts=0 prefill=0 · 0x00000001
  prefill4096 · Prefill b=1 t=4096 rows=4096 drafts=0 prefill=0 · 0x00000001, 0x00020040
  speculative8x1 · Speculative b=8 t=1 rows=8 drafts=0 prefill=0 · 0x00000001, 0x00010024, 0x00040010
  packed-speculative8x8-drafts3-tokens32 · Speculative b=8 t=4 rows=32 drafts=3 prefill=0 · 0x00000001, 0x00020040
[    OK    ] Start server.  http://0.0.0.0:5092 · 32768 tokens/request  4.9s
"""
HUB_START = """Prepared-model cache: /home/flare/.cache/flare/models
Package source: example-org/Example-Model-FLARE@gb202
Resolved package: example-org/Example-Model-FLARE@0123456789abcdef0123456789abcdef01234567 / model.safetensors
Downloading model file: model.safetensors...
"""


def terminal(text: str) -> str:
    """The same log as an interactive terminal draws it: colored rows redrawn in place, a framed prompt."""
    out = []
    for line in text.splitlines():
        if line.startswith("[ PENDING  ]"):
            out.append("\r\x1b[2K[\x1b[36m━━━━━━━━━━\x1b[0m]" + line[12:] + "\r")
            continue
        if line.startswith("[    OK    ]"):
            label, _, rest = line[13:].partition(".  ")
            out.append(f"\r\x1b[2K[\x1b[1;32m    OK    \x1b[0m] {label}.\x1b[2m  {rest}\x1b[0m")
            continue
        if line.startswith(("Publisher:", "Identity:", "Signature:", "Source:", "License:", "Signing key:")):
            out.append(f"\x1b[36m│\x1b[0m {line:<70} \x1b[36m│\x1b[0m")
            continue
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Startup log
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("draw", [str, terminal])
def test_startup_log_steps_trust_and_capacity(draw):
    log = flare.parse_log(draw(STARTUP))
    assert log["started"] and log["earlier_starts"] == 0
    steps = {s["name"]: s for s in log["steps"]}
    assert steps["Load weights"] == {"name": "Load weights", "status": "OK", "detail": "40.0 GiB", "seconds": 21.6}
    assert log["ranks"] == 2 and log["concurrency"] == 8 and log["kv_tokens"] == 1234567
    assert log["model"] == "example-base/Example-Model" and log["tokens_per_request"] == 32768
    assert log["gpus"] == [{"ordinal": 0, "architecture": "SM120"}, {"ordinal": 1, "architecture": "SM120"}]
    trust = log["trust"]
    assert trust["name"] == "Example Publisher, Inc." and trust["namespace"] == "example-org"
    assert trust["verified"] is True and trust["source"] == "https://huggingface.co/example-org/Example-Model-FLARE"
    assert trust["fingerprint"] == FINGERPRINT and trust["license"] == "mit"
    assert trust["license_link"].endswith("/LICENSE")
    assert "\x1b" not in log["text"] and not log["truncated"]


def test_collective_choices_table():
    collectives = flare.parse_log(STARTUP)["collectives"]
    assert collectives["tuning"] == {"graphs": 6, "seconds": 1.25}
    device, = collectives["devices"]
    assert device["device"] == 0
    assert device["groups"] == [{"selectors": ["0x00000001"], "count": 2}, {"selectors": ["0x00010024"], "count": 1},
                                {"selectors": ["0x00020040"], "count": 2}, {"selectors": ["0x00040010"], "count": 1}]
    graphs = {g["name"]: g for g in device["graphs"]}
    assert graphs["speculative8x1"] == {"name": "speculative8x1", "kind": "Speculative", "batch": 8, "tokens": 1,
                                        "rows": 8, "drafts": 0, "prefill": 0,
                                        "selectors": ["0x00000001", "0x00010024", "0x00040010"]}
    assert graphs["packed-speculative8x8-drafts3-tokens32"]["rows"] == 32
    assert len(device["graphs"]) == 4  # the block ends at the server start line
    assert flare.collective_summary(collectives) == (
        "device 0: 6 groups · full/barrier ×2 · bulk_push ×2 · payload_rsag/word_exchange ×1 · pull ×1")


@pytest.mark.parametrize("raw, mode, ctas, name, ambiguous", [
    ("0x00000001", 0, 1, None, True),      # full all-reduce or barrier all-gather
    ("0x00010024", 1, 36, None, True),     # payload_rsag all-reduce or word_exchange all-gather
    ("0x00020040", 2, 64, "bulk_push", False),
    ("0x00030008", 3, 8, "bulk_pull", False),
    ("0x00040010", 4, 16, "pull", False),
    ("0x00050008", 5, 8, "full_phased", False),
    ("0x00060008", 6, 8, "payload_rsag_phased", False),
])
def test_selector_decoding(raw, mode, ctas, name, ambiguous):
    item = flare.decode_selector(raw)
    assert (item["mode"], item["ctas"], item["name"], item["ambiguous"], item["known"]) == (mode, ctas, name, ambiguous, True)
    if ambiguous:
        assert item["operation"] is None and item["all_reduce"] and item["all_gather"]
        assert raw in flare.selector_label(item)  # an ambiguous value is shown by its raw value
    else:
        assert item["operation"] == "all_reduce" and flare.selector_label(item) == f"{name} ×{ctas}"


def test_unknown_selectors_stay_raw():
    unknown = flare.decode_selector("0x00090004")
    assert unknown["mode"] == 9 and unknown["name"] is None and not unknown["known"]
    assert flare.selector_label(unknown) == "0x00090004"
    odd = flare.decode_selector("raw=[01, 02]")
    assert odd["mode"] is None and flare.selector_label(odd) == "raw=[01, 02]"
    log = STARTUP.replace("0x00000001, 0x00020040\n", "0x00000001, raw=[01, 02]\n")
    graphs = {g["name"]: g for g in flare.parse_log(log)["collectives"]["devices"][0]["graphs"]}
    assert graphs["prefill4096"]["selectors"] == ["0x00000001", "raw=[01, 02]"]


def test_the_newest_start_is_read_and_a_remembered_key_comes_from_an_earlier_one():
    # The second start of the container: the trust store remembered the key, so no prompt.
    second = "\n".join(line for line in STARTUP.splitlines() if not line.startswith(
        ("Publisher:", "Identity:", "Signature:", "Source:", "License:", "Signing key")))
    second = second.replace("concurrency 8", "concurrency 4")
    log = flare.parse_log(STARTUP + second)
    assert log["earlier_starts"] == 1 and log["concurrency"] == 4
    assert log["trust"]["from_earlier_start"] and log["trust"]["namespace"] == "example-org"


def test_hub_package_lines():
    log = flare.parse_log(HUB_START + STARTUP)
    assert log["hub"] == {"repository": "example-org/Example-Model-FLARE", "revision": "gb202",
                          "commit": "0123456789abcdef0123456789abcdef01234567", "filename": "model.safetensors"}
    serve = flare.run_options(["flare", "run", "example-org/Example-Model-FLARE", "--revision", "gb202"])
    package = flare.package_identity(serve, log, None, None)
    assert package["kind"] == "hub" and package["repository"] == "example-org/Example-Model-FLARE"
    assert package["revision"] == "gb202" and package["commit"].startswith("01234567")
    assert package["commit_source"] == "log" and package["file"] == {"name": "model.safetensors", "readable": False}
    assert package["publisher"]["fingerprint"] == FINGERPRINT and package["license"]["name"] == "mit"


def test_only_recognized_startup_lines_are_kept():
    text = flare.parse_log("Prepared-model cache: /home/someone/.cache/flare/models\nsome other output\n"
                           + HUB_START + STARTUP)["text"]
    assert "/home/someone" not in text and "some other output" not in text and "PENDING" not in text
    assert "Resolved package: example-org/Example-Model-FLARE@0123" in text and FINGERPRINT in text
    assert "https://huggingface.co/example-base/Example-Model/blob/0123/LICENSE" in text
    assert "0x00000001: 2 groups" in text and "speculative8x1 · Speculative" in text
    log = flare.parse_log("Updating model from example-org/Example-Model-FLARE@gb202\n" + STARTUP)
    assert log["hub"] == {"repository": "example-org/Example-Model-FLARE", "revision": "gb202"}


@pytest.mark.parametrize("stamp, expected", [
    ("2026-11-04T01:00:00.123456789Z", 1793754000.123456),
    ("2026-11-04T01:00:00.1Z", 1793754000.1),  # Docker trims trailing zeros
    ("2026-11-04T01:00:00Z", 1793754000.0),
    ("2026-11-04T02:00:00.5+01:00", 1793754000.5),
    ("0001-01-01T00:00:00Z", None),
])
def test_docker_times(stamp, expected):
    value = flare._unix_time(stamp)
    assert value == pytest.approx(expected) if expected is not None else value is None


def test_a_log_without_startup_tuning():
    static = STARTUP.split("Startup tuning")[0] + "[    OK    ] Start server.  http://0.0.0.0:5092  4.9s\n"
    log = flare.parse_log(static)
    assert "collectives" not in log and log["started"]


# ---------------------------------------------------------------------------
# Docker Engine API on a Unix socket
# ---------------------------------------------------------------------------

def container_doc(tmp_path: Path, **changes) -> dict:
    doc = {
        "Id": "c" * 64, "Name": "/flare-test", "Path": flare.BINARY, "RestartCount": 0,
        "Args": ["run", "/models/model.safetensors", "--publisher-keys", "/models/publisher.json", "--gpus", "0,1",
                 "--max-concurrency", "8", "--mtp-draft-tokens", "3", "--host", "0.0.0.0", "--port", "5092"],
        "Image": "sha256:" + "d" * 64,
        "Config": {"Image": "ghcr.io/local-inference-lab/flare:release-abcdef1-main-20261104", "Tty": False,
                   "Env": ["PATH=/usr/bin", "HF_TOKEN=hf_secret", "NCCL_DEBUG=WARN"], "WorkingDir": "/home/flare",
                   "User": "flare", "Labels": {"org.opencontainers.image.revision": "e" * 40, "com.example": "x"}},
        # A PID above pid_max: nothing under /proc, so files resolve through the mounts.
        "State": {"Pid": 999_999_999, "StartedAt": "2026-11-04T01:00:00.123456789Z"},
        "HostConfig": {"NetworkMode": "host", "DeviceRequests": [
            {"Driver": "", "Count": 0, "DeviceIDs": ["GPU-b", "GPU-d"], "Capabilities": [["gpu"]]}]},
        "Mounts": [{"Type": "bind", "Source": str(tmp_path / "pkg"), "Destination": "/models", "RW": False}],
        "NetworkSettings": {"Ports": {}},
    }
    doc.update(changes)
    return doc


IMAGE_DOC = {
    "Id": "sha256:" + "d" * 64, "Architecture": "amd64", "Created": "2026-11-04T00:00:00Z",
    "Descriptor": {"mediaType": "application/vnd.oci.image.index.v1+json", "digest": "sha256:" + "d" * 64},
    "Manifests": [{"ID": "sha256:" + "8" * 64, "Kind": "image", "Available": True,
                   "Descriptor": {"digest": "sha256:" + "8" * 64, "platform": {"architecture": "amd64"}}},
                  {"ID": "sha256:" + "7" * 64, "Kind": "attestation", "Descriptor": {"digest": "sha256:" + "7" * 64}}],
    "RepoTags": ["ghcr.io/local-inference-lab/flare:release-abcdef1-main-20261104"],
    "RepoDigests": ["ghcr.io/local-inference-lab/flare@sha256:" + "f" * 64],
    "Config": {"Labels": {"org.opencontainers.image.revision": "e" * 40,
                          "org.opencontainers.image.source": "https://github.com/example/flare"}},
}


class DockerAPI:
    """A Docker Engine API on a Unix socket, serving fixed documents and a framed log stream."""

    def __init__(self, path: Path, routes: dict, log_frames: list[bytes]):
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                api.paths.append(self.path)
                route = self.path.split("?")[0]
                if route.endswith("/logs"):
                    body = b"".join(log_frames)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/vnd.docker.multiplexed-stream")
                elif route in routes:
                    body = json.dumps(routes[route]).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                else:
                    body = json.dumps({"message": f"No such object: {route}"}).encode()
                    self.send_response(404)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

        self.paths: list[str] = []
        self.httpd = Server(str(path), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def frame(text: str, stream: int = 2) -> bytes:
    data = text.encode()
    return bytes([stream, 0, 0, 0]) + len(data).to_bytes(4, "big") + data


def test_docker_api_on_its_socket(tmp_path):
    container = container_doc(tmp_path)
    cid = container["Id"]
    routes = {"/containers/json": [{"Id": cid}], f"/containers/{cid}/json": container,
              "/containers/flare-test/json": container,
              f"/images/sha256%3A{'d' * 64}/json": IMAGE_DOC,
              f"/containers/{cid}/changes": [{"Path": "/usr/local/bin/flare", "Kind": 0}]}
    lines = STARTUP.splitlines(keepends=True)
    frames = [frame("".join(lines[i:i + 7])) for i in range(0, len(lines), 7)] + [frame("AFTER THE START\n" * 50)]
    api = DockerAPI(tmp_path / "docker.sock", routes, frames)
    try:
        docker = flare.Docker(socket_path=str(tmp_path / "docker.sock"))
        assert [c["Id"] for c in docker.running()] == [cid]
        assert docker.inspect("flare-test")["Path"] == flare.BINARY
        assert docker.image(container["Image"])["RepoDigests"] == IMAGE_DOC["RepoDigests"]
        assert f"/images/sha256%3A{'d' * 64}/json?manifests=1" in api.paths
        assert docker.changes(cid) == [{"Path": "/usr/local/bin/flare", "Kind": 0}]
        text = docker.logs(container, done=lambda t: bool(flare.START_DONE.search(t)))
        assert "Start server.  http://0.0.0.0:5092" in text and "AFTER THE START" not in text
        # Since one second before the container started (2026-11-04T01:00:00Z).
        assert f"/containers/{cid}/logs?stdout=1&stderr=1&since=1793753999" in api.paths
        assert flare.parse_log(text)["kv_tokens"] == 1234567
        with pytest.raises(flare.DockerError, match="No such object"):
            docker.inspect("missing")
    finally:
        api.close()


def test_docker_access_prefers_the_socket(tmp_path, monkeypatch):
    sock = tmp_path / "d.sock"
    sock.write_text("")
    monkeypatch.setenv("DOCKER_HOST", f"unix://{sock}")
    assert flare.Docker.open().socket_path == str(sock)
    monkeypatch.setenv("DOCKER_HOST", "tcp://10.0.0.1:2375")
    monkeypatch.setattr(flare.shutil, "which", lambda name: "/usr/bin/docker")
    assert flare.Docker.open().cli == "/usr/bin/docker"
    monkeypatch.setattr(flare.shutil, "which", lambda name: None)
    assert flare.Docker.open() is None


# ---------------------------------------------------------------------------
# Which container, which GPUs
# ---------------------------------------------------------------------------

class FakeDocker:
    def __init__(self, containers: list[dict], log: str = STARTUP, changes=(), image=IMAGE_DOC):
        self.containers, self.log, self.changed, self.image_doc = containers, log, list(changes), image

    def describe(self):
        return "fake"

    def running(self):
        return self.containers

    def inspect(self, name):
        found = [c for c in self.containers if name in (c["Id"], c["Name"].lstrip("/"))]
        if not found:
            raise flare.DockerError(f"no container {name}")
        return found[0]

    def image(self, reference):
        return self.image_doc

    def changes(self, container_id):
        return self.changed

    def logs(self, container, done=None, limit=flare.LOG_LIMIT):
        return self.log


def test_container_is_found_by_its_port(tmp_path):
    host = container_doc(tmp_path)
    bridge = container_doc(tmp_path, Id="b" * 64, Name="/flare-bridge", HostConfig={"NetworkMode": "bridge"},
                           Args=["run", "/models/model.safetensors", "--port", "8000"],
                           NetworkSettings={"Ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8001"}]}})
    vllm = container_doc(tmp_path, Id="a" * 64, Name="/vllm", Path="/opt/venv/bin/python", Args=["-m", "vllm"])
    docker = FakeDocker([vllm, host, bridge])
    assert flare.find_container(docker, "http://127.0.0.1:5092")[0]["Name"] == "/flare-test"
    assert flare.find_container(docker, "http://localhost:8001")[0]["Name"] == "/flare-bridge"
    assert flare.container_url(bridge) == "http://127.0.0.1:8001"
    container, why = flare.find_container(docker, "http://127.0.0.1:9999")
    assert container is None and "publishes port 9999" in why
    container, why = flare.find_container(docker, "http://203.0.113.9:5092")
    assert container is None and "--container" in why
    container, why = flare.find_container(docker, None)
    assert container is None and "several Flare containers" in why


DEVICES = [{"index": 0, "uuid": "GPU-a", "pids": set()}, {"index": 1, "uuid": "GPU-b", "pids": {999_999_999}},
           {"index": 2, "uuid": "GPU-c", "pids": set()}, {"index": 3, "uuid": "GPU-d", "pids": {999_999_999}}]


def test_server_gpus_from_nvml_processes_or_container_devices(tmp_path):
    container = container_doc(tmp_path)
    options = flare.run_options(["flare", *container["Args"]])["options"]
    assert flare.server_gpus(999_999_999, container, options, 2, DEVICES) == (["GPU-b", "GPU-d"], "NVML processes")
    blind = [{**d, "pids": set()} for d in DEVICES]  # NVML without the host's PIDs
    assert flare.server_gpus(999_999_999, container, options, 2, blind) == (["GPU-b", "GPU-d"], "--gpus")
    narrowed = flare.run_options(["flare", "run", "/models/model.safetensors", "--gpus", "1"])["options"]
    assert flare.server_gpus(None, container, narrowed, 1, blind) == (["GPU-d"], "--gpus")
    everything = container_doc(tmp_path, HostConfig={"DeviceRequests": [{"Count": -1, "Capabilities": [["gpu"]]}]})
    assert flare.server_gpus(None, everything, {}, 2, blind) == (["GPU-a", "GPU-b"], "package ranks")
    cdi = container_doc(tmp_path, HostConfig={"DeviceRequests": [{"Driver": "cdi", "DeviceIDs": ["nvidia.com/gpu=2"]}]})
    assert flare.container_devices(cdi) == ["2"]
    assert flare.server_gpus(None, None, {}, None, []) == (None, "NVML unavailable")


def test_gpus_are_kept_in_nvml_order():
    blind = [{**d, "pids": set()} for d in DEVICES]
    options = flare.run_options(["flare", "run", "/models/x.safetensors", "--gpus", "1,0"])["options"]
    container = {"HostConfig": {"DeviceRequests": [{"Count": -1, "Capabilities": [["gpu"]]}]}}
    assert flare.server_gpus(None, container, options, 2, blind) == (["GPU-a", "GPU-b"], "--gpus")


def test_host_discovery_prefers_flare_over_vllm_of_other_containers(tmp_path, monkeypatch):
    class Args:
        url, container, server_log = "", "", ""
    first = container_doc(tmp_path)
    docker = FakeDocker([first])
    monkeypatch.setattr(server, "find_server", lambda: {"pid": 77, "argv": ["vllm", "serve", "m"]})
    monkeypatch.setattr(flare, "find_processes", lambda url, proc_root="/proc": [])
    # A vLLM process of another container (another PID namespace) loses to the Flare container.
    monkeypatch.setattr(server, "own_namespace", lambda pid, proc_root="/proc": False)
    assert standard.flare_target(Args, docker) == {"url": "http://127.0.0.1:5092", "container": "flare-test"}
    # lil-bench inside a vLLM container measures that vLLM server.
    monkeypatch.setattr(server, "own_namespace", lambda pid, proc_root="/proc": True)
    assert standard.flare_target(Args, docker) is None
    monkeypatch.setattr(server, "own_namespace", lambda pid, proc_root="/proc": False)
    second = container_doc(tmp_path, Id="f" * 64, Name="/flare-two")
    with pytest.raises(flare.DockerError, match="several Flare containers match .*pass --url or --container"):
        standard.flare_target(Args, FakeDocker([first, second]))
    monkeypatch.setattr(flare, "find_processes", lambda url, proc_root="/proc": [{"pid": 1, "argv": []}] * 2)
    with pytest.raises(flare.DockerError, match="2 flare run processes"):
        standard.flare_target(Args, None)


def test_a_port_that_does_not_speak_http_is_not_flare(monkeypatch):
    import http.client

    def broken(url, timeout=10.0):
        raise http.client.BadStatusLine("garbage")
    monkeypatch.setattr(server, "http_json", broken)
    monkeypatch.setattr(server, "http_text", broken)
    assert server.is_flare("http://127.0.0.1:9") is False


def test_topology_matrices_keep_only_the_servers_gpus():
    topo = ("\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
            "GPU0\t X \tPIX\tNODE\tNODE\tNODE\t0-127\t0\t\tN/A\n"
            "GPU1\tPIX\t X \tNODE\tNODE\tNODE\t0-127\t0\t\tN/A\n"
            "GPU2\tNODE\tNODE\t X \tPIX\tNODE\t0-127\t0\t\tN/A\n"
            "GPU3\tNODE\tNODE\tPIX\t X \tNODE\t0-127\t0\t\tN/A\n"
            "NIC0\tNODE\tNODE\tNODE\tNODE\t X \t\t\t\t\n\nLegend:\n\n  X    = Self\n")
    narrowed = inventory.select_matrix(topo, {1: 0, 3: 1})
    lines = narrowed.split("\n")
    assert lines[0] == "\tGPU0\tGPU1\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID"
    assert lines[1] == "GPU0\t X \tNODE\tNODE\t0-127\t0\t\tN/A"
    assert lines[2] == "GPU1\tNODE\t X \tNODE\t0-127\t0\t\tN/A"
    assert lines[3].startswith("NIC0\tNODE\tNODE\t X ") and "Legend:" in narrowed
    p2p = " \tGPU0\tGPU1\tGPU2\t\n GPU0\tX\tOK\tOK\t\n GPU1\tOK\tX\tCNS\t\n GPU2\tOK\tCNS\tX\t\n\nLegend:\n"
    assert inventory.select_matrix(p2p, {1: 0, 2: 1}).split("\n")[:3] == [" \tGPU0\tGPU1\t", " GPU0\tX\tCNS\t", " GPU1\tCNS\tX\t"]


# ---------------------------------------------------------------------------
# Package, image and integrity
# ---------------------------------------------------------------------------

def package_dir(tmp_path: Path, commit: str | None = None) -> bytes:
    data = b"flare package index" * 1000
    folder = tmp_path / "pkg"
    folder.mkdir(exist_ok=True)
    (folder / "model.safetensors").write_bytes(data)
    (folder / ("0" * 64 + ".safetensors")).write_bytes(b"payload")
    key = list(range(32))
    (folder / "publisher.json").write_text(json.dumps([{"namespace": "example-org", "name": "Example", "public_key": key}]))
    if commit:
        metadata = folder / ".cache" / "huggingface" / "download"
        metadata.mkdir(parents=True, exist_ok=True)
        (metadata / "model.safetensors.metadata").write_text(f"{commit}\nsomeetag\n1793750000.0\n")
    return data


def test_local_package_is_hashed_and_its_download_commit_read(tmp_path):
    data = package_dir(tmp_path, commit="9" * 40)
    container = container_doc(tmp_path)
    serve = flare.run_options([container["Path"], *container["Args"]])
    package = flare.package_identity(serve, flare.parse_log(STARTUP), 999_999_999, container)
    assert package["kind"] == "local" and package["repository"] == "example-org/Example-Model-FLARE"
    assert package["file"] == {"name": "model.safetensors", "size": len(data), "readable": True,
                               "sha256": hashlib.sha256(data).hexdigest()}
    assert package["commit"] == "9" * 40 and package["commit_source"] == "hf download metadata"
    assert package["publisher"]["verified"] is True and package["model"] == "example-base/Example-Model"
    # A package directory names its primary file; payloads beside it are not the package.
    serve_dir = flare.run_options(["flare", "run", "/models", "--publisher-keys", "/models/publisher.json"])
    assert flare.package_identity(serve_dir, {}, None, container)["file"]["sha256"] == hashlib.sha256(data).hexdigest()


def test_publisher_from_the_keys_file_when_the_log_has_no_prompt(tmp_path):
    package_dir(tmp_path)
    container = container_doc(tmp_path)
    serve = flare.run_options([container["Path"], *container["Args"]])
    publisher = flare.package_identity(serve, {}, None, container)["publisher"]
    assert publisher["namespace"] == "example-org" and publisher["verified"] is None
    assert publisher["fingerprint"] == hashlib.sha256(bytes(range(32))).hexdigest()
    assert "--publisher-keys" in publisher["identity"]


def test_image_identity_from_the_container_and_its_image(tmp_path):
    image = flare.image_identity(FakeDocker([]), container_doc(tmp_path))
    assert image["repository"] == "ghcr.io/local-inference-lab/flare" and image["tag"] == "release-abcdef1-main-20261104"
    assert image["revision"] == "e" * 40 and image["repo_digests"] == IMAGE_DOC["RepoDigests"]
    assert image["platform_digests"] == ["sha256:" + "8" * 64] and image["descriptor"] == "sha256:" + "d" * 64
    assert "labels" not in image and "source" not in image and "repo_tags" not in image
    pinned = flare.parse_reference("ghcr.io/local-inference-lab/flare:release-x@sha256:" + "1" * 64)
    assert pinned == {"repository": "ghcr.io/local-inference-lab/flare", "tag": "release-x", "digest": "sha256:" + "1" * 64}
    assert flare.parse_reference("localhost:5000/flare") == {"repository": "localhost:5000/flare", "tag": None, "digest": None}


@pytest.mark.parametrize("change, problem", [
    ({}, None),
    ({"Path": "/models/flare"}, "entrypoint /models/flare"),
    ({"Mounts": [{"Type": "bind", "Source": "/tmp/x", "Destination": "/usr/local/bin/flare", "RW": False}]},
     "mount over /usr/local/bin/flare"),
    ({"Config": {"Env": ["LD_PRELOAD=/tmp/hook.so"], "Image": "flare"}}, "LD_PRELOAD set"),
])
def test_integrity_of_the_container(tmp_path, change, problem):
    code = flare.integrity(FakeDocker([]), container_doc(tmp_path, **change), None)
    assert code["status"] == ("modified" if problem else "unmodified")
    assert (problem or "") in code["reason"]
    changed = flare.integrity(FakeDocker([], changes=[{"Path": "/usr/lib/libcuda.so.1", "Kind": 1},
                                                      {"Path": "/usr/local/bin/flare", "Kind": 0}]),
                              container_doc(tmp_path), None)
    assert changed["status"] == "modified" and changed["changes"] == ["/usr/local/bin/flare"]
    assert flare.integrity(None, None, None)["status"] == "unverified"


def test_describe_a_container(tmp_path):
    package_dir(tmp_path)
    state = {"model_id": "Example-Model", "max_model_len": 32768, "engine": "flare", "flare": {"max_concurrency": 8}}
    described = flare.describe(state, "http://127.0.0.1:5092", docker=FakeDocker([container_doc(tmp_path)]),
                               devices=DEVICES)
    block, limits = described["flare"], described["server"]["limits"]
    assert described["gpus"] == ["GPU-b", "GPU-d"] and described["gpu_indices"] == [1, 3]
    assert described["container_name"] == "flare-test" and not described["notes"]
    assert limits["max_num_seqs"] == 8 and limits["max_num_seqs_source"] == "flare_argv"
    assert limits["tensor_parallel_size"] == 2 and limits["port"] == 5092
    assert described["server"]["environment"] == {"NCCL_DEBUG": "WARN"}  # HF_TOKEN never leaves
    assert described["server"]["argv"][:3] == [flare.BINARY, "run", "/models/model.safetensors"]
    assert block["source"] == "container" and block["log_source"] == "docker logs"
    assert block["revision"] == "e" * 40 and block["image"]["tag"] == "release-abcdef1-main-20261104"
    assert block["container"] == {"network_mode": "host", "tty": False, "started": "2026-11-04T01:00:00.123456789Z",
                                  "restarts": 0, "user": "flare"}
    assert block["collectives"]["profiles"]["0x00020040"]["name"] == "bulk_push"
    assert block["startup"]["kv_tokens"] == 1234567 and block["gpus"] == {"source": "NVML processes", "count": 2}
    assert "hf_secret" not in json.dumps(described, default=str)


def test_describe_without_docker_names_what_is_missing(monkeypatch):
    monkeypatch.setattr(flare, "find_process", lambda url, proc_root="/proc": None)
    described = flare.describe({"flare": {}}, "http://127.0.0.1:5092", docker=None, devices=[])
    flare_block = described["flare"]
    assert flare_block["source"] == "api" and flare_block["image"] == {} and described["image"]["known"] is False
    assert flare_block["collectives"]["status"] == "not reported" and described["integrity"]["status"] == "unverified"
    assert any("no Docker access" in note for note in described["notes"])


def test_server_state_of_flare(monkeypatch):
    models = {"data": [{"id": "Example-Model", "root": "example-base/Example-Model", "max_model_len": 32768,
                        "flare": {"max_concurrency": 8, "prefill_chunk_size": 4096}}]}
    metrics = "flare_total_kv_tokens 1234567\nflare_kv_page_tokens 64\nflare_http_request_capacity 8\n" \
              "flare_active_requests 0\nflare_pending_requests 2\n"
    monkeypatch.setattr(server, "http_json", lambda url, timeout=10.0: models if url.endswith("/v1/models") else {})
    monkeypatch.setattr(server, "http_text", lambda url, timeout=10.0: metrics)
    state = server.server_state("http://x")
    assert state["engine"] == "flare" and state["version"] is None and state["flare"]["max_concurrency"] == 8
    assert state["kv_tokens"] == 1234567 and state["kv_tokens_source"] == "flare" and state["kv_page_tokens"] == 64
    assert standard.server_kv_tokens(state, None) == 1234567
    assert server.busy_requests("http://x") == 2.0 and server.is_flare("http://x")


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------

def test_quick_run_beside_a_flare_container(monkeypatch, capsys, site, tmp_path):  # noqa: F811
    data = package_dir(tmp_path)
    stub = tmp_path / "llm_decode_bench.py"
    stub.write_text(STUB_BENCH)
    monkeypatch.setattr(standard, "BENCH", stub)
    monkeypatch.setattr(standard, "RESULT_DIRS", (str(tmp_path / "results"),))
    docker = FakeDocker([container_doc(tmp_path)])
    monkeypatch.setattr(flare.Docker, "open", classmethod(lambda cls: docker))
    monkeypatch.setattr(flare, "nvml_devices", lambda: DEVICES)
    monkeypatch.setattr(server, "find_server", lambda: pytest.fail("a Flare --url never looks for vLLM"))
    monkeypatch.setattr(server, "is_flare", lambda url: True)
    monkeypatch.setattr(server, "server_state", lambda url: {
        "model_id": "Example-Model", "max_model_len": 32768, "root": "example-base/Example-Model", "engine": "flare",
        "flare": {"max_concurrency": 8}, "kv_tokens": 100000, "kv_tokens_source": "flare", "version": None})
    monkeypatch.setattr(server, "busy_requests", lambda url: 0.0)
    monkeypatch.setattr(standard.time, "sleep", lambda s: None)
    sources = []

    def source(indices=None):
        sources.append(indices)
        return FakeSource([[row(2800), row(2810)] for _ in range(100000)])
    monkeypatch.setattr(telemetry, "open_source", source)
    selected = []

    def collect(gpus_selected=None):
        selected.append(gpus_selected)
        return {"gpus": [{"index": i, "host_index": h, "name": "GPU", "power": {"enforced_limit_w": 300.0}, "clocks": {}}
                         for i, h in enumerate((1, 3))],
                "tuning": [], "pcie": {"gpu_paths": {}}, "cpu": {"lscpu": {"Model name": "CPU"}},
                "memory": {"mem_total_mib": 1024}}
    monkeypatch.setattr(inventory, "collect", collect)
    argv = ["run", "--profile", "quick", "--site", site.url, "--url", "http://127.0.0.1:5092", "--no-p2pmark"]
    code, out = run_main(argv, monkeypatch, capsys, {"LIL_BENCH_TOKEN": GOOD_TOKEN})
    assert code == 0, out
    assert "Flare container flare-test · http://127.0.0.1:5092" in out
    assert "max concurrency 8 (--max-concurrency)" in out
    assert "Flare eeeeeeeeeeee · image ghcr.io/local-inference-lab/flare:release-abcdef1-main-20261104" in out
    assert "package example-org/Example-Model-FLARE · signed by Example Publisher, Inc. (example-org), verified" in out
    assert "startup collectives (docker logs): 6 tuning graphs in 1.25 s" in out
    assert "GPUs 1, 3 of this machine (NVML processes)" in out and "container code: the image's own" in out
    assert sources == [[1, 3]] and selected == [["GPU-b", "GPU-d"]]
    document = site.uploads[0]
    assert document["engine"] == "flare" and document["summary"]["engine"] == "flare"
    assert document["image"]["alias"] == "ghcr.io/local-inference-lab/flare:release-abcdef1-main-20261104"
    assert document["integrity"]["status"] == "unmodified"
    assert document["server"]["limits"]["max_num_seqs"] == 8 and document["server"]["limits"]["kv_tokens"] == 100000
    assert document["server"]["argv"][1] == "run" and document["server"]["url"] == "http://127.0.0.1:5092"
    package = document["flare"]["package"]
    assert package["file"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert document["flare"]["collectives"]["devices"][0]["graphs"][2]["selectors"][1] == "0x00010024"
    assert document["flare"]["log"]["text"].startswith("[    OK    ] Verify package.")
    assert set(document["server_telemetry"]["metrics"]) == set(telemetry.FLARE_SERVER_METRICS)
    assert document["results"]["p2pmark"]["status"] == "skipped"
    assert "hf_secret" not in json.dumps(document) and GOOD_TOKEN not in json.dumps(document)
    saved = list((tmp_path / "results").glob("*.json.gz"))
    assert json.loads(gzip.decompress(saved[0].read_bytes()))["flare"]["revision"] == "e" * 40


def test_a_url_that_is_not_flare_keeps_the_vllm_path(monkeypatch):
    monkeypatch.setattr(server, "is_flare", lambda url: False)

    class Args:
        url, container, server_log = "http://127.0.0.1:8000", "", ""
    assert standard.flare_target(Args, None) is None
    Args.container = "flare-test"
    with pytest.raises(flare.DockerError, match="--container needs"):
        standard.flare_target(Args, None)


def test_concurrency_skips_name_the_flare_setting():
    assert "Flare --max-concurrency 8" in standard.concurrency_skip(16, 8, "flare_argv")
    assert "the Flare package's concurrency 8" in standard.concurrency_skip(16, 8, "flare_package")
    plan = standard.build_plan(standard.PROFILES["standard"], {"max_model_len": 1 << 20, "max_num_seqs": 8,
                                                               "max_num_seqs_source": "flare_package"})
    assert any("package's concurrency" in (c["reason"] or "") for c in plan if c.get("concurrency") == 16)


def test_results_skip_a_missing_cache_volume(monkeypatch, tmp_path):
    is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda self: False if str(self) == "/cache" else is_dir(self))
    monkeypatch.setattr(standard, "RESULT_DIRS", ("/cache/lil-bench", str(tmp_path / "home" / "lil-bench")))
    assert standard.result_dir() == tmp_path / "home" / "lil-bench"


def test_server_recorder_samples_flare_counters(tmp_path):
    text = "flare_generated_tokens_total 10\nflare_active_requests 2\nvllm:generation_tokens_total 5\n"
    recorder = telemetry.ServerRecorder("http://x", 0.0, fetch=lambda: text, proc_root=str(tmp_path),
                                        sys_root=str(tmp_path), names=telemetry.FLARE_SERVER_METRICS)
    recorder.record_once()
    series = recorder.series()
    assert series["metrics"]["flare_generated_tokens_total"] == [10.0]
    assert "vllm:generation_tokens_total" not in series["metrics"]
    assert series["fields"]["metrics"].startswith("Flare Prometheus")
