"""Standardized benchmark of a running Karmic Kraken container (``lil-bench``).

The orchestrator measures a fixed matrix with ``llm_decode_bench.py``,
records the hardware, PCIe topology, GPU clocks and throttling while it
runs, and uploads one result document to docker.local-inference-lab.ai.
"""

VERSION = "1.3.1"
STANDARD = "standard/1"
SCHEMA = "lil-bench-result/1"
DEFAULT_SITE = "https://docker.local-inference-lab.ai"
TOKEN_URL_PATH = "/bench/token"
