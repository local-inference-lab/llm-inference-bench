#!/bin/bash
# lock.sh: resolve requirements.in into requirements.lock (pinned, with hashes) using uv in a container.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
docker run --rm -v "$here:/w" -w /w ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --index-strategy unsafe-best-match --generate-hashes --emit-index-url --no-header -o requirements.lock
