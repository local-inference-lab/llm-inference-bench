#!/bin/bash
# build.sh FLASHINFER_TREE [TAG]: build the lil-collectives image locally.
#
# FLASHINFER_TREE is a FlashInfer source tree (a checkout or `git archive` of the commit to ship);
# its commit is read from .lil-flashinfer-commit or git, and optional patch names (shown in the
# results) from .lil-flashinfer-patches. TAG defaults to lil-collectives:YYYYMMDD.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
tree=$(cd "$1" && pwd)
tag=${2:-lil-collectives:$(date -u +%Y%m%d)}
commit=$(cat "$tree/.lil-flashinfer-commit" 2>/dev/null || git -C "$tree" rev-parse HEAD)
patches=$(cat "$tree/.lil-flashinfer-patches" 2>/dev/null || true)
test -d "$tree/flashinfer/experimental/b12x" || { echo "$tree is not a FlashInfer source tree" >&2; exit 1; }
docker buildx build --load -f "$here/Dockerfile" \
  --build-context flashinfer="$tree" \
  --build-arg FLASHINFER_COMMIT="$commit" --build-arg FLASHINFER_PATCHES="$patches" --build-arg IMAGE_TAG="$tag" \
  -t "$tag" "$root"
echo "built $tag (FlashInfer ${commit:0:12}${patches:+ + $patches})"
