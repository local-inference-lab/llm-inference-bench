# syntax=docker/dockerfile:1.7
# ghcr.io/local-inference-lab/lil-collectives: `lil-bench collectives` with PyTorch, cuda-python,
# CUTLASS DSL and a FlashInfer source tree (b12x collectives compile at run time; nothing is
# prebuilt). Build with docker/collectives/build.sh, which supplies the FlashInfer tree as the
# `flashinfer` build context.
ARG CUDA_DEVEL=nvidia/cuda:13.2.1-devel-ubuntu24.04
ARG RUNTIME_BASE=ubuntu:24.04

FROM ${CUDA_DEVEL} AS p2pmark
RUN apt-get update && apt-get install -y --no-install-recommends libnccl-dev make \
    && rm -rf /var/lib/apt/lists/*
COPY tools/p2pmark/llm_p2pmark.cu tools/p2pmark/Makefile /src/p2pmark/
# Static cudart (nvcc's default); NCCL comes from the PyTorch wheels at run time.
RUN make -C /src/p2pmark -B NVCC=/usr/local/cuda/bin/nvcc LDFLAGS=-lnccl \
    "CFLAGS=-O2 -std=c++20 -gencode arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_121,code=sm_121 -gencode arch=compute_120,code=compute_120"

FROM ${RUNTIME_BASE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv pciutils ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY docker/collectives/requirements.lock /tmp/requirements.lock
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps --require-hashes \
         --extra-index-url https://download.pytorch.org/whl/cu132 -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock
COPY --from=flashinfer flashinfer /opt/flashinfer/flashinfer
COPY lil-bench llm_decode_bench.py /opt/lil/bench/
COPY lil_bench /opt/lil/bench/lil_bench
COPY --from=p2pmark /src/p2pmark/llm_p2pmark /opt/lil/bench/tools/p2pmark/llm_p2pmark
ARG FLASHINFER_COMMIT=""
ARG FLASHINFER_PATCHES=""
ARG IMAGE_TAG=""
ENV PATH=/opt/venv/bin:${PATH} \
    PYTHONPATH=/opt/lil/bench:/opt/flashinfer:/opt/flashinfer/flashinfer/experimental/b12x/_compat \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LIL_FLASHINFER_COMMIT=${FLASHINFER_COMMIT} \
    LIL_FLASHINFER_PATCHES=${FLASHINFER_PATCHES} \
    LIL_COLLECTIVES_IMAGE=${IMAGE_TAG} \
    LIL_BENCH_NO_UPDATE_CHECK=1
RUN printf '#!/bin/sh\nexec /opt/venv/bin/python -m lil_bench "$@"\n' > /usr/local/bin/lil-bench \
    && chmod +x /usr/local/bin/lil-bench \
    && lil-bench collectives --help > /dev/null
WORKDIR /root
ENTRYPOINT ["lil-bench", "collectives"]
