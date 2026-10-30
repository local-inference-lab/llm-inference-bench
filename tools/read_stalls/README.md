# lil_read_stalls.py

One-file check whether peer reads between GPUs stall. It needs only Python 3.8+ and the NVIDIA
driver: no CUDA toolkit, PyTorch, Docker or pip packages (the probe kernel ships as PTX).

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/llm-inference-bench/main/tools/read_stalls/lil_read_stalls.py
python3 lil_read_stalls.py --gpus 0,1,2,3        # nvidia-smi indices, 2-8 idle GPUs
python3 lil_read_stalls.py --gpus 0,4,8,12 --json stalls.json
```

Every GPU runs one persistent kernel at the same time and times every single operation of three
patterns (6 s, 6 s and 3 s by default): reading an 80 KiB message from the next GPU, reading it from
every peer (one-shot pull all-reduce traffic), and writing it to the next GPU as the control. On some
CPU fabrics peer reads stall for 1-3 ms in periodic windows while posted writes never do; pull
all-reduce transports then pay those stalls on average although their median looks best.

It prints `READ STALLS: YES` or `READ STALLS: NONE` with the longest read, the number of reads over
1 ms, the share of time lost, the repeat period of the stall windows and which GPUs saw them. Exit
code 0 means no read stalls, 1 read stalls, 2 the test could not run.

The same test runs in `lil-bench collectives --stalls-only` (dedicated Docker image), which uploads
the result to <https://docker.local-inference-lab.ai/bench/collectives>.
