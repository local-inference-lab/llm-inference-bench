# llm-inference-bench

LLM inference decode throughput benchmark with a Rich TUI dashboard.

Measures token generation speed across a matrix of **concurrency levels** and **context lengths**, giving you a full picture of how your serving engine scales under load.

Supports **SGLang** and **vLLM** engines (auto-detected). Works with any OpenAI-compatible API (OpenRouter, Together AI, etc.).

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)

![screenshot](screenshot.png)

## Features

- **Throughput matrix** — benchmarks every combination of concurrency (1, 2, 4, 8, ...) and context length (0K, 16K, 32K, 64K, 128K)
- **Three benchmark layers** — prefill, sustained decode, and optional Burst / E2E decode
- **Two decode entry points** — default duration-based Sustained Decode, plus request-count `--request-count` Burst / E2E-only mode
- **Inline client latency detail** — aggregate decode cells can show `tok/s + TTFT/ITL` when there is enough terminal width
- **Server-side validation** — optionally scrapes Prometheus `/metrics` for vLLM/SGLang validation, queue, KV, and scheduler signals
- **Live TUI dashboard** — adaptive Rich layout with compact modes for narrower terminals
- **Live hardware panel** — GPU temperature, SM/memory utilization, VRAM usage, watts, clocks, PCIe rx/tx, plus CPU utilization/frequency and CPU package temperatures when exposed by the host
- **Fabric diagnostics** — bundled CUDA/NCCL P2P diagnostic plus AMD CPU NUMA/xGMI bandwidth and latency diagnostic
- **Event log** — right-side live history of warmup, readiness, skips, and cell completion while the dashboard redraws
- **Prefill measurement** — integrated decode scout prefill by default, using client `prompt_tokens / TTFT`, with optional standalone cold-prefill profiling and live ETA for long-prefill rows
- **Completion-token statistics mode** — adaptive task benchmark for long-answer quality/token-efficiency tests such as GLM dense MLA vs NSA; warms prefill once, finds the fastest decode concurrency, then collects completion-token distributions
- **Dataset accuracy profiles** — pinned GSM8K (1319 items), stratified MMLU-Pro (1000 items), and GPQA Diamond (198 items) benchmarks with per-item scoring, Wilson confidence intervals, and per-category accuracy, designed to measure quantization degradation (e.g. NVFP4 w4a16 vs w4a4)
- **Paired A/B comparison** — `--compare-baseline` pairs two runs per item and reports accuracy delta, correct/wrong flips, exact McNemar significance, per-category deltas, and completion-token inflation
- **Effective concurrency detection** — shows `(X/Y)*` when the server cannot actually run all requested concurrent requests
- **Dynamic warmup** — uses scheduler metrics when available, with an OpenAI stream fallback when `/metrics` is disabled
- **JSON output** — structured results saved to `benchmark_results.json` for further analysis
- **Smart test skipping** — reads KV cache budget from the server, automatically skips over-capacity cells
- **Engine auto-detection** — automatically detects SGLang vs vLLM and adapts metric scraping
- **Auto-update** — checks GitHub for new versions on startup, offers one-click upgrade

## Installation

See [CHANGELOG.md](CHANGELOG.md) for versioned methodology changes.

```bash
pip install httpx rich psutil
```

## Usage

```bash
# Default: localhost:5000, tests concurrency 1-128, contexts 0K-128K
python3 llm_decode_bench.py

# Custom port and parameters
python3 llm_decode_bench.py --port 5199 --concurrency 1,2,4 --contexts 0,16384

# Custom max tokens and test duration
python3 llm_decode_bench.py --port 5001 --max-tokens 4096 --duration 60

# Full standalone cold-prefill profile when debugging long-context ingest
python3 llm_decode_bench.py --port 5001 \
    --standalone-prefill --prefill-contexts 8k,16k,32k,64k,128k

# Prefill-only communication sweep: no sustained decode matrix
python3 llm_decode_bench.py --port 5001 \
    --prefill-only --prefill-contexts 8k,64k,128k \
    --display-mode plain --hw-monitor-interval 0.5

# Burst / E2E-only mode: exactly N measured requests per cell
python3 llm_decode_bench.py --port 5001 --skip-prefill \
    --contexts 0 --concurrency 1,4 \
    --request-count 40 --warmup-request-count 4 --max-tokens 64

# Full report: prefill + sustained decode + short Burst / E2E section
python3 llm_decode_bench.py --port 5001 \
    --concurrency 1,4,8 --contexts 0,16k \
    --duration 30 --run-burst --burst-requests-per-concurrency 5

# Built-in completion-token statistics profile for the GLM long-context task
python3 llm_decode_bench.py --port 8001 --model GLM-5 \
    --test-profile estonia \
    --profile-concurrency 8 \
    --profile-runs 30 \
    --max-tokens 40000

# Same Estonia task with a generic high-reasoning-effort wrapper
python3 llm_decode_bench.py --port 8001 --model GLM-5 \
    --test-profile estonia-long \
    --profile-concurrency 8 \
    --profile-runs 30 \
    --max-tokens 40000

# Adaptive completion-token statistics profile search
python3 llm_decode_bench.py --port 8001 --model GLM-5 \
    --test-profile estonia \
    --completion-stats-concurrency-levels 1,2,4,8,16,30 \
    --completion-stats-min-results 30

# GSM8K accuracy benchmark (full pinned 1319-item test set, temperature 0)
python3 llm_decode_bench.py --port 8001 --model GLM-5 \
    --test-profile gsm8k

# MMLU-Pro accuracy benchmark (pinned stratified 1000-question subset)
python3 llm_decode_bench.py --port 8001 --model GLM-5 \
    --test-profile mmlu-pro

# GPQA Diamond accuracy benchmark (198 graduate-level science questions;
# fetched from the official password-protected zip on first use, cached locally)
python3 llm_decode_bench.py --port 8001 --model GLM-5 \
    --test-profile gpqa-diamond

# Quantization A/B: run the baseline quant first, then the candidate with a
# paired per-item comparison (accuracy delta, flips, exact McNemar p-value)
python3 llm_decode_bench.py --port 8001 --model GLM-5-w4a16 \
    --test-profile gsm8k --output gsm8k_w4a16.json
python3 llm_decode_bench.py --port 8002 --model GLM-5-w4a4 \
    --test-profile gsm8k --output gsm8k_w4a4.json \
    --compare-baseline gsm8k_w4a16.json

# Standalone paired comparison of two earlier result files (no server needed)
python3 llm_decode_bench.py \
    --compare-baseline gsm8k_w4a16.json --compare-candidate gsm8k_w4a4.json

# Quick subset run (evenly-spread deterministic 200-item slice)
python3 llm_decode_bench.py --port 8001 --model GLM-5 \
    --test-profile mmlu-pro --profile-runs 200

# Remote API with authentication (OpenRouter, Together AI, etc.)
python3 llm_decode_bench.py --host https://openrouter.ai --api-key sk-or-... --model meta-llama/llama-3-70b

# Skip prefill phase for quick decode-only testing
python3 llm_decode_bench.py --skip-prefill --concurrency 1,2,4 --contexts 0

# Manual KV cache budget (for vLLM where auto-detection is unreliable)
python3 llm_decode_bench.py --port 5199 --kv-budget 692736

# CUDA/NCCL P2P fabric diagnostic only
python3 llm_decode_bench.py --p2pmark-only

# AMD CPU socket fabric / NUMA diagnostic only
python3 llm_decode_bench.py --amd-fabric-only
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--host` | `localhost` | Server hostname or full URL (e.g. `https://openrouter.ai`) |
| `--port` | `5000` | Server port (ignored when `--host` is a URL) |
| `--api-key` | | API key sent as `Authorization: Bearer` header |
| `--model` | `Qwen3.5` | Model name for API requests (auto-detected from server) |
| `--concurrency` | `1,2,4,8,16,32,64,128` | Comma-separated concurrency levels |
| `--contexts` | `0,16384,32768,65536,131072` | Comma-separated context lengths (tokens) |
| `--max-tokens` | `2048` | Max tokens to generate per request |
| `--duration` | `30` | Duration per decode test cell (seconds) |
| `--decode-warmup-seconds` | `3` | Hidden pre-measurement warmup at `C=1` using the largest requested context that fits current model/KV limits. Set `0` to disable |
| `--prefill-contexts` | `8k,64k,128k` | Extra scout prefill contexts in default mode; standalone profile contexts with `--standalone-prefill` |
| `--prefill-metric` | `client` | Prefill headline source: `client`, `auto`, or `prometheus`. `auto` adds Prometheus validation when available |
| `--standalone-prefill` | `false` | Run the old repeated cold-prefill profile before decode |
| `--prefill-only` | `false` | Run standalone cold-prefill profiling and exit before sustained decode; JSON and final table include hardware/PCIe summaries when hardware sampling is enabled |
| `--request-count` | `0` | Burst / E2E-only mode: measured requests per cell. `0` keeps Sustained Decode as the primary mode |
| `--warmup-request-count` | `0` | Warmup requests to discard before each `--request-count` cell |
| `--run-burst` | `false` | After sustained decode, run an additional short Burst / E2E matrix |
| `--burst-request-count` | `0` | Measured requests per Burst / E2E cell. `0` means `concurrency × --burst-requests-per-concurrency` |
| `--burst-warmup-request-count` | `0` | Warmup requests per Burst / E2E cell. `0` means `concurrency` |
| `--burst-requests-per-concurrency` | `5` | Auto Burst / E2E measured request multiplier |
| `--test-profile` | | Built-in task profile. `estonia` embeds the GLM long-context prompt inside the script and implies `--completion-stats`. `gsm8k` and `mmlu-pro` are pinned multi-item accuracy benchmarks for quantization A/B tests |
| `--compare-baseline` | | Path to a previous dataset-profile results JSON; after the run, a paired per-item comparison (accuracy delta, flips, exact McNemar p, per-category deltas, token inflation) is printed and embedded in the output JSON |
| `--compare-candidate` | | Standalone mode: compare `--compare-baseline` against this results JSON and exit without contacting a server |
| `--profile-concurrency` | `0` | Fixed task-profile concurrency. `0` keeps adaptive probing |
| `--profile-runs` | `0` | Fixed task-profile measured request count. `0` uses `--completion-stats-min-results` |
| `--completion-stats` | `false` | Run adaptive completion-token statistics mode instead of the decode matrix |
| `--completion-stats-min-results` | `30` | Minimum completed runs collected at the selected concurrency |
| `--completion-stats-concurrency-levels` | `1,2,4,8,16,30` | Candidate concurrency levels for the adaptive probe |
| `--completion-stats-correct-regex` | `\\bestonia\\b` | Regex used to score final-answer correctness; empty disables scoring |
| `--completion-stats-save-text` | `false` | Store full streamed output/reasoning/content text in JSON instead of only final answer/excerpts |
| `--hw-monitor-interval` | `2` | Live CPU/GPU hardware sampling interval in seconds |
| `--hw-gpu-limit` | `8` | Maximum GPUs shown in the live hardware panel |
| `--no-hw-monitor` | `false` | Disable live hardware sampling |
| `--p2pmark` | `false` | Run the bundled CUDA/NCCL fabric diagnostic before the LLM benchmark and embed it in JSON |
| `--p2pmark-only` | `false` | Run only the bundled fabric diagnostic and exit |
| `--p2pmark-detail` | `false` | Print expanded P2P matrices and per-pair topology/latency tables; default report is compact |
| `--p2pmark-mode` | `all` | Diagnostic mode: `bandwidth`, `latency`, `allreduce`, or `all` |
| `--p2pmark-bin` | bundled | Override path to the `llm_p2pmark` binary; default also has an embedded fallback |
| `--amd-fabric` | `false` | Run the bundled AMD CPU NUMA/xGMI fabric diagnostic before the LLM benchmark and embed it in JSON |
| `--amd-fabric-only` | `false` | Run only the AMD CPU fabric diagnostic and exit |
| `--amd-fabric-detail` | `false` | Print separate full AMD fabric matrices; default output is compact |
| `--amd-fabric-bin` | sidecar/PATH | Override path to the `llm_amd_fabric` helper |
| `--amd-fabric-size-mb` | `512` | Buffer size per NUMA bandwidth measurement |
| `--amd-fabric-latency-mb` | `256` | Pointer-chase latency working-set size per NUMA node |
| `--amd-fabric-threads` | `0` | Threads per NUMA node for bandwidth tests; `0` auto-selects up to 64 CPUs per node |
| `--output` | `benchmark_results.json` | Output file path |
| `--kv-budget` | `0` | KV cache budget in tokens (0 = auto-detect) |
| `--skip-prefill` | | Skip prefill reporting entirely |
| `--serial-server` | | Single-flight servers (mlx-lm style): readiness = tokens flowing on ≥1 stream, auto warmup timeout scales with concurrency; at C>1 aggregate tok/s measures the serial queue, not the model |

## Measurement Methodology

### Prefill

Prefill measures input processing speed. By default, prefill is based on scout
requests. Every non-zero decode context already sends one scout request to
populate the prefix cache before the measured decode cell, and the tool records
that scout as a prefill sample. Contexts listed in `--prefill-contexts` that are
not part of the decode matrix are measured once as lightweight scout-only
samples, so default runs still include the 8k sanity point without restoring the
old repeated standalone prefill phase.

The headline metric is client-side `prompt_tokens / TTFT`. If the engine exports
clean Prometheus prefill counters, standalone mode can also print a server-side
validation value.

Default integrated prefill contexts are the union of the non-zero decode
contexts from `--contexts` and the configured `--prefill-contexts`. This removes
the old extra repeated prefill phase from normal runs while still showing ingest
numbers for the exact prompts used by decode and the small 8k sanity point.

Use `--standalone-prefill --prefill-contexts 8k,16k,32k,64k,128k` when you need
the old repeated cold-prefill curve. Use `--prefill-only` for focused ingest
and PCIe communication sweeps; it implies `--standalone-prefill`, exits before
decode, and keeps hardware sampling active even with `--display-mode plain`
unless `--no-hw-monitor` is set.

Use this section to compare long-context ingest speed. Do not mix it with decode
throughput; they stress different parts of the engine.

### Sustained Decode

Sustained Decode is the default duration-based benchmark. Before the measured
matrix starts, the default run performs one hidden `C=1` warmup at the largest
requested context that fits the current model/KV limits. Each matrix cell then
runs for `--duration` seconds after its own readiness warmup and keeps the
requested concurrency saturated by restarting streams as they finish.

Aggregate decode throughput uses OpenAI stream usage by default. For local
vLLM/SGLang this is exact when `continuous_usage_stats` is supported, because
the stream exposes cumulative `completion_tokens` during the measured window.
Prometheus generation counters are still collected as validation and for
scheduler/effective-concurrency state, but they are not the default headline
metric. If continuous usage is not available, the tool falls back to streamed
content chunks and marks the aggregate source in JSON.

Prometheus `/metrics` is optional. If SGLang is started without
`--enable-metrics`, or if a remote server does not expose metrics, the benchmark
prints a visible warning and continues with OpenAI stream metrics. In that mode,
scheduler/effective-concurrency, KV auto-detection from metrics, and Prometheus
validation fields are unavailable.

Use this section as the main tuning/regression signal for kernels, NCCL, DCP,
MTP, scheduler, and KV-cache changes. It answers: "How much decode throughput
can the engine sustain once it is already running this concurrency?"

### Burst / E2E Decode

Burst / E2E Decode is a finite client-facing request burst. It sends a fixed
number of measured requests, waits until they complete, and reports:

```text
sum(completion_tokens) / profiling_wall_time
```

Enable it after the sustained matrix with `--run-burst`. By default it sends
`concurrency × 5` measured requests and `concurrency` warmup requests per cell.
Override with `--burst-request-count` and `--burst-warmup-request-count`.

Use this section for community-facing "what happens if I throw a
batch of N requests at the server?" numbers. It includes admission, scheduling,
prefill/cache effects for that finite burst, and completion behavior. It should
be compared separately from Sustained Decode.

### Request-Count Only Mode

`--request-count N` switches the primary decode cells to a request-count
Burst / E2E-only model:

- Send `--warmup-request-count` requests first and discard them.
- Send exactly `N` measured requests per cell.
- Wait for all measured requests to complete.
- Compute aggregate throughput as `sum(completion_tokens) / profiling_wall_time`.

This mode requests only final OpenAI usage chunks, not continuous usage chunks,
so its request payload matches AIPerf-style finite burst measurements more
closely. Continuous usage is reserved for duration-based Sustained Decode where
the tool must measure inside an open time window.

This mode is best when you want only finite request bursts without running the
Sustained Decode matrix. For full reports, prefer `--run-burst` so both
Sustained Decode and Burst / E2E Decode are present and labeled separately.

If `--run-burst` is not set, the final report prints an explicit Phase 3 note:
`Burst / E2E Decode: Not run`. This is the default to avoid doubling the runtime
of a full matrix accidentally.

### Completion-Token Statistics

`--test-profile estonia` is the built-in long-answer task benchmark for the
GLM-5.1 dense MLA vs NSA style test. `--test-profile estonia-long` uses the same
task with a generic high-reasoning-effort system message and wrapper that ask
the model to do a slower private pass and verification pass before answering,
without adding task-specific chain or decoy hints. It uses
`max_completion_tokens` for the generation cap and sends MiMo `thinking.enabled`
as a profile request override. The long prompt is embedded directly in
`llm_decode_bench.py` as a compressed blob, so the benchmark can be run from a
single script without copying `testLuke5.txt` around. The important questions are:

- how many decode tokens the model needs before it reaches the final answer,
- whether the final answer is correct,
- which parallel decode concurrency gives the best aggregate throughput for
  this task.

The mode uses the OpenAI-compatible chat/completions stream and does not run the
normal context/concurrency decode matrix. It sends one optional `max_tokens=1`
scout request first to populate the server prefix cache. Measured requests then
reuse the exact same prompt so engines with prefix caching can avoid repeated
prefill and focus the benchmark on parallel decode.

For explicit, reproducible profile runs, use fixed concurrency:

```bash
python3 llm_decode_bench.py --port 8001 --model GLM-5 \
    --test-profile estonia \
    --profile-concurrency 8 \
    --profile-runs 30
```

Without explicit profile controls, `estonia` and `estonia-long` default to
`--profile-concurrency 30 --profile-runs 30`; override these when the server
cannot fit that much parallel work or when you want a smaller diagnostic run.
The example above sends exactly 30 measured requests with up to 8 requests in flight. The
live display shows the scout request, queued/launched/active request counts,
active stream elapsed time, estimated live tokens, estimated live tok/s, recent
final answers/excerpts, running completion-token percentiles, correctness rate,
TTFT, generation throughput, and the same live GPU/CPU hardware panel used by
the normal decode dashboard in the top-right area while the run is still in
progress. The scout row is reported as prefix-cache/prefill measurement with
prompt tokens, TTFT, and prefill tok/s; it is not scored as a normal answer.
Press `q` to stop after the currently completed work and print a partial report.

If `--profile-concurrency` is not set, the adaptive flow is:

- run the prefill scout once, unless `--completion-stats-no-prefill-scout` is set,
- run a pilot/probe at C=1,
- probe the configured concurrency levels,
- stop once aggregate generation throughput no longer improves by
  `--completion-stats-min-improvement` for `--completion-stats-patience` levels,
- collect additional runs at the selected concurrency until
  `--completion-stats-min-results` completed answers are available.

The final report prints per-concurrency probe rows and a selected-concurrency
summary with completion-token avg/p50/p90/p99, elapsed time, TTFT, aggregate
generation tok/s, max-token hits, and correctness rate when scoring is enabled.
Correctness is scored by default from the final non-empty answer line using
`--completion-stats-correct-regex`; this matches the GLM dense-MLA vs NSA
methodology where mentioning the right country during reasoning is not enough.

If `--max-tokens` is not explicitly provided in this mode, the tool defaults to
the built-in profile default, currently `40000` for `estonia` and
`estonia-long`. Override it for shorter tasks. `--prompt` and `--prompt-file`
remain available for custom completion-token statistics, but the reproducible
bundled task should use `--test-profile estonia` or `--test-profile
estonia-long`.

If SGLang is running with DCP/CP and `/get_server_info` reports only the local KV
budget, pass `--dcp-size N` or set `LLM_BENCH_DCP_SIZE=N`. For example, a local
`max_total_num_tokens=200000` with `--dcp-size 4` is displayed and treated as an
effective `800000` token KV budget.

### Dataset Accuracy Profiles (gsm8k, mmlu-pro, gpqa-diamond)

`--test-profile gsm8k`, `--test-profile mmlu-pro`, and `--test-profile
gpqa-diamond` are multi-item accuracy benchmarks built on the completion-stats
machinery. Instead of repeating one prompt, every measured request is a
**different pinned dataset item**, so the reported correctness rate is dataset
accuracy, not a resample pass-rate. They are intended as sensitive, externally
comparable anchors for quantization and engine A/B tests (for example NVFP4
`w4a16` vs `w4a4` of the same checkpoint).

Datasets are pinned by sha256 and resolved in this order: `data/<file>` next to
the script, `~/.cache/llm_decode_bench/datasets/`, then download from the pinned
source with hash verification. A hash mismatch is a hard error, so two machines
can never silently measure different item sets.

- `gsm8k` — the full official GSM8K test split (1319 grade-school math word
  problems, MIT license, downloaded verbatim from `openai/grade-school-math`).
  The model is asked to end with the final number alone on the last line;
  scoring is exact final-number match (thousands separators, `$`/`%` and
  trailing punctuation are tolerated). Multi-step generation makes this the
  most quantization-sensitive standard task benchmark that is still trivially
  verifiable.
- `mmlu-pro` — a deterministic stratified 1000-question subset of the
  TIGER-Lab/MMLU-Pro test split (Apache-2.0), proportional per category via
  largest remainder, floor-stride by `question_id` inside each category,
  shipped in `data/mmlu_pro_1000.jsonl`. Up to 10 options per question; the
  model must end with `Answer: <letter>`; scoring is exact letter match with
  tolerant extraction (bold/parenthesised tags, bare final-line letters,
  last-tag-wins fallback in the visible text). The report includes
  per-category accuracy.
- `gpqa-diamond` — all 198 graduate-level "Google-proof" science questions of
  the GPQA Diamond split (CC BY 4.0; biology, chemistry, physics), 4 options
  per question assigned by a deterministic per-item shuffle (seeded by record
  id, identical on every machine), same `Answer: <letter>` scoring as
  `mmlu-pro`. This is the frontier-difficulty anchor; with only 198 items its
  statistical resolution is coarse (~±5 pp paired), so read it alongside
  `gsm8k` and `mmlu-pro`. The GPQA authors distribute the dataset as a
  password-protected zip and ask that plaintext never be republished online
  (anti-contamination), so this dataset is **not** shipped in `data/`: the
  official zip is downloaded on first use (both the archive and the derived
  JSONL are sha256-pinned) and cached under
  `~/.cache/llm_decode_bench/datasets/` only. `.gitignore` guards against
  committing a local copy; please keep it out of public repos.

All dataset profiles default to temperature 0, `max_tokens` 131072 (a generous
reasoning budget so a healthy baseline essentially never truncates and
candidate `max_tokens` hits read as degradation; override with
`--max-tokens`), fixed concurrency 30, no prefix-cache scout (prompts are
unique), and **all dataset items**. If your server's `max_model_len` is at or
below 128k, engines like vLLM reject requests whose prompt + `max_tokens`
exceed the context window — pass a smaller `--max-tokens` in that case.
`--profile-runs N` selects a deterministic evenly-spread N-item subset — the
same N items every run, so subsets stay comparable across configurations.
Item-level results (`item_id`, expected/parsed answer, per-item correctness,
tokens) are stored in the output JSON. The headline metric is accuracy with a
Wilson 95% interval; completion-token percentiles and `max_tokens` hits are
reported alongside as early damage signals (a damaged quant usually inflates
reasoning tokens before accuracy visibly drops).

A request that hits the `max_tokens` limit while a thinking model is still
reasoning — and therefore never emits an answer — is scored as **TRUNCATED**
(glyph `⊘`), a distinct category from **unparseable (format)**, which is when
the model *did* answer but the letter/number could not be read. Both still
count as wrong, but TRUNCATED is a token-budget artifact, not a model failure:
the report shows the count explicitly (`truncated (no answer)` and a
`hit max_tokens` breakdown of how many produced no answer vs answered before
the cap) so a high number is an unambiguous signal to raise `--max-tokens`
rather than a misleading "unparseable". Because a damaged quant tends to think
longer, a rising TRUNCATED count between two runs is itself a degradation
signal, and the paired comparison reports it per side.

### Paired A/B Comparison

`--compare-baseline previous.json` (with a dataset-profile run) or the
standalone `--compare-baseline a.json --compare-candidate b.json` mode pairs
runs **per item id** and reports: accuracy for both sides with Wilson 95%
intervals, the accuracy delta, the exact item flips (correct only in baseline
vs only in candidate), a two-sided exact McNemar p-value over the discordant
pairs, per-category deltas (worst first), completion-token inflation, and
`max_tokens`-hit counts. The comparison is embedded under `comparison` in the
output JSON, including flip item ids for drill-down.

Recommended protocol for quantization comparisons: keep the engine version and
flags identical between runs, change only the checkpoint/quant config; run the
full item set on each endpoint; run the profile twice against the *same*
endpoint first — that self-flip rate is the noise floor (temperature 0 does not
guarantee bitwise determinism under batching) that a real degradation must
exceed. Paired McNemar statistics resolve roughly 1–2 pp differences on the
full GSM8K set; a 30-run single-prompt profile cannot.

### Client Latency Metrics

Client latency metrics follow OpenAI streaming semantics in both modes:

- TTFT is time from request start to first streamed content token.
- TTST is time from first streamed content token to second streamed content token.
- Request latency ends at the last streamed content token, not at the usage-only chunk or HTTP close.
- ITL is `(last_content_token_time - first_content_token_time) / (output_tokens - 1)`.
- Per-user output throughput is `1 / ITL`.

Sustained-duration cells may stop streams at the measurement boundary. In that
case ITL is still valid if at least two content tokens were observed, because it
uses only first/last received token timestamps and never uses cancel or HTTP
close time as a synthetic last token. Full request latency remains available
only for completed streams.

The main aggregate matrix keeps latency compact: a wide terminal shows cells
like `63.1 1k/14`, meaning aggregate decode throughput `63.1 tok/s`, TTFT
`~1000 ms`, and ITL `14 ms`. Per-request throughput and request latency are
shown in separate per-cell matrices. Completion/sample counts are preserved in
JSON but intentionally not printed in the default report because they are mostly
diagnostic and easy to misread as benchmark failures.

### Live Hardware Panel

The live dashboard samples `nvidia-smi` while the benchmark runs. It shows GPU
SM utilization, memory-controller utilization, VRAM used/total, watts/power
limit, temperature, SM/memory clocks, PCIe rx/tx MB/s, and CPU utilization.

VRAM usage and memory-controller utilization are intentionally separate: VRAM is
capacity pressure, while `Mem` is memory-controller activity. PCIe rx/tx comes
from `nvidia-smi dmon -s t`; treat it as a coarse live diagnostic signal, not a
per-collective NCCL profiler.

When hardware sampling is active, every measured decode cell also gets a compact
hardware summary in JSON and in the final report. Startup diagnostics are saved
to JSON as well: benchmark arguments, relevant `NCCL_`/`VLLM_`/`SGLANG_`/`CUDA_`
environment variables, `uname`, GPU query output, and `nvidia-smi topo -m`.

The benchmark also checks the NVIDIA runtime P2P override at startup by reading
`/proc/driver/nvidia/params`, not just the modprobe file. A green startup panel
means the expected `ForceP2P=0x11`, `RMForceP2PType=1`, `RMPcieP2PType=2`,
`GrdmaPciTopoCheckOverride=1`, and `EnableResizableBar=1` values are actually
loaded. If they are missing, the panel prints the suggested
`/etc/modprobe.d/nvidia-p2p-override.conf` line and reminds that the NVIDIA
module must be reloaded or the host rebooted before the file takes effect.

For a deeper fabric sanity check, run:

```bash
python3 llm_decode_bench.py --p2pmark-only
python3 llm_decode_bench.py --p2pmark --p2pmark-mode all --port 8000
```

The bundled `tools/p2pmark/llm_p2pmark` CUDA binary measures CUDA peer memcpy
bandwidth, peer-distance topology behavior, ring bandwidth, all-to-all stress,
dependent remote-read latency, and allreduce behavior across visible GPUs. The
default allreduce sweep compares custom PCIe allreduce vs NCCL from 256 B to
1 MiB, with winner and speedup ratio per size. Use
`--p2pmark-allreduce-sizes-mb 1,2,4,8,16,32,64` for a larger MiB-only sweep.
The default console report is intentionally compact: one fabric summary, one
peer-distance topology table, one allreduce table, and one per-GPU compact view.
Use `--p2pmark-detail` to print full matrices and pair-pattern tables; JSON
output always contains the full raw data.

For single-file installs, `llm_decode_bench.py` includes a compressed Linux
x86_64 CUDA/NCCL fallback helper. If the sidecar binary is missing, the script
extracts it to `~/.cache/llm_decode_bench/bin/`. The fallback still depends on
compatible runtime libraries (`libcudart.so.13` and `libnccl.so.2`). Build a
local sidecar with `make -C tools/p2pmark` or pass `--p2pmark-bin` if the
runtime does not match.

For AMD dual-socket hosts, run:

```bash
python3 llm_decode_bench.py --amd-fabric-only
python3 llm_decode_bench.py --amd-fabric --port 8000
```

The bundled `tools/amd_fabric/llm_amd_fabric` helper measures CPU execution
NUMA node vs memory allocation NUMA node. It reports NUMA distance, read/write
bandwidth, memcpy bandwidth, dependent pointer-chase latency, and a
bidirectional remote-read test for 2-socket systems. Off-diagonal cells are the
practical CPU-socket fabric signal.

The default console report is compact: one summary panel and one combined
`CPU node -> memory node` table. Use `--amd-fabric-detail` to print separate
distance, read, write, memcpy, and latency matrices; JSON output always contains
the full raw data.

In the compact table, `N0->N0` means CPU threads pinned to NUMA node 0 accessing
memory allocated on NUMA node 0. That is local socket traffic. Cross-socket
fabric traffic is shown by off-diagonal rows such as `N0->N1` and `N1->N0`.
The helper also reports bidirectional remote read/write/memcpy saturation, which
runs both socket directions concurrently and is the more relevant aggregate
fabric number.

Linux does not expose a portable active xGMI socket-link count through standard
sysfs/procfs interfaces. The report therefore labels active xGMI links as "not
exposed" and treats measured remote NUMA bandwidth as authoritative. When
Linux `perf list --details data_fabric` exposes cross-socket `link_N` counter
slots, the report prints the number of visible DF link counter slots as a useful
hint, but this is still not the same as a decoded active/trained xGMI link count.
On AMD EPYC 9004/9005 2P platforms the expected link count is board-dependent;
NPS1 reference topologies commonly use four board-wired xGMI links.

Build the helper with `make -C tools/amd_fabric` or pass `--amd-fabric-bin` if
you want to use a custom binary.

### Prefill Metrics

Prefill headline throughput is client `prompt_tokens / TTFT`. In the default
mode these samples come from decode scout requests plus any scout-only extra
contexts, so the benchmark no longer pays for a repeated standalone prefill
phase. If Prometheus exposes uncontaminated prefill counters, standalone prefill
mode can also show server-side throughput as validation. Prometheus is not
required for the headline prefill number.

See [methodology and tool parity notes](docs/aiperf-parity-report-2026-04-26.md) for current comparison data and known workload-parity limits.

## Output

Results are saved as JSON with metadata and per-cell throughput data:

```json
{
  "metadata": {
    "version": "0.4.8",
    "engine": "vllm",
    "model": "Qwen3_5-397B-A17B-NVFP4",
    "timestamp": "2026-03-13T00:30:53",
    "decode_mode": "duration",
    "primary_decode_layer": "sustained_decode",
    "request_count": 0,
    "warmup_request_count": 0,
    "run_burst": true,
    "burst_e2e_status": "enabled",
    "concurrency_levels": [1, 2, 4, 8, 16, 32, 64, 128],
    "context_lengths": [0, 16384, 32768, 65536, 131072]
  },
  "prefill": { ... },
  "results": [ ... ],
  "summary_table": { ... },
  "burst_results": [ ... ],
  "burst_summary_table": { ... },
  "methodology": { ... }
}
```

## Additional tools

### `llm_cjk_watchdog.py` — CJK character leak detector

A standalone streaming watchdog that runs chat completions against any OpenAI-compatible endpoint and watches for unexpected Chinese / CJK Han ideographs in the output. Useful for catching model drift, KV-cache corruption, quantization damage, or other failure modes where an English task starts emitting Chinese tokens.

```bash
# single shot against local SGLang/vLLM on :5000
python3 llm_cjk_watchdog.py

# loop until the model leaks a Chinese character
python3 llm_cjk_watchdog.py --loop

# remote OpenAI-compatible endpoint
python3 llm_cjk_watchdog.py --host https://api.together.xyz \
    --api-key $TOGETHER_API_KEY --model meta-llama/llama-3-70b

# simulate a 40k-token input context
python3 llm_cjk_watchdog.py --context-tokens 40000 --max-tokens 2000
```

Features:

- **Loop mode** — runs indefinitely, aborts the stream the moment a CJK character appears
- **Two-row live overlay** pinned to the bottom of the terminal: row 1 shows the current iteration's live tok/s, tokens, elapsed time, and CJK counter; row 2 shows last-iteration and cumulative stats so they never scroll away
- **Precise tok/s** — uses `stream_options.continuous_usage_stats` so the live readout is the exact `completion_tokens` reported by the server, not an estimate from chunk counts
- **Padding context** — optional synthetic input of configurable token size to reproduce long-context failure modes
- **Exit code 2** when CJK characters are detected (scripting-friendly)

Requires only `requests`. See `python3 llm_cjk_watchdog.py --help` for the full CLI.

## License

MIT
