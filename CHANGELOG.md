# Changelog

## 0.4.25 - 2026-06-06

### Distinct-prefix KV mode

- New `--distinct-prefixes` flag: each of the `C` concurrent requests in a cell gets its
  own distinct KV prefix instead of the shared prompt the default matrix uses, to find
  where the KV cache runs out. Traverses column-by-column (each concurrency swept through
  contexts ascending) so a lane runs 8k->32k->128k consecutively and reuses its smaller-size
  prefix; fixed-token decode per lane (waits for completion).
- Per-`(ctx, conc)` prefill-speed grid (`C` `max_tokens=1` scouts, `sum(prompt_tokens)/wall`)
  replaces the single-column prefill view in this mode.
- Engine prefix-cache **hit %** (vLLM `prefix_cache_hits`/`_queries_total`, SGLang
  `cache_hit_rate`) shown live in the SERVER panel with a sparkline, inside each prefill
  cell, and as a report grid; measured as a per-cell counter delta.
- Decode and prefill grids render side-by-side (live and in the final summary);
  capacity-limited cells show measured tok/s with a `*` marker instead of `∅`.
- JSON: top-level `prefill_grid`, per-cell `prefill_*`/`prefix_cache_*` fields, metadata
  `prefix_mode` / `matrix_traversal` / `distinct_prefixes`.
- Mutually exclusive with `--completion-stats`/`--test-profile`, `--request-count`/
  `--run-burst`, and `--standalone-prefill`/`--prefill-only`/`--skip-prefill`. For
  per-request `cached_tokens` in `usage`, start vLLM with `--enable-prompt-tokens-details`
  (the cross-size reuse % is then shown next to the hit %).
- Decode is **fixed-work, not fixed-time**: each lane sends ONE request of `--max-tokens`
  (default 512) and the cell waits for completion, rather than measuring a fixed `--duration`
  window. This is deterministic (no re-prefill-gap artifacts), and because requests complete
  the server returns `usage.prompt_tokens_details.cached_tokens`. Aggregate tok/s = total
  output tokens / the decode window (first token → last completion).
- Prefill speed counts only **newly-computed** tokens (`prompt_tokens - cached_tokens`), so
  a larger context that reuses its cached smaller prefix doesn't inflate the rate.
- Two reuse signals shown by their source: the server-side prefix-cache **hit %** (vLLM
  `/metrics` counter delta, baselined after the scout so it measures decode reuse) sits in
  the **prefill grid**; the client-side **c%** (`usage.prompt_tokens_details.cached_tokens`
  reported in the decode response, needs `--enable-prompt-tokens-details`) sits in the
  **decode grid**. Both in JSON.

## 0.4.18 - 2026-05-15

### Decode warmup

- Standard decode runs now perform a hidden pre-measurement warmup at `C=1` using the largest requested context that fits the current model/KV limits.
- The warmup uses a separate prompt prefix so it does not populate the measured prefix-cache key or get recorded as a prefill result.
- `--decode-warmup-seconds` now defaults to `3`; set it to `0` to disable the hidden warmup.

## 0.4.17 - 2026-05-14

### Sustained decode timing

- Fixed duration-based Sustained Decode so the final matrix uses the same observed OpenAI stream-usage window as the live display.
- JSON now includes `measurement_wall_seconds` separately from `measurement_seconds`, making the requested wall window visible without diluting tok/s with the post-token cancel/scrape tail.

## 0.4.16 - 2026-05-14

### Sustained decode timing

- Fixed duration-based vLLM cells where the final aggregate tok/s could be lower than the last live value because the final Prometheus scrape time was accidentally included in the OpenAI stream-usage measurement window.
- The client-side measured window now closes exactly when `--duration` expires; server `/metrics` scraping remains a validation path and no longer extends the OpenAI throughput denominator.

## 0.4.15 - 2026-05-11

### AMD CPU fabric diagnostics

- Added bundled `tools/amd_fabric/llm_amd_fabric` helper source and Makefile for AMD dual-socket NUMA/fabric diagnostics.
- Added `--amd-fabric` and `--amd-fabric-only` to measure CPU-node vs memory-node read/write/copy bandwidth and pointer-chase latency.
- The AMD fabric report now includes NUMA distance, local/remote bandwidth summaries, bidirectional remote-read bandwidth on 2-socket systems, latency matrices, and xGMI reporting notes.
- AMD fabric default console output is now compact: one summary panel plus one combined `CPU node -> memory node` table. `--amd-fabric-detail` prints the separate full matrices.
- AMD fabric helper now auto-selects up to 64 CPUs per NUMA node instead of 32, and reports bidirectional remote read/write/memcpy saturation.
- The AMD fabric report now parses Linux `perf list --details data_fabric` for visible cross-socket `link_N` counter slots when available.
- Active xGMI socket-link count is explicitly marked as not exposed by standard Linux sysfs/procfs; measured off-diagonal NUMA bandwidth is used as the authoritative fabric signal.
- JSON output now embeds `amd_fabric` results and startup diagnostics when the diagnostic is run.

### P2P fabric diagnostics

- Restored the full default `P2P memcpy bandwidth GB/s` matrix while keeping the compact fabric cards and topology/allreduce/per-GPU summaries.
- Reworked the default P2P summary from a dense key/value table into compact human-readable cards.

## 0.4.14 - 2026-05-10

### Estonia profile live progress

- Completion-token profile mode now updates while requests are still streaming instead of only when a request finishes.
- The live view now shows scout status, queued/launched/active request counts, active stream elapsed time, estimated live tokens, estimated live tok/s, and latest answer text excerpts.
- The prefill scout row is now rendered as prefix-cache/prefill measurement with prompt tokens, TTFT, and prefill tok/s instead of a misleading one-token completed answer.
- Completion-token profile mode now includes the same live GPU/CPU hardware panel used by the normal decode dashboard when `nvidia-smi` is available.
- The hardware panel is placed in the top-right of the completion-token live layout instead of between result tables.
- `--test-profile estonia` now defaults to fixed `--profile-concurrency 30` and `--profile-runs 30` unless the user explicitly provides concurrency/runs/adaptive-level options.
- Recent runs now include the final answer or output excerpt, not just `ok/no`.
- `q` now performs a soft stop in completion-token profile mode and returns a partial final report from completed requests.
- `--dcp-size` now also scales SGLang `max_total_num_tokens` when `/get_server_info` reports only the local KV cache budget.

### P2P fabric diagnostics

- Added bundled CUDA/NCCL fabric diagnostic source and binary target under `tools/p2pmark`.
- Added `--p2pmark` and `--p2pmark-only` to measure CUDA peer memcpy and NCCL allreduce before inference and store the parsed result in JSON.
- P2P diagnostic console output now includes peer-access matrix, P2P bandwidth matrix, per-GPU in/out summaries, peer-distance topology probe, single-writer fan-out, ring bandwidth, all-to-all fabric stress, remote-read latency, and all allreduce rows instead of only a compact one-line summary.
- P2P diagnostic default output is now compact: fabric summary, peer-distance topology, allreduce table, and per-GPU compact view. `--p2pmark-detail` restores the expanded matrices and per-pair tables.
- P2P diagnostic default allreduce sweep compares custom PCIe allreduce vs NCCL from 256 B to 1 MiB, with winner and ratio per size; larger MiB sweeps can be requested explicitly.
- Startup now prints whether the NVIDIA P2P override is effectively loaded from `/proc/driver/nvidia/params`, and prints the suggested modprobe line when it is missing.
- `llm_decode_bench.py` now embeds a compressed Linux x86_64 CUDA/NCCL `llm_p2pmark` fallback helper, so raw single-file installs can run `--p2pmark-only` when compatible CUDA/NCCL runtime libraries are present.

## 0.4.13 - 2026-05-09

### Repository transfer

- Updated the auto-update source URL to the canonical `local-inference-lab/llm-inference-bench` repository after the GitHub transfer.

## 0.4.12 - 2026-05-06

### Prefill-only profiling

- Added `--prefill-only`, which implies `--standalone-prefill`, runs the cold-prefill profile, and exits before sustained decode or Burst / E2E.
- Standalone prefill rows now capture hardware summaries, including PCIe RX/TX averages, and store them in JSON.
- Final prefill tables now show PCIe RX/TX averages when hardware sampling is enabled.
- Hardware sampling now remains enabled in `--display-mode plain`; use `--no-hw-monitor` to disable it explicitly.

## 0.4.11 - 2026-05-04

### Built-in task profiles

- Added `--test-profile estonia`, a built-in GLM long-context completion-token profile with the prompt embedded directly in `llm_decode_bench.py` as a compressed `zlib+base64` blob.
- Added fixed profile controls: `--profile-concurrency` / `--completion-stats-concurrency` and `--profile-runs` / `--completion-stats-runs`.
- Selecting a test profile now implies completion-token statistics mode; if `--completion-stats` is used without `--prompt`, `--prompt-file`, or `--test-profile`, it defaults to `estonia`.
- Improved the live completion-token display with profile name, progress bar, active request count, running completion-token percentiles, correctness rate, TTFT, and generation throughput while the test is still running.
- Updated the completion-token final report to show profile metadata, requested runs, fixed/adaptive concurrency, prompt source, prompt size, and scoring configuration.

## 0.4.10 - 2026-05-01

### Completion-token statistics

- Added `--completion-stats`, a separate adaptive task benchmark for long-answer token-efficiency tests such as the GLM-5.1 dense MLA vs NSA comparison.
- The mode sends one optional prefix-cache scout request, probes increasing decode concurrency, selects the fastest aggregate generation-throughput level, and then collects at least `--completion-stats-min-results` completed answers at that selected concurrency.
- Added final report tables for per-concurrency probe results and selected-concurrency completion-token statistics: avg/p50/p90/p99 completion tokens, elapsed time, TTFT, generation tok/s, max-token hits, and correctness rate.
- Added `--prompt`, `--prompt-file`, `--completion-stats-concurrency-levels`, `--completion-stats-min-results`, `--completion-stats-correct-regex`, `--completion-stats-score-source`, `--completion-stats-save-text`, and related adaptive search controls.
- Completion-stats mode defaults to the GLM `testLuke5.txt` prompt when available locally and uses `40000` max tokens unless `--max-tokens` is explicitly provided.

## 0.4.9 - 2026-04-28

### Internal sync

- Synchronized the standalone `/mnt/llm_decode_bench.py` runtime version with the repository copy before adding the completion-token statistics mode.

## 0.4.8 - 2026-04-27

### Hardware monitor

- Added optional CPU temperature monitoring to the live hardware panel.
- Supports multi-socket/package style labels when exposed by `psutil.sensors_temperatures()` or `/sys/class/hwmon`.
- Hardware summary now includes max CPU temperature when available.

## 0.4.7 - 2026-04-27

### Final report

- The final Primary Summary now renders `Prefill tok/s` and `Aggregate decode tok/s` side-by-side on wide terminals, making the last screen easier to screenshot/share.
- Narrow terminals keep the previous stacked layout to avoid wrapping/cropping the matrices.

## 0.4.6 - 2026-04-27

### Prefill stability

- Added an explicit default-mode prefill/JIT warmup before measured integrated scout prefill rows.
- This makes the first reported 8k prefill row less dependent on whether the token-calibration probe ran cold or was loaded from the calibration cache.
- The warmup uses a unique `[WARMUP_*]` prefix, so it warms kernels/graphs without intentionally reusing the measured `[BENCH_*]` prefix-cache entry.

## 0.4.5 - 2026-04-27

### Startup visibility

- Startup diagnostics are now replayed into the live event log after the TUI starts, so engine detection, KV/cache info, prefill setup, token calibration, and related warnings remain visible.
- If `nvidia-smi` is not available, the benchmark prints a startup warning, records it in the event log, and disables the hardware panel instead of showing an empty/stale HW widget.

## 0.4.4 - 2026-04-27

### Live prefill progress

- Fixed integrated decode-scout prefill freezing the dashboard while waiting for the first token on long-context prompts.
- The TUI now refreshes during integrated prefill, scout-only prefill, and standalone cold-prefill requests, so hardware stats, elapsed time, and ETA keep moving while prefill is in flight.

## 0.4.3 - 2026-04-27

### Reverse proxies

- Fixed OpenAI-compatible reverse proxies that forward `/v1/*` but return 502 or non-Prometheus bodies for `/version`, `/get_server_info`, and `/metrics`.
- Such endpoints are now detected as `openai_proxy`, Prometheus diagnostics are disabled, and sustained decode warmup uses client stream activity instead of waiting 60 seconds for nonexistent scheduler metrics.

### Live prefill progress

- Added live prefill ETA/progress text for long-prefill models.
- The ETA uses the nearest completed prefill sample, including the observed tokenizer-token ratio, so long contexts show an approximate remaining time instead of only a static "prefill" status.
- When no prefill baseline exists yet, the dashboard explicitly says it is waiting for the first completed prefill sample.

## 0.4.2 - 2026-04-26

### Graceful Quit

- Fixed early `q` / Ctrl-C final reports losing already measured prefill rows.
- Partial prefill results are now snapshotted whenever an integrated scout, scout-only prefill, or standalone prefill row completes.
- Primary Summary now includes `Prefill tok/s` on interrupted runs as soon as any prefill row has been measured.

## 0.4.1 - 2026-04-26

### Metrics optionality

- SGLang without `--enable-metrics` is no longer fatal.
- Missing `/metrics` now produces a visible warning and the benchmark continues using OpenAI stream metrics for headline throughput.
- Scheduler/effective-concurrency, KV auto-detection from Prometheus, and Prometheus validation are marked unavailable when metrics are disabled.
- Duration warmup falls back to client stream activity when scheduler metrics are unavailable, avoiding the old 60 second wait for `running_reqs`.
- Request-count Burst / E2E mode also skips repeated `/metrics` scrapes when metrics are unavailable.

### Live dashboard

- Improved the mid-width hardware panel: it now gets more horizontal space and uses a real GPU table before falling back to the ultra-compact layout.

## 0.4.0 - 2026-04-26

### Measurement methodology

- Added three clearly separated benchmark layers: integrated prefill, sustained decode, and optional Burst / E2E decode.
- Kept duration-based Sustained Decode as the default tuning/regression signal. The benchmark waits for the server to admit the requested concurrency and pass warmup before measuring.
- Added request-count Burst / E2E-only mode with `--request-count` and `--warmup-request-count`, matching the finite request-burst style used by tools such as AIPerf.
- Added optional post-sustained Burst / E2E matrix via `--run-burst`, `--burst-request-count`, `--burst-warmup-request-count`, and `--burst-requests-per-concurrency`.
- Switched aggregate decode throughput to OpenAI stream usage where available, using continuous `completion_tokens` during the measured window. Prometheus generation counters remain as validation and scheduler telemetry.
- Changed request-count Burst / E2E payloads to request final OpenAI usage only. `continuous_usage_stats` is reserved for duration-based Sustained Decode, where it is required to measure inside an open time window.
- Reworked client latency metrics to follow OpenAI streaming semantics: TTFT starts at request submission, request latency ends at the last content token, and ITL uses the interval between first and last received content tokens.
- Sustained decode now computes observed ITL and per-user decode throughput from partial streams stopped at the measurement boundary, without using cancel or HTTP close time as a synthetic last token.
- Full request latency remains completed-stream-only.

### Prefill

- Integrated default prefill measurement into decode scout requests, avoiding the old extra repeated prefill phase for normal runs.
- Added scout-only extra prefill contexts through `--prefill-contexts`; default prefill contexts include `8k,64k,128k` plus non-zero decode contexts.
- Added `--standalone-prefill` for the previous cold-prefill profile when debugging ingest behavior.
- Added `--prefill-metric` with client headline measurement and optional Prometheus validation.
- Fixed the old misleading baseline-subtraction prefill approach; headline prefill is now client `prompt_tokens / TTFT`.

### Live dashboard

- Reworked the Rich TUI into adaptive wide, medium, and narrow layouts.
- Added a live hardware panel with GPU SM utilization, memory-controller utilization, VRAM usage, power, temperature, clocks, PCIe rx/tx, and CPU utilization/frequency.
- Added an event log for warmup, readiness, skips, and cell completion.
- Made the aggregate decode panel compact instead of stretching across unused screen width.
- Added inline aggregate-cell latency details when horizontal space allows: `tok/s + TTFT/ITL`. Narrower layouts fall back to stacked or compact cells.
- Added a decode speed trace with fixed deviation scaling so small variance does not look like large jitter.
- Improved terminal ergonomics: `q` now behaves like Ctrl-C and prints partial results instead of hard-exiting.
- Improved narrow-terminal rendering so hardware and decode panels avoid Rich ellipsis in important numeric columns.

### Final report

- Reordered final output so primary prefill and aggregate decode summaries are repeated at the end.
- Replaced the misleading global mixed client distribution table with per-cell client matrices.
- Added per-cell request latency matrices while keeping sample counts and full request-level distributions in JSON.
- The aggregate decode matrix can include compact per-cell `TTFT/ITL` detail; per-request throughput and request latency are shown separately.
- Added explicit notes when Burst / E2E was not run.

### Scheduler, KV, and diagnostics

- Added effective-concurrency tracking from scheduler metrics where available.
- Marked cells that cannot fit the configured KV budget and kept exact deficit information in JSON.
- Added `--dcp-size` support for deriving DCP-adjusted KV budget when server-side introspection is not available.
- Added startup diagnostics to JSON: benchmark args, relevant `NCCL_`/`VLLM_`/`SGLANG_`/`CUDA_`/`OMP_` environment variables, `uname`, GPU query output, and `nvidia-smi topo -m`.
- Added hardware summaries per measured cell when hardware sampling is enabled.

### JSON and documentation

- Expanded JSON output with decode mode, primary decode layer, burst settings, prefill mode, startup diagnostics, event log, hardware summaries, request samples, and methodology metadata.
- Updated README with the new benchmark layers, options, methodology, client latency semantics, hardware panel, and JSON structure.
- Added this CHANGELOG for versioned methodology changes.

### Compatibility and validation

- Kept support for vLLM and SGLang auto-detection.
- Kept Prometheus optional: it is used for validation, scheduler/effective-concurrency state, and server-side diagnostics, while OpenAI streaming remains the primary portable data source.
- Added parity-oriented request-count mode so comparable workloads can be checked against AIPerf-style finite request-burst measurements.
