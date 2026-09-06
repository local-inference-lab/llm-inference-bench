# Changelog

## 0.6.2 - 2026-09-06

### Tagged release of guarded decode measurements

- Publishes the default exact-repetition guard, failed-stream invalidation,
  request-specific SGLang cancellation, scheduler-drain checks, and interactive
  model-output viewer as a versioned release.
- The benchmark reports invalid decode cells as errors, retains bounded
  diagnostics in JSON, and leaves summary throughput `null` for those cells.
- Runtime behavior, CLI defaults, sampling parameters, and result schemas are
  unchanged from 0.6.1. The version and release documentation identify the
  downloadable package; the 0.6.0 and 0.6.1 entries describe its functional changes.

## 0.6.1 - 2026-09-06

### Failed decode streams invalidate throughput

- A disconnected or failed stream invalidates its sustained or request-count
  decode cell, including failures before warmup completes. Tables show `ERROR`,
  JSON retains the failure reason, and summary throughput is `null`.
- Missing measurement timestamps cannot raise a secondary exception or turn a
  failed request into a zero-throughput result. Scheduler drain checks remain
  active before another cell starts.
- Regression tests exercise early disconnection, request-count warmup and
  measurement errors, terminal output, and JSON reports.

## 0.6.0 - 2026-09-06

### Decode repetition guard and output preview

- Sustained and request-count decode cells check reasoning and content per
  request for sustained exact repetition by default, including their warmups.
  Four identical cycles spanning at least 4096 Unicode characters invalidate
  the cell: tables show `ERROR: loop` instead of throughput. The existing
  targeted SGLang abort and scheduler-drain boundaries remain active.
- JSON retains bounded repetition evidence and invalid cells; summary speeds
  are `null` for those cells. `--no-loop-detection` permits deliberate repetitive
  workloads without changing their sampling parameters. Detection is a bounded
  exact-text heuristic, not a semantic correctness verdict.
- The `o` key opens a bounded model-output panel in the decode dashboard.
  Space freezes its snapshot, PageUp/PageDown navigate history, End restores
  live scrolling, and `[`/`]` select a worker without interleaving responses.
  Preview controls do not pause the request reader or create inference work.
- Regression tests cover threshold and fragment-boundary behavior, per-request
  and per-channel isolation, concurrent cancellation, report/resume handling,
  terminal key parsing, bounded history, and compact dashboard rendering.

## 0.4.34 - 2026-09-03

### SGLang verifier rates use matched metric snapshots

- SGLang speculative decode now derives verifier steps per second from each
  matched generation-throughput and acceptance-length metric snapshot, then
  reports the median interval rate.
- The final acceptance-length gauge is no longer applied to the complete
  measurement window. That gauge describes only a recent scheduler interval
  and can otherwise make a stable verifier path appear to vary with the final
  generated tokens.

## 0.4.33 - 2026-09-03

### Decode cells require scheduler isolation

- Duration-based SGLang measurements now send a targeted `/abort_request` for
  the request id returned by the OpenAI stream when the timing window closes.
  Closing an HTTP stream alone can leave SGLang generating briefly after the
  client has discarded the request.
- Every decode cell now starts and finishes with a metrics-backed scheduler
  drain barrier. A server that does not reach zero running and queued requests
  fails the cell instead of silently measuring a higher effective concurrency.
- The abort is request-specific; the benchmark never uses SGLang's
  `abort_all` operation and therefore does not cancel unrelated traffic.

## 0.4.32 - 2026-09-01

### Consistency profiles no longer pin sampling

- Reverted the `temperature 0.6` / `top_p 0.95` defaults that 0.4.31 added to `estonia`, `estonia-v1`, `estonia-long` and `hotel-lights`. These profiles again run on the server/model default unless `--completion-stats-temperature` / `--completion-stats-top-p` are given. The effective values (or `None`) stay recorded in result metadata and in the Configuration panel, and `--completion-stats-seed` remains available for reproducible resamples.

## 0.4.31 - 2026-09-01

### Answer scoring rewrite for the consistency profiles (estonia, hotel-lights)

Both scorers could report a pass without the model having answered:

- `estonia` applied `\bestonia\b` to the last non-empty line, so an answer that concluded **Latvia** but mentioned Estonia in a contrast or parenthetical passed, and — because an empty visible answer fell back to the reasoning stream — a run that hit `max_tokens` inside its thinking with the word "Estonia" in the last reasoning line passed too (reproduced from a local MiMo log).
- `hotel-lights` (`numeric_exact`) took the last number anywhere in the selected text: `Not 48` passed, `48 rooms out of 100` failed as "got 100", reasoning-only output could be scored, and a truncated stream was never labelled as such.

Changes:

- **Reasoning is never scored.** Only the visible `content` is used. Inline `<think>…</think>` blocks streamed inside `content` are split off (an unclosed `<think>` counts as reasoning). An empty visible answer is `truncated` / `stalled` / `timeout` / `fail (no visible answer)` depending on how the stream ended — never a pass.
- **Anchored answer extraction.** An explicit `Final answer:` / `Answer:` line wins over the last line; markdown emphasis, bullets and trailing punctuation are stripped. Incomplete streams (`finish_reason=length`, stall, wall-clock) are only scored when they carry an explicit anchored answer or a bare-number line.
- **New `country_exact` scorer for `estonia`, `estonia-v1`, `estonia-long`.** It extracts the country the answer *asserts*: negated mentions (`not Latvia`), parenthetical asides next to an asserted country, and superseded values (`from Latvia to Estonia`, `Latvia -> Estonia`) do not count. Results are labelled `PASS`, `DECOY` (committed to the planted country, Latvia), `NOT_STATED` (the model says the packet does not contain the answer), `AMBIG` (one conclusion line asserts two countries; scored wrong and flagged for review) or `FAIL` (other country / unparseable), with the parsed country in the report.
- **Strict `numeric_exact`.** Accepts a bare-number line, a number anchored at the end of the line (`= 48`, `answer is 48`, `Final answer: 48`) or a line whose only number is not negated or hedged; `Not 48`, `47, not 48`, hedged statements and multi-number lines without an anchor are `unparseable`, reported as such instead of as a wrong number.
- **Incomplete runs are reported, not hidden.** New labels `STALL` and `TIMEOUT` join `TRUNC` (glyph `⊘`). They still count as not-correct in the headline pass rate, but the selected-concurrency table adds `pass rate, finished runs only`, the score line lists every label with a count (`PASS 19 / FAIL 0 / DECOY 4 / NOT_STATED 5 / TRUNC 2`), and the Failed Final Answers table colours them apart. A client-side cancel (`q`) is `CANCEL` and not scored at all.
- **Estonia prompt v2.** The 700k-character packet is byte-identical; only the question tail changed: it now names the *vendor (manufacturer)* — the packet links a vendor account, never a "manufacturer" — and asks for exactly one `Final answer: <country>` line. The old tail is available as `--test-profile estonia-v1` (alias `estonia-legacy`), scored with the new scorer. `estonia-long` wraps the v2 prompt. Result metadata records `profile_version`, `scorer_version` and the sha256 of the prompt actually sent; `--compare-baseline` warns when prompt or scorer versions differ.
- **Sampling made visible.** New `--completion-stats-seed BASE` sends `seed = BASE + run_index` so the N resamples differ from each other but are identical across engines/quants. `--completion-stats-top-p` can now also come from a profile default. The Configuration panel prints the effective sampling (or "server/model defaults (unpinned)") and watchdog settings, and result metadata records temperature, top_p and seed. (0.4.31 also pinned `temperature 0.6` / `top_p 0.95` for the consistency profiles; that was reverted in 0.4.32.)
- **Watchdogs for uncapped runs.** `--completion-stats-stall-timeout` (default 600 s; 0 disables) closes a stream that produces no token for that long and scores it `STALL`. `--completion-stats-request-timeout` (default 0 = off) is a per-request wall-clock limit for models that loop without stalling when `max_tokens` is omitted; such runs are scored `TIMEOUT`. Previously the read timeout was unlimited and a looping model could hold the endpoint forever.
- Tests: `tests/test_answer_scoring.py` covers the fixtures above (real local final answers, the reported false positives, think-block splitting, watchdog labels, prompt v1/v2 integrity and the summary breakdown).

## 0.4.30 - unreleased

- `--forced-token-id`: fixed-token decode route diagnostic (logit_bias=100, temperature 0, ignore_eos) for the sustained/burst matrices.

## 0.4.29 - 2026-07-08

### Truncated vs unparseable scoring

- Dataset accuracy profiles (`gsm8k`, `mmlu-pro`, `gpqa-diamond`) now distinguish a **TRUNCATED** result — the model hit the `max_tokens` limit while still reasoning and never emitted an answer — from a genuine **unparseable (format)** miss where the model answered but the value could not be read. Both still count as wrong (accuracy is unchanged), but they are no longer conflated under the misleading "unparseable" label.
- The report shows a dedicated `truncated (no answer)` count, breaks `hit max_tokens` into "produced no answer" vs "answered before the cap", adds a `⊘` glyph and `TRUNC` label (quality bar, score line, and Failed Answers table), and explains the distinction in the interpretation note. A high truncated count is now an unambiguous signal to raise `--max-tokens`.
- The paired A/B comparison reports `truncated (no answer)` per side (a damaged quant that thinks longer truncates more, so this is itself a degradation signal) and renames the format-only count to `unparseable (format)`.
- Note: GSM8K truncations usually still surface as a wrong number rather than TRUNCATED, because truncated math reasoning almost always contains digits the scorer will read; the TRUNCATED category mainly benefits the multiple-choice profiles.

## 0.4.28 - 2026-07-08

### Dataset profile defaults

- Raised the default `max_tokens` of the dataset accuracy profiles (`gsm8k`, `mmlu-pro`, `gpqa-diamond`) from 65536 to 131072. Note that engines such as vLLM reject requests whose prompt plus `max_tokens` exceed the server `max_model_len`; on servers configured at or below 128k context, pass an explicit smaller `--max-tokens`.

## 0.4.27 - 2026-07-08

### Dataset profile defaults

- Raised the default `max_tokens` of the dataset accuracy profiles (`gsm8k`, `mmlu-pro`, `gpqa-diamond`) from 32768 to 65536, so heavy thinking-mode models never truncate on a healthy baseline and candidate `max_tokens` hits keep reading as degradation rather than budget artifacts. Override with `--max-tokens` as before; the value used is recorded in result metadata and checked implicitly by paired comparisons.

## 0.4.26 - 2026-07-06

### GPQA Diamond profile

- Added `--test-profile gpqa-diamond` (aliases `gpqa`, `gpqa_diamond`): all 198 graduate-level GPQA Diamond science questions (CC BY 4.0) as a frontier-difficulty accuracy anchor, scored by exact option-letter match with per-category (biology/chemistry/physics) accuracy and the same paired `--compare-baseline` support as gsm8k/mmlu-pro.
- Options are assigned to letters by a deterministic per-item shuffle seeded by the GPQA record id, so letter assignments and item ids are identical on every machine and across runs.
- Respecting the GPQA anti-contamination terms, the dataset is never stored in this repository: the official password-protected `dataset.zip` is downloaded from `idavidrein/gpqa` on first use, the archive and the derived canonical JSONL are both sha256-pinned, and the JSONL is cached under `~/.cache/llm_decode_bench/datasets/` only. A `.gitignore` entry guards against committing a local copy.
- Defaults match the other dataset profiles: temperature 0, `max_tokens` 32768, fixed concurrency 30, all 198 items (deterministic evenly-spread subset via `--profile-runs N`).

## 0.4.25 - 2026-07-06

### Dataset accuracy profiles and paired comparison

- Added `--test-profile gsm8k`: the full official GSM8K test split (1319 grade-school math problems, MIT license) as a multi-item accuracy benchmark. Every measured request is a different problem; scoring is exact final-number match with tolerant parsing (thousands separators, `$`/`%`, trailing punctuation).
- Added `--test-profile mmlu-pro`: a pinned deterministic stratified 1000-question subset of the TIGER-Lab/MMLU-Pro test split (Apache-2.0), shipped in `data/mmlu_pro_1000.jsonl` (largest-remainder allocation per category, floor-stride by `question_id`). Scoring is exact option-letter match with tolerant `Answer: <letter>` extraction; the report adds per-category accuracy.
- Datasets are sha256-pinned and resolved from `data/` next to the script, then `~/.cache/llm_decode_bench/datasets/`, then downloaded from the pinned source and verified; a hash mismatch is a hard error so runs on different machines always measure identical item sets.
- Dataset profiles default to temperature 0, `max_tokens` 32768 (common reasoning-model eval budget; a healthy baseline should essentially never truncate, so candidate `max_tokens` hits read as degradation, not artifacts), fixed concurrency 30, no prefix-cache scout, and all dataset items; `--profile-runs N` selects a deterministic evenly-spread N-item subset that stays identical across runs.
- Completion-stats machinery now supports per-item requests: run records carry `item_id`, `category`, expected/parsed answers, and the report gains an `accuracy` block (correct/scored, accuracy, Wilson 95% interval, unparseable count) plus `category_summaries`.
- Added `--compare-baseline`: after a dataset-profile run, results are paired per item id against a previous results JSON and the report/console gain accuracy deltas with Wilson intervals, correct/wrong flip counts and flip item ids, a two-sided exact McNemar p-value, per-category deltas, completion-token inflation, and `max_tokens`-hit counts.
- Added standalone `--compare-baseline a.json --compare-candidate b.json` comparison mode that runs without contacting any server.
- These profiles are intended as quantization-degradation anchors (e.g. GLM NVFP4 `w4a16` vs `w4a4`): run the same profile against each endpoint with identical engine flags and compare paired; the README documents the noise-floor protocol (same-endpoint self-comparison first).

### Completion-token profiles

- Added `--test-profile estonia-long`, which reuses the built-in Estonia prompt with a generic high-reasoning-effort system message and wrapper. It is intended to test whether models avoid premature short unknown/wrong answers without task-specific chain or decoy hints, uses `max_completion_tokens`, and sends MiMo `thinking.enabled` as a request override.

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
