# PD Flip Next Experiments Todo List

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PD flip observation time cover source quiescing, add a `1P3D -> 2P2D` reschedule/KV-stitching experiment, and rerun full-link latency observation with 40 mixed 1k/10k-character requests.

**Architecture:** Treat monitor observation, source drain/quiesce, migration, and final role switch as separate measurable stages. Move the uncertain source-idle wait into the fixed observation/quiesce window, then make post-migration idle only a short bounded assertion. For `1P3D -> 2P2D`, make reschedule explicit: prefill KV segment and decode KV segment must be measured as separate inputs to the target decode continuation path.

**Tech Stack:** SGLang scheduler/runtime-role state machine, PD flip controller, trace replay/generator scripts, migration measurement scripts, CSV/SVG raw artifacts.

## Global Constraints

- Keep raw artifacts for every experiment: trace JSONL/CSV, TTFT CSV, TPOT CSV, SLO CSV/JSON, controller log, migration samples, controller action CSV, full-chain SVG/CSV.
- Every controller stage must have elapsed time in raw output.
- Every migrated request must expose source queue/type: `running`, `waiting_queue`, `decode_prealloc`, `decode_transfer`, or `rescheduled`.
- Post-migration `wait_source_idle` must be bounded and reported separately from the fixed observation/quiesce window.
- Trace for the next experiment: 40 requests, 0.5s interval, mixed short/long prompts by character count, short = about 1k chars, long = about 10k chars.

---

## Todo 1: Cover `wait_source_idle` Inside The Observation Window

**Intent:** The monitor window should not merely decide whether to flip; it should also absorb source drain/quiesce time. After migration commit, `wait_source_idle` should become a short sanity check, not an unbounded variable stage.

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`
- Modify: `scripts/playground/disaggregation/pd_flip_migration_measure.py`
- Modify: `experiments/make_pd_state_machine_latency_diagram.py`
- Modify or add tests under `test/srt/` and/or `test/srt/test_pd_flip_controller*.py`

**Tasks:**

- [ ] Add a controller state before KV migration: `observing_source_quiesce`.
  - On D->P risk, select source decode and target decode early.
  - Immediately `router_drain_source`.
  - Pause source admission if the experiment mode requires strict no-new-work.
  - Start a fixed timer, e.g. `OBSERVE_QUIESCE_SECONDS`.

- [ ] During `observing_source_quiesce`, poll source queues and write these fields into raw samples:
  - `source_running_reqs`
  - `source_waiting_queue_reqs`
  - `source_decode_prealloc_queue_reqs`
  - `source_decode_transfer_queue_reqs`
  - `source_decode_retracted_queue_reqs`
  - `source_total_residual_reqs`
  - `source_quiesce_elapsed_s`

- [ ] At the end of the fixed observation/quiesce window, branch explicitly:
  - If SLO recovered, abort flip and undrain source.
  - If SLO still requires D->P, continue to migration.
  - If source residual requests remain but are not migratable, either reschedule them or mark them as explicit residuals in raw output.

- [ ] Change post-migration idle wait from open-ended waiting to bounded verification:
  - New env/config: `PD_FLIP_POST_MIGRATION_IDLE_TIMEOUT_SECONDS`, default small, e.g. `2`.
  - If source is not idle within this bound, fail the flip with raw residual queue counts.
  - The diagram should show this as `post-migration idle assertion`, not as the main waiting stage.

- [ ] Update the full-chain diagram labels:
  - Add `Observation + source quiesce`.
  - Move the former long `wait_source_idle` time into that box.
  - Keep `Post-migration idle assertion` as a separate bounded stage.

- [ ] Acceptance check:
  - Run a D->P full-link experiment where old `wait_source_idle` would be large.
  - Expected result: total flip still includes the same work, but the unbounded portion appears inside `Observation + source quiesce`.
  - Expected post-migration idle assertion: under configured timeout or explicit failure with residual counts.

## Todo 2: Run `1P3D -> 2P2D` With Explicit Reschedule And KV Stitching

**Intent:** Test the D->P direction from a cluster layout with one prefill and three decode nodes. When one decode becomes prefill, its active decode requests must be rescheduled to another decode target. The experiment should measure whether target decode gets the correct KV state from prefill and decode segments.

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `scripts/playground/disaggregation/pd_flip_migration_measure.py`
- Add: `experiments/pd_flip_1p3d_to_2p2d_runner.sh`
- Add tests: `test/srt/test_pd_flip_reschedule_kv_stitching.py`

**Tasks:**

- [ ] Define the `1P3D -> 2P2D` initial layout:
  - node0: prefill
  - node1: decode
  - node2: decode source to flip into prefill
  - node3: decode target for rescheduled requests

- [ ] Add a manifest schema for reschedule/stitching evidence:
  - `rid`
  - `source_decode_url`
  - `prefill_provider_url`
  - `target_decode_url`
  - `prefill_kv_len`
  - `decode_kv_len`
  - `kv_committed_len`
  - `pd_flip_source_queue`
  - `reschedule_reason`
  - `stitch_plan`: ordered segment list, e.g. `prefill_segment`, `decode_segment`

- [ ] Verify current behavior before changing it:
  - Check whether the source decode already owns the full committed KV range.
  - If yes, record this as `source_decode_full_kv_available=true` and compare against split-provider design.
  - If no, implement two-source stitching: prefill segment from prefill provider, decode segment from source decode.

- [ ] Implement target decode reconstruction:
  - Target decode receives prefill KV segment.
  - Target decode receives decode KV segment.
  - Target decode orders segments by token offset.
  - Target decode only enqueues resumed request after both segments are present.

- [ ] Add reschedule safety checks:
  - Token offsets must be contiguous.
  - `prefill_kv_len + decode_kv_len == kv_committed_len`.
  - Output continuation must use the same `rid`, sampling params, priority, and routing metadata.

- [ ] Measure and write separate timings:
  - prefill KV segment transfer time
  - decode KV segment transfer time
  - target stitching time
  - target held queue time
  - target resumed decode time
  - controller reschedule decision time

- [ ] Acceptance check:
  - Run `1P3D -> 2P2D`.
  - Confirm final roles are node0/node2 prefill and node1/node3 decode.
  - Confirm migrated/rescheduled requests continue on target decode without duplicated or missing output.
  - Produce raw package and full-chain SVG/CSV.

## Todo 3: Replace Trace With 40 Mixed 1k/10k Character Requests

**Intent:** The current trace is too short. Build a smaller but heavier trace that can expose long-request KV migration and decode continuation cost.

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_trace_replay.py`
- Modify or add: `test/srt/test_pd_flip_trace_replay.py`
- Modify experiment runners to accept trace sizing env vars.

**Tasks:**

- [ ] Add trace generator arguments:
  - `--num-requests 40`
  - `--interval-seconds 0.5`
  - `--short-chars 1000`
  - `--long-chars 10000`
  - `--long-count 20`
  - `--short-count 20`

- [ ] Generate prompts by character count, not only word count:
  - Short prompt target: 1,000 chars ± 2%.
  - Long prompt target: 10,000 chars ± 2%.
  - Keep request body deterministic by seed.

- [ ] Preserve per-request SLO fields:
  - `ttft_slo_s`
  - `tpot_slo_s`
  - `prompt_chars`
  - `prompt_kind`
  - `max_tokens`

- [ ] Add request-level raw outputs:
  - `ttft.csv`
  - `tpot.csv`
  - `tpot_tokens.csv`
  - `slo_attainment.csv`
  - `request_metrics.jsonl`
  - `trace_requests.jsonl`
  - `trace_requests.csv`

- [ ] Add migration/control raw outputs:
  - `controller_actions.csv`
  - `migration_status_samples.csv`
  - `router_worker_samples.csv`
  - `worker_pd_flip_samples.csv`
  - `pd_state_machine_full_chain_latency.csv`
  - `pd_state_machine_full_chain_latency.svg`

- [ ] Add latency breakdown fields:
  - control logic time: drain, admission pause, source/start, target/prepare, status polling, commit, source/finish, runtime role set, router role update
  - KV transfer time: source send, target receive, target held
  - decode continuation time after target commit
  - request-level TTFT/TPOT and timeout/error status

- [ ] Acceptance check:
  - Run the 40-request trace at 0.5s interval.
  - Verify the trace contains exactly 20 short and 20 long prompts.
  - Verify raw SLO files contain 40 rows, except per-token TPOT files which contain one row per generated-token interval.
  - Verify full-chain diagram includes controller logic, KV transfer, target held/commit, and post-commit decode continuation.

## Execution Order

- [ ] First implement Todo 1 so the ambiguous `wait_source_idle` stage becomes explainable and bounded.
- [ ] Then run a small D->P validation experiment to prove the observation/quiesce window absorbs the old idle wait.
- [ ] Then implement Todo 3 trace changes so all later experiments use the new 40-request 1k/10k trace.
- [ ] Then implement Todo 2 and run the `1P3D -> 2P2D` reschedule experiment with the new trace.
- [ ] Package every run as `.tar.gz` and keep the raw directory uncompressed locally.

## Expected Deliverables

- `pd_flip_observation_quiesce_<timestamp>.tar.gz`
- `pd_flip_1p3d_to_2p2d_<timestamp>.tar.gz`
- Full-chain SVG and CSV for each run
- Raw request trace, TTFT, TPOT, SLO, controller, and migration-link files
- A short summary markdown in each raw directory explaining:
  - role layout
  - request mix
  - migrated/rescheduled request counts
  - per-stage latency
  - any timeout/error requests
