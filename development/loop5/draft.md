# Loop 5 Draft — Double Sparsity MVP on H200

## Objective

Get a **demonstrable Double Sparsity (DS) MVP** running end-to-end on
the 2-node H200 cluster as fast as possible, without confusing a
hardware smoke milestone with the loop4-complete MVP.

There are two deliverables:

1. **Smoke MVP:** DS-on DeepSeek-V3.2 (FP8) serves real requests on
   H200, produces non-trivial DS selection, has one DS benchmark JSON
   + one DSA benchmark JSON, and passes the paired quality smoke.
2. **Loop4-compatible MVP:** the smoke milestone plus the loop4
   requirements needed to claim comparable default-cookbook behavior:
   TP=8, FP8 KV, page size 64, CUDA graphs represented, chunked
   prefill probed, radix cache enabled for the final run, DSA baseline
   captured with matching knobs, AC-11 comparator run, and AC-12 full
   quality gate run.

If AC-10 radix, AC-11 comparator, or AC-12 full quality are missing,
the result is a useful smoke milestone, not the minimal viable working
version requested by loop4.

## Why a new loop

Loop 4 built deep code-tier scaffolding (comparator validation
gauntlet, bench_serving timing path, M3-B fixture infrastructure,
AC-12 harness, validator helpers) but never executed against
hardware — even though `CLUSTER.md` advertised an 8× H200 local +
8× H200 remote setup the entire time. The CPU-only loop drifted
because the remaining ACs were all hardware-gated and I kept
adding fixture code instead of running the existing code.

The critical artifact `/models/dsv32-fp8-channel-mask.safetensors`
**does not exist on disk**. Generating it unblocks every DS-on AC.
That single missing file is the actual root blocker.

## Hardware (per `CLUSTER.md` + auto-memory)

- Node 0 (local): 8× H200, hostname `h200-10-220-51-16`. Verified
  via `nvidia-smi`: 8 GPUs × 143 GB free.
- Node 1 (remote): 8× H200, hostname `h200-10-220-51-5`. Access via
  `rx devbox run double-sparsity --rank 1 -- <cmd>`.
- DSv3.2 FP8 weights: `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`.
- Default ports: workers 30001, router 30000, prometheus 29000.
- Logs: node 0 `/sgl-workspace/sglang/development/logs/`;
  node 1 `/tmp/sgl_logs/`.

## MVP scope — IN

0. **Close the Round 38 AC-10 producer bug before claiming radix-on.**
   The current capture producer path is unreachable: `_write_token_labels`
   does not accept `forward_batch`, but the capture branch references it
   and then hides the failure. Fix this first:
   - update `_write_token_labels(..., forward_batch: Optional[ForwardBatch] = None)`;
   - pass the live `forward_batch` at the extend, decode, and TRT-LLM
     call sites;
   - keep token-label writes first and publish radix capture only when
     `forward_batch` is present and the mode is extend;
   - add the producer-side regression required by Round 38;
   - verify `/generate` exposes non-empty
     `meta_info["double_sparsity_radix_capture"]` when
     `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`.

1. **Generate the channel mask** (`task-ac4-hwrun`). Single GPU,
   `--tp 1`, ~15–30 min wall-clock. Unblocks every DS-on AC.
   Output: `/models/dsv32-fp8-channel-mask.safetensors`. Validate
   `shape=[L, H, 16]`, `dtype=fp8_e4m3`, `head_dim=128`,
   `page_size=64`, `label_dim=16`.

2. **DS boot smoke** (`task-ac1-hwtest`). Launch
   `serve_double_sparsity.sh` on local 8× H200 TP=8 with the new
   mask; issue one `/generate` request; confirm the server returns
   text and the token-label table populates from the production
   `_write_token_labels` hook (the env-gated capture log built in
   Round 36–38 is the easiest probe: set
   `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, send the request, read
   `meta_info["double_sparsity_radix_capture"]` for non-empty
   `per_token_slot_sha` and `per_layer_written_all_true=True`).

3. **DSA + DS benchmark pair** (`task-ac8-server` + `task-ac9-baseline`).
   - DSA baseline: boot `serve_native_nsa.sh`, run
     `development/benchmark_baseline.sh` with the locked Option B
     flags at conc 16 / 32 / 64. ~30 min total.
   - DS run: boot `serve_double_sparsity.sh`, run
     `development/benchmark.sh` with the same operating point.
     ~30 min total.
   - A radix-off DS run is allowed only as a smoke/debug run. The
     final loop4-compatible MVP run must close AC-10 and run DS and
     DSA with radix cache enabled.
   - A single trial is allowed only for the smoke milestone. The final
     comparable-performance run uses the AC-11 shape: conc 16 / 32 /
     64, 3 trials, 120s warmup, 600s measurement window, median
     comparison.

4. **Quality smoke** (`task-ac8-quality`). Boot both servers
   simultaneously on different ports; run
   `test/manual/test_dsv32_quality_smoke.py` to compare DS-on vs
   DSA outputs on 20 deterministic prompts. Gates: prefix-match
   ≥ 0.80, ROUGE-L ≥ 0.85, NIAH-mini 4/5. ~5 min.

That is the smoke MVP. Demonstrable: one DS benchmark JSON, one DSA
benchmark JSON, one quality smoke artifact, side-by-side. Enough
to say "DS works on V3.2 FP8 with comparable quality at the
locked Option B operating point."

The loop4-compatible MVP additionally requires radix-on final serving,
AC-11 comparator evidence, and AC-12 full quality evidence.

## Smoke-only items that are NOT enough for loop4 MVP

These are allowed to defer only for the smoke milestone. They are not
allowed to remain deferred when claiming the loop4-compatible MVP:

- **AC-10 radix-cache flip.** Radix-off is acceptable for first boot
  and bench smoke only. To claim default-cookbook comparable behavior,
  run the M3-B fixtures, prove producer capture works, flip the guard,
  remove `--disable-radix-cache` from the final DS launch, and run the
  final comparator with radix cache on.
- **AC-11 directional comparator.** Single-trial bench_serving runs
  size the gap but do not prove comparable speed/performance. The final
  result needs the 3-trial DSA + DS sweep at conc 16 / 32 / 64 with
  120s warmup, 600s measurement, medians, DS TPS within 5% of DSA, and
  DS P99 TTFT no worse than 1.10x DSA.
- **AC-6 CUDA-graph capture validation.** Eager-mode DS is useful for
  diagnosis, but the client asks for performant knobs. The final bundle
  must record whether CUDA graphs are enabled and must include a clear
  exception if capture cannot be used.
- **AC-1b chunked-prefill probe.** Run and record the probe. If it
  passes, keep the default chunked-prefill setting. If it fails, disable
  it on both DS and DSA for apples-to-apples evidence and file the
  follow-up.
- **AC-12 full NIAH 4K/16K/64K + MMLU 5-shot.** The 5-minute quality
  smoke gates whether to continue, but the full gate is required before
  declaring loop4 MVP complete.

## Critical path (concrete commands)

```bash
# 0. Sanity
nvidia-smi --query-gpu=index,name,memory.free --format=csv

# 0a. Before radix-on claims
# Patch the Round 38 AC-10 producer bug and verify:
#   SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 /generate returns
#   meta_info["double_sparsity_radix_capture"] with non-empty
#   per-token and per-layer evidence.

# 1. Channel mask (~15-30 min, single GPU)
mkdir -p /models
python -m sglang.srt.layers.attention.double_sparsity.calibrate \
    --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_e4m3 \
    --tp 1 \
    --output /models/dsv32-fp8-channel-mask.safetensors \
    --label-dim 16 \
    --page-size 64 \
    --num-samples 256 \
    --block-size 512 \
    --seed 42 \
    -v 2>&1 | tee /sgl-workspace/sglang/development/logs/calibrate_$(date +%Y%m%d-%H%M%S).log

# 2. Validate mask artifact
python -c "
from sglang.srt.layers.attention.double_sparsity.channel_mask import load_channel_mask
m = load_channel_mask('/models/dsv32-fp8-channel-mask.safetensors')
print(f'dtype={m.dtype} head_dim={m.head_dim} page_size={m.page_size} label_dim={m.label_dim}')
print(f'channel_selection.shape={tuple(m.channel_selection.shape)}')
print(f'content_sha256[:12]={m.content_sha256[:12]}')
"

# 3. DS boot smoke (~5 min, separate terminal)
SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 \
  bash development/serve_double_sparsity.sh &
# Wait for /health on :30000, then:
curl -s -X POST http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello from DS", "sampling_params": {"temperature": 0.0, "max_new_tokens": 32}}' \
  | python -c "import sys,json; r=json.load(sys.stdin); print(r['text'][:200]); print('capture:', bool(r.get('meta_info',{}).get('double_sparsity_radix_capture')))"

# 4. DSA baseline bench (~10-30 min)
MODE=native_nsa CONCURRENCIES="16 32 64" \
  bash development/benchmark_baseline.sh
# (After it finishes, kill the DSA server. Boot DS server.)

# 5. DS bench
MODE=double_sparsity CONCURRENCIES="16 32 64" \
  bash development/benchmark.sh

# 6. Two-column comparator (single trial, --baseline / --ds form)
python development/benchmark_compare.py \
  --baseline development/results/native_nsa_gsp_isl4096_osl512_c64_t1.jsonl \
  --ds       development/results/double_sparsity_gsp_isl4096_osl512_c64_t1.jsonl \
  --output development/results/mvp_compare.md

# 7. Quality smoke (both servers up simultaneously on different ports)
DS_BASE_URL=http://127.0.0.1:30000 \
DSA_BASE_URL=http://127.0.0.1:30001 \
  pytest test/manual/test_dsv32_quality_smoke.py -v

# 8. Final loop4-compatible comparator, after AC-10 radix flip
# Ensure the DS launcher no longer passes --disable-radix-cache.
TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 \
MODE=native_nsa CONCURRENCIES="16 32 64" \
  bash development/benchmark_baseline.sh

TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 \
MODE=double_sparsity CONCURRENCIES="16 32 64" \
  bash development/benchmark.sh

# 9. Full quality gate
DS_BASE_URL=http://127.0.0.1:30000 \
DSA_BASE_URL=http://127.0.0.1:30001 \
  pytest test/manual/test_double_sparsity_v32.py -v
```

## Acceptance evidence — what "MVP done" looks like

A single directory `/sgl-workspace/sglang/runs/<date>_dsv32_mvp/`
containing:

- `calibrate.log` + `dsv32-fp8-channel-mask.safetensors` validation
  output.
- `serve_*.log` for both DS and DSA boots (showing no crashes,
  validator accepted, all 8 GPUs visible).
- Branch and commit SHA.
- Full server args from `/get_server_info`.
- Knob evidence: TP value, `kv_cache_dtype=fp8_e4m3`, `page_size=64`,
  CUDA graph status, radix cache status, chunked-prefill setting,
  DS config path, mask content hash, and whether overlap scheduling /
  piecewise CUDA graph remain disabled under Option B.
- Six bench JSONLs: `native_nsa_*c{16,32,64}_t1.jsonl` and
  `double_sparsity_*c{16,32,64}_t1.jsonl` plus matching `.meta.json`
  sidecars.
- `mvp_compare.md` from `benchmark_compare.py` (single-trial
  AC-7/AC-8 report — TPS, TTFT, no-op detector).
- `dsv32_quality_smoke_*.json` with prefix-match / ROUGE-L /
  NIAH-mini numbers for the paired DS/DSA run.
- Final loop4-compatible evidence, when claiming MVP complete:
  - radix-on DS and DSA launch evidence;
  - AC-11 3-trial comparator artifacts and pass/fail summary;
  - AC-12 NIAH 4K/16K/64K + MMLU 5-shot artifacts and pass/fail
    summary.

The smoke narrative: "DS-on V3.2 FP8 serves at the locked Option B
operating point. Side-by-side with DSA at conc 16/32/64. Quality
smoke passes on 20 paired prompts. Here's the bench JSON and the
comparator report."

The loop4 MVP narrative adds: "The final run used matching production
knobs, including radix cache enabled; CUDA graph and chunked-prefill
status are recorded; AC-11 comparator and AC-12 quality gates are
complete."

## Risks + likely failure modes

1. **Calibrate OOMs at TP=1.** Mitigation: bump to TP=2 with
   `--tp 2 --gpus 0,1`. The calibrate module accepts both.
2. **DS server fails the validator's DEC-2 guard.** Expected:
   the launcher already passes `--disable-radix-cache`, so the
   guard accepts. If it fails on something else (mask hash, page
   size pairing), read the validator error verbatim.
3. **bench_serving crashes on DS selection.** This would mean
   the production `_write_token_labels` hook is buggy on hardware
   (despite passing CPU unit tests). Round 18–20 work claims it's
   wired through `ForwardContext`; the boot smoke (step 3) catches
   this before the bench.
4. **Quality smoke prefix-match ≪ 0.80.** Could indicate either
   (a) DS labels are bad → re-check calibrate output, or
   (b) prompts are too short / sensitive. Investigate by running
   ROUGE-L only on long-output prompts first.
5. **TPS gap > 5%.** Acceptable for the smoke milestone. Not
   acceptable for the loop4-compatible MVP unless the artifact is
   explicitly reported as an AC-11 failure requiring follow-up tuning.

## Loop-runner notes

- Single mainline objective per round: the *next concrete
  command from the critical path above*. No more multi-day fixture
  refactors.
- Each round produces ARTIFACTS in `runs/<date>_dsv32_mvp/`, not
  just code changes. If a round did not produce an artifact, it
  stalled.
- The existing loop-4 code stays as-is unless a specific bench
  failure mode requires patching it. The Round 38 AC-10 producer
  bug is the one known exception that must be patched before a
  radix-on or default-cookbook parity claim.
