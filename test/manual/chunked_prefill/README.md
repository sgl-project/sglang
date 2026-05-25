# Chunked-Prefill Refactor Manual Test Suite

Per-feature accuracy fixtures for the chunked-prefill scheduler refactor. **All fixtures are manual** (not registered with CI) — they are run by hand during refactor development and as part of the manual safety net described in `agent-context/projects/sglang/2026-05-25-chunked-prefill-rewrite/agent-drafts/2026-05-25-testing-strategy-overview.md`.

## What this suite is

Each fixture launches a sglang server with one chunked-prefill-relevant feature flag and runs a mixed-prefix GSM8K eval. The mixed-prefix eval (`eval_name="gsm8k_mixed"`, see `python/sglang/test/simple_eval_gsm8k_mixed.py`) deterministically routes each question through one of four prefix modes — standard / cluster / random-sample / zero-shot — so a single 100-question run exercises radix hit, miss, branching, and short-prefill paths simultaneously instead of repeating one shared prefix.

The primary safety detector is the KV canary (when its PR lands). Score gating in this suite is intentionally loose (`>= 0.50`) — it catches catastrophic regressions (server hung, output garbage) without false alarms from mixed-prefix's lower score baseline.

## What this suite is NOT

- Not a CI gate. No `register_*_ci` calls anywhere. The presence of fixtures here is uncoupled from per-commit / nightly runs.
- Not a complete safety net. Resource leaks (`req` stuck in `waiting_queue` holding KV without ever reading/writing) won't be caught here — those need the fine-grained scripted scheduler tests and runtime invariants (see the strategy doc).
- Not a performance benchmark. Throughput is logged for reference but not gated.

## Fixtures

| Letter | File | Feature | GPU req |
|---|---|---|---|
| a | `test_feature_a_pp.py` | PP + dynamic chunking | 4 (TP=2 × PP=2) |
| b | `test_feature_b_disagg.py` | PD disaggregation | 2+ (prefill + decode) |
| c | `test_feature_c_hybrid_swa.py` | Hybrid SWA (gpt-oss-20b) | 1 large (>=40GB) |
| d | `test_feature_d_hisparse.py` | HiSparse (GLM-5-FP8) | 8 (H200 class) |
| e | `test_feature_e_spec.py` | EAGLE spec decoding | 1 large |
| f | `test_feature_f_radix.py` | Radix prefix match | 1 small |
| g | `test_feature_g_priority.py` | Priority scheduling | 1 small |
| h | `test_feature_h_piecewise_cuda_graph.py` | Piecewise CUDA graph (default-on) | 1 small |
| i | `test_feature_i_lora.py` | LoRA | 1 large |
| j | `test_feature_j_lora_overlap.py` | LoRA overlap loading | 1 large |
| k | `test_feature_k_dp_attention.py` | DP attention | 2 |

Pilot trio (a, e, i) was implemented first to validate the `ChunkedRefactorTestBase` shape before scaling out.

## Common knobs

All knobs live in `common.py`:

- `DEFAULT_CHUNKED_PREFILL_SIZE = 256` — small enough to force chunking on gsm8k 10-shot prompts (~1500-2000 tokens → 6-9 chunks)
- `DEFAULT_NUM_EXAMPLES = 100` — 25 per mode × 4 modes
- `DEFAULT_NUM_SHOTS = 10` — produces enough prompt length to chunk
- `LONG_PROMPT_NUM_SHOTS = 24` — used only by `c` (SWA) and `d` (HiSparse), where the feature is only meaningful above a structural prompt-length threshold
- `SCORE_THRESHOLD = 0.50` — single conservative floor; per-mode scores are logged for debugging but not gated
- `KV_CANARY_ARGS = []` — placeholder. Flip to e.g. `["--enable-kv-canary"]` once the canary PR lands. Every fixture appends it, so one edit covers the whole suite.

To override per fixture, set the corresponding `ClassVar` on the subclass.

## Running

Single fixture:

```bash
python -m unittest test.manual.chunked_prefill.test_feature_a_pp -v
```

All fixtures with the bundled runner (captures logs + json metrics):

```bash
bash test/manual/chunked_prefill/run_all.sh
bash test/manual/chunked_prefill/run_all.sh --only a,e,i      # pilot subset
bash test/manual/chunked_prefill/run_all.sh --skip d,b         # skip expensive ones
RESULTS_DIR=/tmp/my_run bash test/manual/chunked_prefill/run_all.sh
```

Results land in `test/manual/chunked_prefill/results/<fixture>.log` and `.json`.

## Adding a new fixture

1. Create `test_feature_<letter>_<name>.py`.
2. `from test.manual.chunked_prefill.common import ChunkedRefactorTestBase`.
3. Subclass; set `model` and `feature_args` (and optionally `num_shots`, `chunked_prefill_size`, ...).
4. Do **not** call `register_cuda_ci()` / `register_amd_ci()` / etc.
5. Add the letter to `ALL_LETTERS` and `LETTER_TO_MODULE` in `run_all.sh`.

For features with non-standard server setup (multi-engine like disagg), subclass the appropriate `*ServerBase` from `python/sglang/test/server_fixtures/` first, then mix in `ChunkedRefactorTestBase`. See `test_feature_b_disagg.py`.

## Related docs

- `agent-context/projects/sglang/2026-05-25-chunked-prefill-rewrite/agent-drafts/2026-05-25-testing-strategy-overview.md` — overall 5-layer strategy
- `…/2026-05-25-e2e-accuracy-workload-per-feature.md` — per-feature workload rationale
- `…/2026-05-25-mixed-prefix-gsm8k-design.md` — mixed-prefix eval design
- `…/2026-05-25-kv-canary-vs-deterministic-for-accuracy.md` — why canary primary
- `…/2026-05-25-chunked-testing-implementation-plan.md` — this suite's implementation plan
