# Round 20 Code Review

Mainline Progress Verdict: ADVANCED

Round 20 fixes the specific Round 19 AC-6 coding gap. I do not find a high-signal implementation defect in the new `ds_topk_indices_out` ForwardContext lookup. The original Loop 4 plan is still not complete: hardware, benchmark, and quality gates remain open and must not be treated as deferred completion.

## Implementation Review

Verified Round 20 claims:
- `python/sglang/srt/models/deepseek_v2.py:2136-2247` now resolves `_dsa_metadata` from the active `ForwardContext` and reuses it for both `ds_graph_state` and `ds_topk_indices_out`.
- The old synthetic `forward_batch.attn_backend.forward_metadata` runtime lookup is gone from `_select_topk_indices`.
- `python/sglang/srt/layers/attention/dsa_backend.py:1044-1077` allocates both `ds_topk_indices_out` and `ds_graph_state` into `DSAMetadata` for CUDA graph capture.
- `python/sglang/srt/model_executor/cuda_graph_runner.py:1020-1076` still creates a local `ForwardBatch` without DS fields and publishes the backend through `ForwardContext`, matching the new regression.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py:3469-3548` asserts `torch.empty_like` is not called and the returned tensor aliases metadata-owned `ds_topk_indices_out`.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py:3561-3645` removed the manual `forward_batch.ds_topk_indices_out` setup from the CUDA graph replay test.

Validation run:
```text
PYTHONPATH=python pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py -k 'uses_metadata_ds_topk_indices_out_via_forward_context or select_topk_indices_zero_allocs_production_path or select_topk_indices_reads_metadata_buffer_via_forward_context or no_bypass_when_forward_context_use_mha_false'
4 passed, 197 deselected

PYTHONPATH=python pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py
201 passed

PYTHONPATH=python pytest -q test/registered/integration/test_double_sparsity_tp_multiprocess.py
3 passed
```

## Goal Alignment Summary

```text
ACs: 9/15 addressed | Forgotten items: 0 | Unjustified deferrals: 0
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.
Partial: AC-1, AC-4, AC-6.
Not met: AC-1b, AC-8, AC-9, AC-10, AC-11, AC-12.

No original-plan task is missing from Active, Completed, or Deferred after tracker reconciliation. `Explicitly Deferred` is still empty, which is correct. Hardware-gated work remains active/pending, not accepted as deferred.

## Findings By Lane

### Mainline Gaps

1. The Loop 4 MVP is still incomplete because the remaining original-plan gates have not run.

AC-6 coding is now verified, but AC-6 itself still requires the real V3.2 conc=64 full-graph capture, 100 replay steps, and eager-vs-graph deterministic equality. AC-1 still needs the H200 forward population check. AC-4 still needs the generated and validated `/models/dsv32-fp8-channel-mask.safetensors`. AC-1b, AC-8, AC-12, and stretch AC-9 through AC-11 remain unrun.

Required execution plan:
1. Generate the V3.2 mask on the H200 cluster:
   ```bash
   python -m sglang.srt.layers.attention.double_sparsity.calibrate \
     --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 \
     --dtype bfloat16 \
     --kv-cache-dtype fp8_e4m3 \
     --output /models/dsv32-fp8-channel-mask.safetensors \
     --label-dim 16 \
     --page-size 64
   ```
2. Validate the mask with `load_channel_mask`, including shape `[L, H, 16]`, `max(channel_selection) < 128`, metadata dtype `fp8_e4m3`, and content hash.
3. Run `task-ac1-hwtest`: real H200 `forward_extend`, then assert each `token_label_table.signatures[layer_id, out_cache_loc]` row is non-zero.
4. Run `task-ac6-hwrun`: Option B V3.2 conc=64 full-graph capture, 100 replays, no CUDA launch failure, eager/graph `max_abs_diff <= 1e-6`.
5. Run `task-ac1b-probe`: `chunked_prefill_size=4096`, compare labels for tokens 0..4095 against non-chunked baseline, and record the result. Do not remove it from Active without evidence.
6. Run AC-8 server and quality smoke, then AC-12 full NIAH/MMLU. AC-12 is hard and the loop cannot close without it.
7. Complete stretch AC-9, AC-10, and AC-11 instead of treating them as complete-by-deferral.

2. AC-8 and AC-12 quality harnesses are not runnable in their planned form yet.

Evidence:
- `test/manual/test_dsv32_quality_smoke.py` does not exist, although the plan names it as the AC-8 lightweight quality smoke fixture.
- `test/manual/test_double_sparsity_v32.py:48-81` is still a skip-only scaffold; every NIAH/MMLU and negative sensitivity test calls `self.skipTest(...)`.

Required implementation plan:
- Create `test/manual/test_dsv32_quality_smoke.py` with the 20 deterministic prompts, same-session DSA reference generation, commit SHA recording, prefix-match >= 80%, mean ROUGE-L >= 0.85, NIAH-mini recall >= 4/5, and first-8-token divergence check.
- Replace the skip-only AC-12 scaffold with real NIAH 4K/16K/64K and MMLU 5-shot execution against paired DSA/DS servers. Fail the suite when DS deltas exceed 5 pp for NIAH or 1.0 pp for MMLU.

3. The AC-8/AC-9/AC-11 run tooling still needs Option B alignment before hardware evidence is valid.

Evidence:
- `development/serve_double_sparsity.sh:44-54` and `development/serve_native_nsa.sh:29-36` do not pass the locked `--dsa-prefill-backend flashmla_kv`, `--dsa-decode-backend flashmla_kv`, `--disable-overlap-schedule`, and `--disable-piecewise-cuda-graph` flags.
- `development/benchmark.sh:23` and `development/benchmark_baseline.sh:22` default to conc=64 only, while AC-8/AC-9 require conc 16/32/64.
- `development/benchmark_compare.py:252-286` still implements older SLO/no-op checks around selected pages, not the AC-11 3-trial median, DS TPS within 5%, and P99 TTFT <= 1.10x DSA gate.

Required implementation plan:
- Update both server launch scripts to encode the locked Option B flags. Keep `--disable-radix-cache` only on the DS AC-8 path until AC-10 passes; after AC-10, the AC-11 comparison must differ only by DS enablement/config.
- Make benchmark scripts run conc 16/32/64 by default for the gates and persist commit SHA, server args, chunked-prefill setting, seed, warmup, and measurement window in each result.
- Update `benchmark_compare.py` to consume three trials per concurrency, take medians, enforce DS TPS within 5% of DSA and P99 TTFT <= 1.10x DSA, and stop relying on page-named DS fields.

### Blocking Side Issues

None separate from the mainline gaps above. Round 20's specific AC-6 production-object issue is fixed.

### Queued Side Issues

- Fix `_publish_ds_request_summary` page-vs-token naming and sparsity denominator before AC-8 server/quality smoke.
- Align Option B launcher/comparator scripts before AC-8/AC-9/AC-11 hardware runs.
- Clean stale DS bind/runtime comments mentioning `req_to_token_pool.size`.
- Update token-label lifetime docs to describe invalidate-before-selection rather than overwrite-before-read.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 20 Review`.
- Added Round 20 Review rows for AC-6 verification and pending harness/tooling drift.
- Moved `task-ac6-cuda-graph` from Active to Completed and Verified.
- Updated active AC-8/AC-11/AC-12 task notes to reflect missing/runnable harness work.
- Added the Option B script/comparator alignment issue to Queued Side Issues.

NOT COMPLETE
