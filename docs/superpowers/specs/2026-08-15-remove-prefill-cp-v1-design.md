# Remove prefill context parallel v1

Date: 2026-08-15

Status: Approved

## Context

Prefill context parallelism currently has two implementations. The legacy path is concentrated in `python/sglang/srt/layers/utils/cp_utils.py` and model-specific branches. The strategy-based path is concentrated in `python/sglang/srt/layers/cp/` and is gated by `SGLANG_ENABLE_CP_V2`.

The two paths duplicate input sharding, KV materialization, output gathering, server-argument handling, and DSA behavior. Compatibility fields and deprecated CLI aliases also translate old CP terminology into the canonical `--enable-prefill-cp` and `--cp-strategy` interface.

This change makes the strategy-based implementation the only prefill CP implementation and removes its `v2` naming.

## Goals

- Remove `SGLANG_ENABLE_CP_V2` and make the strategy-based CP path unconditional whenever canonical CP configuration is active.
- Delete the legacy CP implementation, compatibility arguments, model-specific CP branches, DSA v1 branches, and v1 tests.
- Make `python/sglang/srt/layers/cp/` the sole implementation boundary for prefill CP.
- Keep model implementations independent of prefill CP sharding and gathering.
- Remove `v2` from every CP-specific API, field, method, variable, test name, and diagnostic string.
- Preserve batch-level fallback to ordinary non-CP execution when a batch cannot use CP.

## Non-goals

- Do not rename model versions such as DeepSeekV2 or Qwen2.
- Do not rename unrelated identifiers such as `DRAFT_EXTEND_V2`.
- Do not change decode context parallelism (DCP).
- Do not redesign the zigzag or interleave algorithms.
- Do not retain a compatibility shim for removed CP v1 environment variables or CLI flags.

## Configuration and API surface

The canonical configuration remains:

- `--enable-prefill-cp`
- `--cp-strategy {zigzag,interleave}`
- `--attn-cp-size`
- Existing canonical CP options that are not v1 compatibility aliases

The following surfaces are removed:

- The `SGLANG_ENABLE_CP_V2` environment variable and documentation.
- Model-class allowlists used only to enable CP v2.
- Legacy server-argument fields such as `enable_prefill_context_parallel`, `enable_dsa_prefill_context_parallel`, `prefill_cp_mode`, and `dsa_prefill_cp_mode`.
- Deprecated CLI aliases including `--enable-prefill-context-parallel`, `--enable-dsa-prefill-context-parallel`, `--enable-nsa-prefill-context-parallel`, `--prefill-cp-mode`, `--dsa-prefill-cp-mode`, and `--nsa-prefill-cp-mode`.
- Compatibility mapping between `in-seq-split`/`round-robin-split` and `zigzag`/`interleave`.

CP-specific names are normalized in place. Examples include:

- `is_cp_v2_active` to `is_cp_active`
- `_execute_extend_cp_v2` to `_execute_extend_cp`
- `cp_v2_input_ids` to `cp_input_ids`
- `enable_cp_v2_bcg_capture` to `enable_cp_bcg_capture`
- `cp_round_robin_input_ids_v2` to an unversioned CP name

## Architecture

### Strategy ownership

`python/sglang/srt/layers/cp/` owns strategy selection, metadata, tensor sharding, attention dispatch, KV materialization, global-token-order restoration, and output gathering.

`python/sglang/srt/layers/utils/cp_utils.py` is deleted. Legacy helpers are not copied into a new compatibility module. A capability that remains necessary must be expressed through the existing strategy interface or a backend-specific primitive.

### Runner and model boundary

The runner determines whether the current forward batch can use CP. For an active batch it builds CP metadata, shards embeddings, input IDs, positions, and speculative hidden state before calling the model body, then gathers hidden states before logits processing.

Files under `python/sglang/srt/models/` do not import prefill CP helpers or branch on CP state. Any CP work currently performed in a model file moves to one of these owners:

- The runner for model-boundary sharding and gathering.
- The selected CP strategy for layout-specific behavior.
- The attention backend for attention- or cache-specific behavior.

Model-specific restrictions that are configuration validation rather than tensor manipulation may remain outside modeling code.

### Attention backends and DSA

Attention backends keep only the strategy-based CP path, with unversioned names. Branches that select between v1 and v2 are collapsed to the strategy path. Non-CP behavior remains available for inactive batches.

DSA-specific indexer, cache, FP8 KV, and communication behavior remains in the DSA backend or CP strategy. Old round-robin helpers and code reachable only with CP v1 are removed. DCP logic remains separate and unchanged.

## Runtime data flow

1. Server arguments validate the canonical CP configuration and initialize one `ContextParallelStrategy`.
2. The runner calls `is_cp_active(forward_batch)`. The predicate checks the forward mode, configured strategy, input availability, and strategy-specific eligibility. It does not read an environment flag.
3. The runner builds and pads strategy metadata for an eligible prefill batch.
4. The runner shards model inputs at the model boundary.
5. The attention backend delegates CP layout and communication operations to the strategy.
6. The runner gathers model outputs into global logical token order before logits processing.
7. Decode batches and ineligible prefill batches use the ordinary non-CP path. They never enter a legacy CP implementation.

## Failure behavior

- Enabling prefill CP without `--cp-strategy` fails during argument validation.
- Invalid topology, strategy, backend, or model combinations fail explicitly during startup or metadata initialization.
- No unsupported combination silently falls back to CP v1.
- A batch that is too short or otherwise ineligible for CP may use normal non-CP execution. This is a batch eligibility decision, not a version fallback.

## Tests and acceptance criteria

Legacy-only tests and cases are deleted, including `test/registered/cp/test_dsa_prefill_cp_legacy.py`. Existing CP tests are updated to remove `SGLANG_ENABLE_CP_V2` overrides and CP-specific `v2` names.

CPU tests cover canonical argument handling, strategy initialization, `is_cp_active`, runner boundary sharding/gathering, inactive-batch fallback, and explicit rejection of unsupported combinations.

Static acceptance checks require:

- No occurrence of `SGLANG_ENABLE_CP_V2`.
- No CP-specific `cp_v2`, `CP_V2`, or equivalent versioned API name.
- No prefill CP helper import or prefill CP branch under `python/sglang/srt/models/`.
- No legacy CP utility file or deprecated CP CLI alias.
- No remaining test that exercises or configures CP v1.

All surviving tests under `test/registered/cp` must pass. Validation uses devboxes matching their registered runner configurations:

| Test | Required environment |
| --- | --- |
| `test_cp_strategy_unit.py` | CPU |
| `test_deepseek_v3_cp_single_node.py` | 8x H200 |
| `test_dsa_prefill_cp.py` | 8x H200 |
| `test_deepseek_v4_flash_fp4_b200_cp.py` | 4x B200 |
| `test_gpt_oss_4gpu_mxfp4_cp.py` | 4x B200 |
| `test_gqa_prefill_cp.py` | 4x H100 |

The deleted legacy test is excluded from this matrix. Local static, syntax, lint, and CPU checks supplement the devbox runs but do not replace the registered GPU tests.
