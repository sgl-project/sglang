# MiMo V2.5 Zigzag Context Parallel Design

## Context

Pull request #29972 currently mixes strategy-neutral MiMo V2.5 context-parallel support with a new implementation of the `interleave` CP strategy. The required production path is FA4 with `--cp-strategy zigzag` on four GPUs. The PR must therefore retain only the MiMo and zigzag work, validate that path on the allocated GB300 runner, and remove every new interleave-specific behavior from its final diff.

## Goals

- Serve Xiaomi MiMo V2.5 with FA4 and CP-v2 using `--cp-strategy zigzag` on four GPUs.
- Preserve the MiMo target and MTP adaptations required when attention TP collapses from the checkpoint's serialized TP=4 layout to runtime attention TP=1 under CP.
- Pass the registered 200-example GSM8K test on the exact pushed tree.
- Prove a zigzag CP plus EAGLE/MTP launch becomes ready, remains healthy, produces a correct response, and emits no traceback.
- Keep the final PR diff free of new interleave implementation and tests.

## Non-Goals

- Completing, testing, or publishing the interleave CP implementation.
- Opening a backup interleave PR.
- Modifying MiMo model or multimodal processor files.
- Refactoring unrelated CP, FA4, model-loading, or speculative-decoding code.

## Branch Strategy

Rebase the existing PR branch onto the latest `origin/main`, then add a narrow corrective commit. This preserves the existing PR and review discussion while making the final diff authoritative. Use `--force-with-lease` only because the requested rebase rewrites the PR commits; verify the remote head immediately before and after pushing.

Process-only design and plan documents are not part of the intended final PR diff.

## Production Changes

### Remove Interleave-Specific Work

- Restore `python/sglang/srt/layers/cp/interleave.py` to `origin/main` exactly.
- Restore `python/sglang/srt/layers/attention/flashattention_backend.py` to `origin/main` exactly. Its PR-only callback extension for per-query page-table row selection is used exclusively by the new interleave strategy.
- Remove the PR-added interleave mechanics tests and import from `test/registered/cp/test_cp_strategy_unit.py`. Retain pre-existing enum and strategy-initialization coverage from `main`.
- Replace interleave-specific wording in new MiMo CP diagnostics with strategy-neutral or zigzag wording.

The phrase "TP-interleaved QKV" in MiMo weight adaptation describes checkpoint serialization, not the CP strategy, and remains valid.

### Retain and Exercise Zigzag/MiMo Work

- Keep FA4 recognition in the CP attention-backend enum.
- Keep MiMo CP-v2 default registration, attention CP sizing, dense-MLP replication, and effective attention-TP validation.
- Keep zigzag metadata handling for uneven token counts, its minimum CP-v2 activation boundary, output reordering, KV materialization, and synchronized all-gather.
- Keep the MiMo fused-QKV block-FP8 loader adapter for target and `MiMoV2MTP` architectures.
- Keep eager-runner handling for MiMo embeddings, temporary speculative hidden-state sharding, tuple-shaped model output, PP-group ownership, and supported inner-model kwargs.
- Keep the prior reviewer-requested removal of asynchronous CP all-gather APIs.

## Runtime Data Flow

1. Server argument resolution enables CP-v2 for MiMo, derives attention CP size 4 from TP=4, and selects dense TP=1.
2. The loader converts the checkpoint's serialized TP=4 QKV groups into the canonical runtime Q/K/V layout before serving.
3. During an eligible extend batch, `ZigzagCPStrategy` builds per-sequence metadata and shards embeddings, positions, and MTP speculative hidden states consistently.
4. FA4 executes the existing two-part zigzag attention dispatch using the original four-argument callback and full request page tables.
5. Zigzag gathers and restores hidden states to original token order before MiMo logits processing; MTP state is restored after the inner forward, including on exceptions.

## Error Handling and Scope Guards

- Retain explicit validation for unsupported MiMo attention-TP and dense-TP layouts.
- Preserve shape checks around QKV repacking and CP gather operations.
- Ensure temporary MTP hidden-state sharding uses `try/finally` restoration.
- Fail the final scope audit if `interleave.py` or the FA4 backend differs from `origin/main`, if the registered MiMo test does not say `zigzag`, or if rejected model/multimodal paths reappear.

## Test Design

### Focused Tests

- Convert the generic speculative hidden-state sharding test to use real zigzag metadata and verify restoration on both success and exception paths.
- Retain zigzag shard, position, hidden-state gather, KV gather, prev/next attention dispatch, uneven padding, and CP-v2 activation-boundary coverage.
- Retain MiMo target/MTP QKV conversion and MTP embedding coverage.
- Add or retain assertions that FA4 maps to the CP flash-attention backend and both final and pre-normalization MiMo hidden tensors are gathered before logits.
- Run changed-file pre-commit checks.

### GB300 End-to-End Tests

On the four-GPU `baizhou-dev` runner, using the exact final source tree and local MiMo V2.5 checkpoint:

1. Run focused CP and MiMo/MTP unit tests.
2. Run `test/registered/cp/test_mimo_cp.py -v` after changing its server command to `--cp-strategy zigzag`; require score at or above 0.93.
3. Launch the cookbook-equivalent MiMo server with FA4, zigzag CP, and EAGLE/MTP; require server readiness, HTTP 200 health, a correct independent math response, and zero traceback entries.
4. Clean all SGLang processes and verify the four GPUs have no remaining compute processes.

## Completion Criteria

- The branch is based on the latest `origin/main`.
- The final PR diff contains no new interleave implementation, FA4 interleave callback, or interleave mechanics tests.
- Focused tests, pre-commit checks, zigzag GSM8K, and zigzag CP+MTP integration all pass on the exact pushed commit.
- PR #29972 title, description, and final review ping describe zigzag only.
