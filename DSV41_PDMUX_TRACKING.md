# DeepSeek-V4.1-Flash PDMux rollout

Track [#42515](https://github.com/sgl-project/sglang/pull/42515) and the four implementation layers split from [#41562](https://github.com/sgl-project/sglang/pull/41562). The common DP/PDMux scheduling path follows [pdmux-standard layer_split reference](https://github.com/Li-brua/sglang/tree/17761860ccbc6cea53b0c3625fa0c89880b48f9e). The tracker is a separate documentation branch and adds no commit to the implementation stack.

## Review stack

| Order | PR | Scope | Branch | Head | Review status |
| --- | --- | --- | --- | --- | --- |
| 1 | [#42507](https://github.com/sgl-project/sglang/pull/42507) | Layerwise prefill and decode overlap | `feat/dsv41-flash-pdmux-core` | `8cbc5df3fc` | Ready for review |
| 2 | [#42509](https://github.com/sgl-project/sglang/pull/42509) | HiCache progress and SWA ownership | `fix/dsv41-flash-pdmux-hicache` | `9ec3a170d8` | Draft |
| 3 | [#42513](https://github.com/sgl-project/sglang/pull/42513) | Attention DP and IDLE participation | `feat/dsv41-flash-pdmux-dp` | `1b7accf6a1` | Draft |
| 4 | [#42514](https://github.com/sgl-project/sglang/pull/42514) | Single-layer NextN/EAGLE and DSpark | `feat/dsv41-flash-pdmux-spec` | `fee732a7c9` | Draft |

Merge order: **#42507 → #42509 → #42513 → #42514**. Every child targets upstream `main` and contains the cumulative prefix until predecessors merge. Each child body links its focused incremental comparison. Rebase downstream heads onto `main` and drop merged dependency commits after each merge. Shared runtime changes overlap the [GLM PDMux stack #41861](https://github.com/sgl-project/sglang/pull/41861); deduplicate common code while retaining model-specific adaptations.

## Integration branch

Use `Li-brua:feat/dsv41-pdmux` for combined validation.

- Base: `5b5d72123935dc2a3fb341790796ec1a987d31ce`.
- Integration/spec HEAD: `fee732a7c9bbcf57dac315b939196583ac13e109`.
- Exactly four feature commits above the base, one for each child PR. The child refs point to those four successive prefixes. Fixes and formatting are folded into their owning commits.
- Original refs and the validated pre-rewrite tree are archived locally. #41562 remains a draft integration reference; merge through the child sequence.

```bash
git fetch https://github.com/Li-brua/sglang.git feat/dsv41-pdmux
git switch --detach FETCH_HEAD
git rev-list --count 5b5d72123935dc2a3fb341790796ec1a987d31ce..HEAD
# Expected: 4
git log --reverse --oneline 5b5d72123935dc2a3fb341790796ec1a987d31ce..HEAD
```

## Behavior and validation boundaries

PDMux uses layerwise prefill directly. Token chunking and layer slicing can be combined; intermediate slices preserve batch-owned mHC, Engram, tail and auxiliary state. With decode present, split budgeting uses the maximum prefill token count across DP ranks and has no extra layer-count cap. With no decode work, ranks execute the remaining layers together. Compressor-plan admission retains its token cap because layer slicing does not reduce a plan's token count.

Stream selection uses the local running decode batch before decode metadata gather. Prefill overrides only the full-TP handle; attention/MoE groups and sampler request-replica groups retain the ordinary handles. The scheduler preserves a global token vector for split boundaries and honors the ordinary skip-gather environment setting. Peer-only work still forms IDLE participants. Each target slice prepares/pads and unpads independently, without a raw-count snapshot or preparation reuse.

The core/cache prefixes admit plain TP. The DP layer adds non-speculative TP/attention DP, including TP8/DP8 and TP8/DP2. The spec layer adds single-NextN MTP/EAGLE and an explicitly adapted DSpark target. Final draft extension or injection follows the reference flow without an extra bidirectional handoff fence. EAGLE IDLE draft extension and final DSpark IDLE results without logits retain their respective completion paths. The optional planner stream follows `SGLANG_ENABLE_OVERLAP_PLAN_STREAM`.

DeepSeek-specific metadata/backend isolation remains: decode/IDLE replanning must not invalidate an unfinished split prefill's attention metadata. EAGLE drafts run eager; target and DSpark retain per-stream graph paths. Model helper-stream/workspace safeguards, HiCache event progress and SWA page ownership remain supported.

Multi-layer EAGLE, EAGLE3, adaptive speculative parameters and other unsupported algorithms remain rejected. Existing bounded replay/tail restrictions on MTP FULL capture and DP remain enforced. DSpark attention DP requires the ordinary DP LM head configuration. EP, CP and DCP are outside the validated matrix; TP8/DP8/EP8 with DSPARK and HiCache still requires a GPU pressure-test rerun.

## CPU validation

| Scope | Test files/selectors | Result |
| --- | --- | --- |
| Core prefix | 30 | 248 passed, 2 skipped, 155 subtests passed |
| Core/cache prefix | 32 | 262 passed, 2 skipped, 159 subtests passed |
| DP prefix | 35 | 278 passed, 2 skipped, 169 subtests passed |
| Complete implementation | 37 | 316 passed, 179 subtests passed, 2 skipped |

These suites overlap; counts are not additive. Coverage includes ordinary/split model parity, batch lifetime, planner admission, HiCache/SWA ownership, active/IDLE split boundaries, per-slice padding, group context and final-only NextN/DSpark processing. Syntax, Ruff lint/format, isort, spelling, registered-test and whitespace checks passed. The final implementation tree is identical before and after the history rewrite.

CPU tensors and runtime imports are real; CUDA kernels, streams and collectives are mocked. GPU collective liveness, output parity, speculative acceptance and performance remain unverified. The reported pressure-test hang has not been verified resolved. Use the [GPU runbook](https://github.com/Li-brua/sglang/blob/feat/dsv41-pdmux/test/manual/pdmux/layerwise_prefill_runbook.md) with identical model/workload settings against the ordinary scheduler.

- [ ] Plain TP1/TP8 greedy output and concurrent long prefill/decode.
- [ ] HiCache reload, eviction and repeated FULL/SWA ownership; include write-back, storage and buffer modes.
- [ ] TP8/DP8 and TP8/DP2, uneven and peer-only work, all-IDLE collective liveness.
- [ ] Single-NextN MTP and DSpark output/acceptance, final-only extension/injection and backend/graph selection.
- [ ] Token chunks, parked chunks, aborts and HiCache on/off in both SM layouts.
- [ ] Per-slice preparation and sustained decode output parity; rerun the reported TP8/DP8/EP8 + DSPARK + HiCache pressure test.
- [ ] TTFT, ITL p50/p99, output tokens/s and per-rank peak memory against ordinary-scheduler baselines.
