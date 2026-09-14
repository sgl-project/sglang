# MiniMax-M3 on MI355X: PR-ready branches (kevin-mii/sglang)

All branches are pushed to `github.com/kevin-mii/sglang`. Each worktree under `/scratch/pr/` holds an untracked `PR_DESCRIPTION.md` (title, motivation, per-commit summary, measurements, tests, scope, risks, footer) ready to paste into the PR. Independent review notes are under `/scratch/pr/reviews/` (one file per branch, `SUMMARY_A.md`, `SUMMARY_B.md`); every REQUEST CHANGES / BLOCKED finding was fixed in a second round and amended into the owning commits, and a third pass applied the radixark pr-cleanup skill (one-line comments, docstring contracts, naming, no module state, tests pruned to those a plausible mistake fails) as separate cleanup commits on each branch. No PRs have been opened.

Upstream base: `origin/main` 39e147443b. Open upstream PRs we stack on (all by zcnrex): 36546 (Gluon sparse prefill), 36549 (fp8 index-K cache), 36559 (small-batch MoE sort), 36560 (wave64 decode top-k), 36574 (MXFP8 dense block convert), 36575 (fused add-RMSNorm fp8 quant), 36576 (ROCm shared-experts fusion). 36527 and 36557 are merged. **Plain main cannot serve MiniMax-M3 MXFP4 correctly without 36549/36559/36560/36574/36575/36576 plus the fixes below**; the integration branch `integration/m3-all` (55141230a7) = main + those seven PR heads + all 16 branches, and is what the end-to-end validation ran on.

## Follow-ups on open upstream PRs (merge into that PR's branch, or land right after it)

| Branch | Base PR | Head | Commits | What |
|---|---|---|---|---|
| `pr/36576-followup-shared-expert-slot` | 36576 | 45dcf57c25 | 2 | Fused shared expert was emitted twice and one routed expert dropped on aiter paths (rows `[3 routed, 128, 128]`; GSM8K 0.81 -> 0.896, MXFP4 output garbage fixed); Triton gate fills the slot itself, removing the append launch. **Required before 36576 lands.** |
| `pr/36546-followup-gluon-fp8-and-scratch` | 36546 | ae8f5a7921 | 4 | Gluon prefill accepts fp8 K/V pools (the PR silently fell back to Triton on fp8 KV); page layout built on device (no pageable H2D sync); dtype-aware configurable scratch cap; score-only index kernel split over KV blocks (2.3 -> 0.08 ms/layer for small extends at 198K). Registered GPU parity tests included. Note for the base PR: the Triton sparse prefill deviates from a torch oracle on synthetic top-k lists (up to 0.77, NaN at block_size_q 4). |
| `pr/36574-followup-mxfp8-aiter-backend` | 36574 | 4175b1ec25 | 2 | `--moe-runner-backend aiter` no longer overridden under the gfx95 mxfp8 override (FlyDSL fused MoE; 257K prefill 18.5 -> 14.7 s); quant kernel emits row-aligned scaled_mm operands (-4 launches per dense GEMM). |
| `pr/36575-followup-fp8-linear-consumes-fused-quant` | 36575 | 9e94a280a9 | 1 | Per-token fp8 linear consumes the fused add-RMSNorm's pre-quantized activation (version-guarded); `SGLANG_FUSED_NORM_FP8_QUANT_MAX_M`. |

## Standalone branches on `origin/main`

| Branch | Head | Commits | What | Review |
|---|---|---|---|---|
| `pr/quark-packed-shard-exclude-fix` | 55639a09c7 | 1 | quark `should_ignore_layer` ignores packed shards absent from the exclude list (index_v_proj not in the checkpoint). Small, land first. | APPROVE |
| `pr/m3-eagle3-chain-verify` | a69b0d262c | 3 | EAGLE chain target-verify for MiniMax-M3 sparse attention on CUDA/ROCm (was NPU-only); packed per-request index scoring (with the local-block fix found in review); small extends over long cached prefixes on the decode kernels. 1 stream 57 -> 84 tok/s at 70K. | fixed after REQUEST CHANGES |
| `pr/triton-verify-shared-kv-long-context` | 744bd6e7b3 | 4 | Split-KV / grouped-head verify kernels for EAGLE draft extend and M3 dense layers; head_dim-128 long-context config (24x195K: 2.02 -> 0.38 ms/call); grouped-head decode via the shared-KV kernel (head_dim 128 only); small constant-length extends routed to the verify kernels (twin of the sparse-backend half in `pr/m3-eagle3-chain-verify`). | fixed after REQUEST CHANGES |
| `pr/triton-extend-long-prefix` | 2b8206eb88 | 3 | Split-prefix extend attention for long cached prefixes (`SGLANG_ENABLE_TRITON_EXTEND_LONG_PREFIX`, renamed from the benchmark-branch name); aiter CK paged batch-prefill for extends >= 2048 rows (`SGLANG_USE_AITER_EXTEND_LONG_PREFIX`); AMD hook keeps an explicit `--triton-attention-num-kv-splits`. 197K fresh prefill 6.2 -> 5.8 s. | fixed after minor changes |
| `pr/spec-page-table-parallel-copies` | 77770da139 | 1 | Token-block-parallel page-table copies in the Triton draft backend and EAGLE metadata (419 -> 30 us and 101 -> 13 us per call at 24x200K). | APPROVE |
| `pr/m3-fused-kv-index-store` | f9710a5e99 | 1 | Fused scale+cast+scatter store for fp8 main and index KV caches (7-8 launches per layer -> 1). | fixed after REQUEST CHANGES |
| `pr/m3-sparse-attention-eager-break-bcg` | 0dcf77d879 | 1 | M3 sparse attention runs as an eager break under `--cuda-graph-backend-prefill breakable` (was captured: garbage + faults on replay). Extend forward at 185K: 75 -> 29 ms (20 tok), 118 -> 38 ms (409 tok). | APPROVE |
| `pr/scheduler-chunked-prefill-fairness` | 544fafd835 | 1 | `SGLANG_CHUNKED_PREFILL_FAIRNESS_RESERVE`: waiting short extends ride along with an in-flight chunked prefill; TTFT p99 -19% at c=24, throughput neutral. Scheduler hunk is one delegate call. | APPROVE |
| `pr/quark-online-fp8-excluded-layers` | c1acbbadf1 | 1 | Opt-in load-time per-channel FP8 for quark-excluded bf16 linear layers (ATOM's PTPC-FP8 attention/dense); GSM8K-1000 0.863 vs 0.854 with tuned aiter rows (rows are an aiter-side contribution, referenced). | APPROVE |
| `pr/router-gemv-128-rows` | 7dd47ba0b2 | 1 | `router_gemv` covers 65-128-row batches (EAGLE verify shapes); base PR 36557 is merged. | APPROVE |
| `pr/openai-chat-log-rejected-stream` | f84fdb7826 | 1 | Log the validation error (with request id) when a streaming chat request is rejected before its first chunk. | APPROVE |

## Suggested landing order

1. `pr/quark-packed-shard-exclude-fix` (unblocks loading the MXFP4 checkpoint on main), `pr/openai-chat-log-rejected-stream`, `pr/router-gemv-128-rows`.
2. Upstream 36576 with its follow-up merged in (the follow-up fixes a correctness bug in the PR); 36574, 36575, 36546 each with their follow-up right after.
3. `pr/m3-sparse-attention-eager-break-bcg`, `pr/m3-fused-kv-index-store`, `pr/spec-page-table-parallel-copies`.
4. `pr/triton-verify-shared-kv-long-context`, then `pr/m3-eagle3-chain-verify` (independent, but the small-extend speedup needs both), then `pr/triton-extend-long-prefix`.
5. `pr/scheduler-chunked-prefill-fairness`, `pr/quark-online-fp8-excluded-layers`.

Not upstreamed (stay on `M3-perf`): `benchmark/minimax_m3_mi355x/` launch and AIPerf scripts, tuned CSVs, `M3_MI350X_STATUS.md`.

## Integration validation

See the section appended below once the end-to-end run on `integration/m3-all` completes.

## Integration validation (done)

`integration/m3-all` @ 55141230a7 = origin/main 39e147443b + upstream PR heads 36546, 36549, 36559, 36560, 36574, 36575, 36576 + all 16 branches (3 additive merge conflicts, recorded in `/scratch/results/integration_notes.md`). Recommended real-acceptance config, TP4 on MI350X:

| Check | Result | Reference (M3-perf) |
|---|---|---|
| Needle 20K / 80K / 200K | coherent | coherent |
| Cached-restart parity (3 prompts) | 2/3 identical, 1 late divergence | known MoE-batch-shape class |
| GSM8K-500, EAGLE3 on | 0.874 | 0.85-0.89 |
| Steady decode N=16 / 24 at 195K | 1,557 / 1,961 tok/s | 1,611-1,697 / 1,949-2,124 |
| 600 s closed loop N=24 | 1,899 tok/s, 1,129 completions, no faults | stable |
| Fresh 197K prefill | 5.5-5.6 s | 5.6-5.9 s |

All engaged kernel paths match the benchmarked server marker for marker. No regression attributable to any branch.

**Load-bearing dependency found during integration:** plain main (even with 36546/36574/36575/36576 and our quark fix) produces garbage for MiniMax-M3 MXFP4 on gfx950; adding **PR 36559** (MoE small-batch sorting with fused mxfp8 quant, which carries the gfx950 aiter small-sort / SwiGLU path) alone restores correct output. Every M3 MXFP4 branch therefore depends on 36559 landing (or its aiter-side fix); the PR descriptions state this. Also: main before #37254 cannot load the MXFP4 checkpoint at all without `pr/quark-packed-shard-exclude-fix`.

Env rename to remember: the PR branches use `SGLANG_ENABLE_TRITON_EXTEND_LONG_PREFIX`; `M3-perf` and its benchmark configs used `SGLANG_TRITON_EXTEND_LONG_PREFIX` (the configs now set both).
