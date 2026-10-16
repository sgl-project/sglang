# DeepSeek-V4.1-Flash low-latency decode (GB300)

`--enable-dsv41-low-latency-decode` replaces a set of stock decode kernels of
DeepSeek-V4.1-Flash with kernels sized for 1 to 48 rows. It targets agentic
serving, where a server decodes a handful of requests at a time with DSpark
speculative decoding and users see the time per output token.

With the switch off, nothing in this package runs and the model takes the stock
paths.

## Validated deployment

The switch refuses every other deployment at startup, listing each requirement
it misses (`switch.check_deployment`):

- DeepSeek-V4.1, one SM103 node (GB300 / B300), `--tp-size 4`;
- `--ep-size 1` and `--moe-a2a-backend none` (the routed-expert GEMMs serve
  every expert locally);
- no pipeline, DP-attention or context parallelism, no two-batch overlap;
- `--speculative-algorithm DSPARK`, the default DeepSeek-V4 attention backend,
  decode CUDA graphs;
- no encoder or decoder SWA bounded replay, PD disaggregation, LoRA or
  deterministic inference;
- no `--enable-fp32-lm-head` or `--rl-on-policy-target` (the low-M LM head
  returns BF16 logits).

## Paths

Each path replaces one stock call. It runs only inside its row bound
(`switch.py`) and otherwise declines, so the stock kernel runs. Rows are the
tokens of one forward: a DSpark verify step at batch `b` has `6 b` rows.

| Path | Replaces | Rows | Module |
| --- | --- | --- | --- |
| `wqkv_a`, `wq_b`, `wo_b` on FlashInfer's `cutedsl_low_latency` MXFP8 GEMM, tactics pinned per GB300 shape | the persistent CuTe DSL MXFP8 GEMM | 1-8 | `mxfp8_linear.py` |
| `wo_b`'s epilogue stores its rows into the Lamport mailbox; the attention boundary runs only the all-reduce + mHC consumer | `wo_b` + the fused all-reduce + mHC post | 1-8 | `wo_b_publish.py`, `lamport/` |
| Routed experts: one route kernel + two `tcgen05` GEMMs (8-row tiles; an expert chosen by more rows gets several tiles), NVRTC-compiled | FlashInfer TRT-LLM-gen routing + batched GEMMs | 1-48 | `routed_expert_gemm.py`, `routed_expert_gemm_src/` |
| Compressor `wkv` / `wkv_gate`, indexer `weights_proj` and `wk` on split-K low-M BF16 GEMMs | cuBLAS and the JIT small GEMMs | 1-8 | `indexer.py`, `low_m_gemm.py`, `ll_bf16/` |
| Indexer query (`weights_proj`, `wq_b`, RoPE + FP4 pack) on a side stream beside `wqkv_a` and the compressor | the same work on the sources stream | low-ratio decode | `indexer.py` |
| LM head (shared with the DSpark draft) on FlashInfer's CuTe DSL low-M BF16 GEMM | cuBLAS split-K + reduce | 1-16 | `lm_head.py` |
| Gluon swap-AB sparse decode attention up to 24 rows, launched with PDL, one-warp combine | FlashMLA sparse decode on 64 zero-padded heads past 8 rows | 6-24 | `switch.SWAPAB_MAX_ROWS` |

## DSpark draft and verify

The same switch moves the DSpark draft and the verify accept onto low-latency
paths (`dspark/`). They take effect only when the target resolved the switch;
there is no second flag.

| Path | Replaces | Rows | Module |
| --- | --- | --- | --- |
| Draft attention `wq_b`, `wo_b` on the low-latency MXFP8 GEMM (same shapes and pinned tactics as the target) | the persistent CuTe DSL MXFP8 GEMM | 1-8 | `dspark/install.py`, `mxfp8_linear.py` |
| Draft MoE on the routed-expert GEMM pair plus a local finalize (`finalize.cu`, FlashInfer's small-batch finalize arithmetic), since draft stages do not defer the finalize | FlashInfer TRT-LLM-gen routed MoE with `do_finalize=True` | 1-48 | `dspark/install.py`, `routed_expert_gemm.py` |
| Whole Markov chain of the folded draft proposal in one vocab-sharded cooperative kernel per rank; ranks exchange one key per row and step through symmetric memory | one Markov step kernel + vocab gather per draft step | 1-8 requests | `dspark/sharded_markov.py`, `.cu` |
| Draft context-KV projection on a side stream beside the accept, inside the verify CUDA graph | the same projection after the accept | static verify graphs | `dspark/static_epilogue.py` |
| Engram history commit recorded in the verify CUDA graph, gated per row | the eager commit after the replay | static verify graphs | `dspark/static_epilogue.py` |
| Temperature-sampling accept (rejection, bonus, accept-site sync, Engram and KV commits) in one CUDA graph per verify batch size, compiled at startup | the eager accept after the replay | all-sampled static steps | `dspark/sampling_accept_graph.py` |
| Block-wise rejection sampler: per-vocab-block stats, a one-warp accept, a block-parallel Gumbel-max resample, counter-based draws | draft and target softmax + single-CTA rejection kernel | no top-k / top-p, no block verification | `dspark/blockwise_rejection.py` |

The block-wise sampler's draws are functions of (server seed, a device step
counter, row, token), so every rank computes the same accept from the same
logits. Its accept syncs at the `dspark-accept-greedy` site instead of
`dspark-accept-sample`: with `SGLANG_SPEC_TP_SYNC=rng` it broadcasts nothing,
with the default `all` it still broadcasts. The sampling-accept graph serves
all-sampled steps only; mixed greedy / sampled, top-k, top-p, grammar and
logit-adjusted steps take the eager accept. A cross-rank wait
in the sharded Markov kernel traps after 20 s (`sharded_markov.TIMEOUT_MS`)
instead of hanging. Every per-step DSpark gate reads batch flags, shapes and
which graphs were captured, all shared by the TP ranks; the sharded chain is
built on every rank or on none (CPU all-reduce at startup).

## Rank lockstep

The fused all-reduce and the mailbox are collectives, so every rank must take the
same path at every step. Each gate reads only host state that all TP ranks share:
row counts, the graph mode and load-time facts. Nothing is read back from the GPU.
A publish whose rows were not reduced by the fused post raises instead of
desynchronizing the mailbox.

## Startup

The MXFP8 projections are tuned at 1, 2, 4 and 8 rows during FlashInfer's
startup autotune, with the pinned tactic as the only choice for 5-8 rows; the
autotune cache key records the switch. The routed-expert GEMMs are compiled with
NVRTC on first use and cached under `SGLANG_CACHE_DIR/dsv41_routed_expert_gemm`.
The mailbox and its publishing GEMM are built on the first eager forward, never
during graph capture.

## Provenance

The routed-expert GEMMs, the wo_b publish, the Lamport all-reduce + mHC changes and the low-M projections were developed at NVIDIA for this model. Files derived from vLLM (vllm-project/vllm@75e3b17c) or FlashInfer keep those projects' SPDX and copyright headers.
