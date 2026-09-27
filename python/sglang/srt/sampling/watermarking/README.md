# Text watermarking design

This package implements keyed Aaronson-Gumbel sampling and model-free detection for SGLang. The [operator guide](../../../../../docs/docs/advanced_features/text_watermarking.mdx) defines the server and request APIs, the complete detector protocol, standalone detector use, compatibility, and limitations. This document describes the implementation and its validation.

## Sampling algorithm

SGLang applies penalties, logit bias, grammar masking, temperature, and top-k/top-p/min-p truncation before watermark selection. For the resulting distribution `p`, the selector computes a deterministic uniform value for every supported token:

```text
context_hash = MurmurHash3_x86_32(last h committed tokens)
H[v]         = mix32(key_lo, key_hi, context_hash, v)
u[v]         = (H[v] + 0.5) / 2^32
token        = argmax_v log(u[v]) / p[v]
```

This is the logarithmic form of `argmax_v u[v]^(1 / p[v])`. Under the ideal independent-uniform hash model it samples exactly from `p`; masked and truncated tokens have zero support and cannot win. Ties select the lowest token ID.

The upper end of the uint32 range needs special handling. In fp32, `(H + 0.5) / 2^32` rounds to `1.0` for the top 128 hashes, which would give `log(u) = 0` and let a very unlikely token win. The shared Triton helper and PyTorch reference use the direct expression below `2^31` and

```text
log1p(-(2^32 - H - 0.5) / 2^32)
```

above it. The integer complement is formed before conversion to preserve relative precision near one.

The current hash is a fixed MurmurHash3-derived 32-bit mix, not a cryptographic PRF. The key is sufficient to detect and forge the watermark; the scheme does not claim indistinguishability or unforgeability.

## Detection contract

For each generated token, the detector reconstructs its preceding `h`-token context, recomputes the selected token's `u`, and adds `-log(1 - u)` to the score. Under the null, the sum over `n` scored contexts follows `Gamma(n, 1)`, so the detector reports the one-sided upper-tail p-value, log p-value, and normal approximation `z = (S - n) / sqrt(n)`.

A compatible detector must:

1. Align the first completion token with the exact prompt token IDs when available; otherwise skip completion positions whose context is unknown.
2. Score each exact context tuple once. Generation uses the 32-bit context hash for admission, so a rare collision can add only an ordinary null term at detection time and conservatively dilute the signal.
3. Stop at the generation history budget, normally the first 4,096 eligible distinct contexts. A lower per-request token-pool capacity lowers that budget.

Greedy and entropy-gated positions are ordinary samples and therefore dilute, rather than create, evidence. Dual-key detection reproduces the deterministic per-context coin and scores each position with the selected key. The [detector specification](../../../../../docs/docs/advanced_features/text_watermarking.mdx#detection) contains the exact Murmur primitives, key parsing, prompt alignment, prefix rule, dual-key coin, and API examples. `detector.py` is the in-tree reference and can be copied with `config.py` for use without SGLang or PyTorch.

## Runtime design

- **GPU ring state.** Each `req_pool_idx` owns the last committed tokens needed for the context window. Prefill initializes it from the logical prompt tail, including chunked-prefill and radix-cache-hit cases. Normal decode appends the sampled token; speculative decode appends only accepted tokens.
- **Repeated-context masking.** A context hash is forced at most once per request. Repeated contexts use ordinary sampling, preventing deterministic short cycles from excluding EOS. Retraction rebuilds history from accepted output. The history is capped at `min(4,096, per-request token-pool capacity)`.
- **Greedy bypass.** Rows normalized to `top_k <= 1`, including `temperature=0`, are neither forced nor recorded.
- **Speculative isolation and state fusion.** NGRAM, EAGLE, EAGLE3, and NEXTN reconstruct each verify row from the tree mask, draft tokens, positions, and committed ring tail. Fused CUDA kernels admit contexts, force tokens, record only accepted contexts, and append only accepted tokens. When verify logprobs are requested, forcing uses a clone so downstream logprob computation reads the pre-watermark target distribution; other verify batches are forced in place.
- **Fused selector.** Triton kernels combine hashing, repeated-context admission, keyed scoring, argmax, and one-hot writeback. Finite `top_k <= 8,192` uses `torch.topk` followed by tie canonicalization and small-k truncation. Unlimited top-k retains the full-sort reference path. Runtime top-k values do not create separate Triton specializations.
- **Disabled-batch bypass.** A host candidate bit follows batch filtering and merging. Prompt initialization, verify cloning, selector work, and append are skipped when the batch has no enabled non-greedy row.
- **State interface.** `ModelRunner.watermark_state` has a class-level `None` default, then receives live state during full initialization. Direct attribute access therefore also works for partial construction paths and test fixtures.

## Code composition

The GitHub pull-request files API for the implementation before this README reported 39 files and `+6,085/-46`. Including this file, the diff contains 40 files and `+6,211/-46`.

| File | +/- | Responsibility |
| --- | ---: | --- |
| `docs/docs.json` | +1/−0 | Registers the operator guide. |
| `docs/docs/advanced_features/text_watermarking.mdx` | +397/−0 | Operator API, detector protocol, compatibility, and limitations. |
| `python/sglang/kernels/ops/sampling/textseal_selector.py` | +1,442/−0 | Normal and speculative Triton selector/state kernels; adapted from Meta TextSeal under Apache-2.0 while retaining this feature's hash contract. |
| `python/sglang/srt/arg_groups/fields/exec_.py` | +44/−0 | Watermark server arguments. |
| `python/sglang/srt/arg_groups/pipeline.py` | +2/−0 | Argument-group composition. |
| `python/sglang/srt/arg_groups/resolution_hooks.py` | +1/−0 | Resolution-hook registration. |
| `python/sglang/srt/arg_groups/serving_hook.py` | +27/−0 | File-backed key configuration. |
| `python/sglang/srt/arg_groups/validation_hook.py` | +88/−0 | Policy validation and fail-closed mode admission. |
| `python/sglang/srt/entrypoints/engine.py` | +22/−11 | Redacted engine information. |
| `python/sglang/srt/entrypoints/grpc_bridge.py` | +4/−1 | gRPC server-info redaction. |
| `python/sglang/srt/entrypoints/http_server.py` | +16/−13 | HTTP server-info redaction. |
| `python/sglang/srt/entrypoints/openai/protocol.py` | +15/−0 | OpenAI request schema. |
| `python/sglang/srt/entrypoints/openai/serving_completions.py` | +1/−0 | Completion request pass-through. |
| `python/sglang/srt/managers/io_struct.py` | +13/−0 | Tokenized-request transport. |
| `python/sglang/srt/managers/tokenizer_manager.py` | +46/−9 | Request admission and crash-dump redaction. |
| `python/sglang/srt/model_executor/forward_batch_info.py` | +15/−0 | Prompt-tail and retraction metadata. |
| `python/sglang/srt/model_executor/model_runner.py` | +51/−0 | Sampling injection and watermark state lifecycle. |
| `python/sglang/srt/sampling/sampling_batch_info.py` | +55/−0 | Per-row configuration and host candidate bookkeeping. |
| `python/sglang/srt/sampling/sampling_params.py` | +7/−0 | Per-request watermark normalization. |
| `python/sglang/srt/sampling/watermarking/__init__.py` | +13/−0 | Public detector exports. |
| `python/sglang/srt/sampling/watermarking/config.py` | +96/−0 | Key parsing, bounded JSON loading, and shared protocol limits. |
| `python/sglang/srt/sampling/watermarking/core.py` | +1,230/−0 | Request policy, batch configuration, ring/history state, selectors, and speculative orchestration. |
| `python/sglang/srt/sampling/watermarking/detector.py` | +247/−0 | Torch-free single/dual-key reference detector. |
| `python/sglang/srt/sampling/watermarking/README.md` | +126/−0 | Algorithm, runtime design, code map, and validation evidence. |
| `python/sglang/srt/server_args.py` | +14/−2 | Startup and launch-command redaction. |
| `python/sglang/srt/speculative/eagle_utils.py` | +54/−1 | Verify-time forcing and accepted-context recording. |
| `python/sglang/srt/speculative/eagle_worker_common.py` | +15/−0 | EAGLE/EAGLE3/NEXTN accepted-token append. |
| `python/sglang/srt/speculative/ngram_worker.py` | +18/−3 | NGRAM accepted-token append. |
| `python/sglang/srt/utils/request_logger.py` | +17/−4 | Request-log redaction. |
| `rust/sglang-server/src/message/sampling.rs` | +43/−0 | Rust/Python sampling schema lockstep. |
| `test/registered/e2e/openai_server/test_watermark.py` | +239/−0 | Endpoint policy, isolation, redaction, and structured-output coverage. |
| `test/registered/kernels/ops/sampling/test_watermark_selector.py` | +417/−0 | Torch/Triton parity, top-k boundaries and ties, dual key, and fp32 hash-boundary regression. |
| `test/registered/kernels/ops/sampling/test_watermark_state.py` | +421/−0 | Ring, retraction, speculative state fusion, and inactive-batch coverage. |
| `test/registered/unit/layers/test_mamba_prefill_track_metadata.py` | +1/−0 | ModelRunner interface fixture. |
| `test/registered/unit/sampling/test_sampling_batch_info.py` | +37/−1 | Filter/merge lifecycle. |
| `test/registered/unit/sampling/test_sampling_metadata_staging.py` | +4/−1 | Sampling metadata staging. |
| `test/registered/unit/sampling/test_watermark.py` | +267/−0 | Batch resolution, greedy bypass, and disabled-path behavior. |
| `test/registered/unit/sampling/test_watermark_config.py` | +430/−0 | Policy matrix, config loading, validation, redaction, and interval boundaries. |
| `test/registered/unit/sampling/test_watermark_detector.py` | +272/−0 | Detector vectors, dual-key partitioning, prefix cap, isolation, and standalone package use. |
| `test/registered/unit/test_model_overrides.py` | +3/−0 | Server-argument migration lockstep. |

## Performance and evidence

All relative performance figures compare watermark ON and OFF within the same engine and setup. Finite-top-k and speculative results use random 1,024-token prompts, 512 output tokens, concurrency 8, temperature 0.8, top-p 0.95, top-k 20, 100 measured requests after 10 warmups, and three repetitions.

| Setup | OFF | ON | Change |
| --- | ---: | ---: | ---: |
| Qwen3-8B, H200, no spec, c1 output tok/s | 197.85 | 198.11 | +0.13% |
| Qwen3-8B, H200, no spec, c8 output tok/s | 1,249.22 | 1,249.47 | +0.02% |
| Qwen3-8B, GB300, no spec, unlimited top-k, c8 output tok/s | 224.88 | 212.84 | −5.35% |
| Qwen3.5-9B, GB300 TP2, NEXTN 3/1/4, c8 TPOT | 2.246 ms | 2.451 ms | +9.17% |
| Same NEXTN run, mean accept length | 2.759 | 2.783 | +0.85% |
| Capability enabled with fully disabled traffic, GB300 EAGLE3 c8 TPOT | 2.465 ms merge-base | 2.469 ms | +0.14% |
| Dual key versus single key, GB300 output tok/s | — | — | −0.21% c1 / −1.50% c8 |

The final merge head (`48f15f3e55`) reproduced normal decode on a fresh B200 at 95/100 GSM8K accuracy, 100% stop rate, zero truncations, watermark p=`4.18e-27`, opted-out p=`0.325`, and a safe greedy stop. Its same-GPU-pair NEXTN spot check measured +8.92% TPOT, −7.28% output throughput, −0.22% accept length, watermark p=`3.72e-19`, and opted-out p=`0.950`.

Full 1,319-example GSM8K runs preserved accuracy and 100% stop rate across normal decode, NGRAM, and EAGLE3. A matched Terminal-Bench 2.1 run produced 68/89 ON and 71/89 OFF, with paired two-sided p=`0.5488`; the ON trajectories scored 5.24 million contexts at z=`783.89`, while the OFF control remained negative at p=`0.42`.

DP attention was verified with Qwen3-8B on two H200 GPUs at DP2/TP2. Normal decode scored 95/100 GSM8K with 100% stop; each rank's 1,024-token watermarked output had p=`1.84e-23`, disabled controls had p=`0.358`, and repeated one-rank-idle batches stayed healthy. A DP2/TP2 NEXTN smoke produced p=`2.14e-36` on both ranks.

### Cross-engine reference

vLLM nightly `0.30.1rc1` used its own benchmark client and PRF/detector contract. Compare each engine's within-engine delta rather than absolute throughput.

| Workload | SGLang | vLLM |
| --- | --- | --- |
| Qwen3.5-9B MTP, TP2 GB300, three draft tokens, c8, top-k 20 | TPOT +9.17%; accept length +0.85% | `dual_key_gumbel`: TPOT +14.24%; accept length +0.72% |
| Qwen3-8B, TP1 H200, no spec, top-k 20, c1/c8 output throughput | +0.13% / +0.02% | plain Gumbel: −1.70% / −2.11% |

The public-model runs showed no material speculative acceptance loss. Unlimited-top-k overhead remains hardware-dependent because it sorts the full vocabulary; measured per-stream throughput changes ranged from −2.77% to −7.24% across H200, GB300, and B300.
