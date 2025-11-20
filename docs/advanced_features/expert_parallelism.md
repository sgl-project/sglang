# Expert Parallelism

SGLang’s Mixture-of-Experts (MoE) stack can scale beyond one GPU by splitting experts across *expert parallel* (EP) ranks while continuing to share activations with tensor parallel (TP), data parallel (DP), and (optionally) pipeline parallel groups. This page explains how EP is wired in the current codebase, how it interacts with DeepEP and Mooncake A2A backends, and which server arguments govern expert placement, redundancy, and monitoring.

## Terminology and prerequisites

- `--tp-size` (`tp_size`) controls how many ranks participate in tensor-parallel compute inside each MoE layer. `--ep-size` (`ep_size`) partitions those TP ranks into EP subgroups. SGLang enforces `tp_size % ep_size == 0` and uses `moe_tp_size = tp_size / ep_size` when building MoE kernels; misconfigured shapes raise validation errors in `python/sglang/srt/model_executor/model_runner.py`.
- `--moe-runner-backend` chooses the per-expert GEMM backend (DeepGEMM, Triton, FlashInfer variants, Cutlass, etc.). Some backends enforce constraints: FP8 Cutlass only works with `ep_size == 1`, and FlashInfer Cutlass (FP4) requires `ep_size ∈ {1, tp_size}` because each backend determines how activations are split (`python/sglang/srt/server_args.py`).
- `--moe-a2a-backend` decides how router outputs are exchanged: `"none"` keeps the default all-reduce broadcast, `"deepep"` uses DeepSeek’s DeepEP transport, `"mooncake"` uses Mooncake EP (`python/sglang/srt/layers/moe/utils.py`).
- `--moe-dense-tp-size` (currently `None` or `1`) lets you clamp dense MLP tensor parallel size when large EP groups make GEMM tile sizes too small.
- EP-aware models read `get_moe_expert_parallel_world_size()` inside their constructors – for example `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/models/gpt_oss.py`, and `python/sglang/srt/models/qwen3_vl_moe.py`. These families (and their quantized variants) are the primary beneficiaries of EP today.

## Configuring `--ep-size`

1. Decide how many experts each GPU can store. `ep_size` should match the number of unique expert shards you want across the TP group. For example, on an 8×H200 DeepSeek-V3.2 deployment, set `--tp 8 --ep 8` so every GPU owns a different expert shard (see `docs/basic_usage/deepseek_v32.md`).
2. Keep `tp_size / ep_size` integral. A non-integer ratio produces `ValueError: tp_size ... must be divisible by moe_ep_size ...` at startup and also breaks the Qwen3-VL FP8 shape guard that checks `moe_intermediate_size / (tp_size / ep_size)` (`python/sglang/srt/model_executor/model_runner.py`).
3. Combine with DP carefully. Under DP-attention, `moe_dense_tp_size` defaults to `None`, so TP activations are gathered before MLPs. Setting `--moe-dense-tp-size 1` relaxes this and matches the fast-path assumptions used when an A2A backend is enabled (`python/sglang/srt/utils/common.py`).
4. When you select `--moe-a2a-backend deepep` or `--moe-a2a-backend mooncake`, SGLang automatically overrides `ep_size = tp_size`. DeepEP “normal” mode also disables CUDA graphs (`python/sglang/srt/server_args.py`).

## Choosing runner and A2A backends

### Runner backend

Use `--moe-runner-backend auto` unless you have to pin a kernel. The enum in `python/sglang/srt/layers/moe/utils.py` exposes the supported backends. Pay attention to these guardrails from `python/sglang/srt/server_args.py`:

```1387:1418:python/sglang/srt/server_args.py
        if self.moe_runner_backend == "flashinfer_cutlass":
            assert (
                self.quantization == "modelopt_fp4"
            ), "modelopt_fp4 quantization is required for Flashinfer Cutlass MOE"
            assert self.ep_size in [
                1,
                self.tp_size,
            ], "The expert parallel size must be 1 or the same as the tensor parallel size"
...
        if self.moe_runner_backend == "cutlass" and self.quantization == "fp8":
            assert (
                self.ep_size == 1
            ), "FP8 Cutlass MoE is only supported with ep_size == 1"
```

### All-to-all backend

| Backend | When to use | Notes |
| --- | --- | --- |
| `none` | Use the default NCCL all-reduce broadcast for token exchange. | Allows `ep_size != tp_size`; relies on the standard fused MoE dispatcher. |
| `deepep` | GPU-only DeepEP deployments. | Forces `ep_size = tp_size`, supports both normal (prefill) and low-latency (decode) modes, and requires DeepEP + DeepGEMM for low-latency unless you run on Ascend NPUs or FlashInfer CuTeDSL FP4. |
| `mooncake` | Clusters with Mooncake EP installed. | Forces `ep_size = tp_size`, currently low-latency only, integrates with Elastic EP for rank health tracking. |

## DeepEP backend

- **Modes.** `--deepep-mode auto` resolves to normal for prefill batches and low-latency for decode batches (`python/sglang/srt/layers/moe/utils.py`). Normal mode keeps CUDA graphing off; low-latency overlaps communication and compute but demands DeepGEMM unless you are on NPU or using FlashInfer CuTeDSL FP4 (`python/sglang/srt/layers/moe/ep_moe/layer.py`).
- **Library + env.** Install DeepEP from upstream (`sgl-project` images already bundle it) and set env vars when needed:
  - `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` (≤ 1024) caps inflight tokens per rank.
  - `SGLANG_DEEPEP_BF16_DISPATCH=1` forces bf16 dispatch, which W4A-FP8 requires (`python/sglang/srt/layers/moe/token_dispatcher/deepep.py` and `python/sglang/srt/layers/moe/ep_moe/layer.py`).
  - `SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS` can rebalance SM usage for low-latency combines (`python/sglang/srt/single_batch_overlap.py`).
- **Configuration.** Pass tuned transport parameters via `--deepep-config` (JSON string or file). DeepEP’s dispatcher also respects redundant experts and EPLB metadata when routing tokens.
- **Example.**

```bash
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3.2-Exp \
  --tp 8 --ep 8 --dp 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --deepep-config /opt/deepep_config.json
```

## Mooncake backend and Elastic EP

- **Requirements.** Install Mooncake with EP support (`mooncake.mooncake_ep_buffer`). The dispatcher allocates buffers sized by `SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` (default 128, must be ≤ 1024) when it is first used (`python/sglang/srt/layers/moe/token_dispatcher/mooncake.py`).
- **Modes.** Only low-latency dispatch is implemented today; normal mode will raise `NotImplementedError`.
- **Elastic EP.** Set `--elastic-ep-backend mooncake` to monitor rank health. When Elastic EP is enabled, `--enable-eplb` must also be active and `--eplb-algorithm` automatically becomes `elasticity_aware` if it was left as `auto` (`python/sglang/srt/server_args.py`). `python/sglang/srt/elastic_ep/elastic_ep.py` tracks active ranks so Mooncake can skip unhealthy endpoints.
- **Networking.** Pin InfiniBand devices with `--mooncake-ib-device mlx5_0,mlx5_1`; Mooncake will call its `set_device_filter` during distributed init (`python/sglang/srt/model_executor/model_runner.py`).

## Expert placement, redundancy, and load balancing

- **Redundant experts.** Add hot spares with `--ep-num-redundant-experts N`. SGLang increases the `num_physical_experts` tensor by `N` per layer (`python/sglang/srt/eplb/expert_location.py`). Some models (e.g., `bailing_moe`) still assert `ep_num_redundant_experts == 0`, so keep the flag off there (`python/sglang/srt/models/bailing_moe.py`).
- **Dispatch algorithms.** `--ep-dispatch-algorithm` supports `static`, `dynamic`, or `fake`. When unspecified, enabling EPLB or providing `--init-expert-location` forces `"static"` so spans stay deterministic (`python/sglang/srt/server_args.py` and `python/sglang/srt/eplb/expert_location_dispatch.py`).
- **Initial layouts.** `--init-expert-location trivial` maps logical experts round-robin. You can also pass a JSON/`.pt` snapshot containing `physical_to_logical_map` or `logical_count` to resume from a previous EPLB snapshot (`python/sglang/srt/eplb/expert_location.py`).
- **EPLB.** `--enable-eplb` activates Expert Placement Load Balancing. The manager periodically triggers rebalancing after `--eplb-rebalance-num-iterations` forward passes, optionally limiting to `--eplb-rebalance-layers-per-chunk` layers per pass and only when average GPU utilization drops below `--eplb-min-rebalancing-utilization-threshold`. EPLB also auto-enables the expert distribution recorder.
- **Elastic EP + EPLB.** When `--elastic-ep-backend` is set (currently only `"mooncake"`), EPLB must run in `elasticity_aware` mode to keep logical-to-physical maps consistent if ranks are muted (`python/sglang/srt/server_args.py`).
- **Metrics.** `--enable-expert-distribution-metrics` exports Prometheus counters. Set `SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL` to a positive integer to log per-layer occupancy histograms (`python/sglang/srt/eplb/expert_distribution.py`).

## Hybrid CPU/GPU experts (KTransformers)

If you want to keep some experts on CPUs (e.g., AMX INT4) while others run on GPUs, wrap the GPU quantization method with the KTransformers EP wrapper (`python/sglang/srt/layers/moe/kt_ep_wrapper.py`). Provide CPU weights, thread counts, and deferred expert limits via `--kt-*` arguments; the wrapper masks CPU-only expert IDs on the GPU side and streams CPU outputs back into the combine step. This is orthogonal to GPU EP sizing—you can still run DeepEP or Mooncake for GPU experts while offloading a tail of experts to CPU cores.

## Monitoring and troubleshooting

- **Common launch errors.**
  - `tp_size ... must be divisible by moe_ep_size ...` → adjust `--tp`/`--ep`.
  - `moe_intermediate_size ... must be divisible by moe_tp_size ...` → some FP8 models require specific TP×EP factorizations.
  - `DeepEP low_latency mode requires deep_gemm` → either enable JIT DeepGEMM (default on Hopper+) or choose `--deepep-mode normal`.
  - `W4AFP8 does not support FP8 dispatch` → set `SGLANG_DEEPEP_BF16_DISPATCH=1` when using W4A-FP8 weights.
- **Metrics to watch.** Enable `--expert-distribution-recorder-mode stat --enable-expert-distribution-metrics` to log `sglang:eplb_gpu_physical_count` histograms every `SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL` passes. DeepEP also logs masked token counts via `get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency`.
- **Environment variables.**
  - `SGLANG_DEEPEP_*` controls DeepEP dispatch precision and buffer sizing.
  - `SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` caps Mooncake queue depth.
  - `SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL` controls histogram cadence.

## Example launch recipes

| Scenario | Command | Notes |
| --- | --- | --- |
| 8×H200 DeepSeek-V3.2 with DeepEP | `python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep 8 --dp 8 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode auto` | Matches the template in `docs/basic_usage/deepseek_v32.md`; DP attention is recommended for DeepSeek models. |
| 8×H200 Qwen3-VL-235B FP8 | `python -m sglang.launch_server --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 --tp 8 --ep 8` | FP8 variants require `(moe_intermediate_size=1536 / moe_tp_size) % weight_block_size_n=128 == 0`, where `moe_tp_size = tp_size / ep_size` → You can choose `ep_size` in 2, 4 or 8. |

## Related docs

- `docs/advanced_features/server_arguments.md` lists every EP-related flag.
- `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md` shows production launch manifests that combine DeepEP, EPLB, and PD disaggregation.
- `docs/advanced_features/hyperparameter_tuning.md` covers how to benchmark different parallelism strategies, including EP.
