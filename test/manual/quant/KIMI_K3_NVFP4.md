# Kimi K3 NVFP4 experimental optimizations

This branch collects five independently controlled optimizations for independent testing on NVIDIA B300 (SM103), targeting `nvidia/Kimi-K3-NVFP4`. It is a draft integration, not a validated all-options combination. Install SGLang from the repository root with `pip install -e "python"`. The tested Cake adapter uses FlashInfer 0.7.2 private prepared-stage APIs and requires the Cake modules in that installation.

Routing/quant fusion and the aligned KV projection CUTLASS path are enabled by default. Cake, CuTe DCP prefill and BCG MHA remain disabled by default. Set the corresponding environment variable to `0` to disable a path. All runtime coverage guards still apply.

## Switches and coverage

| Environment variable (set to `1`) | Coverage | Fallback / interaction |
|---|---|---|
| `SGLANG_K3_NVFP4_ROUTE_QUANT` | TRTLLM-layout NVFP4 MoE, EP1, SM103, 1–64 activation rows, hidden size 3584; BF16 or FP32 input | Uncovered routing uses the existing quantization path. Disabled when Cake is enabled. FP8 KV cache is compatible. |
| `SGLANG_K3_NVFP4_CAKE` | TP8/EP1 NVFP4 SiTU MoE; activation row counts 1, 8, 16, 32, 64, 128; equal gate/up dequant scales | Other shapes use the original runner. Replaces the routing/quant candidate rather than combining with it. Warm supported shapes before CUDA graph capture. |
| `SGLANG_K3_KV_B_PROJ_CUTLASS` | SM103 block128 FP8 GEMM, BF16 input, N=3072, K=512, 1536≤M≤131072, M divisible by128, original backend TRTLLM | Selects scale layout before GEMM. Unaligned tails and small decode shapes keep the original runner. Applicable when prefill actually executes this projection, independently of KV-cache dtype or BCG. |
| `SGLANG_K3_DCP_CUTE_PREFILL` | BCG ordinary extend with DCP metadata, MLA dimensions 512+64 | Uses gathered KV with FlashInfer `cute-dsl-monolithic`. No effect for DCP1. This is an alternative MLA prefill path, not the MHA path below. |
| `SGLANG_K3_BCG_MHA` | Kimi K3, BCG ordinary extend, TP8 with BF16 KV/DCP8 or FP8 E4M3 KV/DCP1, BF16 activations, no attention CP/DP, chunked prefix cache enabled | Dynamic MHA/KV writes between graph segments. Other dtype/parallel configurations retain the existing attention selection. |

Row counts refer to actual GEMM/verify activation rows, not request batch size or accepted output length. All switches must be set before starting the server. Enabling an option does not prove it was used; confirm the kernel path in a separate profile.

## Independent comparisons

Keep the launch command, checkpoint, prompts, cache counts and other environment settings unchanged. Set all five switches to `0` for the baseline; unsetting a variable uses its default. For TP8/FP8-KV/DFlash block8 without DCP, run these candidates separately:

```bash
export SGLANG_K3_NVFP4_ROUTE_QUANT=1  # candidate A; other switches explicitly set to 0
# Run your unchanged launch command and benchmark.
```

```bash
export SGLANG_K3_NVFP4_ROUTE_QUANT=0
export SGLANG_K3_NVFP4_CAKE=1        # candidate B; requires Cake APIs
```

For prefill, set both MoE switches to `0` and separately enable `SGLANG_K3_KV_B_PROJ_CUTLASS=1`. Confirm that the original FP8 GEMM backend is TRTLLM and the measured request reaches the eligible KV projection. CuTe DCP prefill requires a separately labeled DCP configuration. The MHA candidate supports TP8 with either FP8 E4M3 KV/DCP1 or BF16 KV/DCP8 and requires explicitly enabled BCG (`--cuda-graph-backend-prefill breakable`) and `SGLANG_K3_BCG_MHA=1`. When comparing MHA and CuTe MLA, enable only the corresponding attention switch. The combined MHA plus KV-CUTLASS candidate is a separate run.

Use `bench_one_batch_server` for isolated prefill and steady decode, output length1024 and fixed acceptance4 for performance. Measure 8k/32k/128k, and prefill with cache0 and approximately95%, recording actual cache counts. Use natural acceptance for accuracy. A single-node isolated comparison does not establish PD transfer or multi-turn SLO performance; PD hardware is not required to test these kernel paths.

## Regression checks

```bash
python -m pytest -q test/manual/quant/test_kimi_k3_nvfp4_handoff.py
python -m pytest -q test/manual/quant/test_kimi_k3_nvfp4_route.py
python -m pytest -q test/manual/quant/test_kimi_k3_kv_projection.py
python -m pytest -q test/manual/quant/test_kimi_k3_bcg_mha_rows.py
```

The handoff test checks consume-once behavior, scale/layout identity and fallback. The GPU tests cover exact routing/FP4 bytes, changed-input CUDA graph replay, and KV projection dispatch/accuracy with nonuniform scales. Cake and the attention integrations still need full-model startup, replay and quality checks on this main-based branch. These tests are not substitutes for model accuracy checks.

Historical measurements came from earlier experimental branches: routing/quant improved 8k/B1 no-spec decode from101.89 to104.81 tok/s; Cake+CuTe improved fixed-acceptance decode by2.68–4.57% and one128k cached-prefill case from2221.9 to762.5ms. A later BF16-KV/DCP8 comparison improved128k cached-prefill from705.1 to557.6ms with MHA+BCG and then496.1ms with KV CUTLASS. These are separate experiments, not additive gains or results for this exact branch/FP8-KV/DCP1 configuration.

FP8-KV/DCP1 BCG MHA validation: startup and padded graph replay succeeded after matching KV write locations to the real token count; the five row/metadata regression checks passed. On B300x8 TP8, with Modal DFlash block8 and natural acceptance, cached retrieval passed 18/18 for both eager MHA and BCG MHA. BCG MHA scored 196/200 on GSM8K versus 197/200 for eager MHA, with one paired regression (dataset item796) and no fixes. Prompts match and neither arm hit the output-length limit. This is not a claim of accuracy parity; targeted repeatability checks remain pending. This option remains disabled by default.

The following isolated B1 prefill measurements use the PR attention changes through `fb7830c269`, with route/quant and KV CUTLASS enabled in both arms. The only arm differences are the prefill graph backend (disabled versus breakable) and `SGLANG_K3_BCG_MHA` (0 versus 1). Each median uses three formal repetitions after two discarded repetitions. Prompt hashes and actual cached counts match. The timer is `bench_one_batch_server` with OSL1; these are not decode or PD measurements.

| Input tokens | Cached tokens | Hit rate | Extend tokens | Eager MHA (ms) | BCG MHA (ms) | BCG extend tokens/s |
|---|---|---|---|---|---|---|
| 8192 | 7680 | 93.75% | 512 | 303.8 | 98.3 | 5208.5 |
| 32768 | 31232 | 95.31% | 1536 | 316.9 | 175.7 | 8742.2 |
| 131072 | 124416 | 94.92% | 6656 | 554.5 | 554.2 | 12010.1 |

These show a graph benefit versus eager MHA at 8k and 32k; 128k has no demonstrated improvement. They do not isolate MHA versus MLA. Cold-prefill coverage remains pending; the separate decode configuration comparison below covers the requested input lengths and batch sizes. The single-node adaptation used prefill Mamba ratio0.9 and HiCache180; it does not substitute for the decode configuration with Mamba ratio0.015.

## Decode communication configuration comparison

Measured on B300x8 using runtime commit `e973588745`, based on main `a77998e157`. Both arms use NVIDIA NVFP4 weights (revision `b2428a0b83a8b712ff2e1a8448a103d4175341f1`), Modal DFlash (revision `c192d15a43407bf758b5ae0880d5c72052fef1de`), TP8/DCP1, FP8 E4M3 KV, block8, ReplaySSM, and `--enable-symm-mem`. Only `--disable-custom-all-reduce` differs. These are configuration gains, not additional optimization code gains.

The single-node adaptation removes PD transport options and explicitly disables radix cache, matching PD decode's automatic setting. It preserves Mamba ratio0.015 and uses maximum batch4. Each result is steady full-batch output throughput per user, with OSL1024 and fixed acceptance4: median of three formal runs after one discarded run. All runs produced the requested token count without early stopping. Profiler runs are separate.

| Input tokens | Batch | Custom AR disabled (tok/s/user) | Custom AR enabled (tok/s/user) | Gain |
|---|---|---|---|---|
| 8192 | 1 | 239.55 | 253.67 | 5.89% |
| 8192 | 4 | 187.23 | 193.02 | 3.10% |
| 32768 | 1 | 233.49 | 250.95 | 7.48% |
| 32768 | 4 | 173.03 | 179.48 | 3.73% |
| 131072 | 1 | 213.49 | 224.90 | 5.34% |
| 131072 | 4 | 148.92 | 153.42 | 3.02% |

Trace verification: the baseline uses `ncclSymkDevKernel_AllReduce_RSxLDMC_AGxSTMC_sum_bf16`; the enabled arm uses `sglang::all_reduce_2shot_push_kernel`. At 8k/B4, TP0 VERIFY graphs contain 187 all-reduce calls each. Their median cumulative duration across ten graphs changes from2432.161 to1525.979µs, while median graph span changes from17898.042 to17664.688µs. These profiled durations are not the formal benchmark or an estimate of critical-path savings. One enabled-arm all-reduce outside the graphs took23101.501µs; its cause is unresolved, so whole-trace cumulative communication time is not used to infer speedup.

Both traces execute the existing `set_mla_kv_concat_q_fp8_kernel` and routing/NVFP4 quant fusion. Symmetric memory remains enabled under DCP1; K3 AR/norm fusion's automatic path skips this configuration. The separate no-symmetric-memory comparison is recorded below. Natural-acceptance accuracy for these communication configurations remains pending; these fixed-acceptance timings do not validate accuracy or PD SLOs.

Dependencies: PyTorch2.14.1, FlashInfer0.7.2, fastsafetensors0.5.0, transformers5.19.0. From the repository root with the environment active:

```bash
# Start one arm at a time. Stop it before starting the other arm.
bash test/manual/quant/launch_kimi_k3_nvfp4_decode_bench.sh --disable-custom-all-reduce
# Enabled arm:
bash test/manual/quant/launch_kimi_k3_nvfp4_decode_bench.sh
```

In a separate terminal, repeat the following with input lengths8192/32768/131072 and batch sizes1/4, using a distinct result file for each repetition. Discard the first repetition and take the next three medians. The launch script uses the model names; for exact reproduction, use the checkpoint revisions listed above.

```bash
python -m sglang.bench_one_batch_server \
  --model-path nvidia/Kimi-K3-NVFP4 --base-url http://127.0.0.1:30000 \
  --backend sglang --dataset-name random-ids --seed 42 \
  --batch-size 1 --input-len 8192 --output-len 1024 --cache-hit-rate 0 \
  --result-filename decode-8192-b1-r0.jsonl --show-report
```

### Disabling symmetric memory

With custom all-reduce enabled, removing only `--enable-symm-mem` automatically enables the existing K3 AR/norm fusion. This additional configuration comparison uses identical runtime source, dependencies and checkpoints, with the same fixed-acceptance benchmark method above:

| Input tokens | Batch | Custom AR + symmetric memory (tok/s/user) | Custom AR without symmetric memory (tok/s/user) | Additional gain |
|---|---|---|---|---|
| 8192 | 1 | 253.67 | 270.70 | 6.71% |
| 8192 | 4 | 193.02 | 204.84 | 6.12% |

Relative to the original disabled-custom-AR/symmetric-memory configuration, these measured throughputs improve by13.00% at B1 and9.40% at B4. This is a configuration result, not an additional code patch or a sum of independent gains. The 32k/128k no-symmetric-memory cases and natural-acceptance accuracy remain pending.

The 8k/B4 TP0 trace confirms the changed fusion path: each VERIFY graph executes95 `all_reduce_2shot_push`,92 `all_reduce_pull_res`, and92 `all_reduce_push_norm_cluster` calls. Their respective median cumulative durations across ten graphs are670.437,1332.551 and635.365µs. Median graph span is16328.993µs, versus17664.688µs with symmetric memory. This path changes both collective structure and compute overlap; cumulative communication durations are not directly comparable as critical-path savings.

To reproduce this performance configuration, start the server with:

```bash
K3_BENCH_SYMM_MEM=0 bash test/manual/quant/launch_kimi_k3_nvfp4_decode_bench.sh
```

`K3_BENCH_SYMM_MEM` controls this example script only; it is not an SGLang runtime environment variable. The script still fixes acceptance for performance, so it must not be used for accuracy tests.
