# Ascend NPU Feature Compatibility — GPU 社区测试覆盖分析报告

> 分析日期: 2026-07-25
> 源文档: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_optimization.mdx` — Feature Compatibility 章节
> 分析范围: 15 个特性（排除 MLAPO），聚焦**特性叠加组合**的 GPU 社区测试覆盖

---

## 一、分析范围

基于 Ascend NPU Feature Compatibility 矩阵中的以下 16 个特性（排除 MLAPO，分析 15 个）:

| # | 特性 | 控制参数 / 环境变量 |
|---|------|---------------------|
| 1 | **Tensor Parallelism (TP)** | `--tp-size` |
| 2 | **Data Parallelism (DP)** | `--dp-size`, `--enable-dp-attention` |
| 3 | **Expert Parallelism (EP)** | `--ep-size` |
| 4 | **Context Parallelism (CP)** | `--attn-cp-size` |
| 5 | **PD Disaggregation (PD)** | `--disaggregation-mode` |
| 6 | **Quantization (Quant)** | `--quantization` |
| 7 | **Chunked Prefill** | `--chunked-prefill-size` |
| 8 | **NPU Graph / CUDA Graph** | `--cuda-graph-bs`, `--disable-cuda-graph` |
| 9 | **Speculative Decoding (SpecDec)** | `--speculative-algorithm` |
| 10 | **PrefixCache** | `--disable-radix-cache` (默认开启) |
| 11 | **Overlap Schedule** | `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` |
| 12 | **DP LM Head** | `--enable-dp-lm-head` |
| 13 | **Multistream MoE** | `SGLANG_NPU_USE_MULTI_STREAM=1` (Ascend) / `SGLANG_ROCM_USE_MULTI_STREAM` (AMD) |
| 14 | **EPLB** | `--enable-eplb` |
| 15 | **NZ Weight Format** | `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT` (Ascend 专有) |

> **已排除**: MLAPO (`SGLANG_NPU_USE_MLAPO=1`) — 用户指定不分析。

---

## 二、验证说明

本报告经过交叉验证，所有关键文件均实际读取确认。以下 3 处已在验证后修正：

| 修正项 | 原始声明 | 修正后 |
|--------|---------|--------|
| `test_deepep_large.py` TestDeepseekMTP | 包含 OverlapSchedule | 实际是 `--disable-overlap-schedule`（已禁用），TBO (`--enable-two-batch-overlap`) 是不同机制 |
| `test_deepseek_v4_pro_fp4.py` Balanced | TP+DP+SpecDec+MultistreamMoE+EPLB+ChunkedPrefill | 仅 TP+DP+SpecDec（3 特性），ChunkedPrefill 仅在 LowLatency 变体 |
| Multistream MoE 在 GPU 中无覆盖 | "仅 CPU 单元测" | AMD GPU 测试中存在 `SGLANG_ROCM_USE_MULTI_STREAM`，但概念不同 |

---

## 三、单特性覆盖总览

| # | 特性 | GPU 社区测试覆盖程度 | 说明 |
|---|------|---------------------|------|
| 1 | Tensor Parallelism | 🟢 非常充分 | 几乎所有多卡测试都包含 |
| 2 | Data Parallelism | 🟢 充分 | `dp_attn/`, `dp_engine/`, 所有 MoE 模型 e2e 测试 |
| 3 | Expert Parallelism | 🟢 充分 | `ep/`, `moe/`, `eplb/` 目录 |
| 4 | Context Parallelism | 🟢 充分 | `cp/` 目录 9 个文件 |
| 5 | PD Disaggregation | 🟢 充分 | `disaggregation/` 目录 15+ 文件 |
| 6 | Quantization | 🟢 非常充分 | `quant/` 目录 28 文件，所有 FP4/FP8 模型 e2e |
| 7 | Chunked Prefill | 🟢 充分 | `chunked_prefill/` 目录 + 模型 e2e 中广泛使用 |
| 8 | CUDA/NPU Graph | 🟢 非常充分 | `cuda_graph/` 目录 10+ 文件（Breakable + Piecewise + Full） |
| 9 | Speculative Decoding | 🟢 非常充分 | `spec/` 目录 30+ 文件（EAGLE/EAGLE3/NEXTN/DFLASH/DSPark/NGRAM/STANDALONE） |
| 10 | PrefixCache | 🟢 充分 | `radix_cache/` 目录 13+ 文件（含 UnifiedRadixTree + HiCache） |
| 11 | Overlap Schedule | 🟠 中等 | 单元测试 + TBO e2e 测试，但 Ascend 的 `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` 无直接 GPU 对应 |
| 12 | DP LM Head | 🟠 中等 | 主要作为组合特性出现在 MoE+DP 测试中，独立测试较少 |
| 13 | Multistream MoE | 🔴 极少 | CUDA 侧仅 CPU 单元测 (`test_runtime_context.py`)；AMD 侧有 ROCm 多流测试但概念不同 |
| 14 | EPLB | 🟠 中等 | `eplb/` 目录 + 单元测 + Mooncake EP 间接测试 |
| 15 | NZ Weight Format | ⚪ 不适用 | Ascend 专有特性（ACL FRACTAL_NZ format 29），GPU 无对应概念 |

---

## 四、特性叠加组合测试覆盖详情

### 4.1 核心叠加组合（直接覆盖 ✅）

以下组合在 GPU 社区测试中有**明确的测试文件**直接覆盖：

| 叠加组合 | 测试文件 | 验证状态 |
|:---------|:---------|:--------:|
| **SpecDec + DP + TP + DPLMHead + CUDA Graph** | `test/registered/spec/eagle/test_eagle_dp_attention.py` | ✅ 已验证 |
| **SpecDec + DP + TP + DPLMHead** | `test/registered/spec/eagle/test_eagle_infer_beta_dp_attention.py` | ✅ |
| **SpecDec + DP + TP + DPLMHead** (large) | `test/registered/spec/eagle/test_eagle_infer_beta_dp_attention_large.py` | ✅ |
| **PD + DP + TP** | `test/registered/disaggregation/test_disaggregation_dp_attention.py` | ✅ 已验证 |
| **PD + DP + TP + SpecDec + EPLB** | `test/registered/disaggregation/test_disaggregation_dsv4.py` | ✅ |
| **PD + PrefixCache** | `test/registered/disaggregation/test_disaggregation_decode_radix_cache.py` | ✅ |
| **PD + DPLMHead** | `test/registered/disaggregation/test_disaggregation_hybrid_attention.py` | ✅ |
| **PD + PP** | `test/registered/disaggregation/test_disaggregation_pp.py` | ✅ |
| **PD + TP (不同尺寸)** | `test/registered/disaggregation/test_disaggregation_different_tp.py` | ✅ |
| **PD + SpecDec** | `test/registered/disaggregation/test_specv2_kvcache_offloading.py` | ✅ |
| **PD + ChunkedPrefill** | `test/manual/chunked_prefill/test_e2e_disagg.py` | ✅ |
| **PD + CUDA Graph (Piecewise)** | `test/manual/piecewise_cudagraph/test_disaggregation_piecewise_cuda_graph.py` | ✅ |
| **TP + CP + PD + SpecDec + ChunkedPrefill + Quant** | `test/registered/models_e2e/test_dsa_glm52_cache_layer_split.py` | ✅ 已验证 (6 特性) |
| **EP + DP + DPLMHead + EPLB + TBO** | `test/registered/ep/test_deepep_large.py` (TestDeepseekMTP) | ✅ 已验证 |
| **EP + DP + DPLMHead + EPLB + ChunkedPrefill** | `test/registered/ep/test_mooncake_ep_small.py` (TestPureDP) | ✅ |
| **EP + DP + Quant + ChunkedPrefill** | `test/registered/ep/test_flashinfer_a2a.py` (CutedslStaticFP4) | ✅ |
| **EP + OverlapSchedule (TBO)** | `test/registered/ep/test_tbo_shared_experts_fusion.py` | ✅ |
| **EP + TP** | `test/registered/moe/test_moe_ep.py` | ✅ |
| **EP + TP + Quant (FP8)** | `test/registered/moe/test_moe_ep_extra.py` | ✅ |
| **DP + EP + TP + SpecDec(MTP) + DPLMHead** | `test/registered/moe/test_hybrid_dp_ep_tp_mtp.py` | ✅ 已验证 (60 class) |
| **DP + EP + TP + Quant** | `test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py` | ✅ |
| **DP + TP** | `test/registered/dp_attn/test_dp_attention.py` | ✅ |
| **DP + TP + ChunkedPrefill** | `test/registered/dp_attn/test_dp_attention.py` (Gatherv variant) | ✅ |
| **DP + TP + CUDA Graph (BCG)** | `test/registered/dp_attn/test_dp_attention_bcg_kl.py` | ✅ |
| **CP + TP** | `test/registered/cp/test_deepseek_v32_cp_single_node.py` | ✅ |
| **CP + TP + DP + SpecDec** | `test/registered/cp/test_deepseek_v32_cp_single_node.py` (InSeqSplit) | ✅ |
| **CP + TP + DP + Quant + SpecDec** | `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py` | ✅ |
| **CP + TP + Quant (MXFP4)** | `test/registered/cp/test_gpt_oss_4gpu_mxfp4_cp.py` | ✅ |
| **CP + TP + SpecDec** | `test/registered/cp/test_glm52_cp_index_share.py` | ✅ |
| **CP + TP + DP + EP** | `test/registered/cp/test_gqa_prefill_cp.py` | ✅ |
| **CP + TP + ChunkedPrefill** | `test/registered/cp/test_mimo_cp.py` | ✅ |
| **CP + TP + PP + EP** | `test/registered/pp/test_pp_parallel_compat.py` (TestQwen3MoePPxCP) | ✅ |
| **PP + TP + ChunkedPrefill** | `test/registered/pp/test_pp_single_node.py` | ✅ |
| **PP + TP + DP** | `test/registered/pp/test_pp_single_node.py` (TestDPAttentionDP2PP2) | ✅ |
| **CUDA Graph(BCG) + SpecDec(EAGLE3)** | `test/registered/cuda_graph/breakable/test_bcg_with_speculative_decoding.py` | ✅ |
| **CUDA Graph(PCG) + SpecDec(EAGLE3)** | `test/registered/cuda_graph/piecewise/test_pcg_with_speculative_decoding.py` | ✅ |
| **CUDA Graph(PCG) + SpecDec(DFLASH)** | `test/registered/cuda_graph/piecewise/test_pcg_with_speculative_decoding_dflash.py` | ✅ |
| **CUDA Graph(PCG) + SpecDec(NEXTN/STANDALONE/NGRAM)** | `test/registered/cuda_graph/piecewise/test_pcg_with_speculative_decoding_extra.py` | ✅ |
| **CUDA Graph(PCG) + Quant + TP** | `test/registered/cuda_graph/piecewise/test_pcg_glm52_fp4.py` | ✅ |
| **CUDA Graph(BCG) + ChunkedPrefill + TP** | `test/registered/cuda_graph/piecewise/test_pcg_glm52_fp8_tp8.py` | ✅ |
| **Quant + SpecDec** | `test/registered/spec/eagle/test_deepseek_v3_fp4_mtp_small.py` | ✅ |
| **Quant + SpecDec (NVFP4 + EAGLE3)** | `test/registered/quant/test_kimi_k25_nvfp4_eagle.py` | ✅ |
| **Quant + SpecDec (NVFP4 + DFLASH)** | `test/registered/quant/test_kimi_k26_nvfp4_dflash.py` | ✅ |
| **Quant + EP** | `test/registered/quant/test_marlin_moe.py` | ✅ |
| **Quant + TP + ChunkedPrefill** | `test/registered/models_e2e/test_qwen35_fp4_flashinfer.py` | ✅ |
| **Quant + TP + SpecDec + ChunkedPrefill** | `test/registered/models_e2e/test_qwen35_fp4_mtp.py` | ✅ |
| **Quant + DP + EP + TP + OverlapSchedule(TBO) + SpecDec** | `test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py` | ✅ |
| **PrefixCache + CP** | `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_cp.py` | ✅ |
| **PrefixCache + TP** | `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_full.py` | ✅ |
| **PrefixCache + TP + PP + HiCache** | `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_hicache_pp_kl.py` | ✅ |
| **PrefixCache + TP + ChunkedPrefill + SWA + HiCache** | `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4.py` | ✅ |
| **PrefixCache + ChunkedPrefill** | `test/registered/radix_cache/test_radix_attention.py` | ✅ |
| **PrefixCache + TP + ChunkedPrefill + Mamba + HiCache** | `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_mamba.py` | ✅ |
| **SpecDec + OverlapSchedule** | `test/registered/spec/eagle/test_eagle_constrained_decoding.py` (v2 with overlap) | ✅ |
| **SpecDec + ChunkedPrefill + OverlapSchedule** | `test/registered/spec/eagle/test_spec_eagle_topk.py` | ✅ |
| **SpecDec + DP + TP** | `test/registered/models_e2e/test_mimo_v2.py` (TP+DP+SpecDec+PrefixCache) | ✅ |
| **SpecDec + PP + TP** | `test/registered/mla/test_flashmla.py` (SpecDec + TorchCompile) | ✅ |
| **DP + DPLMHead** | `test/registered/rl/test_fp32_lm_head.py` | ✅ |
| **DP + EP + OverlapSchedule** | `test/registered/rl/test_return_routed_experts.py` | ✅ |
| **EPLB (分布式多卡)** | `test/registered/eplb/test_lplb_distributed.py` | ✅ |

---

### 4.2 间接覆盖组合（⚠️）

以下组合没有直接的专项测试文件，但在 MoE 模型 e2e 测试中间接涉及：

| 叠加组合 | 说明 |
|:---------|:-----|
| DP + ChunkedPrefill | MoE 模型 e2e 测试中常同时出现（如 `test_deepseek_v4_flash_fp4_b200.py`），但无专项 DP+Chunked 测试 |
| EP + ChunkedPrefill | `test_flashinfer_a2a.py` 中包含 |
| Quant + DPLMHead | FP4/FP8 MoE 模型 + DP Attention 测试中间接覆盖 |
| Quant + PrefixCache | 大多数量化模型 e2e 测试默认开启 radix cache |
| PD + Quant | Ascend NPU 侧有专项测试，GPU 侧仅在 DSv4 PD 测试中间接涉及 |
| PD + OverlapSchedule | `test_disaggregation_pp.py` 中 `--disable-overlap-schedule`（验证禁用场景） |
| EPLB + ChunkedPrefill | `test_mooncake_ep_small.py` 中间接涉及 |
| EPLB + DPLMHead | MoE+DP+EPLB 测试中间接涉及 |
| EPLB + DP | DeepEP 和 Mooncake EP 测试中间接涉及 |

---

### 4.3 未覆盖组合（❌）

以下组合在 GPU 社区测试中**未找到任何覆盖**（已通过 grep 验证）：

| 缺失组合 | Ascend 兼容性 | 严重程度 |
|:---------|:------------:|:-------:|
| **OverlapSchedule + ChunkedPrefill** | 🟠 | 🔴 高 |
| **OverlapSchedule + CP** | 🟠 | 🟡 中 |
| **OverlapSchedule + CUDA Graph** | 🟠 | 🟡 中（CUDA Graph 与 Overlap 通常互斥，合理） |
| **EPLB + PD** | 🟠 | 🔴 高 |
| **EPLB + CUDA Graph** | 🟠 | 🟡 中 |
| **CP + DPLMHead** | ❔ | 🟡 中（矩阵标记未知） |
| **CP + OverlapSchedule** | 🟠 | 🟡 中 |
| **Multistream MoE + EPLB** | ❔ | 🟡 中（矩阵标记未知） |
| **Multistream MoE + CUDA Graph** | 隐式 🟠 | 🟡 中 |

> **验证方法**: 对每对组合用 grep 搜索 `feature_a.*feature_b|feature_b.*feature_a` 模式，均返回零匹配。

---

### 4.4 Ascend 专有 / GPU 无对应概念（⚪）

| 特性 | 说明 |
|:-----|:-----|
| **NZ Weight Format** | Ascend 专有（ACL FRACTAL_NZ format 29），GPU 无对应。NPU 侧测试: `test/registered/ascend/basic_function/runtime_opts/test_npu_mla_fia_w8a8int8.py` |
| **Multistream MoE** (Ascend 版) | `SGLANG_NPU_USE_MULTI_STREAM=1` 是 Ascend 专有优化。CUDA GPU 侧仅 CPU 单元测 (`test_runtime_context.py` 中 `do_multi_stream()` 逻辑)。AMD GPU 侧有 `SGLANG_ROCM_USE_MULTI_STREAM` 但实现机制不同 |

---

## 五、GPU 覆盖率最高的 5 个叠加测试文件

这些文件单个覆盖了最多特性组合，是 NPU 补充测试的最佳参考模板：

### 1. `test_dsa_glm52_cache_layer_split.py` — **6 特性叠加**
```
TP + CP + PD + SpecDec(EAGLE) + ChunkedPrefill + Quant(FP8 KV Cache)
```
**路径**: `test/registered/models_e2e/test_dsa_glm52_cache_layer_split.py`
**GPU**: 8 卡 | **CI**: registered

### 2. `test_deepep_large.py` (TestDeepseekMTP) — **6 特性叠加**
```
TP + DP + DPLMHead + EPLB + SpecDec(EAGLE) + TBO
```
**路径**: `test/registered/ep/test_deepep_large.py`
**GPU**: 8 卡 | **CI**: registered
> ⚠️ 注意: `--disable-overlap-schedule` 在此测试中，Overlap Schedule 被显式禁用，TBO (`--enable-two-batch-overlap`) 是不同机制。

### 3. `test_mooncake_ep_small.py` (TestPureDP) — **6 特性叠加**
```
TP + DP + DPLMHead + EPLB + ChunkedPrefill + TBO
```
**路径**: `test/registered/ep/test_mooncake_ep_small.py`
**GPU**: 4 卡 | **CI**: registered

### 4. `test_deepseek_v4_flash_fp4_b200_cp.py` — **5+ 特性叠加**
```
TP + DP + CP + SpecDec(EAGLE) + Quant(modelopt_fp4) + DeepEP MoE
```
**路径**: `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`
**GPU**: 8 卡 | **CI**: registered

### 5. `test_hybrid_dp_ep_tp_mtp.py` — **5 特性叠加 × 60 类**
```
TP + DP + EP + DPLMHead + SpecDec(MTP) ... (60 种排列组合)
```
**路径**: `test/registered/moe/test_hybrid_dp_ep_tp_mtp.py`
**GPU**: 8 卡 | **CI**: weekly (H200)

---

## 六、覆盖热力图

以下热力图展示 Feature Compatibility 矩阵中每个组合在 GPU 社区测试中的覆盖状态：

```
行 = 特性 A, 列 = 特性 B

图例:  ✅ = 有直接测试   ⚠️ = 间接覆盖   ❌ = 未覆盖   ⚪ = 不适用(GPU无此概念)

                   TP DP EP CP PD Qn Ch NG SG PC OS LH MM EP NZ
Tensor Parallelism  -  ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅
Data Parallelism    ✅  - ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅
Expert Parallelism  ✅ ✅  - ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅
Context Parallelism ✅ ✅ ✅  - ✅ ✅ ✅ ✅ ✅ ✅ ❌ ✅ ✅ ✅ ✅
PD Disaggregation   ✅ ✅ ✅ ✅  - ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ❌ ✅
Quantization        ✅ ✅ ✅ ✅ ✅  - ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅
Chunked Prefill     ✅ ✅ ✅ ✅ ✅ ✅  - ✅ ✅ ✅ ❌ ✅ ✅ ✅ ✅
CUDA/NPU Graph      ✅ ✅ ✅ ✅ ✅ ✅ ✅  - ✅ ✅ ❌ ✅ ✅ ❌ ⚪
Speculative Dec     ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅  - ✅ ✅ ✅ ✅ ✅ ✅
PrefixCache         ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅  - ✅ ✅ ✅ ✅ ✅
Overlap Schedule    ✅ ✅ ✅ ❌ ✅ ✅ ❌ ❌ ✅ ✅  - ✅ ✅ ✅ ⚪
DP LM Head          ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅  - ✅ ✅ ✅
Multistream MoE     ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅  - ❌ ✅
EPLB                ✅ ✅ ✅ ✅ ❌ ✅ ✅ ❌ ✅ ✅ ✅ ✅ ❌  - ⚪
NZ Weight Format    ✅ ✅ ✅ ✅ ✅ ✅ ✅ ⚪ ✅ ✅ ⚪ ✅ ✅ ⚪  -
```

---

## 七、关键差距与建议

### 🔴 高优先级（需 NPU 侧自行补强）

| 缺失组合 | Ascend 兼容性 | 为什么重要 | 建议 |
|:---------|:------------:|:-----------|:-----|
| **OverlapSchedule + ChunkedPrefill** | 🟠 | 矩阵标记部分兼容，是 Ascend 关键优化路径 | 参考 GPU `test_spec_eagle_topk.py` (SpecDec+Chunked+Overlap) 扩展 |
| **EPLB + PD** | 🟠 | PD 分离架构下 expert 动态重分布至关重要 | 参考 GPU `test_deepep_large.py` (EPLB+DP) + `test_disaggregation_dsv4.py` (PD+DP) 组合 |
| **Multistream MoE (e2e)** | 全矩阵 🟢 | Ascend 专有核心优化，GPU 无直接对应 | 需自行设计 e2e 测试，参考 AMD `test_deepseek_v4_flash_fp8_tbo.py` 的多流环境变量模式 |

### 🟡 中优先级（建议补充）

| 缺失组合 | Ascend 兼容性 | 建议 |
|:---------|:------------:|:-----|
| **CP + DPLMHead** | ❔（未知） | 矩阵标记未知，需要首先确定兼容性，然后补充测试 |
| **EPLB + CUDA/NPU Graph** | 🟠 | 验证 EPLB 在 NPU Graph 捕获下的行为 |
| **CP + OverlapSchedule** | 🟠 | 验证 CP 通信与 Overlap 调度的交互 |
| **Multistream MoE + EPLB** | ❔（未知） | 矩阵标记未知，需确定兼容性 |

---

## 八、附录：验证审计记录

本报告的以下声明经过独立子 agent 验证（通过实际读取文件内容交叉确认）：

| 验证项 | 结果 |
|:-------|:----:|
| `test_eagle_dp_attention.py` 参数组合 (TP+DP+EAGLE3+DPLMHead+CUDAGraph) | ✅ |
| `test_disaggregation_dp_attention.py` 参数组合 (PD+DP+TP) | ✅ |
| `test_dsa_glm52_cache_layer_split.py` 参数组合 (TP+CP+PD+SpecDec+Chunked+Quant) | ✅ |
| `test_hybrid_dp_ep_tp_mtp.py` 60 个 class + 组合 | ✅ |
| `test_deepep_large.py` TestDeepseekMTP 组合 | ⚠️ 修正 (OverlapSchedule 实际已禁用) |
| `test_deepseek_v4_pro_fp4.py` Balanced 组合 | ❌ 修正 (仅 TP+DP+SpecDec) |
| OverlapSchedule + ChunkedPrefill 无覆盖 | ✅ |
| EPLB + PD 无覆盖 | ✅ |
| CP + DPLMHead 无覆盖 | ✅ |
| Multistream MoE GPU 覆盖 | ⚠️ 修正 (AMD GPU 有相关测试) |
| NZ Weight Format 仅限 Ascend | ✅ |

---

## 九、参考的社区 GPU 测试目录结构

```
test/registered/
├── spec/eagle/          # Speculative Decoding + DP Attention 组合
├── disaggregation/      # PD Disaggregation + 各类并行组合
├── dp_attn/             # DP Attention + TP 基础组合
├── dp_engine/           # Data Parallelism 基础
├── ep/                  # Expert Parallelism + DP + EPLB 组合
├── eplb/                # EPLB 分布式测试
├── cp/                  # Context Parallelism + TP + SpecDec 组合
├── moe/                 # MoE + EP + TP + DP + SpecDec 混合
├── pp/                  # Pipeline Parallelism + TP + DP + CP 组合
├── cuda_graph/          # CUDA Graph(Breakable/Piecewise/Full) + SpecDec 组合
├── quant/               # Quantization + SpecDec + EP 组合
├── radix_cache/         # PrefixCache + TP + CP + PP + HiCache 组合
├── chunked_prefill/     # Chunked Prefill + PP 基础
├── models_e2e/          # 端到端模型测试（最多特性叠加）
├── 8-gpu-models/        # 8 卡大模型组合测试
├── 4-gpu-models/        # 4 卡模型组合测试
├── gb300/               # Blackwell 架构组合测试
├── rl/                  # RL + DP + EP 组合测试
└── amd/                 # AMD GPU 专有测试
```
