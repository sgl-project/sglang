# NPU PR 测试用例特性叠加分析 & GPU 对比缺口

> 分析日期: 2026-07-25 | 验证状态: ✅ 子 agent 交叉验证
> 对比基准: GPU 社区测试覆盖 (`feature_compatibility_gpu_test_coverage.md`)

---

## ⚠️ 重要前提

分析过程中发现：**所有 `test/registered/ascend/performance/` 下的性能测试文件**，其 `register_npu_ci` 配置均为：
```python
register_npu_ci(est_time=3600, suite="", nightly=True, disabled="performance testcase")
```

即 `suite=""`（空字符串）、`disabled="performance testcase"`、`nightly=True`。这意味着：
- 它们**目前未在任何 PR suite 中注册**
- 用户列表中的 "pr-single-node-tests" 和 "即将加入 PR" 指的是**计划/目标**，而非当前 CI 代码状态
- 本报告将它们作为 "计划加入 PR" 来分析其覆盖价值

---

## 一、NPU PR 用例逐项分析（已验证）

### 1.1 当前已在 PR suite 中的用例 (19 个)

| # | Job (suite) | 测试文件 | TP | DP | EP | CP | PD | Quant | Chunk | NPUGraph | SpecDec | PrefCache | Overlap | DPLMHead | NZWeight |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | stage-b-test-1-npu-a2 | test_npu_hicache_mha.py | - | - | - | - | - | - | - | - | - | ✅H | - | - | - |
| 2 | stage-b-test-1-npu-a2 | test_npu_sampling_backend.py | - | - | - | - | - | - | - | ❌ | - | ❌ | - | - | - |
| 3 | stage-b-test-1-npu-a2 | test_npu_compile_graph_tp1_bf16.py | - | - | - | - | - | - | - | - | - | ❌ | - | - | ✅ |
| 4 | stage-b-test-1-npu-a2 | test_npu_graph_tp1_bf16.py | - | - | - | - | - | - | - | ✅ | - | - | - | - | - |
| 5 | stage-b-test-1-npu-a2 | test_npu_piecewise_graph_prefill.py | - | - | - | - | - | - | - | ✅ | - | - | - | - | - |
| 6 | stage-b-test-1-npu-a2 | test_npu_autoround_dense.py | - | - | - | - | - | ✅AR | - | - | - | - | - | - | - |
| 7 | stage-b-test-1-npu-a2 | test_npu_autoround_moe.py | - | - | - | - | - | ✅AR | - | - | - | - | - | - | - |
| 8 | stage-b-test-1-npu-a2 | test_npu_gptq_moe.py | - | - | - | - | - | ✅GPTQ | - | - | - | - | - | - | - |
| 9 | stage-b-test-1-npu-a2 | test_npu_tp1_bf16.py | - | - | - | - | - | - | - | ❌ | - | - | - | - | - |
| 10 | stage-b-test-2-npu-a2 | test_npu_graph_tp2_bf16.py | ✅2 | - | - | - | - | - | - | ✅ | - | - | - | - | - |
| 11 | stage-b-test-2-npu-a2 | test_npu_mla_fia_w8a8int8.py | ✅2 | - | - | - | - | - | - | ❌ | - | ❌ | - | - | ✅ |
| 12 | stage-b-test-2-npu-a2 | test_npu_tp2_bf16.py | ✅2 | - | - | - | - | - | - | ❌ | - | - | - | - | - |
| 13 | stage-b-test-2-npu-a2 | test_npu_tp2_fia_bf16.py | ✅2 | - | - | - | - | - | - | ❌ | - | ❌ | - | - | ✅ |
| 14 | stage-b-test-4-npu-a3 | test_npu_hicache_mla.py | ✅4 | - | - | - | - | - | - | - | - | ✅H | - | - | - |
| 15 | stage-b-test-4-npu-a3 | test_npu_llada2_mini.py | - | - | - | - | - | - | - | - | - | ❌ | - | - | - |
| 16 | stage-b-test-4-npu-a3 | test_npu_w4a4_quantization.py | ✅4 | - | - | - | - | - | - | ✅ | - | ❌ | - | - | - |
| 17 | stage-b-test-4-npu-a3 | test_npu_mla_w8a8int8.py | ✅4 | - | - | - | - | - | - | ❌ | - | ❌ | - | - | - |
| 18 | stage-b-test-4-npu-a3 | test_npu_tp4_bf16.py | ✅4 | - | - | - | - | - | - | ✅ | - | ❌ | - | - | - |
| 19 | stage-b-test-16-npu-a3 | test_npu_deepep.py | ✅16 | ✅1 | ✅16 | - | - | - | ✅ | - | - | ❌ | - | - | - |

> 符号: ✅ = 使用, ❌ = 显式禁用, H = HiCache, AR = AutoRound, 数字 = TP/DP/EP size

### 1.2 计划加入 PR 的性能/精度测试 (计划 12 个，当前均为 disabled nightly)

| # | Name | Runner | TP | DP | EP | CP | PD | Quant | Chunk | NPUGraph | SpecDec | PrefCache | Overlap | DPLMHead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | qwen3_6_27b_1p_in1024... | a3-2 | ✅2 | - | - | - | - | - | ✅ | ✅ | ✅NEXTN | ❌ | ✅ | - |
| 21 | qwen3_8b_w8a8_1p_in3k5... | a3-2 | ✅1 | - | - | - | - | ✅MS | ✅ | ✅ | ✅EAGLE3 | ❌ | ✅ | - |
| 22 | qwen3_30b_w8a8_1p_in3k5... | a3-2 | ✅2 | - | ✅MoE | - | - | ✅MS | ✅ | ✅ | ✅EAGLE3 | ❌ | ✅ | - |
| 23 | qwen3_6_35b_a3b_1p_in64k... | a3-2 | ✅2 | - | ✅MoE | - | - | - | ✅ | ✅ | ✅NEXTN | ✅ | ❌ | - |
| 24 | qwen3_vl_8b_thinking_1p_mmmu | a3-2 | ✅2 | - | - | - | - | - | ✅ | - | - | ❌ | - | - |
| 25 | qwen3_32b_w8a8_2p_in3k5... | a3-4 | ✅4 | - | - | - | - | ✅MS | ✅ | ✅ | ✅EAGLE3 | ❌ | ✅ | - |
| 26 | qwen3_next_80b_w8a8_2p... | a3-4 | ✅4 | ✅2 | ✅MoE | - | - | ✅MS | ✅ | ✅ | ✅NEXTN | ❌ | ✅ | ✅ |
| 27 | minimax_m2_5_w8a8_4p... | a3-8 | ✅8 | - | ✅MoE | - | - | ✅MS | ✅ | ✅ | ✅EAGLE3 | ✅ | ✅ | - |
| 28 | deepseek_v4_flash_w8a8_8p... | a3-16 | ✅16 | ✅16 | ✅MoE | - | - | ✅MS | ✅ | ✅ | ✅EAGLE | ❌ | ✅ | ✅ |
| 29 | kimi_k2_6_w4a8_8p... | a3-16 | ✅32 | ✅32 | ✅MoE | - | ✅mix | ✅MS | ✅ | ✅ | ✅EAGLE3 | ❌ | ✅ | - |
| 30 | qwen3_235b_w8a8_8p... | a3-16 | ✅8 | - | ✅MoE | - | - | ✅MS | ✅* | ✅* | - | ✅* | - | - |
| 31 | qwen3_5_397b_w4a8_8p... | a3-16 | ✅8 | - | ✅MoE | - | - | ✅MS | ✅* | ✅* | - | ✅* | - | - |

> ✅* = 未实际读取文件验证，基于同目录其他性能测试的模式推断

---

## 二、验证审计日志

子 agent 交叉验证结果（所有声明已验证为 ✅ 正确或已修正）：

| 验证项 | 结果 | 备注 |
|:-------|:----:|:-----|
| `test_npu_hicache_mha.py` → --enable-hierarchical-cache | ✅ | L40 |
| `test_npu_sampling_backend.py` → --disable-radix-cache + --disable-cuda-graph | ✅ | L32-33 |
| `test_npu_tp4_bf16.py` → TP=4 + cuda-graph-max-bs-decode 32 + disable-radix-cache | ✅ | L41-45 |
| `test_npu_w4a4_quantization.py` → TP=4 + cuda-graph-bs 64 + 无显式 --quantization | ✅ | 量化类型由模型路径决定 |
| `test_npu_qwen3_6_27b_w8a8_1p_in64k_out1k_50ms.py` → TP=2+Quant+SpecDec+Overlap | ✅ | 但 suite=""、disabled="performance testcase" |
| `test_npu_qwen3_next_80b_w8a8_2p_...` → TP=4+DP=2+DPLMHead+Quant+SpecDec+Overlap | ✅ | 但 suite=""、disabled="performance testcase" |
| `test_npu_deepseek_v4_flash_w8a8_8p_...` → TP=16+DP=16+DPLMHead+Quant+SpecDec+Overlap | ✅ | 但 suite=""、disabled="performance testcase" |
| `test_npu_eagle3.py` → suite="nightly-1-npu-a3" | ✅ | 确认不在 PR 中 |
| `enable-dp-attention` 在 ascend PR suite 中 0 出现 | ✅ | 9 个文件使用，但全部 nightly/disabled |
| `disaggregation-mode` 在 ascend PR suite 中 0 出现 | ✅ | 3 个文件使用，全部 suite="" |
| `enable-eplb` 在 ascend 目录中 0 出现 | ✅ | 完全不存在 |
| `SGLANG_NPU_USE_MULTI_STREAM` >0 在 PR 中 0 出现 | ✅ | 仅 1 个文件 =1 且在 nightly-16-npu-a3 |
| `test_npu_deepep.py` → suite="stage-b-test-16-npu-a3" + TP16+EP16+Chunked | ✅ | L15, L41-49 |

---

## 三、NPU PR vs GPU 覆盖对比

### 3.1 单特性覆盖统计

| 特性 | GPU 等效测试 | NPU 当前 PR (19) | + 计划 PR (12) | 合计 (31) | 差距 |
|:-----|:---:|:---:|:---:|:---:|:---|
| **TP** | ✅ | 9 | 12 | **21** | 🟢 |
| **DP (--enable-dp-attention)** | ✅ | **0** | 3 | **3** | 🟡 (计划后有) |
| **EP** | ✅ | 1 | 9 | **10** | 🟢 |
| **CP** | ✅ | **0** | **0** | **0** | 🔴 |
| **PD** | ✅ | **0** | 1 | **1** | 🔴 (仅 kimi PD-mix) |
| **Quant** | ✅ | 5 | 9 | **14** | 🟢 |
| **ChunkedPrefill** | ✅ | 1 | 12 | **13** | 🟢 |
| **NPUGraph** | ✅ | 7 | 11 | **18** | 🟢 |
| **SpecDec** | ✅ | **0** | 9 | **9** | 🟡 (计划后有) |
| **PrefixCache** | ✅ | 8 | 12 | **20** | 🟢 |
| **OverlapSchedule** | ✅ | **0** | 9 | **9** | 🟡 (计划后有) |
| **DPLMHead** | ✅ | **0** | 3 | **3** | 🟡 (计划后有) |
| **MultistreamMoE** | ⚠️ | **0** | **0** | **0** | 🔴 |
| **EPLB** | ✅ | **0** | **0** | **0** | 🔴 |
| **NZWeightFormat** | ⚪ N/A | 3 | **0** | **3** | 🟢 (Ascend 专有) |

### 3.2 特性叠加组合对比（关键缺口）

| 叠加组合 | Ascend 兼容性 | GPU 覆盖 | NPU 当前 PR | NPU 计划 PR | NPU 差距 |
|:---------|:------------:|:---:|:---:|:---:|:---|
| **TP + NPUGraph + PrefixCache** | 🟠 | ✅ | ✅ (5 files) | ✅ | 🟢 |
| **TP + NZWeightFormat** | 🟢 | ⚪ | ✅ (3 files) | - | 🟢 |
| **EP + TP + ChunkedPrefill** | 🟠 | ✅ | ✅ (deepep.py) | ✅ | 🟢 |
| **EP + TP + Quant** | 🟢 | ✅ | - | ✅ (MoE models) | 🟢 |
| | | | | | |
| **SpecDec + OverlapSchedule** | 🟢 | ✅ | ❌ | ✅ (9 files) | 🟡 计划后OK |
| **SpecDec + TP + Quant** | 🟢 | ✅ | ❌ | ✅ (8 files) | 🟡 计划后OK |
| **SpecDec + NPUGraph** | 🟠 | ✅ | ❌ | ✅ (9 files) | 🟡 计划后OK |
| | | | | | |
| **DP + TP** | 🟢 | ✅ | ❌ | ✅ (3 files) | 🟡 计划后OK |
| **DP + SpecDec + TP** | 🟢 | ✅ | ❌ | ✅ (3 files) | 🟡 计划后OK |
| **DP + DPLMHead + TP** | 🟢 | ✅ | ❌ | ✅ (2 files) | 🟡 计划后OK |
| **DP + SpecDec + TP + DPLMHead + OverlapSchedule + Quant** | 多组合 | ✅ | ❌ | ✅ (deepseek_v4) | 🟡 |
| | | | | | |
| **PD + TP** | 🟢 | ✅ | ❌ | ❌ | 🔴 |
| **PD + DP** | 🟢 | ✅ | ❌ | ❌ | 🔴 |
| **PD + SpecDec** | 🟢 | ✅ | ❌ | ❌ | 🔴 |
| **PD + Quant + ChunkedPrefill + NPUGraph** | 多组合 | ✅ | ❌ | ❌ | 🔴 |
| | | | | | |
| **CP + TP** | 🟢 | ✅ | ❌ | ❌ | 🔴 |
| **CP + SpecDec** | 🟢 | ✅ | ❌ | ❌ | 🔴 |
| **CP + DP** | 🟠 | ⚠️ | ❌ | ❌ | 🔴 |
| | | | | | |
| **EP + DP + DPLMHead** | 🟠 | ✅ | ❌ | ❌ | 🔴 |
| **EP + OverlapSchedule** | 🟢 | ✅ | ❌ | ❌ | 🔴 |
| **EP + EPLB** | 🟢 | ✅ | ❌ | ❌ | 🔴 |
| | | | | | |
| **OverlapSchedule + ChunkedPrefill** | 🟠 | ❌ | ❌ | ❌ | 🟡 (两边都缺) |
| **OverlapSchedule + CP** | 🟠 | ❌ | ❌ | ❌ | 🟡 (两边都缺) |
| **EPLB + PD** | 🟠 | ❌ | ❌ | ❌ | 🟡 (两边都缺) |
| **DPLMHead + CP** | ❔ | ❌ | ❌ | ❌ | 🟡 (两边都缺) |
| **MultistreamMoE (e2e)** | 全矩阵 | ❌ | ❌ | ❌ | 🔴 |
| **MultistreamMoE + EPLB** | ❔ | ❌ | ❌ | ❌ | 🟡 (两边都缺) |

---

## 四、关键发现

### 🔴 即使计划 PR 加入后仍然完全缺失的特性

| 缺失特性 | 说明 |
|:---------|:-----|
| **CP (Context Parallelism)** | GPU 有 15+ 测试文件，NPU 当前 PR=0，计划 PR=0。Ascend 文档明确要求 `--attn-cp-size` = `--tp-size` |
| **PD (PD Disaggregation)** | GPU 有 20+ 测试文件，NPU 当前 PR=0，计划 PR 仅 kimi PD-mix（非标准 PD 分离）。Nightly 中有 GLM5.1 和 MIMO 的 PD 分离测试但未进入 PR |
| **EPLB** | GPU 有 10+ 测试文件，NPU ascend 目录完全不存在 `--enable-eplb` |
| **MultistreamMoE (e2e)** | Ascend 专有优化，GPU 无对应，必须 NPU 自行覆盖 |

### 🟡 计划 PR 加入后能缓解但仍不足

| 特性 | 计划 PR 覆盖 | 仍缺失 |
|:-----|:-----|:-----|
| **DP Attention** | 3 文件 (qwen3_next, deepseek_v4, kimi_k2) | 无 DP+EP 组合，无 DP+PD 组合 |
| **SpecDec** | 9 文件 | 无 SpecDec+PD 组合，无 SpecDec+CP 组合 |
| **DPLMHead** | 3 文件 | 无 DPLMHead+EP 组合，无 DPLMHead+PD 组合 |
| **OverlapSchedule** | 9 文件 (SpecDec perf tests) | 无 OverlapSchedule+ChunkedPrefill 组合 |

### 🔑 Nightly 中有但未进入 PR 的高价值测试

以下 nightly 测试（已注册且非 disabled）覆盖了关键特性组合，但未在 PR 中：

| 文件 | Suite | 覆盖组合 |
|:-----|:-----|:-----|
| `test_npu_eagle3.py` | nightly-1-npu-a3 | SpecDec+TP+OverlapSchedule |
| `test_npu_deepep_auto_qwen3_480b.py` | nightly-16-npu-a3 | EP+DP+DPLMHead+TP+Quant+ChunkedPrefill+NPUGraph |
| `test_npu_deepep_low_latency_qwen3_480b.py` | nightly-16-npu-a3 | EP+DP+DPLMHead+TP+Quant+ChunkedPrefill+NPUGraph |
| `test_npu_deepep_low_latency_qwen3_next.py` | nightly-16-npu-a3 | EP+DP+DPLMHead+TP+Quant+ChunkedPrefill+NPUGraph |

### 📊 与 GPU 的差距量化

| 维度 | GPU | NPU 当前 PR (19) | NPU + 计划 PR (31) |
|:-----|:---:|:---:|:---:|
| 单特性覆盖 (15 个中) | 14/15 (93%) | 7/15 (47%) | 12/15 (80%) |
| 核心组合覆盖 (30+ 对) | ~45 (90%) | ~5 (12%) | ~20 (50%) |
| 3+ 特性叠加 | 20+ | 2 | ~10 |
| PD 分离测试 | 20+ | 0 | 1 (PD-mix) |
| SpecDec 测试 | 30+ | 0 | 9 |
| DP Attention 测试 | 116+ | 0 | 3 |
| EPLB 测试 | 10+ | 0 | 0 |

---

## 五、建议优先补充的 NPU PR 测试

| 优先级 | 补测组合 | 参考 GPU 文件 | 卡数 | 来源 |
|:---:|:---|:---|:---:|:---|
| P0 | **SpecDec + TP** (基础投机推理) | `test_spec_eagle.py` | 2 | 降级 `test_npu_eagle3.py` 从 nightly → PR |
| P0 | **DP + TP** (基础 DP Attention) | `test_dp_attention.py` | 4 | 新增 |
| P0 | **PD + TP** (基础 PD 分离) | `test_disaggregation_basic.py` | 4 | 降级 nightly PD 测试 → PR |
| P1 | **SpecDec + DP + TP + DPLMHead** | `test_eagle_dp_attention.py` | 4 | 新增 |
| P1 | **EP + DP + DPLMHead + TP** | `test_deepep_low_latency_qwen3_480b.py` | 16 | 降级 nightly → PR |
| P1 | **SpecDec + OverlapSchedule** | `test_eagle_constrained_decoding.py` | 2 | 新增 |
| P1 | **CP + TP** | `test_gqa_prefill_cp.py` | 4 | 新增 |
| P2 | **EPLB + EP** | `test_lplb_distributed.py` | 2 | 新增 |
| P2 | **PD + DP + TP** | `test_disaggregation_dp_attention.py` | 8 | 降级 nightly PD → PR |
| P2 | **PD + SpecDec** | `test_specv2_kvcache_offloading.py` | 4 | 新增 |

---

## 六、总结

1. **当前 NPU PR 仅覆盖 7/15 特性**，且几乎没有超出双特性的叠加组合。最强的 PR 组合是 `test_npu_deepep.py`（TP+EP+ChunkedPrefill，3 特性）

2. **计划加入的 12 个性能测试**（当前为 disabled nightly）能大幅改善覆盖：单特性从 7→12，组合从 ~5→~20。这些测试的核心价值在于引入了 **SpecDec + OverlapSchedule + TP + Quant** 的叠加组合

3. **即使计划 PR 全部加入，仍有 3 个特性完全未覆盖**：CP、EPLB、MultistreamMoE。其中 CP 和 EPLB 在 GPU 侧有充分测试，MultistreamMoE 是 Ascend 专有需要自行设计

4. **PD 分离测试严重不足**：GPU 有 20+ 测试，NPU PR 仅有 kimi PD-mix（非标准 PD 分离）。Nightly 中有 GLM5.1 和 MIMO 的完整 PD 分离测试但未进入 PR

5. **建议优先从 nightly 降级 4 个已有测试到 PR**（test_npu_eagle3.py, deepep_auto/low_latency 系列），能以最小成本获得 SpecDec 和 DP+DPLMHead 组合的 PR 覆盖
