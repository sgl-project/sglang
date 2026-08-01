# NPU 测试仓 → 社区仓 缺口合入方案

> 分析日期: 2026-07-25
> 测试仓: `D:\00_code\claude\ascend_sglang` (testcases 分支)
> 社区仓: `D:\00_code\claude\sglang` (main 分支)
> 目标: 找出测试仓有但社区仓没有的 NPU 用例，合入社区仓补齐特性缺口

---

## 一、目录级差异

### 测试仓比社区仓多的 basic_function 子目录

| 目录 | 文件数 | 对应缺口 |
|:-----|:---:|:---|
| `dp_attn/` | 1 | **DP Attention** |
| `pd_disaggregation/` | 1 | **PD Disaggregation** |
| `parallel_strategy/data_parallelism/` | 2 | **DP Load Balance** |
| `speculative_inference/` (额外) | +7 | **SpecDec 扩展** |
| `optimization_debug_options/` | 17 | 调试选项 |
| `runtime_options/` | 10 | 运行时选项 |
| `EPD/` | 2 | Encoder-Prefill-Decode |
| `model_tokenizer/` | 若干 | Tokenizer 测试 |
| 其他 (lora/mambacache/forward/等) | 若干 | 其他功能 |

### 测试仓比社区仓多的 expert_parallelism 文件

| 测试仓文件 | 覆盖缺口 |
|:-----|:---|
| `test_npu_eplb_min_rebalancing_utilization_threshold.py` | **EPLB** |
| `test_npu_expert_distribution_recorder_mode.py` | EP 扩展 |
| `test_npu_moe_dense_tp_size.py` | MoE dense TP |
| `test_npu_moe_runner_backend.py` | MoE backend |
| `test_npu_deepep_auto_deepseek_R1_0528_w4a8_per_channel.py` | EP + per-channel quant |
| `test_npu_deepep_auto_deepseek_v2.py` | EP (DeepSeek V2) |
| `test_npu_deepep_auto_qwen3_235b_a22b_w8a8.py` | EP (Qwen3 235B) |
| `test_npu_deepep_auto_qwen3_30b_a3b_w8a8.py` | EP (Qwen3 30B) |
| `test_npu_deepep_low_latency_deepseek_R1_0528_w4a8_per_channel.py` | EP low_latency |
| `test_npu_deepep_low_latency_deepseek_v2.py` | EP low_latency |
| `test_npu_deepep_low_latency_qwen3_235b_a22b_w8a8.py` | EP low_latency |
| `test_npu_deepep_low_latency_qwen3_30b_a3b_w8a8.py` | EP low_latency |
| `test_npu_bucket_adjust_interval_secs_concurrency.py` | 并发调整 |

---

## 二、缺口逐一分析：测试仓覆盖情况

### 缺口 A: DP Attention (--enable-dp-attention)

**社区仓**: 0 个 PR 测试，仅 nightly 有（disabled 的 perf 测试）
**测试仓**: ✅ 有

| 文件 | 路径 | Suite | 卡数 | 特性组合 |
|:-----|:-----|:-----|:---:|:-----|
| `test_npu_dp_attention.py` | `basic_function/dp_attn/` | `full-4-npu-a3` | 4 | **DP+TP**, DP+TP+ChunkedPrefill, DP+TP+PrefixCache, DP+TP+VLM |

**合入建议**: 从测试仓合入 → 社区仓 `test/registered/ascend/basic_function/dp_attn/`，降级 suite 从 `full-4-npu-a3` → `stage-b-test-4-npu-a3`

---

### 缺口 B: PD Disaggregation (--disaggregation-mode)

**社区仓**: 0 个 PR 测试，仅 3 个 disabled 的 nightly perf 测试
**测试仓**: ✅ 有

| 文件 | 路径 | Suite | 卡数 | 特性组合 |
|:-----|:-----|:-----|:---:|:-----|
| `test_npu_pd_disaggregation.py` | `basic_function/pd_disaggregation/` | `full-16-npu-a3` | 4 | **PD+TP+PrefixCache(HiCache)** |

**合入建议**: 从测试仓合入 → 社区仓 `test/registered/ascend/basic_function/pd_disaggregation/`，降级 suite 从 `full-16-npu-a3` → `stage-b-test-4-npu-a3`

---

### 缺口 C: Data Parallelism + Load Balance

**社区仓**: 0 个
**测试仓**: ✅ 有

| 文件 | 路径 | Suite | 卡数 | 特性组合 |
|:-----|:-----|:-----|:---:|:-----|
| `test_npu_load_balance_method.py` | `basic_function/parallel_strategy/data_parallelism/` | `full-16-npu-a3` | 16 | **DP+TP+Quant** (round_robin/auto/total_requests/total_tokens) |
| `test_npu_load_balance_method_pd_disaggregation.py` | `basic_function/parallel_strategy/data_parallelism/` | `full-16-npu-a3` | 16 | **PD+DP+TP+Quant** (PD 分离下的 DP 负载均衡) |

**合入建议**: 从测试仓合入 → 社区仓 `test/registered/ascend/basic_function/parallel_strategy/data_parallelism/`

---

### 缺口 D: Speculative Decoding 扩展

**社区仓**: 仅 `test_npu_eagle3.py`（nightly-1-npu-a3）
**测试仓**: ✅ 有 8 个文件（社区仓的 1 个 + 额外 7 个）

| 文件 | 特性组合 |
|:-----|:-----|
| `test_npu_basic_sanity.py` | SpecDec 基础 sanity |
| `test_npu_basic_sanity_eagle3.py` | SpecDec(EAGLE3) + sanity |
| `test_npu_eagle3.py` | SpecDec(EAGLE3) + OverlapSchedule（社区仓已有） |
| `test_npu_speculative_attention_mode.py` | SpecDec + attention mode |
| `test_npu_speculative_draft_attention_backend.py` | SpecDec + draft attention backend |
| `test_npu_speculative_moe_a2a_backend.py` | **SpecDec + EP(MoE A2A)** |
| `test_npu_speculative_multi_npu.py` | **SpecDec + 多卡** |
| `test_npu_speculative_token_map.py` | SpecDec + token map |

**合入建议**: 全部 7 个新文件合入 → 社区仓 `test/registered/ascend/basic_function/speculative_inference/`

---

### 缺口 E: EPLB (--enable-eplb)

**社区仓**: 0 个（`enable-eplb` 搜索结果为 0）
**测试仓**: ✅ 有

| 文件 | 路径 | Suite | 卡数 | 特性组合 |
|:-----|:-----|:-----|:---:|:-----|
| `test_npu_eplb_min_rebalancing_utilization_threshold.py` | `basic_function/parallel_strategy/expert_parallelism/` | `full-8-npu-a3` | 8 | **EP + TP + Quant + ChunkedPrefill + EPLB** |

**合入建议**: 从测试仓合入 → 社区仓，降级 suite 从 `full-8-npu-a3` → `stage-b-test-8-npu-a3`

---

### 缺口 F: OverlapSchedule (SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1)

**社区仓**: 仅在 nightly 性能测试中有，无 PR 测试
**测试仓**: coverage unchanged（speculative_inference 中的 `test_npu_eagle3.py` 已设置）

**合入建议**: 将 `test_npu_eagle3.py` 降级到 PR（同之前 P0 #1）

---

### 缺口 G: MultistreamMoE (SGLANG_NPU_USE_MULTI_STREAM=1)

**社区仓**: 仅在 `test_npu_deepep_auto_deepseek_v3_2_w8a8.py`（nightly-16-npu-a3）
**测试仓**: 同上，无新增

**合入建议**: 降级现有 `test_npu_deepep_auto_deepseek_v3_2_w8a8.py` 到 PR

---

### 缺口 H: CP + SpecDec + DP

**社区仓**: 无
**测试仓**: 无直接覆盖（`test_npu_qwen3_30b_attn_cp.py` 仅 CP+TP，无 SpecDec/DP）

**状态**: 仍需从 GPU 移植

---

### 缺口 I: PD + DP + SpecDec

**社区仓**: 无
**测试仓**: `test_npu_load_balance_method_pd_disaggregation.py` 覆盖了 PD+DP，但无 SpecDec

**状态**: PD+DP 被覆盖，PD+DP+SpecDec 仍需扩充

---

## 三、合入清单

### 🟢 从测试仓直接合入（零代码修改）

| # | 测试仓文件 | 目标社区仓路径 | 覆盖缺口 |
|---|-----------|--------------|:---|
| 1 | `test_npu_dp_attention.py` | `test/registered/ascend/basic_function/dp_attn/` | DP+TP |
| 2 | `test_npu_pd_disaggregation.py` | `test/registered/ascend/basic_function/pd_disaggregation/` | PD+TP |
| 3 | `test_npu_load_balance_method.py` | `test/registered/ascend/basic_function/parallel_strategy/data_parallelism/` | DP+TP+LoadBalance |
| 4 | `test_npu_load_balance_method_pd_disaggregation.py` | `test/registered/ascend/basic_function/parallel_strategy/data_parallelism/` | PD+DP+TP |
| 5 | `test_npu_eplb_min_rebalancing_utilization_threshold.py` | `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/` | EPLB+EP+TP |
| 6 | `test_npu_expert_distribution_recorder_mode.py` | `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/` | EP 扩展 |
| 7 | `test_npu_moe_dense_tp_size.py` | `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/` | MoE dense TP |
| 8 | `test_npu_moe_runner_backend.py` | `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/` | MoE backend |
| 9 | `test_npu_basic_sanity.py` | `test/registered/ascend/basic_function/speculative_inference/` | SpecDec sanity |
| 10 | `test_npu_basic_sanity_eagle3.py` | `test/registered/ascend/basic_function/speculative_inference/` | SpecDec(EAGLE3) sanity |
| 11 | `test_npu_speculative_attention_mode.py` | `test/registered/ascend/basic_function/speculative_inference/` | SpecDec + attention mode |
| 12 | `test_npu_speculative_draft_attention_backend.py` | `test/registered/ascend/basic_function/speculative_inference/` | SpecDec + draft backend |
| 13 | `test_npu_speculative_moe_a2a_backend.py` | `test/registered/ascend/basic_function/speculative_inference/` | SpecDec + EP(MoE) |
| 14 | `test_npu_speculative_multi_npu.py` | `test/registered/ascend/basic_function/speculative_inference/` | SpecDec + 多卡 |
| 15 | `test_npu_speculative_token_map.py` | `test/registered/ascend/basic_function/speculative_inference/` | SpecDec + token map |
| 16 | `test_npu_bucket_adjust_interval_secs_concurrency.py` | `test/registered/ascend/basic_function/parallel_strategy/` | 并发调整 |

### 🟡 合入后需降级 suite 到 PR

| # | 文件 | 当前 suite | 建议 PR suite | 卡数 |
|---|------|-----------|--------------|:---:|
| 17 | `test_npu_dp_attention.py` | `full-4-npu-a3` | `stage-b-test-4-npu-a3` | 4 |
| 18 | `test_npu_pd_disaggregation.py` | `full-16-npu-a3` | `stage-b-test-4-npu-a3` | 4 |
| 19 | `test_npu_eplb_min_rebalancing_utilization_threshold.py` | `full-8-npu-a3` | `stage-b-test-8-npu-a3` | 8 |
| 20 | `test_npu_eagle3.py` (社区仓已有) | `nightly-1-npu-a3` | `stage-b-test-1-npu-a2` | 1 |

### 🔴 测试仓也无覆盖，仍需从 GPU 移植

| # | 缺口 | GPU 参考文件 | 卡数 |
|---|------|------------|:---:|
| 21 | CP + SpecDec + DP | `test/registered/cp/test_deepseek_v32_cp_single_node.py` | 8 |
| 22 | PD + DP + SpecDec | `test/registered/disaggregation/test_disaggregation_dsv4.py` | 8 |

---

## 四、合入后覆盖对比

| 维度 | 社区仓当前 PR | + 合入测试仓 | + 降级 suite | + GPU 移植 |
|:-----|:---:|:---:|:---:|:---:|
| 单特性 (15) | 7/15 (47%) | 12/15 (80%) | **14/15 (93%)** | **14/15 (93%)** |
| DP Attention 测试 | 0 | 1 | 1 (PR) | 1 |
| PD 测试 | 0 | 1 | 1 (PR) | 1 |
| SpecDec 测试 | 0 (nightly only) | 8 | 8 | 8 |
| EPLB 测试 | 0 | 1 | 1 (PR) | 1 |
| DP+TP+ChunkedPrefill | 0 | 1 | 1 (PR) | 1 |
| PD+DP+TP | 0 | 1 | 1 (PR) | 1 |
| CP+SpecDec+DP | 0 | 0 | 0 | 1 |

### 仍需从 GPU 移植的（测试仓无覆盖）

| # | 缺口 | GPU 参考 | 原因 |
|---|------|---------|:-----|
| 1 | **CP + SpecDec + DP** | `test_deepseek_v32_cp_single_node.py` | 测试仓无 CP+SpecDec 组合 |
| 2 | **PD + DP + SpecDec** | `test_disaggregation_dsv4.py` | 测试仓的 PD+DP 无 SpecDec |
| 3 | **CP + DPLMHead** | `test_pp_parallel_compat.py` | 双边都无覆盖 |
| 4 | **EPLB + PD** | - | 双边都无覆盖 |

---

## 五、执行步骤

```
步骤 1: 从测试仓 testcases 分支合入 16 个文件到社区仓 main
  → 目标: test/registered/ascend/basic_function/{dp_attn,pd_disaggregation,speculative_inference,...}/

步骤 2: 修改 4 个文件的 register_npu_ci，降级 suite 到 PR
  → test_npu_dp_attention.py: full-4-npu-a3 → stage-b-test-4-npu-a3
  → test_npu_pd_disaggregation.py: full-16-npu-a3 → stage-b-test-4-npu-a3
  → test_npu_eplb_min_rebalancing_utilization_threshold.py: full-8-npu-a3 → stage-b-test-8-npu-a3
  → test_npu_eagle3.py: nightly-1-npu-a3 → stage-b-test-1-npu-a2

步骤 3: 从 GPU 移植 2 个文件做 NPU 适配
  → test_deepseek_v32_cp_single_node.py → test_npu_cp_specdec.py
  → test_disaggregation_dsv4.py → test_npu_pd_dp_spec.py
```
