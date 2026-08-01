# NPU PR 特性缺口补齐方案

> 分析日期: 2026-07-25 | 验证: ✅ 12/12 项交叉验证通过
> 目标: 对每个 NPU vs GPU 特性覆盖缺口，给出具体文件级的补齐方案

---

## 分类说明

| 标记 | 含义 | 行动 |
|:---:|:-----|:-----|
| 🟢 | NPU 有现成测试，可直接降级到 PR | 修改 `register_npu_ci` 的 suite 参数 |
| 🟡 | NPU 有但需修改（模型太大/已disabled） | 简化配置或新建轻量版本 |
| 🔴 | NPU 完全没有，需从 GPU 移植 | 参考 GPU 文件，新建 NPU 适配版 |

---

## 一、🟢 可降级到 PR 的现成 NPU 测试

这些测试已在 nightly 中运行且验证通过，只需修改 `register_npu_ci` 添加到 PR suite。**零新代码成本**。

| # | 缺口 | NPU 现成文件 | 当前 suite | 建议 PR suite | 卡数 | 覆盖特性 |
|---|------|------------|-----------|--------------|:---:|:-----|
| 1 | SpecDec + OverlapSchedule | `.../speculative_inference/test_npu_eagle3.py` | `nightly-1-npu-a3` | `stage-b-test-1-npu-a2` | 1 | SpecDec(EAGLE3) + OverlapSchedule |
| 2 | CP + TP | `.../llm_models/test_npu_qwen3_30b_attn_cp.py` | `nightly-4-npu-a3` | `stage-b-test-4-npu-a3` | 4 | CP + TP + EP(moe-dp) |
| 3 | EP + DP + DPLMHead | `.../expert_parallelism/test_npu_deepep_auto_qwen3_480b.py` | `nightly-16-npu-a3` | `stage-b-test-16-npu-a3` | 16 | EP + DP(4) + DPLMHead + TP(16) + Quant + ChunkedPrefill + NPUGraph |
| 4 | EP + DP + DPLMHead (low_latency) | `.../expert_parallelism/test_npu_deepep_low_latency_qwen3_480b.py` | `nightly-16-npu-a3` | `stage-b-test-16-npu-a3` | 16 | EP + DP(4) + DPLMHead + TP(16) + deepep low_latency |
| 5 | MultistreamMoE (e2e) | `.../expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py` | `nightly-16-npu-a3` | `stage-b-test-16-npu-a3` | 16 | EP + MultistreamMoE(SGLANG_NPU_USE_MULTI_STREAM=1) |

### 降级操作示例

以 `test_npu_eagle3.py` 为例，只需增加一行注册：

```python
# 修改前
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

# 修改后 (新增 PR 注册，保留 nightly)
register_npu_ci(est_time=400, suite="stage-b-test-1-npu-a2", nightly=False)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)
```

### ✅ 验证结果

| 验证项 | 结果 |
|:-------|:----:|
| `test_npu_eagle3.py` suite=`nightly-1-npu-a3`, SpecDec=EAGLE3, OverlapSchedule=1 | ✅ |
| `test_npu_qwen3_30b_attn_cp.py` suite=`nightly-4-npu-a3`, attn-cp-size=2, enable-prefill-context-parallel | ✅ |
| `test_npu_deepep_auto_qwen3_480b.py` suite=`nightly-16-npu-a3`, enable-dp-attention, enable-dp-lm-head | ✅ |
| `test_npu_deepep_low_latency_qwen3_480b.py` suite=`nightly-16-npu-a3`, deepep-mode=low_latency, enable-dp-attention | ✅ |
| `test_npu_deepep_auto_deepseek_v3_2_w8a8.py` suite=`nightly-16-npu-a3`, SGLANG_NPU_USE_MULTI_STREAM=1 | ✅ |

---

## 二、🟡 NPU 有但需新建轻量版

### 缺口 B: PD Disaggregation 基础功能测试

**现状**: 3 个 disabled nightly 性能测试有 `--disaggregation-mode`，但 suite=""、disabled="performance testcase"，且均为多节点大模型

**方案**: 新建单节点 PD 基础功能测试

| 项目 | 内容 |
|:-----|:-----|
| **GPU 参考** | `test/registered/disaggregation/test_disaggregation_basic.py` (2-GPU, PDDisaggregationServerBase) |
| **NPU 新建** | `test/registered/ascend/basic_function/disaggregation/test_npu_pd_basic.py` |
| **建议配置** | 4 卡、Qwen3-8B、PD 单节点、`--disaggregation-transfer-backend ascend` |
| **覆盖特性** | PD + TP |
| **建议 PR suite** | `stage-b-test-4-npu-a3` |

### 缺口 C/G 扩展: SpecDec + DP + DPLMHead 轻量测试

**现状**: 降级 #1 覆盖了 SpecDec + OverlapSchedule，但无 DP+DPLMHead 组合的轻量测试

**方案**: 基于 `test_npu_eagle3.py` 新建带 DP Attention 的版本

| 项目 | 内容 |
|:-----|:-----|
| **GPU 参考** | `test/registered/spec/eagle/test_eagle_dp_attention.py` (4-GPU, TP=2/DP=2, EAGLE3, enable-dp-attention, enable-dp-lm-head) |
| **NPU 新建** | `test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3_dp.py` |
| **建议配置** | 4 卡、TP=2、DP=2、enable-dp-attention、enable-dp-lm-head、EAGLE3 |
| **覆盖特性** | SpecDec + DP + TP + DPLMHead |
| **建议 PR suite** | `stage-b-test-4-npu-a3` |

### 缺口 G 扩展: DP Attention + DPLMHead 独立功能测试

**方案**: 新建不含 SpecDec 的纯 DP Attention 功能测试

| 项目 | 内容 |
|:-----|:-----|
| **GPU 参考** | `test/registered/dp_attn/test_dp_attention.py` (4-GPU, TP=2/DP=2, enable-dp-attention) |
| **NPU 新建** | `test/registered/ascend/basic_function/parallel_strategy/test_npu_dp_attention.py` |
| **建议配置** | 4 卡、TP=2、DP=2、enable-dp-attention、enable-dp-lm-head |
| **覆盖特性** | DP + TP + DPLMHead |
| **建议 PR suite** | `stage-b-test-4-npu-a3` |

### ✅ 验证结果

| 验证项 | 结果 |
|:-------|:----:|
| `test_disaggregation_basic.py` 存在，使用 PDDisaggregationServerBase，2-GPU 配置 | ✅ |
| `test_eagle_dp_attention.py` 存在，EAGLE3 + enable-dp-attention + enable-dp-lm-head, TP=2/DP=2 | ✅ |
| `test_dp_attention.py` 存在，enable-dp-attention, TP=2/DP=2 | ✅ |
| NPU `disaggregation-mode` 仅存在于 3 个 disabled 性能测试（suite=""）| ✅ |

---

## 三、🔴 NPU 完全没有，需从 GPU 移植

### 缺口 E: EPLB

**现状**: 整个 `test/registered/ascend/` 目录中 `enable-eplb` 搜索结果 = 0

**需移植的 GPU 文件**:

| 优先级 | GPU 文件 | 覆盖组合 | 卡数 | NPU 适配要点 |
|:---:|:-----|:-----|:---:|:-----|
| P1 | `test/registered/eplb/test_lplb_distributed.py` | EPLB (LPLB solver) | 2 | 纯逻辑测试，不涉及 attention backend，适配成本低 |
| P2 | `test/registered/ep/test_mooncake_ep_small.py` | EP + EPLB + DP + DPLMHead + ChunkedPrefill | 4 | 需替换 Mooncake EP → Ascend DeepEP/ascend_fuseep |

### 缺口 D 扩展: CP + SpecDec + DP 组合

**现状**: 降级 #2 覆盖了 CP + TP，但无 CP+SpecDec+DP 的组合

**需移植的 GPU 文件**:

| 优先级 | GPU 文件 | 覆盖组合 | 卡数 |
|:---:|:-----|:-----|:---:|
| P1 | `test/registered/cp/test_deepseek_v32_cp_single_node.py` (InSeqSplit) | CP + TP + DP + SpecDec(EAGLE) | 8 |

### 缺口 PD 扩展: PD + DP

**需移植的 GPU 文件**:

| 优先级 | GPU 文件 | 覆盖组合 | 卡数 |
|:---:|:-----|:-----|:---:|
| P2 | `test/registered/disaggregation/test_disaggregation_dp_attention.py` | PD + DP + TP | 8 |

### ✅ 验证结果

| 验证项 | 结果 |
|:-------|:----:|
| `test_lplb_distributed.py` 存在，NUM_GPUS=2，EPLB 分布式测试 | ✅ |
| `test_mooncake_ep_small.py` 存在，--enable-eplb + mooncake EP + DPLMHead | ✅ |
| `test_deepseek_v32_cp_single_node.py` 存在，CP + SpecDec + DP Attention, TP=8/DP=2/CP=4 | ✅ |
| `test_disaggregation_dp_attention.py` 存在，PD + DP + TP | ✅ |
| NPU 目录 `enable-eplb` 搜索结果 = 0 | ✅ |

---

## 四、完整补齐清单

### P0 — 立即执行（仅改注册行，零新代码）

| # | 行动 | NPU 文件 | 卡数 | 新增覆盖 |
|---|------|---------|:---:|:-----|
| 1 | 降级 | `test_npu_eagle3.py` | 1 | SpecDec + OverlapSchedule |
| 2 | 降级 | `test_npu_qwen3_30b_attn_cp.py` | 4 | CP + TP |
| 3 | 降级 | `test_npu_deepep_auto_qwen3_480b.py` | 16 | EP + DP + DPLMHead + Quant + ChunkedPrefill + NPUGraph |
| 4 | 降级 | `test_npu_deepep_low_latency_qwen3_480b.py` | 16 | EP + DP + DPLMHead + deepep low_latency |
| 5 | 降级 | `test_npu_deepep_auto_deepseek_v3_2_w8a8.py` | 16 | MultistreamMoE (e2e) |

### P1 — 短期（新建轻量测试，参考 GPU）

| # | 行动 | GPU 参考 | NPU 新建 | 卡数 | 新增覆盖 |
|---|------|---------|---------|:---:|:-----|
| 6 | 新建 | `test_disaggregation_basic.py` | `test_npu_pd_basic.py` | 4 | PD + TP |
| 7 | 新建 | `test_eagle_dp_attention.py` | `test_npu_eagle3_dp.py` | 4 | SpecDec + DP + TP + DPLMHead |
| 8 | 新建 | `test_dp_attention.py` | `test_npu_dp_attention.py` | 4 | DP + TP + DPLMHead |
| 9 | 移植 | `test_lplb_distributed.py` | `test_npu_lplb.py` | 2 | EPLB |
| 10 | 移植 | `test_deepseek_v32_cp_single_node.py` | `test_npu_cp_specdec.py` | 8 | CP + TP + DP + SpecDec |

### P2 — 中期（扩展覆盖深度）

| # | 行动 | GPU 参考 | NPU 新建 | 卡数 | 新增覆盖 |
|---|------|---------|---------|:---:|:-----|
| 11 | 移植 | `test_disaggregation_dp_attention.py` | `test_npu_pd_dp.py` | 8 | PD + DP + TP |
| 12 | 移植 | `test_mooncake_ep_small.py` | `test_npu_eplb_ep.py` | 4 | EP + EPLB + DP + ChunkedPrefill |

---

## 五、补齐前后覆盖对比

| 维度 | GPU | NPU 当前 PR | P0 后 | P0+P1 后 | P0+P1+P2 后 |
|:-----|:---:|:---:|:---:|:---:|:---:|
| 单特性 (15) | 14/15 | 7/15 | **12/15** | **14/15** | **14/15** |
| 核心组合 (30+) | ~45 | ~5 | ~15 | ~30 | ~35 |
| 3+ 特性叠加 | 20+ | 2 | 7 | 12 | 15+ |
| PD 测试 | 20+ | 0 | 0 | 1 | 2 |
| SpecDec 测试 | 30+ | 0 | 1 | 2 | 2 |
| DP Attention 测试 | 116+ | 0 | 3 | 5 | 6 |
| EPLB 测试 | 10+ | 0 | 0 | 1 | 2 |
| CP 测试 | 15+ | 0 | 1 | 2 | 2 |

### 剩余无法覆盖

| 缺口 | 原因 |
|:-----|:-----|
| MultistreamMoE 轻量测试 | Ascend 专有，需 16 卡大模型，P0 #5 已覆盖 e2e |
| CP + DPLMHead | GPU 侧也无覆盖（兼容性标记 ❔） |
| EPLB + PD | GPU 侧也无覆盖（兼容性标记 🟠） |
| OverlapSchedule + ChunkedPrefill | GPU 侧也无覆盖（兼容性标记 🟠） |
