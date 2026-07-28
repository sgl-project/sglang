# GLM-5.2 Support in SGLang - Source Analysis v2

## 1. Current Repository Version

- Commit: `e4976683f4` — `[Docs] Fix broken links in cookbook (#29261)`
- Branch: `main`
- Status: clean working tree (only untracked files: `CLAUDE.md`, `docs/glm52_analysis/`, `protoc-25.3-linux-x86_64.zip`)

## 2. Executive Summary

**GLM-5.2 在当前 SGLang 仓库中仍然没有被显式引用。** 搜索 `glm-5.2`、`glm52`、`GLM-5.2`、`glm5.2` 等所有变体，在 `python/sglang/`、`test/`、`docs/`、`examples/` 中均无命中（仅旧报告文件自身命中）。

然而，GLM-5 和 GLM-5.1 有完整的代码路径支持。核心架构类 `GlmMoeDsaForCausalLM` 继承自 `DeepseekV2ForCausalLM`，复用 DeepSeek V3.2 的 DSA 稀疏注意力、MTP 多 token 预测、权重加载和 MoE 基础设施。如果 GLM-5.2 沿用 `GlmMoeDsaForCausalLM` 架构且 config 中包含 `index_topk` 字段，则可被现有代码路径隐式支持——但代码中没有任何针对 "GLM-5.2" 的特殊处理。

## 3. Search Results: What Changed After git pull

与旧报告相比，当前仓库状态的核心结论未变：

- **GLM-5.2 字符串**：仍未出现在任何源代码、测试或文档中。
- **GLM-5 / GLM-5.1**：继续受到完整支持，测试覆盖广泛。
- **新增/变化**：DSA 后端代码有更新（如 `dsa_backend.py` 中新增 AIter 相关逻辑、HiSparse 支持、prefill CP 支持等），但无 GLM-5.2 专属代码。
- **测试文件**：测试目录中 GLM-5 相关测试数量丰富，包括 FP8、NVFP4、MXFP4、DSA+MTP、HiSparse 等多种配置。

## 4. Model Registration and Class Mapping

### 4.1 Model Registry 机制

模型注册通过 `python/sglang/srt/models/registry.py` 中的 `ModelRegistry` 实现：
- `import_model_classes()` (line 95) 扫描 `sglang.srt.models` 包下所有模块。
- 每个模块通过 `EntryClass` 变量声明可注册的模型类 (line 111-125)。
- `resolve_model_cls()` (line 80) 根据 HF config 的 `architectures` 字段查找对应类。

### 4.2 GLM 相关 EntryClass

| 文件 | EntryClass | 说明 |
|------|-----------|------|
| `python/sglang/srt/models/glm4_moe.py:1482` | `[Glm4MoeForCausalLM, GlmMoeDsaForCausalLM]` | GLM-4.5/4.6/4.7 及 GLM-5 DSA |
| `python/sglang/srt/models/glm4_moe_nextn.py:168` | `[Glm4MoeForCausalLMNextN]` | GLM-4.x MTP draft model |
| `python/sglang/srt/models/glm4_moe_lite.py:1302` | `[Glm4MoeLiteForCausalLM]` | GLM-4.x Lite 变体 |
| `python/sglang/srt/models/glm4_moe_lite_nextn.py:182` | `[Glm4MoeLiteForCausalLMNextN]` | GLM-4.x Lite MTP draft |
| `python/sglang/srt/models/deepseek_v2.py:2941` | `[DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]` | DeepSeek V2/V3/V3.2（GlmMoeDsaForCausalLM 的父类） |

### 4.3 DSA 模型检测

`python/sglang/srt/configs/model_config.py:103` — `is_deepseek_dsa()` 函数检测以下架构并要求 `index_topk` 非空：
- `DeepseekV3ForCausalLM`
- `DeepseekV32ForCausalLM`
- `DeepseekV3ForCausalLMNextN`
- `MistralLarge3ForCausalLM`
- `PixtralForConditionalGeneration`
- `GlmMoeDsaForCausalLM`

### 4.4 Draft Model (MTP) 架构映射

`python/sglang/srt/configs/model_config.py:526-541` — `_config_draft_model()` 将主模型架构映射到 NextN draft 架构：
- `GlmMoeDsaForCausalLM` → `DeepseekV3ForCausalLMNextN` (line 529-531)
- `Glm4MoeForCausalLM` → `Glm4MoeForCausalLMNextN` (line 540-541)
- `Glm4MoeLiteForCausalLM` → `Glm4MoeLiteForCausalLMNextN` (line 543-546)

## 5. Main Model Architecture

### 5.1 GlmMoeDsaForCausalLM（GLM-5 架构）

定义于 `python/sglang/srt/models/glm4_moe.py:1477`：

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    def determine_num_fused_shared_experts(self):
        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")
```

这是一个极简子类，仅重写 `determine_num_fused_shared_experts`，其余全部继承自 `DeepseekV2ForCausalLM`。

### 5.2 DeepseekV2ForCausalLM（父类）

定义于 `python/sglang/srt/models/deepseek_v2.py:2634`，继承 `nn.Module` 和 `DeepseekV2WeightLoaderMixin`。

核心组件：
- **Decoder Layer**: `DeepseekV2DecoderLayer` (line 2014) — 支持 sparse/dense 层切换、DSA prefill CP。
- **Attention**: `DeepseekV2AttentionMLA` (line 1541) — MLA (Multi-head Latent Attention) 实现，支持 q_lora_rank、kv_lora_rank、DSA。
- **MoE**: `DeepseekV2MoE` (line 531) 和 `MoEGate` (line 421) — grouped top-k routing，注释明确标注覆盖 "V3/V3.2/GLM-5/Glm4MoeLite" (line 652)。
- **Model**: `DeepseekV2Model` (line 2317) — 包含 `use_dsa = is_deepseek_dsa(config)` (line 2328)。
- **RMSNorm**: 来自 `sglang.srt.layers.layernorm.RMSNorm`。

### 5.3 Glm4MoeForCausalLM（GLM-4.5/4.6/4.7 架构）

定义于 `python/sglang/srt/models/glm4_moe.py:1167`，独立于 DeepSeek V2，拥有自己的：
- `Glm4MoeAttention` (line 180) — 标准 GQA 注意力，使用 `QKVParallelLinear`。
- `Glm4MoeMLP` (line 122) — dense MLP。
- `Glm4MoeSparseMoeBlock` (line 391) — sparse MoE。
- `Glm4MoeDecoderLayer` (line 785) — decoder layer。
- `Glm4MoeModel` (line 1041) — model body。
- `determine_num_fused_shared_experts()` (line 1196) — shared experts 融合逻辑，含硬件能力检测。

### 5.4 DeepseekV3ForCausalLM / DeepseekV32ForCausalLM

均定义于 `python/sglang/srt/models/deepseek_v2.py`：
- `DeepseekV3ForCausalLM` (line 2894) — 空类，直接继承 `DeepseekV2ForCausalLM`。
- `DeepseekV32ForCausalLM` (line 2898) — 空类，直接继承 `DeepseekV2ForCausalLM`。

## 6. DSA / MLA / Attention Backend

### 6.1 DSA 后端

`python/sglang/srt/layers/attention/dsa_backend.py` — `DeepseekSparseAttnBackend` 类 (line 304)。

- 初始化时检测 DSA：`self.use_dsa = is_deepseek_dsa(model_runner.model_config.hf_config)` (line 322)。
- 读取 `index_topk`：`self.dsa_index_topk = get_dsa_index_topk(...)` (line 327)。
- 支持 prefill 实现：flashmla / triton / aiter / trtllm (line 344-347)。
- 支持 decode 实现：同上。
- MTP precompute mixin：`DeepseekSparseAttnBackendMTPPrecomputeMixin` (line 24, import)。
- GLM-5.1 在 AMD gfx950 上的验证：line 62 注释提及 `GLM-5.1 @ TP4`。
- GLM-5 head padding：line 375 注释提及 `GLM-5 64 heads / TP8 = 8` 需要 pad 到 16。

### 6.2 DSA 后端注册

`python/sglang/srt/layers/attention/attention_registry.py:111-115`：
```python
@register_attention_backend("dsa")
def create_dsa_backend(runner):
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    return DeepseekSparseAttnBackend(runner)
```
`nsa` 是 `dsa` 的已弃用别名 (line 118-126)。

### 6.3 DSA 相关函数

- `is_deepseek_dsa()` — `python/sglang/srt/configs/model_config.py:103`
- `get_dsa_index_topk()` — `python/sglang/srt/configs/model_config.py:175`
- `dsa_layer_skips_topk()` — `python/sglang/srt/configs/model_config.py:180` — 支持 `index_topk_pattern` 和 `index_topk_freq`。
- `get_dsa_index_n_heads()` — `python/sglang/srt/configs/model_config.py:203`
- `get_dsa_index_head_dim()` — `python/sglang/srt/configs/model_config.py:125`

### 6.4 DSA 后端选择

`python/sglang/srt/server_args.py:3603-3612` — 当 `model_arch` 在 DSA 模型列表中（包含 `GlmMoeDsaForCausalLM`）且 `is_deepseek_dsa(hf_config)` 为真时，自动配置 DSA 相关参数。

### 6.5 MLA 架构配置

`python/sglang/srt/configs/model_config.py:726-756` — 当架构为 DSA 模型（含 `GlmMoeDsaForCausalLM`）时，设置 `attention_arch = AttentionArch.MLA`，读取 `kv_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim`、`index_head_dim`。

## 7. MTP / NextN / Speculative Decoding

### 7.1 MTP Draft Model 架构

- `DeepseekV3ForCausalLMNextN` — `python/sglang/srt/models/deepseek_nextn.py:261`，继承 `DeepseekV3ForCausalLM`。GLM-5 的 draft model 使用此架构（通过 `_config_draft_model` 映射）。
- `Glm4MoeForCausalLMNextN` — `python/sglang/srt/models/glm4_moe_nextn.py:119`，继承 `Glm4MoeForCausalLM`。GLM-4.x 的 draft model 使用此架构。

### 7.2 DSA + MTP 交互

`python/sglang/srt/speculative/draft_utils.py:107-116` — 在 DSA 模型上启用 speculative decoding 时，导入 `DeepseekSparseAttnBackend` 相关组件。

`python/sglang/srt/models/deepseek_nextn.py:27` — `is_deepseek_dsa` 用于判断是否在 DSA 模式下运行 NextN。

### 7.3 EAGLE3 支持

`python/sglang/srt/models/glm4_moe.py:1462` — `set_eagle3_layers_to_capture()` 方法用于 EAGLE3 speculative decoding。
`python/sglang/srt/models/deepseek_v2.py:2864` — 同名方法在 DeepseekV2ForCausalLM 中。

## 8. Weight Loading

### 8.1 DeepseekV2WeightLoaderMixin

定义于 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:96`。

- `do_load_weights()` (line 105) — 核心权重加载方法，处理 stacked params mapping (`gate_up_proj`)、expert params mapping、qkv fusion。
- `load_weights()` — `DeepseekV2ForCausalLM.load_weights()` (line 2842) 调用 `self.do_load_weights()`。
- `GlmMoeDsaForCausalLM` 继承此 mixin，无需自定义权重加载逻辑。

### 8.2 GLM-4.x 权重加载

`Glm4MoeForCausalLM.load_weights()` — `python/sglang/srt/models/glm4_moe.py:1264`，有独立的权重加载实现。

### 8.3 量化权重

- FP8：`DeepseekV2MoE` 中支持 FP8 fused clamp + deep_gemm (line 328-379 of `deepseek_v2.py`)。
- NVFP4：`_maybe_quant_weights_to_fp8_ue8m0` (line 118 of `deepseek_weight_loader.py`)，用于 NVFP4 checkpoint 的 FP8 注意力量化。
- W4AFP8：`glm4_moe.py:1216` 检测 `w4afp8` 量化配置。
- MXFP4：DeepSeek V3 NextN 支持amd MXFP4 重命名 (line 263-269 of `deepseek_nextn.py`)。

## 9. Runtime Inference Call Chain

推理调用链（高层）：

1. **Request** → HTTP API → `Scheduler.event_loop_normal()` (`scheduler.py:1516`)
2. **Scheduler** → `Scheduler.run_batch()` (`scheduler.py:3185`) → 组装 `ForwardBatch`
3. **ModelRunner** → `ModelRunner.forward()` (`model_runner.py:2915`)
4. **Model forward** → `GlmMoeDsaForCausalLM.forward()` (继承自 `DeepseekV2ForCausalLM`)
5. **Model body** → `DeepseekV2Model.forward()` → 遍历 `DeepseekV2DecoderLayer`
6. **Decoder Layer** → `DeepseekV2DecoderLayer.forward()` → attention + MoE/MLP + RMSNorm
7. **Attention** → `DeepseekV2AttentionMLA` → `RadixAttention` → `DeepseekSparseAttnBackend`（DSA 模型）
8. **MoE** → `DeepseekV2MoE` → `MoEGate` routing + expert dispatch
9. **LM Head** → `ParallelLMHead` → `LogitsProcessor`
10. **Sampling** → sampler → output tokens

## 10. Quantization / FP8 / NVFP4 / TP / EP / MoE

### 10.1 量化支持

| 量化格式 | 支持情况 | 关键文件 |
|----------|---------|---------|
| FP8 | 完整支持 | `deepseek_v2.py` (fused clamp, deep_gemm), `glm4_moe.py` (per-token quant) |
| NVFP4 | 支持 | `deepseek_weight_loader.py:118`, 测试 `test_pcg_glm5_fp4.py`, `test_glm5_nvfp4.py` |
| W4AFP8 | GLM-4.x 支持 | `glm4_moe.py:1216` |
| MXFP4 | DeepSeek/AMD 支持 | `deepseek_nextn.py:263`, AMD 测试 `test_glm5_mxfp4_*` |

### 10.2 Tensor Parallel / Expert Parallel

- `Glm4MoeForCausalLM` 中 TP 用于 attention 和 dense MLP (line 205-219)。
- `DeepseekV2MoE` 支持 `moe_ep_size` (line 403 of `deepseek_v2.py`)。
- Shared experts fusion 在 EP > 1 时有限制 (line 1208-1215 of `glm4_moe.py`)。
- DeepEP / Mori all-to-all 后端与 shared experts fusion 不兼容 (line 1213-1215)。

### 10.3 AllReduce Fusion

`python/sglang/srt/server_args.py:4396-4409` — `GlmMoeDsaForCausalLM` 在 flashinfer allreduce fusion backend 支持列表中。

### 10.4 硬件特定

- AMD gfx950：DSA Triton prefill 路径 (line 62-65 of `dsa_backend.py`)。
- AMD AIter：DSA decode 路径含 head padding 逻辑 (line 375-397)。
- Ascend NPU：GLM-5 支持文档见 `docs/platforms/ascend/ascend_npu_glm5_examples.md`。
- GB300：有 GLM-5 FP8 和 NVFP4 测试 (line 13, 11 of `test_glm5_fp8.py`, `test_glm5_nvfp4.py`)。

## 11. Tokenizer / Reasoning Parser / Tool Parser

### 11.1 Reasoning Parser

`python/sglang/srt/parser/reasoning_parser.py:1073`：
- `"glm45"` → `Glm45Detector` (line 338) — 用于 GLM-4.5/4.6/5/5.1 的 reasoning 解析。

### 11.2 Tool Call Parser

`python/sglang/srt/function_call/function_call_parser.py:66-68`：
- `"glm"` / `"glm45"` → `Glm4MoeDetector` — GLM-4.5/4.6 tool call 解析。
- `"glm47"` → `Glm47MoeDetector` — GLM-4.7/5 tool call 解析 (`glm47_moe_detector.py:165`)。

### 11.3 Template Detection

`python/sglang/srt/managers/template_detection.py`：
- `_is_glm45()` (line 193) — 检测 GLM-4.5/4.6 模板特征。
- `_is_glm47()` (line 207) — 在 `_is_glm45` 基础上增加 GLM-4.7 模板特征。
- `TOOL_CALL_PARSER_RULES` (line 367) 和 reasoning parser rules (line 331) 中均注册了 GLM 规则。

### 11.4 Chat Template / 文档建议

`docs/basic_usage/glm45.md:38` — GLM-4.7 使用 `--tool-call-parser glm47`，GLM-4.5/4.6 使用 `--tool-call-parser glm45`。
`docs/basic_usage/deepseek_v32.md:134-137` — GLM-5 使用 `--reasoning-parser glm45 --tool-call-parser glm47`。

## 12. Tests and Docs

### 12.1 测试文件

| 测试文件 | 模型 | 说明 |
|----------|------|------|
| `test/registered/models_e2e/test_dsa_glm5_tp_mtp.py` | `zai-org/GLM-5-FP8` | DSA + TP + MTP |
| `test/registered/models_e2e/test_dsa_glm5_dp_mtp.py` | `zai-org/GLM-5-FP8` | DSA + DP + MTP |
| `test/registered/models_e2e/test_dsa_glm5_hisparse.py` | `zai-org/GLM-5-FP8` | DSA + HiSparse |
| `test/registered/8-gpu-models/test_glm_51_fp8.py` | `zai-org/GLM-5.1-FP8` | GLM-5.1 FP8, 8-GPU |
| `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp4.py` | `nvidia/GLM-5-NVFP4` | PCG + NVFP4 |
| `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py` | `zai-org/GLM-5-FP8` | PCG + FP8 TP8 |
| `test/registered/gb300/test_glm5_fp8.py` | `zai-org/GLM-5.1-FP8` | GB300 FP8 |
| `test/registered/gb300/test_glm5_nvfp4.py` | `nvidia/GLM-5-NVFP4` | GB300 NVFP4 |
| `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` | GLM-5 | AMD MI35x 精度 |
| `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` | GLM-5 MXFP4 | AMD MI35x MXFP4 |
| `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` | GLM-5 | AMD MI35x 性能 |
| `test/registered/moe/test_glm4_moe_models.py` | GLM-4.x MoE | MoE 基础测试 |
| `test/registered/8-gpu-models/test_glm_46.py` | GLM-4.6 | GLM-4.6 测试 |
| `test/registered/stress/test_stress_glm_4_6.py` | GLM-4.6 | 压力测试 |

### 12.2 文档

| 文档 | 说明 |
|------|------|
| `docs/basic_usage/glm45.md` | GLM-4.5/4.6/4.7 启动指南 |
| `docs/basic_usage/deepseek_v32.md` | DeepSeek V3.2 / GLM-5 共用指南，含 DSA/MTP 说明 |
| `docs/platforms/ascend/ascend_npu_glm5_examples.md` | Ascend NPU 上的 GLM-5 部署指南 |

## 13. What Is Implemented

- `GlmMoeDsaForCausalLM` 类（GLM-5 DSA 架构）— `python/sglang/srt/models/glm4_moe.py:1477`
- `Glm4MoeForCausalLM` 类（GLM-4.5/4.6/4.7 架构）— `python/sglang/srt/models/glm4_moe.py:1167`
- `Glm4MoeForCausalLMNextN`（MTP draft model）— `python/sglang/srt/models/glm4_moe_nextn.py:119`
- `Glm4MoeLiteForCausalLM`（Lite 变体）— `python/sglang/srt/models/glm4_moe_lite.py:898`
- DSA 稀疏注意力后端 — `python/sglang/srt/layers/attention/dsa_backend.py:304`
- DSA 模型检测与配置 — `python/sglang/srt/configs/model_config.py:103-222`
- `DeepseekV2WeightLoaderMixin` 权重加载 — `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:96`
- MLA 注意力实现 — `python/sglang/srt/models/deepseek_v2.py:1541`
- MoE routing（grouped top-k, shared experts fusion）— `python/sglang/srt/models/deepseek_v2.py:531,421`
- FP8 / NVFP4 / W4AFP8 / MXFP4 量化路径
- GLM-4.5/4.7 reasoning parser (`glm45`) — `python/sglang/srt/parser/reasoning_parser.py:1073`
- GLM-4.5/4.6 tool call parser (`glm45`/`glm`) 和 GLM-4.7/5 tool call parser (`glm47`)
- Template 自动检测 — `python/sglang/srt/managers/template_detection.py:193-210`
- EAGLE3 speculative decoding 支持
- AMD (MI30x/MI35x) 和 Ascend NPU 平台支持
- GLM-5 / GLM-5.1 的 E2E 测试（FP8, NVFP4, MXFP4, DSA+MTP, HiSparse）

## 14. What Is Not Found / Unclear

- **"GLM-5.2" 字符串在仓库中完全不存在** — 没有代码、配置、测试或文档引用此版本号。
- 没有 `GlmMoeDsaV2ForCausalLM` 或任何 GLM-5.2 专属类。
- 没有针对 GLM-5.2 的特殊 config 解析、权重映射或量化逻辑。
- 没有针对 GLM-5.2 的测试文件。
- 没有针对 GLM-5.2 的文档。
- GLM-5.2 是否需要新的 `index_topk_pattern` 或其他架构变更：无法从代码判断。
- GLM-5.2 是否沿用 `GlmMoeDsaForCausalLM` 架构：无法从代码确认。
- GLM-5.2 是否需要新的 tool call / reasoning parser：无法从代码确认。

## 15. Risks and Verification Checklist

### 风险

1. 如果 GLM-5.2 使用了新的 `architectures` 名称（非 `GlmMoeDsaForCausalLM`），当前代码将无法识别，会回退到 `TransformersForCausalLM`。
2. 如果 GLM-5.2 的 config 中缺少 `index_topk` 字段，`is_deepseek_dsa()` 将返回 False，不会启用 DSA 后端。
3. 如果 GLM-5.2 的权重命名规则有变，`DeepseekV2WeightLoaderMixin.do_load_weights()` 可能无法正确加载。

### 验证步骤

```bash
# 1. 确认 GLM-5.2 不在仓库中
grep -RIn "glm.5.2\|GLM-5.2\|glm52" python/sglang test docs examples

# 2. 确认 GlmMoeDsaForCausalLM 的注册
grep -n "EntryClass" python/sglang/srt/models/glm4_moe.py

# 3. 确认 DSA 检测逻辑
grep -n "is_deepseek_dsa\|GlmMoeDsaForCausalLM" python/sglang/srt/configs/model_config.py

# 4. 确认 DSA 后端
grep -n "class DeepseekSparseAttnBackend" python/sglang/srt/layers/attention/dsa_backend.py

# 5. 确认 GLM-5 测试存在
find test -iname "*glm5*" -o -iname "*glm_5*"
```

## 16. Evidence Matrix

| Feature | Evidence | Conclusion |
|---------|----------|------------|
| GLM-5.2 显式代码 | 搜索 `glm52`/`GLM-5.2` 零结果 | 不存在 |
| GLM-5 DSA 架构 | `glm4_moe.py:1477` `GlmMoeDsaForCausalLM` | 已实现 |
| GLM-5.1 支持 | `test_glm_51_fp8.py`, `test_glm5_fp8.py` (GB300) | 已测试 |
| DSA 注意力后端 | `dsa_backend.py:304` `DeepseekSparseAttnBackend` | 已实现 |
| MTP / NextN | `deepseek_nextn.py:261`, `glm4_moe_nextn.py:119` | 已实现 |
| 权重加载 | `deepseek_weight_loader.py:96` `DeepseekV2WeightLoaderMixin` | 已实现 |
| FP8 量化 | `deepseek_v2.py:328-379`, `glm4_moe.py:887-914` | 已实现 |
| NVFP4 量化 | `deepseek_weight_loader.py:118`, `test_pcg_glm5_fp4.py` | 已实现 |
| Tool Call Parser | `function_call_parser.py:66-68` (`glm45`, `glm47`) | 已实现 |
| Reasoning Parser | `reasoning_parser.py:1073` (`glm45`) | 已实现 |
| AMD 支持 | `dsa_backend.py:62,375`, `test_glm5_*_mi35x.py` | 已实现 |
| Ascend 支持 | `ascend_npu_glm5_examples.md` | 已实现 |
| GLM-5.2 测试 | 无 | 不存在 |
| GLM-5.2 文档 | 无 | 不存在 |

## 17. Appendix: Commands Used

```bash
git log -1 --oneline
git branch --show-current
git status --short

grep -RIn "GLM-5.2\|glm-5.2\|glm52\|GLM52\|GLM5.2\|glm5.2" python docs test examples
grep -RIn "GLM-5\|GLM 5\|glm5\|GlmMoe\|GlmMoeDsa\|glm4_moe" python/sglang test docs examples
grep -RIn "DSA\|dsa\|DeepseekV32\|DeepseekV3\|DeepseekV2\|index_topk\|MTP\|NextN\|EAGLE" python/sglang/srt/layers/attention/dsa_backend.py
grep -RIn "is_deepseek_dsa\|get_dsa_index_topk\|dsa_backend\|DsaBackend" python/sglang/srt
find python/sglang test docs examples -iname "*glm*" -o -iname "*chatglm*"
grep -RIn "class DeepseekV2ForCausalLM\|class DeepseekV2Model\|class DeepseekV2MLA\|class DeepseekV2DecoderLayer\|class DeepseekV2MoE" python/sglang/srt/models/deepseek_v2.py
grep -RIn "class Glm4MoeForCausalLM\|class Glm4MoeModel\|class Glm4MoeDecoderLayer\|class Glm4MoeMoE\|class Glm4MoeMLP" python/sglang/srt/models/glm4_moe.py
grep -RIn "DeepseekV2WeightLoaderMixin" python/sglang/srt
grep -RIn "EntryClass" python/sglang/srt/models/registry.py
grep -RIn "glm45\|glm47" python/sglang/srt/parser/reasoning_parser.py python/sglang/srt/function_call/function_call_parser.py
grep -RIn "glm\|GLM\|Glm" python/sglang/srt/managers/template_detection.py
grep -RIn "fp8\|nvfp4\|modelopt_fp4\|w4afp8\|mxfp4" python/sglang/srt/models/deepseek_v2.py python/sglang/srt/models/glm4_moe.py
grep -RIn "glm\|GLM\|Glm" test/registered/models_e2e/test_dsa_glm5_tp_mtp.py test/registered/8-gpu-models/test_glm_51_fp8.py
grep -RIn "glm\|GLM" docs/basic_usage/glm45.md docs/basic_usage/deepseek_v32.md docs/platforms/ascend/ascend_npu_glm5_examples.md
```
