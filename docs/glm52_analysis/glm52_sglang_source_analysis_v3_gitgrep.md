# GLM-5.2 Support in SGLang - v3 Git-tracked Source Analysis

## 1. Repository Version

- Commit: `e4976683f4` — `[Docs] Fix broken links in cookbook (#29261)`
- Branch: `main`
- Tracked files under `python/`, `test/`, `docs/`, `examples/`: 4864
- Working tree: clean (untracked: `CLAUDE.md`, `docs/glm52_analysis/`, `protoc-25.3-linux-x86_64.zip`)

## 2. Executive Summary

**GLM-5.2 在当前 Git-tracked SGLang 文件中已显式出现**，但仅在 `docs_new/` 目录下的文档和配置代码中。Python 源码（`python/sglang/`）中没有 "GLM-5.2" 字符串，也没有 GLM-5.2 专属类或函数。

具体而言：

- `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` 是完整的 GLM-5.2 部署指南，明确标注 GLM-5.2 使用 `glm_moe_dsa` 架构（即 `GlmMoeDsaForCausalLM`），含 78 层、256 experts（top-8）、1M 上下文、MTP。
- `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` 包含 GLM-5.2 的完整部署配置矩阵（FP8/BF16，H200/B200/GB300/B300，low-latency/balanced/high-throughput），含实测 benchmark 数据。
- `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` 包含 Ascend NPU 上 GLM-5.2 的部署脚本。
- `docs_new/docs/basic_usage/anthropic_api.mdx` 引用 GLM-5.2 作为 Anthropic API 兼容端点的示例模型。

Python 源码中，`GlmMoeDsaForCausalLM`（`python/sglang/srt/models/glm4_moe.py:1477`）是 GLM-5/5.1/5.2 共用的架构类，继承自 `DeepseekV2ForCausalLM`，复用 DSA、MTP、权重加载等全部基础设施。GLM-5.2 没有独立的模型类——文档明确指出其架构名为 `glm_moe_dsa`，即同一个 `GlmMoeDsaForCausalLM`。

## 3. Git grep Search Results

### 3.1 GLM-5.2 精确搜索

`git grep -n -i -E "GLM-5\.2|glm-5\.2|glm52|glm_52|GLM52|GLM5\.2|glm5\.2" -- python test docs examples`

在 `python/`、`test/`、`examples/` 中：**零命中**。

在 `docs_new/` 中（上述命令未覆盖，需单独搜索）：**大量命中**，涵盖 cookbook、配置、Ascend 文档、Anthropic API 文档。

### 3.2 GLM-5 / GlmMoeDsa 搜索

`git grep -n -i -E "GLM-5|GLM 5|glm5|glm_5|GlmMoe|GlmMoeDsa|glm4_moe" -- python test docs examples`

命中分布：
- `python/sglang/srt/configs/model_config.py` — `GlmMoeDsaForCausalLM` 出现在 DSA 检测、draft model 映射、MLA 配置中。
- `python/sglang/srt/models/glm4_moe.py:1477` — `GlmMoeDsaForCausalLM` 类定义。
- `python/sglang/srt/server_args.py` — DSA 后端选择、allreduce fusion、speculative 配置。
- `python/sglang/srt/arg_groups/speculative_hook.py` — MTP 默认参数。
- `python/sglang/srt/eplb/lplb_solver.py:44` — LPLB 支持模型列表。
- `python/sglang/srt/utils/hf_transformers/config.py:72` — GlmMoeDsaConfig 字段修复。
- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:577` — 排除 GlmMoeDsaForCausalLM 的特殊处理。
- 多个测试文件覆盖 GLM-5 和 GLM-5.1。

### 3.3 DSA / MLA / MTP / NextN 搜索

命中集中在 `dsa_backend.py`、`deepseek_v2.py`、`deepseek_nextn.py`、`model_config.py`、`server_args.py`、`attention_registry.py` 及多个文档。

## 4. Model Registration and Class Mapping

### 4.1 注册机制

`python/sglang/srt/models/registry.py:95-127` — `import_model_classes()` 扫描 `sglang.srt.models` 包，通过每个模块的 `EntryClass` 变量注册模型类。`resolve_model_cls()` 根据 HF config 的 `architectures` 字段查找。

### 4.2 GLM 相关 EntryClass

| 文件 | 行号 | EntryClass |
|------|------|------------|
| `python/sglang/srt/models/glm4_moe.py` | 1482 | `[Glm4MoeForCausalLM, GlmMoeDsaForCausalLM]` |
| `python/sglang/srt/models/glm4_moe_nextn.py` | 168 | `[Glm4MoeForCausalLMNextN]` |
| `python/sglang/srt/models/glm4_moe_lite.py` | 1302 | `[Glm4MoeLiteForCausalLM]` |
| `python/sglang/srt/models/glm4_moe_lite_nextn.py` | 182 | `[Glm4MoeLiteForCausalLMNextN]` |
| `python/sglang/srt/models/deepseek_v2.py` | 2941 | `[DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]` |
| `python/sglang/srt/models/deepseek_nextn.py` | 362 | `[DeepseekV3ForCausalLMNextN]` |

### 4.3 DSA 模型检测

`python/sglang/srt/configs/model_config.py:103-115` — `is_deepseek_dsa()` 检测架构名在 `GlmMoeDsaForCausalLM` 等中且 `index_topk` 非空。

### 4.4 Draft Model 映射

`python/sglang/srt/configs/model_config.py:526-531` — `GlmMoeDsaForCausalLM` 映射到 `DeepseekV3ForCausalLMNextN` 作为 MTP draft model。

### 4.5 GlmMoeDsaConfig 修复

`python/sglang/srt/utils/hf_transformers/config.py:70-88` — 当架构为 `GlmMoeDsaForCausalLM` 时，从 raw config.json 恢复 `qk_rope_head_dim`、`index_topk_freq` 等被 HuggingFace `GlmMoeDsaConfig` 丢弃的字段。

## 5. Main Model Architecture

### 5.1 GlmMoeDsaForCausalLM

`python/sglang/srt/models/glm4_moe.py:1477-1479`:

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    def determine_num_fused_shared_experts(self):
        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")
```

极简子类，仅重写 `determine_num_fused_shared_experts`。所有模型逻辑继承自 `DeepseekV2ForCausalLM`。

### 5.2 DeepseekV2ForCausalLM（父类）

`python/sglang/srt/models/deepseek_v2.py:2634` — 继承 `nn.Module` 和 `DeepseekV2WeightLoaderMixin`。

关键组件：
- `DeepseekV2Model` (line 2317) — model body，`use_dsa = is_deepseek_dsa(config)` (line 2328)。
- `DeepseekV2DecoderLayer` (line 2014) — decoder layer，支持 sparse/dense 切换。
- `DeepseekV2AttentionMLA` (line 1541) — MLA 注意力，支持 `kv_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim`、DSA。
- `DeepseekV2MoE` (line 531) + `MoEGate` (line 421) — grouped top-k routing，注释 line 652: "Covers V3/V3.2/GLM-5/Glm4MoeLite"。
- `DeepseekV3ForCausalLM` (line 2894) 和 `DeepseekV32ForCausalLM` (line 2898) — 空子类。

### 5.3 GLM-5.2 架构参数（来自文档）

`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx:65`:
- 78 transformer layers
- 256 routed experts, 8 active per token
- 1M context (1,048,576)
- 1 MTP layer
- top-2048 DSA indexer
- FP8 (`zai-org/GLM-5.2-FP8`) 和 BF16 (`zai-org/GLM-5.2`)

## 6. DSA / MLA / Attention Backend

### 6.1 DSA 后端

`python/sglang/srt/layers/attention/dsa_backend.py:304` — `DeepseekSparseAttnBackend`。

- `is_deepseek_dsa()` 检测 (line 322)
- `get_dsa_index_topk()` 读取 topk (line 327)
- Prefill 后端：`flashmla_sparse` / `triton` / `aiter` / `trtllm` (line 344)
- Decode 后端：同上 (line 347)
- MTP precompute：`DeepseekSparseAttnBackendMTPPrecomputeMixin` (line 24)
- AMD gfx950 验证注释 (line 62): "GLM-5.1 @ TP4"
- GLM-5 head padding (line 375): "GLM-5 64 heads / TP8 = 8"

### 6.2 后端注册

`python/sglang/srt/layers/attention/attention_registry.py:111-115` — `@register_attention_backend("dsa")` 创建 `DeepseekSparseAttnBackend`。`nsa` 是已弃用别名 (line 118)。

### 6.3 DSA 函数

- `is_deepseek_dsa()` — `model_config.py:103`
- `get_dsa_index_topk()` — `model_config.py:175`
- `dsa_layer_skips_topk()` — `model_config.py:180`，支持 `index_topk_pattern` / `index_topk_freq`
- `get_dsa_index_n_heads()` — `model_config.py:203`
- `get_dsa_index_head_dim()` — `model_config.py:125`

### 6.4 自动后端选择

`python/sglang/srt/server_args.py:3603-3612` — `GlmMoeDsaForCausalLM` 在 DSA 模型列表中，触发 DSA 参数配置。注释 "DeepSeek 3.2/GLM 5"。

### 6.5 MLA 配置

`python/sglang/srt/configs/model_config.py:726-756` — DSA 模型（含 `GlmMoeDsaForCausalLM`）设置 `AttentionArch.MLA`，读取 `kv_lora_rank` 等。

### 6.6 GLM-5.2 DSA 细节（来自文档）

`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx:95`:
- 自动选择 DSA 后端：`flashmla_sparse` prefill, `fa3` decode, `sgl-kernel` indexer topk
- 自动选择 KV cache dtype：`fp8_e4m3` on Blackwell, `bf16` on Hopper
- 支持 DSA prefill Context Parallelism (CP)，仅 Hopper 验证

## 7. MTP / NextN / Speculative Decoding

### 7.1 NextN 架构

- `DeepseekV3ForCausalLMNextN` — `python/sglang/srt/models/deepseek_nextn.py:261`，GLM-5/5.2 的 MTP draft model 通过 `_config_draft_model()` 映射使用此架构 (`model_config.py:529-531`)。
- `Glm4MoeForCausalLMNextN` — `python/sglang/srt/models/glm4_moe_nextn.py:119`，GLM-4.x 的 MTP draft。

### 7.2 MTP 默认参数

`python/sglang/srt/arg_groups/speculative_hook.py:522-538` — `GlmMoeDsaForCausalLM` 的默认 speculative 参数为 `(3, 1, 4)`（num_steps=3, eagle_topk=1, num_draft_tokens=4）。

### 7.3 DSA + MTP 交互

`python/sglang/srt/speculative/draft_utils.py:107-116` — DSA 模型启用 speculative decoding时导入 `DeepseekSparseAttnBackend`。

### 7.4 GLM-5.2 MTP 细节（来自文档）

`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx:96`:
- 1 个 nextn layer
- 推荐 low-latency: `--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`
- 推荐 balanced: `1-1-2`
- `index_share_for_mtp_iteration` 复用 DSA indexer topk（仅 topk==1 时有效）
- Accept length 高（4+，低延迟可达 5-6）

## 8. Weight Loading

### 8.1 DeepseekV2WeightLoaderMixin

`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:96` — `GlmMoeDsaForCausalLM` 通过继承 `DeepseekV2ForCausalLM` 获得此 mixin。

- `do_load_weights()` (line 105) — 处理 `gate_up_proj` stacking、expert params mapping、qkv fusion。
- NVFP4 checkpoint FP8 量化 (line 118): `_maybe_quant_weights_to_fp8_ue8m0()`
- `deepseek_weight_loader.py:577` — 明确排除 `GlmMoeDsaForCausalLM` 不走 `DeepseekV3ForCausalLM` 的 MXFP4 quark 路径。

### 8.2 权重加载入口

`DeepseekV2ForCausalLM.load_weights()` — `deepseek_v2.py:2842`，调用 `self.do_load_weights()`。

## 9. Runtime Inference Call Chain

1. HTTP Request → `Scheduler.event_loop_normal()` (`scheduler.py:1516`)
2. `Scheduler.run_batch()` (`scheduler.py:3185`) → 组装 `ForwardBatch`
3. `ModelRunner.forward()` (`model_runner.py:2915`)
4. `GlmMoeDsaForCausalLM.forward()` → 继承自 `DeepseekV2ForCausalLM`
5. `DeepseekV2Model.forward()` → 遍历 `DeepseekV2DecoderLayer`
6. `DeepseekV2DecoderLayer.forward()` → `DeepseekV2AttentionMLA` + `DeepseekV2MoE` + `RMSNorm`
7. `RadixAttention` → `DeepseekSparseAttnBackend`（DSA 模型）
8. `DeepseekV2MoE` → `MoEGate` routing + expert dispatch
9. `ParallelLMHead` → `LogitsProcessor` → sampling

## 10. Quantization / FP8 / NVFP4 / MXFP4 / TP / EP / MoE

### 10.1 量化支持

| 格式 | 支持 | 关键证据 |
|------|------|---------|
| FP8 | 是 | `deepseek_v2.py:328-379` (fused clamp + deep_gemm), `glm4_moe.py:887-914` |
| NVFP4 | 是 | `deepseek_weight_loader.py:118`, `test_pcg_glm5_fp4.py`, `test_glm5_nvfp4.py` |
| W4AFP8 | GLM-4.x | `glm4_moe.py:1216` |
| MXFP4 | DeepSeek/AMD | `deepseek_nextn.py:263`, AMD 测试 |
| W8A8 (modelslim) | Ascend | `ascend_npu_glm5.2_examples.mdx:114` |

### 10.2 GLM-5.2 量化（来自文档）

`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx:65` + `glm-5.2.jsx:15-18`:
- FP8: `zai-org/GLM-5.2-FP8` — 推荐部署
- BF16: `zai-org/GLM-5.2` — 需 8xB300 或多节点
- Ascend: `GLM-5.2-w8a8` (modelslim 量化，不含 MTP)

### 10.3 并行策略

- TP: `Glm4MoeForCausalLM` 中 attention 和 dense MLP 使用 TP (`glm4_moe.py:205-219`)。
- EP: `DeepseekV2MoE` 支持 `moe_ep_size` (`deepseek_v2.py:403`)。
- DP-Attention: GLM-5.2 balanced/high-throughput 推荐配置 (`glm-5.2.jsx:196-197`)。
- DeepEP: `--moe-a2a-backend deepep` (`glm-5.2.jsx:122`)。
- DSA Prefill CP: 仅 Hopper 验证 (`glm-5.2.jsx:108-110`)。
- LPLB: `GlmMoeDsaForCausalLM` 在支持列表中 (`lplb_solver.py:44`)。

### 10.4 AllReduce Fusion

`python/sglang/srt/server_args.py:4396-4409` — `GlmMoeDsaForCausalLM` 在 flashinfer allreduce fusion 列表中。

## 11. Tokenizer / Reasoning Parser / Tool Parser

### 11.1 Reasoning Parser

`python/sglang/srt/parser/reasoning_parser.py:1073`: `"glm45"` → `Glm45Detector` (line 338)。

GLM-5.2 使用 `--reasoning-parser glm45`（`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx:107`）。

### 11.2 Tool Call Parser

`python/sglang/srt/function_call/function_call_parser.py:66-68`:
- `"glm"` / `"glm45"` → `Glm4MoeDetector`
- `"glm47"` → `Glm47MoeDetector` (`glm47_moe_detector.py:165`，注释 "GLM-4.7 and GLM-5")

GLM-5.2 使用 `--tool-call-parser glm47`（`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx:153`），因为 GLM-5.2 使用新的 `<tool_call>…` 格式。

### 11.3 Template Detection

`python/sglang/srt/managers/template_detection.py:193-210`:
- `_is_glm45()` 检测 GLM-4.5+ 模板特征
- `_is_glm47()` 在此基础上增加 GLM-4.7 特征
- 注册于 `REASONING_PARSER_RULES` (line 331) 和 `TOOL_CALL_PARSER_RULES` (line 367)

## 12. Tests and Docs

### 12.1 Git-tracked 测试文件

| 文件 | 模型 | 配置 |
|------|------|------|
| `test/registered/models_e2e/test_dsa_glm5_tp_mtp.py` | `zai-org/GLM-5-FP8` | DSA+TP+MTP |
| `test/registered/models_e2e/test_dsa_glm5_dp_mtp.py` | `zai-org/GLM-5-FP8` | DSA+DP+MTP |
| `test/registered/models_e2e/test_dsa_glm5_hisparse.py` | `zai-org/GLM-5-FP8` | DSA+HiSparse |
| `test/registered/8-gpu-models/test_glm_51_fp8.py` | `zai-org/GLM-5.1-FP8` | FP8 TP8 |
| `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp4.py` | `nvidia/GLM-5-NVFP4` | PCG+NVFP4 |
| `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py` | `zai-org/GLM-5-FP8` | PCG+FP8 TP8 |
| `test/registered/gb300/test_glm5_fp8.py` | `zai-org/GLM-5.1-FP8` | GB300 FP8 |
| `test/registered/gb300/test_glm5_nvfp4.py` | `nvidia/GLM-5-NVFP4` | GB300 NVFP4 |
| `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` | GLM-5 | AMD 精度 |
| `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` | GLM-5 MXFP4 | AMD MXFP4 |
| `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` | GLM-5 | AMD 性能 |
| `test/manual/8-gpu-models/test_dsa_models_basic.py` | `zai-org/GLM-5-FP8` | DSA 基础 |
| `test/registered/moe/test_glm4_moe_models.py` | GLM-4.x MoE | MoE 基础 |

**GLM-5.2 专属测试文件：无。**

### 12.2 Git-tracked 文档

| 文档 | 说明 |
|------|------|
| `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` | GLM-5.2 完整部署指南 |
| `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` | 配置矩阵 + benchmark |
| `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` | 实测性能数据 |
| `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` | Ascend 部署 |
| `docs_new/docs/basic_usage/anthropic_api.mdx` | Anthropic API 集成 |
| `docs_new/cookbook/autoregressive/intro.mdx:31` | Cookbook 索引链接 |
| `docs_new/docs.json:1000` | 文档导航注册 |
| `docs/basic_usage/glm45.md` | GLM-4.5/4.6/4.7 指南 |
| `docs/basic_usage/deepseek_v32.md` | DeepSeek V3.2/GLM-5 共用指南 |
| `docs/platforms/ascend/ascend_npu_glm5_examples.md` | Ascend GLM-5 部署 |
| `docs/advanced_features/hisparse_guide.md` | HiSparse（提及 GLM-5.1） |

## 13. What Is Implemented

- `GlmMoeDsaForCausalLM` 类 — `python/sglang/srt/models/glm4_moe.py:1477`，GLM-5/5.1/5.2 共用架构
- `GlmMoeDsaConfig` 字段修复 — `python/sglang/srt/utils/hf_transformers/config.py:70-88`
- DSA 稀疏注意力后端 — `python/sglang/srt/layers/attention/dsa_backend.py:304`
- DSA 模型检测与配置 — `python/sglang/srt/configs/model_config.py:103-222`
- `DeepseekV2WeightLoaderMixin` 权重加载 — `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:96`
- MLA 注意力 — `python/sglang/srt/models/deepseek_v2.py:1541`
- MoE routing（grouped top-k, shared experts fusion）— `python/sglang/srt/models/deepseek_v2.py:531,421`
- MTP / NextN — `python/sglang/srt/models/deepseek_nextn.py:261`
- FP8 / NVFP4 / W4AFP8 / MXFP4 / W8A8 量化路径
- EAGLE / EAGLE3 speculative decoding
- `glm45` reasoning parser — `python/sglang/srt/parser/reasoning_parser.py:1073`
- `glm47` tool call parser — `python/sglang/srt/function_call/function_call_parser.py:68`
- Template 自动检测 — `python/sglang/srt/managers/template_detection.py:193-210`
- LPLB 支持 — `python/sglang/srt/eplb/lplb_solver.py:44`
- DSA Prefill Context Parallelism
- HiSparse（DSA 模型 KV offload）
- AMD MI30x/MI35x 和 Ascend NPU 平台支持
- GLM-5 / GLM-5.1 E2E 测试（FP8, NVFP4, MXFP4, DSA+MTP, HiSparse）
- GLM-5.2 完整部署文档和 benchmark 数据 — `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`
- GLM-5.2 Ascend NPU 部署文档 — `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx`

## 14. What Is Not Found / Unclear

- **Python 源码中没有 "GLM-5.2" 字符串** — 无专属 GLM-5.2 类、函数或常量。
- **没有 GLM-5.2 专属测试文件** — 测试覆盖止于 GLM-5 和 GLM-5.1。
- **GLM-5.2 是否有独立的 `architectures` 名称**：文档说 `glm_moe_dsa`（`GLM-5.2.mdx:95`），但 Python 代码中注册的架构名是 `GlmMoeDsaForCausalLM`，二者对应关系为大小写/下划线转换。
- **GLM-5.2 的 `index_topk` 值**：文档说 top-2048（`GLM-5.2.mdx:65`），但代码中 `get_dsa_index_topk()` 从 config 动态读取，不硬编码。
- **GLM-5.2 BF16 多节点 recipe**：标注 `verified: false`（`glm-5.2.jsx:483` 等）。
- **DSA Prefill CP 在 Blackwell 上**：`glm-5.2.jsx:110` 注明 "not yet adapted"。
- **旧 `docs/` 目录**：`docs/basic_usage/deepseek_v32.md` 仅提及 GLM-5，未提及 GLM-5.2。

## 15. Risks and Verification Checklist

### 风险

1. GLM-5.2 依赖 `GlmMoeDsaForCausalLM` 架构名。如果 HF checkpoint 使用不同名称，需确认 `architectures` 字段。
2. `GlmMoeDsaConfig` 字段修复（`hf_transformers/config.py:70-88`）依赖 transformers 版本，升级后可能需要调整。
3. DSA Prefill CP 仅在 Hopper 验证，Blackwell 上不可用。
4. BF16 多节点部署未经实测验证。
5. GLM-5.2 无专属测试，问题只能通过 GLM-5/5.1 测试间接覆盖。

### 验证步骤

```bash
# 确认 GLM-5.2 在 docs_new 中的存在
git grep -n -i "GLM-5.2" -- docs_new/

# 确认 Python 源码中无 GLM-5.2
git grep -n -i "GLM-5.2" -- python/

# 确认 GlmMoeDsaForCausalLM 注册
git grep -n "EntryClass" -- python/sglang/srt/models/glm4_moe.py

# 确认 DSA 检测逻辑
git grep -n "is_deepseek_dsa\|GlmMoeDsaForCausalLM" -- python/sglang/srt/configs/model_config.py

# 确认 GlmMoeDsaConfig 修复
git grep -n "GlmMoeDsa" -- python/sglang/srt/utils/hf_transformers/config.py

# 确认 GLM-5.2 cookbook 存在
git ls-files docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx
```

## 16. Evidence Matrix

| Feature | Evidence | Conclusion |
|---------|----------|------------|
| GLM-5.2 显式文档 | `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` | 已存在 |
| GLM-5.2 配置矩阵 | `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` | 已存在，含 benchmark |
| GLM-5.2 Ascend 文档 | `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` | 已存在 |
| GLM-5.2 Python 代码 | `git grep -i "GLM-5.2" -- python/` 零命中 | 不存在 |
| GLM-5.2 专属测试 | 无 | 不存在 |
| 架构类 | `glm4_moe.py:1477` `GlmMoeDsaForCausalLM` | GLM-5/5.1/5.2 共用 |
| DSA 后端 | `dsa_backend.py:304` `DeepseekSparseAttnBackend` | 已实现 |
| GlmMoeDsaConfig 修复 | `hf_transformers/config.py:70-88` | 已实现 |
| MTP / NextN | `deepseek_nextn.py:261`, `model_config.py:529` | 已实现 |
| 权重加载 | `deepseek_weight_loader.py:96` `DeepseekV2WeightLoaderMixin` | 已实现 |
| FP8 | `deepseek_v2.py:328-379` | 已实现 |
| NVFP4 | `deepseek_weight_loader.py:118` | 已实现 |
| Tool Call Parser | `function_call_parser.py:68` `glm47` | 已实现 |
| Reasoning Parser | `reasoning_parser.py:1073` `glm45` | 已实现 |
| DSA Prefill CP | `glm-5.2.jsx:108` (仅 Hopper) | 部分实现 |
| GLM-5/5.1 测试 | 多个 `test_glm5_*` / `test_glm_51_*` 文件 | 已实现 |
| LPLB | `lplb_solver.py:44` | 已实现 |

## 17. Appendix: Commands Used

```bash
git log -1 --oneline
git branch --show-current
git status --short
git ls-files | grep -E '^(python|test|docs|examples)/' | wc -l

git grep -n -i -E "GLM-5\.2|glm-5\.2|glm52|glm_52|GLM52|GLM5\.2|glm5\.2" -- python test docs examples
git grep -n -i -E "GLM-5\.2|glm-5\.2|glm52" -- docs_new/
git grep -n -i -E "GLM-5|GLM 5|glm5|glm_5|GlmMoe|GlmMoeDsa|glm4_moe" -- python test docs examples
git grep -n -i -E "DSA|dsa|index_topk|DeepseekV32|DeepseekV3|DeepseekV2|MLA|MTP|NextN|EAGLE" -- python test docs examples
git grep -n -i -E "fp8|nvfp4|mxfp4|modelopt_fp4|kv_cache|moe|expert|deepep|tensor_parallel|tp_size" -- python test docs examples
git ls-files | grep -iE "glm|chatglm"
git grep -n "GlmMoeDsa" -- python/
git grep -n "EntryClass" -- python/sglang/srt/models/glm4_moe.py python/sglang/srt/models/deepseek_v2.py
git grep -n "is_deepseek_dsa\|GlmMoeDsaForCausalLM" -- python/sglang/srt/configs/model_config.py
git grep -n "GlmMoeDsa" -- python/sglang/srt/utils/hf_transformers/config.py
git grep -n "GlmMoeDsaForCausalLM" -- python/sglang/srt/arg_groups/speculative_hook.py
git grep -n "GlmMoeDsaForCausalLM" -- python/sglang/srt/eplb/lplb_solver.py
git grep -n "GlmMoeDsaForCausalLM" -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
```
