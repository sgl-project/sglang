# 最新版 SGLang 中 GLM-5.2 支持的源码分析

## 1. 结论先行

GLM-5.2 在 SGLang 中的支持状态可分为两个层面：

- **文档层面**：`docs_new/` 下有完整的 GLM-5.2 部署指南、配置矩阵、benchmark 数据和 Ascend NPU 部署脚本，明确标注 GLM-5.2 使用 `glm_moe_dsa` 架构。
- **代码层面**：Python 源码中没有 "GLM-5.2" 字符串，也没有 `Glm52` 专属类。GLM-5.2 通过 `GlmMoeDsaForCausalLM`（`python/sglang/srt/models/glm4_moe.py:1477`）获得支持，该类继承 `DeepseekV2ForCausalLM`，复用 DSA 稀疏注意力、MTP、权重加载等全部基础设施。GLM-5/5.1/5.2 共用同一个模型类，区别仅在 HuggingFace checkpoint 的 `architectures` 字段和 config 参数。

## 2. 仓库版本

- Commit: `e4976683f4` — `[Docs] Fix broken links in cookbook (#29261)`
- Branch: `main`
- Git-tracked 文件（`python/`、`test/`、`docs/`、`docs_new/`、`examples/`）共 5296 个
- 工作区干净，仅有未跟踪文件 `CLAUDE.md`、`docs/glm52_analysis/`、`protoc-25.3-linux-x86_64.zip`

## 3. GLM-5.2 的显式证据

以下 Git-tracked 文件中显式提及 GLM-5.2：

**Cookbook 部署指南** — `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`:
- 标题: "GLM-5.2"（line 2）
- 描述: DSA MoE 模型，MTP 推测解码，1M 上下文，支持 H200/B200/B300/GB300（line 3）
- 架构: `glm_moe_dsa`，78 层，256 experts（top-8），top-2048 DSA indexer（line 65, 95）
- Checkpoint: `zai-org/GLM-5.2-FP8`（FP8）和 `zai-org/GLM-5.2`（BF16）（line 77, 82）
- DSA 自动后端: `flashmla_sparse` prefill, `fa3` decode, `sgl-kernel` indexer topk（line 95）
- MTP: 1 个 nextn layer, `index_share_for_mtp_iteration` 复用 DSA topk（line 96）
- Parser: `--reasoning-parser glm45`, `--tool-call-parser glm47`（line 107, 153）

**配置矩阵** — `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`:
- `modelName: "GLM-5.2"`（line 5）
- 支持 FP8 和 BF16（line 15-18）
- 硬件: H200, B200, GB300, B300（line 7-9）
- 策略: low-latency, balanced, high-throughput（line 19-23）
- 含 12 个 `verified: true` 的单节点配置 cell 和 9 个 `verified: false` 的 BF16 多节点 cell

**Benchmark 数据** — `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`:
- 在 H200/B200/GB300/B300 上的实测 TTFT、TPOT、tokens/sec/gpu 数据
- SGLang 版本: `0.5.13.post1`（line 12 等）

**Ascend NPU 部署** — `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx`:
- GLM-5.2 BF16 和 W8A8（modelslim）版本（line 11-12）
- 单节点/多节点/PD 分离部署脚本

**Anthropic API 文档** — `docs_new/docs/basic_usage/anthropic_api.mdx`:
- GLM-5.2-FP8 作为 Anthropic 兼容端点的示例模型（line 25, 31）

**Cookbook 索引** — `docs_new/cookbook/autoregressive/intro.mdx:31` 链接到 GLM-5.2 页面。

## 4. Python 源码里有没有 GLM-5.2 专属类？

**没有。**

- `git grep -n -i -E "GLM-5\.2|glm52|glm_52" -- python/` 返回零结果。
- 没有 `Glm52ForCausalLM`、`GlmMoeDsaV2ForCausalLM` 或任何 GLM-5.2 专属类/函数。
- GLM-5.2 的运行时实现完全依赖 `GlmMoeDsaForCausalLM`，这是一个 GLM-5/5.1/5.2 共用的架构类。

## 5. 模型注册与架构映射

### 5.1 ModelRegistry

`python/sglang/srt/models/registry.py:94-127` — `import_model_classes()` 扫描 `sglang.srt.models` 下所有模块，读取 `EntryClass` 变量注册模型类。`ModelRegistry.register("sglang.srt.models")`（line 131）完成全局注册。

### 5.2 EntryClass

`python/sglang/srt/models/glm4_moe.py:1482`:
```python
EntryClass = [Glm4MoeForCausalLM, GlmMoeDsaForCausalLM]
```

这意味着 HuggingFace config 中 `architectures: ["GlmMoeDsaForCausalLM"]` 的 checkpoint 会被映射到此类。文档中 `glm_moe_dsa` 是 `GlmMoeDsaForCausalLM` 的小写/下划线形式。

### 5.3 DSA 模型检测

`python/sglang/srt/configs/model_config.py:103-115` — `is_deepseek_dsa()` 检测架构名（含 `GlmMoeDsaForCausalLM`）且 `index_topk` 非空。

### 5.4 GlmMoeDsaConfig 字段修复

`python/sglang/srt/utils/hf_transformers/config.py:70-88` — 当 `architectures[0] == "GlmMoeDsaForCausalLM"` 时，从 raw config.json 恢复 `qk_rope_head_dim` 和 `index_topk_freq` 等被 HuggingFace `GlmMoeDsaConfig` 丢弃的字段。

## 6. 主模型类：GlmMoeDsaForCausalLM

`python/sglang/srt/models/glm4_moe.py:1477-1479`:

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    def determine_num_fused_shared_experts(self):
        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")
```

这是一个极简子类，仅重写 `determine_num_fused_shared_experts`。所有模型逻辑——forward、attention、MoE、权重加载——全部继承自 `DeepseekV2ForCausalLM`。

## 7. DeepSeekV2 / DeepSeekV3.2 风格路径复用

### 7.1 DeepseekV2ForCausalLM

`python/sglang/srt/models/deepseek_v2.py:2634` — 继承 `nn.Module` 和 `DeepseekV2WeightLoaderMixin`。

关键继承组件：
- `DeepseekV2Model`（line 2317）— model body，`use_dsa = is_deepseek_dsa(config)`（line 2328）
- `DeepseekV2DecoderLayer`（line 2014）— decoder layer，支持 sparse/dense 切换
- `DeepseekV2AttentionMLA`（line 1541）— MLA 注意力，支持 `kv_lora_rank`、DSA
- `DeepseekV2MoE`（line 531）+ `MoEGate`（line 421）— grouped top-k routing，注释 line 652: "Covers V3/V3.2/GLM-5/Glm4MoeLite"
- `DeepseekV3ForCausalLM`（line 2894）和 `DeepseekV32ForCausalLM`（line 2898）— 空子类

### 7.2 权重加载

`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:96` — `DeepseekV2WeightLoaderMixin`，`GlmMoeDsaForCausalLM` 通过继承获得。`do_load_weights()`（line 105）处理 `gate_up_proj` stacking、expert params mapping、qkv fusion。

## 8. DSA / MLA / index_topk 路径

### 8.1 DSA 后端

`python/sglang/srt/layers/attention/dsa_backend.py:303` — `DeepseekSparseAttnBackend`：
- `is_deepseek_dsa()` 检测（line 322）
- `get_dsa_index_topk()` 读取 topk（line 327）
- Prefill 后端: `flashmla_sparse` / `triton` / `aiter` / `trtllm`（line 344）
- Decode 后端: 同上（line 347）
- GLM-5.1 AMD 验证注释（line 62）、GLM-5 head padding 注释（line 375）

### 8.2 后端注册

`python/sglang/srt/layers/attention/attention_registry.py:111-115` — `@register_attention_backend("dsa")` → `DeepseekSparseAttnBackend`。`nsa` 是已弃用别名（line 118）。

### 8.3 DSA 函数

- `is_deepseek_dsa()` — `model_config.py:103`
- `get_dsa_index_topk()` — `model_config.py:175`
- `dsa_layer_skips_topk()` — `model_config.py:180`，支持 `index_topk_pattern` / `index_topk_freq`
- `get_dsa_index_n_heads()` — `model_config.py:203`
- `get_dsa_index_head_dim()` — `model_config.py:125`

### 8.4 自动后端选择

`python/sglang/srt/server_args.py:3603-3612` — `GlmMoeDsaForCausalLM` 在 DSA 模型列表中，触发 DSA 参数自动配置。注释: "DeepSeek 3.2/GLM 5"。

### 8.5 MLA 配置

`python/sglang/srt/configs/model_config.py:726-756` — DSA 模型（含 `GlmMoeDsaForCausalLM`）设置 `AttentionArch.MLA`。

### 8.6 GLM-5.2 DSA 细节（文档）

`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx:95`:
- 自动选择: `flashmla_sparse` prefill, `fa3` decode, `sgl-kernel` indexer topk
- 自动 KV cache dtype: `fp8_e4m3` on Blackwell, `bf16` on Hopper
- 支持 DSA prefill Context Parallelism（仅 Hopper 验证）

## 9. MTP / NextN / EAGLE 推测解码

### 9.1 NextN 架构

`python/sglang/srt/configs/model_config.py:526-531` — `_config_draft_model()` 将 `GlmMoeDsaForCausalLM` 映射到 `DeepseekV3ForCausalLMNextN`（`python/sglang/srt/models/deepseek_nextn.py:261`）作为 MTP draft model。

### 9.2 MTP 默认参数

`python/sglang/srt/arg_groups/speculative_hook.py:522-538` — `GlmMoeDsaForCausalLM` 的默认 speculative 参数: `(3, 1, 4)`。

### 9.3 GLM-5.2 MTP 细节（文档）

`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx:96`:
- 1 个 nextn layer
- 推荐 low-latency: `--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`
- `index_share_for_mtp_iteration` 复用 DSA indexer topk（仅 topk==1 有效）
- Accept length 高（4+，低延迟可达 5-6）

## 10. 权重加载与 checkpoint 兼容性

### 10.1 加载器

`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:96` — `DeepseekV2WeightLoaderMixin`。

- `do_load_weights()`（line 105）— `gate_up_proj` stacking、expert params mapping、qkv fusion
- `_maybe_quant_weights_to_fp8_ue8m0()`（line 118）— NVFP4 checkpoint 的 FP8 注意力量化
- `deepseek_weight_loader.py:577` — 明确排除 `GlmMoeDsaForCausalLM` 不走 `DeepseekV3ForCausalLM` 的 MXFP4 quark 路径

### 10.2 Checkpoint 要求

- `architectures` 字段须为 `GlmMoeDsaForCausalLM`
- config 须包含 `index_topk`（否则 `is_deepseek_dsa()` 返回 False）
- `qk_rope_head_dim` 和 `index_topk_freq` 字段会被 `hf_transformers/config.py:70-88` 恢复

## 11. 推理调用链

1. HTTP Request → `Scheduler.event_loop_normal()`（`python/sglang/srt/managers/scheduler.py:1516`）
2. `Scheduler.run_batch()`（`scheduler.py:3185`）→ 组装 `ForwardBatch`
3. `ModelRunner.forward()`（`python/sglang/srt/model_executor/model_runner.py:2915`）
4. `GlmMoeDsaForCausalLM.forward()` → 继承自 `DeepseekV2ForCausalLM`
5. `DeepseekV2Model.forward()` → 遍历 `DeepseekV2DecoderLayer`
6. `DeepseekV2DecoderLayer.forward()` → `DeepseekV2AttentionMLA` + `DeepseekV2MoE` + `RMSNorm`
7. `RadixAttention` → `DeepseekSparseAttnBackend`（DSA 模型）
8. `DeepseekV2MoE` → `MoEGate` routing + expert dispatch
9. `ParallelLMHead` → `LogitsProcessor` → sampling

## 12. 量化、并行与部署能力

### 12.1 量化支持

| 格式 | 代码证据 | 文档证据 | 结论 |
|------|---------|---------|------|
| FP8 | `deepseek_v2.py:328-379`（fused clamp + deep_gemm） | `glm-5.2.jsx:16` `zai-org/GLM-5.2-FP8` | 已实现+已验证 |
| NVFP4 | `deepseek_weight_loader.py:118` | `test_pcg_glm5_fp4.py`, `test_glm5_nvfp4.py` | 已实现（GLM-5） |
| MXFP4 | `deepseek_nextn.py:263`（AMD 重命名） | AMD 测试 `test_glm5_mxfp4_*` | 已实现（GLM-5/AMD） |
| W8A8 | — | `ascend_npu_glm5.2_examples.mdx:114` `--quantization modelslim` | Ascend 文档化 |

### 12.2 并行策略

- **TP**: attention 和 dense MLP 使用 TP（`glm4_moe.py:205-219`）
- **EP**: `DeepseekV2MoE` 支持 `moe_ep_size`（`deepseek_v2.py:403`）
- **DP-Attention**: GLM-5.2 balanced/high-throughput 推荐（`glm-5.2.jsx:196-197`）
- **DeepEP**: `--moe-a2a-backend deepep`（`glm-5.2.jsx:122`）
- **DSA Prefill CP**: 仅 Hopper 验证（`glm-5.2.jsx:108-110`）
- **LPLB**: `GlmMoeDsaForCausalLM` 在支持列表（`python/sglang/srt/eplb/lplb_solver.py:44`）
- **AllReduce Fusion**: `server_args.py:4399` 列表包含 `GlmMoeDsaForCausalLM`

### 12.3 HiSparse

`docs/advanced_features/hisparse_guide.md:5` — HiSparse 支持 DSA 架构（DeepSeek-V3.2, GLM-5.1），需 PD 分离模式。

### 12.4 硬件支持

| 平台 | 证据 |
|------|------|
| NVIDIA H200 | `glm-5.2.jsx` 配置 cell，`verified: true` |
| NVIDIA B200 | `glm-5.2.jsx` 配置 cell，`verified: true` |
| NVIDIA GB300 | `glm-5.2.jsx` 配置 cell，`verified: true` |
| NVIDIA B300 | `glm-5.2.jsx` 配置 cell，FP8+BF16 `verified: true` |
| AMD MI30x/MI35x | `test_glm5_*_mi35x.py` 测试文件 |
| Ascend NPU | `ascend_npu_glm5.2_examples.mdx` 部署脚本 |

## 13. Tokenizer / Reasoning Parser / Tool Parser

### 13.1 Reasoning Parser

`python/sglang/srt/parser/reasoning_parser.py:1073` — `"glm45"` → `Glm45Detector`（line 338）。

GLM-5.2 使用 `--reasoning-parser glm45`（`GLM-5.2.mdx:107`）。

### 13.2 Tool Call Parser

`python/sglang/srt/function_call/function_call_parser.py:66-68`:
- `"glm"` / `"glm45"` → `Glm4MoeDetector`
- `"glm47"` → `Glm47MoeDetector`（`glm47_moe_detector.py:165`，注释 "GLM-4.7 and GLM-5"）

GLM-5.2 使用 `--tool-call-parser glm47`（`GLM-5.2.mdx:153`），因为 GLM-5.2 使用新的 `<tool_call>…` 格式。

### 13.3 Template Detection

`python/sglang/srt/managers/template_detection.py:193-210` — `_is_glm45()` 和 `_is_glm47()` 检测 GLM 模板特征。

## 14. 测试与文档覆盖

### 14.1 测试文件

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
| `test/manual/8-gpu-models/test_dsa_models_basic.py` | `zai-org/GLM-5-FP8` | DSA 基础 |

**GLM-5.2 专属测试文件：无。**

### 14.2 文档

| 文档 | 说明 |
|------|------|
| `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` | GLM-5.2 完整部署指南 |
| `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` | 配置矩阵 |
| `docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` | 实测 benchmark |
| `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` | Ascend 部署 |
| `docs_new/docs/basic_usage/anthropic_api.mdx` | Anthropic API 集成 |
| `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx` | GLM-5 部署指南 |
| `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx` | GLM-5.1 部署指南 |
| `docs/basic_usage/deepseek_v32.md` | DeepSeek V3.2/GLM-5 共用指南 |
| `docs/basic_usage/glm45.md` | GLM-4.5/4.6/4.7 指南 |
| `docs/advanced_features/hisparse_guide.md` | HiSparse（提及 GLM-5.1） |

## 15. 已实现、文档化、未确认

### 已在 Python 源码中实现

- `GlmMoeDsaForCausalLM` 类 — `glm4_moe.py:1477`
- `GlmMoeDsaConfig` 字段修复 — `hf_transformers/config.py:70-88`
- DSA 稀疏注意力后端 — `dsa_backend.py:303`
- DSA 模型检测与配置 — `model_config.py:103-222`
- `DeepseekV2WeightLoaderMixin` 权重加载 — `deepseek_weight_loader.py:96`
- MLA 注意力 — `deepseek_v2.py:1541`
- MoE routing（grouped top-k, shared experts fusion）— `deepseek_v2.py:531,421`
- MTP / NextN — `deepseek_nextn.py:261`, `model_config.py:529`
- FP8 / NVFP4 / MXFP4 量化路径
- EAGLE / EAGLE3 speculative decoding
- `glm45` reasoning parser, `glm47` tool call parser
- Template 自动检测 — `template_detection.py:193-210`
- LPLB 支持 — `lplb_solver.py:44`

### 已在 docs_new/ 中文档化

- GLM-5.2 完整部署指南（含 DSA、MTP、CP、HiCache 说明）
- 配置矩阵（FP8/BF16 × H200/B200/GB300/B300 × 3 策略）
- 实测 benchmark 数据（v0.5.13.post1）
- Ascend NPU 部署脚本（单节点/多节点/PD 分离）
- Anthropic API 集成示例

### 需要运行时验证

- GLM-5.2 无专属测试，代码正确性仅通过 GLM-5/5.1 测试间接覆盖
- BF16 多节点 recipe 标注 `verified: false`
- DSA Prefill CP 在 Blackwell 上 "not yet adapted"
- `GlmMoeDsaConfig` 字段修复依赖 transformers 版本（PR #46338）

## 16. Evidence Matrix

| Feature | Evidence | Conclusion |
|---------|----------|------------|
| GLM-5.2 文档 | `docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx` | 已存在 |
| GLM-5.2 配置+benchmark | `docs_new/src/snippets/configs/zai-org/glm-5.2.jsx` | 已存在 |
| GLM-5.2 Ascend | `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx` | 已存在 |
| GLM-5.2 Python 代码 | `git grep -i "GLM-5.2" -- python/` 零命中 | 不存在 |
| GLM-5.2 专属测试 | 无 | 不存在 |
| 架构类 | `glm4_moe.py:1477` `GlmMoeDsaForCausalLM` | GLM-5/5.1/5.2 共用 |
| DSA 后端 | `dsa_backend.py:303` | 已实现 |
| Config 修复 | `hf_transformers/config.py:70-88` | 已实现 |
| MTP / NextN | `deepseek_nextn.py:261`, `model_config.py:529` | 已实现 |
| 权重加载 | `deepseek_weight_loader.py:96` | 已实现 |
| FP8 | `deepseek_v2.py:328-379` | 已实现 |
| NVFP4 | `deepseek_weight_loader.py:118` | 已实现 |
| Tool Call Parser | `function_call_parser.py:68` `glm47` | 已实现 |
| Reasoning Parser | `reasoning_parser.py:1073` `glm45` | 已实现 |
| DSA Prefill CP | `glm-5.2.jsx:108`（仅 Hopper） | 部分实现 |
| GLM-5/5.1 测试 | 多个 `test_glm5_*` 文件 | 已实现 |

## 17. 手工验证命令

```bash
# GLM-5.2 在 docs_new 中的存在
git grep -n -i "GLM-5.2" -- docs_new/

# GLM-5.2 在 Python 源码中不存在
git grep -n -i "GLM-5.2" -- python/

# GlmMoeDsaForCausalLM 注册
git grep -n "EntryClass" -- python/sglang/srt/models/glm4_moe.py

# DSA 检测逻辑
git grep -n "is_deepseek_dsa\|GlmMoeDsaForCausalLM" -- python/sglang/srt/configs/model_config.py

# GlmMoeDsaConfig 修复
git grep -n "GlmMoeDsa" -- python/sglang/srt/utils/hf_transformers/config.py

# DSA 后端
git grep -n "class DeepseekSparseAttnBackend" -- python/sglang/srt/layers/attention/dsa_backend.py

# Draft model 映射
git grep -n "GlmMoeDsaForCausalLM" -- python/sglang/srt/configs/model_config.py

# 权重加载
git grep -n "class DeepseekV2WeightLoaderMixin" -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py

# GLM-5.2 配置文件
git ls-files docs_new/src/snippets/configs/zai-org/glm-5.2.jsx

# GLM 测试文件
git ls-files | grep -iE "test.*glm5"
```
