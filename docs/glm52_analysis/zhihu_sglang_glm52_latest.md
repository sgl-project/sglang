# 从最新版 SGLang 源码看 GLM-5.2：文档显式支持，代码走 GlmMoeDsaForCausalLM + DSA 路径

> 本文基于 SGLang 仓库 `main` 分支（commit `e4976683f4`）的 Git-tracked 文件，使用 `git grep` / `git ls-files` 检索，分析 GLM-5.2 在 SGLang 中的支持现状。所有结论均有源码或文档行号佐证。

## 一、一句话结论

GLM-5.2 在 SGLang 的 `docs_new/` 目录中有完整的部署指南、配置矩阵和 benchmark 数据；但在 Python 源码中没有 GLM-5.2 专属类，运行时完全依赖 `GlmMoeDsaForCausalLM`——一个继承自 `DeepseekV2ForCausalLM` 的极简子类，复用 DeepSeek V3.2 的 DSA 稀疏注意力、MTP 多 token 预测和权重加载基础设施。

## 二、GLM-5.2 在哪里被显式提及？

在 `docs_new/` 目录下，以下 Git-tracked 文件显式提及 GLM-5.2：

**1. Cookbook 部署指南**（`docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx`）

这是最核心的 GLM-5.2 文档。它明确写道：

- GLM-5.2 是 Z.ai 的 MoE 模型，基于 DeepSeek Sparse Attention（DSA），indexer 选择 top-2048 的稀疏 key tokens（line 65）
- 架构名为 `glm_moe_dsa`，78 层 transformer，256 个 routed experts（8 active），1M 上下文窗口，1 个 MTP layer（line 65, 95）
- 两个 checkpoint：`zai-org/GLM-5.2-FP8`（FP8，推荐）和 `zai-org/GLM-5.2`（BF16，约 1.5 TB）（line 77, 82）
- SGLang 自动选择 DSA 注意力后端：`flashmla_sparse` prefill、`fa3` decode、`sgl-kernel` indexer topk（line 95）
- 自动选择 KV cache dtype：Blackwell 用 `fp8_e4m3`，Hopper 用 `bf16`（line 95）
- MTP 推荐参数：low-latency 用 `5-1-6`，balanced 用 `1-1-2`（line 96）
- `index_share_for_mtp_iteration` 在 topk==1 时复用 DSA indexer 的 topk（line 96）
- Reasoning parser: `glm45`；Tool call parser: `glm47`（line 107, 153）

**2. 配置矩阵**（`docs_new/src/snippets/configs/zai-org/glm-5.2.jsx`）

包含 12 个 `verified: true` 的单节点配置（FP8/BF16 × H200/B200/GB300/B300 × 3 策略）和 9 个 `verified: false` 的 BF16 多节点配置。每个 cell 都有完整的启动命令。

**3. Benchmark 数据**（`docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`）

在 H200/B200/GB300/B300 上的实测 TTFT、TPOT、tokens/sec/gpu 数据，基于 SGLang `0.5.13.post1` 版本。例如 H200 FP8 low-latency（8K in / 1K out）：concurrency=1 时 TTFT 662ms、TPOT 3.03ms；concurrency=16 时 TTFT 5080ms、TPOT 12.44ms。

**4. Ascend NPU 部署**（`docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx`）

包含 GLM-5.2 BF16 和 W8A8（modelslim 量化）在 Atlas 800 A3/A2 上的单节点、多节点和 PD 分离部署脚本。

**5. Anthropic API 集成**（`docs_new/docs/basic_usage/anthropic_api.mdx`）

以 GLM-5.2-FP8 为示例模型，展示 SGLang 的 `/v1/messages` 兼容端点。

## 三、Python 源码里有 GLM-5.2 专属类吗？

**没有。**

对 `python/` 目录执行 `git grep -n -i "GLM-5.2"` 返回零结果。不存在 `Glm52ForCausalLM`、`GlmMoeDsaV2ForCausalLM` 或任何 GLM-5.2 专属的类、函数或常量。

GLM-5.2 的运行时实现完全落在 `GlmMoeDsaForCausalLM` 上——这是 GLM-5、GLM-5.1、GLM-5.2 三个版本共用的架构类。不同版本的区别仅在于 HuggingFace checkpoint 中的 `architectures` 字段（值为 `GlmMoeDsaForCausalLM`）和 config 参数（如 `index_topk`、`index_topk_freq` 等）。

## 四、模型注册：从 architectures 到 Python 类

SGLang 的模型注册机制定义在 `python/sglang/srt/models/registry.py`。`import_model_classes()`（line 94）扫描 `sglang.srt.models` 包下所有模块，读取每个模块的 `EntryClass` 变量，将类名作为 key 注册到全局字典。

在 `python/sglang/srt/models/glm4_moe.py` 的 line 1482：

```python
EntryClass = [Glm4MoeForCausalLM, GlmMoeDsaForCausalLM]
```

当 HuggingFace config 的 `architectures` 字段为 `["GlmMoeDsaForCausalLM"]` 时，`ModelRegistry.resolve_model_cls()` 会返回 `GlmMoeDsaForCausalLM` 类。文档中的 `glm_moe_dsa` 就是 `GlmMoeDsaForCausalLM` 的小写/下划线形式。

此外，`python/sglang/srt/utils/hf_transformers/config.py:70-88` 有一段专门针对 `GlmMoeDsaForCausalLM` 的修复逻辑：HuggingFace 的 `GlmMoeDsaConfig` 会丢弃 `qk_rope_head_dim` 和 `index_topk_freq` 等 raw checkpoint 字段，SGLang 在加载时从 config.json 重新读取并恢复这些字段。注释标注该问题已在 transformers PR #46338 中修复。

## 五、GlmMoeDsaForCausalLM：一个极简子类

`python/sglang/srt/models/glm4_moe.py:1477-1479`：

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    def determine_num_fused_shared_experts(self):
        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")
```

仅此而已。这个类只重写了 `determine_num_fused_shared_experts`，其余全部继承自 `DeepseekV2ForCausalLM`。这意味着 GLM-5.2 的 forward、attention、MoE、权重加载、logits 计算等所有逻辑都走 DeepSeek V2 的代码路径。

## 六、DeepSeek V2 路径：GLM-5.2 的真正引擎

`DeepseekV2ForCausalLM` 定义于 `python/sglang/srt/models/deepseek_v2.py:2634`，继承 `nn.Module` 和 `DeepseekV2WeightLoaderMixin`。GLM-5.2 复用的核心组件包括：

- **Model body**: `DeepseekV2Model`（line 2317），在初始化时通过 `is_deepseek_dsa(config)` 判断是否启用 DSA 模式（line 2328）
- **Decoder Layer**: `DeepseekV2DecoderLayer`（line 2014），支持 sparse/dense 层切换
- **Attention**: `DeepseekV2AttentionMLA`（line 1541），MLA 注意力实现，支持 `kv_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim` 和 DSA
- **MoE**: `DeepseekV2MoE`（line 531）+ `MoEGate`（line 421），grouped top-k routing，注释明确标注 "Covers V3/V3.2/GLM-5/Glm4MoeLite"（line 652）
- **权重加载**: `DeepseekV2WeightLoaderMixin`（`deepseek_common/deepseek_weight_loader.py:96`），处理 `gate_up_proj` stacking、expert params mapping、qkv fusion

`DeepseekV3ForCausalLM`（line 2894）和 `DeepseekV32ForCausalLM`（line 2898）本身也是 `DeepseekV2ForCausalLM` 的空子类。所以 GLM-5.2 与 DeepSeek V3.2 在运行时共享几乎完全相同的代码路径。

## 七、DSA 稀疏注意力：GLM-5.2 的核心机制

### 7.1 DSA 检测

`python/sglang/srt/configs/model_config.py:103-115` 定义了 `is_deepseek_dsa()` 函数。当架构名为 `GlmMoeDsaForCausalLM`（以及其他 DeepSeek 架构）且 config 中 `index_topk` 字段非空时，返回 `True`。这是触发 DSA 路径的入口条件。

### 7.2 DSA 后端

`python/sglang/srt/layers/attention/dsa_backend.py:303` — `DeepseekSparseAttnBackend` 是 DSA 注意力的核心实现。初始化时：

- 调用 `is_deepseek_dsa()` 确认 DSA 模式（line 322）
- 调用 `get_dsa_index_topk()` 读取模型的 `index_topk`（line 327，GLM-5.2 为 2048）
- 配置 prefill 后端（`flashmla_sparse` / `triton` / `aiter` / `trtllm`，line 344）
- 配置 decode 后端（同上，line 347）

后端通过 `@register_attention_backend("dsa")` 注册（`attention_registry.py:111-115`）。`nsa` 是 `dsa` 的已弃用别名。

### 7.3 index_topk 与跳层优化

DSA 支持通过 `index_topk_pattern` 或 `index_topk_freq` 配置跳层优化——部分层复用上一层的 top-k 索引，减少计算量。相关函数 `dsa_layer_skips_topk()` 在 `model_config.py:180`。GLM-5 的文档推荐使用特定的 pattern 字符串 `FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS`（`docs/basic_usage/deepseek_v32.md:77`）。

### 7.4 自动后端选择

`python/sglang/srt/server_args.py:3603-3612` — 当 `model_arch` 在 DSA 模型列表中（包含 `GlmMoeDsaForCausalLM`）且 `is_deepseek_dsa(hf_config)` 为真时，自动配置 DSA 相关参数。注释标注 "DeepSeek 3.2/GLM 5"。

## 八、MTP / NextN：GLM-5.2 的推测解码

### 8.1 架构映射

GLM-5.2 的 MTP draft model 使用 `DeepseekV3ForCausalLMNextN`（`python/sglang/srt/models/deepseek_nextn.py:261`）。映射逻辑在 `model_config.py:526-531`：当 `is_draft_model` 为真且架构为 `GlmMoeDsaForCausalLM` 时，将架构替换为 `DeepseekV3ForCausalLMNextN`。

### 8.2 默认参数

`python/sglang/srt/arg_groups/speculative_hook.py:522-538` — `GlmMoeDsaForCausalLM` 的默认 speculative 参数为 `(3, 1, 4)`（num_steps=3, eagle_topk=1, num_draft_tokens=4）。文档推荐 low-latency 场景调到 `5-1-6`。

### 8.3 DSA + MTP 协同

GLM-5.2 的 config 包含 `index_share_for_mtp_iteration`，在 `--speculative-eagle-topk 1` 时允许 MTP draft steps 复用 DSA indexer 的 top-k 结果，减少重复计算。文档指出 GLM-5.2 的 MTP head 质量很高，accept length 通常在 4+ 以上（`GLM-5.2.mdx:96`）。

## 九、权重加载

`DeepseekV2WeightLoaderMixin`（`deepseek_common/deepseek_weight_loader.py:96`）是 GLM-5.2 权重加载的核心。`do_load_weights()`（line 105）处理：

- `gate_up_proj` stacking（将 gate_proj 和 up_proj 合并）
- Expert params mapping（routed experts 的 gate/down/up 权重映射）
- QKV fusion（当 `q_lora_rank` 存在时合并 `q_a_proj` 和 `kv_a_proj_with_mqa`）
- NVFP4 checkpoint 的 FP8 注意力量化（`_maybe_quant_weights_to_fp8_ue8m0()`，line 118）

值得注意的是，`deepseek_weight_loader.py:577` 有一行注释明确排除 `GlmMoeDsaForCausalLM` 不走 `DeepseekV3ForCausalLM` 的 MXFP4 quark 路径，说明权重加载有架构级别的差异化处理。

Checkpoint 必须满足：`architectures` 字段为 `GlmMoeDsaForCausalLM`，config 包含 `index_topk` 字段。

## 十、量化与并行

### 量化

| 格式 | 代码证据 | 文档/测试证据 |
|------|---------|-------------|
| FP8 | `deepseek_v2.py:328-379` fused clamp + deep_gemm | `glm-5.2.jsx` `zai-org/GLM-5.2-FP8`，12 个 verified 配置 |
| NVFP4 | `deepseek_weight_loader.py:118` | `test_pcg_glm5_fp4.py`, `test_glm5_nvfp4.py`（GLM-5） |
| MXFP4 | `deepseek_nextn.py:263` AMD 重命名 | AMD 测试 `test_glm5_mxfp4_*`（GLM-5） |
| W8A8 | — | `ascend_npu_glm5.2_examples.mdx` `--quantization modelslim` |

FP8 是 GLM-5.2 的推荐部署格式，在 H200/B200/GB300/B300 上均有 verified 配置。

### 并行策略

- **TP**: 注意力和 dense MLP 使用 tensor parallel（`glm4_moe.py:205-219`）
- **EP + DeepEP**: `DeepseekV2MoE` 支持 `moe_ep_size`（`deepseek_v2.py:403`），文档推荐 balanced/high-throughput 配置使用 `--moe-a2a-backend deepep`
- **DP-Attention**: 文档推荐 balanced/high-throughput 策略使用 `--enable-dp-attention`
- **DSA Prefill CP**: DSA prefill Context Parallelism，仅 Hopper（H200）验证，Blackwell 尚未适配（`glm-5.2.jsx:108-110`）
- **LPLB**: `GlmMoeDsaForCausalLM` 在 LPLB 支持列表中（`lplb_solver.py:44`）
- **AllReduce Fusion**: `server_args.py:4399` 包含 `GlmMoeDsaForCausalLM`
- **HiSparse**: DSA 架构的 KV offload 优化，需 PD 分离模式（`hisparse_guide.md:5`）

## 十一、Parser 与 Template

GLM-5.2 的 reasoning 解析使用 `glm45` parser（`reasoning_parser.py:1073` → `Glm45Detector`），tool call 解析使用 `glm47` parser（`function_call_parser.py:68` → `Glm47MoeDetector`）。文档指出 GLM-5.2 使用新的 tool call 格式，必须用 `glm47` 而非 `glm45`（`GLM-5.2.mdx:153`）。

模板自动检测在 `template_detection.py:193-210` 中实现，`_is_glm45()` 和 `_is_glm47()` 通过 tokenizer 中的特殊 token 和模板结构特征进行识别。

## 十二、推理调用链

GLM-5.2 的完整推理路径：

1. HTTP 请求到达 → `Scheduler.event_loop_normal()`（`scheduler.py:1516`）
2. `Scheduler.run_batch()`（`scheduler.py:3185`）组装 `ForwardBatch`
3. `ModelRunner.forward()`（`model_runner.py:2915`）
4. `GlmMoeDsaForCausalLM.forward()` → 继承自 `DeepseekV2ForCausalLM`
5. `DeepseekV2Model.forward()` → 遍历 `DeepseekV2DecoderLayer`
6. 每个 decoder layer: `DeepseekV2AttentionMLA`（MLA 注意力）+ `DeepseekV2MoE`（MoE routing）+ `RMSNorm`
7. `RadixAttention` 分发到 `DeepseekSparseAttnBackend`（DSA 模型自动选择）
8. `ParallelLMHead` → `LogitsProcessor` → sampling

## 十三、测试覆盖

当前仓库中有丰富的 GLM-5 和 GLM-5.1 测试，但**没有 GLM-5.2 专属测试**：

| 测试文件 | 模型 | 覆盖 |
|---------|------|------|
| `test_dsa_glm5_tp_mtp.py` | GLM-5-FP8 | DSA + TP + MTP |
| `test_dsa_glm5_dp_mtp.py` | GLM-5-FP8 | DSA + DP + MTP |
| `test_dsa_glm5_hisparse.py` | GLM-5-FP8 | DSA + HiSparse |
| `test_glm_51_fp8.py` | GLM-5.1-FP8 | FP8 TP8 |
| `test_pcg_glm5_fp4.py` | GLM-5-NVFP4 | PCG + NVFP4 |
| `test_glm5_fp8_tp8.py` | GLM-5-FP8 | PCG + FP8 TP8 |
| `test_glm5_fp8.py` (GB300) | GLM-5.1-FP8 | GB300 FP8 |
| `test_glm5_nvfp4.py` (GB300) | GLM-5-NVFP4 | GB300 NVFP4 |
| 多个 AMD 测试 | GLM-5 / GLM-5.1 | MI30x/MI35x 精度+性能 |

GLM-5.2 的代码正确性目前通过 GLM-5/5.1 的测试间接覆盖。

## 十四、已实现 vs 需验证

**已在 Python 源码中实现：**
- `GlmMoeDsaForCausalLM` 类和 `DeepseekV2ForCausalLM` 全套基础设施
- DSA 稀疏注意力后端（prefill + decode + MTP precompute）
- `GlmMoeDsaConfig` 字段修复
- MTP / NextN 架构映射和 EAGLE 推测解码
- FP8 / NVFP4 / MXFP4 量化路径
- `glm45` reasoning parser 和 `glm47` tool call parser
- TP / EP / DP-Attention / LPLB / AllReduce Fusion / HiSparse

**已在 docs_new/ 中文档化：**
- GLM-5.2 完整部署指南（DSA、MTP、CP、HiCache）
- 配置矩阵（4 种硬件 × 2 种量化 × 3 种策略）
- 实测 benchmark 数据
- Ascend NPU 部署脚本

**需要运行时验证：**
- GLM-5.2 无专属测试，正确性仅通过 GLM-5/5.1 间接覆盖
- BF16 多节点 recipe 标注 `verified: false`
- DSA Prefill CP 在 Blackwell 上 "not yet adapted"
- `GlmMoeDsaConfig` 字段修复依赖 transformers 版本

## 十五、总结

GLM-5.2 在 SGLang 中的支持体现了"架构复用"的设计哲学：不需要为每个模型版本写新代码，而是通过统一的 `GlmMoeDsaForCausalLM` → `DeepseekV2ForCausalLM` 继承链，复用 DSA、MTP、MoE、量化等全部基础设施。只要 HuggingFace checkpoint 的 `architectures` 字段为 `GlmMoeDsaForCausalLM` 且 config 包含 `index_topk`，SGLang 就能自动走 DSA 路径加载和运行。

文档方面，`docs_new/` 提供了非常完善的部署指南——从 low-latency 到 high-throughput，从 H200 到 Ascend NPU，从 FP8 到 BF16，每种组合都有明确的启动命令和实测数据。这说明 SGLang 团队已经对 GLM-5.2 做了系统性的工程验证。

代码方面，GLM-5.2 没有独立测试文件，运行时兼容性依赖 GLM-5/5.1 测试的间接覆盖。如果 GLM-5.2 引入了新的 config 字段或权重命名变更，可能需要额外的测试来确认兼容性。
