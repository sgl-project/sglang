# 我在 SGLang 里追了一遍 GLM-5.2：核心路径其实是 GlmMoeDsaForCausalLM + DeepSeek DSA

## 1. 为什么要看 SGLang 里的 GLM-5.2

GLM 系列模型从 GLM-4.5 开始就在 SGLang 里有不错的支持。到了 GLM-5，架构发生了比较大的变化——直接复用了 DeepSeek V3.2 的 DSA（DeepSeek Sparse Attention）稀疏注意力机制。那 GLM-5.2 呢？我在 SGLang 源码里做了一次完整的追踪，想搞清楚：代码里到底有没有 GLM-5.2 的影子？如果没有，最接近的路径是什么？

本文所有结论均基于 SGLang 仓库源码，每条关键论断都附有文件路径和类名/函数名，方便读者自行验证。

## 2. 最关键的结论

先说结论：

- **"GLM-5.2" 这个字符串在整个 SGLang 仓库中没有出现。** 无论是 Python 代码、测试文件还是文档，搜不到任何 GLM-5.2、glm-5.2、glm52 的引用。
- GLM-5 和 GLM-5.1 是有完整支持的。GLM-5 通过 `GlmMoeDsaForCausalLM` 类实现，这个类继承了 `DeepseekV2ForCausalLM`，复用了 DeepSeek 的 DSA 注意力和 MTP 多 token 预测基础设施。
- 如果 GLM-5.2 确实存在，最可能的路径就是通过同一个 `GlmMoeDsaForCausalLM` 架构来支持——但这只是推测，代码里没有任何针对 GLM-5.2 的特殊处理。

验证命令：`rg -rn "glm.5.2\|GLM-5.2\|glm52" python/ test/ docs/` 返回零结果。

## 3. 源码里有没有直接写 GLM-5.2？

没有。我用了以下方式全面搜索：

- 在 `python/sglang/srt/` 下递归搜索所有 `.py` 文件
- 在 `test/` 下搜索所有测试文件
- 在 `docs/` 下搜索所有 `.md` 和 `.rst` 文件

结果：零匹配。仓库里能找到的最高版本是 GLM-5.1，出现在测试文件中（`test/registered/8-gpu-models/test_glm_51_fp8.py`，使用模型 `zai-org/GLM-5.1-FP8`）。

GLM-5 的引用则散布在多个地方：
- 测试文件：`test/registered/models_e2e/test_dsa_glm5_tp_mtp.py`，使用 `zai-org/GLM-5-FP8`
- 文档：`docs/basic_usage/deepseek_v32.md` 明确写了 GLM-5 的使用方法
- Ascend NPU 文档：`docs/platforms/ascend/ascend_npu_glm5_examples.md`

## 4. GLM MoE 的主模型类：GlmMoeDsaForCausalLM

GLM-5 在 SGLang 里的入口是 `GlmMoeDsaForCausalLM`，定义在 `python/sglang/srt/models/glm4_moe.py`（约第 1477 行）：

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    def determine_num_fused_shared_experts(self):
        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")
```

你没看错——就这三行代码。它继承自 `DeepseekV2ForCausalLM`（定义在 `python/sglang/srt/models/deepseek_v2.py` 约第 2539 行），所有的 forward 逻辑、权重加载、MoE 计算全部来自父类。

同文件第 1482 行注册了两个 `EntryClass`：`Glm4MoeForCausalLM` 和 `GlmMoeDsaForCausalLM`。SGLang 的模型注册机制（`python/sglang/srt/models/registry.py`，`_ModelRegistry` 类，第 131 行）会自动扫描所有模型模块的 `EntryClass`，按架构名注册。

其他 GLM 模型类：
- `Glm4ForCausalLM`（`glm4.py` 第 419 行）——标准 GQA 注意力，非 MoE
- `Glm4MoeForCausalLM`（`glm4_moe.py` 第 1167 行）——MoE + GQA 注意力
- `Glm4MoeLiteForCausalLM`（`glm4_moe_lite.py` 第 896 行）——MoE + MLA 注意力，用于 GLM-4.7-Flash
- `ChatGLMModel`（`chatglm.py` 第 423 行）——老版 ChatGLM2/3

## 5. 为什么说它复用了 DeepSeekV2 / DeepSeek V3.2 风格路径

`GlmMoeDsaForCausalLM` 继承 `DeepseekV2ForCausalLM` 后，拿到了三样东西：

1. **权重加载**：通过 `DeepseekV2WeightLoaderMixin`（`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` 第 96 行），包括 `do_load_weights()`（第 105 行）和 `post_load_weights()`（第 411 行）。
2. **MLA 注意力**：`DeepseekV2AttentionMLA`（`deepseek_v2.py` 第 1464 行），不过 GLM-5 DSA 路径走的是 DSA 后端，不是纯 MLA。
3. **完整的前向传播和 MoE 逻辑**：`DeepseekV2ForCausalLM.forward()`（`deepseek_v2.py` 第 2691 行）。

在配置层面，`python/sglang/srt/configs/model_config.py` 的 `is_deepseek_dsa()` 函数（第 103 行）会检测架构名是否为 `GlmMoeDsaForCausalLM` 且 config 中设置了 `index_topk`。如果是，就判定为 DSA 模型。

`server_args.py`（第 2004-2024 行）中，`GlmMoeDsaForCausalLM` 被列在和 `DeepseekV3ForCausalLM` 同一组里，自动设置 `attention_backend = "dsa"`。注释里直接写了 `# DeepSeek 3.2/GLM 5`。

## 6. DSA / MLA 注意力路径

GLM 模型在 SGLang 里有三条不同的注意力路径：

**DSA 路径（GLM-5 使用）：**
- 检测：`is_deepseek_dsa()`（`model_config.py` 第 103 行）
- 后端：`DeepseekSparseAttnBackend`（`python/sglang/srt/layers/attention/dsa_backend.py` 第 287 行）
- 自动选择：`server_args.py` 第 2023 行设 `attention_backend = "dsa"`
- 可配置后端：`--dsa-prefill-backend` 和 `--dsa-decode-backend`
- 可选 kernel：`flashmla_sparse`、`flashmla_kv`、`fa3`、`tilelang`、`trtllm`、`aiter`

DSA 的核心思想是通过 indexer 选取 top-k 个 KV token 做稀疏注意力，而非全量 attention。`DSAMetadata`（第 125 行）和 `DSAIndexerMetadata`（第 203 行）封装了稀疏选择所需的元数据。

**MLA 路径（GLM-4.7-Flash 使用）：**
- `Glm4MoeLiteDecoderLayer`（`glm4_moe_lite.py` 第 538 行）在第 557 行实例化 `DeepseekV2AttentionMLA`
- MLA 参数：`qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim`、`q_lora_rank`、`kv_lora_rank`
- `model_config.py` 第 620-646 行对 `GlmMoeDsaForCausalLM` 和 `Glm4MoeLiteForCausalLM` 都设 `AttentionArch.MLA`

**标准 GQA 路径（GLM-4/4.5/4.6 使用）：**
- `Glm4MoeAttention`（`glm4_moe.py` 第 180 行）用 `QKVParallelLinear`（第 229 行），不走 MLA

## 7. 权重加载路径

**GlmMoeDsaForCausalLM** 的权重加载完全继承自 `DeepseekV2ForCausalLM.load_weights()`（`deepseek_v2.py` 第 2747 行）。关键组件：

- `DeepseekV2WeightLoaderMixin.do_load_weights()`（`deepseek_weight_loader.py` 第 105 行）——主入口
- `post_load_weights()`（第 411 行）——后处理，包括 MLA 权重融合
- `_maybe_quant_weights_to_fp8_ue8m0()`（第 655 行）——FP8 量化支持
- 堆叠参数映射：`qkv_proj` 和 `gate_up_proj`

**Glm4MoeLiteForCausalLM** 有自己的 `load_weights()`（`glm4_moe_lite.py` 第 1050 行），其中第 1223 行有一段注释 `# GLM NOTE: for MLA`，实现了把 `q_a_proj` 和 `kv_a_proj_with_mqa` 融合成 `fused_qkv_a_proj_with_mqa` 的逻辑。这是 GLM 特有的权重融合，DeepSeek 原始代码里没有。

**Glm4MoeForCausalLM** 的 `load_weights()` 在 `glm4_moe.py` 第 1264 行。

## 8. 推理调用链：从请求到采样

完整的推理调用链（以 GLM-5 DSA 为例）：

1. **HTTP 请求入口**：`python/sglang/srt/entrypoints/http_server.py` 接收 OpenAI 兼容 API 请求。
2. **调度器**：`Scheduler`（`python/sglang/srt/managers/scheduler.py` 第 291 行）的 `run_batch()`（第 3106 行）接收 `ScheduleBatch`，决定 prefill/decode 模式，分发到 TP worker。
3. **TP Worker**：`TpModelWorker`（`python/sglang/srt/managers/tp_worker.py` 第 218 行）的 `forward_batch_generation()`（第 467 行）创建 `ForwardBatch`（`forward_batch_info.py` 第 263 行），调用 `model_runner.forward()`。
4. **ModelRunner**：`ModelRunner`（`model_runner.py` 第 371 行）的 `forward()`（第 3493 行）调用 `_forward_raw()`（第 3584 行），按模式分支：`forward_decode()`（第 3288 行）、`forward_extend()`（第 3338 行）、`forward_split_prefill()`（第 3466 行）。
5. **模型前向**：调用 `DeepseekV2ForCausalLM.forward()`（`deepseek_v2.py` 第 2691 行），逐层执行 decoder layer。
6. **DSA 注意力**：`DeepseekSparseAttnBackend` 处理稀疏 KV 选择和 prefill/decode。
7. **MoE 前向**：`Glm4MoeSparseMoeBlock.forward()`（`glm4_moe.py` 第 560 行）分发到 `forward_normal()`、`forward_normal_dual_stream()`（双流并行）或 `forward_deepep()`（DeepEP A2A）。
8. **Logits 处理**：`LogitsProcessor`（`logits_processor.py` 第 260 行）计算 logits，`_preprocess_logits()`（第 3682 行）应用 vocab mask 和 bias。
9. **采样**：`Sampler`（`sampler.py` 第 68 行）执行 temperature、top-k、top-p 采样。
10. **KV 缓存**：`DSATokenToKVPool`（`memory_pool.py` 第 2492 行，继承 `MLATokenToKVPool`）存储压缩 KV。`RadixCache`（`radix_cache.py` 第 285 行）管理前缀缓存。HiSparse 场景下用 `HiSparseDSATokenToKVPool`（`hisparse_memory_pool.py` 第 28 行）做主机内存卸载。

## 9. FP8、NVFP4、MoE、TP、EP 支持

**FP8**：`Glm4MoeForCausalLM` 和 `Glm4MoeLiteForCausalLM` 构造函数都接受 `quant_config`。测试中使用 `zai-org/GLM-5-FP8` 和 `zai-org/GLM-5.1-FP8`。权重加载器有 `_maybe_quant_weights_to_fp8_ue8m0()`（`deepseek_weight_loader.py` 第 655 行）。

**NVFP4**：`nvidia/GLM-5-NVFP4` 在 `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp4.py` 中测试，使用 `modelopt_fp4` 量化方法。仅限 B200。

**Tensor Parallel**：`Glm4MoeAttention` 用 `attn_tp_size` 做 QKV 分片。MoE 层用 `get_parallel().tp_size` 和 `moe_ep_size`。

**Expert Parallel**：`Glm4MoeSparseMoeBlock`（`glm4_moe.py` 第 391 行）支持 DeepEP A2A 后端（`forward_deepep` 第 649 行）。`Glm4MoeLiteSparseMoeBlock`（`glm4_moe_lite.py` 第 176 行）也支持 EP。

**MoE 共享专家融合**：`determine_num_fused_shared_experts()` 控制是否将共享专家融合到路由专家中。双流执行（`forward_normal_dual_stream`）在不同 CUDA stream 上并行执行共享专家和路由专家。

**KV Cache**：DSA 模型支持 FP8 KV cache。`server_args.py` 的 `_set_default_dsa_backends()`（第 1915 行）根据 `kv_cache_dtype` 自动设置默认 DSA 后端。

## 10. Tokenizer、Reasoning Parser、Tool Parser

**Reasoning Parser**：`Glm45Detector`（`python/sglang/srt/parser/reasoning_parser.py` 第 338 行）处理 GLM-4.5+ 的推理格式，注册名 `"glm45"`（第 1073 行）。注意：`glm45` 已被弃用，在 `server_args.py` 第 1206 行映射到 `"glm"`。

**Tool Call Parser**：有两个检测器：
- `Glm4MoeDetector`（`function_call/glm4_moe_detector.py` 第 134 行）——用于 GLM-4.5/4.6，注册名 `"glm"` 和 `"glm45"`
- `Glm47MoeDetector`（`function_call/glm47_moe_detector.py` 第 165 行）——用于 GLM-4.7+，注册名 `"glm47"`，支持 xgrammar structural tag

GLM-5.1 的测试配置（`test_glm_51_fp8.py` 第 16-18 行）使用 `--reasoning-parser=glm45` 和 `--tool-call-parser=glm47`。

**Thinking Budget**：`Glm4MoeThinkingBudgetLogitProcessor`（`python/sglang/srt/sampling/custom_logit_processor.py` 第 116 行）提供思考预算控制。

**模板检测**：`python/sglang/srt/managers/template_detection.py` 第 192 行有 `_is_glm45()` 函数检测 GLM-4.5+ 的聊天模板。

没有发现 GLM-5 专属的 tokenizer 或聊天模板代码，GLM-5 复用 `glm45`/`glm47` 的解析器。

## 11. 哪些是明确实现的，哪些没有找到

**明确实现的：**
- GLM-5 DSA 架构（`GlmMoeDsaForCausalLM` 继承 `DeepseekV2ForCausalLM`）
- GLM-5.1 FP8 量化支持和测试
- GLM-5 NVFP4 量化支持和测试
- DSA 注意力后端，多种 kernel 可选
- MTP / EAGLE 推测解码
- TP、DP attention、EP 支持
- Tool call parser（`glm45`、`glm47`）
- Reasoning parser（`glm45`）
- Thinking budget logit processor
- Ascend NPU 支持
- Index Cache 优化（含 GLM-5 专属 `index_topk_pattern` 推荐）
- HiSparse 支持（GLM-5.1）

**没有找到 / 不明确的：**
- "GLM-5.2" 在仓库中完全不存在
- 没有独立的 GLM-5 模型文件——全靠 `GlmMoeDsaForCausalLM` 这 3 行子类
- 没有 GLM-5 专属的 tokenizer 或聊天模板
- GLM-5.2 是否需要新的架构改动：从代码层面无法判断
- GLM MoE 模型没有使用 sliding window attention

## 12. 风险和下一步验证

**风险：**
- GLM-5 支持和 DeepSeek V3.2 代码深度耦合。`deepseek_v2.py` 或 `deepseek_weight_loader.py` 的任何 breaking change 都可能影响 GLM-5。
- `GlmMoeDsaForCausalLM` 只有 3 行代码，所有逻辑在父类里，GLM-5 特有的 bug 很难隔离。
- `glm45` reasoning parser 已弃用（映射到 `glm`），可能造成混淆。
- DSA 后端选择复杂且依赖硬件，配错了可能性能大幅回退。
- FP4/NVFP4 仅在 B200 上测试过，可移植性有限。
- `index_topk_pattern` 是一串 F/S 字符的原始字符串，脆弱且难以验证。

**下一步验证清单：**
- 确认 GLM-5.2 是否是真实版本号
- 如果存在，检查是否用同一个 `GlmMoeDsaForCausalLM` 架构
- 验证 GLM-5.2 权重与 `DeepseekV2WeightLoaderMixin` 的兼容性
- 检查是否需要新的 tool call parser 或 reasoning parser
- 持续关注 DeepSeek V3.2 代码变更对 GLM-5/5.1 的影响
- 如果模型发布，补充 GLM-5.2 专属测试

## 13. 总结

在 SGLang 仓库里，GLM-5.2 并不存在。但 GLM-5 和 GLM-5.1 的支持是实打实的——核心就是 `GlmMoeDsaForCausalLM` 这个极简子类，挂在 `DeepseekV2ForCausalLM` 之上，复用了 DeepSeek V3.2 的 DSA 稀疏注意力、MTP 多 token 预测、权重加载和 MoE 基础设施。

如果 GLM-5.2 沿用相同的 DSA 架构，SGLang 很可能已经"免费"支持了它——不需要改一行代码，只要模型的 `architectures` 字段写的是 `GlmMoeDsaForCausalLM`，config 里有 `index_topk`，就能自动走 DSA 路径。但这是推测，不是事实。

最后，所有关键结论都可以用一条命令验证：

```
rg -rn "glm.5.2\|GLM-5.2\|glm52" python/ test/ docs/
```

如果这条命令返回了结果，说明仓库已更新，本文的结论需要修正。
