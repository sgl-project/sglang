# 第 12 章 多模态支持：图像、视频、音频的接入方式

## 12.1 多模态请求长什么样

一个多模态请求与文本请求的唯一区别，是 `GenerateReqInput` 里多了 `image_data` / `video_data` / `audio_data`（`managers/io_struct.py`），而文本侧完全不变。因此多模态的核心问题只有一个：**把非文本输入变成模型能理解的 token（embedding），并和文本 token 拼在一起进入同一套调度/执行流水线**。

```python
req = GenerateReqInput(
    text="这张图里有什么？",
    image_data="https://example.com/cat.png",   # 也支持 base64/本地路径
)
```

## 12.2 处理链路

```text
HTTP 请求 (image_data)
  → TokenizerManager._prepare_tokenizer_input / _tokenize_one_request
      → MultimodalProcessor（srt/multimodal/）
          → Processor 生成 image tokens + 对应的 kv cache 预留
      → BatchTokenizedGenerateReqInput（含 mm 特征）
  → Scheduler：多模态 token 与文本 token 一样参与调度与缓存
  → ModelRunner：vision encoder 前向产出 image embeddings
  → 模型主干 forward（文本 + 图像 embedding 拼接）
```

关键代码：`managers/multimodal_processor.py`（`MultimodalProcessor` 类）与 `srt/multimodal/mm_utils.py`。

## 12.3 Processor 生态：每类视觉编码器一个文件

`srt/multimodal/processors/` 下按模型组织：

- `llava.py`、`clip.py`：经典 LLaVA/CLIP 视觉塔；
- `qwen_vl.py`、`qwen3_vl.py`（`configs/qwen3_vl.py`）：Qwen 系列；
- `internvl.py`、`glm4v.py`、`gemma3.py`：各厂商 VLM；
- `whisper.py`、`qwen_audio.py`：音频；
- `mllama4.py`、`pixtral.py`、`phi4mm.py`：更多新架构；
- `transformers_auto.py`：兜底，用 HuggingFace transformers 的 processor。

每个 processor 负责：加载图像/视频 → 预处理（缩放、patch）→ 产出模型需要的输入格式（image tokens 的 ids、位置掩码、attention mask 等）。

## 12.4 多模态与调度/缓存的交互

多模态 token 与文本 token 一样参与 `req_to_token_pool`，所以：

- **前缀缓存同样生效**：相同图像（`mm_hashes` 提供外部哈希）的视觉 token 可以命中 KV 缓存，避免重复编码——`mm_hashes` 字段就是为外部路由/KV 复用设计的；
- `multimodal_cache.py` 提供视觉特征缓存；
- 调度器初始化时有 `init_mm_processor` 相关逻辑，processor 也可像模型一样用 CUDA graph 加速（`vit_cuda_graph_runner.py`、`internvl_vit_cuda_graph_runner.py`）。

## 12.5 多模态的并行与内存

- 视觉编码器可以单独 DP/TP（`docs_new/docs/advanced_features/dp_for_multi_modal_encoder.mdx` 讨论过）；`srt/configs` 里也有专门的 encoder 配置；
- 图像 token 数量远大于文本 token，`mm_utils.py` 里有 padding/截断策略，防止长图占满 batch；
- PD 分离下，多模态 prefill 也有专门路径（`disaggregation` 与 `mm_processor` 协作）。

## 12.6 快速上手示例

```bash
python examples/runtime/engine/offline_batch_inference_vlm.py \
  --model-path Qwen/Qwen2-VL-7B-Instruct
```

线上服务用法与 OpenAI 兼容接口一致：

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2-vl",
       "messages": [{"role": "user",
                     "content": [
                       {"type": "image_url", "image_url": {"url": "https://.../cat.png"}},
                       {"type": "text", "text": "这是什么？"}]}]}'
```

## 12.7 本章小结

- 多模态只是"输入侧多一类数据"，调度、批处理、KV 缓存、采样全部复用。
- Processor 生态按模型组织，新增模型通常只需新增一个 processor。
- 相同图像可通过 `mm_hashes`/缓存复用视觉特征，这是多模态吞吐的关键优化。
- 下一部分进入进阶级：分布式、投机解码、性能调优等生产话题。
