# 第 12 章 多模态实现走读：图像如何变成 token

> 代码来自 `python/sglang/srt/multimodal/` 与 `managers/`。

## 12.1 多模态请求的特殊之处

一个多模态请求与文本请求的唯一区别：`GenerateReqInput`（`managers/io_struct.py`）多了几个字段：

```python
image_data: Optional[MultimodalDataInputFormat] = None
video_data: Optional[MultimodalDataInputFormat] = None
audio_data: Optional[MultimodalDataInputFormat] = None
mm_hashes: Optional[Union[List[str], List[List[str]]]] = None   # 外部哈希，用于缓存复用
```

除此之外的一切（调度、批处理、KV 缓存、采样）都与文本请求共用。所以多模态的工程问题只有一个：**怎么把图片/视频/音频变成模型认识的 token，并塞进文本序列**。

## 12.2 处理链路的两个阶段

```text
阶段一（预处理）：原始数据 → 视觉特征（image embeddings）
  输入：URL / base64 / 本地路径
  处理：下载 → 解码 → resize → 图像编码器前向（可在 CPU 或 GPU）
  输出：每张图 N 个 visual token 的 embedding + 对应的特殊 token（<image> 等）

阶段二（主干模型）：visual token 与文本 token 拼在一起，正常走 prefill/decode
```

阶段划分对应代码里的两个位置：`TokenizerManager` 的多模态预处理（tokenize 之前）与 `ModelRunner` 的图像编码器前向（GPU 前向时）。

## 12.3 Processor 生态：BaseMultimodalProcessor

`multimodal/processors/base_processor.py` 第 180 行定义抽象基类：

```python
class BaseMultimodalProcessor(ABC):
    models = []                # 注册：这个 processor 服务于哪些模型
    gpu_image_decode = True    # 图像解码是否用 GPU
    ...
    def __init__(self, hf_config, server_args, _processor, transport_mode, ...):
        self._processor = _processor          # HuggingFace processor
        self.image_config = self.server_args.mm_process_config.get("image", {})
        ...
```

`models = []` 是注册表：模型加载时按名字找到对应 processor。`multimodal/processors/` 下每个模型一个文件：

- `llava.py` / `clip.py`：经典视觉塔；
- `qwen_vl.py`、`internvl.py`、`glm4v.py`：各厂商 VLM；
- `whisper.py`、`qwen_audio.py`：音频；
- `transformers_auto.py`：兜底，直接用 HF processor。

新增一个多模态模型 = 新增一个 processor 文件 + 把模型名注册进 `models` 列表。

## 12.4 输出契约：BaseMultiModalProcessorOutput

processor 的输出是标准化的（`base_processor.py:49`）：

```python
@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    input_text: str                            # 占位符展开后的完整文本
    input_ids: Optional[...] = None            # 预分词 ids
    images: list[...] = field(default_factory=list)
    videos: list[...] = field(default_factory=list)
    audios: list[...] = field(default_factory=list)
```

流程：处理器把 `"<image> 这是什么"` 这种模板文本展开成多个 `<image>` 占位（每个对应一个视觉 token），同时产出图像特征。文本和特征在 `ForwardBatch` 里分别走 embedding 查询和视觉编码器，最终拼成一个完整的 token 序列。

## 12.5 多模态与缓存：mm_hashes 的价值

相同图像被重复请求（比如一个 Agent 反复看同一张截图）时，视觉编码是重复劳动。SGLang 的解法：

1. `mm_hashes` 让调用方提供图像哈希（或服务端自己算）；
2. 视觉 token 的 KV 一样走 RadixCache——相同图像的视觉 token 序列相同，前缀命中后**图像编码结果直接复用**；
3. `multimodal_cache.py` 专门做视觉特征缓存。

这就是第 8 章前缀缓存技术在多模态上的延伸：图像"重新 prefill"可以变成"命中缓存"。

## 12.6 工程细节：图像 token 很占地方

一张 336×336 的图通常产生几百个 visual token，比一段短文本还多。后果：

- 长图/多图请求会撑爆 batch 的 token 预算 → `mm_utils.py` 有 padding/截断策略；
- 视觉编码器可以单独 DP/TP（`docs_new` 里有 dp_for_multi_modal_encoder 方案）；
- 视觉编码器也可以用 CUDA graph 加速（`vit_cuda_graph_runner.py`、`internvl_vit_cuda_graph_runner.py`）。

## 12.7 自己动手的实验

1. 用 VLM 模型发两次**同一张图 + 同一文本**的请求，打开 `--enable-cache-report`，第二次的 `cached_tokens` 应覆盖视觉 token 部分。
2. 对比：同一张图 + 不同文本，观察缓存命中范围（图像部分命中，文本部分未命中）。
3. 发一张大图，看日志里的视觉 token 数量，体会它对 batch token 预算的占用。

## 12.8 本章小结

- 多模态 = 把非文本数据变成 visual token，其余全走文本流水线。
- Processor 按模型注册，输出契约统一（文本 + 特征 + 模态数据）。
- 相同图像的视觉 token 可以命中 RadixCache，`mm_hashes` 支持外部哈希对齐。
- 图像 token 量大会挤压 batch 预算，需要专门策略。

> 入门到进阶的旅程到此结束。第三部分开始：分布式、性能、生产环境的"为什么"与"怎么办"。
