# DP for Multi-Modal Encoder in SGLang

A typical VLM architecture involves two main components: an multi-modal encoder and a text decoder.

Most VLMs utilize a Vision Transformer (ViT) as their multi-modal encoder, it is responsible for processing visual data, extracting features (objects, colors, textures, etc.), and transforming them into a format that can be understood by the model.

The text deocoder is based on LLM. It processes textual data and generates output based on the encoded visual features.

However, since the size of ViT is very small compared to language decoders,
there is relatively little gain from TP. On the other hand, TP incurs significant communication
overhead because of all-reduce being performed after every layer.

Placing the ViT in data parallel while keeping the LLM in tensor parallel consistently lowers TTFT and boosts end-to-end throughput. In this hybrid layout, the vision front-end becomes parallel and lightweight, while scarce interconnect bandwidth and collective ops are reserved for the LLM.

Data parallelism replicates the entire model across multiple GPU sets and processes different batches of requests in parallel.

## Command Example
You can enable batch-level DP by setting `mm-enable-dp-encoder`, for example:
```
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --tp 2 \
    --mm-enable-dp-encoder
```

## Known supported models
- Qwen2.5-VL (<https://github.com/sgl-project/sglang/pull/13126>)
- Qwen3-VL (<https://github.com/sgl-project/sglang/pull/13724>)
- InternVL (<https://github.com/sgl-project/sglang/pull/13925>)
- GLM-4.5V & GLM-4.6V (<https://github.com/sgl-project/sglang/pull/14097>)
