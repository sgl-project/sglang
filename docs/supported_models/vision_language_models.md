# Vision Language Models

These models accept multi-modal inputs (e.g., images and text) and generate text output. They augment language models with visual encoders and require a specific chat template for handling vision prompts.

## Example launch Command

```shell
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \  # example HF/local path
  --host 0.0.0.0 \
  --port 30000 \
```

## Supporting Matrixs

| Model Family (Variants)        | Example HuggingFace Identifier                     | Chat Template        | Description                                                                            |
|--------------------------------|--------------------------------------------------|----------------------|----------------------------------------------------------------------------------------|
| **Qwen-VL** (Qwen2 series)     | `Qwen/Qwen2.5-VL-7B-Instruct`                    | `qwen2-vl`           | Alibaba’s vision-language extension of Qwen; for example, Qwen2.5-VL (7B and larger variants) can analyze and converse about image content. |
| **DeepSeek-VL2**               | `deepseek-ai/deepseek-vl2`                       | `deepseek-vl2`       | Vision-language variant of DeepSeek (with a dedicated image processor), enabling advanced multimodal reasoning on image and text inputs. |
| **Janus-Pro** (1B, 7B)      | `deepseek-ai/Janus-Pro-7B`                     | `janus-pro`       | DeepSeek’s open-source multimodal model capable of both image understanding and generation. Janus-Pro employs a decoupled architecture for separate visual encoding paths, enhancing performance in both tasks. |
| **MiniCPM-V / MiniCPM-o**      | `openbmb/MiniCPM-V-2_6`                          | `minicpmv`           | MiniCPM-V (2.6, ~8B) supports image inputs, and MiniCPM-o adds audio/video; these multimodal LLMs are optimized for end-side deployment on mobile/edge devices. |
| **Llama 3.2 Vision** (11B)     | `meta-llama/Llama-3.2-11B-Vision-Instruct`        | `llama_3_vision`     | Vision-enabled variant of Llama 3 (11B) that accepts image inputs for visual question answering and other multimodal tasks. |
| **Pixtral** (12B, 124B)              | `mistral-community/pixtral-12b`                           | `mistral`     | Pixtral is a vision-language model from Mistral AI that can process both text and images. |
| **LLaVA** (v1.5 & v1.6)        | *e.g.* `liuhaotian/llava-v1.5-13b`               | `vicuna_v1.1`        | Open vision-chat models that add an image encoder to LLaMA/Vicuna (e.g. LLaMA2 13B) for following multimodal instruction prompts. |
| **LLaVA-NeXT** (8B, 72B)       | `lmms-lab/llava-next-72b`                        | `chatml-llava`       | Improved LLaVA models (with an 8B Llama3 version and a 72B version) offering enhanced visual instruction-following and accuracy on multimodal benchmarks. |
| **LLaVA-OneVision**            | `lmms-lab/llava-onevision-qwen2-7b-ov`           | `chatml-llava`       | Enhanced LLaVA variant integrating Qwen as the backbone; supports multiple images (and even video frames) as inputs via an OpenAI Vision API-compatible format. |
| **Gemma 3 (Multimodal)**       | `google/gemma-3-4b-it`                           | `gemma-it`           | Gemma 3’s larger models (4B, 12B, 27B) accept images (each image encoded as 256 tokens) alongside text in a combined 128K-token context. |
| **Kimi-VL** (A3B)              | `moonshotai/Kimi-VL-A3B-Instruct`                | `kimi-vl`            | Kimi-VL is a multimodal model that can understand and generate text from images. |
