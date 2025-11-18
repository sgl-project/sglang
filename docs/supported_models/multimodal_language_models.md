# Multimodal Language Models

These models accept multi-modal inputs (e.g., images and text) and generate text output. They augment language models with multimodal encoders.

## Example launch Command

```shell
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \  # example HF/local path
  --host 0.0.0.0 \
  --port 30000 \
```

> See the [OpenAI APIs section](https://docs.sglang.ai/basic_usage/openai_api_vision.html) for how to send multimodal requests.

## Supported models

Below the supported models are summarized in a table.

If you are unsure if a specific architecture is implemented, you can search for it via GitHub. For example, to search for `Qwen2_5_VLForConditionalGeneration`, use the expression:

```
repo:sgl-project/sglang path:/^python\/sglang\/srt\/models\// Qwen2_5_VLForConditionalGeneration
```

in the GitHub search bar.


| Model Family (Variants)    | Example HuggingFace Identifier             | Description                                                                                                                                                                                                     | Notes |
|----------------------------|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| **Qwen-VL** | `Qwen/Qwen3-VL-235B-A22B-Instruct`              | Alibaba's vision-language extension of Qwen; for example, Qwen2.5-VL (7B and larger variants) can analyze and converse about image content.                                                                     |  |
| **DeepSeek-VL2**           | `deepseek-ai/deepseek-vl2`                 | Vision-language variant of DeepSeek (with a dedicated image processor), enabling advanced multimodal reasoning on image and text inputs.                                                                        |  |
| **Janus-Pro** (1B, 7B)     | `deepseek-ai/Janus-Pro-7B`                 | DeepSeek's open-source multimodal model capable of both image understanding and generation. Janus-Pro employs a decoupled architecture for separate visual encoding paths, enhancing performance in both tasks. |  |
| **MiniCPM-V / MiniCPM-o**  | `openbmb/MiniCPM-V-2_6`                    | MiniCPM-V (2.6, ~8B) supports image inputs, and MiniCPM-o adds audio/video; these multimodal LLMs are optimized for end-side deployment on mobile/edge devices.                                                 |  |
| **Llama 3.2 Vision** (11B) | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Vision-enabled variant of Llama 3 (11B) that accepts image inputs for visual question answering and other multimodal tasks.                                                                                     |  |
| **LLaVA** (v1.5 & v1.6)    | *e.g.* `liuhaotian/llava-v1.5-13b`         | Open vision-chat models that add an image encoder to LLaMA/Vicuna (e.g. LLaMA2 13B) for following multimodal instruction prompts.                                                                               |  |
| **LLaVA-NeXT** (8B, 72B)   | `lmms-lab/llava-next-72b`                  | Improved LLaVA models (with an 8B Llama3 version and a 72B version) offering enhanced visual instruction-following and accuracy on multimodal benchmarks.                                                       |  |
| **LLaVA-OneVision**        | `lmms-lab/llava-onevision-qwen2-7b-ov`     | Enhanced LLaVA variant integrating Qwen as the backbone; supports multiple images (and even video frames) as inputs via an OpenAI Vision API-compatible format.                                                 |  |
| **Gemma 3 (Multimodal)**   | `google/gemma-3-4b-it`                     | Gemma 3's larger models (4B, 12B, 27B) accept images (each image encoded as 256 tokens) alongside text in a combined 128K-token context.                                                                        |  |
| **Kimi-VL** (A3B)          | `moonshotai/Kimi-VL-A3B-Instruct`          | Kimi-VL is a multimodal model that can understand and generate text from images.                                                                                                                                |  |
| **Mistral-Small-3.1-24B**  | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Mistral 3.1 is a multimodal model that can generate text from text or images input. It also supports tool calling and structured output. |  |
| **Phi-4-multimodal-instruct**  | `microsoft/Phi-4-multimodal-instruct` | Phi-4-multimodal-instruct is the multimodal variant of the Phi-4-mini model, enhanced with LoRA for improved multimodal capabilities. It supports text, vision and audio modalities in SGLang. |  |
| **MiMo-VL** (7B)           | `XiaomiMiMo/MiMo-VL-7B-RL`                 | Xiaomi's compact yet powerful vision-language model featuring a native resolution ViT encoder for fine-grained visual details, an MLP projector for cross-modal alignment, and the MiMo-7B language model optimized for complex reasoning tasks. |  |
| **GLM-4.5V** (106B) /  **GLM-4.1V**(9B)           | `zai-org/GLM-4.5V`                   | GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning                                                                                                                                                                                                      | Use `--chat-template glm-4v` |
| **DotsVLM** (General/OCR)  | `rednote-hilab/dots.vlm1.inst`             | RedNote's vision-language model built on a 1.2B vision encoder and DeepSeek V3 LLM, featuring NaViT vision encoder trained from scratch with dynamic resolution support and enhanced OCR capabilities through structured image data training. |  |
| **DotsVLM-OCR**            | `rednote-hilab/dots.ocr`                   | Specialized OCR variant of DotsVLM optimized for optical character recognition tasks with enhanced text extraction and document understanding capabilities. | Don't use `--trust-remote-code` |
| **NVILA** (8B, 15B, Lite-2B, Lite-8B, Lite-15B) | `Efficient-Large-Model/NVILA-8B` | `chatml` | NVILA explores the full stack efficiency of multi-modal design, achieving cheaper training, faster deployment and better performance. |

## Video Input Support

SGLang supports video input for Vision-Language Models (VLMs), enabling temporal reasoning tasks such as video question answering, captioning, and holistic scene understanding. Video clips are decoded, key frames are sampled, and the resulting tensors are batched together with the text prompt, allowing multimodal inference to integrate visual and linguistic context.

| Model Family | Example Identifier | Video notes |
|--------------|--------------------|-------------|
| **Qwen-VL** (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-Omni) | `Qwen/Qwen3-VL-235B-A22B-Instruct` | The processor gathers `video_data`, runs Qwen's frame sampler, and merges the resulting features with text tokens before inference. |
| **GLM-4v** (4.5V, 4.1V, MOE) | `zai-org/GLM-4.5V` | Video clips are read with Decord, converted to tensors, and passed to the model alongside metadata for rotary-position handling. |
| **NVILA** (Full & Lite) | `Efficient-Large-Model/NVILA-8B` | The runtime samples eight frames per clip and attaches them to the multimodal request when `video_data` is present. |
| **LLaVA video variants** (LLaVA-NeXT-Video, LLaVA-OneVision) | `lmms-lab/LLaVA-NeXT-Video-7B` | The processor routes video prompts to the LlavaVid video-enabled architecture, and the provided example shows how to query it with `sgl.video(...)` clips. |

Use `sgl.video(path, num_frames)` when building prompts to attach clips from your SGLang programs.

Example OpenAI-compatible request that sends a video clip:

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s happening in this video?"},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://github.com/sgl-project/sgl-test-files/raw/refs/heads/main/videos/jobs_presenting_ipod.mp4"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print(response.text)
```

## Usage Notes

### Performance Optimization

For multimodal models, you can use the `--keep-mm-feature-on-device` flag to optimize for latency at the cost of increased GPU memory usage:

- **Default behavior**: Multimodal feature tensors are moved to CPU after processing to save GPU memory
- **With `--keep-mm-feature-on-device`**: Feature tensors remain on GPU, reducing device-to-host copy overhead and improving latency, but consuming more GPU memory

Use this flag when you have sufficient GPU memory and want to minimize latency for multimodal inference.
