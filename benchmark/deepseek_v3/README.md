# SGLang v0.4.1 - DeepSeek V3 Support

We're excited to announce [SGLang v0.4.1](https://github.com/sgl-project/sglang/releases/tag/v0.4.1), which now supports [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) - currently the strongest open-source LLM, even surpassing GPT-4o.

The SGLang and DeepSeek teams worked together to get DeepSeek V3 FP8 running on NVIDIA and AMD GPU **from day one**. We've also supported MLA optimization and DP attention before, making SGLang one of the best open-source LLM engines for running DeepSeek models.

Special thanks to Meituan's Search & Recommend Platform Team and Baseten's Model Performance Team for their support, and DataCrunch for providing GPU resources.

## Hardware Recommendation
- 8 x NVIDIA H200 GPUs

## Installation & Launch

### Using Docker (Recommended)
```bash
docker run --gpus all --shm-size 32g -p 30000:30000 -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3-Base --enable-dp-attention --tp 8 --trust-remote-code --port 30000
```

### Using pip
```bash
# Installation
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer

# Launch
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3-Base --enable-dp-attention --tp 8 --trust-remote-code
```

If you see errors when launching the server, please check if it has finished downloading the weights. It is recommended to download the weights before launching, or to launch multiple times until all the weights have been downloaded.

### Example with OpenAI API

```python3
import openai
client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

# Chat completion
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print(response)
```

## DeepSeek V3 optimization plan

https://github.com/sgl-project/sglang/issues/2591

## Appendix

SGLang is the inference engine officially recommended by the DeepSeek team.

https://github.com/deepseek-ai/DeepSeek-V3/tree/main?tab=readme-ov-file#62-inference-with-sglang-recommended
