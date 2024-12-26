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
