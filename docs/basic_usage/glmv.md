# GLM-4.6V / GLM-4.5V Usage

## Launch commands for SGLang

Below are suggested launch commands tailored for different hardware / precision modes

### FP8 (quantised) mode

For high memory-efficiency and latency optimized deployments (e.g., on H100, H200) where FP8 checkpoint is supported:

```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.6V-FP8 \
  --tp 2 \
  --ep 2 \
  --host 0.0.0.0 \
  --port 30000 \
  --keep-mm-feature-on-device
```

### Non-FP8 (BF16 / full precision) mode
For deployments on A100/H100 where BF16 is used (or FP8 snapshot not used):
```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.6V \
  --tp 4 \
  --ep 4 \
  --host 0.0.0.0 \
  --port 30000
```

## Hardware-specific notes / recommendations

- On H100 with FP8: Use the FP8 checkpoint for best memory efficiency.
- On A100 / H100 with BF16 (non-FP8): It’s recommended to use `--mm-max-concurrent-calls` to control parallel throughput and GPU memory usage during image/video inference.
- On H200 & B200: The model can be run “out of the box”, supporting full context length plus concurrent image + video processing.

## Sending Image/Video Requests

### Image input:

```python
import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "zai-org/GLM-4.6V",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
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

### Video Input:

```python
import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "zai-org/GLM-4.6V",
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

## Important Server Parameters and Flags

When launching the model server for **multimodal support**, you can use the following command-line arguments to fine-tune performance and behavior:

- `--mm-attention-backend`: Specify multimodal attention backend. Eg. `fa3`(Flash Attention 3)
- `--mm-max-concurrent-calls <value>`: Specifies the **maximum number of concurrent asynchronous multimodal data processing calls** allowed on the server. Use this to control parallel throughput and GPU memory usage during image/video inference.
- `--mm-per-request-timeout <seconds>`: Defines the **timeout duration (in seconds)** for each multimodal request. If a request exceeds this time limit (e.g., for very large video inputs), it will be automatically terminated.
- `--keep-mm-feature-on-device`: Instructs the server to **retain multimodal feature tensors on the GPU** after processing. This avoids device-to-host (D2H) memory copies and improves performance for repeated or high-frequency inference workloads.
- `--mm-enable-dp-encoder`: Placing the ViT in data parallel while keeping the LLM in tensor parallel consistently lowers TTFT and boosts end-to-end throughput.
- `SGLANG_USE_CUDA_IPC_TRANSPORT=1`: Shared memory pool based CUDA IPC for multi-modal data transport. For significantly improving e2e latency.

### Example usage with the above optimizations:
```bash
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_VLM_CACHE_SIZE_MB=0 \
python -m sglang.launch_server \
  --model-path zai-org/GLM-4.6V \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --tp-size 8 \
  --enable-cache-report \
  --log-level info \
  --max-running-requests 64 \
  --mem-fraction-static 0.65 \
  --chunked-prefill-size 8192 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --mm-enable-dp-encoder \
  --enable-metrics
```

### Thinking Budget for GLM-4.5V / GLM-4.6V

In SGLang, we can implement thinking budget with `CustomLogitProcessor`.

Launch a server with `--enable-custom-logit-processor` flag on. and using `Glm4MoeThinkingBudgetLogitProcessor` in the request likes `GLM-4.6` example in [glm45.md](./glm45.md).
