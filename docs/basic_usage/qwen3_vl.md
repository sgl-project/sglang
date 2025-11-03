# Qwen3-VL Usage

[Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl)
is Alibaba’s latest multimodal large language model with strong text, vision, and reasoning capabilities.
SGLang supports Qwen3-VL Family of models with Image and Video input support.

## Launch Qwen3-VL with SGLang

To serve the model:

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0
  --tp 4
```

## Sending Image/Video Requests

#### Image input:

```python
import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
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

#### Video Input:

```python
import requests

url = f"http://localhost:30000/v1/chat/completions"

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
