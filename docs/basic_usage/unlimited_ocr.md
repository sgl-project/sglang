# Baidu Unlimited-OCR

Unlimited-OCR is a multimodal OCR model for document parsing. It uses a
sliding-window language backbone while keeping the prompt and image tokens
visible during decode.

## Launch server

```shell
python -m sglang.launch_server \
  --model-path baidu/Unlimited-OCR \
  --attention-backend fa3 \
  --page-size 1 \
  --context-length 32768 \
  --enable-custom-logit-processor \
  --host 0.0.0.0 \
  --port 30000
```

CUDA graph and the overlap scheduler can stay enabled. The model uses a
prefill-aware sliding-window KV cache path, so prompt and image tokens remain
available throughout long OCR generations.

## Image modes

Use `images_config.image_mode` to select the OCR image mode:

- `tiny`
- `small`
- `base`
- `gundam`

`gundam` uses more image tokens and is useful for high-detail document parsing.
Lower modes reduce prefill cost for simpler images.

## OpenAI-compatible request example

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "baidu/Unlimited-OCR",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "document parsing."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/your_document.png"},
                },
            ],
        }
    ],
    "images_config": {"image_mode": "base"},
    "temperature": 0,
    "max_tokens": 2048,
}

response = requests.post(url, json=data)
print(response.json()["choices"][0]["message"]["content"])
```

## curl example

```shell
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "baidu/Unlimited-OCR",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "document parsing."},
        {"type": "image_url", "image_url": {"url": "https://example.com/your_document.png"}}
      ]
    }],
    "images_config": {"image_mode": "base"},
    "temperature": 0,
    "max_tokens": 2048
  }'
```
