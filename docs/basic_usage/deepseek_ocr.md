# DeepSeek OCR (OCR-1 / OCR-2)

DeepSeek OCR models are multimodal (image + text) models for OCR and document understanding.

## Launch server

```shell
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-OCR-2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000
```

> You can replace `deepseek-ai/DeepSeek-OCR-2` with `deepseek-ai/DeepSeek-OCR`.

## Prompt examples

Recommended prompts from the model card:

```
<image>
<|grounding|>Convert the document to markdown.
```

```
<image>
Free OCR.
```

## OpenAI-compatible request example

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "deepseek-ai/DeepSeek-OCR-2",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>\n<|grounding|>Convert the document to markdown."},
                {"type": "image_url", "image_url": {"url": "https://example.com/your_image.jpg"}},
            ],
        }
    ],
    "max_tokens": 512,
}

response = requests.post(url, json=data)
print(response.text)
```
