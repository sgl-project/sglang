# Classification API

This document describes the `/v1/classify` API endpoint implementation in SGLang, which is compatible with vLLM's classification API format.

## Overview

The classification API allows you to classify text inputs using classification models. This implementation follows the same format as vLLM's 0.7.0 classification API.

## API Endpoint

```
POST /v1/classify
```

## Request Format

```json
{
  "model": "model_name",
  "input": "text to classify"
}
```

### Parameters

- `model` (string, required): The name of the classification model to use
- `input` (string, required): The text to classify
- `user` (string, optional): User identifier for tracking
- `rid` (string, optional): Request ID for tracking
- `priority` (integer, optional): Request priority

## Response Format

```json
{
  "id": "classify-9bf17f2847b046c7b2d5495f4b4f9682",
  "object": "list",
  "created": 1745383213,
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "data": [
    {
      "index": 0,
      "label": "Default",
      "probs": [0.565970778465271, 0.4340292513370514],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```

### Response Fields

- `id`: Unique identifier for the classification request
- `object`: Always "list"
- `created`: Unix timestamp when the request was created
- `model`: The model used for classification
- `data`: Array of classification results
  - `index`: Index of the result
  - `label`: Predicted class label
  - `probs`: Array of probabilities for each class
  - `num_classes`: Total number of classes
- `usage`: Token usage information
  - `prompt_tokens`: Number of input tokens
  - `total_tokens`: Total number of tokens
  - `completion_tokens`: Number of completion tokens (always 0 for classification)
  - `prompt_tokens_details`: Additional token details (optional)

## Example Usage

### Using curl

```bash
curl -v "http://127.0.0.1:8000/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": "Loved the new café—coffee was great."
  }'
```

### Using Python

```python
import requests
import json

# Make classification request
response = requests.post(
    "http://127.0.0.1:8000/v1/classify",
    headers={"Content-Type": "application/json"},
    json={
        "model": "jason9693/Qwen2.5-1.5B-apeach",
        "input": "Loved the new café—coffee was great."
    }
)

# Parse response
result = response.json()
print(json.dumps(result, indent=2))
```

## Supported Models

The classification API works with any classification model supported by SGLang, including:

### Classification Models (Multi-class)
- `LlamaForSequenceClassification` - Multi-class classification
- `Qwen2ForSequenceClassification` - Multi-class classification
- `Qwen3ForSequenceClassification` - Multi-class classification
- `BertForSequenceClassification` - Multi-class classification
- `Gemma2ForSequenceClassification` - Multi-class classification

**Label Mapping**: The API automatically uses the `id2label` mapping from the model's `config.json` file to provide meaningful label names instead of generic class names. If `id2label` is not available, it falls back to `LABEL_0`, `LABEL_1`, etc., or `Class_0`, `Class_1` as a last resort.

### Reward Models (Single score)
- `InternLM2ForRewardModel` - Single reward score
- `Qwen2ForRewardModel` - Single reward score
- `LlamaForSequenceClassificationWithNormal_Weights` - Special reward model

**Note**: The `/classify` endpoint in SGLang was originally designed for reward models but now supports all non-generative models. Our `/v1/classify` endpoint provides a standardized vLLM-compatible interface for classification tasks.

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid request format or missing required fields
- `500 Internal Server Error`: Server-side processing error

Error response format:
```json
{
  "error": "Error message",
  "type": "error_type",
  "code": 400
}
```

## Implementation Details

The classification API is implemented using:

1. **Rust Model Gateway**: Handles routing and request/response models in `sgl-model-gateway/src/protocols/spec.rs`
2. **Python HTTP Server**: Implements the actual endpoint in `python/sglang/srt/entrypoints/http_server.py`
3. **Classification Service**: Handles the classification logic in `python/sglang/srt/entrypoints/openai/serving_classify.py`

## Testing

Use the provided test script to verify the implementation:

```bash
python test_classify_api.py
```

## Compatibility

This implementation is compatible with vLLM's classification API format, allowing seamless migration from vLLM to SGLang for classification tasks.
