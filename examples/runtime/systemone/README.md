# Jev's `/v1/systemone` on an SGLang server

`systemone_server.py` serves [TypeSafe's Jev API](https://docs.typesafe.ai/api)
(`choice` / `score` / `noul` questions about a `state`, answered with probabilities
and a confidence, no generated text) in front of any chat model on SGLang. Each
question becomes a prompt that ends where the answer label starts; one `/v1/score`
call with `label_token_ids` returns the label distribution for every question.

## Run it

```bash
# 1. Any chat model. On Apple Silicon (MLX backend) add --mlx-enable-sampling,
#    which is what enables output logprobs there.
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000

# 2. The proxy (needs the model's tokenizer for its chat template).
python systemone_server.py --upstream http://127.0.0.1:30000 \
    --model Qwen/Qwen2.5-0.5B-Instruct --port 8300

# 3. A request in Jev's shape.
curl -s localhost:8300/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, invoicing, refunds",
                                "technical": "Bugs, outages, integrations",
                                "sales": "Pricing, upgrades, new accounts"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
  }
}'
```

Response (Qwen2.5-0.5B-Instruct, Apple M2):

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "answers": {
    "department": {"type": "choice", "choice": "technical",
                   "probabilities": {"billing": 0.037139, "technical": 0.957834, "sales": 0.005026},
                   "confidence": 0.936752},
    "frustration": {"type": "score", "score": 0.993795,
                    "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                    "probabilities": {"0": 0.00669, "1": 0.992826, "2": 0.000485},
                    "confidence": 0.989239},
    "is_urgent": {"type": "noul", "noul": 0.914901}
  },
  "usage": {"input_tokens": 271, "output_tokens": 0}
}
```

`state`, `instructions` and `criteria` may be strings or JSON structure (objects and
arrays are rendered as JSON in the prompt). Schema errors return HTTP 400 with an
`error` object; upstream failures return 502.

## How each answer is computed

- **choice**: options are labelled `A`, `B`, `C`, … in the prompt. `probabilities` is the
  softmax over the label tokens, `choice` is the argmax.
- **score**: levels are labelled the same way, in order. `score = Σ i · p_i` over level
  indices, and `legend` maps each index back to its description.
- **noul**: the labels are `Yes` and `No`; `noul` is P(`Yes`).
- **confidence** (choice and score): TypeSafe's published formula,
  `clamp((n · max(p) − 1) / (n − 1), 0, 1)`; 0 for uniform, 1 for certain.
- `--temperature` scales the label logits before the softmax; fit it on labelled data.

## Differences from the hosted model

- Probabilities come from a generative model's next-token distribution, not a model
  trained for calibrated decisions; check them against labels before thresholding.
- At most 26 options (letter labels; Jev allows 255) and 10 levels (as in Jev).
- Labels must tokenize to distinct leading tokens; otherwise 400.
- `model` in the request is ignored; the served model answers.
- Text state only.
