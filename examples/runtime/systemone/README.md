# Typed decisions: Jev's `/v1/systemone` on an SGLang server

[TypeSafe's Jev API](https://docs.typesafe.ai/api) takes a `state` and a map of typed
`questions` and returns one typed answer per question, with no generated text:

| question `type` | answer |
|---|---|
| `choice` | the chosen option, a probability per option, `confidence` |
| `score` | a probability-weighted position over ordered levels, a probability per level, `confidence` |
| `noul` | the probability that a yes/no statement is true |

`systemone_server.py` serves that API in front of any instruction-tuned model running
on SGLang. Each question becomes a chat prompt that ends where the answer label would
start, and the answer is read from the model's next-token distribution over the label
tokens. That is what SGLang's `/v1/score` endpoint computes (`label_token_ids` +
`apply_softmax`), and one `/v1/score` call scores all questions of a request as one
batch, so every question is evaluated in parallel and in isolation against the same
state, as in Jev.

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
- **confidence** (choice and score) uses the formula TypeSafe publishes for its own
  answers: `clamp((n · max(p) − 1) / (n − 1), 0, 1)` for `n` options, so a uniform
  distribution scores 0 and a certain one scores 1.
- `--temperature` scales the label logits before the softmax (SGLang's `/v1/score`
  `temperature`). Fit it on a labelled sample before thresholding on probabilities.

## Differences from the hosted model

- The probabilities are a generative model's next-token distribution, not the output of
  a model trained for calibrated decisions. Treat the numbers as uncalibrated until you
  have checked them against labels.
- Options are limited to 26 (one letter label each); Jev allows 255. Levels are limited
  to 10, as in Jev.
- Answer labels must tokenize to distinct leading tokens under the served model's
  tokenizer; the proxy checks this and returns 400 if they do not.
- `model` in the request is accepted and ignored; the served model answers, and its name
  is returned.
- Images and other non-text state are not supported.
