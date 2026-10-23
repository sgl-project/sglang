---
title: Kev
description: "Deploy the Kev decision models with SGLang. Kev returns calibrated probabilities for typed questions about a document, in sizes from 0.8B to 27B, each on one NVIDIA H200 in BF16."
tag: NEW
---

## Deployment

Every Kev size runs on one GPU in BF16 with TP=1 and needs no extra flag. SGLang reads the checkpoint's `head.pt` and configures the server.

```bash Command
sglang serve --model-path jaredpalmer/kev-27b --host 0.0.0.0 --port 30000
```

Replace the model path with any size in the table below. Each was validated on one H200.

| Model | Hugging Face ID | Checkpoint | Backbone | GPU memory for weights |
| --- | --- | --- | --- | --- |
| Kev-27B | `jaredpalmer/kev-27b` | full weights | Qwen3.8-27B | 47.8 GB |
| Kev-9B | `jaredpalmer/kev-9b` | LoRA adapter | Qwen3.5-9B-Base | 14.9 GB |
| Kev-4B | `jaredpalmer/kev-4b` | LoRA adapter | Qwen3.5-4B-Base | 8.0 GB |
| Kev-0.8B | `jaredpalmer/kev-0.8b` | LoRA adapter | Qwen3.5-0.8B-Base | 1.5 GB |

For a LoRA adapter, SGLang loads the base revision that the adapter's `head.pt` pins and merges the adapter into its weights at startup, in FP32 with one rounding to BF16, as the reference code merges it. The base is downloaded on first use.

## 1. Model introduction

[Kev](https://huggingface.co/collections/jaredpalmer/kev) is a family of decision models published by Jared Palmer under the Apache-2.0 license, with [reference code](https://github.com/jaredpalmer/kev). Each model keeps a Qwen3.5 or Qwen3.8 backbone and replaces the LM head with a pointer head. Each question runs as its own prompt after the shared state. The head compares the final hidden state at each option's closing token with the one at the prompt's last token and returns one logit per option, with no text generation. A temperature stored in `head.pt` calibrates the probabilities.

Pass a text string, JSON object, or JSON array as the state. `/v1/systemone` returns one answer per question: a `choice` over allowed options, a `noul` probability of true, or a `score` over ordered levels. JSON states are flattened into labeled text as the reference code does.

## 2. Configuration tips

- **Route.** Send requests to `/v1/systemone`, or set your System One client's base URL to the server. This includes TypeSafe SDK clients. See the [API reference](/docs/supported-models/decision_models#system-one-compatible-api) for request and response fields. Kev refuses `/v1/decisions`, chat, generation, and `/encode` requests.
- **One prefill per question.** The model reads the state once for every question, as the reference does, so a request with many questions over a long state costs that many prefills. SGLang enables embedding mode and disables radix caching, chunked prefill, and CUDA graphs. Keep TP=1 and PP=1.
- **Answers.** `probabilities` are the softmax of the head's logits divided by the calibrated temperature. `confidence` uses the System One adapter formulas, as Kev's own server does. SGLang does not round the values to four decimals. Answers omit `x_label_mass` because the head scores only the allowed options.
- **Long states.** The reference server accepts states of up to 65,536 tokens. In SGLang each question's prompt must also fit one prefill, 16,384 tokens by default, so longer prompts are refused with an error that gives both lengths. Raising `--max-prefill-tokens` is not supported for this model yet, because the server does not reserve memory for longer unchunked prefills and can run out of memory under load.
- **Inputs.** Requests are text only, so `images` and `chat_template_kwargs` are refused. A `score` question takes up to 10 levels and a `choice` question up to 255 options.
- **Not supported.** Kev-27B v1, the `v1-lora` revision of `jaredpalmer/kev-27b`, is an adapter trained on a BF16 backbone that the reference applies unmerged, so SGLang refuses it and serves the current full weight release instead. Kev-0.5B is a prototype on a Qwen2.5 base and is refused too.

## 3. Advanced usage

### 3.1 TypeSafe SDK

```bash Command
pip install typesafe-sdk
```

<Accordion title="Text decision example (Python)">

```python Example
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://localhost:30000", model="kev-latest")
response = client.system_one(
    state="I was charged twice for order 1182. Please refund one of the charges.",
    questions={
        "team": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Charges and refunds", "shipping": "Deliveries", "returns": "Exchanges"},
        ),
        "urgent": Noul(instructions="Does this need a reply today?"),
    },
)
print(response.choices["team"].choice, response.nouls["urgent"].noul)
```

</Accordion>

<Accordion title="Example output">

Captured with Kev-27B on one H200. Probabilities can vary with the model size, hardware, and numerical backend.

```text Output
billing 0.7591662905989526
```

</Accordion>
