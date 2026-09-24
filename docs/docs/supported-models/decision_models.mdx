---
title: Decision models
description: Answer typed choice, score, and yes or no questions with a probability for every option from a chat model, without generating text.
---

`/v1/decisions` turns a chat model into a decision model. You send an input and a list of typed questions, and each answer comes back with the probability of every option, read from the model's next-token scores at the answer position. No text is generated and no output is parsed. It needs no special checkpoint: any generation model served with a Jinja chat template can answer, as long as the checks below pass for it.

<Note>
  `/v1/decisions` is an SGLang extension under `/v1`, like `/v1/score` and `/v1/rerank`. It is not part of the OpenAI API, so call it over HTTP rather than through an OpenAI SDK method. Until a release contains it, install a [nightly build](/docs/get-started/install#nightly-builds), which is built from the main branch.
</Note>

## Supported models

| Model | Example HuggingFace identifier | Notes |
| --- | --- | --- |
| **Qwen3.8-27B** | `Qwen/Qwen3.8-27B` | Validated on one H200 in BF16, with and without NEXTN speculative decoding. See the [model page](/cookbook/autoregressive/Qwen/Qwen3.8-27B). |
| **Qwen3.5-35B-A3B** | `Qwen/Qwen3.5-35B-A3B` | Validated on one H200 in BF16. See the [model page](/cookbook/autoregressive/Qwen/Qwen3.5). |

Other chat models are served when the answer labels are single tokens at the answer position and the answer does not start inside a reasoning block. The server checks these per request and returns a 400 that names the reason when one fails, see [Errors](#errors). It does not check how the chat template renders the question or where the template itself would put an answer, so inspect `prompt_token_ids` from `return_prompt_token_ids` before relying on a new model.

## Launch command

```bash Command
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --host 127.0.0.1 --port 30000
```

No extra flag is needed. The same server keeps serving `/v1/chat/completions` and `/generate`, and decisions can run alongside that traffic. A server launched with `--enable-mis` or `--dllm-algorithm`, or one that uses a built-in conversation template instead of the tokenizer's Jinja template, refuses decisions.

## Example request

```python Example
import requests

URL = "http://127.0.0.1:30000"

response = requests.post(
    f"{URL}/v1/decisions",
    json={
        "input": "I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales.",
        "questions": [
            {
                "id": "team",
                "type": "choice",
                "question": "Which team should handle this ticket?",
                "options": [
                    {"name": "billing", "description": "Payment or subscription issues"},
                    {"name": "technical", "description": "Bugs or integration problems"},
                    {"name": "sales", "description": "Pricing or account questions"},
                ],
            },
            {
                "id": "frustration",
                "type": "score",
                "question": "How frustrated is the customer?",
                "levels": ["Calm", "Frustrated but civil", "Very angry"],
            },
            {
                "id": "urgent",
                "type": "yes_no",
                "question": "The customer needs an answer today.",
            },
        ],
    },
    timeout=60,
)
response.raise_for_status()
for question_id, answer in response.json()["answers"].items():
    print(question_id, answer)
```

Each answer contains:

- `type`: the question type, `choice`, `score`, or `yes_no`.
- `probabilities`: one value per option name, level index, or `yes` and `no`, summing to 1. For a yes or no question, `probabilities["yes"]` is the answer.
- `choice` for a choice question (the most probable option) or `score` for a score question (the probability-weighted mean level index).
- `label_mass`: the full-vocabulary probability of the answer labels at the answer position. A low value means the model puts most of its probability outside the offered answers.

Questions are answered independently and scored together in one batch. To route the ticket in this example, send it to the chosen team, and hand it to a person when the top probability falls below a threshold you validate on your own labeled tickets.

## How answers are computed

The server renders each question as one user message with the model's chat template and thinking turned off, and labels the answers `A` to `Z` for options, `0` to `9` for levels, and `yes` and `no`. It checks that each label is one distinct token at the answer position, then runs one prefill pass per question through the scoring path of `/v1/score` and reads the next-token log-probabilities of the labels over the full vocabulary. For a yes or no question with log-probabilities `lp_yes` and `lp_no` and request temperature `T`:

- `probabilities["yes"] = exp(lp_yes / T) / (exp(lp_yes / T) + exp(lp_no / T))`, which equals a softmax of the two label logits divided by `T`, because the vocabulary normalizer cancels.
- `label_mass = exp(lp_yes) + exp(lp_no)`, which does not depend on `T`.

Choice and score questions work the same way over their labels. None of these values is a calibrated probability that the decision is correct. Validate any threshold on labeled data from your workload.

## Request and response reference

Request fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `input` | string, object, or array | What the questions are about. Objects and arrays are rendered as compact JSON. Must not be blank. |
| `questions` | list of questions | At least one. Each question has a unique `id`, which keys its answer. |
| `temperature` | number above 0, default 1 | Divides the label logits before the softmax over the labels. It does not change `label_mass`. |
| `chat_template_kwargs` | object, default empty | Extra chat template arguments, applied after `--default-chat-template-kwargs`. The thinking toggle that the chat template or its reasoning parser names stays off in either case. |
| `prompt_format_version` | integer, optional | Pins the prompt wording. A server that serves another version returns 400. |
| `return_prompt_token_ids` | boolean, default false | Adds `prompt_token_ids` and `label_token_ids` to each answer. |
| `model` | string, optional | Echoed in the response, as `default` when omitted. It does not select a LoRA adapter, and the `base:adapter` form returns 400. |

Question fields:

| Type | Fields | Labels |
| --- | --- | --- |
| `choice` | `id`, `question`, and `options`: 2 to 26 objects with a `name` and an optional `description` | `A` to `Z` in list order |
| `score` | `id`, `question`, and `levels`: 2 to 10 level descriptions, lowest first | `0` to `9` |
| `yes_no` | `id`, `question`, and optional `yes` and `no` descriptions | `yes` and `no` |

`question`, descriptions, and levels can also be JSON objects or arrays. Each question becomes one user message: the input, a blank line, the question line, one line per option, level, or described yes or no answer, and a closing instruction to answer with one label only. An option renders as `A: billing` or `A: billing - <description>`, a level as `0: <description>`, and a described yes or no answer as `yes: <description>`.

The response has `object` set to `decisions`, `model`, `prompt_format_version`, `answers` keyed by question id, and `usage` with `prompt_tokens` for all questions and `completion_tokens` 0.

## Errors

The request fails with HTTP 400 for:

- an unknown question type or an unknown field anywhere in the request
- option or level counts outside the ranges above
- a blank input, question, question id, or level
- a repeated question id, or option names that are blank, contain control or line break characters, or repeat another name after trimming and case folding
- a label that is not one distinct token at the answer position for the served tokenizer
- a prompt that does not fit the model's context length
- a chat template that always thinks before answering or starts every answer with a reasoning block, `chat_template_kwargs` that set the thinking toggle to anything but `false`, a rendered prompt that leaves a reasoning block open, or a model whose reasoning parser expects answers to start inside a reasoning block when the rendered prompt does not close one
- a `prompt_format_version` other than the served one, or a `model` that names a LoRA adapter
- a server launched with `--enable-mis` or `--dllm-algorithm`, or one that uses a built-in conversation template, whether named with `--chat-template`, loaded from a JSON template file, or inferred from the model path
- a model whose chat route uses a built-in encoder instead of a chat template, a tokenizer that does not encode the rendered chat text back to the same ids, or a model that is not a generation model
- a chat template that raises an error for the question message, or a server launched with `--skip-tokenizer-init`

Errors about one question name its id or its position in `questions`.

## Pin and replay the prompt

The server owns the prompt wording of `/v1/decisions` and versions it. Every response carries `prompt_format_version`, and a change to the wording ships as a new version. Send `prompt_format_version` to fail loudly instead of receiving answers from different wording after a server upgrade.

To keep an answer reproducible independently of the server's wording, ask for the scored ids and replay them through `/v1/score`, which only scores the ids you send:

```python Example
import requests

URL = "http://127.0.0.1:30000"

body = {
    "input": "The integration keeps failing.",
    "questions": [{"id": "urgent", "type": "yes_no", "question": "The customer needs an answer today."}],
    "prompt_format_version": 1,
    "return_prompt_token_ids": True,
}
answer = requests.post(f"{URL}/v1/decisions", json=body, timeout=60).json()["answers"]["urgent"]
replay = requests.post(
    f"{URL}/v1/score",
    json={
        "query": [],
        "items": [answer["prompt_token_ids"]],
        "label_token_ids": [answer["label_token_ids"]],
        "apply_softmax": True,
        "return_token_logprobs": True,
    },
    timeout=60,
).json()
print(answer["probabilities"], replay["scores"][0])
```

The replayed scores equal the answer's probabilities when both requests use the same `temperature` and run in the same cache state. If the decision set `temperature`, send the same value in the `/v1/score` body. A replay right after the decision reuses its cached prefix, so it can differ slightly, as described in [Reproducibility](#reproducibility).

## Thinking models

Qwen3.8-27B and Qwen3.5-35B-A3B think by default. `/v1/decisions` turns thinking off for every question, refuses a request that turns it back on, and refuses a rendered prompt that leaves a reasoning block open, so the answer is never read inside the reasoning. Chat requests to the same server keep the model's default, and launching with `--reasoning-parser qwen3` affects chat only.

## Reproducibility

These models mix full and linear attention. Answer probabilities can move by up to several hundredths (0.07 in our checks), and `label_mass` by up to about 0.14, between cold and prefix-cached requests and across batch compositions, while the chosen option stayed the same in our checks. On Qwen3.5-35B-A3B, `--disable-radix-cache` gave identical values across sequential repeats, at the cost of prefix reuse.

## System One compatible API

`POST /v1/systemone` serves the same decisions in the request and response shape of the System One API, with a `state`, a map of `noul`, `choice`, and `score` questions keyed by your ids, and one answer per id. Clients written for that API, including the official TypeSafe SDKs, get decisions by pointing their base URL at the server, with the exceptions listed below. The route uses the same rendering, label checks, and scoring as `/v1/decisions`, and refuses the same servers, chat templates, and prompts, listed in [Errors](#errors). Its request fields are checked as described below.

```bash Command
pip install typesafe-sdk
```

```python Example
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

# A server started without --api-key accepts any key, so a placeholder works.
client = TypeSafeClient(base_url="http://127.0.0.1:30000", api_key="unused")
result = client.system_one(
    "I've been trying to connect my Stripe account for 3 days and the integration keeps failing.",
    {
        "team": Choice(
            instructions="Which team should handle this ticket?",
            criteria={"billing": None, "technical": "Bugs or integration problems", "sales": None},
        ),
        "urgent": Noul(instructions="The customer needs an answer today."),
        "frustration": Score(
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated but civil", "Very angry"],
        ),
    },
)
for name, answer in result.answers.items():
    print(name, answer)
```

What to expect from this route:

- Any `model` name is accepted, including the SDK default `jev-latest`, unless it names a LoRA adapter after a colon, as in `base:adapter`. The response `model` is always the served model.
- A `noul` answer is the probability of yes. A `choice` answer has the most probable option and a `score` answer the probability-weighted mean level, each with `probabilities` and `confidence`, and a score answer echoes its levels in `legend`.
- `confidence` follows the formulas TypeSafe publishes for its adapter. It is a statistic of the returned probabilities, which are renormalized over the answer labels and ignore probability outside them. It is not calibrated, it can read 0 for a score split between distant levels, and thresholds tuned on another provider may not transfer.
- Each answer also carries `x_label_mass`, the full-vocabulary probability of its labels, which `/v1/decisions` returns as `label_mass`. The Python SDK drops fields it does not define, so read it from `result.raw_http_response.json()`.
- A choice takes 1 to 255 options and a score 1 to 10 levels. Option names that are blank, contain control or line break characters, or repeat another name after trimming and case folding are refused, and so are blank or null levels, as on `/v1/decisions`.
- Beyond 26 options, every option gets a two-letter label (`AA`, `AB`, and so on) checked to be one token for the served tokenizer. These labels include common words and have unequal priors, so answers above 26 options can depend on option order. More than 26 options also need a Hugging Face tokenizer that reports its added tokens, and a chat template with an added token between the message and the answer position whose following text tokenizes the same on its own, as the Qwen models do with `</think>`. Without them, a choice is limited to 26 options.
- `instructions` are optional. A choice or score question without them drops its question line, and a `noul` question is answered from its `true` and `false` descriptions. A `noul` question with neither, such as the SDK's bare `Noul()`, is refused.
- Unknown top-level fields are ignored. Unknown keys inside a question or inside `noul` criteria are refused, and so are the `/v1/decisions` fields `temperature`, `prompt_format_version`, and `return_prompt_token_ids` with any value other than null. `chat_template_kwargs` is accepted as an SGLang extension.
- Unlike `/v1/decisions`, an empty `state` and empty or blank question ids are accepted, as the System One schema allows.
- Invalid requests return 422 with the location of the error, other refusals return 400, and neither is retried by the SDKs.
- `usage.input_tokens` counts the prompt of every question, so the state is counted once per question, and `usage.output_tokens` is 0.
- The SDKs' `models.list()` and response `request_id` are not supported, because `GET /v1/models` keeps its OpenAI format and no request id header is sent.
- The SDKs time out after 10 seconds and retry, so raise their timeout for requests with many questions over a long state. A request whose client gives up still finishes its encoding and part of its prefill, so each retry repeats at least the encoding.

## Build decisions without `/v1/decisions`

When you need your own prompt or labels, build the prompt on the client and score it through `/v1/score`, which takes token ids and one label list per item. This needs `transformers` and `jinja2` on the client. Every label must be exactly one token at the answer position:

```python Example
import requests
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.8-27B"
URL = "http://127.0.0.1:30000"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

options = ["billing", "technical", "sales"]
letters = ["A", "B", "C"]
prompt = (
    "I've been trying to connect my Stripe account for 3 days and the integration keeps failing.\n\n"
    "Question: Which team should handle this ticket?\n"
    + "\n".join(f"{letter}: {name}" for letter, name in zip(letters, options))
    + "\nAnswer with the letter of one option only."
)
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False, tokenize=False
)
input_ids = tokenizer.encode(text, add_special_tokens=False)
label_ids = []
for letter in letters:
    ids = tokenizer.encode(text + letter, add_special_tokens=False)
    if ids[: len(input_ids)] != input_ids or len(ids) != len(input_ids) + 1:
        raise ValueError(f"Label {letter} is not one token at the answer position")
    label_ids.append(ids[-1])

scores = requests.post(
    f"{URL}/v1/score",
    json={"query": [], "items": [input_ids], "label_token_ids": [label_ids], "apply_softmax": True},
    timeout=60,
).json()["scores"][0]
print(dict(zip(options, scores)))
```

`enable_thinking=False` renders a closed think block, so the answer token follows it directly. Do not score a prompt that leaves the think block open, because the scored position then falls inside the reasoning.

Structured output gives the same decision through generation. A one-token regex such as `(A|B|C)` with `temperature` 0 and logprobs returns probabilities renormalized over the labels, equal to the `/v1/score` values for one-character labels in the same cache state. A JSON schema with `enum` fields fills several fields in one response, but gives no clean per-option probability, because an enum value can span several tokens. See [Structured outputs](/docs/advanced_features/structured_outputs) for the request formats.

## Limitations

- Labels must be one token at the answer position. `/v1/decisions` assigns one-token labels for you, and a client-built prompt must do the same.
- For `yes_no`, `label_mass` counts only the lowercase `yes` and `no` tokens. The model also puts probability on `Yes` and `No`, so this value reads lower than for choice and score questions even on clear cases, while `probabilities["yes"]` is unaffected.
- The server tells from the chat template and its reasoning parser whether answers could start inside a reasoning block, using the parser's reasoning tags in the generation prompt and in the template's own rendering of a finished answer. Reasoning that shows neither is not detected, so check `prompt_token_ids` for such a model. Some templates of models that do not reason still contain reasoning tags without ever closing a block in the generation prompt, and those models are refused.
- A default temperature in `--preferred-sampling-params` can reach scoring requests and scale both the probabilities and `label_mass` when a decision shares a batch with generation or runs under speculative decoding. Leave it unset for decisions.
- The server tokenizes each question's full prompt once on the HTTP event loop, as `/v1/score` does for its items, and lets other requests run between questions. Tokenizers whose label check falls back to the full prompt encode it once more per label.
- A decision takes a single `input`, not a chat history. To decide about a conversation, pass the history as the input text or as a JSON array.
