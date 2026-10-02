---
title: pplx-decider-v1-27b
description: "Deploy pplx-decider-v1-27b with SGLang. Perplexity's decision model, fine-tuned from Qwen3.8-27B, answers choice, yes or no, and score questions on one NVIDIA H200."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

Until a release includes pplx-decider support, use a [nightly build](/docs/get-started/install#nightly-builds). The nightly image carries it:

```bash Command
docker pull lmsysorg/sglang:dev
```

</Accordion>

The recipe below serves the model in BF16 on one H200 with no extra flag. SGLang reads the checkpoint's `decision_config.json`, loads its decision readout, and answers on `/v1/systemone` with the prompt, answer codes, and calibrated temperature the model was trained with.

import { Deployment } from "/src/snippets/_deployment.jsx"
import { config } from "/src/snippets/configs/perplexity-ai/pplx-decider-v1-27b.jsx"

<Deployment config={config} />

## 1. Model Introduction

**pplx-decider-v1-27b** is a decision model from Perplexity, fine-tuned from Qwen3.8-27B under the Apache-2.0 license. It replaces the language model head with a readout over 255 answer codes, and answers choice, yes or no (`noul`), and score questions about a text state and optional images with a probability for every option, without generating text.

**Resources:** [HuggingFace](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b).

## 2. Configuration Tips

- **Route.** Send requests to `/v1/systemone`, the System One API the checkpoint was trained for. `/v1/decisions` refuses this checkpoint because its labels and prompt are not the ones the readout learned, and chat or generation requests are not meaningful without a language model head.
- **Clients.** Clients of the System One API, including the TypeSafe SDKs, work by pointing their base URL at the server. See [System One compatible API](/docs/supported-models/decision_models#system-one-compatible-api) for the request and response reference.
- **Images.** Pass each image in `images` as base64 bytes or a URL. They precede the text in every question, as in the checkpoint's reference code.

## 3. Advanced Usage

### 3.1 Text Decisions

<Accordion title="Text Decision Example (Python)">

```python Example
import requests

response = requests.post(
    "http://localhost:30000/v1/systemone",
    json={
        "model": "perplexity-ai/pplx-decider-v1-27b",
        "state": "My Stripe integration keeps failing. I'm losing sales. Please help ASAP.",
        "questions": {
            "routing": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "billing": "Charges and refunds",
                    "technical_support": "Integration errors",
                    "sales": "Questions about buying a product",
                },
            },
            "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
        },
    },
    timeout=60,
)
response.raise_for_status()
for name, answer in response.json()["answers"].items():
    print(name, answer)
```

</Accordion>

<Accordion title="Example Output">

```text Output
routing {'type': 'choice', 'choice': 'technical_support', 'confidence': 0.987854396908662, 'probabilities': {'billing': 0.004840538662952211, 'technical_support': 0.9919029312724412, 'sales': 0.0032565300646064375}}
urgency {'type': 'noul', 'noul': 0.9929980993021188}
```

</Accordion>

### 3.2 Image Decisions

The example output below came from a mostly red test image.

<Accordion title="Image Decision Example (Python)">

```python Example
import base64
from pathlib import Path

import requests

image = base64.b64encode(Path("screenshot.png").read_bytes()).decode("ascii")
response = requests.post(
    "http://localhost:30000/v1/systemone",
    json={
        "model": "perplexity-ai/pplx-decider-v1-27b",
        "state": "Look at the supplied image.",
        "images": [image],
        "questions": {
            "color": {
                "type": "choice",
                "instructions": "What is the dominant color?",
                "criteria": {"red": "Red", "green": "Green", "blue": "Blue", "other": "Another color"},
            }
        },
    },
    timeout=60,
)
response.raise_for_status()
print(response.json()["answers"]["color"])
```

</Accordion>

<Accordion title="Example Output">

```text Output
{'type': 'choice', 'choice': 'red', 'confidence': 0.9498445161931509, 'probabilities': {'red': 0.9623833871448632, 'green': 0.009006826727044857, 'blue': 0.00926546931819573, 'other': 0.01934431680989631}}
```

</Accordion>
