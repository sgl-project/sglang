---
title: Flint
description: "Deploy Recursive's Flint with SGLang — a 40B Mixture-of-Experts model for fast, local agentic coding with a 262K context window — on NVIDIA H200, B200, B300, GB300, and DGX Spark."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

Flint runs on the `lmsysorg/sglang:dev-flint` Docker image:

```bash Command
docker pull lmsysorg/sglang:dev-flint
```

For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). The **Docker** toggle in the command panel below produces the full `docker run` command; the **Python** toggle produces the inner `sglang serve ...` to run inside the container.

</Accordion>

Pick your hardware to generate the launch command.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/RecursiveSI/flint.jsx";

<Deployment config={config} />

## 1. Model Introduction

**[Flint](https://huggingface.co/RecursiveSI/flint)** is a new open-weight **40B Mixture-of-Experts** model from Recursive, built for fast, local AI coding. Flint was developed using Recursive's Eureka Machine for automated AI research and is released under Apache 2.0.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left",  padding: "10px 12px", fontWeight: 700}}>Model</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>Total params</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>Context</th>
      <th style={{textAlign: "left",  padding: "10px 12px", fontWeight: 700}}>Use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/RecursiveSI/flint">Flint</a></strong></td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>40B (MoE)</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>262,144</td>
      <td style={{padding: "9px 12px"}}>Fast, local agentic coding</td>
    </tr>
  </tbody>
</table>

**License:** Apache 2.0

**Resources:** [Hugging Face](https://huggingface.co/RecursiveSI/flint).

## 2. Advanced Usage

### 2.1 Reasoning

Launch with `--reasoning-parser flint` (included in the commands above). Thinking is separated into `message.reasoning_content`, and the final answer is returned in `message.content`.

<Accordion title="Reasoning Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="flint",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
)

message = response.choices[0].message
print("=============== Reasoning ===============")
print(message.reasoning_content)
print("=============== Answer ===============")
print(message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
=============== Reasoning ===============
We need answer simple. Need maybe show calculation. User asks "What is 15% of 240?" We can answer. Need be careful maybe just 36. Let's craft concise.
=============== Answer ===============
15% of 240 is **36**.
```

</Accordion>

### 2.2 Tool Calling

Launch with `--tool-call-parser flint` (included in the commands above) to surface structured tool calls via `message.tool_calls`.

<Accordion title="Tool Calling Example (Python)">

```python Example
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="flint",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
)

message = response.choices[0].message
print("=============== Reasoning ===============")
print(getattr(message, "reasoning_content", None))
print("=============== Content ===============")
print(message.content)
print("=============== Tool Calls ===============")
for call in message.tool_calls or []:
    print(call.function.name, json.loads(call.function.arguments))
```

</Accordion>

<Accordion title="Example Output">

```text Output
=============== Reasoning ===============
The user asked for the weather in Paris, and I have a get_weather tool available that takes a city parameter. I should call it with city 'Paris' to get the current weather.
=============== Content ===============
I'll get the current weather for Paris for you.
=============== Tool Calls ===============
get_weather {'city': 'Paris'}
```

</Accordion>
