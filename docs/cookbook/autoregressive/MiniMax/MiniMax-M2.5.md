---
title: MiniMax-M2.5
metatags:
    description: "Deploy MiniMax-M2.5 with SGLang - community contribution guide for MiniMax M2.5 model deployment."
tag: NEW
---

## 1. Model Introduction

[MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) is a powerful language model developed by MiniMax, built for real-world productivity with state-of-the-art performance across coding, reasoning, agentic tasks, and tool use.

As the latest iteration in the MiniMax model series, MiniMax-M2.5 achieves comprehensive enhancements across multiple domains. Details are as follows:

- **Superior coding performance**: Achieves 79.7 on Droid and 76.1 on OpenCode, surpassing Opus 4.6 (78.9 and 75.9 respectively). Strong results on SWE-bench Verified, SWE-bench Multilingual, SWE-bench-pro, and Multi-SWE-bench.
- **Advanced reasoning**: Demonstrates strong performance on AIME25 and other reasoning benchmarks, with robust tool use during inference.
- **More capable agents**: Excels in agentic tasks including web browsing (BrowseComp, Wide Search), information retrieval (RISE), and complex tool use scenarios (Terminal Bench 2, MEWC, Finance Modeling).
- **Real-world productivity**: Designed for production-grade workloads with strong performance on practical coding, data analysis, and multi-step reasoning tasks.

For more details, please refer to the [official MiniMax-M2.5 announcement](https://www.minimax.io/news/minimax-m25).

## 2. SGLang Installation

SGLang offers multiple installation methods. You can choose the most suitable installation method based on your hardware platform and requirements.

Please refer to the [official SGLang installation guide](https://docs.sglang.ai/get_started/install.html) for installation instructions.

## 3. Model Deployment

This section provides deployment configurations optimized for different hardware platforms and use cases.

### 3.1 Basic Configuration

**Interactive Command Generator**: Use the configuration selector below to automatically generate the appropriate deployment command for your hardware platform, deployment strategy, and feature capabilities.

import { MiniMaxM25Deployment } from '/src/snippets/autoregressive/minimax-m25-deployment.jsx'

<MiniMaxM25Deployment />

### 3.2 Configuration Tips

**Key Parameters:**

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "33.3%"}} />
    <col style={{width: "33.3%"}} />
    <col style={{width: "33.3%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Parameter</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Description</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Recommended Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`--tool-call-parser`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Tool call parser for function calling support</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>`minimax-m2`</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`--reasoning-parser`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Reasoning parser for thinking mode</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>`minimax-append-think`</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`--trust-remote-code`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Required for MiniMax model loading</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Always enabled</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`--mem-fraction-static`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Static memory fraction for KV cache</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>`0.85`</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`--tp-size`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Tensor parallelism size</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>`4` (4-GPU) or `8` (8-GPU)</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`--ep-size`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Expert parallelism size</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>`8` (for 8-GPU deployment)</td>
    </tr>
  </tbody>
</table>

**Hardware Requirements:**

- **4-GPU deployment**: Requires 4× high-memory GPUs (e.g., H200, B200, A100, H100) with TP=4
- **8-GPU deployment**: Requires 8× GPUs (e.g., H200, B200, A100, H100) with TP=8 and EP=8

## 4. Model Invocation

### 4.1 Basic Usage

For basic API usage and request examples, please refer to:

- [SGLang Basic Usage Guide](https://docs.sglang.ai/basic_usage/send_request.html)

**Testing Deployment:**

After startup, you can test the SGLang OpenAI-compatible API with the following command:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniMaxAI/MiniMax-M2.5",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Who won the world series in 2020?"}]}
        ]
    }'
```

**Simple Completion Example:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M2.5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ],
    max_tokens=1024
)

print(response.choices[0].message.content)
```
**Example Output**:
```
<think>The user asks: "Who won the world series in 2020?" That is a straightforward factual question. The answer: the Los Angeles Dodgers. They won the 2020 World Series, beating the Tampa Bay Rays. The user is presumably expecting that answer.

We must follow the policies. The question is safe: no disallowed content. It's just a factual question. Provide answer.

We must ensure compliance: Use no disallowed content. Should we provide context? Just answer straightforwardly.

The user simply asks "Who won the world series in 2020?" We'll answer: The Los Angeles Dodgers.

No additional relevant info needed, but could elaborate briefly: They beat the Tampa Bay Rays in six games, the series was played in a bubble at Globe Life Field in Arlington, Texas due to COVID-19.

No need for any extra. That's it.
</think>

The Los Angeles Dodgers won the 2020 World Series, defeating the Tampa Bay Rays in six games.
```
### 4.2 Advanced Usage

#### 4.2.1 Reasoning Parser

MiniMax-M2.5 supports Thinking mode. Enable the reasoning parser during deployment to separate the thinking and the content sections:

```shell
python -m sglang.launch_server \
  --model-path MiniMaxAI/MiniMax-M2.5 \
  --tp-size 4 \
  --reasoning-parser minimax-append-think \
  --trust-remote-code \
  --mem-fraction-static 0.85
```

**Streaming with Thinking Process**


With `minimax-append-think`, the thinking content is wrapped in `<think>...</think>` tags within the `content` field. You can parse these tags on the client side to separate the thinking and content sections:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Enable streaming to see the thinking process in real-time
response = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M2.5",
    messages=[
        {"role": "user", "content": "Solve this problem step by step: What is 15% of 240?"}
    ],
    temperature=0.7,
    max_tokens=2048,
    stream=True
)

# Process the stream, separating <think>...</think> from content
in_think = False
think_printed_header = False
content_printed_header = False
buffer = ""

for chunk in response:
    if chunk.choices and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if delta.content:
            buffer += delta.content

            while buffer:
                if in_think:
                    # Look for closing </think> tag
                    end_idx = buffer.find("</think>")
                    if end_idx != -1:
                        print(buffer[:end_idx], end="", flush=True)
                        buffer = buffer[end_idx + len("</think>"):]
                        in_think = False
                    else:
                        # Still in thinking, print what we have
                        print(buffer, end="", flush=True)
                        buffer = ""
                else:
                    # Look for opening <think> tag
                    start_idx = buffer.find("<think>")
                    if start_idx != -1:
                        # Print any content before <think>
                        before = buffer[:start_idx]
                        if before:
                            if not content_printed_header:
                                print("=============== Content =================", flush=True)
                                content_printed_header = True
                            print(before, end="", flush=True)
                        buffer = buffer[start_idx + len("<think>"):]
                        in_think = True
                        if not think_printed_header:
                            print("=============== Thinking =================", flush=True)
                            think_printed_header = True
                    else:
                        # No <think> tag, print as content
                        if not content_printed_header and think_printed_header:
                            print("\n=============== Content =================", flush=True)
                            content_printed_header = True
                        print(buffer, end="", flush=True)
                        buffer = ""

print()
```

**Output Example:**

```text
=============== Thinking =================
The user asks: "Solve this problem step by step: What is 15% of 240?" This is straightforward: 15% = 0.15; 0.15*240 = 36. So answer: 36. Provide step-by-step: convert percent to decimal, multiply.

We need to obey policies. There's no policy violation. Just answer. Provide step by step. Should respond with solution.

We can also mention alternative method: 15% = 15/100 = 3/20. Multiply 240 * 3/20 = (240/20)*3 = 12*3 = 36.

Thus answer 36.

We can add step-by-step. That's it.

=============== Content =================


**Step‑by‑step solution**

1. **Convert the percent to a decimal**
   \[
   15\% = \frac{15}{100}=0.15
   \]

2. **Multiply the decimal by the number**
   \[
   0.15 \times 240 = 36
   \]

(You can also think of it as \(15\% = \frac{3}{20}\) and then \(240 \times \frac{3}{20}=12 \times 3 = 36\).)

\[
\boxed{36}
\]
```

**Note:** The `minimax-append-think` reasoning parser embeds the thinking process in `<think>...</think>` tags within the `content` field. The code above parses these tags in real-time to display thinking and content separately.

#### 4.2.2 Tool Calling

MiniMax-M2.5 supports tool calling capabilities. Enable the tool call parser:

```shell
python -m sglang.launch_server \
  --model-path MiniMaxAI/MiniMax-M2.5 \
  --tp-size 4 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --trust-remote-code \
  --mem-fraction-static 0.85
```

**Python Example:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Non-streaming request
response = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M2.5",
    messages=[
        {"role": "user", "content": "What's the weather in Beijing?"}
    ],
    tools=tools,
    temperature=0.7
)

message = response.choices[0].message

# Check for tool calls
if message.tool_calls:
    for tool_call in message.tool_calls:
        print(f"Tool Call: {tool_call.function.name}")
        print(f"   Arguments: {tool_call.function.arguments}")
else:
    print(message.content)
```

**Output Example**:
```
Tool Call: get_weather
   Arguments: {"location": "Beijing"}
```

**Note:**

- Tool calls are returned in `message.tool_calls` with the function name and arguments
- You can then execute the function and send the result back to continue the conversation

**Handling Tool Call Results:**

```python
# After getting the tool call, execute the function
def get_weather(location, unit="celsius"):
    # Your actual weather API call here
    return f"The weather in {location} is 22°{unit[0].upper()} and sunny."

# Send tool result back to the model
messages = [
    {"role": "user", "content": "What's the weather in Beijing?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Beijing", "unit": "celsius"}'
            }
        }]
    },
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": get_weather("Beijing", "celsius")
    }
]

final_response = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M2.5",
    messages=messages,
    temperature=0.7
)

print(final_response.choices[0].message.content)
# Output: "The weather in Beijing is currently 22°C and sunny."
```

## 5. Benchmark

This section uses **industry-standard configurations** for comparable benchmark results.

### 5.1 Speed Benchmark

**Test Environment**:

- Hardware: NVIDIA B200 GPU (8x)
- Model: Minimax-M2.5
- Tensor Parallelism: 8
- Expert Parallelism: 8
- sglang version: 0.5.8

#### 5.1.1 Standard Scenario Benchmark
- Model Deployment Command:
```
python3 -m sglang.bench_serving \
    --model-path MiniMaxAI/MiniMax-M2.5 \
    --tp 8 \
    --ep 8 \
    --reasoning-parser minimax-append-think \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --tool-call-parser minimax-m2
```
##### 5.1.1.1 Low Concurrency
- Benchmark Command:
```
python3 -m sglang.bench_serving \
  --backend sglang \
  --model MiniMaxAI/MiniMax-M2.5 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-prompts 10 \
  --max-concurrency 1
```
- Test Results:
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 1
Successful requests:                     10
Benchmark duration (s):                  42.99
Total input tokens:                      6091
Total input text tokens:                 6091
Total generated tokens:                  4220
Total generated tokens (retokenized):    3804
Request throughput (req/s):              0.23
Input token throughput (tok/s):          141.70
Output token throughput (tok/s):         98.17
Peak output token throughput (tok/s):    102.00
Peak concurrent requests:                2
Total token throughput (tok/s):          239.87
Concurrency:                             1.00
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   4295.92
Median E2E Latency (ms):                 3419.28
P90 E2E Latency (ms):                    7832.04
P99 E2E Latency (ms):                    9601.40
---------------Time to First Token----------------
Mean TTFT (ms):                          130.57
Median TTFT (ms):                        116.10
P99 TTFT (ms):                           190.90
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.89
Median TPOT (ms):                        9.89
P99 TPOT (ms):                           9.91
---------------Inter-Token Latency----------------
Mean ITL (ms):                           9.89
Median ITL (ms):                         9.89
P95 ITL (ms):                            10.15
P99 ITL (ms):                            10.32
Max ITL (ms):                            14.46
==================================================
```
##### 5.1.1.2 Medium Concurrency
- Benchmark Command:
```
python3 -m sglang.bench_serving \
  --backend sglang \
  --model MiniMaxAI/MiniMax-M2.5 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-prompts 80 \
  --max-concurrency 16
```
- Test Results:
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 16
Successful requests:                     80
Benchmark duration (s):                  48.43
Total input tokens:                      39588
Total input text tokens:                 39588
Total generated tokens:                  40805
Total generated tokens (retokenized):    37142
Request throughput (req/s):              1.65
Input token throughput (tok/s):          817.37
Output token throughput (tok/s):         842.49
Peak output token throughput (tok/s):    1184.00
Peak concurrent requests:                21
Total token throughput (tok/s):          1659.86
Concurrency:                             13.67
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   8274.32
Median E2E Latency (ms):                 8692.90
P90 E2E Latency (ms):                    13690.70
P99 E2E Latency (ms):                    16104.18
---------------Time to First Token----------------
Mean TTFT (ms):                          305.44
Median TTFT (ms):                        106.75
P99 TTFT (ms):                           1053.26
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.20
Median TPOT (ms):                        16.06
P99 TPOT (ms):                           26.75
---------------Inter-Token Latency----------------
Mean ITL (ms):                           15.65
Median ITL (ms):                         13.63
P95 ITL (ms):                            14.90
P99 ITL (ms):                            87.99
Max ITL (ms):                            483.53
==================================================
```
##### 5.1.1.3 High Concurrency
- Benchmark Command:
```
python3 -m sglang.bench_serving \
  --backend sglang \
  --model MiniMaxAI/MiniMax-M2.5 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-prompts 500 \
  --max-concurrency 100
```
- Test Results:
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 100
Successful requests:                     500
Benchmark duration (s):                  92.31
Total input tokens:                      249331
Total input text tokens:                 249331
Total generated tokens:                  252662
Total generated tokens (retokenized):    218975
Request throughput (req/s):              5.42
Input token throughput (tok/s):          2700.94
Output token throughput (tok/s):         2737.02
Peak output token throughput (tok/s):    4479.00
Peak concurrent requests:                109
Total token throughput (tok/s):          5437.97
Concurrency:                             91.19
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   16835.82
Median E2E Latency (ms):                 16042.08
P90 E2E Latency (ms):                    31027.63
P99 E2E Latency (ms):                    34787.91
---------------Time to First Token----------------
Mean TTFT (ms):                          391.06
Median TTFT (ms):                        133.12
P99 TTFT (ms):                           1712.92
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          33.04
Median TPOT (ms):                        34.29
P99 TPOT (ms):                           41.98
---------------Inter-Token Latency----------------
Mean ITL (ms):                           32.61
Median ITL (ms):                         21.67
P95 ITL (ms):                            87.76
P99 ITL (ms):                            118.81
Max ITL (ms):                            1145.62
==================================================
```
#### 5.1.2 Summarization Scenario Benchmark
- Model Deployment Command:
```
python3 -m sglang.bench_serving \
    --model-path MiniMaxAI/MiniMax-M2.5 \
    --tp 8 \
    --ep 8 \
    --reasoning-parser minimax-append-think \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --tool-call-parser minimax-m2
```
##### 5.1.2.1 Low Concurrency
- Benchmark Command:
```
python3 -m sglang.bench_serving \
  --backend sglang \
  --model MiniMaxAI/MiniMax-M2.5 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 10 \
  --max-concurrency 1
```
- Test Results:
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 1
Successful requests:                     10
Benchmark duration (s):                  43.49
Total input tokens:                      41941
Total input text tokens:                 41941
Total generated tokens:                  4220
Total generated tokens (retokenized):    4220
Request throughput (req/s):              0.23
Input token throughput (tok/s):          964.42
Output token throughput (tok/s):         97.04
Peak output token throughput (tok/s):    102.00
Peak concurrent requests:                2
Total token throughput (tok/s):          1061.46
Concurrency:                             1.00
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   4346.83
Median E2E Latency (ms):                 3508.84
P90 E2E Latency (ms):                    7972.23
P99 E2E Latency (ms):                    9659.71
---------------Time to First Token----------------
Mean TTFT (ms):                          131.50
Median TTFT (ms):                        126.76
P99 TTFT (ms):                           182.52
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.00
Median TPOT (ms):                        10.01
P99 TPOT (ms):                           10.12
---------------Inter-Token Latency----------------
Mean ITL (ms):                           10.01
Median ITL (ms):                         10.02
P95 ITL (ms):                            10.29
P99 ITL (ms):                            10.44
Max ITL (ms):                            14.11
==================================================
```
##### 5.1.2.2 Medium Concurrency
- Benchmark Command:
```
python3 -m sglang.bench_serving \
  --backend sglang \
  --model MiniMaxAI/MiniMax-M2.5 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 80 \
  --max-concurrency 16
```
- Test Results:
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 16
Successful requests:                     80
Benchmark duration (s):                  50.12
Total input tokens:                      300020
Total input text tokens:                 300020
Total generated tokens:                  41669
Total generated tokens (retokenized):    41662
Request throughput (req/s):              1.60
Input token throughput (tok/s):          5986.00
Output token throughput (tok/s):         831.38
Peak output token throughput (tok/s):    1152.00
Peak concurrent requests:                20
Total token throughput (tok/s):          6817.38
Concurrency:                             13.93
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   8727.66
Median E2E Latency (ms):                 9170.52
P90 E2E Latency (ms):                    14220.00
P99 E2E Latency (ms):                    16896.54
---------------Time to First Token----------------
Mean TTFT (ms):                          282.56
Median TTFT (ms):                        149.37
P99 TTFT (ms):                           1278.62
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.60
Median TPOT (ms):                        16.61
P99 TPOT (ms):                           25.17
---------------Inter-Token Latency----------------
Mean ITL (ms):                           16.24
Median ITL (ms):                         13.89
P95 ITL (ms):                            15.96
P99 ITL (ms):                            105.79
Max ITL (ms):                            1065.02
==================================================
```
##### 5.1.2.3 High Concurrency
- Benchmark Command:
```
python3 -m sglang.bench_serving \
  --backend sglang \
  --model MiniMaxAI/MiniMax-M2.5 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 320 \
  --max-concurrency 64
```
- Test Results:
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 64
Successful requests:                     320
Benchmark duration (s):                  93.92
Total input tokens:                      1273893
Total input text tokens:                 1273893
Total generated tokens:                  170000
Total generated tokens (retokenized):    169999
Request throughput (req/s):              3.41
Input token throughput (tok/s):          13563.30
Output token throughput (tok/s):         1810.01
Peak output token throughput (tok/s):    2881.00
Peak concurrent requests:                71
Total token throughput (tok/s):          15373.31
Concurrency:                             58.87
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   17277.69
Median E2E Latency (ms):                 16827.33
P90 E2E Latency (ms):                    29045.40
P99 E2E Latency (ms):                    33496.77
---------------Time to First Token----------------
Mean TTFT (ms):                          692.26
Median TTFT (ms):                        188.46
P99 TTFT (ms):                           4932.70
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          32.19
Median TPOT (ms):                        32.69
P99 TPOT (ms):                           50.46
---------------Inter-Token Latency----------------
Mean ITL (ms):                           31.28
Median ITL (ms):                         21.59
P95 ITL (ms):                            101.35
P99 ITL (ms):                            136.74
Max ITL (ms):                            4649.23
==================================================
```

### 5.2 Accuracy Benchmark
#### 5.2.1 GSM8K Benchmark
- Benchmark Command:
```
python benchmark/gsm8k/bench_sglang.py --port 30000
```
- Test Results:
```
Accuracy: 0.950
Invalid: 0.000
Latency: 18.033 s
Output throughput: 1130.161 token/s
```
#### 5.2.2 MMLU Benchmark
- Benchmark Command:
```
cd benchmark/mmlu
bash download_data.sh
python3 bench_sglang.py --port 30000
```
- Test Results:
```
subject: abstract_algebra, #q:100, acc: 0.620
subject: anatomy, #q:135, acc: 0.830
subject: astronomy, #q:152, acc: 0.928
subject: business_ethics, #q:100, acc: 0.810
subject: clinical_knowledge, #q:265, acc: 0.891
subject: college_biology, #q:144, acc: 0.951
subject: college_chemistry, #q:100, acc: 0.670
subject: college_computer_science, #q:100, acc: 0.820
subject: college_mathematics, #q:100, acc: 0.660
subject: college_medicine, #q:173, acc: 0.832
subject: college_physics, #q:102, acc: 0.814
subject: computer_security, #q:100, acc: 0.880
subject: conceptual_physics, #q:235, acc: 0.915
subject: econometrics, #q:114, acc: 0.719
subject: electrical_engineering, #q:145, acc: 0.834
subject: elementary_mathematics, #q:378, acc: 0.902
subject: formal_logic, #q:126, acc: 0.698
subject: global_facts, #q:100, acc: 0.710
subject: high_school_biology, #q:310, acc: 0.926
subject: high_school_chemistry, #q:203, acc: 0.793
subject: high_school_computer_science, #q:100, acc: 0.910
subject: high_school_european_history, #q:165, acc: 0.879
subject: high_school_geography, #q:198, acc: 0.955
subject: high_school_government_and_politics, #q:193, acc: 0.964
subject: high_school_macroeconomics, #q:390, acc: 0.908
subject: high_school_mathematics, #q:270, acc: 0.600
subject: high_school_microeconomics, #q:238, acc: 0.954
subject: high_school_physics, #q:151, acc: 0.781
subject: high_school_psychology, #q:545, acc: 0.956
subject: high_school_statistics, #q:216, acc: 0.847
subject: high_school_us_history, #q:204, acc: 0.922
subject: high_school_world_history, #q:237, acc: 0.916
subject: human_aging, #q:223, acc: 0.839
subject: human_sexuality, #q:131, acc: 0.893
subject: international_law, #q:121, acc: 0.934
subject: jurisprudence, #q:108, acc: 0.861
subject: logical_fallacies, #q:163, acc: 0.890
subject: machine_learning, #q:112, acc: 0.750
subject: management, #q:103, acc: 0.883
subject: marketing, #q:234, acc: 0.944
subject: medical_genetics, #q:100, acc: 0.920
subject: miscellaneous, #q:783, acc: 0.936
subject: moral_disputes, #q:346, acc: 0.829
subject: moral_scenarios, #q:895, acc: 0.632
subject: nutrition, #q:306, acc: 0.863
subject: philosophy, #q:311, acc: 0.833
subject: prehistory, #q:324, acc: 0.907
subject: professional_accounting, #q:282, acc: 0.720
subject: professional_law, #q:1534, acc: 0.640
subject: professional_medicine, #q:272, acc: 0.923
subject: professional_psychology, #q:612, acc: 0.871
subject: public_relations, #q:110, acc: 0.773
subject: security_studies, #q:245, acc: 0.845
subject: sociology, #q:201, acc: 0.930
subject: us_foreign_policy, #q:100, acc: 0.940
subject: virology, #q:166, acc: 0.614
subject: world_religions, #q:171, acc: 0.895
Total latency: 81.468
Average accuracy: 0.825
```
