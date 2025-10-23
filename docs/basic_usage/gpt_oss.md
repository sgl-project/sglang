# GPT OSS Usage

Please refer to [https://github.com/sgl-project/sglang/issues/8833](https://github.com/sgl-project/sglang/issues/8833).

## Responses API & Built-in Tools

### Responses API

GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use. You can set reasoning level via `instructions`, e.g., "Reasoning: high" (also supports "medium" and "low") — levels: low (fast), medium (balanced), high (deep).

### Built-in Tools

GPT‑OSS can call built‑in tools for web search and Python execution. You can use the demo tool server or connect to external MCP tool servers.

#### Python Tool

- Executes short Python snippets for calculations, parsing, and quick scripts.
- By default runs in a Docker-based sandbox. To run on the host, set `PYTHON_EXECUTION_BACKEND=UV` (this executes model-generated code locally; use with care).
- Ensure Docker is available if you are not using the UV backend. It is recommended to run `docker pull python:3.11` in advance.

#### Web Search Tool

- Uses the Exa backend for web search.
- Requires an Exa API key; set `EXA_API_KEY` in your environment. Create a key at `https://exa.ai`.

### Tool & Reasoning Parser

- We support OpenAI Reasoning and Tool Call parser, as well as our SGLang native api for tool call and reasoning. Refer to [reasoning parser](../advanced_features/separate_reasoning.ipynb) and [tool call parser](../advanced_features/function_calling.ipynb) for more details.


## Notes

- Use **Python 3.12** for the demo tools. And install the required `gpt-oss` packages.
- The default demo integrates the web search tool (Exa backend) and a demo Python interpreter via Docker.
- For search, set `EXA_API_KEY`. For Python execution, either have Docker available or set `PYTHON_EXECUTION_BACKEND=UV`.

Examples:
```bash
export EXA_API_KEY=YOUR_EXA_KEY
# Optional: run Python tool locally instead of Docker (use with care)
export PYTHON_EXECUTION_BACKEND=UV
```

Launch the server with the demo tool server:

```bash
python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --tool-server demo \
  --tp 2
```

For production usage, sglang can act as an MCP client for multiple services. An [example tool server](https://github.com/openai/gpt-oss/tree/main/gpt-oss-mcp-server) is provided. Start the servers and point sglang to them:
```bash
mcp run -t sse browser_server.py:mcp
mcp run -t sse python_server.py:mcp

python -m sglang.launch_server ... --tool-server ip-1:port-1,ip-2:port-2
```
The URLs should be MCP SSE servers that expose server information and well-documented tools. These tools are added to the system prompt so the model can use them.

### Quick Demo

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-123456"
)

tools = [
    {"type": "code_interpreter"},
    {"type": "web_search_preview"},
]

# Reasoning level example
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant."
    reasoning_effort="high" # Supports high, medium, or low
    input="In one sentence, explain the transformer architecture.",
)
print("====== reasoning: high ======")
print(response.output_text)

# Test python tool
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helfpul assistant, you could use python tool to execute code.",
    input="Use python tool to calculate the sum of 29138749187 and 29138749187", # 58,277,498,374
    tools=tools
)
print("====== test python tool ======")
print(response.output_text)

# Test browser tool
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helfpul assistant, you could use browser to search the web",
    input="Search the web for the latest news about Nvidia stock price",
    tools=tools
)
print("====== test browser tool ======")
print(response.output_text)
```

Example output:
```
====== test python tool ======
The sum of 29,138,749,187 and 29,138,749,187 is **58,277,498,374**.
====== test browser tool ======
**Recent headlines on Nvidia (NVDA) stock**

| Date (2025) | Source | Key news points | Stock‑price detail |
|-------------|--------|----------------|--------------------|
| **May 13** | Reuters | The market data page shows Nvidia trading “higher” at **$116.61** with no change from the previous close. | **$116.61** – latest trade (delayed ≈ 15 min)【14†L34-L38】 |
| **Aug 18** | CNBC | Morgan Stanley kept an **overweight** rating and lifted its price target to **$206** (up from $200), implying a 14 % upside from the Friday close. The firm notes Nvidia shares have already **jumped 34 % this year**. | No exact price quoted, but the article signals strong upside expectations【9†L27-L31】 |
| **Aug 20** | The Motley Fool | Nvidia is set to release its Q2 earnings on Aug 27. The article lists the **current price of $175.36**, down 0.16 % on the day (as of 3:58 p.m. ET). | **$175.36** – current price on Aug 20【10†L12-L15】【10†L53-L57】 |

**What the news tells us**

* Nvidia’s share price has risen sharply this year – up roughly a third according to Morgan Stanley – and analysts are still raising targets (now $206).
* The most recent market quote (Reuters, May 13) was **$116.61**, but the stock has surged since then, reaching **$175.36** by mid‑August.
* Upcoming earnings on **Aug 27** are a focal point; both the Motley Fool and Morgan Stanley expect the results could keep the rally going.

**Bottom line:** Nvidia’s stock is on a strong upward trajectory in 2025, with price targets climbing toward $200‑$210 and the market price already near $175 as of late August.

```
