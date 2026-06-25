# GPT OSS Usage

Please refer to [https://github.com/sgl-project/sglang/issues/8833](https://github.com/sgl-project/sglang/issues/8833).

## Responses API & Built-in Tools

### Responses API

GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use. You can set reasoning level via `instructions`, e.g., "Reasoning: high" (also supports "medium" and "low") — levels: low (fast), medium (balanced), high (deep).

### Built-in Tools

GPT‑OSS can call built-in tools for web search and Python execution. SGLang provides native web search when an Exa API key is configured on the server. The demo tool server and external MCP tool servers remain available for Python execution and custom tools.

#### Python Tool

- Executes short Python snippets for calculations, parsing, and quick scripts.
- By default runs in a Docker-based sandbox. To run on the host, set `PYTHON_EXECUTION_BACKEND=UV` (this executes model-generated code locally; use with care).
- Ensure Docker is available if you are not using the UV backend. It is recommended to run `docker pull python:3.11` in advance.

#### Web Search Tool

- Uses Exa as the default backend for native web search.
- Requires an Exa API key; set `EXA_API_KEY` in the SGLang server environment. Create a key at `https://dashboard.exa.ai/api-keys`.
- Uses server-side defaults: `numResults=10`, search `type="auto"`, and `contents.highlights=true`.
- Tags native Exa requests with `x-exa-integration: sglang` for integration attribution.

### Tool & Reasoning Parser

- We support OpenAI Reasoning and Tool Call parser, as well as our SGLang native api for tool call and reasoning. Refer to [reasoning parser](../advanced_features/separate_reasoning.ipynb) and [tool parser](../advanced_features/tool_parser.ipynb) for more details.


## Notes

- Native web search does not require `--tool-server demo`; set `EXA_API_KEY` before launching SGLang.
- Use **Python 3.12** for the demo Python tool. For Python execution, either have Docker available or set `PYTHON_EXECUTION_BACKEND=UV`.
- MCP remains available for advanced/custom tools. Native SGLang web search is tagged automatically; custom MCP servers are responsible for their own upstream request headers.

Examples:
```bash
export EXA_API_KEY=YOUR_EXA_KEY
export SGLANG_EXA_NUM_RESULTS=10
export SGLANG_EXA_SEARCH_TYPE=auto
export SGLANG_EXA_INCLUDE_HIGHLIGHTS=true

# Optional: run Python tool locally instead of Docker (use with care)
export PYTHON_EXECUTION_BACKEND=UV
```

Launch the server with native web search:

```bash
python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --tp 2
```

Add `--tool-server demo` only when you also want the demo Python tool server.

For production usage, sglang can act as an MCP client for multiple services. An [example tool server](https://github.com/openai/gpt-oss/tree/main/gpt-oss-mcp-server) is provided. Start the servers and point sglang to them:
```bash
mcp run -t sse browser_server.py:mcp
mcp run -t sse python_server.py:mcp

python -m sglang.launch_server ... --tool-server ip-1:port-1,ip-2:port-2
```
The URLs should be MCP SSE servers that expose server information and well-documented tools. These tools are added to the system prompt so the model can use them.

## Speculative Decoding

SGLang supports speculative decoding for GPT-OSS models using EAGLE3 algorithm. This can significantly improve decoding speed, especially for small batch sizes.

**Usage**:
Add `--speculative-algorithm EAGLE3` along with the draft model path.
```bash
python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 \
  --tp 2
```

```{tip}
To enable the experimental overlap scheduler for EAGLE3 speculative decoding, set the environment variable `SGLANG_ENABLE_SPEC_V2=1`. This can improve performance by enabling overlap scheduling between draft and verification stages.
```

### Quick Demo

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-123456"
)

search_tools = [{"type": "web_search"}]
python_tools = [{"type": "code_interpreter"}]

# Reasoning level example
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant.",
    reasoning_effort="high", # Supports high, medium, or low
    input="In one sentence, explain the transformer architecture.",
)
print("====== reasoning: high ======")
print(response.output_text)

# Test python tool. Requires launching SGLang with --tool-server demo.
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant, you could use python tool to execute code.",
    input="Use python tool to calculate the sum of 29138749187 and 29138749187", # 58,277,498,374
    tools=python_tools
)
print("====== test python tool ======")
print(response.output_text)

# Test web search tool
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant, you can search the web when needed.",
    input="Search the web for the latest news about Nvidia stock price",
    tools=search_tools
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
