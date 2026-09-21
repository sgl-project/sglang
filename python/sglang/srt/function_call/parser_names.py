"""Tool-call parser names, kept in a dependency-free module.

`server_args` needs these for the `--tool-call-parser` CLI choices, and
importing `function_call_parser` for them costs seconds (OpenAI protocol
models -> xgrammar -> transformers -> torch.distributed). A unit test checks
this list against `FunctionCallParser.ToolCallParserEnum`; add a name here
when you register a parser.
"""

TOOL_CALL_PARSER_NAMES = [
    "apertus2509",
    "cohere_command4",
    "deepseekv3",
    "deepseekv31",
    "deepseekv32",
    "deepseekv4",
    "deepseekv41",
    "dots",
    "glm",
    "glm45",
    "glm47",
    "gpt-oss",
    "k2_horizon",
    "kimi_k2",
    "kimi_k3",
    "lfm2",
    "ling3",
    "llama3",
    "mimo",
    "minicpm5",
    "mistral",
    "muse",
    "poolside_v1",
    "pythonic",
    "qwen",
    "qwen25",
    "qwen3_coder",
    "spark25",
    "step3",
    "step3p5",
    "minimax-m2",
    "minimax-m3",
    "nanbeige",
    "trinity",
    "interns1",
    "hermes",
    "hunyuan",
    "gigachat3",
    "gigachat35",
    "gemma4",
    "inkling",
]
