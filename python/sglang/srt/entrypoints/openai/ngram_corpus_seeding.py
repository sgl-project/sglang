from typing import Any

from sglang.srt.sampling.sampling_params import MAX_REQUEST_NGRAM_CORPUS_SEEDS_TOKENS


# Placeholder values are arbitrary; templates only see their JSON type.
def _make_placeholder(schema: Any) -> Any:
    kind = schema.get("type", "string") if isinstance(schema, dict) else "string"
    if kind in ("integer", "number"):
        return 0
    if kind == "boolean":
        return True
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return "X"


def _build_example_arguments(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function", tool)
    properties = (function.get("parameters") or {}).get("properties", {})
    return {name: _make_placeholder(schema) for name, schema in properties.items()}


# The prompt carries each tool's schema; the call syntax the model emits is the
# chat template's, a different token sequence that is not in the corpus until
# some request has produced it.
def render_tool_call_seeds(
    *,
    tokenizer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    template_kwargs: dict[str, Any],
    encode_kwargs: dict[str, Any],
) -> list[list[int]]:
    """Render one example assistant call per tool and return its tokens after the prompt."""

    def render(rendered_messages: list[dict[str, Any]]) -> str:
        return tokenizer.apply_chat_template(
            rendered_messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
            return_dict=False,
            **template_kwargs,
        )

    try:
        prefix = render(messages)
    except Exception:
        return []

    seeds: list[list[int]] = []
    num_tokens = 0
    for tool in tools:
        function = tool.get("function", tool)
        example_call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "0",  # arbitrary; a template that prints the id seeds it verbatim
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": _build_example_arguments(tool),
                    },
                }
            ],
        }
        try:
            rendered = render(messages + [example_call])
        except Exception:
            continue
        if not rendered.startswith(prefix):
            continue
        seed_tokens = list(tokenizer.encode(rendered[len(prefix) :], **encode_kwargs))
        # The budget is the flat list set_request_ngram_corpus_seeds builds, so
        # it includes one separator per seed already collected.
        if not seed_tokens or (
            num_tokens + len(seeds) + len(seed_tokens)
            > MAX_REQUEST_NGRAM_CORPUS_SEEDS_TOKENS
        ):
            continue
        seeds.append(seed_tokens)
        num_tokens += len(seed_tokens)
    return seeds
