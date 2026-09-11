"""
Generate engine-vs-router chat render parity fixtures.

For each cached model and request shape, reproduce the engine's
`OpenAIServingChat._apply_jinja_template` (or the DeepSeek-V4 encoder path)
with SGLang's own helpers and record the prompt token ids. The Rust test
`tests/component/tokenizer/render_parity.rs` renders the same requests through
the router's `ChatEncoder` and asserts bit-identical ids.

Run from an environment with sglang importable and the models in the HF cache:
    python tests/scripts/generate_chat_render_parity.py
"""

import copy
import json
import pathlib
import sys

from sglang.srt.entrypoints.openai import encoding_dsv4
from sglang.srt.entrypoints.openai.chat_encoding import (
    resolve_dsv4_reasoning_effort_profile,
)
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import (
    ThinkingMode,
    normalize_assistant_tool_call_arguments,
    normalize_tool_content,
)
from sglang.srt.parser.jinja_template_utils import (
    detect_jinja_template_content_format,
    process_content_for_template_format,
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from transformers.utils.hub import cached_file

ROOT = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "chat_render_parity"

MODELS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "glm-5.2": "zai-org/GLM-5.2-FP8",
    "minimax-m3": "MiniMaxAI/MiniMax-M3",
    "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
}

LONG = "Résumé of the plan: " + "第一步，收集数据。Then we iterate. " * 8

SHAPES = {
    "user_only": {"messages": [{"role": "user", "content": "Say hi in one sentence."}]},
    "system_user": {
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is 2+2?"},
        ]
    },
    "multi_turn": {
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": LONG},
        ]
    },
    "thinking_off": {
        "messages": [{"role": "user", "content": "Think about it."}],
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
    },
    "thinking_on": {
        "messages": [{"role": "user", "content": "Think about it."}],
        "chat_template_kwargs": {"enable_thinking": True, "thinking": True},
    },
}


def snapshot_dir(model_id):
    return pathlib.Path(cached_file(model_id, "config.json", local_files_only=True)).parent


def engine_messages(request, content_format):
    messages = [m.model_dump() for m in request.messages]
    for message in messages:
        normalize_assistant_tool_call_arguments(message)
    out = []
    for msg in copy.deepcopy(messages):
        if msg.get("content") is None:
            msg["content"] = ""
        processed = process_content_for_template_format(msg, content_format, [], [], [], [])
        processed["content"] = normalize_tool_content(processed["role"], processed.get("content"))
        out.append(processed)
    return out


def engine_prompt_ids(model_id, tok, request):
    """Mirror `_apply_jinja_template` for a text-only request without tools."""
    snapshot = snapshot_dir(model_id)
    model_type = json.load(open(snapshot / "config.json")).get("model_type")
    if model_type == "deepseek_v4":
        messages = engine_messages(request, "string")
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        thinking = (request.chat_template_kwargs or {}).get("thinking", False)
        text = encoding_dsv4.encode_messages(
            messages,
            thinking_mode=ThinkingMode.THINKING if thinking else ThinkingMode.CHAT,
            reasoning_effort=None,
            reasoning_effort_profile=resolve_dsv4_reasoning_effort_profile(
                model_path=str(snapshot)
            ),
        )
        return tok.encode(text)

    template = tok.chat_template
    if not isinstance(template, str):
        raise RuntimeError(f"{model_id}: named template dict is not supported here")
    messages = engine_messages(request, detect_jinja_template_content_format(template))
    extra = {}
    if request.reasoning_effort is not None:
        extra["reasoning_effort"] = request.reasoning_effort
    if request.chat_template_kwargs:
        extra.update(request.chat_template_kwargs)
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=None, return_dict=False, **extra
    )
    encode_kwargs = {"add_special_tokens": False} if len(tok.encode("")) > 0 else {}
    return tok.encode(rendered, **encode_kwargs)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for slug, model_id in MODELS.items():
        try:
            snapshot = snapshot_dir(model_id)
        except Exception as e:
            print(f"skip {model_id}: {e}", file=sys.stderr)
            continue
        tok = get_tokenizer(str(snapshot))
        cases = []
        for shape, body in SHAPES.items():
            request = ChatCompletionRequest(model=model_id, **copy.deepcopy(body))
            ids = engine_prompt_ids(model_id, tok, request)
            cases.append({"shape": shape, "request": body, "expected_token_ids": ids})
        out = ROOT / f"{slug}.json"
        lines = [json.dumps(case, ensure_ascii=False) for case in cases]
        out.write_text(
            '{"model_id": %s, "cases": [\n%s\n]}\n' % (json.dumps(model_id), ",\n".join(lines))
        )
        print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
