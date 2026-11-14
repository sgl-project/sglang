"""Replay a local LLM-jp checkpoint's tokenization and saved model generations.

Example:
    PYTHONPATH=python python test/manual/validate_llm_jp_harmony.py \
        /path/to/llm-jp-4.1-32b-a3b-thinking --generated-json output.json

Generation JSON contains ``token_ids`` and ``tools`` (OpenAI function schemas).
No model weights or network access are needed for replay.
"""

import argparse
import json
from pathlib import Path

import xgrammar as xgr
from transformers import AutoTokenizer

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.llm_jp_harmony_detector import LlmJpHarmonyDetector
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.parser.template_detection import (
    detect_reasoning_parser,
    detect_reasoning_pattern,
    detect_tool_call_parser,
)


def _parse(text, tools, chunk_size=None):
    reasoning = ReasoningParser("llm-jp-harmony")
    parser = FunctionCallParser(tools, "llm-jp-harmony")
    if chunk_size is None:
        analysis, forwarded = reasoning.parse_non_stream(text)
        normal, calls = parser.parse_non_stream(forwarded)
    else:
        analysis, normal, calls = "", "", []
        for offset in range(0, len(text), chunk_size):
            thinking, forwarded = reasoning.parse_stream_chunk(
                text[offset : offset + chunk_size]
            )
            analysis += thinking or ""
            content, items = parser.parse_stream_chunk(forwarded or "")
            normal += content
            calls += items
        thinking, forwarded = reasoning.parse_stream_end()
        analysis += thinking or ""
        content, items = parser.parse_stream_chunk(forwarded or "")
        normal += content
        calls += items
        content, items = parser.parse_stream_end()
        normal += content
        calls += items
    # Non-stream tool_index refers to the tool definition; streaming indexes
    # refer to call ordinals. Compare the observable names and JSON arguments.
    return {
        "reasoning": analysis,
        "content": normal,
        "calls": [
            {"name": item.name, "arguments": json.loads(item.parameters)}
            for item in calls
        ],
    }


def main():
    args_parser = argparse.ArgumentParser(description=__doc__)
    args_parser.add_argument("model_path")
    args_parser.add_argument("--generated-json", action="append", default=[])
    args = args_parser.parse_args()
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    calls = [
        {"name": "get_weather", "arguments": {"city": city}}
        for city in ("Tokyo", "Osaka")
    ]
    replay = (
        "<|channel|>analysis<|message|>Two cities.<|end|>"
        '<|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|> json<|message|>{"city":"Tokyo"}<|end|>'
        '<|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|> json<|message|>{"city":"Osaka"}<|call|>'
    )
    report = []
    for trust_remote_code in (False, True):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        force, config = detect_reasoning_pattern(tokenizer.chat_template)
        for detect in (detect_reasoning_parser, detect_tool_call_parser):
            assert (
                detect(tokenizer.chat_template, tokenizer, config, force)
                == "llm-jp-harmony"
            )
        assert tokenizer.convert_tokens_to_ids("<|call|>") == 13
        samples = [
            {
                "name": "parallel-tokenizer-replay",
                "token_ids": tokenizer.encode(replay, add_special_tokens=False),
                "tools": [tool],
            }
        ]
        samples += [json.loads(Path(path).read_text()) for path in args.generated_json]
        for sample in samples:
            token_ids = sample["token_ids"]
            # Mirror the llm-jp-harmony detokenizer's EOS trimming.
            if token_ids and token_ids[-1] in (2, 13):
                token_ids = token_ids[:-1]
            text = tokenizer.decode(token_ids, skip_special_tokens=False)
            tools = [Tool.model_validate(t) for t in sample.get("tools") or []]
            result = _parse(text, tools)
            for chunk_size in (1, 2, 7, 31):
                assert _parse(text, tools, chunk_size) == result
            if sample["name"].startswith("parallel"):
                assert result["calls"] == calls, result
                assert result["content"] == "", result
                # String-level grammar tests miss tokenizer-specific spaces.
                # Match the actual token IDs, including the closing EOS token.
                info = xgr.TokenizerInfo.from_huggingface(
                    tokenizer, vocab_size=196608, stop_token_ids=[2, 13]
                )
                tag = LlmJpHarmonyDetector().get_structural_tag(
                    tools=tools, tool_choice="required"
                )
                compiled = xgr.GrammarCompiler(info).compile_structural_tag(tag)
                matcher = xgr.GrammarMatcher(compiled)
                generated = sample["token_ids"]
                tool_ids = generated[generated.index(11) + 1 :]
                for token_id in tool_ids:
                    assert matcher.accept_token(token_id), token_id
                assert matcher.is_terminated()
            elif sample["name"] == "final":
                assert result["content"] == "2", result
                assert result["reasoning"], result
            # Reproduce the failure of the GPT-OSS reuse approach on tool output.
            if sample["name"] == "parallel-tokenizer-replay":
                _, old_forwarded = ReasoningParser("gpt-oss").parse_non_stream(text)
                _, old_calls = FunctionCallParser(tools, "gpt-oss").parse_non_stream(
                    old_forwarded
                )
                assert old_calls == []
            report.append(
                {
                    "tokenizer": type(tokenizer).__name__,
                    "sample": sample["name"],
                    **result,
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
