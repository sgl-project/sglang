"""CPU parity: real Python/Rust chat formatting and tokenization, without a server.

Run: python test/registered/unit/entrypoints/test_deepseek_v4_rust.py
Requires cargo; downloads only the official checkpoint's tokenizer/config files.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from huggingface_hub import snapshot_download

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="base-a-test-cpu")


def chat_cases():
    user = {"role": "user", "content": "What is the capital of France?"}
    yield "default", {"messages": [user]}
    yield "system", {"messages": [{"role": "system", "content": "Be concise."}, user]}
    yield "text_parts", {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the capital"},
                    {"type": "text", "text": "of France?"},
                ],
            }
        ]
    }
    for thinking in (False, True):
        for effort in (None, "low", "high", "max", "medium"):
            yield f"thinking={thinking},effort={effort}", {
                "messages": [user],
                "chat_template_kwargs": {"thinking": thinking},
                "reasoning_effort": effort,
            }
    yield "reasoning_history", {
        "messages": [
            {"role": "user", "content": "What is 1+1?"},
            {
                "role": "assistant",
                "content": "2",
                "reasoning_content": "Add one and one.",
            },
            user,
        ],
        "chat_template_kwargs": {"thinking": True},
    }
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    yield "tools", {"messages": [user], "tools": [tool], "tool_choice": "none"}
    yield "tool_history", {
        "tools": [tool],
        "tool_choice": "none",
        "chat_template_kwargs": {"thinking": True},
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Look up the weather.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny."},
            {"role": "user", "content": "Summarize the result."},
        ],
    }


class TestDeepSeekV4RustParity(unittest.TestCase):
    @patch.dict(
        os.environ, {"SGLANG_DEFAULT_THINKING": "0", "SGLANG_DSV4_REASONING_EFFORT": ""}
    )
    def test_chat_parity(self):
        model = "deepseek-ai/DeepSeek-V4-Flash-0731"
        path = snapshot_download(
            model,
            revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            allow_patterns=[
                "config.json",
                "tokenizer*.json",
                "special_tokens_map.json",
            ],
        )
        tokenizer = get_tokenizer(path)
        # Exercise the serving formatter without constructing a model or scheduler.
        encoder = Mock(wraps=tokenizer.encode)
        serving = object.__new__(OpenAIServingChat)
        serving.chat_encoding_spec = "dsv4"
        serving._dsv4_reasoning_effort_profile = "official"
        serving.template_manager = SimpleNamespace(
            jinja_template_content_format="string"
        )
        serving.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=encoder)
        )
        cases = []
        for name, body in chat_cases():
            body = {"model": model, **body}
            result = serving._apply_jinja_template(
                ChatCompletionRequest(**body), tools=None, is_multimodal=False
            )
            cases.append(
                dict(
                    name=name,
                    request=body,
                    prompt=encoder.call_args.args[0],
                    input_ids=result.prompt_ids,
                )
            )
        config = json.loads((Path(path) / "config.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "parity.json"
            fixture.write_text(
                json.dumps(
                    dict(
                        tokenizer_path=path,
                        model_type=config["model_type"],
                        cases=cases,
                    )
                )
            )
            subprocess.run(
                [
                    "cargo",
                    "test",
                    "-p",
                    "sglang-server",
                    "--lib",
                    "--locked",
                    "python_rust_chat_tokenization_parity",
                    "--",
                    "--ignored",
                    "--nocapture",
                ],
                cwd=Path(__file__).resolve().parents[4] / "rust",
                env={**os.environ, "SGLANG_TEST_CHAT_PARITY_FIXTURE": str(fixture)},
                check=True,
            )
        print(f"Python/Rust prompts and token IDs match for {len(cases)} cases")


if __name__ == "__main__":
    unittest.main()
