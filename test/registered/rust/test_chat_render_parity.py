"""Python vs Rust chat rendering and tokenization parity for a template-less model.

Python renders through its real serving path; the Rust server renders the same
requests with Dynamo's built-in formatter. No server or GPU is involved.
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

register_cpu_ci(est_time=180, suite="base-a-test-cpu")

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
USER = {"role": "user", "content": "What is the capital of France?"}
# Omitted where stock Dynamo differs from Python: an omitted `thinking` (Dynamo
# defaults on), an omitted or unsupported effort while thinking (Dynamo defaults
# high), and multi-part text content (Dynamo joins parts without spaces).
CASES = {
    "system": {
        "messages": [{"role": "system", "content": "Be concise."}, USER],
        "chat_template_kwargs": {"thinking": False},
    },
    "reasoning_history": {
        "messages": [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2", "reasoning_content": "Add."},
            USER,
        ],
        "chat_template_kwargs": {"thinking": True},
        "reasoning_effort": "low",
    },
    **{
        f"thinking={thinking},effort={effort}": {
            "messages": [USER],
            "chat_template_kwargs": {"thinking": thinking},
            "reasoning_effort": effort,
        }
        for thinking, efforts in (
            (False, (None, "low", "high", "max", "medium")),
            (True, ("low", "high", "max")),
        )
        for effort in efforts
    },
}


class TestChatRenderParity(unittest.TestCase):
    @patch.dict(
        os.environ, {"SGLANG_DEFAULT_THINKING": "0", "SGLANG_DSV4_REASONING_EFFORT": ""}
    )
    def test_prompts_and_token_ids_match(self):
        path = snapshot_download(
            MODEL,
            revision=REVISION,
            allow_patterns=[
                "config.json",
                "tokenizer*.json",
                "special_tokens_map.json",
            ],
        )
        encoder = Mock(wraps=get_tokenizer(path).encode)
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
        for name, body in CASES.items():
            body = {"model": MODEL, **body}
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

        model_type = json.loads((Path(path) / "config.json").read_text())["model_type"]
        with tempfile.NamedTemporaryFile("w", suffix=".json") as fixture:
            json.dump(
                dict(tokenizer_path=path, model_type=model_type, cases=cases), fixture
            )
            fixture.flush()
            subprocess.run(
                [
                    "cargo",
                    "test",
                    "-p",
                    "sglang-server",
                    "--lib",
                    "--locked",
                    "python_rust_render_parity",
                    "--",
                    "--ignored",
                ],
                cwd=Path(__file__).resolve().parents[3] / "rust",
                env={**os.environ, "SGLANG_CHAT_PARITY_FIXTURE": fixture.name},
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
