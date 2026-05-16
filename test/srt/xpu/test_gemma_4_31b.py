"""
Gemma 4 31B: simple text Q&A on XPU (OpenAI /v1), same shape as
``test_gemma_4_e2b.py``.

Model card: https://huggingface.co/google/gemma-4-31B-it

  - XPU test runs when Intel XPU is available.
  - Dense 31B model; requires tp=4 on 4x Arc Pro B60 (24 GB each).

Run from test/srt::

  python3 -m unittest xpu.test_gemma_4_31b.TestGemma431BXPU.test_simple_code_qa

Appends to ``gemma_4_31b_comparison.txt`` in this directory.

Tensor dumps (all layers) go under ``debug_tensor_dump_output/gemma_4_31b/``.

Server is started with ``sglang serve`` (``--model-impl sglang``).
"""

from __future__ import annotations

import os
import re
import unittest
from datetime import datetime, timezone
from pathlib import Path

import openai

from sglang.srt.utils.common import is_xpu
from sglang.test.test_utils import CustomTestCase
from sglang.test.vlm_utils import (
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_server,
)

MODEL = "google/gemma-4-31B-it"

COMPARISON_LOG_PATH = Path(__file__).resolve().parent / "gemma_4_31b_comparison.txt"
DEBUG_TENSOR_DUMP_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "debug_tensor_dump_output" / "gemma_4_31b"
)
LAUNCH_TIMEOUT = 1200


def _server_subprocess_env() -> dict:
    return {
        "TORCHDYNAMO_VERBOSE": "0",
        "TORCHINDUCTOR_VERBOSE": "0",
        "TORCH_COMPILE_DEBUG": "0",
        "TORCH_SHOW_CPP_STACKTRACES": "0",
    }


def _prettify_spm_style_text(s: str) -> str:
    """Turn SentencePiece-style space/newline markers in API strings into normal text."""
    if not s:
        return s
    return s.replace("Ċ", "\n").replace("Ġ", " ")


def setUpModule():
    DEBUG_TENSOR_DUMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for _pt in DEBUG_TENSOR_DUMP_OUTPUT_DIR.rglob("*.pt"):
        _pt.unlink(missing_ok=True)
    COMPARISON_LOG_PATH.write_text(
        "Gemma-4-31B — device comparison log\n"
        f"Model: {MODEL}\n"
        f"Run started (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"{'=' * 80}\n\n",
        encoding="utf-8",
    )


def _append_comparison_log(
    *,
    title: str,
    device_cli: str,
    extra_server_notes: str,
    user_prompt: str,
    response,
) -> None:
    msg = response.choices[0].message
    content = _prettify_spm_style_text(msg.content or "")
    reasoning = _prettify_spm_style_text(getattr(msg, "reasoning_content", None) or "")
    usage = response.usage
    block = (
        f"\n{'#' * 80}\n"
        f"{title}\n"
        f"Server device flag: {device_cli}\n"
        f"{extra_server_notes}\n"
        f"{'#' * 80}\n"
        f"--- user prompt ---\n{user_prompt}\n"
        f"--- assistant message.content ---\n{content}\n"
        f"--- assistant message.reasoning_content (if any) ---\n{reasoning}\n"
        f"--- usage ---\n"
        f"  prompt_tokens: {getattr(usage, 'prompt_tokens', None)}\n"
        f"  completion_tokens: {getattr(usage, 'completion_tokens', None)}\n"
        f"  total_tokens: {getattr(usage, 'total_tokens', None)}\n"
        f"{'=' * 80}\n"
    )
    with COMPARISON_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(block)


# Gemma 4 31B does not ship a chat_template in its tokenizer.
# Reuse the Gemma-family jinja that sits next to this test file.
_CHAT_TEMPLATE_PATH = str(
    Path(__file__).resolve().parent / "gemma4_chat_template.jinja"
)

# 31B dense model: tp=4 on 4x Arc Pro B60. Memory-tightening flags
# copied from the authorized launch shape in
# model_enablement/gemma/run_performance_benchmark.py (DEFAULT_TP["31b"]=4,
# mem_fraction_static=0.92, context_length=8192, chunked_prefill_size=1024,
# max_running_requests=8).
XPU_SERVER_ARGS = [
    "--device",
    "xpu",
    "--tp=4",
    "--trust-remote-code",
    "--disable-overlap-schedule",
    "--page-size",
    "64",
    "--attention-backend",
    "intel_xpu",
    "--model-impl",
    "sglang",
    "--chat-template",
    _CHAT_TEMPLATE_PATH,
    "--mem-fraction-static",
    "0.92",
    "--context-length",
    "8192",
    "--chunked-prefill-size",
    "1024",
    "--max-running-requests",
    "8",
    "--cuda-graph-max-bs",
    "8",
]

_SIMPLE_CODE_PROMPT = (
    "Write a minimal Python function `def add(a, b):` that returns a+b. "
    "Reply with only the function, give a brief explanation. "
    "Finish with asking me How can I help you today?"
)


def _simple_text_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _SIMPLE_CODE_PROMPT},
            ],
        }
    ]


def _compact_code_text(s: str) -> str:
    t = s.replace("Ġ", " ").replace("Ċ", "\n")
    return re.sub(r"\s+", "", t.lower())


def _assert_code_reply(response):
    assert response.choices[0].message.role == "assistant"
    msg = response.choices[0].message
    text = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    combined = f"{text} {reasoning}".strip()
    assert len(combined) > 0
    lower = combined.lower()
    assert (
        "def" in lower and "add" in lower
    ), f"expected a Python `def add` in reply, got: {combined!r}"
    assert "return" in lower, f"expected `return` in reply, got: {combined!r}"
    compact = _compact_code_text(combined)
    assert (
        "a+b" in compact
    ), f"expected `a+b` (allowing spaces) in reply, got: {combined!r}"
    assert response.usage is not None
    assert response.usage.completion_tokens > 0


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestGemma431BXPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        os.environ["SGLANG_USE_SGL_XPU"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=LAUNCH_TIMEOUT,
            api_key=cls.api_key,
            other_args=list(XPU_SERVER_ARGS),
            device="xpu",
            env=_server_subprocess_env(),
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_simple_code_qa(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=_simple_text_messages(),
            temperature=0,
            max_tokens=96,
        )
        _assert_code_reply(response)
        _append_comparison_log(
            title="OUTPUT FROM --device XPU (Gemma-4-31B)",
            device_cli="--device xpu",
            extra_server_notes="SGLANG_USE_SGL_XPU=1; tp=4; see XPU_SERVER_ARGS in test source.",
            user_prompt=_SIMPLE_CODE_PROMPT,
            response=response,
        )

    def test_sliding_window_long_context(self):
        """Generate >511 tokens to exercise decode past the SWA window boundary.

        Gemma 4 uses sliding_window=512 (511 in SGLang exclusive).
        With ~30 prompt tokens + 600 generated tokens, the total sequence
        (~630 tokens) exceeds the window, forcing the SWA decode kernel
        to actually mask out-of-window tokens. If page table translation
        or kernel masking is broken, this test will produce garbage or crash.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Write a detailed, step-by-step tutorial on how to "
                                "build a simple web server in Python using the socket "
                                "module. Include complete code examples with comments."
                            ),
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=600,
        )
        msg = response.choices[0].message
        text = msg.content or ""
        assert len(text) > 0, "expected non-empty response"
        assert response.usage is not None
        assert response.usage.completion_tokens >= 500, (
            f"expected >= 500 completion tokens to exceed SWA window, "
            f"got {response.usage.completion_tokens}"
        )
        _append_comparison_log(
            title="OUTPUT FROM --device XPU (Gemma-4-31B) [SWA long context]",
            device_cli="--device xpu",
            extra_server_notes="SWA window=511; generated >500 tokens to exceed window.",
            user_prompt="(long context SWA test)",
            response=response,
        )

    def test_sliding_window_3k_tokens(self):
        """Generate ~3000 tokens — approximately 6x the SWA window.

        At 3000 tokens the sliding window (511) has rolled many times,
        stressing the decode kernel's local masking and KV cache page
        table management over an extended generation.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Write a comprehensive guide to Python data structures. "
                                "Cover lists, tuples, dictionaries, sets, double-ended queues, "
                                "namedtuples, dataclasses, and custom linked lists. "
                                "For each one, provide multiple code examples, explain "
                                "time complexity of common operations, compare trade-offs, "
                                "and show real-world use cases. Be extremely thorough."
                            ),
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=3000,
        )
        msg = response.choices[0].message
        text = msg.content or ""
        assert len(text) > 0, "expected non-empty response"
        assert response.usage is not None
        assert response.usage.completion_tokens >= 2500, (
            f"expected >= 2500 completion tokens (6x SWA window), "
            f"got {response.usage.completion_tokens}"
        )
        _append_comparison_log(
            title="OUTPUT FROM --device XPU (Gemma-4-31B) [SWA 3k tokens]",
            device_cli="--device xpu",
            extra_server_notes="SWA window=511; generated ~3000 tokens (6x window).",
            user_prompt="(3k token SWA stress test)",
            response=response,
        )


if __name__ == "__main__":
    unittest.main()
