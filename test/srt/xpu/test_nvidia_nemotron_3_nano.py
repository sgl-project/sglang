"""
NVIDIA Nemotron 3 Nano 30B-A3B (BF16, hybrid MoE + Mamba): simple text Q&A on
XPU (OpenAI /v1), same shape as ``test_gemma_4_31b.py``.

Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

  - XPU test runs when Intel XPU is available.
  - Requires tp=4 (30B params with ~3B active; ~60 GB weights sharded 4-way
    → ~15 GB per rank). On 24 GB Arc Pro B60 that leaves headroom for the
    Mamba state cache + KV pool.

Run from test/srt::

  python3 -m unittest xpu.test_nvidia_nemotron_3_nano.TestNemotron3Nano30BXPU.test_simple_code_qa

Appends to ``nemotron_3_nano_30b_comparison.txt`` in this directory.

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

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

COMPARISON_LOG_PATH = (
    Path(__file__).resolve().parent / "nemotron_3_nano_30b_comparison.txt"
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
    if not s:
        return s
    return s.replace("Ċ", "\n").replace("Ġ", " ")


def setUpModule():
    COMPARISON_LOG_PATH.write_text(
        "NVIDIA-Nemotron-3-Nano-30B-A3B — device comparison log\n"
        f"Model: {MODEL}\n"
        f"Run started (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"{'=' * 80}\n\n",
        encoding="utf-8",
    )


def _append_comparison_log(
    *, title, device_cli, extra_server_notes, user_prompt, response
):
    msg = response.choices[0].message
    content = _prettify_spm_style_text(msg.content or "")
    reasoning = _prettify_spm_style_text(getattr(msg, "reasoning_content", None) or "")
    usage = response.usage
    block = (
        f"\n{'#' * 80}\n{title}\nServer device flag: {device_cli}\n"
        f"{extra_server_notes}\n{'#' * 80}\n"
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


# 30B-A3B hybrid (Mamba + attention + MoE) model: requires tp=4 across XPU
# devices. The BF16 variant's weights are ~60 GB, sharded 4-way → ~15 GB per
# rank of weights. On a 24 GB Arc Pro B60 partition this leaves ~9 GB headroom
# per rank for the Mamba state cache, KV pool, and cuda-graph buffers. The
# combination below mirrors ``test_gemma_4_31b.py`` and has been shaped to fit
# at tp=4 on 4x Arc Pro B60.
#
# ``qwen3_coder`` tool-call parser and ``deepseek-r1`` reasoning parser match
# the registered 2-GPU BF16 configuration for this model
# (see ``test/registered/models/test_nvidia_nemotron_3_nano.py``).
XPU_SERVER_ARGS = [
    "--device",
    "xpu",
    "--tp=4",
    "--trust-remote-code",
    "--disable-overlap-schedule",
    "--disable-radix-cache",
    "--page-size",
    "64",
    "--attention-backend",
    "intel_xpu",
    "--model-impl",
    "sglang",
    "--tool-call-parser",
    "qwen3_coder",
    "--reasoning-parser",
    "deepseek-r1",
    "--mem-fraction-static",
    "0.85",
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
        {"role": "user", "content": [{"type": "text", "text": _SIMPLE_CODE_PROMPT}]}
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
    ), f"expected `def add` in reply, got: {combined!r}"
    assert "return" in lower, f"expected `return` in reply, got: {combined!r}"
    compact = _compact_code_text(combined)
    assert "a+b" in compact, f"expected `a+b` in reply, got: {combined!r}"
    assert response.usage is not None
    assert response.usage.completion_tokens > 0


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestNemotron3Nano30BXPU(CustomTestCase):
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
            max_tokens=256,
        )
        # Log the raw response BEFORE assertions so the .txt always captures
        # what the model produced, even when the content assertions fail.
        _append_comparison_log(
            title="OUTPUT FROM --device XPU (NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)",
            device_cli="--device xpu",
            extra_server_notes=(
                "SGLANG_USE_SGL_XPU=1; tp=4; intel_xpu attention backend; "
                "hybrid Mamba+attention MoE; tool-call-parser=qwen3_coder; "
                "reasoning-parser=deepseek-r1."
            ),
            user_prompt=_SIMPLE_CODE_PROMPT,
            response=response,
        )
        _assert_code_reply(response)


if __name__ == "__main__":
    unittest.main()
