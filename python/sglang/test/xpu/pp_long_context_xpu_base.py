"""Shared server fixture for the Intel XPU pipeline-parallel long-context tests.

Launches Llama-3.1-8B-Instruct at PP2 on the ``intel_xpu`` attention backend,
captures the server log, and builds passkey prompts of an exact token length.
"""

from __future__ import annotations

import io
import re

import requests
from transformers import AutoTokenizer

from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

MODEL = DEFAULT_MODEL_NAME_FOR_TEST
FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)
BASE_ARGS = [
    "--attention-backend",
    "intel_xpu",
    "--pp-size",
    "2",
    "--disable-radix-cache",
]


def passkey_prompt_ids(tokenizer, num_tokens: int, passkey: int) -> list[int]:
    """Chat prompt of exactly ``num_tokens`` tokens with the passkey at mid depth."""
    prefix, suffix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "\x00"}],
        add_generation_prompt=True,
        tokenize=False,
    ).split("\x00")

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    head = encode(prefix + "Find the pass key hidden in the text below.\n\n")
    needle = encode(
        f" The pass key is {passkey}. Remember it. {passkey} is the pass key. "
    )
    tail = encode("\n\nWhat is the pass key? Answer with the number only." + suffix)
    filler = encode(FILLER)
    hay_len = num_tokens - len(head) - len(needle) - len(tail)
    hay = (filler * (hay_len // len(filler) + 1))[:hay_len]
    return head + hay[: hay_len // 2] + needle + hay[hay_len // 2 :] + tail


class PPLongContextXPUServerBase(CustomTestCase):
    mem_fraction_static = "0.80"
    other_args: list[str] = []
    env: dict = {}

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        cls.stdout, cls.stderr = io.StringIO(), io.StringIO()
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=BASE_ARGS
            + ["--mem-fraction-static", cls.mem_fraction_static]
            + cls.other_args,
            env=cls.env or None,
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def server_log(self) -> str:
        return self.stdout.getvalue() + self.stderr.getvalue()

    def pp0_chunk_sizes(self) -> list[int]:
        """New-token counts of the single-request prefill batches PP0 has logged."""
        return [
            int(m)
            for m in re.findall(
                r" PP0 [^\]]*\] Prefill batch, #new-seq: 1, #new-token: (\d+)",
                self.server_log(),
            )
        ]

    def generate(self, input_ids: list[int], max_new_tokens: int = 12) -> dict:
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            },
            timeout=1200,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def assert_passkey(self, num_tokens: int, passkey: int) -> dict:
        ids = passkey_prompt_ids(self.tokenizer, num_tokens, passkey)
        out = self.generate(ids)
        self.assertIn(str(passkey), out["text"], f"{num_tokens=} {out['text']=}")
        return out
