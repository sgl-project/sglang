"""Pipeline parallelism for long context on Intel XPU (PP=2, Llama-3.1-8B-Instruct).

Covers the paths the PP long-context guide documents: many-chunk prefill up to
~120K tokens, uneven layer partitions, PP communication on a dedicated stream,
dynamic chunking, and the dynamic-chunking fallback when its probe runs out of
memory.
"""

import io
import re
import unittest

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_xpu_ci(est_time=720, suite="nightly-kernel-main-test-2-gpu-xpu", nightly=True)

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


def _passkey_prompt_ids(tokenizer, num_tokens: int, passkey: int) -> list[int]:
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


class _PPServerBase(CustomTestCase):
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

    def _pp0_chunk_sizes(self) -> list[int]:
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
        ids = _passkey_prompt_ids(self.tokenizer, num_tokens, passkey)
        out = self.generate(ids)
        self.assertIn(str(passkey), out["text"], f"{num_tokens=} {out['text']=}")
        return out


@unittest.skipUnless(torch.xpu.is_available(), "needs Intel XPU")
class TestXPUPPLongContext(_PPServerBase):
    """Fixed 4K chunks, an uneven layer split and PP comm on its own stream."""

    other_args = ["--chunked-prefill-size", "4096"]
    env = {"SGLANG_PP_LAYER_PARTITION": "15,17", "SGLANG_PP_COMM_OVERLAP": "1"}

    def test_comm_overlap_stream_is_used(self):
        self.assertIn(
            "PP tensor communication overlap enabled on a dedicated xpu stream",
            self.server_log(),
        )

    def test_passkey_retrieval_across_many_chunks(self):
        # 120K tokens is ~30 chunks, each crossing the stage boundary.
        for num_tokens, passkey in ((32768, 61442), (120000, 27819)):
            with self.subTest(num_tokens=num_tokens):
                out = self.assert_passkey(num_tokens, passkey)
                self.assertEqual(out["meta_info"]["prompt_tokens"], num_tokens)


@unittest.skipUnless(torch.xpu.is_available(), "needs Intel XPU")
class TestXPUPPDynamicChunking(_PPServerBase):
    """Dynamic chunking from a 16K initial chunk.

    Its 20K-token probes take most of the memory the KV pool leaves free, and the PP
    communicator is built only after them, so the server must release that memory
    before it serves.
    """

    BASE = 16384
    other_args = ["--chunked-prefill-size", str(BASE), "--enable-dynamic-chunking"]

    def test_chunks_shrink_as_the_prefix_grows(self):
        self.assertRegex(
            self.server_log(), r"\[PP Dynamic Chunk\] \[PP0\] Predictor ready"
        )
        before = len(self._pp0_chunk_sizes())
        self.assert_passkey(65536, 40913)
        sizes = self._pp0_chunk_sizes()[before:]
        # The last chunk is the remainder; every predicted chunk before it is
        # no larger than the previous one and no smaller than a quarter of the base.
        predicted = sizes[:-1]
        self.assertEqual(predicted[0], self.BASE, sizes)
        self.assertEqual(predicted, sorted(predicted, reverse=True), sizes)
        self.assertLess(predicted[-1], self.BASE, sizes)
        self.assertGreaterEqual(min(predicted), self.BASE // 4, sizes)
        self.assertEqual(sum(sizes), 65536, sizes)


@unittest.skipUnless(torch.xpu.is_available(), "needs Intel XPU")
class TestXPUPPDynamicChunkingProbeFallback(_PPServerBase):
    """A probe that runs out of memory disables dynamic chunking, not the server.

    On a 24 GB card a 0.85 static fraction leaves PP0 too little headroom for the
    15K-token probe; the server must still come up and serve. A full 12K chunk does
    not fit at this fraction either, so the request stays below one chunk.
    """

    mem_fraction_static = "0.85"
    other_args = ["--chunked-prefill-size", "12288", "--enable-dynamic-chunking"]

    def test_server_keeps_serving_after_the_probe_fails(self):
        if "Failed to profile prefill latency" not in self.server_log():
            self.skipTest("the probe fit in memory on this card")
        self.assertIn("Dynamic chunking will be disabled", self.server_log())
        self.assert_passkey(2048, 73104)


if __name__ == "__main__":
    unittest.main()
