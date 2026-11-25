"""Start-up warm-up of DeepSeek-V4 token-parallel prefill."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.entrypoints import warmup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTokenizerManager:
    def __init__(self, chunked_prefill_size, context_len):
        self.server_args = SimpleNamespace(chunked_prefill_size=chunked_prefill_size)
        self.context_len = context_len
        self.prompt_lens = []

    async def generate_request(self, req, _request):
        self.prompt_lens.append(len(req.input_ids))
        yield {}


class TestDefaultWarmups(CustomTestCase):
    def _names(self, env):
        with mock.patch.dict(os.environ, env, clear=True):
            return warmup.default_warmups()

    def test_on_only_with_token_parallel_prefill(self):
        self.assertEqual(self._names({}), [])
        self.assertEqual(
            self._names({"SGLANG_DSV4_TOKEN_PARALLEL_PREFILL": "1"}),
            ["dsv4_token_parallel"],
        )
        self.assertEqual(
            self._names(
                {
                    "SGLANG_DSV4_TOKEN_PARALLEL_PREFILL": "1",
                    "SGLANG_DSV4_TOKEN_PARALLEL_WARMUP": "0",
                }
            ),
            [],
        )

    def test_default_name_is_registered(self):
        with mock.patch.dict(
            os.environ, {"SGLANG_DSV4_TOKEN_PARALLEL_PREFILL": "1"}, clear=True
        ):
            for name in warmup.default_warmups():
                self.assertIn(name, warmup._warmup_registry)


class TestTokenParallelWarmupPrompts(unittest.IsolatedAsyncioTestCase):
    async def _prompt_lens(self, chunk, context_len, env=None):
        tm = _FakeTokenizerManager(chunk, context_len)
        with mock.patch.dict(os.environ, env or {}, clear=True):
            await warmup.dsv4_token_parallel("null", tm)
        return tm.prompt_lens

    async def test_one_full_chunk_plus_each_eager_tail(self):
        # Each prompt is one full chunk and then a tail chunk of a size that runs
        # token-parallel (at least the minimum rows, at most one chunk).
        lens = await self._prompt_lens(chunk=8192, context_len=1 << 20)
        self.assertEqual(
            lens, [8192 + t for t in (2048, 2049, 3072, 4071, 4096, 6144, 8175, 8192)]
        )

    async def test_tails_respect_min_tokens_and_context(self):
        lens = await self._prompt_lens(
            chunk=8192,
            context_len=8192 + 4096,
            env={"SGLANG_DSV4_TOKEN_PARALLEL_MIN_TOKENS": "3000"},
        )
        self.assertEqual(lens, [8192 + 3072, 8192 + 4071])

    async def test_skipped_without_chunked_prefill(self):
        self.assertEqual(await self._prompt_lens(chunk=None, context_len=1 << 20), [])


if __name__ == "__main__":
    unittest.main()
