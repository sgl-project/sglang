"""Exact prompt IDs must reach the corpus loader without retokenization."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExternalCorpusTokenChunks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = TokenizerControlMixin()
        self.manager.auto_create_handle_loop = Mock()
        self.manager.server_args = SimpleNamespace(
            speculative_ngram_external_corpus_max_tokens=4
        )
        self.manager.tokenizer = Mock()
        self.manager.model_config = SimpleNamespace(vocab_size=128256)
        self.manager.add_external_corpus_communicator = AsyncMock(
            return_value=[
                AddExternalCorpusReqOutput(
                    success=True, corpus_id="prompt", loaded_token_count=3
                )
            ]
        )
        spec = patch(
            "sglang.srt.managers.tokenizer_control_mixin.get_spec",
            return_value=SimpleNamespace(speculative_algorithm="NGRAM"),
        )
        spec.start()
        self.addCleanup(spec.stop)

    async def test_exact_tokens_include_bos_and_do_not_use_tokenizer(self):
        request = AddExternalCorpusReqInput(
            corpus_id="prompt", token_chunks=[[128000, 17, 0]]
        )
        result = await self.manager.add_external_corpus(request)
        self.assertTrue(result.success)
        self.assertEqual(result.loaded_token_count, 3)
        self.manager.add_external_corpus_communicator.assert_awaited_once_with(request)
        self.assertEqual(request.token_chunks, [[128000, 17, 0]])
        self.manager.tokenizer.encode.assert_not_called()

    async def test_invalid_ids_or_shapes_never_reach_native_loader(self):
        for chunks in (
            [],
            [[]],
            [1],
            [[-1]],
            [[128256]],
            [[2**31]],
            [[True]],
            [[1.5]],
            [["1"]],
        ):
            with self.subTest(chunks=chunks):
                result = await self.manager.add_external_corpus(
                    AddExternalCorpusReqInput(token_chunks=chunks)
                )
                self.assertFalse(result.success)
        self.manager.add_external_corpus_communicator.assert_not_awaited()

    async def test_exact_tokens_fail_instead_of_truncating_over_budget(self):
        result = await self.manager.add_external_corpus(
            AddExternalCorpusReqInput(token_chunks=[[1, 2], [3, 4, 5]])
        )
        self.assertFalse(result.success)
        self.assertIn("exceeds", result.message)
        self.manager.add_external_corpus_communicator.assert_not_awaited()

    async def test_ambiguous_sources_are_rejected(self):
        for extra in ({"documents": ["code"]}, {"file_path": "corpus.jsonl"}):
            result = await self.manager.add_external_corpus(
                AddExternalCorpusReqInput(token_chunks=[[1]], **extra)
            )
            self.assertFalse(result.success)
        self.manager.add_external_corpus_communicator.assert_not_awaited()

    async def test_existing_document_path_still_tokenizes(self):
        self.manager.tokenizer.encode.return_value = [4, 5, 6]
        request = AddExternalCorpusReqInput(corpus_id="prompt", documents=["code"])
        result = await self.manager.add_external_corpus(request)
        self.assertTrue(result.success)
        self.manager.tokenizer.encode.assert_called_once_with(
            "code", add_special_tokens=False
        )
        self.assertEqual(request.token_chunks, [[4, 5, 6]])


if __name__ == "__main__":
    unittest.main()
