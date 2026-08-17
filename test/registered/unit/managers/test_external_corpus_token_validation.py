import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    ListExternalCorporaReqOutput,
)
from sglang.srt.managers.tokenizer_control_mixin import (
    TokenizerControlMixin,
    _validate_external_corpus_token_chunks,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ExplodingChunk(list):
    def __iter__(self):
        raise AssertionError("over-limit chunks must not be scanned")


class TestExternalCorpusTokenValidation(unittest.TestCase):
    def test_canonical_lists_are_not_copied(self):
        chunks = [[1, 2], [3, 4]]

        validated = _validate_external_corpus_token_chunks(
            chunks,
            max_tokens=4,
            vocab_size=10,
            separator_token=-(2**31),
        )

        self.assertIs(validated, chunks)
        self.assertIs(validated[0], chunks[0])

    def test_over_limit_fails_before_scanning_tokens(self):
        chunks = [_ExplodingChunk([1, 2, 3])]

        with self.assertRaisesRegex(ValueError, "exceeds"):
            _validate_external_corpus_token_chunks(
                chunks,
                max_tokens=2,
                vocab_size=10,
                separator_token=-(2**31),
            )

    def test_dynamic_mutations_reject_dp_or_pp_topology_before_dispatch(self):
        methods = (
            ("add_external_corpus", (AddExternalCorpusReqInput(token_chunks=[[1]]),)),
            ("remove_external_corpus", ("docs",)),
        )
        for dp_size, pp_size in ((2, 1), (1, 2)):
            for method_name, args in methods:
                with self.subTest(method=method_name, dp_size=dp_size, pp_size=pp_size):
                    communicator = AsyncMock()
                    fake = SimpleNamespace(
                        server_args=SimpleNamespace(
                            speculative_algorithm="NGRAM",
                            dp_size=dp_size,
                            pp_size=pp_size,
                        ),
                        auto_create_handle_loop=lambda: None,
                        add_external_corpus_communicator=communicator,
                        remove_external_corpus_communicator=communicator,
                    )

                    result = asyncio.run(
                        getattr(TokenizerControlMixin, method_name)(fake, *args)
                    )

                    self.assertFalse(result.success)
                    self.assertIn("dp_size=1 and pp_size=1", result.message)
                    communicator.assert_not_called()

    def test_distributed_list_is_read_only_and_detects_divergence(self):
        communicator = AsyncMock(
            return_value=[
                ListExternalCorporaReqOutput(
                    success=True, corpus_token_counts={"docs": 3}
                ),
                ListExternalCorporaReqOutput(
                    success=True, corpus_token_counts={"docs": 3}
                ),
            ]
        )
        fake = SimpleNamespace(
            server_args=SimpleNamespace(
                speculative_algorithm="NGRAM", dp_size=2, pp_size=1
            ),
            auto_create_handle_loop=lambda: None,
            list_external_corpora_communicator=communicator,
        )

        result = asyncio.run(TokenizerControlMixin.list_external_corpora(fake))
        self.assertTrue(result.success)
        self.assertEqual(result.corpus_token_counts, {"docs": 3})

        communicator.return_value[1] = ListExternalCorporaReqOutput(
            success=True, corpus_token_counts={"other": 4}
        )
        result = asyncio.run(TokenizerControlMixin.list_external_corpora(fake))
        self.assertFalse(result.success)
        self.assertEqual(result.corpus_token_counts, {})
        self.assertIn("inconsistent", result.message)

    def test_dynamic_add_requires_positive_sam_budget(self):
        communicator = AsyncMock()
        fake = SimpleNamespace(
            server_args=SimpleNamespace(
                speculative_algorithm="NGRAM",
                speculative_ngram_external_sam_budget=0,
                dp_size=1,
                pp_size=1,
            ),
            auto_create_handle_loop=lambda: None,
            add_external_corpus_communicator=communicator,
        )

        result = asyncio.run(
            TokenizerControlMixin.add_external_corpus(
                fake, AddExternalCorpusReqInput(token_chunks=[[1]])
            )
        )

        self.assertFalse(result.success)
        self.assertIn("external-sam-budget", result.message)
        communicator.assert_not_called()

    def test_documents_over_token_limit_are_truncated_with_message(self):
        communicator = AsyncMock(
            return_value=[
                AddExternalCorpusReqOutput(
                    success=True,
                    corpus_id="docs",
                    message="loaded",
                    loaded_token_count=2,
                )
            ]
        )
        fake = SimpleNamespace(
            server_args=SimpleNamespace(
                speculative_algorithm="NGRAM",
                speculative_ngram_external_sam_budget=1,
                speculative_ngram_external_corpus_max_tokens=3,
                dp_size=1,
                pp_size=1,
            ),
            tokenizer=SimpleNamespace(
                encode=lambda _doc, add_special_tokens=False: [1, 2]
            ),
            auto_create_handle_loop=lambda: None,
            add_external_corpus_communicator=communicator,
        )

        result = asyncio.run(
            TokenizerControlMixin.add_external_corpus(
                fake,
                AddExternalCorpusReqInput(
                    corpus_id="docs", documents=["first", "second"]
                ),
            )
        )

        self.assertTrue(result.success)
        self.assertIn("truncated", result.message)
        dispatched = communicator.call_args.args[0]
        self.assertEqual(dispatched.token_chunks, [[1, 2]])


if __name__ == "__main__":
    unittest.main()
