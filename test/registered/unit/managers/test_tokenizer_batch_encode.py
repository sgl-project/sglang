"""CPU regressions for tokenizer-manager batch encoding with a local tokenizer."""

import asyncio
import unittest
from types import SimpleNamespace

from transformers import BertTokenizerFast

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import EmbeddingReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCrossEncoderBatchTokenization(CustomTestCase):
    def setUp(self):
        self.tokenizer = BertTokenizerFast(
            vocab={
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4,
                "query": 5,
                "first": 6,
                "second": 7,
                "document": 8,
            }
        )
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.tokenizer = self.tokenizer
        self.manager.async_dynamic_batch_tokenizer = None
        self.manager.is_generation = False
        self.manager.model_config = SimpleNamespace(
            is_embedding_gemma=False, vocab_size=len(self.tokenizer)
        )
        self.manager.context_len = 32
        self.manager.num_reserved_tokens = 0
        self.manager.allow_auto_truncate = False
        self.manager.validate_total_tokens = True
        self.manager.preferred_sampling_params = None
        self.manager.sampling_params_class = SamplingParams
        self.manager.rid_to_state = {}

    def test_cross_encoder_pairs_keep_per_request_token_sequences(self):
        """Batch reranking must not add a nesting level or flatten a one-pair batch."""
        for pairs in (
            [["query", "first document"]],
            [["query", "first document"], ["query", "second document document"]],
        ):
            with self.subTest(batch_size=len(pairs)):
                request = EmbeddingReqInput(text=pairs, is_cross_encoder_request=True)
                request.normalize_batch_and_arguments()
                self.manager.rid_to_state = {
                    rid: SimpleNamespace(time_stats=APIServerReqTimeStats())
                    for rid in request.rid
                }

                tokenized = asyncio.run(
                    self.manager._batch_tokenize_and_process(
                        request.batch_size, request
                    )
                )
                expected = self.tokenizer(pairs, return_token_type_ids=True)
                self.assertEqual(
                    [list(req.input_ids) for req in tokenized], expected["input_ids"]
                )
                self.assertEqual(
                    [req.token_type_ids for req in tokenized],
                    expected["token_type_ids"],
                )


if __name__ == "__main__":
    unittest.main()
