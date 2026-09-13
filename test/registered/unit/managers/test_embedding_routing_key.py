"""Regression tests for embedding routing-key propagation."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    EmbeddingReqInput,
    TokenizedEmbeddingReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEmbeddingRoutingKey(CustomTestCase):
    def test_batch_split_preserves_routing_key(self):
        cases = (
            (["first", "second"], False),
            ([["query 1", "document 1"], ["query 2", "document 2"]], True),
        )

        for text, is_cross_encoder_request in cases:
            with self.subTest(cross_encoder=is_cross_encoder_request):
                req = EmbeddingReqInput(
                    text=text,
                    is_cross_encoder_request=is_cross_encoder_request,
                    routing_key="session-a",
                )
                req.normalize_batch_and_arguments()

                self.assertEqual(
                    [req[i].routing_key for i in range(2)],
                    ["session-a", "session-a"],
                )

    def test_tokenization_preserves_routing_key(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.preferred_sampling_params = None
        manager.sampling_params_class = SamplingParams
        manager.tokenizer = None
        manager.model_config = SimpleNamespace(vocab_size=128)
        manager.rid_to_state = {"embed-req": SimpleNamespace(time_stats=MagicMock())}
        req = EmbeddingReqInput(
            rid="embed-req",
            text="hello",
            sampling_params={},
            routing_key="session-a",
        )

        tokenized_req = manager._create_tokenized_object(req, "hello", [1, 2])

        self.assertEqual(tokenized_req.routing_key, "session-a")

    def test_msgpack_round_trip_preserves_routing_key(self):
        req = TokenizedEmbeddingReqInput(
            rid="embed-req",
            input_text="hello",
            input_ids=array("q", [1, 2]),
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(),
            routing_key="session-a",
        )

        decoded = msgpack_decode(msgpack_encode(req))

        self.assertEqual(decoded.routing_key, "session-a")

    @patch("sglang.srt.managers.scheduler.validate_input_length", return_value=None)
    @patch("sglang.srt.managers.scheduler.get_serving")
    @patch("sglang.srt.managers.scheduler.Req")
    def test_scheduler_receives_routing_key(
        self, req_cls, get_serving, _validate_input_length
    ):
        get_serving.return_value = SimpleNamespace(allow_auto_truncate=False)
        scheduler_req = MagicMock()
        scheduler_req.origin_input_ids = array("q", [1, 2])
        req_cls.return_value = scheduler_req

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tokenizer = None
        scheduler.max_req_input_len = 128
        scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
        scheduler._add_request_to_queue = MagicMock()
        tokenized_req = TokenizedEmbeddingReqInput(
            rid="embed-req",
            input_text="hello",
            input_ids=array("q", [1, 2]),
            sampling_params=SamplingParams(),
            mm_inputs=None,
            token_type_ids=None,
            routing_key="session-a",
        )

        scheduler.handle_embedding_request(tokenized_req)

        self.assertEqual(req_cls.call_args.kwargs["routing_key"], "session-a")


if __name__ == "__main__":
    unittest.main()
