"""Unit tests for batched request routing in TokenizerManager."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    SamplingParams,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    stamp_http_worker_ipc,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _generate_request(rid: str) -> TokenizedGenerateReqInput:
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text="",
        input_ids=[1, 2],
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        time_stats=APIServerReqTimeStats(),
    )


def _embedding_request(rid: str) -> TokenizedEmbeddingReqInput:
    return TokenizedEmbeddingReqInput(
        rid=rid,
        input_text="",
        input_ids=[1, 2],
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        time_stats=APIServerReqTimeStats(),
    )


class TestBatchRequestRouting(CustomTestCase):
    def _assert_routing(self, requests, expected_type):
        manager = TokenizerManager.__new__(TokenizerManager)
        dispatched = []

        def dispatch(batch_request):
            stamp_http_worker_ipc(batch_request, "ipc://http-worker")
            dispatched.append(batch_request)

        manager._dispatch_to_scheduler = dispatch
        manager._send_batch_request(requests)

        self.assertEqual(len(dispatched), 1)
        self.assertIsInstance(dispatched[0], expected_type)
        self.assertEqual(dispatched[0].rids, ["request-0", "request-1"])
        self.assertEqual(
            dispatched[0].http_worker_ipcs,
            ["ipc://http-worker", "ipc://http-worker"],
        )

    def test_generate_batch_preserves_rids(self):
        self._assert_routing(
            [_generate_request("request-0"), _generate_request("request-1")],
            BatchTokenizedGenerateReqInput,
        )

    def test_embedding_batch_preserves_rids(self):
        self._assert_routing(
            [_embedding_request("request-0"), _embedding_request("request-1")],
            BatchTokenizedEmbeddingReqInput,
        )


if __name__ == "__main__":
    unittest.main()
