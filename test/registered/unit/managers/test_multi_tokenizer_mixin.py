import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    BatchStrOutput,
    BatchTokenizedGenerateReqInput,
    SamplingParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import _handle_output_by_index
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    stamp_http_worker_ipc,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_batch_str_output() -> BatchStrOutput:
    return BatchStrOutput(
        rids=["rid-0", "rid-1"],
        spec_verify_ct=[0, 0],
        spec_num_correct_drafts=[0, 0],
        spec_correct_drafts_histogram=[[], []],
        finished_reasons=[None, {"type": "length"}],
        output_strs=["first", "second"],
        output_ids=[[1], [2]],
        prompt_tokens=[10, 20],
        completion_tokens=[1, 2],
        reasoning_tokens=[0, 0],
        cached_tokens=[3, 4],
        cached_tokens_details=[
            {"device": 3, "host": 0},
            {"device": 1, "host": 3},
        ],
        input_token_logprobs_val=[[], []],
        input_token_logprobs_idx=[[], []],
        output_token_logprobs_val=[[], []],
        output_token_logprobs_idx=[[], []],
        input_top_logprobs_val=[[], []],
        input_top_logprobs_idx=[[], []],
        output_top_logprobs_val=[[], []],
        output_top_logprobs_idx=[[], []],
        input_token_ids_logprobs_val=[[], []],
        input_token_ids_logprobs_idx=[[], []],
        output_token_ids_logprobs_val=[[], []],
        output_token_ids_logprobs_idx=[[], []],
        output_token_entropy_val=[0.0, 0.0],
        output_hidden_states=[None, None],
        routed_experts=[None, None],
        indexer_topk=[None, None],
        placeholder_tokens_idx=[None, None],
        placeholder_tokens_val=[None, None],
        retraction_counts=[0, 0],
    )


def _make_tokenized_generate_req(rid: str) -> TokenizedGenerateReqInput:
    req = TokenizedGenerateReqInput(
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
    )
    req.time_stats = APIServerReqTimeStats()
    return req


class TestMultiTokenizerMixin(unittest.TestCase):

    def test_send_batch_request_sets_batch_rids(self):
        tokenizer_manager = TokenizerManager.__new__(TokenizerManager)
        dispatched = []
        tokenizer_manager._dispatch_to_scheduler = dispatched.append

        tokenizer_manager._send_batch_request(
            [
                _make_tokenized_generate_req("rid-0"),
                _make_tokenized_generate_req("rid-1"),
            ]
        )

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].rids, ["rid-0", "rid-1"])

    def test_stamp_http_worker_ipc_fills_batch_and_child_requests(self):
        batch = BatchTokenizedGenerateReqInput(
            batch=[
                _make_tokenized_generate_req("rid-0"),
                _make_tokenized_generate_req("rid-1"),
            ]
        )

        stamp_http_worker_ipc(batch, "ipc-0")

        self.assertEqual(batch.rids, ["rid-0", "rid-1"])
        self.assertEqual(batch.http_worker_ipcs, ["ipc-0", "ipc-0"])
        self.assertEqual(
            [req.http_worker_ipc for req in batch.batch], ["ipc-0", "ipc-0"]
        )

    def test_stamp_http_worker_ipc_preserves_existing_batch_rids(self):
        batch = BatchTokenizedGenerateReqInput(
            rids=["existing-rid"],
            batch=[_make_tokenized_generate_req("child-rid")],
        )

        stamp_http_worker_ipc(batch, "ipc-1")

        self.assertEqual(batch.rids, ["existing-rid"])
        self.assertEqual(batch.http_worker_ipcs, ["ipc-1"])
        self.assertEqual(batch.batch[0].http_worker_ipc, "ipc-1")

    def test_batch_str_output_preserves_cached_tokens_details(self):
        output = _make_batch_str_output()

        single_output = _handle_output_by_index(output, 1)

        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.cached_tokens, [4])
        self.assertEqual(
            single_output.cached_tokens_details,
            [{"device": 1, "host": 3}],
        )


if __name__ == "__main__":
    unittest.main()
