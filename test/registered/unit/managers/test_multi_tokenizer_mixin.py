import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import BatchStrOutput, BatchTokenizedGenerateReqInput
from sglang.srt.managers.multi_tokenizer_mixin import (
    SenderWrapper,
    _attach_http_worker_info,
    _handle_output_by_index,
)

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


class TestMultiTokenizerMixin(unittest.TestCase):
    def test_batch_str_output_preserves_cached_tokens_details(self):
        output = _make_batch_str_output()

        single_output = _handle_output_by_index(output, 1)

        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.cached_tokens, [4])
        self.assertEqual(
            single_output.cached_tokens_details,
            [{"device": 1, "host": 3}],
        )

    def test_sender_wrapper_attaches_ipc_to_batch_request(self):
        batch_req = BatchTokenizedGenerateReqInput(
            batch=[
                SimpleNamespace(rid="rid-0", http_worker_ipc=None),
                SimpleNamespace(rid="rid-1", http_worker_ipc=None),
            ]
        )
        sender = SimpleNamespace(sent=[])
        sender.send_pyobj = sender.sent.append
        wrapper = SenderWrapper(
            SimpleNamespace(tokenizer_ipc_name="ipc://tokenizer-0"), sender
        )

        wrapper.send_pyobj(batch_req)

        self.assertIs(sender.sent[0], batch_req)
        self.assertEqual(batch_req.rids, ["rid-0", "rid-1"])
        self.assertEqual(
            batch_req.http_worker_ipcs,
            ["ipc://tokenizer-0", "ipc://tokenizer-0"],
        )
        self.assertEqual(
            [item.http_worker_ipc for item in batch_req.batch],
            ["ipc://tokenizer-0", "ipc://tokenizer-0"],
        )

    def test_split_batch_output_gets_single_and_batch_ipc(self):
        output = _make_batch_str_output()
        output.http_worker_ipcs = ["ipc://tokenizer-0", "ipc://tokenizer-1"]

        single_output = _handle_output_by_index(output, 1)
        _attach_http_worker_info(single_output, output.http_worker_ipcs[1])

        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.http_worker_ipc, "ipc://tokenizer-1")
        self.assertEqual(single_output.http_worker_ipcs, ["ipc://tokenizer-1"])


if __name__ == "__main__":
    unittest.main()
