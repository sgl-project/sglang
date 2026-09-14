import unittest
from array import array

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.io_struct import BatchStrOutput, BatchTokenIDOutput
from sglang.srt.managers.multi_tokenizer_mixin import _handle_output_by_index
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeTokenizer:
    is_fast = True

    def __init__(self):
        self.calls = []

    def batch_decode(self, ids_list, **kwargs):
        rows = [list(ids) for ids in ids_list]
        self.calls.append(rows)
        return [",".join(str(token_id) for token_id in row) for row in rows]


def _make_manager():
    manager = object.__new__(DetokenizerManager)
    manager.tokenizer = _FakeTokenizer()
    manager.vocab_size = 1024
    manager.decode_status = {}
    manager.disable_tokenizer_batch_decode = False
    manager.is_tool_call_parser_gpt_oss = False
    return manager


def _make_output(mask):
    n = 2
    return BatchTokenIDOutput(
        rids=["text", "tokens"],
        http_worker_ipcs=["worker", "worker"],
        finished_reasons=[None] * n,
        decoded_texts=[""] * n,
        decode_ids=[array("q", [65]), array("q", [66])],
        read_offsets=[0] * n,
        output_ids=[array("q", [65]), array("q", [66])],
        output_text_required=mask,
        skip_special_tokens=[True] * n,
        spaces_between_special_tokens=[True] * n,
        no_stop_trim=[False] * n,
        prompt_tokens=[1] * n,
        reasoning_tokens=[0] * n,
        completion_tokens=[1] * n,
        cached_tokens=[0] * n,
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
        output_token_entropy_val=None,
        output_token_sampling_mask=None,
        output_token_sampling_logprobs=None,
        output_hidden_states=None,
        routed_experts=None,
        indexer_topk=None,
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
    )


class TestTokenOnlyDetokenization(unittest.TestCase):
    def test_all_token_rows_bypass_detokenizer(self):
        manager = _make_manager()
        recv_obj = _make_output([False, False])

        output = manager.handle_batch_token_id_out(recv_obj)

        self.assertIs(output, recv_obj)
        self.assertEqual(manager.tokenizer.calls, [])
        self.assertEqual(manager.decode_status, {})

    def test_mixed_batch_decodes_only_text_rows(self):
        manager = _make_manager()

        output = manager.handle_batch_token_id_out(_make_output([True, False]))

        self.assertIsInstance(output, BatchStrOutput)
        self.assertEqual(output.rids, ["text", "tokens"])
        self.assertEqual([list(ids) for ids in output.output_ids], [[65], [66]])
        self.assertEqual(output.prompt_tokens, [1, 1])
        self.assertEqual(output.output_strs, ["65", ""])
        self.assertEqual(manager.tokenizer.calls, [[[65]]])
        self.assertEqual(set(manager.decode_status), {"text"})

    def test_invalid_mask_falls_back_to_full_decode(self):
        for mask in (None, [False], [True, 0]):
            with self.subTest(mask=mask):
                manager = _make_manager()

                output = manager.handle_batch_token_id_out(_make_output(mask))

                self.assertEqual(output.output_strs, ["65", "66"])
                self.assertEqual(manager.tokenizer.calls, [[[65], [66]]])
                self.assertEqual(set(manager.decode_status), {"text", "tokens"})

    def test_multi_detokenizer_split_preserves_mask(self):
        output = _handle_output_by_index(_make_output([True, False]), 1)

        self.assertEqual(output.rids, ["tokens"])
        self.assertEqual(output.output_text_required, [False])


if __name__ == "__main__":
    unittest.main()
