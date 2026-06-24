import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import BatchEmbeddingOutput, BatchStrOutput
from sglang.srt.managers.multi_tokenizer_mixin import _handle_output_by_index

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

    def test_batch_embedding_output_preserves_fields(self):
        # Regression: the embedding branch of _handle_output_by_index used to
        # omit retraction_counts (a required field), so splitting any embedding
        # batch raised TypeError and crashed the multi-tokenizer routing loop.
        hidden0, hidden1 = object(), object()
        time0, time1 = object(), object()
        output = BatchEmbeddingOutput(
            rids=["rid-0", "rid-1"],
            finished_reasons=[None, {"type": "length"}],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            prompt_tokens=[10, 20],
            cached_tokens=[3, 4],
            placeholder_tokens_idx=[None, None],
            placeholder_tokens_val=[None, None],
            retraction_counts=[0, 2],
            cached_tokens_details=[
                {"device": 3, "host": 0},
                {"device": 1, "host": 3},
            ],
            time_stats=[time0, time1],
            pooled_hidden_states=[hidden0, hidden1],
        )

        single_output = _handle_output_by_index(output, 1)

        self.assertIsInstance(single_output, BatchEmbeddingOutput)
        self.assertIsNot(single_output, output)
        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.embeddings, [[0.3, 0.4]])
        self.assertEqual(single_output.prompt_tokens, [20])
        self.assertEqual(single_output.cached_tokens, [4])
        self.assertEqual(single_output.retraction_counts, [2])
        self.assertEqual(
            single_output.cached_tokens_details,
            [{"device": 1, "host": 3}],
        )
        self.assertEqual(single_output.time_stats, [time1])
        self.assertEqual(len(single_output.pooled_hidden_states), 1)
        self.assertIs(single_output.pooled_hidden_states[0], hidden1)


if __name__ == "__main__":
    unittest.main()
