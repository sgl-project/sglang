"""A CP hook that returns no embeddings must not let the runner embed a shard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.model_executor.runner.eager_runner import _cp_full_sequence_embeds
from sglang.test.test_utils import CustomTestCase


class TestCpFullSequenceEmbeds(CustomTestCase):
    """The runner owns the full layout until cp_shard_model_inputs shards it."""

    def setUp(self):
        self.num_tokens = 6
        self.input_ids = torch.arange(self.num_tokens, dtype=torch.long)
        self.forward_batch = SimpleNamespace(input_ids=self.input_ids)
        self.embed = Mock(return_value=torch.ones(self.num_tokens, 8))
        self.model = SimpleNamespace(
            get_input_embeddings=Mock(return_value=self.embed),
            pp_group=SimpleNamespace(is_first_rank=True),
        )

    def test_hook_embeddings_are_returned_untouched(self):
        prepared = torch.full((self.num_tokens, 8), 2.0)

        out = _cp_full_sequence_embeds(self.model, self.forward_batch, prepared)

        self.assertIs(out, prepared)
        self.embed.assert_not_called()

    def test_missing_embeddings_are_embedded_over_the_full_layout(self):
        out = _cp_full_sequence_embeds(self.model, self.forward_batch, None)

        # One row per *unsharded* token: embedding a rank's slice would all-reduce
        # unrelated rows inside VocabParallelEmbedding.
        self.assertEqual(tuple(out.shape), (self.num_tokens, 8))
        self.embed.assert_called_once()
        self.assertIs(self.embed.call_args.args[0], self.input_ids)

    def test_non_first_pipeline_rank_keeps_the_hook_result(self):
        self.model.pp_group.is_first_rank = False

        out = _cp_full_sequence_embeds(self.model, self.forward_batch, None)

        self.assertIsNone(out)
        self.embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
