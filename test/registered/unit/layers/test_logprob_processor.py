"""DECODE-stage token-ids logprobs for mixed batches under no_copy_to_cpu.

Both DECODE producers call ``get_token_ids_logprobs(..., no_copy_to_cpu=True)``
unconditionally (``OutputLogprobProcessor.compute_logprobs`` for the normal
decode path, ``compute_spec_v2_logprobs`` for spec v2), and a batch is
processed whenever ANY request asks for token-ids logprobs -- requests that
did not ask contribute a ``None`` entry. Every val entry must stay
tensor-typed: the non-overlap result path
(``SchedulerBatchResultProcessor.move_logprobs_to_cpu``) calls ``.tolist()``
on every entry unconditionally, so a bare ``[]`` placeholder crashes the
scheduler with AttributeError on the first mixed decode batch.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logprob_processor import (
    LogprobStage,
    get_token_ids_logprobs_raw,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDecodeTokenIdsLogprobsMixedBatch(CustomTestCase):
    """Mixed batch: request 0 did NOT ask for token-ids logprobs (None entry),
    request 1 asked for tokens [1, 2]."""

    @staticmethod
    def _mixed_vals_idxs():
        logprobs = torch.log_softmax(
            torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8), dim=-1
        )
        return get_token_ids_logprobs_raw(
            logprobs,
            [None, [1, 2]],
            stage=LogprobStage.DECODE,
            no_copy_to_cpu=True,
        )

    def test_none_entry_stays_tensor_typed(self):
        vals, idxs = self._mixed_vals_idxs()
        # The None entry must be an EMPTY TENSOR (not a bare []), keeping the
        # val list homogeneous for downstream tensor-only consumers.
        self.assertTrue(torch.is_tensor(vals[0]))
        self.assertEqual(vals[0].numel(), 0)
        self.assertEqual(idxs[0], [])
        self.assertTrue(torch.is_tensor(vals[1]))
        self.assertEqual(vals[1].shape, (2,))
        self.assertEqual(idxs[1], [1, 2])

    def test_move_logprobs_to_cpu_survives_mixed_batch(self):
        # The REAL non-overlap consumer: move_logprobs_to_cpu calls
        # `v.tolist()` on every val entry unconditionally.
        vals, idxs = self._mixed_vals_idxs()
        logits_output = SimpleNamespace(
            next_token_logprobs=None,
            input_token_logprobs=None,
            next_token_top_logprobs_val=None,
            next_token_top_logprobs_idx=None,
            next_token_token_ids_logprobs_val=vals,
            next_token_token_ids_logprobs_idx=idxs,
        )
        batch = SimpleNamespace(return_logprob=True)

        SchedulerBatchResultProcessor.move_logprobs_to_cpu(
            object.__new__(SchedulerBatchResultProcessor),
            batch=batch,
            logits_output=logits_output,
        )

        converted = logits_output.next_token_token_ids_logprobs_val
        self.assertEqual(converted[0], [])
        self.assertEqual(len(converted[1]), 2)
        self.assertIsInstance(converted[1], list)

    def test_copy_to_cpu_path_unchanged(self):
        # no_copy_to_cpu=False (the plain synchronous path) keeps returning
        # plain lists for every entry.
        logprobs = torch.log_softmax(torch.randn(2, 8), dim=-1)
        vals, idxs = get_token_ids_logprobs_raw(
            logprobs,
            [None, [3]],
            stage=LogprobStage.DECODE,
            no_copy_to_cpu=False,
        )
        self.assertEqual(vals[0], [])
        self.assertIsInstance(vals[1], list)
        self.assertEqual(idxs[1], [3])


if __name__ == "__main__":
    unittest.main()
