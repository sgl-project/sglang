"""dp-attention divert seam of the multi-layer EAGLE worker's extend branch.

forward_batch_generation routes on ``batch.forward_mode.is_extend() or
batch.is_extend_in_batch``: dp_attention stamps the GLOBAL is_extend_in_batch,
so a rank whose LOCAL batch is not an extend can be diverted into the target
prefill branch with ``input_ids`` nulled by the scheduler for the spec
iteration.

Under the default spec+dp-attention scheduling the no-mix sync converts
would-be decode ranks to IDLE while a peer prefills, so the only reachable
non-extend batch here is the staggered-IDLE one: the divert-fill must no-op on
empty tensors instead of forwarding ``input_ids=None`` into the target model.
A NON-empty locally-DECODE/globally-EXTEND batch is only possible when the
no-mix sync is skipped (``--speculative-skip-dp-mlp-sync``) -- the worker has
no divert synthesis for it and must trip loud BEFORE the target forward, not
crash later in rotate_input_ids with peer ranks hung in the dp gather.

White-box unit test in the ``__new__`` + SimpleNamespace style: only the
attributes the extend branch touches are provided.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleWorkerV2,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StopAtTargetForward(Exception):
    pass


def _stub_worker() -> SimpleNamespace:
    def _target_forward(*args, **kwargs):
        raise _StopAtTargetForward()

    return SimpleNamespace(
        speculative_algorithm=SpeculativeAlgorithm.EAGLE,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(8 * 16, dtype=torch.int32).reshape(8, 16)
        ),
        target_worker=SimpleNamespace(forward_batch_generation=_target_forward),
    )


def _diverted_batch(bonus_tokens, seq_lens, forward_mode) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=forward_mode,
        is_extend_in_batch=True,
        spec_info=EagleDraftInput(bonus_tokens=bonus_tokens),
        seq_lens=seq_lens,
        req_pool_indices=torch.arange(seq_lens.numel(), dtype=torch.int64),
        input_ids=None,
        out_cache_loc=None,
    )


class TestMultiLayerEagleDivertGuard(CustomTestCase):
    def test_non_empty_decode_divert_fails_loud(self):
        worker = _stub_worker()
        batch = _diverted_batch(
            bonus_tokens=torch.tensor([11, 12], dtype=torch.int64),
            seq_lens=torch.tensor([9, 17], dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
        )

        with self.assertRaises(RuntimeError) as context:
            MultiLayerEagleWorkerV2.forward_batch_generation(worker, batch)

        self.assertIn("diverted locally-DECODE", str(context.exception))
        # Tripped before the fill and before the target forward.
        self.assertIsNone(batch.input_ids)
        self.assertIsNone(batch.out_cache_loc)

    def test_non_empty_decode_divert_fails_loud_without_bonus_shape(self):
        # The tripwire must not be gated on the spec_info carrying a
        # bonus-shaped EagleDraftInput: a non-empty divert with a None (or
        # shape-mismatched) bonus is the SAME unsupported case, and a
        # bonus-shape-gated guard would let it fall through to the anonymous
        # downstream crash the guard exists to preempt.
        worker = _stub_worker()
        batch = _diverted_batch(
            bonus_tokens=None,
            seq_lens=torch.tensor([9, 17], dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
        )

        with self.assertRaisesRegex(RuntimeError, "diverted locally-DECODE"):
            MultiLayerEagleWorkerV2.forward_batch_generation(worker, batch)
        self.assertIsNone(batch.input_ids)
        self.assertIsNone(batch.out_cache_loc)

    def test_staggered_idle_fill_noops_into_target_forward(self):
        worker = _stub_worker()
        batch = _diverted_batch(
            bonus_tokens=torch.empty(0, dtype=torch.int64),
            seq_lens=torch.empty(0, dtype=torch.int64),
            forward_mode=ForwardMode.IDLE,
        )

        with self.assertRaises(_StopAtTargetForward):
            MultiLayerEagleWorkerV2.forward_batch_generation(worker, batch)

        # The fill ran (idle no-op): empty int64 tensors, no None left for
        # downstream padding helpers to trip on.
        self.assertEqual(batch.input_ids.numel(), 0)
        self.assertEqual(batch.input_ids.dtype, torch.int64)
        self.assertEqual(batch.out_cache_loc.numel(), 0)
        self.assertEqual(batch.out_cache_loc.dtype, torch.int64)

    def test_local_extend_unaffected(self):
        # A genuinely-extend batch (the normal prefill) must reach the target
        # forward untouched by the guard, whatever its spec_info.
        worker = _stub_worker()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            is_extend_in_batch=True,
            spec_info=None,
            seq_lens=torch.tensor([4], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            out_cache_loc=torch.arange(4, dtype=torch.int64),
        )
        with self.assertRaises(_StopAtTargetForward):
            MultiLayerEagleWorkerV2.forward_batch_generation(worker, batch)
        self.assertEqual(batch.input_ids.tolist(), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
