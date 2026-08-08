"""dp-attention divert seam of the multi-layer EAGLE worker's extend branch.

``forward_batch_generation`` routes on ``batch.forward_mode.is_extend() or
batch.is_extend_in_batch``: dp_attention stamps the GLOBAL is_extend_in_batch,
so a rank whose LOCAL batch is not an extend can be diverted into the target
prefill branch.

Under the default spec+dp-attention scheduling the pre-merge no-mix sync
converts would-be decode ranks to IDLE while a peer prefills, so the only
reachable non-extend batch here is the staggered-IDLE one built by
``prepare_for_idle``: empty tensors, ``spec_info=None``. That shape must pass
through to the target forward UNCHANGED (behavior pin -- this also holds
before the guard was added). A NON-empty locally-DECODE/globally-EXTEND batch
is only reachable when the no-mix sync is skipped
(``--speculative-skip-dp-mlp-sync``): the scheduler relays its next-iteration
input_ids through ``spec_info`` and leaves ``batch.input_ids=None``, and the
worker has no divert synthesis for it -- it must trip loud BEFORE the target
forward, not forward ``input_ids=None`` into the model with peer ranks blocked
in the dp gather.

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


def _stub_worker(captured: dict) -> SimpleNamespace:
    def _target_forward(batch, **kwargs):
        captured["input_ids"] = batch.input_ids
        captured["out_cache_loc"] = batch.out_cache_loc
        raise _StopAtTargetForward()

    return SimpleNamespace(
        speculative_algorithm=SpeculativeAlgorithm.EAGLE,
        target_worker=SimpleNamespace(forward_batch_generation=_target_forward),
    )


def _diverted_decode_batch(spec_info, seq_lens) -> SimpleNamespace:
    """Locally-DECODE batch carrying the global extend stamp; input_ids=None
    because the scheduler relays spec decode ids via spec_info
    (resolve_forward_inputs skips the rebuild when a spec algorithm is set)."""
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=True,
        spec_info=spec_info,
        seq_lens=seq_lens,
        req_pool_indices=torch.arange(seq_lens.numel(), dtype=torch.int64),
        input_ids=None,
        out_cache_loc=None,
    )


class TestMultiLayerEagleDivertGuard(CustomTestCase):
    def test_non_empty_decode_divert_fails_loud(self):
        # The relay shape: spec_info is an EagleDraftInput whose bonus_tokens
        # carry the next-iteration ids that never got materialized into
        # input_ids. Without the guard this forwards input_ids=None into the
        # target forward and dies as an anonymous NoneType error downstream.
        captured = {}
        worker = _stub_worker(captured)
        batch = _diverted_decode_batch(
            spec_info=EagleDraftInput(
                bonus_tokens=torch.tensor([11, 12], dtype=torch.int64)
            ),
            seq_lens=torch.tensor([9, 17], dtype=torch.int64),
        )

        with self.assertRaises(RuntimeError) as context:
            MultiLayerEagleWorkerV2.forward_batch_generation(worker, batch)

        self.assertIn("diverted locally-DECODE", str(context.exception))
        self.assertIn("--speculative-skip-dp-mlp-sync", str(context.exception))
        # Tripped before the target forward; the batch is untouched.
        self.assertEqual(captured, {})
        self.assertIsNone(batch.input_ids)
        self.assertIsNone(batch.out_cache_loc)

    def test_non_empty_decode_divert_fails_loud_without_spec_info(self):
        # The tripwire must not depend on spec_info: a non-empty divert with
        # spec_info=None is the SAME unsupported shape, and a spec_info-gated
        # guard would let it fall through to the anonymous downstream crash
        # the guard exists to preempt.
        captured = {}
        worker = _stub_worker(captured)
        batch = _diverted_decode_batch(
            spec_info=None,
            seq_lens=torch.tensor([9, 17], dtype=torch.int64),
        )

        with self.assertRaisesRegex(RuntimeError, "diverted locally-DECODE"):
            MultiLayerEagleWorkerV2.forward_batch_generation(worker, batch)
        self.assertEqual(captured, {})
        self.assertIsNone(batch.input_ids)
        self.assertIsNone(batch.out_cache_loc)

    def test_staggered_idle_passes_through_unchanged(self):
        # BEHAVIOR PIN (green before and after the guard): the reachable
        # staggered-IDLE divert is the prepare_for_idle product -- empty
        # tensors, spec_info=None -- stamped with the global extend flag. It
        # must reach the target forward bit-identical: same (empty) input_ids
        # and out_cache_loc objects, no synthesis, no error.
        captured = {}
        worker = _stub_worker(captured)
        input_ids = torch.empty(0, dtype=torch.int64)
        out_cache_loc = torch.empty(0, dtype=torch.int64)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            is_extend_in_batch=True,
            spec_info=None,
            seq_lens=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            input_ids=input_ids,
            out_cache_loc=out_cache_loc,
        )

        with self.assertRaises(_StopAtTargetForward):
            MultiLayerEagleWorkerV2.forward_batch_generation(worker, batch)

        self.assertIs(captured["input_ids"], input_ids)
        self.assertIs(captured["out_cache_loc"], out_cache_loc)
        self.assertEqual(batch.input_ids.numel(), 0)

    def test_local_extend_unaffected(self):
        # A genuinely-extend batch (the normal prefill) must reach the target
        # forward untouched by the guard, whatever its spec_info.
        captured = {}
        worker = _stub_worker(captured)
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
        self.assertEqual(captured["input_ids"].tolist(), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
