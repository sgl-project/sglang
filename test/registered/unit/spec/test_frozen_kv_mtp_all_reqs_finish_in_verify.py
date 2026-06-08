import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.frozen_kv_mtp_info import (
    FrozenKVMTPDraftInput,
    FrozenKVMTPVerifyInput,
)
from sglang.srt.speculative.frozen_kv_mtp_worker import FrozenKVMTPWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

HIDDEN_SIZE = 8
TOPK = 1


def _stale_verify_input() -> FrozenKVMTPVerifyInput:
    """Placeholder for the verify input installed before target verification."""
    return FrozenKVMTPVerifyInput.__new__(FrozenKVMTPVerifyInput)


def _make_prefill_draft_input() -> FrozenKVMTPDraftInput:
    """A non-idle draft input shaped like a single-req prefill arrival."""
    return FrozenKVMTPDraftInput(
        topk_p=torch.ones(1, TOPK, dtype=torch.float32),
        topk_index=torch.zeros(1, TOPK, dtype=torch.int64),
        hidden_states=torch.zeros(1, HIDDEN_SIZE, dtype=torch.float32),
        bonus_tokens=torch.zeros(1, dtype=torch.int32),
    )


class _FakeVerifyOutput(SimpleNamespace):
    """Fake verify result for worker versions that return either an object or a tuple."""

    def __iter__(self):
        yield self.logits_output
        yield self
        yield self.can_run_cuda_graph


class TestFrozenKVMTPWorker(CustomTestCase):
    def _make_worker(self):
        worker = FrozenKVMTPWorker.__new__(FrozenKVMTPWorker)
        worker.device = torch.device("cpu")
        worker.topk = TOPK
        worker.model_config = SimpleNamespace(dtype=torch.float32)
        worker.server_args = SimpleNamespace(enable_dp_attention=False)
        worker._model_runner = SimpleNamespace(
            tp_group=None, model=SimpleNamespace(backbone_hidden_size=HIDDEN_SIZE)
        )
        worker.draft_tp_context = lambda _: nullcontext()

        stale_verify = _stale_verify_input()
        worker.draft = Mock(return_value=stale_verify)
        worker.verify = Mock(
            return_value=_FakeVerifyOutput(
                # Empty input_ids is the verify postcondition for:
                # has_finished=True and no unfinished requests remain.
                draft_extend_input=SimpleNamespace(
                    input_ids=torch.empty((0,), dtype=torch.int64)
                ),
                logits_output=SimpleNamespace(),
                accept_tokens=torch.empty((0,), dtype=torch.int64),
                num_correct_drafts_per_req_cpu=[0, 0],
                can_run_cuda_graph=False,
            )
        )
        worker.forward_draft_extend_after_decode = Mock()
        return worker, stale_verify

    def _make_decode_batch(self):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: False,
                is_idle=lambda: False,
            ),
            is_extend_in_batch=False,
            reqs=[SimpleNamespace(), SimpleNamespace()],
            spec_info=None,
        )

    def _forward_generation(self, worker, batch):
        with (
            patch(
                "sglang.srt.speculative.frozen_kv_mtp_worker."
                "speculative_moe_backend_context",
                lambda: nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.frozen_kv_mtp_worker."
                "speculative_moe_a2a_backend_context",
                lambda: nullcontext(),
            ),
        ):
            return worker.forward_batch_generation(batch)

    def test_forward_generation_installs_idle_draft_when_verify_finishes_all_reqs(
        self,
    ):
        worker, stale_verify = self._make_worker()
        batch = self._make_decode_batch()

        result = self._forward_generation(worker, batch)

        worker.forward_draft_extend_after_decode.assert_not_called()
        self.assertIsNot(batch.spec_info, stale_verify)
        self.assertIsInstance(batch.spec_info, FrozenKVMTPDraftInput)
        self.assertEqual(batch.spec_info.topk_index.shape, (0, TOPK))
        self.assertEqual(batch.spec_info.hidden_states.shape, (0, HIDDEN_SIZE))
        self.assertEqual(result.num_correct_drafts, 0)

    def test_idle_draft_input_accepts_next_iter_prefill_merge(self):
        worker, _ = self._make_worker()
        batch = self._make_decode_batch()

        self._forward_generation(worker, batch)

        # This mirrors the scheduler's next-iter failure mode:
        # running_batch.spec_info.merge_batch(other.spec_info). Without the
        # all-reqs-finished else branch, batch.spec_info is still the stale
        # FrozenKVMTPVerifyInput and this raises AttributeError.
        example_prefill_draft_input = _make_prefill_draft_input()
        batch.spec_info.merge_batch(example_prefill_draft_input)

        self.assertIsInstance(batch.spec_info, FrozenKVMTPDraftInput)
        self.assertEqual(batch.spec_info.topk_index.shape, (1, TOPK))
        self.assertEqual(batch.spec_info.hidden_states.shape, (1, HIDDEN_SIZE))


if __name__ == "__main__":
    unittest.main(verbosity=3)
