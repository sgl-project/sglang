"""Opt-in protocol of the greedy draft argmax exchange (no GPU).

Only draft-model LogitsProcessors whose enable_draft_argmax_exchange() returns
True may return argmax partials, and only under the draft_argmax_only forward
flag, which compiled forwards never read."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.speculative.eagle_worker_v2 as eagle_worker_v2
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.runtime_context import ForwardFlags, get_forward
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _bare_processor(cls=LogitsProcessor, *, eligible=True):
    proc = cls.__new__(cls)
    torch.nn.Module.__init__(proc)
    proc.can_exchange_argmax = eligible
    proc.draft_argmax_exchange = False
    proc.vocab_size = 1024
    return proc


class _DecliningProcessor(LogitsProcessor):
    """A draft processor that keeps its own top-1 path; it calls super() first,
    so the worker must clear the attribute it set."""

    def enable_draft_argmax_exchange(self):
        super().enable_draft_argmax_exchange()
        return False


class _ModelStub(torch.nn.Module):
    def __init__(self, proc):
        super().__init__()
        self.logits_processor = proc


def _opt_in(draft_proc, *, rejection_sampling=False, shared_head=False):
    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.topk = 1
    target_proc = draft_proc if shared_head else _bare_processor()
    worker.target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(model=_ModelStub(target_proc))
    )
    worker.draft_runner = SimpleNamespace(model=_ModelStub(draft_proc))
    with (
        patch.object(eagle_worker_v2, "_is_cuda", True),
        patch.object(
            eagle_worker_v2,
            "get_spec",
            lambda: SimpleNamespace(
                speculative_use_rejection_sampling=rejection_sampling
            ),
        ),
        patch.object(
            eagle_worker_v2, "get_parallel", lambda: SimpleNamespace(tp_rank=0)
        ),
    ):
        worker._init_draft_argmax_exchange()
    return worker, target_proc


class TestDraftArgmaxExchangeOptIn(CustomTestCase):
    def test_flag_is_scoped_and_eager_only(self):
        self.assertIn("draft_argmax_only", ForwardFlags._DEFAULTS)
        self.assertNotIn("draft_argmax_only", ForwardFlags._GRAPH_VISIBLE)
        with get_forward().scoped(draft_argmax_only=True):
            with get_forward().scoped(draft_argmax_only=False):
                self.assertFalse(get_forward().draft_argmax_only)
            self.assertTrue(get_forward().draft_argmax_only)
        self.assertFalse(get_forward().draft_argmax_only)

    def test_stock_draft_processor_opts_in(self):
        draft = _bare_processor()
        worker, target = _opt_in(draft)
        self.assertTrue(worker.draft_argmax_exchange)
        self.assertTrue(worker.draft_extend_argmax_only)
        self.assertEqual(worker.draft_argmax_vocab_size, draft.vocab_size)
        with get_forward().scoped(draft_argmax_only=True):
            self.assertTrue(draft._draft_argmax_requested())
            self.assertFalse(target._draft_argmax_requested())
        self.assertFalse(draft._draft_argmax_requested())

    def test_declining_and_ineligible_processors_keep_logits(self):
        for draft in (
            _bare_processor(_DecliningProcessor),
            _bare_processor(eligible=False),
        ):
            with self.subTest(processor=type(draft).__name__):
                worker, _ = _opt_in(draft)
                self.assertFalse(worker.draft_argmax_exchange)
                self.assertFalse(worker.draft_extend_argmax_only)
                self.assertFalse(draft.draft_argmax_exchange)
                with get_forward().scoped(draft_argmax_only=True):
                    self.assertFalse(draft._draft_argmax_requested())

    def test_head_shared_with_the_target_is_not_opted_in(self):
        """The target's processor also serves verify, which needs logits."""
        draft = _bare_processor()
        worker, _ = _opt_in(draft, shared_head=True)
        self.assertFalse(worker.draft_argmax_exchange)
        self.assertFalse(draft.draft_argmax_exchange)

    def test_rejection_sampling_keeps_draft_extend_logits(self):
        worker, _ = _opt_in(_bare_processor(), rejection_sampling=True)
        self.assertTrue(worker.draft_argmax_exchange)
        self.assertFalse(worker.draft_extend_argmax_only)

    def test_compiled_gate_never_takes_partials(self):
        draft = _bare_processor()
        draft.enable_draft_argmax_exchange()
        torch._dynamo.reset()
        compiled = torch.compile(
            lambda x: x + 1 if draft._draft_argmax_requested() else x - 1,
            fullgraph=True,
            backend="eager",
        )
        try:
            with get_forward().scoped(draft_argmax_only=True):
                self.assertEqual(compiled(torch.zeros(1)).item(), -1.0)
        finally:
            torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
