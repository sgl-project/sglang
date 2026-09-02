"""Gating of the per-round batch_is_full reset in _get_new_batch_prefill_raw:
cleared for hybrid SWA and for a hybrid SSM model with a mamba-aware cache,
kept for a hybrid SSM model whose cache lacks mamba support and for a model
that is neither hybrid SSM nor hybrid SWA (no preemption in any case)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _scheduler(
    *, is_hybrid_ssm: bool, is_hybrid_swa: bool = False, supports_mamba: bool = True
) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = MagicMock()
    s.grammar_manager.has_waiting_grammars.return_value = False
    s.enable_hierarchical_cache = False
    s.enable_hicache_storage = False
    s.enable_priority_preemption = False
    s.is_hybrid_swa = is_hybrid_swa
    s.is_hybrid_ssm = is_hybrid_ssm
    s.tree_cache = SimpleNamespace(supports_mamba=lambda: supports_mamba)
    s.waiting_queue = []
    s.chunked_req = None
    return s


class TestHybridSsmAdmissionLatch(CustomTestCase):
    def _latch_after_round(self, s):
        running_batch = SimpleNamespace(batch_is_full=True, reqs=[])
        with patch(
            "sglang.srt.managers.scheduler.get_memory",
            return_value=SimpleNamespace(enable_flexkv=False),
        ):
            ret, rb = Scheduler._get_new_batch_prefill_raw(
                s, prefill_delayer_single_pass=None, running_batch=running_batch
            )
        self.assertIsNone(ret)
        return rb.batch_is_full

    def test_hybrid_ssm_with_mamba_radix_cache_resets_latch(self):
        self.assertFalse(self._latch_after_round(_scheduler(is_hybrid_ssm=True)))

    def test_hybrid_ssm_without_mamba_cache_keeps_latch(self):
        self.assertTrue(
            self._latch_after_round(
                _scheduler(is_hybrid_ssm=True, supports_mamba=False)
            )
        )

    def test_hybrid_swa_resets_latch(self):
        self.assertFalse(
            self._latch_after_round(_scheduler(is_hybrid_ssm=False, is_hybrid_swa=True))
        )

    def test_non_hybrid_model_keeps_latch(self):
        self.assertTrue(self._latch_after_round(_scheduler(is_hybrid_ssm=False)))


if __name__ == "__main__":
    unittest.main()
