"""Unit tests for the Elastic EP forward-path fast-path optimization.

These tests verify that `maybe_recover_ep_ranks` treats `active_ranks_cpu`
as the authoritative signal for the "no recovery needed" fast path, avoiding
a host-device synchronization on the forward path.

Reference: `python/sglang/srt/elastic_ep/elastic_ep.py::maybe_recover_ep_ranks`
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.elastic_ep import elastic_ep as elastic_ep_module
from sglang.srt.elastic_ep.elastic_ep import maybe_recover_ep_ranks
from sglang.test.test_utils import CustomTestCase


def _make_tp_group(active_ranks_cpu, active_ranks_gpu=None):
    """Build a minimal `tp_group` stand-in exposing `active_ranks*` fields.

    When `active_ranks_gpu` is None, we deliberately set the GPU field to a
    tensor that would raise on `.all()` if it were consulted; the whole point
    of this test suite is to prove the CPU-only fast path never touches the
    GPU tensor.
    """
    tp_group = SimpleNamespace()
    tp_group.active_ranks_cpu = torch.tensor(active_ranks_cpu, dtype=torch.int32)
    if active_ranks_gpu is None:
        tp_group.active_ranks = _ExplodingTensor()
    else:
        tp_group.active_ranks = torch.tensor(active_ranks_gpu, dtype=torch.int32)
    return tp_group


class _ExplodingTensor:
    """Sentinel that raises if any attribute is accessed.

    Used to prove the fast path never consults the GPU tensor.
    """

    def __getattr__(self, name):
        raise AssertionError(
            f"Fast path must not consult `active_ranks` (accessed .{name})"
        )


class TestElasticEpForwardFastPath(CustomTestCase):
    """V1 semantic equivalence tests for the fast-path branch."""

    def _call(self, tp_group):
        return maybe_recover_ep_ranks(
            tp_group=tp_group,
            eplb_manager=MagicMock(),
            model_config=MagicMock(),
            moe_ep_rank=0,
        )

    def test_fast_path_returns_false_when_cpu_all_active(self):
        """All CPU-side bits set => fast path returns False without touching GPU."""
        tp_group = _make_tp_group(active_ranks_cpu=[1, 1, 1, 1])
        # If the impl consults `active_ranks.all()`, `_ExplodingTensor` raises.
        self.assertFalse(self._call(tp_group))

    def test_fast_path_returns_false_for_single_rank_group(self):
        """Degenerate 1-rank tp group still routes through the fast path."""
        tp_group = _make_tp_group(active_ranks_cpu=[1])
        self.assertFalse(self._call(tp_group))

    def test_slow_path_entered_when_cpu_has_zero(self):
        """Any zero in `active_ranks_cpu` must exit the fast path."""
        tp_group = _make_tp_group(
            active_ranks_cpu=[1, 1, 0, 1],
            active_ranks_gpu=[1, 1, 0, 1],
        )
        with patch.object(
            elastic_ep_module, "try_recover_ranks", return_value=False
        ) as mock_recover:
            result = self._call(tp_group)
        self.assertFalse(result)  # `try_recover_ranks` returned False
        mock_recover.assert_called_once_with([2])

    def test_slow_path_computes_ranks_to_recover_from_cpu_mirror(self):
        """`ranks_to_recover` is derived from the AND of both tensors.

        Because CPU is the lock-step mirror, both tensors normally agree; the
        AND is a defensive guard. This test asserts that with matching zeros
        in both, the recover list contains those indices.
        """
        tp_group = _make_tp_group(
            active_ranks_cpu=[1, 0, 1, 0],
            active_ranks_gpu=[1, 0, 1, 0],
        )
        with (
            patch.object(
                elastic_ep_module, "try_recover_ranks", return_value=True
            ) as mock_recover,
            patch.object(
                elastic_ep_module, "broadcast_global_expert_location_metadata"
            ),
            patch.object(
                elastic_ep_module,
                "get_healthy_expert_location_src_rank",
                return_value=0,
            ),
            patch.object(
                elastic_ep_module.ElasticEPStateManager,
                "instance",
                return_value=MagicMock(),
            ),
        ):
            eplb_manager = MagicMock()
            result = maybe_recover_ep_ranks(
                tp_group=tp_group,
                eplb_manager=eplb_manager,
                model_config=MagicMock(),
                moe_ep_rank=0,
            )
        self.assertTrue(result)
        mock_recover.assert_called_once_with([1, 3])
        eplb_manager.reset_generator.assert_called_once()

    def test_all_cpu_zero_still_enters_slow_path(self):
        """Corner case: all-zero CPU tensor triggers recovery attempt."""
        tp_group = _make_tp_group(
            active_ranks_cpu=[0, 0, 0, 0],
            active_ranks_gpu=[0, 0, 0, 0],
        )
        with patch.object(
            elastic_ep_module, "try_recover_ranks", return_value=False
        ) as mock_recover:
            result = self._call(tp_group)
        self.assertFalse(result)
        mock_recover.assert_called_once_with([0, 1, 2, 3])


class TestElasticEpForwardFastPathGpu(CustomTestCase):
    """V2 (light) verification that on CUDA the fast path completes without
    depending on GPU-side state.

    A strict microbenchmark (which measures the actual host-device sync
    savings) lives in `test/manual/ep/bench_elastic_ep_forward_fast_path.py`.
    This test only verifies that the fast path returns the correct value on
    a CUDA-backed tp_group while background work is queued on the compute
    stream, i.e. that no GPU-side collective is issued.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fast_path_returns_false_on_cuda_with_pending_work(self):
        device = torch.device("cuda")
        tp_group = SimpleNamespace(
            active_ranks=torch.ones(8, dtype=torch.int32, device=device),
            active_ranks_cpu=torch.ones(8, dtype=torch.int32),
        )

        # Queue some work on the current stream. The point is that the fast
        # path should not consult `tp_group.active_ranks`, so the queued
        # work does not have to complete before the check returns.
        big = torch.randn(4096, 4096, device=device)
        for _ in range(64):
            big = big @ big

        result = maybe_recover_ep_ranks(
            tp_group=tp_group,
            eplb_manager=MagicMock(),
            model_config=MagicMock(),
            moe_ep_rank=0,
        )
        self.assertFalse(result)
        # Leave the test environment clean.
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
