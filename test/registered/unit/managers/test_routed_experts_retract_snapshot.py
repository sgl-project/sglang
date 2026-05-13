import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _make_req(prefix=None, snap_seqlen=0, seqlen=None, req_pool_idx=0):
    req = MagicMock()
    req._retract_routed_experts_prefix = prefix
    req._retract_snapshot_seqlen = snap_seqlen
    req.req_pool_idx = req_pool_idx
    req.seqlen = seqlen if seqlen is not None else snap_seqlen
    return req


class TestMaybeCollectRoutedExpertsWithRetract(unittest.TestCase):
    def _call(self, req, suffix):
        scheduler = MagicMock()
        scheduler.req_to_token_pool = MagicMock()
        capturer = MagicMock()
        capturer.get_routed_experts.return_value = suffix
        with patch(
            "sglang.srt.managers.scheduler_output_processor_mixin."
            "get_global_experts_capturer",
            return_value=capturer,
        ):
            SchedulerOutputProcessorMixin.maybe_collect_routed_experts(scheduler, req)
        return req.routed_experts

    def test_no_prefix_returns_suffix(self):
        suffix = torch.arange(12).reshape(4, 3)
        req = _make_req(prefix=None, snap_seqlen=0, seqlen=5)
        result = self._call(req, suffix)
        self.assertTrue(torch.equal(result, suffix))

    def test_prefix_only_when_suffix_none(self):
        prefix = torch.arange(6).reshape(2, 3)
        req = _make_req(prefix=prefix, snap_seqlen=3, seqlen=3)
        result = self._call(req, None)
        self.assertTrue(torch.equal(result, prefix))

    def test_stitches_prefix_and_sliced_suffix(self):
        prefix = torch.tensor([[0, 0, 0], [1, 1, 1]])
        suffix = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
            ]
        )
        # snapshot was taken at seqlen=3 → snap = 3 - 1 = 2; tail = suffix[2:]
        req = _make_req(prefix=prefix, snap_seqlen=3, seqlen=6)
        result = self._call(req, suffix)
        expected = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
            ]
        )
        self.assertTrue(torch.equal(result, expected))

    def test_snap_beyond_suffix_drops_tail(self):
        prefix = torch.tensor([[7, 7, 7], [8, 8, 8]])
        suffix = torch.tensor([[1, 1, 1]])
        # snap_seqlen - 1 = 4 ≥ suffix.shape[0]; tail must be empty
        req = _make_req(prefix=prefix, snap_seqlen=5, seqlen=6)
        result = self._call(req, suffix)
        self.assertTrue(torch.equal(result, prefix))

    def test_snap_zero_keeps_full_suffix(self):
        prefix = torch.tensor([[9, 9, 9]])
        suffix = torch.tensor([[1, 1, 1], [2, 2, 2]])
        # snap = max(0, 0 - 1) = 0; tail = suffix[0:] = full suffix
        req = _make_req(prefix=prefix, snap_seqlen=0, seqlen=3)
        result = self._call(req, suffix)
        expected = torch.cat([prefix, suffix], dim=0)
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
