"""Waiting out Mooncake's membership view on the mlp_sync gather after a bare fault.
The refusal precedes posting the collective, which is what makes waiting safe."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.elastic_ep import elastic_ep
from sglang.srt.elastic_ep.elastic_ep import mooncake_all_gather_settling
from sglang.test.test_utils import CustomTestCase

INACTIVE = "mooncakePgAllGather failed: invalid state: rank is not active in this group"


class _Gather:
    """Stands in for ``all_gather_single``, refusing the first ``refusals`` calls."""

    def __init__(self, refusals, exc=None):
        self.refusals = refusals
        self.calls = 0
        self.exc = exc or RuntimeError(INACTIVE)

    def __call__(self, output, input_, *, group):
        self.calls += 1
        if self.calls <= self.refusals:
            raise self.exc


def _run(gather):
    mooncake_all_gather_settling(None, None, group=None, gather=gather)


class TestGatherSettle(CustomTestCase):
    def test_the_gather_lands_once_the_view_settles(self):
        for refusals in (0, 3):
            with self.subTest(refusals=refusals):
                gather = _Gather(refusals)
                _run(gather)
                self.assertEqual(gather.calls, refusals + 1)

    def test_an_unrelated_runtime_error_is_not_retried(self):
        """Only the inactive refusal is ours to wait on; everything else propagates."""
        gather = _Gather(1, exc=RuntimeError("CUDA error: out of memory"))
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            _run(gather)
        self.assertEqual(gather.calls, 1)

    def test_a_view_that_never_settles_still_fails(self):
        """Loudly, and naming the condition rather than passing Mooncake's text up."""
        with patch.object(elastic_ep, "_GATHER_SETTLE_BUDGET_S", 0.2):
            with self.assertRaisesRegex(
                RuntimeError, "membership view never"
            ) as caught:
                _run(_Gather(10**9))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIn("not active", str(caught.exception.__cause__))


class TestMlpSyncPostsOneCollectiveOnMooncake(CustomTestCase):
    """The retry is safe for one collective only, so Mooncake keeps one all-gather."""

    def _posted(self, on_mooncake):
        import torch

        from sglang.srt.managers.scheduler_components import dp_attn
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        info = dp_attn.MLPSyncBatchInfo(
            num_dp_ranks=2,
            tp_size=1,
            cp_size=1,
            num_tokens=4,
            num_tokens_for_logprob=1,
            can_run_decode_cuda_graph=True,
            can_run_draft_cuda_graph=True,
            can_run_prefill_cuda_graph=True,
            is_extend_in_batch=False,
            local_can_run_tbo=False,
            local_forward_mode=ForwardMode.DECODE.value,
        )
        local = info._get_local_tensor(device="cpu")
        posted = []

        def gather_as(name):
            def gather(output, input_, *, group):
                posted.append(name)
                output.copy_(torch.cat([local, local]))

            return gather

        parallel = SimpleNamespace(
            tp_group=SimpleNamespace(active_ranks_cpu=torch.ones(2, dtype=torch.int64))
        )
        with (
            patch.object(elastic_ep, "is_mooncake_pg", return_value=on_mooncake),
            patch.object(dp_attn, "get_parallel", return_value=parallel),
            patch.object(dp_attn, "all_gather_single", gather_as("all_gather")),
            patch.object(dp_attn, "exchange_sync_rows", gather_as("exchange")),
        ):
            info.all_gather(device="cpu", group=object())
        return posted

    def test_mooncake_posts_the_single_all_gather_other_groups_the_exchange(self):
        self.assertEqual(self._posted(on_mooncake=True), ["all_gather"])
        self.assertEqual(self._posted(on_mooncake=False), ["exchange"])


if __name__ == "__main__":
    unittest.main()
