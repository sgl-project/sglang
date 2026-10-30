"""PP consensus on the outcome of PD decode KV transfers.

Each PP stage of a disaggregated decode receives the KV of its own layers. A
prefill abort that lands after the first prefill stage finished sending fails
only the later decode stages, so the stages must agree not only on which
transfers concluded but also on whether each one failed, and a failure on one
stage must reach every stage down the release chain.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _stage(first_rank: bool, succeeded, failed, prev=None):
    """A PP stage whose own receivers report `succeeded` / `failed` rids."""
    return SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=first_rank),
        disagg_decode_transfer_queue=SimpleNamespace(queue=[]),
        get_rids=lambda queue, is_send, *groups: (list(succeeded), list(failed)),
        _pp_recv_pyobj_from_prev_stage=lambda: prev,
    )


def _consensus(stage):
    return SchedulerPPMixin._pp_pd_get_decode_transferred_ids(stage)


class TestPPDecodeTransferOutcomeConsensus(CustomTestCase):
    def test_first_stage_reports_its_own_outcomes(self):
        stage = _stage(first_rank=True, succeeded=["a"], failed=["b"])
        self.assertEqual(_consensus(stage), {"a": False, "b": True})

    def test_failure_on_any_stage_fails_the_transfer_on_every_stage(self):
        stage1 = _stage(
            first_rank=False,
            succeeded=["c"],
            failed=["r"],
            prev={"r": False, "c": False},
        )
        self.assertEqual(_consensus(stage1), {"r": True, "c": False})
        stage2 = _stage(first_rank=False, succeeded=["r"], failed=[], prev={"r": True})
        self.assertEqual(_consensus(stage2), {"r": True})

    def test_only_transfers_concluded_on_every_stage_are_released(self):
        stage = _stage(
            first_rank=False, succeeded=["a", "b"], failed=[], prev={"b": False}
        )
        self.assertEqual(_consensus(stage), {"b": False})

    def test_empty_previous_consensus_arrives_as_a_list(self):
        # point_to_point_pyobj delivers an empty payload as [].
        stage = _stage(first_rank=False, succeeded=["a"], failed=[], prev=[])
        self.assertEqual(_consensus(stage), {})


class _Receiver:
    def __init__(self, poll):
        self._poll = poll
        self.abort_notified = False
        self.cleared = False

    def poll(self):
        return self._poll

    def clear(self):
        self.cleared = True

    def failure_exception(self):
        raise KVTransferError(
            7, "Failed due to an unknown reason from another rank", True
        )


def _decode_req(rid, poll):
    return SimpleNamespace(
        req=SimpleNamespace(
            rid=rid, bootstrap_room=7, return_logprob=False, finished_reason=None
        ),
        kv_receiver=_Receiver(poll),
        metadata_buffer_index=1,
        hicache_restore_status=None,
        host_staged=False,
    )


def _transfer_queue(decode_reqs):
    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.queue = list(decode_reqs)
    queue.enable_staging = False
    queue.enable_host_receive = False
    queue.enable_deferred_kv_release = False
    queue.gloo_group = MagicMock()
    queue.req_to_metadata_buffer_idx_allocator = MagicMock()
    queue.tp_rank = 0
    queue.tree_cache = MagicMock()
    queue.metadata_buffers = MagicMock()
    queue._clean_hicache_prefetch_resources = MagicMock()
    queue._commit_transfer_to_req = MagicMock()
    queue._record_transfer_failure = MagicMock()
    queue._release_request = MagicMock()
    queue.scheduler = SimpleNamespace(
        enable_decode_hicache=False,
        enable_hisparse=False,
        metrics_reporter=SimpleNamespace(enable_metrics=False),
        output_streamer=SimpleNamespace(stream_output=MagicMock()),
    )
    return queue


class TestPopTransferredFollowsConsensusOutcome(CustomTestCase):
    def setUp(self):
        super().setUp()
        # The release collective sizes itself from the gloo group; answer as one rank.
        for target, kwargs in (
            ("torch.distributed.get_world_size", {"return_value": 1}),
            (
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                {
                    "side_effect": lambda pollers, group, **kw: [
                        p.poll() for p in pollers
                    ]
                },
            ),
            ("sglang.srt.disaggregation.decode.prepare_abort", {}),
            ("sglang.srt.disaggregation.decode.release_kv_cache", {}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_consensus_failure_overrides_a_local_success(self):
        admitted = _decode_req("s", KVPoll.Success)
        failed_elsewhere = _decode_req("f", KVPoll.Success)
        not_yet_agreed = _decode_req("h", KVPoll.Success)
        queue = _transfer_queue([admitted, failed_elsewhere, not_yet_agreed])

        reqs, outcomes = queue.pop_transferred_with_outcomes({"s": False, "f": True})

        self.assertEqual(reqs, [admitted.req])
        self.assertEqual(outcomes, {"s": False, "f": True})
        self.assertEqual(queue.queue, [not_yet_agreed])
        queue._commit_transfer_to_req.assert_called_once_with(admitted)
        queue._release_request.assert_called_once_with(failed_elsewhere)
        queue._record_transfer_failure.assert_called_once_with(
            "Failed due to an unknown reason from another rank"
        )

    def test_local_failure_is_reported_downstream(self):
        failed_here = _decode_req("f", KVPoll.Failed)
        queue = _transfer_queue([failed_here])

        reqs, outcomes = queue.pop_transferred_with_outcomes({"f": False})

        self.assertEqual(reqs, [])
        self.assertEqual(outcomes, {"f": True})
        self.assertEqual(queue.queue, [])

    def test_non_pp_path_keeps_the_request_list(self):
        admitted = _decode_req("s", KVPoll.Success)
        queue = _transfer_queue([admitted])
        self.assertEqual(queue.pop_transferred(), [admitted.req])


class TestProcessDecodeTransferQueueForwardsOutcomes(CustomTestCase):
    def test_forwards_every_concluded_outcome(self):
        admitted = SimpleNamespace(rid="s")
        scheduler = SimpleNamespace(
            disagg_decode_transfer_queue=SimpleNamespace(
                resolve_deferred_releases=MagicMock(),
                pop_transferred_with_outcomes=MagicMock(
                    return_value=([admitted], {"s": False, "f": True})
                ),
            ),
            enable_hisparse=False,
            waiting_queue=[],
        )
        forwarded = SchedulerPPMixin.process_decode_transfer_queue(
            scheduler, {"s": False, "f": True, "h": False}
        )
        self.assertEqual(forwarded, {"s": False, "f": True})
        self.assertEqual(scheduler.waiting_queue, [admitted])

    def test_no_consensus_means_nothing_to_forward(self):
        scheduler = SimpleNamespace(
            disagg_decode_transfer_queue=SimpleNamespace(
                resolve_deferred_releases=MagicMock(),
                pop_transferred_with_outcomes=MagicMock(),
            ),
            enable_hisparse=False,
            waiting_queue=[],
        )
        self.assertIsNone(
            SchedulerPPMixin.process_decode_transfer_queue(scheduler, None)
        )
        scheduler.disagg_decode_transfer_queue.pop_transferred_with_outcomes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
