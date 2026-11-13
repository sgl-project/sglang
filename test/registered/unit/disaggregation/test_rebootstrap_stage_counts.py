"""Rebootstrap accounting follows the real scheduler queue representations."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestRebootstrapStageCounts(CustomTestCase):
    def test_each_stage_is_counted_once_until_completion(self):
        owner = DecodePreallocQueue.__new__(DecodePreallocQueue)
        owner.held_rebootstrap_reqs = IndexedReqList(ListKind.PD_HELD)
        owner.pending_reqs = []
        owner.queue = IndexedReqList(ListKind.PD_PREALLOC, req_of=lambda dr: dr.req)
        transfer = IndexedReqList(ListKind.PD_RECEIVING, req_of=lambda dr: dr.req)
        owner.transfer_queue = SimpleNamespace(queue=transfer)
        request = object()
        dr = SimpleNamespace(req=request, is_rebootstrap=True)
        ordinary = SimpleNamespace(req=object(), is_rebootstrap=False)
        owner.queue.join([ordinary])
        owner.held_rebootstrap_reqs.join([request])
        self.assertEqual(owner.num_rebootstrap_in_flight, 1)
        owner.held_rebootstrap_reqs.clear()
        owner.pending_reqs.append(dr)
        self.assertEqual(owner.num_rebootstrap_in_flight, 1)
        owner.pending_reqs.clear()
        owner.queue.join([dr])
        self.assertEqual(owner.num_rebootstrap_in_flight, 1)
        owner.queue.leave([request])
        transfer.join([dr])
        self.assertEqual(owner.num_rebootstrap_in_flight, 1)
        transfer.leave([request])
        self.assertEqual(owner.num_rebootstrap_in_flight, 0)

    def test_failed_rebootstrap_logs_the_request_identity(self):
        from unittest.mock import Mock, patch

        owner = DecodeTransferQueue.__new__(DecodeTransferQueue)
        owner._is_host_destination = Mock(return_value=False)
        request = SimpleNamespace(
            rid="retry", bootstrap_room=17, disagg_prefill_dp_rank=2
        )
        with patch("sglang.srt.disaggregation.decode.logger.info") as log:
            owner._abort_failed_transfer(
                SimpleNamespace(req=request, is_rebootstrap=True)
            )
        self.assertEqual(log.call_args.args[1:], ("retry", 17, 2))


if __name__ == "__main__":
    unittest.main()
