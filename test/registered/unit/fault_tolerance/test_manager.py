import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.fault_tolerance.ft_state import FaultToleranceState
from sglang.srt.fault_tolerance.manager import FaultToleranceManager
from sglang.srt.fault_tolerance.protocol import parse_apply_request
from sglang.srt.managers.io_struct import ActiveRanksOutput, ProcessActiveRanksOutput
from sglang.srt.managers.scheduler import Scheduler


def make_manager(*, dp_size=2, ranks_per_dp=1, strategy="pause"):
    return FaultToleranceManager(
        server_args=SimpleNamespace(
            dp_size=dp_size,
            tp_size=dp_size * ranks_per_dp,
            fault_tolerance_on_error_strategy=strategy,
            fault_tolerance_timeout=1,
        ),
        zmq_context=Mock(),
        send_to_scheduler=AsyncMock(),
    )


class TestFaultTolerance(unittest.IsolatedAsyncioTestCase):
    def test_protocol_and_state_contract(self):
        request = parse_apply_request(
            b'{"instruction":"scale_down","params":{"removed_dp_ranks":[1]}}'
        )
        self.assertEqual(request.params.removed_dp_ranks, [1])
        with self.assertRaisesRegex(ValueError, "Invalid instruction"):
            parse_apply_request(b'{"instruction":"recover"}')

        state = FaultToleranceState(dp_size=2, strategy="pause", global_rank_count=4)
        state.observe_process_active_ranks([2], active=False)
        self.assertEqual(state.process_alive_dp_mask(), [True, False])
        self.assertEqual(state.status_response()["engines"][1]["status"], "dead")
        self.assertEqual(
            state.expand_dp_mask_to_global_rank_mask([True, False]),
            [True, True, False, False],
        )

        manager = make_manager()
        manager._finish_submitted_apply("request-1", None)
        status = manager.status()[1]
        self.assertEqual(status["last_ft_request_id"], "request-1")
        self.assertNotIn("ft_error", status)
        self.assertNotIn("last_ft_request_id", status["engines"][0])
        manager._finish_submitted_apply("request-2", "failed")
        self.assertEqual(manager.status()[1]["ft_error"], "failed")

    async def test_retry_uses_expected_topology(self):
        manager = make_manager(dp_size=4)
        manager.state.expected_dp_mask = [True, True, False, True]
        manager._send_command_collect = AsyncMock()
        manager._publish_route_dp_mask = AsyncMock()

        self.assertIsNone(await manager._apply_retry(1))
        manager._send_command_collect.assert_awaited_once_with(
            command="retry", target_ranks=[0, 1, 3], timeout_sec=1
        )
        manager._publish_route_dp_mask.assert_awaited_once_with(
            [True, True, False, True], 1
        )

    async def test_scale_down_orders_shutdown_command_and_route(self):
        manager = make_manager(dp_size=4, ranks_per_dp=2)
        events = []
        manager._shutdown_dp_processes = AsyncMock(
            side_effect=lambda *_: events.append("shutdown")
        )
        manager._send_command_collect = AsyncMock(
            side_effect=lambda **_: events.append("command")
        )
        manager._publish_route_dp_mask = AsyncMock(
            side_effect=lambda *_: events.append("route")
        )

        self.assertIsNone(await manager._apply_scale_down([2], 1))
        self.assertEqual(events, ["shutdown", "command", "route"])
        manager._send_command_collect.assert_awaited_once_with(
            command="scale_down",
            target_ranks=[0, 1, 3],
            timeout_sec=1,
            active_global_rank_mask=[
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
            ],
        )
        self.assertEqual(manager.state.expected_dp_mask, [True, True, False, True])

    async def test_continue_routes_only_ready_processes(self):
        manager = make_manager(dp_size=2, ranks_per_dp=2, strategy="continue")

        down = manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2, 3], active=False)
        )
        self.assertEqual(down.status, [True, False])
        self.assertEqual(manager.state.expected_dp_mask, [True, True])

        manager.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=[2, 3], active=True)
        )
        up = manager.observe_active_ranks(ActiveRanksOutput(status=[True, True]))
        self.assertEqual(up.status, [True, True])
        manager.send_to_scheduler.assert_not_awaited()

    def test_abort_precedes_resource_cleanup(self):
        req = Mock(
            rid="request-1",
            kv=SimpleNamespace(holds_kv=True, holds_mamba=False, kv_committed_len=3),
            origin_input_ids=[1, 2],
            output_ids=[3],
            finished_reason=None,
        )
        req.finished.side_effect = lambda: req.finished_reason is not None
        batch = SimpleNamespace(reqs=[req])
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.cur_batch_for_debug = batch
        scheduler.last_batch = batch
        scheduler.running_batch = batch
        scheduler.chunked_req = req
        failed_result_queue = deque([(batch, object())])
        scheduler._ft_result_queue = failed_result_queue
        scheduler.result_queue = deque()
        scheduler.tree_cache = Mock()
        scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=Mock())

        with patch(
            "sglang.srt.managers.scheduler.release_kv_cache"
        ) as release_kv_cache:
            self.assertTrue(scheduler._ft_abort_inflight_window())
            release_kv_cache.assert_not_called()
            self.assertIs(scheduler.running_batch, batch)
            self.assertTrue(failed_result_queue)

            self.assertTrue(scheduler._ft_discard_inflight_window())
            release_kv_cache.assert_called_once_with(
                req,
                scheduler.tree_cache,
                is_insert=False,
                allow_non_spec_overallocated=True,
            )

        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()
        self.assertFalse(scheduler.running_batch.reqs)
        self.assertFalse(failed_result_queue)
        self.assertIsNone(scheduler._ft_result_queue)
        self.assertIsNone(scheduler.chunked_req)

    def test_finished_requests_without_resources_are_not_discarded(self):
        req = SimpleNamespace(
            rid="finished",
            kv=SimpleNamespace(holds_kv=False, holds_mamba=False),
            finished=lambda: True,
        )
        batch = SimpleNamespace(reqs=[req])
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.cur_batch_for_debug = batch
        scheduler.last_batch = batch
        scheduler.running_batch = batch
        scheduler.chunked_req = None
        scheduler._ft_result_queue = None

        self.assertEqual(scheduler._ft_inflight_reqs(), {})
        req.kv.holds_mamba = True
        self.assertEqual(scheduler._ft_inflight_reqs(), {req.rid: req})

    def test_recovery_precedes_discard(self):
        events = []
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = SimpleNamespace(dp_rank=0, attn_tp_rank=0, attn_cp_rank=0)
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                update_fault_tolerance_active_ranks=lambda _: events.append("recover")
            )
        )
        scheduler._ft_discard_inflight_window = lambda: events.append("discard") or True
        scheduler._engine_paused = True
        scheduler._ft_pause_deadline = 1

        scheduler.handle_fault_tolerance_command(
            SimpleNamespace(command="retry", target_ranks=[0], request_id="retry-1")
        )

        self.assertEqual(events, ["recover", "discard"])
        self.assertFalse(scheduler._engine_paused)


if __name__ == "__main__":
    unittest.main()
