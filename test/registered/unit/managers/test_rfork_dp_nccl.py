"""Regression tests for R-Fork NCCL weight transfer with dp_size > 1.

Two bugs are fixed together:

1. tokenizer_control_mixin.py — `assert dp_size == 1` in
   init_weights_send_group_for_remote_instance and send_weights_to_remote_instance
   blocks R-Fork on any seed with dp_size > 1, even though the communicators
   already use fan_out=dp_size by default (correct for NCCL).

2. data_parallel_controller.py — launch_dp_schedulers starts all DP replica
   threads in parallel and waits for all events afterwards. With
   load_format=remote_instance the seed's FanOutCommunicator is queueing
   (one in-flight request at a time), so all replicas hitting it
   simultaneously causes contention. Fix: start and wait for each replica
   sequentially when load_format=remote_instance, keep parallel for everything
   else.

Note: launch_dp_schedulers is only used when --enable-dp-attention=False.
The dp_attention path goes through launch_dp_attention_schedulers, which
launches a single tp world via mp.Process and is not affected by this patch.
"""

import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.data_parallel_controller import DataParallelController
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server_args(dp_size: int = 8, load_format: str = "remote_instance"):
    sa = MagicMock()
    sa.dp_size = dp_size
    sa.load_format = load_format
    sa.tp_size = 1
    sa.pp_size = 1
    sa.gpu_id_step = 1
    # node_rank != 0 skips the self.workers[dp_rank] = ... branch in
    # launch_dp_schedulers, which keeps the test free of zmq sockets.
    sa.node_rank = 1
    return sa


def _make_mixin_obj(dp_size: int):
    """Minimal stand-in for TokenizerManager that has just what the mixin needs."""
    obj = MagicMock()
    obj.server_args = MagicMock()
    obj.server_args.dp_size = dp_size
    obj.auto_create_handle_loop = MagicMock()

    fake_result = MagicMock()
    fake_result.success = True
    fake_result.message = "ok"

    communicator = AsyncMock(return_value=[fake_result])
    obj.init_weights_send_group_for_remote_instance_communicator = communicator
    obj.send_weights_to_remote_instance_communicator = communicator
    return obj


# ---------------------------------------------------------------------------
# Test 1 — tokenizer_control_mixin: dp_size==1 assertions removed
# ---------------------------------------------------------------------------


class TestTokenizerControlMixinDPAssertions(unittest.IsolatedAsyncioTestCase):
    """init_weights_send_group and send_weights must work with any dp_size."""

    async def test_init_weights_send_group_allows_dp_gt_1(self):
        """dp_size=8 must not raise — old code had assert dp_size == 1."""
        obj = _make_mixin_obj(dp_size=8)
        success, _ = (
            await TokenizerControlMixin.init_weights_send_group_for_remote_instance(
                obj, MagicMock()
            )
        )
        self.assertTrue(success)
        obj.auto_create_handle_loop.assert_called_once()

    async def test_send_weights_allows_dp_gt_1(self):
        """dp_size=8 must not raise — old code had assert dp_size == 1."""
        obj = _make_mixin_obj(dp_size=8)
        success, _ = await TokenizerControlMixin.send_weights_to_remote_instance(
            obj, MagicMock()
        )
        self.assertTrue(success)

    async def test_dp_size_1_still_works(self):
        """dp_size=1 must keep working — no regression for the common case."""
        obj = _make_mixin_obj(dp_size=1)
        success, _ = (
            await TokenizerControlMixin.init_weights_send_group_for_remote_instance(
                obj, MagicMock()
            )
        )
        self.assertTrue(success)
        success, _ = await TokenizerControlMixin.send_weights_to_remote_instance(
            obj, MagicMock()
        )
        self.assertTrue(success)


# ---------------------------------------------------------------------------
# Test 2 — data_parallel_controller: sequential start for remote_instance
# ---------------------------------------------------------------------------


class TestDataParallelControllerSequentialLoad(unittest.TestCase):
    """launch_dp_schedulers must start replicas one-by-one for remote_instance."""

    # 30 ms is long enough that any concurrent threads visibly overlap with
    # millisecond-scale scheduling jitter, but still keeps the whole suite under
    # ~1 s for dp_size=3.
    WORKER_LATENCY_S = 0.03

    def _run_launch(self, load_format: str, dp_size: int = 3):
        """Run launch_dp_schedulers with a fake worker that records timestamps.

        Returns a dict with:
          start_order, complete_order — dp_rank lists in observed order
          start_ts, complete_ts       — dict[dp_rank] -> monotonic timestamp
        """
        start_order = []
        complete_order = []
        start_ts = {}
        complete_ts = {}
        lock = threading.Lock()

        def fake_worker(server_args, port_args, base_gpu_id, dp_rank, ready_event):
            with lock:
                start_order.append(dp_rank)
                start_ts[dp_rank] = time.monotonic()
            time.sleep(self.WORKER_LATENCY_S)
            with lock:
                complete_order.append(dp_rank)
                complete_ts[dp_rank] = time.monotonic()
            ready_event.set()

        controller = DataParallelController.__new__(DataParallelController)
        controller.workers = [None] * dp_size
        controller.context = MagicMock()
        controller.launch_tensor_parallel_group_thread = fake_worker

        server_args = _make_server_args(dp_size=dp_size, load_format=load_format)

        mock_port_args = MagicMock()
        mock_port_args.nccl_port = 9999
        mock_port_args.tokenizer_ipc_name = "ipc://tok"
        mock_port_args.detokenizer_ipc_name = "ipc://detok"
        mock_port_args.scheduler_input_ipc_name = "ipc://sched"

        mock_socket = MagicMock()
        mock_socket.close = MagicMock()

        with patch(
            "sglang.srt.managers.data_parallel_controller.PortArgs.init_new",
            return_value=mock_port_args,
        ), patch(
            "sglang.srt.managers.data_parallel_controller.bind_port",
            return_value=mock_socket,
        ):
            controller.launch_dp_schedulers(server_args, mock_port_args)

        return {
            "start_order": start_order,
            "complete_order": complete_order,
            "start_ts": start_ts,
            "complete_ts": complete_ts,
        }

    def test_sequential_for_remote_instance(self):
        """Each replica must complete before the next one starts."""
        r = self._run_launch(load_format="remote_instance", dp_size=3)

        self.assertEqual(
            sorted(r["start_order"]), [0, 1, 2], "all dp_ranks started"
        )
        # Strict invariant: replica i finishes before replica i+1 starts.
        for i in range(2):
            self.assertLess(
                r["complete_ts"][i],
                r["start_ts"][i + 1],
                f"dp_rank {i} must complete before dp_rank {i + 1} starts",
            )

    def test_parallel_for_other_load_formats(self):
        """Non-remote_instance load formats keep the original parallel start.

        Strict invariant: every replica starts BEFORE the first replica
        completes — i.e. all start timestamps precede the earliest complete
        timestamp. With ``WORKER_LATENCY_S`` worker work this is observable
        even with scheduling jitter.
        """
        for fmt in ("auto", "safetensors", "runai_streamer"):
            with self.subTest(load_format=fmt):
                r = self._run_launch(load_format=fmt, dp_size=3)
                self.assertEqual(sorted(r["start_order"]), [0, 1, 2])

                earliest_complete = min(r["complete_ts"].values())
                latest_start = max(r["start_ts"].values())
                self.assertLess(
                    latest_start,
                    earliest_complete,
                    f"{fmt}: all replicas must start before any completes "
                    f"(latest_start={latest_start:.4f}, "
                    f"earliest_complete={earliest_complete:.4f})",
                )

    def test_sequential_all_replicas_loaded(self):
        """All dp_size replicas complete their load for remote_instance."""
        dp_size = 4
        r = self._run_launch(load_format="remote_instance", dp_size=dp_size)
        self.assertEqual(sorted(r["complete_order"]), list(range(dp_size)))


if __name__ == "__main__":
    unittest.main()
