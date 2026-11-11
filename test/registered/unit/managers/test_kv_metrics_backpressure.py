import unittest
from types import SimpleNamespace
from unittest.mock import patch

import zmq

from sglang.srt.managers.io_struct import sock_recv
from sglang.srt.managers.scheduler_components import kv_events_publisher as module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKvMetricsBackpressure(unittest.TestCase):
    def setUp(self):
        self.context = zmq.Context()
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.setsockopt(zmq.LINGER, 0)
        self.sender.setsockopt(zmq.SNDHWM, 1)
        # Bound failures on unpatched code; production uses an infinite timeout.
        self.sender.setsockopt(zmq.SNDTIMEO, 100)
        self.addCleanup(self.context.destroy, linger=0)
        with patch.object(module.SchedulerKvEventsPublisher, "init_kv_events"):
            self.publisher = module.SchedulerKvEventsPublisher(
                kv_events_config=None,
                tree_cache=None,
                send_metrics_from_scheduler=self.sender,
                max_running_requests=16,
                max_total_num_tokens=1024,
                get_stats=lambda: SimpleNamespace(
                    num_running_reqs=SimpleNamespace(total=2),
                    num_queue_reqs=SimpleNamespace(total=3),
                    token_usage=0.25,
                    cache_hit_rate=0.5,
                ),
                enable_kv_cache_events=True,
            )
        parallel = patch.object(
            module, "get_parallel", return_value=SimpleNamespace(dp_rank=2)
        )
        parallel.start()
        self.addCleanup(parallel.stop)

    def receiver(self):
        receiver = self.context.socket(zmq.PULL)
        receiver.setsockopt(zmq.RCVHWM, 1)
        receiver.setsockopt(zmq.RCVTIMEO, 1000)
        receiver.bind("inproc://kv-metrics-test")
        self.sender.connect("inproc://kv-metrics-test")
        return receiver

    def test_unbound_receiver_drops_and_warns_once(self):
        self.sender.connect("inproc://missing-receiver")
        # Exhaust even the queue allocated for a not-yet-bound inproc peer.
        for _ in range(4096):
            try:
                self.sender.send(b"queued", zmq.NOBLOCK)
            except zmq.Again:
                break
        else:
            self.fail("Socket did not apply its high-water mark")
        with patch.object(module.logger, "warning") as warning:
            self.publisher.emit_kv_metrics()
            self.publisher.emit_kv_metrics()
        warning.assert_called_once()

    def test_full_queue_recovers_when_receiver_drains(self):
        receiver = self.receiver()
        queued = 0
        for _ in range(4096):
            try:
                self.sender.send(b"queued", zmq.NOBLOCK)
                queued += 1
            except zmq.Again:
                break
        else:
            self.fail("Socket did not apply its high-water mark")
        with patch.object(module.logger, "warning") as warning:
            self.publisher.emit_kv_metrics()
        warning.assert_called_once()
        for _ in range(queued):
            self.assertEqual(receiver.recv(), b"queued")
        self.assertTrue(self.sender.poll(1000, zmq.POLLOUT))
        self.publisher.emit_kv_metrics()
        metrics = sock_recv(receiver)
        self.assertIsInstance(metrics, module.KvMetrics)
        self.assertEqual(metrics.request_active_slots, 2)
        self.assertEqual(metrics.kv_active_blocks, 256)
        self.assertEqual(metrics.data_parallel_rank, 2)

    def test_closed_socket_is_ignored(self):
        self.sender.close()
        self.publisher.emit_kv_metrics()

    def test_other_socket_errors_are_not_hidden(self):
        with patch.object(module, "sock_send", side_effect=zmq.ZMQError(zmq.EINVAL)):
            with self.assertRaises(zmq.ZMQError):
                self.publisher.emit_kv_metrics()


if __name__ == "__main__":
    unittest.main()
