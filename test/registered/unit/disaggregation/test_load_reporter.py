"""Production Reporter streams, renewal, bounded freshness and target isolation."""

import asyncio
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import grpc
from sglang.srt.disaggregation.load_report_pb2 import LoadReportAck, LoadSample
from sglang.srt.disaggregation.load_reporter import LoadReporter


class TestLoadReporter(unittest.TestCase):
    def test_http_registration_errors_return_valid_responses(self):
        from sglang.srt.entrypoints import http_server

        request = SimpleNamespace(json=AsyncMock(return_value={}))
        state = SimpleNamespace(tokenizer_manager=SimpleNamespace(server_args=object()))
        for error, status in (
            (ValueError("bad target"), 400),
            (RuntimeError("offline"), 503),
        ):
            with (
                patch.object(http_server, "_global_state", state),
                patch(
                    "sglang.srt.disaggregation.load_reporter.register_worker_reporting",
                    AsyncMock(side_effect=error),
                ),
            ):
                response = asyncio.run(http_server.start_reporting(request))
                self.assertEqual(response.status_code, status)

    def test_independent_targets_epoch_samples_and_lease_expiry(self):
        received = [[], []]
        servers = []
        reporter = LoadReporter("ns", "worker", 2, "publisher-epoch")
        stopped = threading.Event()
        try:
            for bucket in received:

                def report(samples, context, bucket=bucket):
                    for sample in samples:
                        bucket.append(sample)
                    return LoadReportAck()

                server = grpc.server(ThreadPoolExecutor(max_workers=2))
                server.add_generic_rpc_handlers(
                    [
                        grpc.method_handlers_generic_handler(
                            "kv_load.v1.LoadMonitor",
                            {
                                "Report": grpc.stream_unary_rpc_method_handler(
                                    report,
                                    request_deserializer=LoadSample.FromString,
                                    response_serializer=LoadReportAck.SerializeToString,
                                )
                            },
                        )
                    ]
                )
                port = server.add_insecure_port("127.0.0.1:0")
                server.start()
                servers.append((server, f"http://127.0.0.1:{port}"))
            for _, target in servers:
                identity = reporter.register(target, "token", 1)
                self.assertEqual(identity["worker_epoch"], "publisher-epoch")

            def measurements():
                while not stopped.wait(0.05):
                    reporter.update(
                        SimpleNamespace(num_running_reqs=3, num_waiting_reqs=7)
                    )

            thread = threading.Thread(target=measurements)
            thread.start()
            deadline = time.monotonic() + 3
            while min(map(len, received)) < 2 and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(all(len(samples) >= 2 for samples in received))
            reporter.register(servers[1][1], "token", 3)
            time.sleep(1.3)
            old = len(received[0])
            other = len(received[1])
            time.sleep(0.3)
            self.assertEqual(len(received[0]), old)
            self.assertGreater(len(received[1]), other)
            for bucket in received:
                self.assertTrue(
                    all(
                        s.worker_epoch == "publisher-epoch" and s.dp_rank == 2
                        for s in bucket
                    )
                )
                self.assertTrue(
                    all(
                        s.num_running_reqs == 3 and s.num_waiting_reqs == 7
                        for s in bucket
                    )
                )
                self.assertEqual(
                    sorted({s.sequence for s in bucket}), [s.sequence for s in bucket]
                )
            stopped.set()
            thread.join()
            time.sleep(0.3)
            count = len(received[1])
            time.sleep(1.1)
            self.assertEqual(
                len(received[1]),
                count,
                "stalled scheduler samples must not be refreshed",
            )
        finally:
            stopped.set()
            reporter.close()
            for server, _ in servers:
                server.stop(0).wait()

    def test_rejects_invalid_registration_before_starting_stream(self):
        reporter = LoadReporter("ns", "worker", 0, "epoch")
        try:
            for target, token, lease in [
                ("file:///tmp/x", "token", 6),
                ("http://host:1", "", 6),
                ("http://host:1", "token", 1000),
            ]:
                with self.assertRaises(ValueError):
                    reporter.register(target, token, lease)
            self.assertEqual(reporter.targets, {})
        finally:
            reporter.close()


if __name__ == "__main__":
    unittest.main()
