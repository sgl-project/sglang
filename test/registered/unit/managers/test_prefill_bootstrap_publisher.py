"""Check bounded publication and the actual DP dispatch hook."""

import asyncio
import json
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.srt.managers.data_parallel_controller import DataParallelController
from sglang.srt.managers.prefill_bootstrap_publisher import PrefillBootstrapPublisher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillBootstrapPublisher(CustomTestCase):
    def publisher(self):
        with patch(
            "sglang.srt.managers.prefill_bootstrap_publisher.ThreadPoolExecutor"
        ):
            return PrefillBootstrapPublisher(12345, max_pending=1)

    def test_queue_bound_and_release_after_completion(self):
        publisher = self.publisher()
        future = Future()
        publisher._executor.submit.return_value = future
        self.assertTrue(publisher.publish(11, 2))
        self.assertFalse(publisher.publish(12, 3))
        future.set_result(None)
        self.assertTrue(publisher.publish(12, 3))

    def test_submit_failure_releases_slot(self):
        publisher = self.publisher()
        publisher._executor.submit.side_effect = RuntimeError("shutdown")
        self.assertFalse(publisher.publish(11, 2))
        self.assertFalse(publisher.publish(12, 3))
        self.assertEqual(publisher._executor.submit.call_count, 2)

    def test_missing_or_invalid_metadata_is_ignored(self):
        publisher = self.publisher()
        for room, rank in ((None, 0), (1, -1), (True, 0), (1, None)):
            self.assertFalse(publisher.publish(room, rank))
        publisher._executor.submit.assert_not_called()

    def test_payload_and_connection_cleanup(self):
        publisher = self.publisher()
        with patch(
            "sglang.srt.managers.prefill_bootstrap_publisher.http.client.HTTPConnection"
        ) as connection:
            connection.return_value.getresponse.return_value.status = 200
            publisher._post(11, 2)
            args = connection.return_value.request.call_args.args
            self.assertEqual(args[:2], ("POST", "/register_dp_rank"))
            self.assertEqual(
                json.loads(args[2]),
                {"bootstrap_room": 11, "dp_rank": 2, "advisory": True},
            )
            connection.return_value.close.assert_called_once()
            connection.return_value.request.side_effect = OSError("offline")
            publisher._post(11, 2)
            self.assertEqual(connection.return_value.close.call_count, 2)

    def test_selected_dispatch_is_published_after_send(self):
        controller = DataParallelController.__new__(DataParallelController)
        controller.workers = [Mock()]
        controller.prefill_bootstrap_publisher = Mock()
        request = SimpleNamespace(bootstrap_room=11)
        with patch("sglang.srt.managers.data_parallel_controller.sock_send") as send:
            controller._send_selected_worker(0, request)
            send.assert_called_once_with(controller.workers[0], request)
        controller.prefill_bootstrap_publisher.publish.assert_called_once_with(11, 0)
        controller.prefill_bootstrap_publisher.reset_mock()
        with patch(
            "sglang.srt.managers.data_parallel_controller.sock_send",
            side_effect=OSError,
        ):
            with self.assertRaises(OSError):
                controller._send_selected_worker(0, request)
        controller.prefill_bootstrap_publisher.publish.assert_not_called()

    def test_authoritative_registration_wins_in_either_arrival_order(self):
        async def register(server, rank, advisory=False):
            request = SimpleNamespace(
                json=AsyncMock(
                    return_value={
                        "bootstrap_room": 11,
                        "dp_rank": rank,
                        "advisory": advisory,
                    }
                )
            )
            return await server._handle_register_dp_rank(request)

        async def check():
            for hint_first in (True, False):
                with self.subTest(hint_first=hint_first):
                    server = CommonKVBootstrapServer.__new__(CommonKVBootstrapServer)
                    server.lock = asyncio.Lock()
                    server.room_to_dp_rank = {}
                    if hint_first:
                        await register(server, 1, advisory=True)
                        self.assertEqual(server.room_to_dp_rank[11]["dp_rank"], 1)
                    await register(server, 2)
                    authoritative = server.room_to_dp_rank[11].copy()
                    await register(server, 1, advisory=True)
                    self.assertEqual(server.room_to_dp_rank[11], authoritative)
                    self.assertEqual(authoritative["dp_rank"], 2)

        asyncio.run(check())

    def test_cancelled_publication_releases_queue_slot(self):
        publisher = self.publisher()
        future = Future()
        publisher._executor.submit.return_value = future
        self.assertTrue(publisher.publish(11, 2))
        future.cancel()
        self.assertTrue(publisher.publish(12, 3))


if __name__ == "__main__":
    unittest.main()
