"""Exercise the real Rust gRPC handler with a non-terminating Python engine owner."""

import asyncio
import threading
import unittest

import grpc

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.managers.tokenizer_manager import (
    ServerStatus,
    SignalHandler,
    TokenizerManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.grpc_shutdown_helpers import (
    load_test_grpc_extension,
    runtime_client,
    runtime_proto,
)
from sglang.test.test_utils import find_available_port

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestRuntimeShutdownRpc(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        try:
            extension = load_test_grpc_extension()
        except ModuleNotFoundError as error:
            self.skipTest(f"requires the built sglang-grpc extension: {error}")
        self.proto, service = runtime_proto()
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager._engine_state_changed_callback = None
        self.manager._gracefully_exit = False
        self.manager._server_status = ServerStatus.Up
        self.manager._is_pause = False
        self.manager.signal_handler_class = SignalHandler
        self.manager.rid_to_state = {"in-flight": object()}
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = self.manager
        handle._event_loop = asyncio.get_running_loop()
        handle.get_model_info = lambda: "{}"
        handle.get_server_info = lambda: "{}"
        self.handle = handle
        port = find_available_port(41000)
        self.server = extension.start_server(
            "127.0.0.1", port, handle, worker_threads=1
        )
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        self.client = runtime_client(self.channel, service)
        self.addAsyncCleanup(self._close_server)
        await asyncio.wait_for(self.channel.channel_ready(), 5)

    async def _close_server(self):
        if hasattr(self, "channel"):
            await self.channel.close()
            await asyncio.to_thread(self.server.shutdown)

    async def test_concurrent_acceptance_runs_existing_sigterm_path(self):
        watch = self.client.WatchEngineState(
            self.proto.WatchEngineStateRequest(), timeout=10
        )
        initial = await watch.read()
        self.assertTrue(initial.healthy)
        replies = await asyncio.gather(
            *(
                self.client.Shutdown(self.proto.ShutdownRequest(), timeout=2)
                for _ in range(8)
            )
        )
        self.assertTrue(all(reply.SerializeToString() == b"" for reply in replies))
        while True:
            snapshot = await watch.read()
            self.assertNotEqual(snapshot, grpc.aio.EOF)
            if not snapshot.healthy:
                self.assertGreater(snapshot.revision, initial.revision)
                break
        self.assertTrue(self.manager.gracefully_exit)
        self.assertIn("in-flight", self.manager.rid_to_state)
        self.assertFalse(
            (await self.client.HealthCheck(self.proto.HealthCheckRequest())).healthy
        )
        # Leave the watch open: explicit stream teardown must release the server.
        await asyncio.wait_for(asyncio.to_thread(self.server.shutdown), 5)
        while await watch.read() != grpc.aio.EOF:
            pass

    async def test_listener_shutdown_allows_pending_python_snapshot_to_finish(self):
        entered = threading.Event()
        release = threading.Event()
        health_check = self.handle.health_check

        def pending_health_check():
            entered.set()
            release.wait(timeout=2)
            return health_check()

        self.handle.health_check = pending_health_check
        self.manager.is_pause = True
        self.assertTrue(await asyncio.to_thread(entered.wait, 2))
        timer = threading.Timer(0.1, release.set)
        timer.start()
        try:
            # Closing the listener joins its runtime, including the pending
            # spawn_blocking snapshot. That join must release the Python GIL.
            await asyncio.wait_for(asyncio.to_thread(self.server.shutdown), 3)
        finally:
            release.set()
            timer.join()


if __name__ == "__main__":
    unittest.main(verbosity=2)
