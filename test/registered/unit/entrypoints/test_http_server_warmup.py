import asyncio
import multiprocessing
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import _send_disaggregation_warmup_requests
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestDisaggregationServerWarmup(unittest.IsolatedAsyncioTestCase):
    async def test_sends_concurrent_scalar_request_to_each_dp_rank(self):
        from sglang.srt.runtime_context import get_context

        # The warmup fan-out width comes from the published topology.
        override = get_context().override_server_args(dp_size=4)
        override.install()
        self.addCleanup(override.restore)
        server_args = SimpleNamespace(dp_size=4)
        all_started = asyncio.Event()
        calls = []
        sessions = []

        class Response:
            status = 200

            async def __aenter__(self):
                if len(calls) == server_args.dp_size:
                    all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=5)
                return self

            async def __aexit__(self, *args):
                pass

            async def read(self):
                return b""

        class Session:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                sessions.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, *args, **kwargs):
                calls.append((args, kwargs))
                return Response()

        with patch("sglang.srt.entrypoints.http_server.aiohttp.ClientSession", Session):
            status_codes = await _send_disaggregation_warmup_requests(
                url="http://localhost:30000",
                headers={"Authorization": "Bearer token"},
                ssl_verify=False,
                timeout=123,
            )

        self.assertEqual(status_codes, [200] * server_args.dp_size)
        self.assertEqual(len(calls), server_args.dp_size)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(
            sessions[0].kwargs["headers"], {"Authorization": "Bearer token"}
        )
        self.assertEqual(sessions[0].kwargs["timeout"].total, 123)

        calls_by_rank = {
            kwargs["json"]["routed_dp_rank"]: (args, kwargs) for args, kwargs in calls
        }
        self.assertEqual(set(calls_by_rank), set(range(server_args.dp_size)))

        for dp_rank, (args, kwargs) in calls_by_rank.items():
            self.assertEqual(args, ("http://localhost:30000/generate",))
            self.assertEqual(kwargs["json"]["input_ids"], [10, 11, 12, 13])
            self.assertEqual(kwargs["json"]["bootstrap_host"], FAKE_BOOTSTRAP_HOST)
            self.assertEqual(kwargs["json"]["bootstrap_room"], dp_rank)
            self.assertFalse(kwargs["ssl"])


class TestMultiTokenizerWarmupSerialization(unittest.TestCase):
    def test_each_worker_warms_and_freezes_without_overlapping(self):
        ctx = multiprocessing.get_context("fork")
        worker_count = 4
        start = ctx.Barrier(worker_count)
        state_lock = ctx.Lock()
        active = ctx.Value("i", 0)
        peak = ctx.Value("i", 0)
        completed = ctx.Value("i", 0)
        callbacks = ctx.Value("i", 0)
        freezes = ctx.Value("i", 0)
        server_args = SimpleNamespace(gc_threshold=123)

        def run_worker(worker_id):
            events = []

            def warmup(args):
                with state_lock:
                    active.value += 1
                    peak.value = max(peak.value, active.value)
                time.sleep(0.05)
                with state_lock:
                    active.value -= 1
                    completed.value += 1
                events.append("warmup")
                return True

            def freeze_worker(label, threshold):
                self.assertEqual(events, ["warmup"])
                self.assertEqual(label, f"Tokenizer worker {worker_id}")
                self.assertEqual(threshold, 123)
                events.append("callback")
                with state_lock:
                    callbacks.value += 1

            def freeze_server(args):
                self.assertEqual(events, ["warmup", "callback"])
                with state_lock:
                    freezes.value += 1

            with (
                patch.object(http_server, "freeze_gc", side_effect=freeze_worker),
                patch.object(
                    http_server,
                    "_freeze_gc_after_server_warmup",
                    side_effect=freeze_server,
                ),
            ):
                kwargs = http_server._multi_tokenizer_warmup_kwargs(
                    server_args, worker_id
                )
                start.wait(timeout=10)
                http_server._wait_and_warmup(execute_warmup_func=warmup, **kwargs)

        with (
            tempfile.TemporaryDirectory() as lock_dir,
            patch.object(http_server.tempfile, "gettempdir", return_value=lock_dir),
            patch.object(http_server, "get_main_process_id", return_value=os.getpid()),
            patch.object(
                http_server,
                "get_serving",
                return_value=SimpleNamespace(
                    tokenizer_worker_num=worker_count, skip_server_warmup=False
                ),
            ),
            patch.object(
                http_server,
                "get_model",
                return_value=SimpleNamespace(
                    checkpoint_engine_wait_weights_before_ready=False,
                    delete_ckpt_after_loading=False,
                ),
            ),
            patch.object(
                http_server,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(is_ep_scale_joiner=False)
                ),
            ),
            patch.object(
                http_server,
                "get_observability",
                return_value=SimpleNamespace(debug_tensor_dump_input_file=None),
            ),
        ):
            workers = [
                ctx.Process(target=run_worker, args=(i,)) for i in range(worker_count)
            ]
            try:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=15)
                    self.assertEqual(worker.exitcode, 0)
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)

        self.assertEqual(peak.value, 1)
        self.assertEqual(active.value, 0)
        self.assertEqual(completed.value, worker_count)
        self.assertEqual(callbacks.value, worker_count)
        self.assertEqual(freezes.value, worker_count)

    def test_single_tokenizer_does_not_open_lock_file(self):
        with (
            patch.object(
                http_server,
                "get_serving",
                return_value=SimpleNamespace(tokenizer_worker_num=1),
            ),
            patch("builtins.open") as open_lock,
        ):
            with http_server._multi_tokenizer_warmup_lock():
                pass
            open_lock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
