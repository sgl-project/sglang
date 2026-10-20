"""Standalone engine shutdown. Set SGLANG_TEST_SHUTDOWN_TP/DP for multi-rank runs.

A source build can be selected with SGLANG_TEST_GRPC_EXTENSION. No Dynamo
process is involved. Emergency cleanup runs only after the exit assertions.
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import grpc
import psutil
import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.grpc_shutdown_helpers import runtime_client, runtime_proto
from sglang.test.test_utils import find_available_port

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-small")


class TestEngineShutdown(unittest.TestCase):
    def test_shutdown_exits_engine_and_schedulers(self):
        proto, service = runtime_proto()
        http_port = find_available_port(31000)
        grpc_port = find_available_port(http_port + 1000)
        tp = int(os.environ.get("SGLANG_TEST_SHUTDOWN_TP", "1"))
        dp = int(os.environ.get("SGLANG_TEST_SHUTDOWN_DP", "1"))
        env = os.environ.copy()
        env["SGLANG_RUST_SERVER"] = "0"
        env["SGLANG_FORCE_SHUTDOWN"] = "0"
        env.pop("SGL_FORCE_SHUTDOWN", None)
        env["SGLANG_RUST_BUILD_MODE"] = "never"
        # Preload an explicitly selected extension in the engine process. With
        # a wheel, load_test_grpc_extension simply loads the bundled module.
        bootstrap = (
            "from sglang.test.grpc_shutdown_helpers import load_test_grpc_extension; "
            "load_test_grpc_extension(); import runpy; "
            "runpy.run_module('sglang.launch_server', run_name='__main__')"
        )
        command = [
            sys.executable,
            "-c",
            bootstrap,
            "--model-path",
            os.environ.get("SGLANG_TEST_SHUTDOWN_MODEL", "Qwen/Qwen3-0.6B"),
            "--host",
            "127.0.0.1",
            "--port",
            str(http_port),
            "--grpc-port",
            str(grpc_port),
            "--tp-size",
            str(tp),
            "--dp-size",
            str(dp),
            "--mem-fraction-static",
            "0.3",
            "--context-length",
            "2048",
            "--disable-cuda-graph",
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
        ]
        base = f"http://127.0.0.1:{http_port}"
        with tempfile.TemporaryFile(mode="w+") as log:
            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=log, start_new_session=True
            )
            channel = grpc.insecure_channel(
                f"127.0.0.1:{grpc_port}", options=(("grpc.enable_retries", 0),)
            )
            client = runtime_client(channel, service)
            schedulers = []
            shutdown_verified = False
            try:
                deadline = time.monotonic() + 300
                while True:
                    self.assertIsNone(process.poll(), "engine exited during startup")
                    try:
                        if requests.get(f"{base}/health", timeout=1).status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    self.assertLess(
                        time.monotonic(), deadline, "engine never became ready"
                    )
                    time.sleep(0.2)
                grpc.channel_ready_future(channel).result(timeout=10)
                response = requests.post(
                    f"{base}/generate",
                    json={
                        "text": "Hello",
                        "sampling_params": {"max_new_tokens": 1},
                    },
                    timeout=30,
                )
                response.raise_for_status()
                schedulers = [
                    child
                    for child in psutil.Process(process.pid).children(recursive=True)
                    if " ".join(child.cmdline()).startswith("sglang::scheduler")
                ]
                self.assertEqual(
                    len(schedulers), tp * dp, "missing required scheduler ranks"
                )
                watch = client.WatchEngineState(
                    proto.WatchEngineStateRequest(), timeout=90
                )
                self.assertTrue(next(watch).healthy)

                # Consume the long-lived watch throughout shutdown. It must not
                # hold Tonic's graceful listener shutdown open indefinitely.
                with ThreadPoolExecutor(max_workers=1) as executor:

                    def consume_watch():
                        snapshots = []
                        try:
                            snapshots.extend(watch)
                        except grpc.RpcError as error:
                            self.assertIn(
                                error.code(),
                                (
                                    grpc.StatusCode.UNAVAILABLE,
                                    grpc.StatusCode.CANCELLED,
                                ),
                            )
                        return snapshots

                    observed = executor.submit(consume_watch)
                    try:
                        reply = client.Shutdown(proto.ShutdownRequest(), timeout=5)
                        self.assertEqual(reply.SerializeToString(), b"")
                    except grpc.RpcError as error:
                        self.assertIn(
                            error.code(),
                            (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED),
                        )
                    try:
                        deadline = time.monotonic() + 5
                        while client.HealthCheck(
                            proto.HealthCheckRequest(), timeout=2
                        ).healthy:
                            self.assertLess(time.monotonic(), deadline)
                            time.sleep(0.05)
                    except grpc.RpcError as error:
                        self.assertEqual(error.code(), grpc.StatusCode.UNAVAILABLE)
                    self.assertEqual(process.wait(timeout=60), 0)
                    observed.result(timeout=5)
                _, alive = psutil.wait_procs(schedulers, timeout=5)
                self.assertFalse(
                    alive, f"scheduler processes survived: {[p.pid for p in alive]}"
                )
                with self.assertRaises(requests.RequestException):
                    requests.get(f"{base}/health", timeout=2)
                shutdown_verified = True
            except BaseException:
                log.seek(0)
                print(log.read()[-20000:], file=sys.stderr)
                raise
            finally:
                channel.close()
                if not shutdown_verified:
                    # Also remove orphaned test children if startup failed
                    # after the parent exited. This cannot satisfy assertions.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
