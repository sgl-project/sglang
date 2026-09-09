import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

_TEST_ROOT = Path(__file__).resolve().parents[3]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from registered.openai_server import rust_renderer

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestStandaloneRustRenderer(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.engine = mock.Mock(pid=101)
        self.renderer = mock.Mock(pid=202)
        self.engine.poll.return_value = None
        self.renderer.poll.return_value = None
        self.launch_engine = stack.enter_context(
            mock.patch.object(
                rust_renderer, "popen_launch_server", return_value=self.engine
            )
        )
        self.launch_renderer = stack.enter_context(
            mock.patch.object(
                rust_renderer.subprocess, "Popen", return_value=self.renderer
            )
        )
        self.kill = stack.enter_context(
            mock.patch.object(rust_renderer, "kill_process_tree")
        )
        self.which = stack.enter_context(
            mock.patch.object(
                rust_renderer.shutil, "which", return_value="/bin/sglang-renderer"
            )
        )
        stack.enter_context(mock.patch.dict(rust_renderer.os.environ, {}, clear=True))
        stack.enter_context(
            mock.patch.object(rust_renderer, "get_free_port", return_value=31000)
        )
        self.get = stack.enter_context(mock.patch.object(rust_renderer.requests, "get"))
        self.clock = stack.enter_context(
            mock.patch.object(rust_renderer.time, "monotonic", return_value=0)
        )
        stack.enter_context(mock.patch.object(rust_renderer.time, "sleep"))

    def test_readiness_requires_renderer_ownership_and_cleanup_owns_both_processes(
        self,
    ):
        self.get.side_effect = [
            mock.Mock(status_code=200, headers={"x-sglang-renderer": "ready"}),
            mock.Mock(status_code=204, headers={}),
            mock.Mock(status_code=204, headers={"x-sglang-renderer": "ready"}),
        ]
        with rust_renderer.launch_rust_renderer(
            "model",
            "http://127.0.0.1:30000",
            timeout=1,
            engine_args=["--tp", "2"],
            renderer_args=["--tool-call-parser", "llama3"],
        ):
            self.kill.assert_not_called()
        self.assertEqual(self.get.call_count, 3)
        self.get.assert_called_with(
            "http://127.0.0.1:30000/_sglang_renderer/ready", timeout=1
        )
        self.launch_engine.assert_called_once_with(
            "model",
            "http://127.0.0.1:31000",
            timeout=1,
            other_args=["--tp", "2"],
            env={"SGLANG_RUST_SERVER": "1"},
        )
        self.launch_renderer.assert_called_once_with(
            [
                "/bin/sglang-renderer",
                "model",
                "--engine-url",
                "http://127.0.0.1:31000",
                "--host",
                "127.0.0.1",
                "--port",
                "30000",
                "--proxy-unhandled-routes",
                "--sampling-defaults",
                "openai",
                "--tool-call-parser",
                "llama3",
            ]
        )
        self.assertEqual(self.kill.call_args_list, [mock.call(202), mock.call(101)])

    def test_renderer_startup_failures_clean_up_every_started_process(self):
        for failure in ("spawn", "exit", "timeout"):
            with self.subTest(failure=failure):
                self.kill.reset_mock()
                self.launch_renderer.side_effect = (
                    OSError("cannot launch renderer") if failure == "spawn" else None
                )
                self.renderer.poll.return_value = 1 if failure == "exit" else None
                self.clock.side_effect = [0, 0, 2] if failure == "timeout" else None
                self.get.return_value = mock.Mock(status_code=503, headers={})
                error = {
                    "spawn": OSError,
                    "exit": RuntimeError,
                    "timeout": TimeoutError,
                }[failure]
                with self.assertRaises(error):
                    with rust_renderer.launch_rust_renderer(
                        "model", "http://127.0.0.1:30000", timeout=1
                    ):
                        self.fail("startup failure must not yield a ready fixture")
                expected = [mock.call(101)]
                if failure != "spawn":
                    expected.insert(0, mock.call(202))
                self.assertEqual(self.kill.call_args_list, expected)

    def test_missing_renderer_fails_before_starting_engine(self):
        self.which.return_value = None
        with self.assertRaisesRegex(FileNotFoundError, "SGLANG_RENDERER_BIN"):
            with rust_renderer.launch_rust_renderer(
                "model", "http://127.0.0.1:30000", timeout=1
            ):
                self.fail("missing renderer must fail")
        self.launch_engine.assert_not_called()
        self.launch_renderer.assert_not_called()
        self.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
