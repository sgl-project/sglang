import asyncio
import unittest
from unittest.mock import Mock, call, patch

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHTTPWorkerParentDeath(CustomTestCase):
    def test_arms_parent_death_signal_before_loading_worker_args(self):
        startup = Mock()
        startup.load_args.side_effect = FileNotFoundError
        with (
            patch.object(http_server, "get_main_process_id", return_value=100),
            patch.object(http_server.os, "getppid", return_value=100),
            patch.object(http_server, "kill_itself_when_parent_died", startup.arm),
            patch.object(http_server, "read_from_shared_memory", startup.load_args),
        ):
            with self.assertRaises(FileNotFoundError):
                asyncio.run(http_server.init_multi_tokenizer())

        self.assertEqual(
            startup.mock_calls, [call.arm(), call.load_args("multi_tokenizer_args_100")]
        )

    def test_exits_before_loading_worker_args_when_launcher_is_gone(self):
        with (
            patch.object(http_server, "get_main_process_id", return_value=100),
            patch.object(http_server.os, "getppid", return_value=1),
            patch.object(http_server, "kill_itself_when_parent_died") as arm,
            patch.object(http_server, "read_from_shared_memory") as load_args,
            patch.object(http_server.os, "_exit", side_effect=SystemExit(1)) as exit_,
        ):
            with self.assertRaises(SystemExit):
                asyncio.run(http_server.init_multi_tokenizer())

        arm.assert_called_once_with()
        exit_.assert_called_once_with(1)
        load_args.assert_not_called()


if __name__ == "__main__":
    unittest.main()
