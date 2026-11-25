import tempfile
import unittest
from unittest.mock import Mock, patch

from sglang.benchmark import offline_throughput as bench
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestOfflineProfile(unittest.TestCase):
    def test_waits_only_for_local_traces(self):
        backend = Mock()
        backend.generate.return_value = [{"meta_info": {"completion_tokens": 1}}]
        backend.get_server_info.return_value = {
            "internal_states": [{"last_gen_throughput": 1}]
        }
        req = bench.DatasetRow(prompt="hello", prompt_len=1, output_len=1)
        for activities, should_wait in [
            (["CUDA_PROFILER"], False),
            (["CPU", "CUDA_PROFILER"], True),
            (None, True),
        ]:
            with (
                self.subTest(activities=activities),
                tempfile.TemporaryDirectory() as directory,
                patch.dict("os.environ", SGLANG_TORCH_PROFILER_DIR=directory),
                patch.object(bench, "monitor_trace_file") as monitor,
            ):
                backend.reset_mock()
                bench.throughput_test_once(
                    "engine", backend, [req], False, {}, True, activities
                )
                backend.stop_profile.assert_called_once_with()
                self.assertEqual(monitor.called, should_wait)


if __name__ == "__main__":
    unittest.main()
