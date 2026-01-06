import json
import tempfile
import time
import unittest
from pathlib import Path

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
    run_bench_serving,
)

register_cuda_ci(est_time=300, suite="nightly-1-gpu", nightly=True)

MODEL = "Qwen/Qwen3-0.6B"


class TestBenchServingFunctionality(CustomTestCase):
    def test_gsp_multi_turn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = popen_launch_server(
                MODEL,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--mem-fraction-static", "0.7",
                    "--log-requests",
                    "--log-requests-level", "3",
                    "--log-requests-format", "json",
                    "--log-requests-target", temp_dir,
                ],
            )
            try:
                args = get_benchmark_args(
                    base_url=DEFAULT_URL_FOR_TEST,
                    dataset_name="generated-shared-prefix",
                    num_prompts=4,
                    request_rate=float("inf"),
                    disable_ignore_eos=True,
                    gsp_num_groups=2,
                    gsp_prompts_per_group=2,
                    gsp_system_prompt_len=64,
                    gsp_question_len=16,
                    gsp_output_len=16,
                    gsp_num_turns=3,
                )
                res = run_benchmark(args)

                self.assertEqual(res["completed"], 4 * 3)
                self.assertGreater(res["output_throughput"], 0)

                time.sleep(1)

                log_files = list(Path(temp_dir).glob("*.log"))
                self.assertGreater(len(log_files), 0, "No log files found")

                file_content = "".join(f.read_text() for f in log_files)
                self._verify_multi_turn_logs(file_content, num_conversations=4, num_turns=3)
            finally:
                kill_process_tree(process.pid)

    def _verify_multi_turn_logs(self, content: str, num_conversations: int, num_turns: int):
        finished_requests = []
        for line in content.splitlines():
            if not line.strip() or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            rid = data.get("rid", "")
            if rid.startswith("HEALTH_CHECK"):
                continue

            if data.get("event") == "request.finished":
                finished_requests.append(data)

        expected_total = num_conversations * num_turns
        self.assertEqual(
            len(finished_requests),
            expected_total,
            f"Expected {expected_total} finished requests, got {len(finished_requests)}",
        )

        prompt_lengths = []
        for req in finished_requests:
            obj = req.get("obj", {})
            prompt = obj.get("text", "")
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 0, "Prompt should not be empty")
            prompt_lengths.append(len(prompt))

        prompt_lengths.sort()
        short_prompts = prompt_lengths[:num_conversations]
        long_prompts = prompt_lengths[-num_conversations:]
        avg_short = sum(short_prompts) / len(short_prompts)
        avg_long = sum(long_prompts) / len(long_prompts)
        self.assertGreater(
            avg_long,
            avg_short * 1.5,
            f"Later turns should have longer prompts due to history. "
            f"Shortest avg: {avg_short}, Longest avg: {avg_long}",
        )


if __name__ == "__main__":
    unittest.main()
