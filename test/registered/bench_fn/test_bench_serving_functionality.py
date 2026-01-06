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
)

register_cuda_ci(est_time=300, suite="nightly-1-gpu", nightly=True)

MODEL = "Qwen/Qwen3-0.6B"
NUM_CONVERSATIONS, NUM_TURNS = 4, 3


class TestBenchServingFunctionality(CustomTestCase):
    def test_gsp_multi_turn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = popen_launch_server(
                MODEL,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--mem-fraction-static",
                    "0.7",
                    "--log-requests",
                    "--log-requests-level",
                    "3",
                    "--log-requests-format",
                    "json",
                    "--log-requests-target",
                    temp_dir,
                ],
            )
            try:
                args = get_benchmark_args(
                    base_url=DEFAULT_URL_FOR_TEST,
                    backend="sglang-oai-chat",
                    tokenizer=MODEL,
                    dataset_name="generated-shared-prefix",
                    num_prompts=NUM_CONVERSATIONS,
                    request_rate=float("inf"),
                    gsp_num_groups=2,
                    gsp_prompts_per_group=2,
                    gsp_system_prompt_len=64,
                    gsp_question_len=16,
                    gsp_output_len=16,
                    gsp_num_turns=NUM_TURNS,
                )
                args.warmup_requests = 0
                res = run_benchmark(args)
                self.assertEqual(res["completed"], NUM_CONVERSATIONS * NUM_TURNS)

                time.sleep(1)
                logs = "".join(f.read_text() for f in Path(temp_dir).glob("*.log"))
                self._verify_multi_turn_logs(logs)
            finally:
                kill_process_tree(process.pid)

    def _verify_multi_turn_logs(self, content: str):
        reqs = [
            json.loads(line)
            for line in content.splitlines()
            if line.startswith("{")
            and "request.finished" in line
            and not json.loads(line).get("rid", "").startswith("HEALTH_CHECK")
        ]
        self.assertEqual(len(reqs), NUM_CONVERSATIONS * NUM_TURNS)

        messages_list = [r.get("obj", {}).get("messages", []) for r in reqs]
        turn1_reqs = [m for m in messages_list if len(m) == 1]
        turn2_reqs = [m for m in messages_list if len(m) == 3]
        turn3_reqs = [m for m in messages_list if len(m) == 5]

        self.assertEqual(len(turn1_reqs), NUM_CONVERSATIONS, "Turn 1 should have 1 msg")
        self.assertEqual(len(turn2_reqs), NUM_CONVERSATIONS, "Turn 2 should have 3 msgs")
        self.assertEqual(len(turn3_reqs), NUM_CONVERSATIONS, "Turn 3 should have 5 msgs")

        prefix_count = 0
        for t2 in turn2_reqs:
            if any(t2[:1] == t1 for t1 in turn1_reqs):
                prefix_count += 1
        for t3 in turn3_reqs:
            if any(t3[:3] == t2 for t2 in turn2_reqs):
                prefix_count += 1
        expected = NUM_CONVERSATIONS * 2
        self.assertEqual(prefix_count, expected, f"Expected {expected} prefix pairs")


if __name__ == "__main__":
    unittest.main()
