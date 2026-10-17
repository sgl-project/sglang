# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the portable launch command and result collector."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from summarize_perf import summarize

HERE = Path(__file__).resolve().parent


class TestReproduction(unittest.TestCase):
    def test_launch_accuracy_clears_synthetic_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / "python3"
            fake.write_text(
                f"#!{sys.executable}\n"
                "import os,sys,json\n"
                "print(json.dumps({'args':sys.argv[1:], 'env':"
                "{k:v for k,v in os.environ.items() if k.startswith(('SGLANG_', 'AITER_'))}}))\n"
            )
            fake.chmod(0o755)
            env = dict(os.environ, PATH=f"{folder}:{os.environ['PATH']}")
            env["SGLANG_SIMULATE_ACC_LEN"] = "9"
            env["SGLANG_TEST_STALE_EXPERIMENT"] = "1"
            env["AITER_TEST_STALE_EXPERIMENT"] = "1"
            env["AITER_JIT_DIR"] = "/allocation/cache/aiter"
            env["SGLANG_JIT_CACHE_DIR"] = "/allocation/cache/sglang"
            for mode, fusion in (("accuracy", "0"), ("perf", "1")):
                result = json.loads(
                    subprocess.check_output(
                        [
                            "bash",
                            str(HERE / "launch_server.sh"),
                            "/model",
                            mode,
                            fusion,
                            "30000",
                        ],
                        env=env,
                        text=True,
                    )
                )
                args, actual = result["args"], result["env"]
                self.assertEqual(args[:2], ["-m", "sglang.launch_server"])
                self.assertEqual(args[args.index("--tp") + 1], "4")
                self.assertEqual(args[args.index("--ep-size") + 1], "1")
                self.assertEqual(
                    args[args.index("--speculative-dspark-block-size") + 1], "5"
                )
                self.assertEqual(actual["SGLANG_SET_CPU_AFFINITY"], "0")
                self.assertEqual(actual["SGLANG_DSV41_SHARED_ROUTER_FUSION"], fusion)
                self.assertNotIn("SGLANG_TEST_STALE_EXPERIMENT", actual)
                self.assertNotIn("AITER_TEST_STALE_EXPERIMENT", actual)
                self.assertEqual(actual["AITER_JIT_DIR"], "/allocation/cache/aiter")
                self.assertEqual(
                    actual["SGLANG_JIT_CACHE_DIR"], "/allocation/cache/sglang"
                )
                if mode == "accuracy":
                    self.assertNotIn("SGLANG_SIMULATE_ACC_LEN", actual)
                else:
                    self.assertEqual(actual["SGLANG_SIMULATE_ACC_LEN"], "3.51")
                    self.assertEqual(
                        actual["SGLANG_SIMULATE_ACC_METHOD"], "match-expected"
                    )
                    self.assertEqual(actual["SGLANG_RAGGED_VERIFY_MODE"], "static")

    def test_collector_rejects_incomplete_repetitions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            requests = [{"prompt_tokens": 8192, "requested_output_tokens": 1024}] * 160
            row = {
                "completed": 160,
                "num_prompts": 160,
                "errors": [None] * 160,
                "output_lens": [1024] * 160,
                "input_lens": [8192] * 160,
                "total_output_tokens": 160 * 1024,
                "median_tpot_ms": 3.0,
                "median_itl_ms": 10.0,
                "median_ttft_ms": 250.0,
                "output_throughput": 600.0,
            }
            for repeat in (1, 2, 3):
                dest = root / f"repeat-{repeat:02d}"
                dest.mkdir()
                (dest / "benchmark.json").write_text(json.dumps(row))
                (dest / "requests.json").write_text(json.dumps({"requests": requests}))
                for phase, calls, tokens in (("before", 0, 0), ("after", 100, 351)):
                    (dest / f"metrics.{phase}.prom").write_text(
                        f'sglang:spec_verify_calls_total{{model="test"}} {calls}\n'
                        f'sglang:generation_tokens_total{{model="test"}} {tokens}\n'
                    )
            self.assertEqual(summarize(root)["status"], "valid_perf_only")
            row["errors"][0] = "failed request"
            (root / "repeat-02/benchmark.json").write_text(json.dumps(row))
            with self.assertRaises(AssertionError):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
