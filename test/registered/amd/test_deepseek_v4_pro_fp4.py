"""MI35x DeepSeek-V4-Pro FP4 Test (8-GPU)

Combined accuracy + performance test for DeepSeek-V4-Pro (1.6T) FP4 on
MI35x ROCm 7.2.
- Accuracy: GSM8K few-shot eval
- Performance: bench_one_batch_server with input_len=8192, output_len=1024 (bs=1)

Both tests share a single launched server.

Registry: nightly-amd-8-gpu-mi35x-deepseek-v4-pro suite
"""

import json
import os
import subprocess
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=14400, suite="nightly-amd-8-gpu-mi35x-deepseek-v4-pro", nightly=True
)

DEEPSEEK_V4_PRO_FP4_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_PRO_MODEL_PATH_FP4", "deepseek-ai/DeepSeek-V4-Pro"
)
# Pro is 1.6T; weight load + warmup is much longer than Flash 285B.
SERVER_LAUNCH_TIMEOUT = 5400

# Common DeepSeek-V4 env vars (AMD ROCm 7.2 path: tilelang + AITER + ROCm700A).
# Source of truth: python/run_dsv4.sh.
COMMON_ENV_VARS = {
    "SGLANG_OPT_USE_FUSED_COMPRESS": "false",
    "SGLANG_OPT_USE_OLD_COMPRESSOR": "true",
    "SGLANG_OPT_USE_TILELANG_SWA_PREPARE": "false",
    "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": "false",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "false",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
    "SGLANG_ENABLE_THINKING": "1",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_ROCM700A": "1",
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
    "SGLANG_OPT_DPSK_V4_RADIX": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "false",
    "SGLANG_OPT_USE_FUSED_STORE_CACHE": "false",
    "SGLANG_TOPK_TRANSFORM_512_TORCH": "1",
    "SGLANG_OPT_USE_TILELANG_INDEXER": "true",
    "SGLANG_HACK_FLASHMLA_BACKEND": "tilelang",
    "SGLANG_REASONING_EFFORT": "max",
}

# FP4 variant: FP4 mixed-precision experts.
FP4_ENV_VARS = {
    "SGLANG_DSV4_FP4_EXPERTS": "true",
    "SGLANG_FORCE_TRITON_MOE_FP8": "0",
}


class TestDeepseekV4ProFp4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_PRO_FP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = os.environ.copy()
        env.update(COMMON_ENV_VARS)
        env.update(FP4_ENV_VARS)

        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--disable-radix-cache",
            "--attention-backend",
            "compressed",
            "--max-running-requests",
            "256",
            "--page-size",
            "256",
            "--chunked-prefill-size",
            "8192",
            "--disable-shared-experts-fusion",
            "--tool-call-parser",
            "deepseekv4",
            "--reasoning-parser",
            "deepseek-v4",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        # `a` prefix to run first (alphabetical) and warm up the server.
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v4-pro-fp4)\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )
            self.assertGreater(metrics["accuracy"], 0.92)

    def test_b_perf_8k_1k(self):
        json_output = "/tmp/deepseek_v4_pro_fp4_perf.json"
        if os.path.exists(json_output):
            os.remove(json_output)

        # First "1" is a warmup; the markdown report below skips it.
        batch_sizes = ["1", "1", "2", "4", "8", "16", "32"]
        cmd = [
            "python3",
            "-m",
            "sglang.bench_one_batch_server",
            "--model",
            "None",
            "--base-url",
            self.base_url,
            "--batch-size",
            *batch_sizes,
            "--input-len",
            "8192",
            "--output-len",
            "1024",
            "--show-report",
            f"--pydantic-result-filename={json_output}",
            "--no-append-to-github-summary",
            "--trust-remote-code",
        ]
        print(f"Running benchmark: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            self.fail(f"bench_one_batch_server failed (rc={result.returncode})")

        self.assertTrue(
            os.path.exists(json_output),
            f"Benchmark JSON output {json_output} not found",
        )
        with open(json_output) as f:
            results_data = json.load(f)
        self.assertTrue(results_data, "No benchmark results returned")

        if (
            len(results_data) > 1
            and results_data[0]["batch_size"] == results_data[1]["batch_size"]
        ):
            report_results = results_data[1:]
        else:
            report_results = results_data

        summary_lines = [
            "### test_perf_8k_1k (deepseek-v4-pro-fp4)",
            "input_len=8192 output_len=1024",
            "",
            "| batch size | latency (s) | input throughput (tok/s) | output throughput (tok/s) | ITL (ms) |",
            "| ---------- | ----------- | ------------------------ | ------------------------- | -------- |",
        ]
        for r in report_results:
            bs = r["batch_size"]
            latency = r.get("latency", 0.0)
            in_tp = r.get("input_throughput", 0.0)
            out_tp = r.get("output_throughput", 0.0)
            itl = 1 / (out_tp / bs) * 1000 if out_tp > 0 else float("inf")
            summary_lines.append(
                f"| {bs} | {latency:.2f} | {in_tp:.2f} | {out_tp:.2f} | {itl:.2f} |"
            )
            print(
                f"bs={bs} latency={latency:.2f}s "
                f"in_tp={in_tp:.2f} tok/s out_tp={out_tp:.2f} tok/s ITL={itl:.2f}ms"
            )

        if is_in_ci():
            write_github_step_summary("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    unittest.main()
