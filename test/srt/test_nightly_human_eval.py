import os
import shutil
import signal
import subprocess
import unittest
from types import SimpleNamespace

from test_nightly_gsm8k_eval import parse_models

from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestEvalAccuracyLarge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_groups = [
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1), False, False),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2), False, True),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1), True, False),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2), True, True),
        ]
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None
        cls.eval_process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_child_process(cls.process.pid)
        if cls.eval_process:
            kill_child_process(cls.eval_process.pid)

    def launch_server(self, model, is_fp8, is_tp2):
        other_args = ["--log-level-http", "warning", "--trust-remote-code"]
        if is_fp8:
            if "Llama-3" in model or "gemma-2" in model:
                # compressed-tensors
                other_args.extend(["--kv-cache-dtype", "fp8_e5m2"])
            elif "Qwen2-72B-Instruct-FP8" in model:
                # bug
                other_args.extend(["--quantization", "fp8"])
            else:
                other_args.extend(
                    ["--quantization", "fp8", "--kv-cache-dtype", "fp8_e5m2"]
                )
        if is_tp2:
            other_args.extend(["--tp", "2"])
        if "DeepSeek" in model:
            other_args.extend(["--mem-frac", "0.85"])
        if "AWQ" in model:
            other_args.extend(["--quantization", "awq"])
        elif "GPTQ" in model:
            other_args.extend(["--quantization", "gptq"])

        self.process = popen_launch_server(
            model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def run_evalplus(self, model):
        print("Delete evalplus results")
        shutil.rmtree("evalplus_results", ignore_errors=True)
        cmd = [
            "evalplus.evaluate",
            "--model",
            model,
            "--dataset",
            "humaneval",
            "--backend",
            "openai",
            "--base-url",
            "http://localhost:6157/v1",
            "--greedy",
        ]

        try:
            self.eval_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,
            )

            stdout, stderr = self.eval_process.communicate(timeout=600)

            if self.eval_process.returncode != 0:
                print(f"Fail to human eval model={model} err={stderr}")

            print("=" * 42)
            print(stdout)
            print("=" * 42)
        except subprocess.TimeoutExpired:
            if self.eval_process:
                os.killpg(os.getpgid(self.eval_process.pid), signal.SIGTERM)
            print(f"Timeout during evaluation for model={model}")
        except Exception as e:
            print(f"Error running evalplus for model={model} {str(e)}")
            if self.eval_process:
                os.killpg(os.getpgid(self.eval_process.pid), signal.SIGTERM)

    def test_human_eval_all_models(self):
        for model_group, is_fp8, is_tp2 in self.model_groups:
            for model in model_group:
                # NOTE: only Llama for now
                if "Llama" in model:
                    with self.subTest(model=model):
                        self.launch_server(model, is_fp8, is_tp2)
                        self.run_evalplus(model)
                        self.tearDownClass()


if __name__ == "__main__":
    unittest.main()
