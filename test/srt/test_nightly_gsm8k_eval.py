import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_QUANT_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


def parse_models(model_string):
    return [model.strip() for model in model_string.split(",") if model.strip()]


class TestEvalAccuracyLarge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_groups = [
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1), False, False),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2), False, True),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1), True, False),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2), True, True),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_QUANT_TP1), False, False),
        ]
        cls.base_url = DEFAULT_URL_FOR_TEST

    def setUp(self):
        self.process = None

    def tearDown(self):
        if self.process:
            kill_child_process(self.process.pid)

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

    def test_mgsm_en_all_models(self):
        for model_group, is_fp8, is_tp2 in self.model_groups:
            for model in model_group:
                with self.subTest(model=model):
                    self.launch_server(model, is_fp8, is_tp2)

                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=model,
                        eval_name="mgsm_en",
                        num_examples=None,
                        num_threads=1024,
                    )

                    metrics = run_eval(args)
                    print(
                        f"{'=' * 42}\n{model} - metrics={metrics} score={metrics['score']}\n{'=' * 42}\n"
                    )
                    # loosely threshold
                    assert metrics["score"] > 0.5, f"score={metrics['score']} <= 0.5"

                    self.tearDown()


if __name__ == "__main__":
    unittest.main()
