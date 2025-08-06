from types import SimpleNamespace
from typing import Literal, List

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

from sglang.test.test_utils import CustomTestCase


class BaseTestGptOss(CustomTestCase):
    def run_test(
        self,
        model_variant: Literal["20b", "120b"],
        quantization: Literal["mxfp4", "bf16"],
        expected_score: float,
        other_args: List[str] = [],
    ):
        model = {
            ("20b", "bf16"): "lmsys/gpt-oss-20b-bf16",
            ("120b", "bf16"): "lmsys/gpt-oss-120b-bf16",
            ("20b", "mxfp4"): "openai/gpt-oss-20b",
            ("120b", "mxfp4"): "openai/gpt-oss-120b",
        }[(model_variant, quantization)]
        self._run_test_raw(
            model=model,
            expected_score=expected_score,
            other_args=other_args,
        )

    def _run_test_raw(
        self,
        model: str,
        expected_score: float,
        other_args: List[str],
    ):
        base_url = DEFAULT_URL_FOR_TEST

        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="gpqa",
                num_examples=198,
                # use enough threads to allow parallelism
                num_threads=198,
                # simple-evals by default use 0.5 and is better than 0.0 temperature
                # but here for reproducibility, we use 0.1
                temperature=0.1,
            )

            metrics = run_eval(args)
            print(f"Evaluation Result: {model=} {metrics}")
            self.assertGreaterEqual(metrics["score"], expected_score)
        finally:
            kill_process_tree(process.pid)
