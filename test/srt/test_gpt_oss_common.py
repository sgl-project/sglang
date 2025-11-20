import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Dict, List, Literal, Optional

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

_base_url = DEFAULT_URL_FOR_TEST
_is_hip = is_hip()


class BaseTestGptOss(CustomTestCase):
    def run_test(
        self,
        model_variant: Literal["20b", "120b"],
        quantization: Literal["mxfp4", "bf16"],
        expected_score_of_reasoning_effort: Dict[str, float],
        other_args: Optional[List[str]] = None,
    ):
        if other_args is None:
            other_args = []

        model = {
            ("20b", "bf16"): "lmsys/gpt-oss-20b-bf16",
            ("120b", "bf16"): "lmsys/gpt-oss-120b-bf16",
            ("20b", "mxfp4"): "openai/gpt-oss-20b",
            ("120b", "mxfp4"): "openai/gpt-oss-120b",
        }[(model_variant, quantization)]

        if model_variant == "20b":
            other_args += ["--cuda-graph-max-bs", "600"]
        if _is_hip:
            os.environ["SGLANG_USE_AITER"] = "0"
        self._run_test_raw(
            model=model,
            expected_score_of_reasoning_effort=expected_score_of_reasoning_effort,
            other_args=other_args,
        )

    def _run_test_raw(
        self,
        model: str,
        expected_score_of_reasoning_effort: Dict[str, float],
        other_args: List[str],
    ):
        process = popen_launch_server(
            model,
            _base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        try:
            # run multiple tests in parallel since we are mostly bound by the longest generate sequence
            # instead of the number of questions
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(
                    executor.map(
                        lambda d: self._run_one_eval(**d),
                        [
                            dict(
                                model=model,
                                reasoning_effort=reasoning_effort,
                                expected_score=expected_score,
                            )
                            for reasoning_effort, expected_score in expected_score_of_reasoning_effort.items()
                        ],
                    )
                )
        finally:
            kill_process_tree(process.pid)

    def _run_one_eval(self, model, reasoning_effort, expected_score):
        args = SimpleNamespace(
            base_url=_base_url,
            model=model,
            eval_name="gpqa",
            num_examples=198,
            # use enough threads to allow parallelism
            num_threads=198,
            # TODO 4k is still not enough, we need e.g. 64k token, but that is super slow
            # otherwise a lot of questions are not answered
            max_tokens=4096,
            # simple-evals by default use 0.5 and is better than 0.0 temperature
            # but here for reproducibility, we use 0.1
            temperature=0.1,
            reasoning_effort=reasoning_effort,
        )

        setup = f"model={model} reasoning_effort={reasoning_effort} expected_score={expected_score}"

        print(f"Evaluation start: {setup}")
        metrics = run_eval(args)
        print(f"Evaluation end: {setup} {metrics=}")
        self.assertGreaterEqual(metrics["score"], expected_score)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gpt_oss_common\n"
                f"Setup: {setup}\n"
                f"Score: {metrics['score']:.2f}\n"
            )
