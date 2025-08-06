from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

from sglang.test.test_utils import CustomTestCase


class BaseTestGptOssCommon(CustomTestCase):
    def run_test(
        self,
        model: str,
        expected_score: float,
        other_args=[],
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
            self.assertGreaterEqual(metrics["score"], expected_score)
        finally:
            kill_process_tree(process.pid)
