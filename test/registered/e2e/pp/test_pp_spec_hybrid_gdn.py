import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=480, stage="extra-b", runner_config="2-gpu-large")

MTP_ARGS = """
--trust-remote-code --speculative-algorithm NEXTN --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--mamba-radix-cache-strategy extra_buffer --mamba-track-interval 128
--reasoning-parser qwen3 --mem-fraction-static 0.7
""".split()  # noqa: SIM905


class TestPPSpecHybridGDN(CustomTestCase):
    model = try_cached_model(DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST)

    def _run(self, pp_size):
        args = MTP_ARGS.copy()
        if pp_size > 1:
            args += ["--pp-size", str(pp_size), "--disable-overlap-schedule"]
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env={**os.environ, "SGLANG_ENABLE_PP_SPEC": "1"},
        )
        try:
            return run_eval(
                SimpleNamespace(
                    base_url=DEFAULT_URL_FOR_TEST,
                    model=self.model,
                    eval_name="gsm8k",
                    api="completion",
                    max_tokens=2048,
                    num_examples=32,
                    num_threads=16,
                )
            )["score"]
        finally:
            kill_process_tree(process.pid)

    def test_pp2_matches_pp1(self):
        pp1_score, pp2_score = self._run(1), self._run(2)
        self.assertLessEqual(abs(pp2_score - pp1_score), 0.02)


if __name__ == "__main__":
    unittest.main()
