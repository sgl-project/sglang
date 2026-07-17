import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="2-gpu-large")

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


class TestKimiLinear(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = KIMI_LINEAR_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--tp-size", "2", "--trust-remote"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.88)


class TestKimiLinearExtraBuffer(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    """Regression guard: KDA never wrote mamba track snapshots, so states
    donated to the radix cache under mamba_radix_cache_strategy=extra_buffer
    were garbage and prefix-cache hits restored wrong KDA state (GSM8K
    0.150 pre-fix vs 0.895 post-fix). Pre-fix, launching KimiLinear with
    extra_buffer also fails the arch allowlist assert."""

    model = KIMI_LINEAR_MODEL
    cache_chunk_size = 64
    gsm8k_score_threshold = 0.88
    kl_div_thres = 0.002
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "2",
        "--chunked-prefill-size",
        "2048",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "2",
    ]


if __name__ == "__main__":
    unittest.main()
