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

register_cuda_ci(est_time=900, stage="base-b", runner_config="2-gpu-large")

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
    kl_div_thres = 0.02
    kl_div_trust_remote_code = True
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


class TestKimiLinearUnifiedMemory(
    GSM8KMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    """BUG REGRESSION. The only per-PR cell that pairs the unified pool with an
    MLA model, so it is the only one that reaches the MLA chunked-prefix /
    MHA-one-shot path -- the producer that fetches its translator off
    `get_attn_backend()` and skips translation when a wrapper backend drops it
    (gsm8k 0.365 vs 0.895 measured on Kimi-Linear, flashinfer, TP2).

    No `--attention-backend` is pinned on purpose: both defects found in review
    on #32972 were reachable only under a resolved default, which a pinned test
    hides by construction. CUDA graphs stay on -- graph-off did not reproduce.

    `test_kimi_linear_unified_memory.py` runs this same configuration nightly.
    The duplication is deliberate: that suite resolves a different default on
    its H100 runner, and this one has to gate every PR.
    """

    model = KIMI_LINEAR_MODEL
    cache_chunk_size = 64
    # Same bar as the static-pool cell above: unified memory must not cost
    # accuracy (measured 0.917 unified vs 0.915 static, 1 sigma ~= 0.015).
    gsm8k_score_threshold = 0.88
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "2",
        "--chunked-prefill-size",
        "2048",
        "--enable-unified-memory",
    ]


if __name__ == "__main__":
    unittest.main()
