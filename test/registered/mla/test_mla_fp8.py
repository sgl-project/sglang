import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import MGSMEnMixin
from sglang.test.test_utils import (
    DEFAULT_MLA_FP8_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

# MLA FP8 KV cache test with MGSM evaluation
register_cuda_ci(est_time=106, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=800, suite="stage-b-test-1-gpu-small-amd")


class TestMLA(CustomTestCase, MGSMEnMixin):
    mgsm_en_score_threshold = 0.8

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_FP8_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--kv-cache-dtype",
            "fp8_e5m2",
            # Pin MoE expert dispatch and kernel reduction order so MGSM
            # scores don't drift across runs. The eval already uses greedy
            # decoding, but FP8 dequant + non-deterministic MoE top-k
            # tie-breaks produce ~1–3 point swings without this flag and
            # straddle the 0.8 threshold. With deterministic inference,
            # the score becomes a fixed function of (model, weights, CUDA
            # stack), so threshold-edge flakes stop being random noise.
        ]
        if not is_in_amd_ci():
            # On AMD, the default attention backend (aiter) is not in the deterministic-inference allowlist, so the server fails to start, disable it.
            other_args.append("--enable-deterministic-inference")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
