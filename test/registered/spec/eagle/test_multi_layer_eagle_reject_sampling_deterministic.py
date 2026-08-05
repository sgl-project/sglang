import unittest
from urllib.parse import urlparse

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_deterministic import (
    BenchArgs,
)
from sglang.test.test_deterministic import test_deterministic as run_deterministic
from sglang.test.test_utils import DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

register_cuda_ci(est_time=480, stage="extra-b", runner_config="8-gpu-h200")

STEP3P5_FLASH_MODEL_PATH = "stepfun-ai/Step-3.5-Flash"


class TestStep3p5FlashMultiLayerEagleRSDeterministic(DefaultServerBase):
    model = STEP3P5_FLASH_MODEL_PATH
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3
    other_args = [
        "--tp",
        "8",
        "--trust-remote-code",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--attention-backend",
        "fa3",
        "--enable-multi-layer-eagle",
        "--speculative-use-rejection-sampling",
        "--enable-deterministic-inference",
        "--mem-fraction-static",
        "0.75",
        "--chunked-prefill-size",
        "4096",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
    ]

    def test_seeded_determinism(self):
        endpoint = urlparse(self.base_url)
        results = run_deterministic(
            BenchArgs(
                host=endpoint.hostname,
                port=endpoint.port,
                temperature=1.0,
                sampling_seed=42,
                max_new_tokens=32,
                test_mode="single",
                n_trials=6,
            )
        )
        self.assertEqual(results, [1])


if __name__ == "__main__":
    unittest.main()
