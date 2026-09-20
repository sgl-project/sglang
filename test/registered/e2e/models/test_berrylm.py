"""BerryLM-OS: hybrid MoE with KDA gated delta-net linear attention and block
attention residuals (1 GPU, GSM8K accuracy through the default server fixture)."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

# TODO(maintainers): confirm the stage; 1-gpu-large = the runner class of the other ~18B-parameter model tests.
register_cuda_ci(
    est_time=300,
    disabled="RWB/BerryLM-OS is not public yet",
    stage="extra-a",
    runner_config="1-gpu-large",
)

BERRYLM_MODEL = "RWB/BerryLM-OS"


class TestBerryLM(GSM8KMixin, DefaultServerBase):
    model = BERRYLM_MODEL
    gsm8k_accuracy_thres = 0.85  # TODO: set from the release gate run
    other_args = [
        "--tp-size",
        "1",
        "--reasoning-parser",
        "berrylm",
        "--tool-call-parser",
        "berrylm",
    ]


if __name__ == "__main__":
    unittest.main()
