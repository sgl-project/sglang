import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.mmmu_vlm_kit import MMMUMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.server_fixtures.mmmu_fixture import MMMUServerBase

register_cuda_ci(
    est_time=200,
    suite="nightly-2-gpu",
    nightly=True,
)

MODEL = "mistralai/Mistral-Small-4-119B-2603"


class TestMistralSmall4TextOnly(GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.9
    model = MODEL
    other_args = ["--tp-size", "2", "--trust-remote-code"]


class TestMistralSmall4MMMU(MMMUMixin, MMMUServerBase):
    accuracy = 0.45
    model = MODEL
    other_args = ["--tp-size", "2", "--trust-remote-code"]
    mmmu_args = ["--limit=0.1"]
    """`--limit=0.1`: 10 percent of each task - this is fine for testing since the nominal result isn't interesting - this run is just to prevent relative regressions."""


if __name__ == "__main__":
    unittest.main()
