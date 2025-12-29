import unittest

from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.kits.mmmu_vlm_kit import MMMUMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.server_fixtures.mmmu_fixture import MMMUServerBase

MODEL = "mistralai/Ministral-3-3B-Instruct-2512"


class TestMinistral3TextOnly(GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.6
    model = MODEL
    other_args = ["--trust-remote-code"]


class TestMinistral3MMMU(MMMUMixin, MMMUServerBase):
    accuracy = 0.3
    model = MODEL
    other_args = ["--trust-remote-code"]
    mmmu_args = ["--limit=0.1"]
    """`--limit=0.1`: 10 percent of each task - this is fine for testing since the nominal result isn't interesting - this run is just to prevent relative regressions."""


if __name__ == "__main__":
    unittest.main()
