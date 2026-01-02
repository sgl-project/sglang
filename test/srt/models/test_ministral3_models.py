import unittest
from types import SimpleNamespace

from sglang.test.gsm8k_mixin import GSM8KMixin
from sglang.test.mmmu_vlm_mixin import MMMUVLMMixin
from sglang.test.test_utils import CustomTestCase

MODEL = "mistralai/Ministral-3-3B-Instruct-2512"


class TestMinistral3TextOnly(GSM8KMixin, CustomTestCase):
    accuracy = 0.6
    model = MODEL
    other_args = ["--trust-remote-code"]


class TestMinistral3MMMU(MMMUVLMMixin, CustomTestCase):
    accuracy = 0.3
    model = MODEL
    other_args = ["--trust-remote-code"]
    mmmu_args = ["--limit=0.1"]
    """`--limit=0.1`: 10 percent of each task - this is fine for testing since the nominal result isn't interesting - this run is just to prevent relative regressions."""

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test(
            SimpleNamespace(model=self.model, mmmu_accuracy=self.accuracy), "./logs"
        )


if __name__ == "__main__":
    unittest.main()
