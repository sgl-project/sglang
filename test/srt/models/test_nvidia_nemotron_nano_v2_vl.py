import unittest
from types import SimpleNamespace

from sglang.test.gsm8k_mixin import GSM8KMixin
from sglang.test.mmmu_vlm_mixin import MMMUVLMMixin
from sglang.test.test_utils import CustomTestCase

MODEL = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"


class TestNvidiaNemotronNanoV2VLTextOnly(GSM8KMixin, CustomTestCase):
    accuracy = 0.87
    model = MODEL
    other_args = ["--max-mamba-cache-size", "256", "--trust-remote-code"]


class TestNvidiaNemotronNanoV2VLMMMU(MMMUVLMMixin, CustomTestCase):
    accuracy = 0.454
    model = MODEL
    other_args = ["--max-mamba-cache-size", "128", "--trust-remote-code"]
    mmmu_args = ["--limit=0.1"]
    """`--limit=0.1`: 10 percent of each task - this is fine for testing since the nominal result isn't interesting - this run is just to prevent relative regressions."""

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test(
            SimpleNamespace(model=self.model, mmmu_accuracy=self.accuracy), "./logs"
        )


if __name__ == "__main__":
    unittest.main()
