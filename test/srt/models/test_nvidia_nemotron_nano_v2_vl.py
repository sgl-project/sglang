import unittest

from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.kits.mmmu_vlm_kit import MMMUMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.server_fixtures.mmmu_fixture import MMMUServerBase

MODEL = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"


class TestNvidiaNemotronNanoV2VLTextOnly(GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.87
    model = MODEL
    other_args = ["--max-mamba-cache-size", "256", "--trust-remote-code"]


class TestNvidiaNemotronNanoV2VLMMMU(MMMUMixin, MMMUServerBase):
    accuracy = 0.444
    model = MODEL
    other_args = ["--max-mamba-cache-size", "128", "--trust-remote-code"]
    mmmu_args = ["--limit=0.1"]
    """`--limit=0.1`: 10 percent of each task - this is fine for testing since the nominal result isn't interesting - this run is just to prevent relative regressions."""


if __name__ == "__main__":
    unittest.main()
