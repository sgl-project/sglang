import unittest

from sglang.srt.utils import is_blackwell
from sglang.test.gsm8k_mixin import GSM8KMixin
from sglang.test.test_utils import CustomTestCase


class TestNvidiaNemotronNanoV2BF16(GSM8KMixin, CustomTestCase):
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    accuracy = 0.87
    other_args = ["--max-mamba-cache-size", "256"]


class TestNvidiaNemotronNanoV2FP8(GSM8KMixin, CustomTestCase):
    accuracy = 0.87
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8"
    other_args = ["--max-mamba-cache-size", "256"]


@unittest.skipIf(not is_blackwell(), "NVFP4 only supported on blackwell")
class TestNvidiaNemotronNanoV2NVFP4(GSM8KMixin, CustomTestCase):
    accuracy = 0.855
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4"
    other_args = ["--max-mamba-cache-size", "256"]


if __name__ == "__main__":
    unittest.main()
