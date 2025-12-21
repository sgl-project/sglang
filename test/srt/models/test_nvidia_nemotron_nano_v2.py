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


@unittest.skip(
    "STANDALONE speculative decoding does not yet support target and draft models "
    "with different hidden sizes (Nemotron-9B: 4480, Llama-3.2-1B: 2048)"
)
class TestNvidiaNemotronNanoV2SpeculativeDecoding(GSM8KMixin, CustomTestCase):
    accuracy = 0.87
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    other_args = [
        "--speculative-algorithm",
        "STANDALONE",
        "--speculative-num-steps",
        "2",
        "--speculative-eagle-topk",
        "3",
        "--speculative-num-draft-tokens",
        "5",
        "--speculative-draft-model-path",
        "meta-llama/Llama-3.2-1B",
        "--speculative-draft-load-format",
        "dummy",
        "--max-running-requests",
        "8",
        "--max-total-tokens",
        "2048",
        "--json-model-override-args",
        '{"vocab_size": 131072}',
    ]


@unittest.skip(
    "STANDALONE speculative decoding does not yet support target and draft models "
    "with different hidden sizes (Nemotron-9B: 4480, Llama-3.2-1B: 2048)"
)
class TestNvidiaNemotronNanoV2SpeculativeDecodingBF16Cache(GSM8KMixin, CustomTestCase):
    accuracy = 0.87
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    other_args = [
        "--speculative-algorithm",
        "STANDALONE",
        "--speculative-num-steps",
        "2",
        "--speculative-eagle-topk",
        "3",
        "--speculative-num-draft-tokens",
        "5",
        "--speculative-draft-model-path",
        "meta-llama/Llama-3.2-1B",
        "--speculative-draft-load-format",
        "dummy",
        "--max-running-requests",
        "8",
        "--max-total-tokens",
        "2048",
        "--json-model-override-args",
        '{"vocab_size": 131072}',
        "--mamba-ssm-dtype",
        "bfloat16",
    ]


if __name__ == "__main__":
    unittest.main()
