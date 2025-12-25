import unittest

from sglang.srt.utils import is_blackwell
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.mamba_scheduler_strategy_test_utils import MambaSchedulerStrategyMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=132, suite="stage-b-test-large-2-gpu")

NVIDIA_NEMOTRON_NANO_V2_MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"


class TestNvidiaNemotronNanoV2BF16(GSM8KMixin, DefaultServerBase):
    model = NVIDIA_NEMOTRON_NANO_V2_MODEL
    gsm8k_accuracy_thres = 0.87
    other_args = ["--max-mamba-cache-size", "256"]


class TestNvidiaNemotronNanoV2BF16PP(GSM8KMixin, DefaultServerBase):
    model = NVIDIA_NEMOTRON_NANO_V2_MODEL
    gsm8k_accuracy_thres = 0.87
    other_args = ["--max-mamba-cache-size", "256", "--pp-size", "2"]

class TestNvidiaNemotronNanoV2BF16ExtraBuffer(
    GSM8KMixin, MambaSchedulerStrategyMixin, DefaultServerBase
):
    model = NVIDIA_NEMOTRON_NANO_V2_MODEL
class TestNvidiaNemotronNanoV2BF16PP(GSM8KMixin, DefaultServerBase):
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    gsm8k_accuracy_thres = 0.87
    other_args = ["--max-mamba-cache-size", "256", "--pp-size", "2"]


class TestNvidiaNemotronNanoV2FP8(GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.87
    kl_div_thres = 0.008
    other_args = [
        "--max-mamba-cache-size",
        "256",
        "--mamba-scheduler-strategy",
        "extra_buffer",
    ]

    def test_input_output_logprobs_match_prefill_cache_hit(self):
        self._test_input_output_logprobs_match_prefill_cache_hit_helper(
            max_samples=32,
            max_new_tokens=512,
        )

    def test_input_output_logprobs_match_decode_cache_hit(self):
        self._test_input_output_logprobs_match_decode_cache_hit_helper(
            max_samples=32,
            max_new_tokens=512,
        )

    def test_prefix_cache_branching(self):
        self._test_prefix_cache_branching_helper(256)


class TestNvidiaNemotronNanoV2FP8(GSM8KMixin, DefaultServerBase):
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8"
    gsm8k_accuracy_thres = 0.87
    other_args = ["--max-mamba-cache-size", "256"]


@unittest.skipIf(not is_blackwell(), "NVFP4 only supported on blackwell")
class TestNvidiaNemotronNanoV2NVFP4(GSM8KMixin, DefaultServerBase):
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4"
    gsm8k_accuracy_thres = 0.855
    other_args = ["--max-mamba-cache-size", "256"]


class TestNvidiaNemotronNanoV2SpeculativeDecoding(GSM8KMixin, DefaultServerBase):
    model = NVIDIA_NEMOTRON_NANO_V2_MODEL
    gsm8k_accuracy_thres = 0.87
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
        '{"vocab_size": 131072, "hidden_size": 4480}',
    ]


class TestNvidiaNemotronNanoV2SpeculativeDecodingExtraBuffer(
    GSM8KMixin, DefaultServerBase
):
    model = NVIDIA_NEMOTRON_NANO_V2_MODEL
    gsm8k_accuracy_thres = 0.87
    other_args = TestNvidiaNemotronNanoV2SpeculativeDecoding.other_args + [
        "--mamba-scheduler-strategy",
        "extra_buffer",
    ]


class TestNvidiaNemotronNanoV2SpeculativeDecodingBF16Cache(
    GSM8KMixin, DefaultServerBase
):
    model = NVIDIA_NEMOTRON_NANO_V2_MODEL
    gsm8k_accuracy_thres = 0.87
    other_args = TestNvidiaNemotronNanoV2SpeculativeDecoding.other_args + [
        "--mamba-ssm-dtype",
        "bfloat16",
    ]


if __name__ == "__main__":
    unittest.main()
