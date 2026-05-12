import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import is_blackwell
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=249, suite="stage-b-test-2-gpu-large")


class TestNvidiaNemotronNanoV2BF16(GSM8KMixin, DefaultServerBase):
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    gsm8k_accuracy_thres = 0.87
    other_args = ["--max-mamba-cache-size", "256"]


class TestNvidiaNemotronNanoV2BF16PP(GSM8KMixin, DefaultServerBase):
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    gsm8k_accuracy_thres = 0.87
    other_args = ["--max-mamba-cache-size", "256", "--pp-size", "2"]


class TestNvidiaNemotronNanoV2FP8(GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.87
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8"
    other_args = ["--max-mamba-cache-size", "256"]


@unittest.skipIf(not is_blackwell(), "NVFP4 only supported on blackwell")
class TestNvidiaNemotronNanoV2NVFP4(GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.855
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4"
    other_args = ["--max-mamba-cache-size", "256"]


class TestNvidiaNemotronNanoV2SpeculativeDecoding(GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.87
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    # NemotronH + STANDALONE + radix cache requires spec v2 +
    # `mamba-scheduler-strategy extra_buffer` (see server_args.py guard).
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
        "--mamba-scheduler-strategy",
        "extra_buffer",
    ]

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_ENABLE_SPEC_V2.set(True)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        envs.SGLANG_ENABLE_SPEC_V2.clear()

    def test_radix_attention(self):
        run_radix_attention_test(self.base_url)
        assert self.process.poll() is None


class TestNvidiaNemotronNanoV2SpeculativeDecodingBF16Cache(
    GSM8KMixin, DefaultServerBase
):
    gsm8k_accuracy_thres = 0.87
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
        "--mamba-scheduler-strategy",
        "extra_buffer",
    ]

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_ENABLE_SPEC_V2.set(True)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        envs.SGLANG_ENABLE_SPEC_V2.clear()

    def test_radix_attention(self):
        run_radix_attention_test(self.base_url)
        assert self.process.poll() is None


if __name__ == "__main__":
    unittest.main()
