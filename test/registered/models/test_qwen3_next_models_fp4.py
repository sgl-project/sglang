import unittest

from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=500, suite="nightly-4-gpu-b200", nightly=True)

QWEN3_NEXT_MODEL_FP4 = "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4"


@unittest.skipIf(
    get_device_sm() < 100, "Test requires CUDA SM 100 or higher (Blackwell)"
)
class TestQwen3NextFp4(GSM8KMixin, DefaultServerBase):
    model = QWEN3_NEXT_MODEL_FP4
    gsm8k_accuracy_thres = 0.93
    other_args = [
        "--tp-size",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--quantization",
        "modelopt_fp4",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
    ]


if __name__ == "__main__":
    unittest.main()
