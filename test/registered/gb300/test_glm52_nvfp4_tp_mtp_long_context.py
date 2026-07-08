import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.long_context_bench_kit import LongContextBenchServingMixin
from sglang.test.server_fixtures.dsa_mtp_fixture import DsaMtpServerBase

register_cuda_ci(
    est_time=1800,
    suite="nightly-4-gpu-gb300-glm5-nvfp4",
    nightly=True,
)


class TestGLM52NVFP4TPMTPLongContext(DsaMtpServerBase, LongContextBenchServingMixin):
    model = "nvidia/GLM-5.2-NVFP4"
    tp_size = 4
    mem_fraction_static = 0.8
    extra_server_args = [
        "--moe-runner-backend",
        "flashinfer_trtllm",
        "--quantization",
        "modelopt_fp4",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "8192",
        "--max-running-requests",
        "8",
    ]


if __name__ == "__main__":
    unittest.main()
