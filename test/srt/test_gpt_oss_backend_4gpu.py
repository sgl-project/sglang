import unittest

from test_gpt_oss_common import BaseTestGptOss

from sglang.srt.utils import get_device_sm


class TestGptOssBackend4Gpu(BaseTestGptOss):
    @unittest.skipIf(get_device_sm() < 100, "Test requires CUDA SM 100 or higher")
    def test_mxfp4_trtllm_moe_120b(self):
        self.run_test(
            model_variant="120b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.58,
            },
            other_args=[
                "--tp",
                "4",
                "--ep",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--attention-backend",
                "trtllm_mha",
                "--moe-runner-backend",
                "flashinfer_trtllm",
                "--cuda-graph-max-bs",
                "640",
                "--mem-fraction-static",
                "0.93",
            ],
        )

    @unittest.skipIf(get_device_sm() < 100, "Test requires CUDA SM 100 or higher")
    def test_mxfp4_cutlass_moe_120b(self):
        self.run_test(
            model_variant="120b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.58,
            },
            other_args=[
                "--tp",
                "4",
                "--ep",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--attention-backend",
                "trtllm_mha",
                "--moe-runner-backend",
                "flashinfer_cutlass",
                "--cuda-graph-max-bs",
                "640",
                "--mem-fraction-static",
                "0.93",
            ],
        )


if __name__ == "__main__":
    unittest.main()
