import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatform

ENV_PATH = (
    "sglang.multimodal_gen.runtime.platforms.cuda.envs."
    "SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND"
)


def _backend(env_value, *, sm120):
    CudaPlatform.get_modelopt_flashinfer_fp4_backend.cache_clear()
    try:
        with (
            patch(ENV_PATH, env_value),
            patch.object(CudaPlatform, "is_sm120", classmethod(lambda cls: sm120)),
        ):
            return CudaPlatform.get_modelopt_flashinfer_fp4_backend()
    finally:
        CudaPlatform.get_modelopt_flashinfer_fp4_backend.cache_clear()


class TestModeloptFp4BackendDefault(unittest.TestCase):
    """The default FP4 GEMM backend must exist on the device.

    flashinfer.mm_fp4 raises "does not support backend 'trtllm' with capability
    120" on sm_120, so defaulting to trtllm there makes every NVFP4 model fail to
    run at all.
    """

    def test_sm120_does_not_default_to_a_backend_it_lacks(self):
        self.assertEqual(_backend(None, sm120=True), "auto")

    def test_other_gpus_keep_the_trtllm_default(self):
        self.assertEqual(_backend(None, sm120=False), "trtllm")

    def test_explicit_backends_are_passed_through(self):
        for env_value, expected in [
            ("trtllm", "trtllm"),
            ("flashinfer_trtllm", "trtllm"),
            ("cutlass", "cutlass"),
            ("flashinfer_cutlass", "cutlass"),
            ("cudnn", "cudnn"),
            ("flashinfer_cudnn", "cudnn"),
            ("auto", "auto"),
        ]:
            with self.subTest(env=env_value):
                self.assertEqual(_backend(env_value, sm120=True), expected)

    def test_unknown_backend_falls_back_to_the_platform_default(self):
        self.assertEqual(_backend("nonsense", sm120=True), "auto")
        self.assertEqual(_backend("nonsense", sm120=False), "trtllm")


if __name__ == "__main__":
    unittest.main()
