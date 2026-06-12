from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPyproject(CustomTestCase):
    def test_pyproject_excludes_cuda_only_dependencies(self):
        pyproject = Path("python/pyproject_mlu.toml").read_text()
        cuda_only = [
            "cuda-python",
            "flashinfer_python",
            "flashinfer_cubin",
            "nvidia-cutlass-dsl",
            "flash-attn-4",
            "sgl-deep-gemm",
            "sglang-kernel",
            "tilelang",
            "tokenspeed_mla",
            "torch==",
            "torchaudio==",
            "torchvision",
        ]
        for dependency in cuda_only:
            with self.subTest(dependency=dependency):
                self.assertNotIn(dependency, pyproject)

        self.assertIn("srt_mlu", pyproject)
        self.assertIn("all_mlu", pyproject)
        self.assertIn("dev_mlu", pyproject)
