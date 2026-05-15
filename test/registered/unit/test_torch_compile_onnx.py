import tempfile
import unittest

import torch

from sglang.srt.compilation.torch_compile import sgl_compile
from sglang.srt.environ import envs
from sglang.srt.platforms.builtin import CudaOnnxPlatform


class TestTorchCompileOnnx(unittest.TestCase):
    def test_cuda_onnx_platform_runs_exported_program(self):
        try:
            import numpy  # noqa: F401
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
            import onnxscript  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"ONNX optional dependency is unavailable: {exc}")

        import sglang.srt.platforms as platforms

        saved_platform = platforms._current_platform
        platforms._current_platform = CudaOnnxPlatform()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with envs.SGLANG_EXPORT_DIR.override(tmpdir):

                    @sgl_compile(key="unit_cuda_onnx_add")
                    def add_one(x):
                        return x + 1

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    x = torch.ones(4, device=device)
                    y = add_one(x)
                    torch.testing.assert_close(y, x + 1)
        finally:
            platforms._current_platform = saved_platform

    def test_cuda_onnx_platform_copies_output_to_mutated_arg(self):
        try:
            import numpy  # noqa: F401
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
            import onnxscript  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"ONNX optional dependency is unavailable: {exc}")

        import sglang.srt.platforms as platforms

        saved_platform = platforms._current_platform
        platforms._current_platform = CudaOnnxPlatform()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with envs.SGLANG_EXPORT_DIR.override(tmpdir):

                    @sgl_compile(
                        key="unit_cuda_onnx_mutate",
                        copy_output_to_arg_index=0,
                    )
                    def scale_in_place(x, scale):
                        x[:] = x * scale

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    x = torch.ones(4, device=device)
                    scale_in_place(x, torch.full((4,), 3.0, device=device))
                    torch.testing.assert_close(x, torch.full((4,), 3.0, device=device))
        finally:
            platforms._current_platform = saved_platform


if __name__ == "__main__":
    unittest.main()
