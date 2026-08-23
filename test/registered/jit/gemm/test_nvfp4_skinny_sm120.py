import unittest

import torch
from flashinfer import fp4_quantize, mm_fp4

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

_SHAPES = (
    (5120, 34816),
    (17408, 5120),
    (5120, 248320),
)


def _make_inputs(m: int, k: int, n: int):
    device = "cuda"
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    input_global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    x, x_sf = fp4_quantize(x_bf16, input_global_scale)
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device).t()
    weight_sf = torch.ones((n, k // 16), dtype=torch.float8_e4m3fn, device=device).t()
    alpha = torch.tensor(0.03125, dtype=torch.float32, device=device)
    return x, weight, x_sf, weight_sf, alpha


def _reference(args):
    return mm_fp4(*args, torch.bfloat16, backend="auto")


class TestNvfp4SkinnySm120(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        props = torch.cuda.get_device_properties(0)
        if (props.major, props.minor) != (12, 0) or props.multi_processor_count < 160:
            raise unittest.SkipTest("requires an SM120 GPU with at least 160 SMs")
        cls.env_override = envs.SGLANG_ENABLE_QWEN38_FP4_SKINNY_GEMM.override(True)
        cls.env_override.__enter__()
        cls.roles_override = envs.SGLANG_QWEN38_FP4_SKINNY_GEMM_ROLES.override("all")
        cls.roles_override.__enter__()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "roles_override"):
            cls.roles_override.__exit__(None, None, None)
        if hasattr(cls, "env_override"):
            cls.env_override.__exit__(None, None, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_matches_flashinfer(self):
        from sglang.kernels.ops.gemm.nvfp4_skinny_sm120 import (
            nvfp4_skinny_sm120,
        )

        torch.manual_seed(0)
        for k, n in _SHAPES:
            for m in (1, 9, 16):
                args = _make_inputs(m, k, n)
                expected = _reference(args)
                actual = nvfp4_skinny_sm120(*args, torch.bfloat16, n)
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
                del args, expected, actual
            torch.cuda.empty_cache()

    def test_cuda_graph_replay(self):
        from sglang.kernels.ops.gemm.nvfp4_skinny_sm120 import (
            nvfp4_skinny_sm120,
        )

        k, n = _SHAPES[1]
        args = _make_inputs(9, k, n)
        expected = _reference(args)
        nvfp4_skinny_sm120(*args, torch.bfloat16, n)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = nvfp4_skinny_sm120(*args, torch.bfloat16, n)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)

    def test_safe_roles_keep_down_verification_on_flashinfer(self):
        from sglang.kernels.ops.gemm.nvfp4_skinny_sm120 import (
            use_nvfp4_skinny_sm120,
        )

        safe_roles = "gate_up,down_m1,lm_head"
        with envs.SGLANG_QWEN38_FP4_SKINNY_GEMM_ROLES.override(safe_roles):
            for m, k, n, expected in (
                (1, 17408, 5120, True),
                (9, 17408, 5120, False),
                (9, 5120, 34816, True),
            ):
                args = _make_inputs(m, k, n)
                self.assertEqual(
                    use_nvfp4_skinny_sm120(*args, torch.bfloat16, n), expected
                )
                del args
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
