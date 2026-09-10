"""MI35X correctness for GLM-5.3-Flash KDA PTPC projections."""

import unittest

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.glm5_next import (
    GLM53_KDA_PTPC_BF16_MAX_M,
    Glm5NextLinearAttention,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "requires one gfx950 GPU",
)
class TestGLM53KDAPTPC(CustomTestCase):
    TP_SHAPES = {
        4: {"qkv_proj": (6144, 4096)},
        8: {"qkv_proj": (3072, 4096)},
    }

    def _run_shape(self, module_name: str, m: int, n: int, k: int):
        from aiter.tuned_gemm import tgemm

        generator = torch.Generator(device="cuda")
        generator.manual_seed(m + n + k)
        x = (
            torch.randn(
                m,
                k,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        weight = (
            torch.randn(
                n,
                k,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        layer = torch.nn.Module()
        parameter = torch.nn.Parameter(weight, requires_grad=False)
        layer.register_parameter("weight", parameter)
        layer._glm53_kda_ptpc_module = module_name
        layer._fp8_ptpc_bf16_max_m = GLM53_KDA_PTPC_BF16_MAX_M[module_name]
        method = UnquantizedLinearMethod()
        method._repack_bf16_to_fp8_ptpc(layer)

        expected = tgemm.mm(x, parameter, otype=torch.bfloat16)
        ptpc_input = Glm5NextLinearAttention._maybe_quantize_ptpc_input(layer, x)
        actual = method.apply(layer, ptpc_input)
        repeated = method.apply(layer, ptpc_input)

        if m <= GLM53_KDA_PTPC_BF16_MAX_M[module_name]:
            self.assertIs(ptpc_input, x)
            cosine = torch.nn.functional.cosine_similarity(
                actual.float().flatten(), expected.float().flatten(), dim=0
            )
            mean_abs = (actual.float() - expected.float()).abs().mean()
            self.assertGreater(cosine.item(), 0.999)
            self.assertLess(mean_abs.item(), 0.002)
        else:
            self.assertIsInstance(ptpc_input, tuple)
            cosine = torch.nn.functional.cosine_similarity(
                actual.float().flatten(), expected.float().flatten(), dim=0
            )
            mean_abs = (actual.float() - expected.float()).abs().mean()
            self.assertGreater(cosine.item(), 0.995)
            self.assertLess(mean_abs.item(), 0.01)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, repeated, atol=0.002, rtol=0.01)
        self.assertIs(layer.weight, parameter)
        self.assertIn("_fp8_ptpc_weight", dict(layer.named_buffers()))
        self.assertNotIn("_fp8_ptpc_weight", layer.state_dict())

        del actual, expected, layer, parameter, ptpc_input, repeated, weight, x
        torch.cuda.empty_cache()

    def test_tp4_and_tp8_projection_shapes_and_boundaries(self):
        for tp, shapes in self.TP_SHAPES.items():
            for module_name, (n, k) in shapes.items():
                threshold = GLM53_KDA_PTPC_BF16_MAX_M[module_name]
                m_values = sorted(
                    {
                        1,
                        8,
                        17,
                        threshold,
                        threshold + 1,
                        8192,
                        16384,
                        131072,
                    }
                )
                for m in m_values:
                    with self.subTest(tp=tp, module=module_name, m=m):
                        self._run_shape(module_name, m, n, k)


if __name__ == "__main__":
    unittest.main(verbosity=3)
