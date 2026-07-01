import importlib
import unittest

import torch

from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-b200")


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _per_token_group_quant_dequant_ref(
    x_q: torch.Tensor,
    x: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize FP8 per-token-group activations with logical UE8M0 scales."""
    *prefix, hidden_size = x.shape
    x_view = x.float().view(*prefix, hidden_size // group_size, group_size)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = _ceil_to_ue8m0(x_view.abs().amax(dim=-1).clamp(min=1e-10) / fp8_max)
    x_q_view = x_q.float().view(*prefix, hidden_size // group_size, group_size)
    return (x_q_view * scale.unsqueeze(-1)).view_as(x).float()


class TestDeepSeekV4FP8WoA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Test requires CUDA SM 100 or higher")

        try:
            cls.deep_gemm = importlib.import_module("deep_gemm")
        except ImportError as exc:
            raise unittest.SkipTest("deep_gemm is required") from exc

        try:
            cls.deepseek_v4 = importlib.import_module("sglang.srt.models.deepseek_v4")
        except ImportError as exc:
            raise unittest.SkipTest(
                "DeepSeek-V4 model dependencies are required"
            ) from exc

        if not hasattr(
            cls.deepseek_v4, "_fp8_wo_a_group_major_quant_ue8m0_for_deep_gemm"
        ):
            raise AssertionError(
                "DeepSeek-V4 FP8 wo_a DeepGEMM quant helper is missing"
            )

        try:
            fp8_utils = importlib.import_module(
                "sglang.srt.layers.quantization.fp8_utils"
            )
        except ImportError as exc:
            raise unittest.SkipTest("FP8 quantization utilities are required") from exc

        cls.quant_helper = staticmethod(
            cls.deepseek_v4._fp8_wo_a_group_major_quant_ue8m0_for_deep_gemm
        )
        cls.quant_weight_ue8m0 = staticmethod(fp8_utils.quant_weight_ue8m0)
        cls.transform_scale_ue8m0 = staticmethod(fp8_utils.transform_scale_ue8m0)
        cls.block_quant_dequant = staticmethod(fp8_utils.block_quant_dequant)

    def test_fp8_wo_a_einsum_uses_group_major_activation_scales(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        T, G, D, R = 5, 2, 256, 256
        group_size = 128
        device = torch.device("cuda")

        token_scale = torch.linspace(0.5, 1.5, T, device=device).view(T, 1, 1)
        group_scale = torch.tensor([0.25, 2.0], device=device).view(1, G, 1)
        o = (
            torch.randn(T, G, D, device=device, dtype=torch.float32)
            * token_scale
            * group_scale
            * 0.2
        ).to(torch.bfloat16)
        weight = (torch.randn(G, R, D, device=device, dtype=torch.float32) * 0.2).to(
            torch.bfloat16
        )

        o_group_major = o.transpose(0, 1).contiguous()
        o_fp8, o_s = self.quant_helper(o_group_major, group_size=group_size)

        self.assertEqual(o_fp8.shape, (G, T, D))
        self.assertEqual(o_s.shape, (G, T, 1))
        self.assertFalse(o_s.is_contiguous())

        weight_fp8, weight_s_raw = self.quant_weight_ue8m0(
            weight, weight_block_size=[128, 128]
        )
        weight_s = self.transform_scale_ue8m0(weight_s_raw, mn=R)

        output = torch.empty(T, G, R, device=device, dtype=torch.bfloat16)
        self.deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8.transpose(0, 1), o_s.transpose(0, 1)),
            (weight_fp8, weight_s),
            output,
            recipe=(1, 1, 128),
        )

        o_dequant = _per_token_group_quant_dequant_ref(
            o_fp8, o_group_major, group_size=group_size
        ).transpose(0, 1)
        weight_dequant = self.block_quant_dequant(
            weight_fp8,
            weight_s_raw,
            block_size=[128, 128],
            dtype=torch.float32,
        )
        ref = torch.einsum("tgd,grd->tgr", o_dequant, weight_dequant).to(torch.bfloat16)

        torch.testing.assert_close(output, ref, atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
