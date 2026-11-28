"""SM90 block-[32, 32] FP8 GEMM. Serving dispatch is NVIDIA H20-3e only."""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_H20_3E = "NVIDIA H20-3e"


def _is_h20_3e() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_name(0) == _H20_3E


def _dequant_ref(act, weight, act_scale, weight_scale):
    m, k = act.shape
    n = weight.shape[0]
    af = (act.float().reshape(m, k // 32, 32) * act_scale.unsqueeze(-1)).reshape(m, k)
    pad = (32 - n % 32) % 32
    bf = torch.nn.functional.pad(weight.float(), (0, 0, 0, pad))
    bf = (
        bf.reshape(-1, 32, k // 32, 32) * weight_scale.float()[:, None, :, None]
    ).reshape(-1, k)[:n]
    return (af @ bf.T).to(torch.bfloat16)


def _inputs(n, k, m=1, seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    act = (torch.randint(-8, 9, (m, k), device="cuda", generator=g) / 4).to(
        torch.float8_e4m3fn
    )
    weight = (torch.randint(-8, 9, (n, k), device="cuda", generator=g) / 4).to(
        torch.float8_e4m3fn
    )
    act_scale = torch.exp2(
        torch.randint(-6, 1, (m, k // 32), device="cuda", generator=g).float()
    )
    weight_scale = torch.exp2(
        torch.randint(
            -6, 1, ((n + 31) // 32, k // 32), device="cuda", generator=g
        ).float()
    )
    return act, weight, act_scale, weight_scale


class TestFp8Block32H20Gate(CustomTestCase):
    def test_device_name_gate(self):
        from sglang.kernels.ops.gemm import fp8_block32_sm90 as mod

        for name, expected in (
            ("NVIDIA H20-3e", True),
            ("NVIDIA H20", False),
            ("NVIDIA H100 80GB HBM3", False),
            ("NVIDIA H200", False),
        ):
            with mock.patch.object(torch.cuda, "get_device_name", return_value=name):
                mod._is_h20_3e.cache_clear()
                self.assertEqual(mod._is_h20_3e(0), expected, name)
        mod._is_h20_3e.cache_clear()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_predicate_rejects_other_gpus(self):
        from sglang.kernels.ops.gemm.fp8_block32_sm90 import (
            _is_h20_3e,
            use_fp8_block32_gemm,
        )

        if torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("predicate tensor check needs an SM90 GPU")
        n, k = 64, 64
        act, weight, act_scale, weight_scale = _inputs(n, k)
        with mock.patch(
            "sglang.kernels.ops.gemm.fp8_block32_sm90._is_h20_3e", return_value=False
        ):
            self.assertFalse(
                use_fp8_block32_gemm(
                    act, weight, act_scale, weight_scale, [32, 32], torch.bfloat16
                )
            )
        _is_h20_3e.cache_clear()
        if _is_h20_3e(0):
            self.assertTrue(
                use_fp8_block32_gemm(
                    act, weight, act_scale, weight_scale, [32, 32], torch.bfloat16
                )
            )


@unittest.skipIf(
    not _is_h20_3e(), "block-32 WGMMA GEMM is enabled only on NVIDIA H20-3e"
)
class TestFp8Block32Sm90(CustomTestCase):
    def _check(self, n, k, split_k=None, m=1):
        from sglang.kernels.ops.gemm.fp8_block32_sm90 import fp8_block32_gemm

        act, weight, act_scale, weight_scale = _inputs(n, k, m=m)
        out = fp8_block32_gemm(act, weight, act_scale, weight_scale, split_k=split_k)
        ref = _dequant_ref(act, weight, act_scale, weight_scale)
        err = (out.float() - ref.float()).abs().max().item()
        self.assertEqual(tuple(out.shape), (m, n))
        self.assertFalse(torch.isnan(out).any().item(), (m, n, k, split_k))
        self.assertLessEqual(err, 0.02, (m, n, k, split_k, err))

    def test_shapes(self):
        for m, n, k, split in (
            (1, 64, 32, 1),
            (1, 64, 64, 1),
            (1, 64, 96, 1),
            (1, 100, 128, 1),
            (1, 160, 256, 2),
            (1, 512, 5120, None),
            (1, 5120, 288, None),
            (1, 128, 512, None),
            (7, 96, 64, 1),
            (8, 128, 128, 1),
            (16, 160, 256, 2),
            (32, 64, 96, 1),
            (48, 160, 128, None),
            (96, 200, 160, None),
            (128, 256, 256, None),
            (130, 288, 288, None),
            # Wide enough that the 64x128 tile is selected (M>=48 and tiles >= SMs/2).
            (256, 4096, 256, None),
        ):
            self._check(n, k, split, m)

    def test_decode_stays_on_split_k(self):
        from sglang.kernels.ops.gemm.fp8_block32_sm90 import use_fp8_block32_large_tile

        # wq_b / wo_b at decode batch 1 must not take the 64-row tile.
        self.assertFalse(use_fp8_block32_large_tile(1, 4096, 1280))
        self.assertFalse(use_fp8_block32_large_tile(1, 5120, 1024))
        self.assertFalse(use_fp8_block32_large_tile(32, 4096, 1280))
        # Prefill-sized M on the same weights does.
        self.assertTrue(use_fp8_block32_large_tile(998, 4096, 1280))
        self.assertTrue(use_fp8_block32_large_tile(998, 5120, 1024))
        self.assertTrue(use_fp8_block32_large_tile(4096, 5120, 5120))

    def test_linear_uses_kernel_for_m_gt1(self):
        from sglang.kernels.ops.gemm.fp8_block32_sm90 import use_fp8_block32_gemm
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )
        from sglang.srt.layers.quantization.fp8_utils import (
            triton_w8a8_block_fp8_linear,
        )

        n, k = 128, 256
        _, weight, _, weight_scale = _inputs(n, k, seed=1)
        x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
        q, scale = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
        self.assertTrue(
            use_fp8_block32_gemm(
                q, weight, scale, weight_scale, [32, 32], torch.bfloat16
            )
        )
        out = triton_w8a8_block_fp8_linear(
            x, weight, [32, 32], weight_scale, act_scale_ue8m0=True
        )
        ref = _dequant_ref(q, weight, scale, weight_scale)
        err = (out.float() - ref.float()).abs().max().item()
        # bf16 rounding of the dequant reference can be one ulp (0.03125)
        # on this shape; the kernel matches Triton on the same inputs.
        self.assertLessEqual(err, 0.0625, err)

        x4 = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
        q4, scale4 = sglang_per_token_group_quant_fp8(x4, 32, scale_ue8m0=True)
        self.assertTrue(
            use_fp8_block32_gemm(
                q4, weight, scale4, weight_scale, [32, 32], torch.bfloat16
            )
        )
        out4 = triton_w8a8_block_fp8_linear(
            x4, weight, [32, 32], weight_scale, act_scale_ue8m0=True
        )
        ref4 = _dequant_ref(q4, weight, scale4, weight_scale)
        rel4 = (out4.float() - ref4.float()).abs() / ref4.float().abs().clamp(min=1)
        self.assertLessEqual(rel4.max().item(), 0.02, rel4.max().item())


if __name__ == "__main__":
    unittest.main()
