"""Tiny block-FP8 GEMM with fused UE8M0 activation quantization.

Checks the fused kernel against the two-launch DeepGEMM path it replaces
(per-token-group quantization + ``w8a8_block_fp8_matmul_deepgemm``) on every
configured shape at M = 1..16, that reruns are bitwise deterministic, that a
row's result does not depend on M, and that a caller-owned output is used.
"""

import unittest

import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

FP8_MAX = 448.0


def _quantize_fp8_blockwise_ue8m0(w: torch.Tensor, block: int = 128):
    """Checkpoint-format block-FP8 weight with power-of-two scales."""
    n, k = w.shape
    n_pad = -(-n // block) * block  # N may end in a partial 128-row block
    w = torch.nn.functional.pad(w, (0, 0, 0, n_pad - n))
    tiles = w.float().reshape(n_pad // block, block, k // block, block)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    w_fp8 = (tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return w_fp8.reshape(n_pad, k)[:n].contiguous(), scale


@unittest.skipUnless(
    get_device_sm() // 10 == 10 and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    "SM10x with UE8M0 DeepGEMM",
)
class TestTinyGemmFp8Blockwise(CustomTestCase):
    @staticmethod
    def _weights(n: int, k: int):
        from sglang.srt.layers.quantization.fp8_utils import transform_scale_ue8m0

        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        w_fp8, scale = _quantize_fp8_blockwise_ue8m0(w)
        packed = transform_scale_ue8m0(scale, mn=n)
        return w_fp8, packed

    @staticmethod
    def _reference(x, w_fp8, packed):
        from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_deepgemm
        from sglang.srt.layers.quantization.fp8_utils import (
            sglang_per_token_group_quant_fp8,
        )

        q, s = sglang_per_token_group_quant_fp8(
            x, 128, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True
        )
        return w8a8_block_fp8_matmul_deepgemm(
            q, w_fp8, s, packed, [128, 128], output_dtype=torch.bfloat16
        )

    def test_matches_deepgemm_path(self):
        from sglang.kernels.ops.gemm.tiny_gemm_fp8_blockwise import (
            _CONFIGS,
            can_use_tiny_gemm_fp8_blockwise,
            tiny_gemm_fp8_blockwise,
        )

        torch.manual_seed(3)
        for n, k in _CONFIGS:
            w_fp8, packed = self._weights(n, k)
            x16 = torch.randn((16, k), device="cuda", dtype=torch.bfloat16)
            out16 = None
            for m in range(1, 17):
                with self.subTest(shape=(n, k), m=m):
                    x = x16[:m].contiguous()
                    self.assertTrue(can_use_tiny_gemm_fp8_blockwise(x, w_fp8, packed))
                    out = tiny_gemm_fp8_blockwise(x, w_fp8, packed)
                    ref = self._reference(x, w_fp8, packed)
                    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
                    # Bitwise deterministic reruns.
                    torch.testing.assert_close(
                        tiny_gemm_fp8_blockwise(x, w_fp8, packed), out, rtol=0, atol=0
                    )
                    if m == 16:
                        out16 = out
            # A row's result does not depend on how many rows share the launch.
            for m in (1, 4, 9):
                torch.testing.assert_close(
                    tiny_gemm_fp8_blockwise(x16[:m].contiguous(), w_fp8, packed),
                    out16[:m],
                    rtol=0,
                    atol=0,
                )

    def test_caller_owned_output(self):
        from sglang.kernels.ops.gemm.tiny_gemm_fp8_blockwise import (
            tiny_gemm_fp8_blockwise,
        )

        torch.manual_seed(5)
        n, k = 2304, 1536
        w_fp8, packed = self._weights(n, k)
        x = torch.randn((8, k), device="cuda", dtype=torch.bfloat16)
        out = torch.full((8, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        ret = tiny_gemm_fp8_blockwise(x, w_fp8, packed, out)
        self.assertIs(ret, out)
        torch.testing.assert_close(out, tiny_gemm_fp8_blockwise(x, w_fp8, packed))
        with self.assertRaises(ValueError):
            tiny_gemm_fp8_blockwise(x, w_fp8, packed, out[:, : n // 2])

    def test_gate_rejects_other_shapes(self):
        from sglang.kernels.ops.gemm.tiny_gemm_fp8_blockwise import (
            can_use_tiny_gemm_fp8_blockwise,
        )

        w_fp8, packed = self._weights(1024, 1536)
        x = torch.randn((4, 1536), device="cuda", dtype=torch.bfloat16)
        self.assertFalse(can_use_tiny_gemm_fp8_blockwise(x, w_fp8, packed))
        w_fp8, packed = self._weights(2304, 1536)
        x = torch.randn((17, 1536), device="cuda", dtype=torch.bfloat16)
        self.assertFalse(can_use_tiny_gemm_fp8_blockwise(x, w_fp8, packed))


if __name__ == "__main__":
    unittest.main()
