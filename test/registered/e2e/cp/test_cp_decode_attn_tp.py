"""CP weight slicing must preserve Blackwell DeepGEMM's packed-scale layout."""

import unittest

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="4-gpu-b200")


class TestCpDecodeAttnTpScales(CustomTestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
        "Packed UE8M0 DeepGEMM requires SM100/SM103",
    )
    def test_packed_scale_shards(self):
        """Column slices kept the full N stride; row slices lost MN-major order.

        Both broke GLM-5.3 CP8 decode-TP startup. Compare real GEMMs against
        independently packed per-shard scales, including cached reactivation.
        """
        import deep_gemm
        from deep_gemm.utils.layout import (
            get_mn_major_tma_aligned_packed_ue8m0_tensor,
        )

        from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
        from sglang.srt.layers.quantization.fp8_utils import transform_scale_ue8m0

        torch.manual_seed(42)
        # GLM-5.3 q_b_proj and o_proj shapes. The latter also exercises K slicing.
        for dim, n, k in [(0, 16384, 2048), (1, 6144, 16384)]:
            weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
            block_scale = torch.exp2(
                torch.randint(-3, 3, (n // 128, k // 128), device="cuda").float()
            )
            packed = transform_scale_ue8m0(block_scale, mn=n)
            for rank in [0, 3, 7]:
                with self.subTest(dim=dim, rank=rank):
                    with get_parallel().override(
                        enable_cp_decode_attn_tp=True,
                        attn_cp_size=8,
                        attn_cp_rank=rank,
                    ):
                        ctx = CpDecodeAttnTpContext()
                    layer = torch.nn.Module()
                    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                    layer.weight_scale_inv = torch.nn.Parameter(
                        packed, requires_grad=False
                    )
                    attrs = ["weight", "weight_scale_inv"]
                    original = {name: getattr(layer, name).data for name in attrs}
                    for name in attrs:
                        ctx._activate(layer, name, dim)
                    shard_n, shard_k = layer.weight.shape
                    scale_chunk = block_scale.shape[dim] // 8
                    reference_scale = transform_scale_ue8m0(
                        block_scale.narrow(dim, rank * scale_chunk, scale_chunk),
                        mn=shard_n,
                    )
                    torch.testing.assert_close(
                        layer.weight_scale_inv, reference_scale, rtol=0, atol=0
                    )
                    a = torch.randn(16, shard_k, device="cuda").to(torch.float8_e4m3fn)
                    a_scale = get_mn_major_tma_aligned_packed_ue8m0_tensor(
                        torch.ones(16, shard_k // 128, device="cuda")
                    )
                    expected = torch.empty(
                        16, shard_n, device="cuda", dtype=torch.bfloat16
                    )
                    actual = torch.empty_like(expected)
                    deep_gemm.fp8_gemm_nt(
                        (a, a_scale),
                        (layer.weight, reference_scale),
                        expected,
                        disable_ue8m0_cast=True,
                    )
                    deep_gemm.fp8_gemm_nt(
                        (a, a_scale),
                        (layer.weight, layer.weight_scale_inv),
                        actual,
                        disable_ue8m0_cast=True,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    pointers = {name: getattr(layer, name).data_ptr() for name in attrs}
                    for name in attrs:
                        ctx._restore(layer, name)
                        self.assertEqual(
                            getattr(layer, name).data_ptr(), original[name].data_ptr()
                        )
                        self.assertEqual(
                            getattr(layer, name).stride(), original[name].stride()
                        )
                        ctx._activate(layer, name, dim)
                        self.assertEqual(
                            getattr(layer, name).data_ptr(), pointers[name]
                        )
                        ctx._restore(layer, name)
                    torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
