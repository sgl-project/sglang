"""
python3 -m unittest test_moe_ld_padding.py
"""

import unittest
import unittest.mock

import torch

from sglang.srt.layers.moe.utils import (
    XPU_MOE_LD_PADDING_BYTES,
    xpu_moe_ld_padding_elems,
)
from sglang.srt.layers.quantization.unquant import _empty_xpu_moe_expert_weight
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")


class TestXpuMoeLdPadding(CustomTestCase):
    def test_padding_selects_aliasing_shapes(self):
        # bf16: row bytes = 2 * K. Aliasing when row bytes is a multiple of
        # 2048 with an odd cofactor >= 3.
        pad = XPU_MOE_LD_PADDING_BYTES // 2
        for k in (3072, 5120, 6144, 7168, 14336):
            self.assertEqual(xpu_moe_ld_padding_elems(k, 2), pad, f"K={k}")
        # Pure powers of two are already well distributed, so are shapes whose
        # row size is not a multiple of 2048.
        for k in (1024, 2048, 4096, 8192, 1536, 2880):
            self.assertEqual(xpu_moe_ld_padding_elems(k, 2), 0, f"K={k}")

    def test_padding_scales_with_itemsize(self):
        # The pad is a fixed byte count, so the element count scales inversely
        # with itemsize, and the aliasing test is on bytes not elements.
        self.assertEqual(
            xpu_moe_ld_padding_elems(3072, 4), XPU_MOE_LD_PADDING_BYTES // 4
        )
        self.assertEqual(xpu_moe_ld_padding_elems(6144, 1), XPU_MOE_LD_PADDING_BYTES)
        # 3072 bytes is not a multiple of 2048, so fp8/int8 K=3072 is fine.
        self.assertEqual(xpu_moe_ld_padding_elems(3072, 1), 0)

    def test_allocation_keeps_shape_and_pads_stride(self):
        E, N, K = 4, 64, 3072
        pad = xpu_moe_ld_padding_elems(K, 2)
        self.assertGreater(pad, 0)

        padded = _empty_xpu_moe_expert_weight(E, N, K, torch.bfloat16)
        plain = torch.empty(E, N, K, dtype=torch.bfloat16)

        # Logical shape is identical -- this is what keeps the weight loader,
        # which indexes purely by shape, working unchanged.
        self.assertEqual(padded.shape, plain.shape)
        self.assertEqual(padded.stride(1), K + pad)
        self.assertFalse(padded.is_contiguous())
        self.assertTrue(plain.is_contiguous())

        # A non-aliasing K allocates normally even on the XPU path.
        unpadded = _empty_xpu_moe_expert_weight(E, N, 1024, torch.bfloat16)
        self.assertTrue(unpadded.is_contiguous())

    def test_only_pads_weights_that_land_on_xpu(self):
        # SGLANG_USE_SGL_XPU only says an XPU exists on the machine; the weights
        # can still be built for CPU/CUDA. create_weights takes no device
        # argument, so the gate reads the ambient device context. Padding a
        # non-XPU weight would make it non-contiguous for no benefit.
        from sglang.srt.layers.moe import MoeRunnerConfig
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.moe_runner_config = MoeRunnerConfig(activation="silu")
                self.moe_runner_config.is_gated = True

        def build(device, use_triton_kernels=False):
            method = UnquantizedFusedMoEMethod(use_triton_kernels=use_triton_kernels)
            layer = _Layer()
            with torch.device(device):
                method.create_weights(
                    layer=layer,
                    num_experts=8,
                    hidden_size=3072,
                    intermediate_size_per_partition=3072,
                    params_dtype=torch.bfloat16,
                    with_bias=False,
                )
            return layer.w13_weight, layer.w2_weight

        with unittest.mock.patch(
            "sglang.srt.layers.quantization.unquant.use_intel_xpu_backend",
            return_value=True,
        ):
            # Env var on but building for CPU -> must stay contiguous.
            w13_cpu, w2_cpu = build("cpu")
            self.assertTrue(w13_cpu.is_contiguous())
            self.assertTrue(w2_cpu.is_contiguous())

            if torch.xpu.is_available():
                w13_xpu, _ = build("xpu")
                self.assertFalse(w13_xpu.is_contiguous())
                # The Triton path stores B transposed and ignores row stride.
                w13_triton, _ = build("xpu", use_triton_kernels=True)
                self.assertTrue(w13_triton.is_contiguous())

        # Backend off entirely -> never padded, even on XPU.
        with unittest.mock.patch(
            "sglang.srt.layers.quantization.unquant.use_intel_xpu_backend",
            return_value=False,
        ):
            device = "xpu" if torch.xpu.is_available() else "cpu"
            w13, w2 = build(device)
            self.assertTrue(w13.is_contiguous())
            self.assertTrue(w2.is_contiguous())

    def test_loader_style_copy_into_padded_view(self):
        # Mirrors _load_w13 / _load_w2: narrow the destination along a dim and
        # copy_ the checkpoint slice in. Must be exact despite the row gaps.
        E, N, K = 4, 64, 3072
        dst = _empty_xpu_moe_expert_weight(E, N, K, torch.bfloat16)
        dst.zero_()
        ref = torch.empty(E, N, K, dtype=torch.bfloat16).normal_()
        half = N // 2
        for e in range(E):
            dst[e].narrow(0, 0, half).copy_(ref[e].narrow(0, 0, half))
            dst[e].narrow(0, half, half).copy_(ref[e].narrow(0, half, half))
        self.assertTrue(torch.equal(dst, ref))
        # Still a padded view after the copies.
        self.assertEqual(dst.stride(1), K + xpu_moe_ld_padding_elems(K, 2))


@unittest.skipUnless(
    torch.xpu.is_available(), "sgl-kernel-xpu grouped GEMM requires an XPU"
)
class TestXpuMoePaddedWeightsNumerics(CustomTestCase):
    """The Xe20 grouped GEMM reads B's row stride from the tensor, so padded
    weights must give bit-identical results to contiguous ones."""

    def _run(self, hidden, inter, num_tokens, num_experts=8, topk=2):
        from sgl_kernel import fused_experts

        dtype, dev = torch.bfloat16, "xpu"
        torch.manual_seed(0)
        x = torch.empty(num_tokens, hidden, dtype=dtype, device=dev).normal_(0, 0.02)
        gate = torch.randn(num_tokens, num_experts, device=dev, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(torch.softmax(gate, -1), topk, -1)
        topk_weights = topk_weights.to(dtype)

        def alloc(n_dim, k_dim, pad):
            with torch.device(dev):
                if pad:
                    return _empty_xpu_moe_expert_weight(
                        num_experts, n_dim, k_dim, dtype
                    )
                return torch.empty(num_experts, n_dim, k_dim, dtype=dtype)

        w13 = alloc(2 * inter, hidden, False).normal_(0, 0.02)
        w2 = alloc(hidden, inter, False).normal_(0, 0.02)
        w13_pad = alloc(2 * inter, hidden, True)
        w2_pad = alloc(hidden, inter, True)
        w13_pad.copy_(w13)
        w2_pad.copy_(w2)

        out = fused_experts(x, w13, w2, topk_weights, topk_ids)
        out_pad = fused_experts(x, w13_pad, w2_pad, topk_weights, topk_ids)
        torch.xpu.synchronize()
        self.assertTrue(
            torch.equal(out, out_pad),
            f"padded weights changed the result for hidden={hidden} inter={inter}",
        )

    def test_padded_weights_bitwise_identical(self):
        for hidden, inter in ((3072, 3072), (7168, 1024), (2880, 2880)):
            for num_tokens in (64, 256):
                with self.subTest(hidden=hidden, inter=inter, num_tokens=num_tokens):
                    self._run(hidden, inter, num_tokens)


if __name__ == "__main__":
    unittest.main()
