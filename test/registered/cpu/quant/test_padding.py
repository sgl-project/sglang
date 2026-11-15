import copy
import unittest

import sgl_kernel  # noqa: F401
import torch

from sglang.srt.hardware_backend.cpu.quantization.gptq_kernels import (
    GPTQIntelAMXMoEKernel,
    _normalize_moe_int4_for_amx,
)
from sglang.srt.layers.amx_utils import CPUQuantMethod
from sglang.srt.utils import cpu_has_amx_support
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import torch_naive_fused_moe

register_cpu_ci(est_time=8, suite="stage-a-test-cpu-intel")


def _make_layer_nemotron_like(
    *,
    e=128,
    k=2688,
    n_w13=1856,
    n_w2=2688,
    g_w13=42,
    g_w2=14,
    group_size=64,
    dtype=torch.bfloat16,
):
    class Layer(torch.nn.Module):
        pass

    layer = Layer()

    # GPTQ checkpoint-like raw tensors (pre-pack)
    layer.w13_qweight = torch.nn.Parameter(
        torch.randint(0, 2**31 - 1, (e, k // 8, n_w13), dtype=torch.int32),
        requires_grad=False,
    )
    layer.w13_qzeros = torch.nn.Parameter(
        torch.randint(0, 2**31 - 1, (e, g_w13, n_w13 // 8), dtype=torch.int32),
        requires_grad=False,
    )
    layer.w13_scales = torch.nn.Parameter(
        (torch.rand(e, g_w13, n_w13, dtype=torch.float32) * 1e-2 + 1e-4).to(dtype),
        requires_grad=False,
    )

    # Intentional mismatch from Nemotron INT4 case:
    # K from qweight = 928, but K from scales/qzeros = 896
    layer.w2_qweight = torch.nn.Parameter(
        torch.randint(0, 2**31 - 1, (e, 116, n_w2), dtype=torch.int32),
        requires_grad=False,
    )
    layer.w2_qzeros = torch.nn.Parameter(
        torch.randint(0, 2**31 - 1, (e, g_w2, n_w2 // 8), dtype=torch.int32),
        requires_grad=False,
    )
    layer.w2_scales = torch.nn.Parameter(
        (torch.rand(e, g_w2, n_w2, dtype=torch.float32) * 1e-2 + 1e-4).to(dtype),
        requires_grad=False,
    )

    layer.w13_g_idx = torch.nn.Parameter(
        torch.arange(k, dtype=torch.int32)
        .div(group_size, rounding_mode="floor")
        .repeat(e, 1),
        requires_grad=False,
    )
    layer.w2_g_idx = torch.nn.Parameter(
        torch.arange(116 * 8, dtype=torch.int32)
        .div(group_size, rounding_mode="floor")
        .repeat(e, 1),
        requires_grad=False,
    )

    return layer


def _unpack_gptq_qweight_3d(qweight: torch.Tensor) -> torch.Tensor:
    # [E, K/8, N] int32 -> [E, K, N] int8 (0..15)
    shifts = (
        torch.arange(8, device=qweight.device, dtype=torch.int32).view(1, 1, 8, 1) * 4
    )
    unpacked = ((qweight.unsqueeze(2) >> shifts) & 0xF).to(torch.int8)
    e, k_div8, _, n = unpacked.shape
    return unpacked.reshape(e, k_div8 * 8, n)


def _unpack_gptq_qzeros_3d(qzeros: torch.Tensor) -> torch.Tensor:
    # [E, G, N/8] int32 -> [E, G, N] int8 (1..16)
    shifts = (
        torch.arange(8, device=qzeros.device, dtype=torch.int32).view(1, 1, 1, 8) * 4
    )
    unpacked = ((qzeros.unsqueeze(-1) >> shifts) & 0xF).to(torch.int8)
    e, g, n_div8, _ = unpacked.shape
    return unpacked.reshape(e, g, n_div8 * 8) + 1


def _dequantize_gptq_moe_weight(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
) -> torch.Tensor:
    # return dense [E, N, K]
    uq_w = _unpack_gptq_qweight_3d(qweight)
    uq_z = _unpack_gptq_qzeros_3d(qzeros)

    out = []
    for e in range(uq_w.size(0)):
        idx = g_idx[e].to(torch.long).clamp_(0, scales[e].size(0) - 1)
        s = scales[e].index_select(0, idx)
        z = uq_z[e].index_select(0, idx).to(s.dtype)
        w = (uq_w[e].to(s.dtype) - z) * s
        out.append(w.transpose(0, 1).contiguous())
    return torch.stack(out, dim=0)


@unittest.skipUnless(
    cpu_has_amx_support(),
    "padding/packing test requires Intel AMX backend",
)
class TestPaddingThenPacking(unittest.TestCase):
    def test_nemotron_padding_then_packing_matches_fallback_reference(self):
        torch.manual_seed(7)

        # Build two copies from Nemotron-like problematic shapes.
        layer_raw = _make_layer_nemotron_like()
        layer_pack = copy.deepcopy(layer_raw)

        # Fallback reference path (the prior implementation logic):
        # normalize shape contract first, then dequantized native result.
        _normalize_moe_int4_for_amx(layer_raw, group_size=64)
        w13_ref = _dequantize_gptq_moe_weight(
            layer_raw.w13_qweight,
            layer_raw.w13_qzeros,
            layer_raw.w13_scales,
            layer_raw.w13_g_idx,
        )
        w2_ref = _dequantize_gptq_moe_weight(
            layer_raw.w2_qweight,
            layer_raw.w2_qzeros,
            layer_raw.w2_scales,
            layer_raw.w2_g_idx,
        )

        # New path under test: normalize + AMX prepack in process_weights_after_loading.
        quant_cfg = type("QCfg", (), {"group_size": 64})()
        kernel = GPTQIntelAMXMoEKernel(quant_cfg)
        kernel.process_weights_after_loading(layer_pack)

        # Input/routing
        m = 2
        k = w13_ref.size(2)
        e = w13_ref.size(0)
        topk = 6
        a = torch.randn(m, k, dtype=torch.bfloat16)
        score = torch.randn(m, e, dtype=torch.bfloat16)
        score_softmax = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score_softmax, topk)

        ref = torch_naive_fused_moe(
            a,
            w13_ref,
            w2_ref,
            score,
            topk,
            renormalize=False,
            activation="silu",
        )

        got = torch.ops.sgl_kernel.fused_experts_cpu(
            a,
            layer_pack.w13_qweight,
            layer_pack.w2_qweight,
            topk_weight,
            topk_ids.to(torch.int32),
            False,
            CPUQuantMethod.INT4_W4A8,
            layer_pack.w13_scales,
            layer_pack.w2_scales,
            layer_pack.w13_qzeros,
            layer_pack.w2_qzeros,
            None,
            None,
            None,
            None,
            None,
            True,
            "silu",
        )

        # Shape checks prove padded tails were discarded before AMX packing.
        self.assertEqual(tuple(layer_raw.w2_qweight.shape), (128, 112, 2688))
        self.assertEqual(tuple(layer_raw.w13_qweight.shape), (128, 336, 1792))
        self.assertEqual(tuple(layer_raw.w13_qzeros.shape), (128, 42, 224))

        # AMX pack output should be blocked 4-D layout.
        self.assertEqual(layer_pack.w2_qweight.dim(), 4)
        self.assertEqual(layer_pack.w13_qweight.dim(), 4)
        self.assertEqual(layer_pack.w2_qweight.shape[0], 128)
        self.assertEqual(layer_pack.w13_qweight.shape[0], 128)
        self.assertEqual(layer_pack.w2_qweight.shape[-1], 1152)
        self.assertEqual(layer_pack.w13_qweight.shape[-1], 1152)

        # Large-shape INT4 path can differ at bf16 accumulation order after packing.
        torch.testing.assert_close(ref.bfloat16(), got, atol=7e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
