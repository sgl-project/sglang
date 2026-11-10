"""SM70 GPU numerical tests: MXFP4 Marlin W4A16 vs torch dequant-GEMM.

Tiny fixtures only. Do not load the 476 GiB DSV4.1-Flash checkpoint.
Do not register this for GPU CI.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.marlin_utils import (
    DSV41_FLASH_HIDDEN_SIZE,
    DSV41_FLASH_MOE_INTERMEDIATE_SIZE,
    DSV41_FLASH_NUM_EXPERTS_PER_TOK,
    DSV41_FLASH_NUM_ROUTED_EXPERTS,
    SM70_MXFP4_UE8M0_FP16_EXACT_MAX,
    SM70_MXFP4_UE8M0_FP16_EXACT_MIN,
    _sm70_marlin_v100_available,
    _sm70_marlin_v100_gemm_op,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=30,
    stage="base-b",
    runner_config="1-gpu-large",
    disabled="V100 bring-up; do not register GPU CI",
)

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_RTOL = 2e-2
_ATOL = 2e-2


def _pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    even = codes[..., 0::2]
    odd = codes[..., 1::2]
    return (even | (odd << 4)).to(torch.uint8)


def dequant_mxfp4_e2m1_ue8m0(
    packed: torch.Tensor, scale_u8: torch.Tensor, group_size: int = 32
) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    sign = 1.0 - 2.0 * ((codes >> 3) & 1).to(torch.float32)
    mag = (codes & 7).to(torch.long)
    lut = torch.tensor(_E2M1, device=packed.device, dtype=torch.float32)
    values = sign * lut[mag]
    scales = torch.pow(
        torch.tensor(2.0, device=packed.device), scale_u8.to(torch.float32) - 127
    )
    scales = scales.repeat_interleave(group_size, dim=-1)
    return values * scales


def _quantize_mxfp4_weight(weight: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a small FP tensor as MXFP4 e2m1 + UE8M0 g32 (checkpoint layout)."""
    n, k = weight.shape
    reshaped = weight.to(torch.float32).reshape(n, k // group_size, group_size)
    amax = reshaped.abs().amax(dim=-1).clamp_min(2.0**-14)
    descale = amax / 6.0
    exponent = torch.ceil(torch.log2(descale)).clamp(
        SM70_MXFP4_UE8M0_FP16_EXACT_MIN - 127,
        SM70_MXFP4_UE8M0_FP16_EXACT_MAX - 127,
    )
    scale_u8 = (exponent + 127).to(torch.uint8)
    scaled = reshaped / torch.pow(
        torch.tensor(2.0, device=weight.device), exponent
    ).unsqueeze(-1)
    bounds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=weight.device,
        dtype=torch.float32,
    )
    mag = (scaled.abs().unsqueeze(-1) - bounds).gt(0).sum(dim=-1).to(torch.uint8)
    sign_bit = (scaled < 0).to(torch.uint8) << 3
    codes = (sign_bit | mag).reshape(n, k)
    packed = _pack_e2m1_codes(codes)
    return packed, scale_u8


def _random_mxfp4_linear(n: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn(n, k, device=device, dtype=torch.float32) * 0.05
    return _quantize_mxfp4_weight(weight)


class _DummyMxfp4Moe(torch.nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device,
    ):
        super().__init__()
        self.orig_dtype = torch.float16
        w13_pack = []
        w13_scale = []
        w2_pack = []
        w2_scale = []
        self._w13_fp32 = []
        self._w2_fp32 = []
        for _ in range(num_experts):
            w13, s13 = _random_mxfp4_linear(2 * intermediate_size, hidden_size, device)
            w2, s2 = _random_mxfp4_linear(hidden_size, intermediate_size, device)
            w13_pack.append(w13)
            w13_scale.append(s13)
            w2_pack.append(w2)
            w2_scale.append(s2)
            self._w13_fp32.append(dequant_mxfp4_e2m1_ue8m0(w13, s13))
            self._w2_fp32.append(dequant_mxfp4_e2m1_ue8m0(w2, s2))
        self.w13_weight = torch.nn.Parameter(
            torch.stack(w13_pack).view(torch.int8), False
        )
        self.w2_weight = torch.nn.Parameter(
            torch.stack(w2_pack).view(torch.int8), False
        )
        self.w13_weight_scale_inv = torch.nn.Parameter(
            torch.stack(w13_scale).view(torch.float8_e8m0fnu), False
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            torch.stack(w2_scale).view(torch.float8_e8m0fnu), False
        )
        self._w13_fp32 = torch.stack(self._w13_fp32)
        self._w2_fp32 = torch.stack(self._w2_fp32)

    def torch_moe(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.zeros_like(hidden, dtype=torch.float32)
        for token in range(hidden.shape[0]):
            acc = torch.zeros(hidden.shape[1], dtype=torch.float32, device=hidden.device)
            x = hidden[token].float()
            for route in range(topk_ids.shape[1]):
                expert = int(topk_ids[token, route])
                w13 = self._w13_fp32[expert]
                w2 = self._w2_fp32[expert]
                gate_up = F.linear(x, w13)
                n = w13.shape[0] // 2
                y = F.silu(gate_up[:n] if gate_up.ndim == 1 else gate_up[..., :n]) * (
                    gate_up[n:] if gate_up.ndim == 1 else gate_up[..., n:]
                )
                acc = acc + topk_weights[token, route].float() * F.linear(y, w2)
            out[token] = acc
        return out.to(torch.float16)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (7, 0)
    and _sm70_marlin_v100_available(),
    "SM70 MXFP4 Marlin GEMM requires a V100 with marlin_v100",
)
class TestMxfp4Sm70MarlinGemm(CustomTestCase):
    def test_one_linear_dequant_gemm(self):
        device = torch.device("cuda")
        torch.manual_seed(1)
        hidden, intermediate, m = 256, 64, 4
        layer = _DummyMxfp4Moe(1, hidden, intermediate, device)
        dequant = layer._w2_fp32[0]
        x = torch.randn(m, intermediate, dtype=torch.float16, device=device) * 0.1
        ref = (x.float() @ dequant.T).to(torch.float16)
        prepare_moe_mxfp4_layer_for_marlin(layer)

        gemm = _sm70_marlin_v100_gemm_op()
        self.assertIsNotNone(gemm)
        _, scalar_types = get_scalar_types()
        out = gemm(
            x,
            None,
            layer.w2_weight[0],
            None,
            layer.w2_weight_scale[0],
            None,
            None,
            None,
            None,
            None,
            layer.workspace,
            scalar_types.float4_e2m1f.id,
            m,
            hidden,
            intermediate,
            True,
            False,
            True,
            False,
        )
        torch.testing.assert_close(out, ref, rtol=_RTOL, atol=_ATOL)

    def test_one_expert_fused_moe_vs_dequant_gemm(self):
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        device = torch.device("cuda")
        torch.manual_seed(2)
        hidden, intermediate, experts, tokens, topk = 256, 64, 2, 4, 2
        layer = _DummyMxfp4Moe(experts, hidden, intermediate, device)
        x = torch.randn(tokens, hidden, dtype=torch.float16, device=device) * 0.1
        topk_ids = torch.tensor(
            [[0, 1], [1, 0], [0, 0], [1, 1]], dtype=torch.int32, device=device
        )
        topk_weights = torch.softmax(
            torch.randn(tokens, topk, device=device), dim=-1
        ).to(torch.float16)
        ref = layer.torch_moe(x, topk_ids, topk_weights)
        prepare_moe_mxfp4_layer_for_marlin(layer)
        self.assertEqual(layer.w13_weight.shape[0], experts)
        out = fused_marlin_moe(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gating_output=torch.zeros(
                tokens, experts, dtype=torch.float16, device=device
            ),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace=layer.workspace,
            num_bits=4,
        )
        torch.testing.assert_close(out, ref, rtol=_RTOL, atol=_ATOL)

    def test_official_one_expert_shape(self):
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        device = torch.device("cuda")
        torch.manual_seed(3)
        layer = _DummyMxfp4Moe(
            1, DSV41_FLASH_HIDDEN_SIZE, DSV41_FLASH_MOE_INTERMEDIATE_SIZE, device
        )
        x = torch.randn(2, DSV41_FLASH_HIDDEN_SIZE, dtype=torch.float16, device=device) * 0.1
        topk_ids = torch.zeros((2, 1), dtype=torch.int32, device=device)
        topk_weights = torch.ones((2, 1), dtype=torch.float16, device=device)
        ref = layer.torch_moe(x, topk_ids, topk_weights)
        prepare_moe_mxfp4_layer_for_marlin(layer)
        out = fused_marlin_moe(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gating_output=torch.zeros(2, 1, dtype=torch.float16, device=device),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace=layer.workspace,
            num_bits=4,
        )
        torch.testing.assert_close(out, ref, rtol=_RTOL, atol=_ATOL)

    def test_384_experts_fit_one_fused_decode(self):
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        device = torch.device("cuda")
        torch.manual_seed(4)
        hidden, intermediate = 256, 64
        layer = _DummyMxfp4Moe(
            DSV41_FLASH_NUM_ROUTED_EXPERTS, hidden, intermediate, device
        )
        x = torch.randn(1, hidden, dtype=torch.float16, device=device) * 0.1
        topk_ids = torch.arange(
            DSV41_FLASH_NUM_EXPERTS_PER_TOK, dtype=torch.int32, device=device
        ).view(1, -1)
        topk_weights = torch.full(
            (1, DSV41_FLASH_NUM_EXPERTS_PER_TOK),
            1.0 / DSV41_FLASH_NUM_EXPERTS_PER_TOK,
            dtype=torch.float16,
            device=device,
        )
        prepare_moe_mxfp4_layer_for_marlin(layer)
        self.assertEqual(layer.w13_weight.shape[0], DSV41_FLASH_NUM_ROUTED_EXPERTS)
        out = fused_marlin_moe(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gating_output=torch.zeros(
                1,
                DSV41_FLASH_NUM_ROUTED_EXPERTS,
                dtype=torch.float16,
                device=device,
            ),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace=layer.workspace,
            num_bits=4,
        )
        self.assertEqual(tuple(out.shape), (1, hidden))
        self.assertTrue(torch.isfinite(out).all())

    def test_dsv41_decode_pin_matches_auto(self):
        """Pinned CTA must match marlin_v100 auto at DSV4.1 decode shapes."""
        import os

        import sglang.kernels.ops.moe.moe_wna16_marlin as marlin_mod
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

        device = torch.device("cuda")
        torch.manual_seed(7)
        e, m, n13, k, topk, block = 6, 1, 4608, 5120, 6, 8
        _, scalar_types = get_scalar_types()
        b_q_type = scalar_types.float4_e2m1f
        w13 = torch.randint(
            -(2**31), 2**31 - 1, (e, k // 16, n13 * 2), dtype=torch.int32, device=device
        )
        s13 = torch.full((e, k // 32, n13), 127, dtype=torch.uint8, device=device).view(
            torch.float8_e8m0fnu
        )
        hidden = torch.randn(m, k, dtype=torch.float16, device=device) * 0.1
        topk_ids = torch.arange(topk, dtype=torch.int32, device=device).view(1, -1)
        topk_weights = torch.full((1, topk), 1.0 / topk, dtype=torch.float32, device=device)
        sorted_ids, expert_ids, n_post = moe_align_single_token(topk_ids, block)
        workspace = marlin_make_workspace(device, 4)
        out = torch.empty((topk, n13), dtype=torch.float16, device=device)

        def run():
            return moe_wna16_marlin_gemm(
                hidden,
                out,
                w13,
                None,
                s13,
                None,
                None,
                None,
                None,
                workspace,
                sorted_ids,
                expert_ids,
                n_post,
                topk_weights,
                moe_block_size=block,
                top_k=topk,
                mul_topk_weights=False,
                is_ep=False,
                b_q_type=b_q_type,
                size_m=m,
                size_n=n13,
                size_k=k,
                is_k_full=True,
                use_atomic_add=False,
                use_fp32_reduce=True,
                is_zp_float=False,
            )

        prev_tuning = marlin_mod._sm70_marlin_user_tuning
        marlin_mod._sm70_marlin_user_tuning = True
        for name in marlin_mod._SM70_MARLIN_MOE_ENV_NAMES:
            os.environ.pop(name, None)
        try:
            run()
            torch.cuda.synchronize()
            auto = out.detach().clone()
            marlin_mod._sm70_marlin_user_tuning = False
            marlin_mod._sm70_marlin_tuning_stage = None
            run()
            torch.cuda.synchronize()
            pinned = out.detach().clone()
            pinned_geom = os.environ.get("SM70_MARLIN_MOE_CTA_GEOMETRY")
        finally:
            marlin_mod._sm70_marlin_user_tuning = prev_tuning
            marlin_mod._sm70_marlin_tuning_stage = None
            for name in marlin_mod._SM70_MARLIN_MOE_ENV_NAMES:
                os.environ.pop(name, None)

        self.assertEqual(pinned_geom, "32x128x32x4x32x32x32")
        torch.testing.assert_close(pinned, auto, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
