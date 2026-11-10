"""SM70 MXFP4 decode GEMV vs Marlin at DSV4.1 Flash shapes.

Not registered for GPU CI (V100 worktree only).
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    DSV41_FLASH_HIDDEN_SIZE,
    DSV41_FLASH_MOE_INTERMEDIATE_SIZE,
    DSV41_FLASH_NUM_EXPERTS_PER_TOK,
    SM70_MXFP4_UE8M0_FP16_EXACT_MAX,
    SM70_MXFP4_UE8M0_FP16_EXACT_MIN,
    _sm70_marlin_v100_available,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=60,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

_RTOL = 2e-2
_ATOL = 2e-2


def _sm70() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (7, 0)
        and _sm70_marlin_v100_available()
    )


def _pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    even = codes[..., 0::2]
    odd = codes[..., 1::2]
    return (even | (odd << 4)).to(torch.uint8)


def _quantize_mxfp4_weight(weight: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
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
    return _pack_e2m1_codes(codes), scale_u8


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
        for _ in range(num_experts):
            w13 = torch.randn(2 * intermediate_size, hidden_size, device=device) * 0.05
            w2 = torch.randn(hidden_size, intermediate_size, device=device) * 0.05
            p13, s13 = _quantize_mxfp4_weight(w13)
            p2, s2 = _quantize_mxfp4_weight(w2)
            w13_pack.append(p13)
            w13_scale.append(s13)
            w2_pack.append(p2)
            w2_scale.append(s2)
        self.w13_weight = torch.nn.Parameter(torch.stack(w13_pack).view(torch.int8), False)
        self.w2_weight = torch.nn.Parameter(torch.stack(w2_pack).view(torch.int8), False)
        self.w13_weight_scale_inv = torch.nn.Parameter(
            torch.stack(w13_scale).view(torch.float8_e8m0fnu), False
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            torch.stack(w2_scale).view(torch.float8_e8m0fnu), False
        )


def _timed_ms(fn, n_warm: int = 5, n_iter: int = 20) -> float:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    starter.record()
    for _ in range(n_iter):
        fn()
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender) / n_iter


@unittest.skipUnless(_sm70(), "SM70 + marlin_v100 required")
class TestSm70Dsv41Mxfp4MoeDecode(CustomTestCase):
    def test_decode_shape_matches_marlin(self):
        from sglang.kernels.ops.moe.sm70_dsv41_mxfp4_moe_decode import (
            sm70_dsv41_mxfp4_moe_decode,
            sm70_dsv41_mxfp4_moe_decode_eligible,
        )
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        device = torch.device("cuda")
        torch.manual_seed(7)
        hidden = DSV41_FLASH_HIDDEN_SIZE
        intermediate = DSV41_FLASH_MOE_INTERMEDIATE_SIZE
        topk = DSV41_FLASH_NUM_EXPERTS_PER_TOK
        layer = _DummyMxfp4Moe(2, hidden, intermediate, device)
        x = torch.randn(1, hidden, dtype=torch.float16, device=device) * 0.1
        topk_ids = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.int32, device=device)
        topk_weights = torch.tensor(
            [[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]], dtype=torch.float32, device=device
        )
        prepare_moe_mxfp4_layer_for_marlin(layer)
        self.assertTrue(
            sm70_dsv41_mxfp4_moe_decode_eligible(
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                topk_ids,
                swiglu_limit=10.0,
            )
        )
        self.assertFalse(
            sm70_dsv41_mxfp4_moe_decode_eligible(
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                topk_ids,
                gemm1_alpha=1.702,
                gemm1_clamp_limit=7.0,
            )
        )
        ref = fused_marlin_moe(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gating_output=torch.zeros(1, 2, dtype=torch.float16, device=device),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace=layer.workspace,
            num_bits=4,
            clamp_limit=10.0,
        )
        out = sm70_dsv41_mxfp4_moe_decode(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_ids,
            topk_weights,
            swiglu_limit=10.0,
        )
        torch.testing.assert_close(out, ref, rtol=_RTOL, atol=_ATOL)

        topk_ids_skip = torch.tensor(
            [[0, -1, 1, -1, 0, -1]], dtype=torch.int32, device=device
        )
        ref_skip = fused_marlin_moe(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gating_output=torch.zeros(1, 2, dtype=torch.float16, device=device),
            topk_weights=topk_weights,
            topk_ids=topk_ids_skip,
            workspace=layer.workspace,
            num_bits=4,
            is_expert_parallel=True,
            clamp_limit=10.0,
        )
        out_skip = sm70_dsv41_mxfp4_moe_decode(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_ids_skip,
            topk_weights,
            swiglu_limit=10.0,
        )
        torch.testing.assert_close(out_skip, ref_skip, rtol=_RTOL, atol=_ATOL)

        gemv_ms = _timed_ms(
            lambda: sm70_dsv41_mxfp4_moe_decode(
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                topk_ids,
                topk_weights,
                swiglu_limit=10.0,
            )
        )
        marlin_ms = _timed_ms(
            lambda: fused_marlin_moe(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                gating_output=torch.zeros(1, 2, dtype=torch.float16, device=device),
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                workspace=layer.workspace,
                num_bits=4,
                clamp_limit=10.0,
            )
        )
        print(
            f"D6 GEMV {gemv_ms:.3f} ms vs Marlin {marlin_ms:.3f} ms "
            f"({marlin_ms / gemv_ms:.2f}x)",
            flush=True,
        )

    def test_eligible_rejects_prefill_m(self):
        from sglang.kernels.ops.moe.sm70_dsv41_mxfp4_moe_decode import (
            sm70_dsv41_mxfp4_moe_decode_eligible,
        )

        device = torch.device("cuda")
        m = 8
        hidden = DSV41_FLASH_HIDDEN_SIZE
        x = torch.empty(m, hidden, dtype=torch.float16, device=device)
        topk_ids = torch.zeros(
            (m, DSV41_FLASH_NUM_EXPERTS_PER_TOK), dtype=torch.int32, device=device
        )
        w13 = torch.empty(2, hidden // 16, 2 * DSV41_FLASH_MOE_INTERMEDIATE_SIZE * 2, dtype=torch.int32, device=device)
        w2 = torch.empty(2, DSV41_FLASH_MOE_INTERMEDIATE_SIZE // 16, hidden * 2, dtype=torch.int32, device=device)
        s13 = torch.empty(
            2, hidden // 32, 2 * DSV41_FLASH_MOE_INTERMEDIATE_SIZE, dtype=torch.float8_e8m0fnu, device=device
        )
        s2 = torch.empty(2, DSV41_FLASH_MOE_INTERMEDIATE_SIZE // 32, hidden, dtype=torch.float8_e8m0fnu, device=device)
        self.assertFalse(
            sm70_dsv41_mxfp4_moe_decode_eligible(x, w13, w2, s13, s2, topk_ids)
        )
        self.assertFalse(
            sm70_dsv41_mxfp4_moe_decode_eligible(
                x, w13, w2, s13, s2, topk_ids, swiglu_limit=10.0
            )
        )


if __name__ == "__main__":
    unittest.main()
