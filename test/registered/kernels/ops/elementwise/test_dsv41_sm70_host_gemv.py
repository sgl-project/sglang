"""Host MXFP4 GEMV vs SM70 GPU decode at DSV4.1 Flash shapes.

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
    est_time=90,
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


def _pack_layer(device: torch.device, n_experts: int = 2):
    hidden = DSV41_FLASH_HIDDEN_SIZE
    intermediate = DSV41_FLASH_MOE_INTERMEDIATE_SIZE
    layer = _DummyMxfp4Moe(n_experts, hidden, intermediate, device)
    prepare_moe_mxfp4_layer_for_marlin(layer)
    return layer


@unittest.skipUnless(_sm70(), "SM70 + marlin_v100 required")
class TestSm70Dsv41HostGemv(CustomTestCase):
    @classmethod
    def tearDownClass(cls):
        try:
            from sglang.kernels.ops.moe.sm70_dsv41_spill_host_gemv import host_gemv_stop

            host_gemv_stop()
        except Exception:
            pass
        super().tearDownClass()

    def test_cpu_expert_matches_gpu_gemv(self):
        from sglang.kernels.ops.moe.sm70_dsv41_mxfp4_moe_decode import (
            sm70_dsv41_mxfp4_moe_decode,
        )
        from sglang.kernels.ops.moe.sm70_dsv41_spill_host_gemv import (
            host_mxfp4_moe_expert,
        )

        device = torch.device("cuda")
        torch.manual_seed(11)
        layer = _pack_layer(device, 1)
        hidden = DSV41_FLASH_HIDDEN_SIZE
        x = torch.randn(1, hidden, dtype=torch.float16, device=device) * 0.1
        topk_ids = torch.tensor([[0, -1, -1, -1, -1, -1]], dtype=torch.int32, device=device)
        topk_weights = torch.tensor(
            [[0.7, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        ref = sm70_dsv41_mxfp4_moe_decode(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_ids,
            topk_weights,
        )
        y = host_mxfp4_moe_expert(
            x.view(-1).cpu(),
            layer.w13_weight[0].cpu(),
            layer.w13_weight_scale[0].cpu(),
            layer.w2_weight[0].cpu(),
            layer.w2_weight_scale[0].cpu(),
            weight=0.7,
        )
        torch.testing.assert_close(y.cuda(), ref.view(-1), rtol=_RTOL, atol=_ATOL)

    def test_mailbox_request_join_matches_gpu(self):
        from sglang.kernels.ops.moe.sm70_dsv41_mxfp4_moe_decode import (
            sm70_dsv41_mxfp4_moe_decode,
        )
        from sglang.kernels.ops.moe.sm70_dsv41_spill_host_gemv import (
            host_gemv_start,
            spill_join,
            spill_request,
        )

        device = torch.device("cuda")
        torch.manual_seed(13)
        layer = _pack_layer(device, 1)
        hidden = DSV41_FLASH_HIDDEN_SIZE
        x = torch.randn(1, hidden, dtype=torch.float16, device=device) * 0.1
        topk_ids = torch.tensor([[0, -1, -1, -1, -1, -1]], dtype=torch.int32, device=device)
        topk_weights = torch.tensor(
            [[0.55, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device
        )
        ref = sm70_dsv41_mxfp4_moe_decode(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_ids,
            topk_weights,
        )

        w13_h = layer.w13_weight[0].cpu().contiguous()
        s13_h = layer.w13_weight_scale[0].cpu().contiguous()
        w2_h = layer.w2_weight[0].cpu().contiguous()
        s2_h = layer.w2_weight_scale[0].cpu().contiguous()
        self.assertTrue(host_gemv_start(2))
        map_table = torch.tensor([-1], dtype=torch.int32, device=device)
        host_map = torch.tensor([0], dtype=torch.int32, device=device)
        bases = torch.tensor(
            [
                w13_h.data_ptr(),
                s13_h.data_ptr(),
                w2_h.data_ptr(),
                s2_h.data_ptr(),
                1,
            ],
            dtype=torch.int64,
            device=device,
        )
        ids = topk_ids.clone()
        spill_request(ids, topk_weights, x, map_table, host_map, bases)
        self.assertTrue(torch.equal(ids, torch.full_like(ids, -1)))
        out = torch.zeros_like(x)
        spill_join(out)
        torch.testing.assert_close(out, ref, rtol=_RTOL, atol=_ATOL)

    def test_zero_hits_join_is_noop(self):
        from sglang.kernels.ops.moe.sm70_dsv41_spill_host_gemv import (
            host_gemv_start,
            spill_join,
            spill_request,
        )

        device = torch.device("cuda")
        torch.manual_seed(17)
        hidden = DSV41_FLASH_HIDDEN_SIZE
        x = torch.randn(1, hidden, dtype=torch.float16, device=device) * 0.1
        dummy = torch.zeros(320, 9216, dtype=torch.int32)
        dummy_s = torch.zeros(160, 4608, dtype=torch.uint8)
        dummy2 = torch.zeros(144, 10240, dtype=torch.int32)
        dummy_s2 = torch.zeros(72, 5120, dtype=torch.uint8)
        self.assertTrue(host_gemv_start(2))
        map_table = torch.tensor([0], dtype=torch.int32, device=device)
        host_map = torch.tensor([-1], dtype=torch.int32, device=device)
        bases = torch.tensor(
            [
                dummy.data_ptr(),
                dummy_s.data_ptr(),
                dummy2.data_ptr(),
                dummy_s2.data_ptr(),
                1,
            ],
            dtype=torch.int64,
            device=device,
        )
        ids = torch.tensor([[0, -1, -1, -1, -1, -1]], dtype=torch.int32, device=device)
        wts = torch.ones(1, DSV41_FLASH_NUM_EXPERTS_PER_TOK, dtype=torch.float32, device=device)
        spill_request(ids, wts, x, map_table, host_map, bases)
        self.assertEqual(int(ids[0, 0].item()), 0)
        out = torch.randn_like(x)
        before = out.clone()
        spill_join(out)
        torch.testing.assert_close(out, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
