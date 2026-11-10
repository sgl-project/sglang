from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.quantization.awq.awq import AWQConfig
from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import (
    AWQLinearKernel,
    can_use_sm70_turbomind_awq,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="TurboMind SM70 AWQ kernels require an NVIDIA V100",
)


class _DummyAWQLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        k, n, group_size = 1536, 5120, 128
        self.prefix = "model.layers.0.self_attn.o_proj"
        self.tp_size = 4
        self.qweight = torch.nn.Parameter(
            torch.randint(
                -(2**31),
                2**31 - 1,
                (k, n // 8),
                dtype=torch.int32,
                device="cuda",
            ),
            requires_grad=False,
        )
        self.qzeros = torch.nn.Parameter(
            torch.randint(
                -(2**31),
                2**31 - 1,
                (k // group_size, n // 8),
                dtype=torch.int32,
                device="cuda",
            ),
            requires_grad=False,
        )
        self.scales = torch.nn.Parameter(
            torch.rand(
                k // group_size,
                n,
                dtype=torch.float16,
                device="cuda",
            ).mul_(0.02),
            requires_grad=False,
        )


def test_compact_decode_and_bounded_exact_prefill_match_dense_weight():
    torch.manual_seed(31)
    assert can_use_sm70_turbomind_awq()
    layer = _DummyAWQLinear()
    kernel = AWQLinearKernel(
        AWQConfig(weight_bits=4, group_size=128, zero_point=True)
    )
    kernel.process_weights_after_loading(layer)

    assert layer._awq_sm70_prepared
    assert layer.qweight.numel() == 0
    assert layer.qzeros.numel() == 0
    assert layer.scales.numel() == 0
    workspace = layer._awq_sm70_prefill_dense_workspace
    assert workspace.numel() * workspace.element_size() == 85 * 1024 * 1024
    dense = workspace[: 1536 * 5120].view(1536, 5120)
    torch.ops.sglang_sm70_turbomind.awq_dequantize_out(
        dense,
        layer._awq_sm70_weight,
        layer._awq_sm70_scales,
        128,
    )

    decode_input = torch.randn(
        1, 1536, dtype=torch.float16, device="cuda"
    ).mul_(0.1)
    decode = kernel.apply(layer, decode_input)
    dense_decode = torch.mm(decode_input, dense)
    torch.testing.assert_close(decode, dense_decode, rtol=2e-3, atol=5e-4)

    prefill_input = torch.randn(
        4096, 1536, dtype=torch.float16, device="cuda"
    ).mul_(0.1)
    prefill = kernel.apply(layer, prefill_input)
    dense_prefill = torch.mm(prefill_input, dense)
    torch.testing.assert_close(prefill, dense_prefill, rtol=0, atol=0)
