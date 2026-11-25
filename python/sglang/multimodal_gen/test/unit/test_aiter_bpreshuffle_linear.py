# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 QKV / gate+up through AITER's pre-shuffled-B asm GEMM
(SGLANG_DIFFUSION_H3_AITER_BPRESHUFFLE)."""

import sys
import types
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.layers import linear
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTBlock


def _is_gfx950() -> bool:
    return (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    )


requires_gfx950 = pytest.mark.skipif(not _is_gfx950(), reason="needs a ROCm gfx950 GPU")


class _Layer(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)


def test_flagged_layer_dispatches_to_the_preshuffled_gemm():
    layer = _Layer(torch.randn(8, 4))
    x = torch.randn(3, 4)
    method = UnquantizedLinearMethod()
    with patch.object(linear, "apply_aiter_bpreshuffle_linear") as asm:
        torch.testing.assert_close(method.apply(layer, x), F.linear(x, layer.weight))
        asm.assert_not_called()
        layer.aiter_bpreshuffle_asm = True
        method.apply(layer, x)
        asm.assert_called_once_with(layer, x, None)


@pytest.fixture
def fake_aiter():
    calls = {"shuffle": [], "gemm": []}

    def shuffle_weight(w, layout):
        calls["shuffle"].append(layout)
        return w.flip(0).contiguous()

    def gemm_a16w16_asm(a, b, out, bias, split_k, kernel_name, bpreshuffle):
        calls["gemm"].append(
            dict(
                a_shape=tuple(a.shape),
                a_contiguous=a.is_contiguous(),
                b=b,
                bias=bias,
                split_k=split_k,
                kernel_name=kernel_name,
                bpreshuffle=bpreshuffle,
            )
        )
        out.copy_(F.linear(a, b, bias))
        return out

    aiter = types.ModuleType("aiter")
    ops = types.ModuleType("aiter.ops")
    gemm = types.ModuleType("aiter.ops.gemm_op_a16w16")
    gemm.gemm_a16w16_asm = gemm_a16w16_asm
    shuffle = types.ModuleType("aiter.ops.shuffle")
    shuffle.shuffle_weight = shuffle_weight
    aiter.ops, ops.gemm_op_a16w16, ops.shuffle = ops, gemm, shuffle
    with patch.dict(
        sys.modules,
        {
            "aiter": aiter,
            "aiter.ops": ops,
            "aiter.ops.gemm_op_a16w16": gemm,
            "aiter.ops.shuffle": shuffle,
        },
    ):
        yield calls


def test_weight_is_shuffled_once_and_feeds_the_asm_kernel(fake_aiter):
    weight = torch.randn(8, 4)
    layer = _Layer(weight.clone())
    bias = torch.randn(8)
    # a non-contiguous 3D activation, as a transposed view
    x = torch.randn(4, 3, 2).permute(2, 1, 0)
    for _ in range(2):
        out = linear.apply_aiter_bpreshuffle_linear(layer, x, bias)
        assert out.shape == (2, 3, 8)
        torch.testing.assert_close(out, F.linear(x, weight.flip(0), bias))
    assert fake_aiter["shuffle"] == [(16, 16)]
    assert layer._aiter_weight_shuffled
    assert torch.equal(layer.weight, weight.flip(0))
    for call in fake_aiter["gemm"]:
        assert call["a_shape"] == (6, 4) and call["a_contiguous"]
        assert call["b"] is layer.weight
        assert call["bias"] is bias
        assert call["split_k"] is None
        assert call["kernel_name"] == linear._AITER_BPRESHUFFLE_KERNEL
        assert call["bpreshuffle"] is True


_ARCH = MiniMaxH3DiTArchConfig(num_layers=1, hidden_size=4, time_embed_dim=3)


class _Linear(nn.Module):
    def __init__(self, quant_method, params_dtype):
        super().__init__()
        self.quant_method = quant_method
        self.params_dtype = params_dtype


def _stubs(quant_method, params_dtype):
    class Attention(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.qkv_proj = _Linear(quant_method, params_dtype)

    class MLP(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.fc1 = _Linear(quant_method, params_dtype)

    return Attention, MLP


@pytest.mark.parametrize(
    "enabled, gfx950, quantized, dtype, expected",
    [
        (True, True, False, torch.bfloat16, True),
        (False, True, False, torch.bfloat16, False),
        (True, False, False, torch.bfloat16, False),
        (True, True, True, torch.bfloat16, False),
        (True, True, False, torch.float16, False),
    ],
)
def test_block_flags_qkv_and_gate_up_only_on_gfx950_bf16_unquantized(
    monkeypatch, enabled, gfx950, quantized, dtype, expected
):
    quant_method = object() if quantized else UnquantizedLinearMethod()
    attention, mlp = _stubs(quant_method, dtype)
    monkeypatch.setattr(h3, "MiniMaxH3Attention", attention)
    monkeypatch.setattr(h3, "MiniMaxH3MLP", mlp)
    monkeypatch.setattr(h3, "_H3_AITER_BPRESHUFFLE", enabled)
    monkeypatch.setattr(h3, "_is_gfx950", lambda: gfx950)
    block = MiniMaxH3DiTBlock(_ARCH, None, prefix="blocks.0", use_adaln_cache=True)
    for layer in (block.attn.qkv_proj, block.mlp.fc1):
        assert getattr(layer, "aiter_bpreshuffle_asm", False) is expected


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU answer")
def test_gfx950_probe_is_false_without_a_gpu():
    assert h3._is_gfx950() is False


@requires_gfx950
@torch.inference_mode()
@pytest.mark.parametrize(
    "n", [21504, 28672], ids=["qkv", "gate_up"]
)  # 3 x 56 x 128 heads; 2 x 14336 FFN
def test_preshuffled_gemm_is_bit_identical_to_f_linear_on_gfx950(n):
    pytest.importorskip("aiter", reason="AITer is a ROCm-only dependency")
    m, k = 9440, 5376
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = (torch.randn(n, k, device="cuda", generator=generator) * k**-0.5).to(
        torch.bfloat16
    )
    x = torch.randn(m, k, device="cuda", generator=generator).to(torch.bfloat16)
    expected = F.linear(x, weight)
    layer = _Layer(weight.clone())
    layer.aiter_bpreshuffle_asm = True
    method = UnquantizedLinearMethod()
    first = method.apply(layer, x)
    second = method.apply(layer, x)
    assert layer._aiter_weight_shuffled
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)
