"""The dynamic LoRA delta: bit-exact at quality "exact", inside the GEMM above it.

At "exact" the delta keeps the reference chain (GEMM, alpha scale, strength,
request scale, add) and only drops multiplications by 1.0, which are exact. At
"lossless" the second GEMM accumulates the scaled delta onto the base output and
rounds once, and a row-parallel bias moves into the base GEMM's epilogue.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.layers.lora.linear import (
    LinearWithLoRA,
    RowParallelLinearWithLoRA,
    _compute_lora_delta,
    _request_allows_lossless,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.test.quality_tier_admission import (
    assert_error_no_worse_than_reference,
)

_TP_RANK_PATCH = "sglang.multimodal_gen.runtime.layers.lora.linear.get_tp_rank"
TOKENS, IN_DIM, OUT_DIM, RANK = 256, 192, 96, 16


def _batch(quality: str, runtime_scale: float = 1.0):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(quality=quality),
        runtime_lora_scale=runtime_scale,
    )


def _activations(rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, generator=g)
    x[:, ::17] *= 8.0  # the outlier channels a DiT block carries
    return x


def _adapter(stacked: bool, dtype: torch.dtype, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    if stacked:
        a = torch.randn(3, RANK, IN_DIM, generator=g) * 0.05
        b = torch.randn(3, OUT_DIM // 3, RANK, generator=g) * 0.05
    else:
        a = torch.randn(RANK, IN_DIM, generator=g) * 0.05
        b = torch.randn(OUT_DIM, RANK, generator=g) * 0.05
    return a.to(dtype), b.to(dtype)


def _layer(a, b, alpha: int, strength: float) -> LinearWithLoRA:
    base = nn.Linear(IN_DIM, OUT_DIM, bias=True, dtype=torch.bfloat16)
    layer = LinearWithLoRA(base, lora_rank=RANK, lora_alpha=alpha)
    layer.set_lora_weights(a, b, strength=strength, merge_weights=False)
    return layer


def _reference_chain(layer, out, x, runtime_scale):
    """The dynamic delta as main computed it."""
    a, b = layer.lora_A, layer.lora_B
    delta = _compute_lora_delta(x.to(a.dtype), a, b)
    if layer.lora_alpha != layer.lora_rank:
        delta = delta * (layer.lora_alpha / layer.lora_rank)
    delta = delta * layer.strength * runtime_scale
    return out + delta.to(dtype=out.dtype)


@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize(
    "alpha, strength, runtime_scale",
    [(RANK, 1.0, 1.0), (2 * RANK, 1.0, 1.0), (RANK, 0.7, 1.0), (8, 0.7, 0.5)],
)
def test_exact_tier_keeps_the_reference_chain_bit_for_bit(
    stacked, alpha, strength, runtime_scale
):
    a, b = _adapter(stacked, torch.bfloat16)
    layer = _layer(a, b, alpha, strength)
    x = _activations(TOKENS, IN_DIM, seed=2).to(torch.bfloat16)
    with torch.inference_mode(), set_forward_context(0, None, _batch("exact")):
        out = layer.base_layer(x)
        actual = layer._add_lora_delta(out.clone(), x, runtime_scale)
        expected = _reference_chain(layer, out, x, runtime_scale)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "stacked, adapter_dtype",
    [(False, torch.bfloat16), (True, torch.bfloat16), (False, torch.float32)],
)
def test_lossless_tier_is_admitted_against_the_fp64_math(stacked, adapter_dtype):
    a, b = _adapter(stacked, adapter_dtype)
    layer = _layer(a, b, alpha=2 * RANK, strength=0.8)
    x = _activations(TOKENS, IN_DIM, seed=3).to(torch.bfloat16)
    with torch.inference_mode(), set_forward_context(0, None, _batch("lossless")):
        out = layer.base_layer(x)
        reference = out.double() + _compute_lora_delta(
            x.double(), a.double(), b.double()
        ) * (2.0 * 0.8)
        baseline = _reference_chain(layer, out, x, 1.0)
        candidate = layer._add_lora_delta(out.clone(), x, 1.0)
    assert_error_no_worse_than_reference(
        reference_fp64=reference,
        baseline=baseline,
        candidate=candidate,
        label=f"LoRA delta in the GEMM ({'stacked' if stacked else '2D'}, {adapter_dtype})",
    )


def test_the_default_quality_takes_the_lossless_path():
    assert _request_allows_lossless()  # no forward context
    with set_forward_context(0, None, SimpleNamespace(runtime_lora_scale=1.0)):
        assert _request_allows_lossless()
    with set_forward_context(0, None, _batch("exact")):
        assert not _request_allows_lossless()
    with set_forward_context(0, None, _batch("high")):
        assert _request_allows_lossless()


class _RowLinear(nn.Module):
    """The RowParallelLinear surface the LoRA wrapper reads, at TP 1."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(4)
        self.weight = nn.Parameter(
            (torch.randn(OUT_DIM, IN_DIM, generator=g) * 0.05).to(torch.bfloat16)
        )
        self.bias = nn.Parameter(torch.randn(OUT_DIM, generator=g).to(torch.bfloat16))
        self.input_is_parallel = True
        self.reduce_results = True
        self.skip_bias_add = False
        self.tp_size = 1
        self.tp_rank = 0
        self.input_size_per_partition = IN_DIM
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x, bias=None: F.linear(x, layer.weight, bias)
        )

    def forward(self, x):
        return F.linear(x, self.weight, self.bias), None


def test_row_parallel_bias_moves_into_the_gemm_only_above_exact():
    a, b = _adapter(False, torch.bfloat16)
    layer = RowParallelLinearWithLoRA(_RowLinear(), lora_rank=RANK, lora_alpha=RANK)
    layer.set_lora_weights(a, b, merge_weights=False)
    base = layer.base_layer
    x = _activations(TOKENS, IN_DIM, seed=5).to(torch.bfloat16)
    with patch(_TP_RANK_PATCH, return_value=0), torch.inference_mode():
        with set_forward_context(0, None, _batch("exact")):
            exact, _ = layer(x)
        with set_forward_context(0, None, _batch("lossless")):
            lossless, _ = layer(x)
        reference_chain = (
            _reference_chain(layer, F.linear(x, base.weight), x, 1.0) + base.bias
        )
        reference = (
            x.double() @ base.weight.double().t()
            + base.bias.double()
            + _compute_lora_delta(x.double(), a.double(), b.double())
        )
    assert torch.equal(exact, reference_chain)
    assert_error_no_worse_than_reference(
        reference_fp64=reference,
        baseline=exact,
        candidate=lossless,
        label="row-parallel LoRA with the bias in the GEMM",
    )
