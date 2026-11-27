"""Prefill-sized launches of the gated RMSNorm and the GDN gating on SM100+ must write
the bytes of the launches they replace: several one-warp row tiles a program for the
norm, all heads of a token in one program for the gating."""

import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
# Nightly sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=30, stage="nightly", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or torch.cuda.get_device_capability()[0] < 10,
    reason="the prefill launches engage on SM100+ CUDA only",
)

from sglang.kernels.ops.attention.fla import layernorm_gated as lg  # noqa: E402
from sglang.kernels.ops.attention.linear import fused_gdn_gating as gating  # noqa: E402

DEV = "cuda"
_use_prefill_tiles = lg._use_prefill_tiles
_heads_per_program = gating._heads_per_program


def _same(a, b):
    return a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _norm(*, x, w, z, tiles, monkeypatch, **kwargs):
    monkeypatch.setattr(
        lg,
        "_use_prefill_tiles",
        _use_prefill_tiles if tiles else lambda **kwargs: False,
    )
    y, mean, rstd = lg._layer_norm_fwd(
        x, w, kwargs.pop("bias", None), 1e-6, z=z, **kwargs
    )
    return y, mean, rstd


def _norm_bytes_equal(*, x, w, z, monkeypatch, **kwargs):
    stock = _norm(x=x, w=w, z=z, tiles=False, monkeypatch=monkeypatch, **kwargs)
    tiled = _norm(x=x, w=w, z=z, tiles=True, monkeypatch=monkeypatch, **kwargs)
    for s, t in zip(stock, tiled):
        assert (s is None and t is None) or _same(s, t)


@pytest.mark.parametrize(
    "tokens", get_ci_test_range([1, 64, 1023, 1024, 4097, 32768], [1023, 1024, 4097])
)
@pytest.mark.parametrize("scale", (1.0, 40.0))
def test_gated_norm_gdn_layout_bytes_equal_stock(tokens, scale, monkeypatch):
    # z is a [T, H, D] column slice of a wider projection, as the GDN layers pass it.
    heads, dim = 64, 128
    g = torch.Generator(device=DEV).manual_seed(tokens)
    x = (torch.randn((tokens * heads, dim), generator=g, device=DEV) * scale).to(
        torch.bfloat16
    )
    proj = (torch.randn((tokens, 20480), generator=g, device=DEV) * 2).to(
        torch.bfloat16
    )
    z = proj[:, 12288:].view(tokens, heads, dim)
    w = (torch.randn(dim, generator=g, device=DEV) * 0.3 + 1).to(torch.bfloat16)
    rows = tokens * heads
    engaged = _use_prefill_tiles(
        M=rows, num_warps=1, rows_per_block=lg.MAX_ROWS_PER_BLOCK, quant_heads=0
    )
    assert engaged is (rows >= lg.PREFILL_TILE_MIN_ROWS)
    _norm_bytes_equal(x=x, w=w, z=z, monkeypatch=monkeypatch, is_rms_norm=True)


@pytest.mark.parametrize("dim", get_ci_test_range([64, 128, 256], [64, 256]))
@pytest.mark.parametrize(
    "norm_before_gate,activation",
    get_ci_test_range(
        [(True, "swish"), (True, "sigmoid"), (False, "swish"), (False, "sigmoid")],
        [(True, "swish"), (False, "sigmoid")],
    ),
)
def test_gated_norm_widths_and_gates_bytes_equal_stock(
    dim, norm_before_gate, activation, monkeypatch
):
    rows = 70_000
    g = torch.Generator(device=DEV).manual_seed(dim)
    x = torch.randn((rows, dim), generator=g, device=DEV).to(torch.bfloat16)
    z = torch.randn((rows, dim), generator=g, device=DEV).to(torch.bfloat16)
    w = torch.randn(dim, generator=g, device=DEV).to(torch.bfloat16)
    _norm_bytes_equal(
        x=x,
        w=w,
        z=z,
        monkeypatch=monkeypatch,
        is_rms_norm=True,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )


@pytest.mark.parametrize("group_size", (None, 64))
def test_layer_norm_with_bias_bytes_equal_stock(group_size, monkeypatch):
    rows, dim = 70_000, 128
    g = torch.Generator(device=DEV).manual_seed(7)
    x = torch.randn((rows, dim), generator=g, device=DEV).to(torch.bfloat16)
    z = torch.randn((rows, dim), generator=g, device=DEV).to(torch.bfloat16)
    w = torch.randn(dim, generator=g, device=DEV).to(torch.bfloat16)
    bias = torch.randn(dim, generator=g, device=DEV).to(torch.bfloat16)
    _norm_bytes_equal(
        x=x,
        w=w,
        z=z,
        monkeypatch=monkeypatch,
        bias=bias,
        group_size=group_size,
        is_rms_norm=False,
    )


@pytest.mark.parametrize("tokens", get_ci_test_range([1024, 4096, 32768], [4096]))
@pytest.mark.parametrize("heads", (64, 48, 32, 16))
def test_gdn_gating_one_program_bytes_equal_stock(tokens, heads, monkeypatch):
    g = torch.Generator(device=DEV).manual_seed(tokens * heads)
    a_log = torch.randn(heads, generator=g, device=DEV)
    dt_bias = torch.randn(heads, generator=g, device=DEV)
    ab = torch.randn(tokens, 2 * heads, generator=g, device=DEV).to(torch.bfloat16)
    a, b = ab[:, :heads], ab[:, heads:]
    assert _heads_per_program(num_tokens=tokens, num_heads=heads) > 8
    monkeypatch.setattr(gating, "_heads_per_program", lambda **kwargs: 8)
    stock = gating.fused_gdn_gating(a_log, a, b, dt_bias)
    monkeypatch.setattr(gating, "_heads_per_program", _heads_per_program)
    one_program = gating.fused_gdn_gating(a_log, a, b, dt_bias)
    for s, o in zip(stock, one_program):
        assert _same(s, o)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
