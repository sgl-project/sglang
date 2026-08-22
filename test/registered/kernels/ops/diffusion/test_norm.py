"""``diffusion.norm``: GroupNorm / RMSNorm / LayerNorm and their fused epilogues.

This domain has the most implementations of any in the package (see the
selection matrix in ``sglang/kernels/ops/diffusion/README.md``), so the suite
is organized by *kernel*, and each section states which oracle it is held to:

- ``triton_group_norm_silu`` / ``apply_group_norm_silu`` -> ``F.group_norm`` +
  ``F.silu`` with a per-dtype tolerance (fp32 statistics, different reduction).
- the two-pass channels-last GroupNorm -> same oracle, plus its support
  predicates (the kernels raise on an unsupported input rather than returning
  ``None``).
- ``rmsnorm_scale`` / ``rmsnorm_tanh_residual`` -> a bf16-native reference that
  reproduces Z-Image's own norm, with a tolerance for Triton's exp-based tanh.
- the CuTe-DSL fused norm+scale/shift -> an fp32 reference chain.

The FlyDSL norms live in ``test_norm_flydsl.py``: they are ROCm gfx950-only,
so they run on a CI lane this file does not, and keeping them here dragged the
CUDA-only CuTe-DSL cases onto the AMD runner.

The *bit-exact* norms (``fused_rmsnorm_scale_shift_bitexact``,
``fused_layernorm_modulate``, ``zimage_qk_rmsnorm_native``) are exercised
through their model wrappers in ``test_model_fast_paths.py``, where the live
eager chain they must reproduce is available.
"""

import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.kernels.ops.diffusion import (
    apply_group_norm_silu,
    can_use_group_norm_silu_4d,
    can_use_wan_rmsnorm_silu,
    group_norm_silu_4d,
    rmsnorm_scale,
    rmsnorm_tanh_residual,
    triton_group_norm_silu,
    wan_rmsnorm_silu,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=70, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
EPS = 1e-5


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.bfloat16:
        return 7e-2, 2e-2
    return 3e-3, 3e-3


@pytest.fixture(autouse=True)
def cuda_setup():
    torch.cuda.manual_seed(0)


def _cl3d(shape, dtype):
    return torch.randn(shape, device=DEVICE, dtype=dtype).contiguous(
        memory_format=torch.channels_last_3d
    )


# ---------------------------------------------------------------------------
# GroupNorm + SiLU
# ---------------------------------------------------------------------------

GN_CASES = [
    pytest.param((2, 64, 32, 32), 32, id="image_2d"),
    pytest.param((1, 64, 4, 16, 16), 32, id="video_3d"),
    pytest.param((4, 128), 32, id="token_2d"),
]


def _gn_silu_reference(x, weight, bias, num_groups, eps=EPS):
    return F.silu(F.group_norm(x, num_groups, weight=weight, bias=bias, eps=eps))


@torch.no_grad()
@pytest.mark.parametrize("shape,num_groups", GN_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_group_norm_silu(shape, num_groups, dtype):
    channels = shape[1]
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    weight = torch.randn(channels, device=DEVICE, dtype=dtype)
    bias = torch.randn(channels, device=DEVICE, dtype=dtype)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(
        triton_group_norm_silu(x, weight, bias, num_groups=num_groups),
        _gn_silu_reference(x, weight, bias, num_groups),
        atol=atol,
        rtol=rtol,
    )


@torch.no_grad()
def test_triton_group_norm_silu_large_tile_bf16():
    # A tile large enough to force the chunked launch path (128 channels over
    # 20x256x256), which the small cases above never reach.
    shape, num_groups = (1, 128, 20, 256, 256), 32
    x = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(shape[1], device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(shape[1], device=DEVICE, dtype=torch.bfloat16)

    atol, rtol = _tol(torch.bfloat16)
    torch.testing.assert_close(
        triton_group_norm_silu(x, weight, bias, num_groups=num_groups),
        _gn_silu_reference(x, weight, bias, num_groups),
        atol=atol,
        rtol=rtol,
    )


@torch.no_grad()
@pytest.mark.parametrize("shape,num_groups", GN_CASES[:2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_apply_group_norm_silu_module_wrapper(shape, num_groups, dtype):
    # The nn.Module-taking wrapper must match the eager module pair it stands
    # in for, including its own guard set (affine=True, non-inplace SiLU).
    norm = nn.GroupNorm(num_groups, shape[1], eps=EPS, affine=True).to(
        device=DEVICE, dtype=dtype
    )
    activation = nn.SiLU()
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(
        apply_group_norm_silu(x, norm, activation),
        activation(norm(x)),
        atol=atol,
        rtol=rtol,
    )


@torch.no_grad()
def test_group_norm_silu_4d_channels_last_and_guards():
    gn = nn.GroupNorm(32, 128, eps=1e-6).to(DEVICE, torch.bfloat16)
    x = torch.randn(1, 128, 64, 64, device=DEVICE, dtype=torch.bfloat16).to(
        memory_format=torch.channels_last
    )
    assert can_use_group_norm_silu_4d(x, gn.weight, gn.bias, 32)
    out = group_norm_silu_4d(x, gn.weight, gn.bias, 32, 1e-6)
    assert out.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(out.float(), F.silu(gn(x)).float(), atol=0.06, rtol=0)

    # Guards: the kernel exists only for channels_last inputs with device-side
    # affine params and a non-empty spatial extent.  Each rejected case must
    # fail the predicate *and* raise if called anyway -- silently returning
    # ``None`` is what this protocol replaced.
    rejected = [
        (x.contiguous(), gn.weight, gn.bias),  # contiguous (NCHW) layout
        (x, gn.weight.cpu(), gn.bias),  # host-side affine
        (x[..., :0, :], gn.weight, gn.bias),  # empty spatial extent
    ]
    for args in rejected:
        assert not can_use_group_norm_silu_4d(*args, 32)
        with pytest.raises(ValueError):
            group_norm_silu_4d(*args, 32, 1e-6)


# ---------------------------------------------------------------------------
# BF16-native RMSNorm fusions (Z-Image / Ideogram)
# ---------------------------------------------------------------------------


def _native_bf16_rmsnorm(x, weight):
    """Z-Image's own norm: every step materialized in bf16, no fp32 carry."""
    square = (x * x).to(torch.bfloat16)
    mean_square = square.mean(dim=-1, keepdim=True).to(torch.bfloat16)
    rstd = torch.rsqrt((mean_square + EPS).to(torch.bfloat16).float()).to(
        torch.bfloat16
    )
    return ((x * rstd).to(torch.bfloat16) * weight).to(torch.bfloat16)


@pytest.mark.parametrize("shape", [(1, 32, 2560), (2, 17, 256)])
def test_rmsnorm_scale_matches_native_bf16(shape):
    batch, _, dim = shape
    x = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16)
    scale = torch.randn(batch, 1, dim, device=DEVICE, dtype=torch.bfloat16)

    actual = rmsnorm_scale(x, weight, scale, EPS)
    assert actual is not None
    torch.testing.assert_close(
        actual,
        (_native_bf16_rmsnorm(x, weight) * scale).to(torch.bfloat16),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.parametrize("shape", [(1, 32, 2560), (2, 17, 256)])
def test_rmsnorm_tanh_residual_matches_native_bf16(shape):
    batch, _, dim = shape
    x = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    gate = torch.randn(batch, 1, dim, device=DEVICE, dtype=torch.bfloat16)
    residual = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16)

    actual = rmsnorm_tanh_residual(x, gate, residual, weight, EPS)
    norm = _native_bf16_rmsnorm(x, weight)
    gated = (torch.tanh(gate.float()).to(torch.bfloat16) * norm).to(torch.bfloat16)

    assert actual is not None
    # Triton's exp-based tanh can differ slightly from torch.tanh in bf16.
    torch.testing.assert_close(
        actual, (residual + gated).to(torch.bfloat16), atol=4e-2, rtol=2e-2
    )


@pytest.mark.parametrize("on_host", [True, False])
def test_native_bf16_rmsnorm_rejects_unsupported_inputs(on_host):
    # Host tensors and a hidden size past the kernel limit are both outside
    # the contract; these entry points signal that by returning None (they
    # are internal fast-path probes, not public predicate+kernel pairs).
    device = "cpu" if on_host else DEVICE
    dim = 16 if on_host else 8448
    x = torch.randn(2, 3, dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(dim, dtype=torch.bfloat16, device=device)
    modulation = torch.randn(2, 1, dim, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(x)

    assert rmsnorm_scale(x, weight, modulation, EPS) is None
    assert rmsnorm_tanh_residual(x, modulation, residual, weight, EPS) is None
    if on_host:
        # Mismatched trailing dims are rejected too.
        assert rmsnorm_scale(x, weight[:-1], modulation, EPS) is None
        assert (
            rmsnorm_tanh_residual(x, modulation, residual[..., :-1], weight, EPS)
            is None
        )


# ---------------------------------------------------------------------------
# Wan VAE channels_last_3d RMSNorm + SiLU
# ---------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize(
    "x_dtype,affine_dtype,atol,rtol",
    [
        (torch.float32, torch.float32, 1e-5, 1e-5),  # FastWan2.2 fp32 decode
        (torch.bfloat16, torch.float32, 1.5e-1, 3e-2),  # Wan2.1 bf16 autocast
    ],
)
def test_wan_rmsnorm_silu_numerics(x_dtype, affine_dtype, atol, rtol):
    x = _cl3d((1, 96, 3, 10, 14), x_dtype)
    gamma = torch.randn((96, 1, 1, 1), device=DEVICE, dtype=affine_dtype)
    for bias in (None, torch.randn_like(gamma)):
        expected = F.silu(
            F.normalize(x, dim=1) * 96**0.5 * gamma + (0 if bias is None else bias)
        )
        actual = wan_rmsnorm_silu(x, gamma, bias)
        assert actual.dtype == expected.dtype
        # The kernel must preserve the channels_last_3d layout; a relayout
        # here would undo the reason the decoder runs in that format.
        assert actual.stride() == x.stride()
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
def test_wan_rmsnorm_silu_rejects_empty_input():
    x = torch.empty(1, 96, 0, 2, 2, device=DEVICE, dtype=torch.bfloat16).to(
        memory_format=torch.channels_last_3d
    )
    gamma = torch.ones(96, 1, 1, 1, device=DEVICE, dtype=torch.bfloat16)
    assert not can_use_wan_rmsnorm_silu(x, gamma, None)
    with pytest.raises(ValueError):
        wan_rmsnorm_silu(x, gamma)


# ---------------------------------------------------------------------------
# CuTe-DSL fused (residual +) norm + scale/shift
# ---------------------------------------------------------------------------

SHAPE_MAP = {
    "1": lambda B, S, F_, D: (1,),
    "D": lambda B, S, F_, D: (D,),
    "1D": lambda B, S, F_, D: (1, D),
    "BD": lambda B, S, F_, D: (B, D),
    "11D": lambda B, S, F_, D: (1, 1, D),
    "B1D": lambda B, S, F_, D: (B, 1, D),
    "1SD": lambda B, S, F_, D: (1, S, D),
    "BSD": lambda B, S, F_, D: (B, S, D),
    "BF1D": lambda B, S, F_, D: (B, F_, 1, D),
}
# (B, S, F, D)
CUTE_SHAPES = [
    (1, 115200, 1, 3072),  # HunyuanVideo
    (1, 32760, 1, 1536),  # Wan
    (1, 6, 1, 3072),  # Qwen-Image
    (1, 1024, 8, 3072),
    (4, 512, 16, 3072),
]
NORM_TYPES = ["layer", "rms"]
AFFINE_MODES = ["D", "NAT"]
INDEX_MODES = ["BSD", "1", "1SD", "BD", "B1D", "D", "1D", "11D", "BF1D"]


def _import_cutedsl():
    """Import the CuTe-DSL entry points, skipping when the backend is absent.

    This file is registered on the AMD lane for its FlyDSL section, but the
    CuTe-DSL norms need cuda-python and CUTLASS, which the ROCm image does not
    ship.  Guarded per test rather than by dropping this file from the AMD
    lane, so the Triton and FlyDSL sections keep running there.
    """
    try:
        from sglang.kernels.ops.diffusion import (
            fused_norm_scale_shift,
            fused_scale_residual_norm_scale_shift,
        )
    except ImportError as exc:  # pragma: no cover - platform-dependent
        pytest.skip(f"CuTe-DSL backend unavailable: {exc}")
    return fused_norm_scale_shift, fused_scale_residual_norm_scale_shift


def _make_tensor(index_mode, shape, dtype):
    if index_mode == "NAT":
        return None
    return torch.randn(*SHAPE_MAP[index_mode](*shape), device=DEVICE, dtype=dtype)


def _apply_scale_shift(y, scale, shift):
    if scale.ndim == 4:
        num_frame = scale.shape[1]
        return rearrange(
            rearrange(y, "b (f l) d -> b f l d", f=num_frame) * (1 + scale) + shift,
            "b f l d -> b (f l) d",
        )
    scale = rearrange(scale, "b d -> b 1 d") if scale.ndim == 2 else scale
    shift = rearrange(shift, "b d -> b 1 d") if shift.ndim == 2 else shift
    return y * (1 + scale) + shift


def _cute_reference(residual, x, gate, weight, bias, scale, shift, norm_type, eps):
    """fp32 oracle for both variants; ``residual is None`` = no-residual form."""
    original_dtype = x.dtype
    residual, x, gate, weight, bias, scale, shift = (
        v.float() if isinstance(v, torch.Tensor) else v
        for v in (residual, x, gate, weight, bias, scale, shift)
    )
    residual_out = None
    if residual is not None:
        if isinstance(gate, int):
            x = residual + gate * x
        elif gate.ndim == 4:
            folded = rearrange(x, "b (f l) d -> b f l d", f=gate.shape[1])
            x = residual + rearrange(folded * gate, "b f l d -> b (f l) d")
        else:
            g = rearrange(gate, "b d -> b 1 d") if gate.ndim == 2 else gate
            x = residual + g * x
        residual_out = x.to(original_dtype)
    if norm_type == "layer":
        norm = torch.layer_norm(x, x.shape[-1:], eps=eps, weight=weight, bias=bias)
    else:
        norm = torch.rms_norm(x, x.shape[-1:], eps=eps, weight=weight)
    return _apply_scale_shift(norm, scale, shift).to(original_dtype), residual_out


@torch.no_grad()
def _run_cute(
    with_residual,
    shape=CUTE_SHAPES[0],
    dtype=DTYPES[0],
    affine_dtype=DTYPES[0],
    mod_dtype=DTYPES[0],
    norm_type=NORM_TYPES[0],
    affine_mode=AFFINE_MODES[0],
    gate_mode="B1D",
    index_mode="BSD",
    eps=EPS,
):
    fused_norm_scale_shift, fused_scale_residual_norm_scale_shift = _import_cutedsl()

    x = _make_tensor("BSD", shape, dtype)
    weight = _make_tensor(affine_mode, shape, affine_dtype)
    bias = _make_tensor(affine_mode, shape, affine_dtype)
    scale = _make_tensor(index_mode, shape, mod_dtype)
    shift = _make_tensor(index_mode, shape, mod_dtype)
    tol = 1e-5 if dtype == torch.float32 else 5e-2

    if with_residual:
        residual = _make_tensor("BSD", shape, dtype)
        gate = _make_tensor(gate_mode, shape, dtype)
        y, res = fused_scale_residual_norm_scale_shift(
            residual, x, gate, weight, bias, scale, shift, norm_type, eps
        )
        y_ref, res_ref = _cute_reference(
            residual, x, gate, weight, bias, scale, shift, norm_type, eps
        )
        torch.testing.assert_close(res, res_ref, atol=tol, rtol=tol)
    else:
        y = fused_norm_scale_shift(x, weight, bias, scale, shift, norm_type, eps)
        y_ref, _ = _cute_reference(
            None, x, None, weight, bias, scale, shift, norm_type, eps
        )
    torch.testing.assert_close(y, y_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("norm_type", NORM_TYPES)
@pytest.mark.parametrize("shape", CUTE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cutedsl_norm_scale_shift_shapes(with_residual, norm_type, shape, dtype):
    _run_cute(with_residual, shape=shape, dtype=dtype, norm_type=norm_type)


@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("norm_type", NORM_TYPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("operand", ["affine", "modulation"])
def test_cutedsl_norm_scale_shift_mixed_operand_dtypes(
    with_residual, norm_type, dtype, operand
):
    # The affine params and the modulation rows may each arrive in a dtype
    # other than the activation's; both combinations must dispatch.
    kwargs = {"affine_dtype" if operand == "affine" else "mod_dtype": dtype}
    _run_cute(with_residual, norm_type=norm_type, **kwargs)


@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("norm_type", NORM_TYPES)
@pytest.mark.parametrize("affine_mode", AFFINE_MODES)
def test_cutedsl_norm_scale_shift_affine_modes(with_residual, norm_type, affine_mode):
    _run_cute(with_residual, norm_type=norm_type, affine_mode=affine_mode)


@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("norm_type", NORM_TYPES)
@pytest.mark.parametrize("index_mode", INDEX_MODES)
def test_cutedsl_norm_scale_shift_index_modes(with_residual, norm_type, index_mode):
    _run_cute(with_residual, norm_type=norm_type, index_mode=index_mode)


@pytest.mark.parametrize("norm_type", NORM_TYPES)
@pytest.mark.parametrize("index_mode", INDEX_MODES)
def test_cutedsl_scale_residual_gate_index_modes(norm_type, index_mode):
    _run_cute(True, norm_type=norm_type, gate_mode=index_mode)


def test_validate_scale_shift_rejects_non_divisible_frames():
    _import_cutedsl()
    from sglang.kernels.ops.diffusion import validate_scale_shift

    with pytest.raises(ValueError, match=r"S\(10\) must be divisible by F\(4\)"):
        validate_scale_shift(
            torch.empty((1, 4, 1, 256), device=DEVICE, dtype=torch.float16), 1, 10, 256
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
