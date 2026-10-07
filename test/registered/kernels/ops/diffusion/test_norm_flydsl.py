"""``diffusion.norm``: the FlyDSL fused norm + scale/shift kernels (ROCm).

Split out of ``test_norm.py`` rather than merged with the other norm backends:
FlyDSL is an AMD gfx950-only compiler, so these run on the AMD CI lane and
nothing else in that file does.  Keeping them together forced the CUDA-only
CuTe-DSL cases onto the ROCm runner, where cuda-python does not exist.

Oracle: an fp32 reference chain, with a tolerance -- the kernel keeps fp32
statistics but reorders the reduction.
"""

import os
import subprocess
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

DEVICE = "cuda"

FLYDSL_MODULE = "sglang.kernels.ops.diffusion.norm.fused_residual_norm_flydsl"
FLYDSL_D = 5120
FLYDSL_EPS = 1e-6


def _require_rocm():
    if not torch.version.hip:
        pytest.skip("ROCm/HIP required for FlyDSL kernels")


def _flydsl_ops():
    """Resolve the FlyDSL exports, skipping when the installed FlyDSL is too old.

    Resolved inside each test: the FlyDSL compiler only exists on ROCm, and the
    facade imports the submodule the moment an export is named -- a module-level
    import would fail collection of this whole file on CUDA.

    The kernel module raises ImportError when the stable FlyDSL surface it needs
    is absent, which is also how ``layernorm.py`` detects that it must fall back
    to the native path.  A runner image predating that surface should skip here
    rather than report a kernel regression.
    """
    _require_rocm()
    try:
        from sglang.kernels.ops.diffusion import (
            flydsl_fused_residual_norm_scale_shift,
            flydsl_norm_scale_shift,
        )
    except ImportError as exc:
        pytest.skip(f"FlyDSL unavailable or too old for the fused norm kernels: {exc}")
    return flydsl_fused_residual_norm_scale_shift, flydsl_norm_scale_shift


def _mk(shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device=DEVICE, dtype=dtype)


def _ref_rms_norm(x_f32, eps):
    var = x_f32.pow(2).mean(-1, keepdim=True)
    return x_f32 * torch.rsqrt(var + eps)


def _apply_affine(normed, weight, bias, norm_type):
    """Reproduce the kernel's affine stage.

    The kernel applies weight and bias together, and skips both when weight is
    absent -- a standalone bias is not reachable through the public op.
    """
    if weight is None:
        return normed
    out = normed * weight.float()
    if norm_type == "layer" and bias is not None:
        out = out + bias.float()
    return out


def _ref_norm(x_bf16, weight, bias, norm_type, eps):
    base = x_bf16.float()
    if norm_type == "layer":
        mean = base.mean(-1, keepdim=True)
        var = base.var(-1, keepdim=True, unbiased=False)
        normed = (base - mean) * torch.rsqrt(var + eps)
    else:
        normed = _ref_rms_norm(base, eps)
    return _apply_affine(normed, weight, bias, norm_type)


def _ref_fused_residual_norm_ss(
    residual, x, gate, weight, bias, scale, shift, norm_type, eps
):
    ref_res = residual.float() + x.float() * (gate.float() if gate is not None else 1)
    ref_res_bf16 = ref_res.to(torch.bfloat16)
    normed = _ref_norm(ref_res_bf16, weight, bias, norm_type, eps)
    y = (normed * (1.0 + scale.float()) + shift.float()).to(torch.bfloat16)
    return y, ref_res_bf16


def _ref_norm_ss(x, weight, bias, scale, shift, norm_type, eps):
    normed = _ref_norm(x, weight, bias, norm_type, eps)
    return (normed * (1.0 + scale.float()) + shift.float()).to(torch.bfloat16)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)


# norm_type, B, L, has_gate, has_weight, scale/shift layout
FUSED_CASES = [
    ("rms", 1, 16, True, True, "bcast"),
    ("rms", 2, 16, True, True, "bcast"),
    ("layer", 2, 16, True, True, "bcast"),
    ("rms", 1, 90000, True, True, "bcast"),
    # gate-free path: a distinct compiled specialization
    ("rms", 2, 16, False, True, "bcast"),
    ("layer", 2, 16, False, True, "bcast"),
    # no-affine path: the production default
    ("rms", 2, 16, True, False, "bcast"),
    ("layer", 2, 16, True, False, "bcast"),
    # per-row scale/shift exercises the non-zero row stride
    ("rms", 2, 16, True, True, "perrow"),
    ("layer", 2, 16, True, True, "perrow"),
]


@pytest.mark.parametrize("norm_type,B,L,has_gate,has_weight,ss", FUSED_CASES)
def test_flydsl_fused_residual_norm_scale_shift(
    norm_type, B, L, has_gate, has_weight, ss
):
    fused_op, _ = _flydsl_ops()

    ss_shape = (B, 1, FLYDSL_D) if ss == "bcast" else (B, L, FLYDSL_D)
    residual = _mk((B, L, FLYDSL_D))
    x = _mk((B, L, FLYDSL_D))
    gate = _mk((B, 1, FLYDSL_D)) if has_gate else None
    weight = _mk((FLYDSL_D,), torch.float32) if has_weight else None
    bias = (
        _mk((FLYDSL_D,), torch.float32)
        if (has_weight and norm_type == "layer")
        else None
    )
    scale = _mk(ss_shape)
    shift = _mk(ss_shape)

    y, res_out = fused_op(
        residual, x, gate, weight, bias, scale, shift, norm_type, FLYDSL_EPS
    )
    y_ref, res_ref = _ref_fused_residual_norm_ss(
        residual, x, gate, weight, bias, scale, shift, norm_type, FLYDSL_EPS
    )
    torch.testing.assert_close(res_out, res_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(y, y_ref, atol=1.0, rtol=5e-2)


NSS_CASES = [
    ("rms", 2, 16, True, "bcast"),
    ("layer", 2, 16, True, "bcast"),
    ("rms", 1, 90000, True, "bcast"),
    ("layer", 1, 90000, True, "bcast"),
    # no-affine path
    ("rms", 2, 16, False, "bcast"),
    ("layer", 2, 16, False, "bcast"),
    # per-row scale/shift
    ("rms", 2, 16, True, "perrow"),
    ("layer", 2, 16, True, "perrow"),
]


@pytest.mark.parametrize("norm_type,B,L,has_weight,ss", NSS_CASES)
def test_flydsl_norm_scale_shift(norm_type, B, L, has_weight, ss):
    _, nss_op = _flydsl_ops()

    ss_shape = (B, 1, FLYDSL_D) if ss == "bcast" else (B, L, FLYDSL_D)
    x = _mk((B, L, FLYDSL_D))
    weight = _mk((FLYDSL_D,), torch.float32) if has_weight else None
    bias = (
        _mk((FLYDSL_D,), torch.float32)
        if (has_weight and norm_type == "layer")
        else None
    )
    scale = _mk(ss_shape)
    shift = _mk(ss_shape)

    y = nss_op(x, weight, bias, scale, shift, norm_type, FLYDSL_EPS)
    y_ref = _ref_norm_ss(x, weight, bias, scale, shift, norm_type, FLYDSL_EPS)
    torch.testing.assert_close(y, y_ref, atol=1.0, rtol=5e-2)


def test_flydsl_fused_frame_gate_4d_scale_shift():
    """4D (B, NF, 1, D) scale/shift is expanded to per-row by _prep_slices."""
    fused_op, _ = _flydsl_ops()

    B, L, NF = 2, 16, 4
    residual, x = _mk((B, L, FLYDSL_D)), _mk((B, L, FLYDSL_D))
    gate = _mk((B, 1, FLYDSL_D))
    weight = _mk((FLYDSL_D,), torch.float32)
    scale4, shift4 = _mk((B, NF, 1, FLYDSL_D)), _mk((B, NF, 1, FLYDSL_D))

    y, res_out = fused_op(
        residual, x, gate, weight, None, scale4, shift4, "rms", FLYDSL_EPS
    )

    def expand(t):
        return t.expand(B, NF, L // NF, FLYDSL_D).reshape(B, L, FLYDSL_D)

    y_ref, res_ref = _ref_fused_residual_norm_ss(
        residual,
        x,
        gate,
        weight,
        None,
        expand(scale4),
        expand(shift4),
        "rms",
        FLYDSL_EPS,
    )
    torch.testing.assert_close(res_out, res_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(y, y_ref, atol=1.0, rtol=5e-2)


def test_flydsl_multi_iteration_dim():
    """D=10240 drives NUM_ITERS=2, exercising more than one register-cached tile."""
    _, nss_op = _flydsl_ops()

    big_d, B, L = 10240, 1, 16
    x = _mk((B, L, big_d))
    weight = _mk((big_d,), torch.float32)
    scale, shift = _mk((B, 1, big_d)), _mk((B, 1, big_d))

    y = nss_op(x, weight, None, scale, shift, "rms", FLYDSL_EPS)
    y_ref = _ref_norm_ss(x, weight, None, scale, shift, "rms", FLYDSL_EPS)
    torch.testing.assert_close(y, y_ref, atol=1.0, rtol=5e-2)


def test_flydsl_compile_cache_reuse_across_row_counts_and_layouts():
    """Guard for the compile-cache/shape-specialization hazard.

    The cache key is (D, is_rms, has_gate, has_weight) and deliberately excludes
    the row count and whether scale/shift are broadcast. One compiled kernel must
    therefore stay correct when both of those change between calls.
    """
    _, nss_op = _flydsl_ops()

    weight = _mk((FLYDSL_D,), torch.float32)
    for L, ss in ((16, "bcast"), (90000, "bcast"), (16, "perrow"), (64, "bcast")):
        ss_shape = (1, 1, FLYDSL_D) if ss == "bcast" else (1, L, FLYDSL_D)
        x = _mk((1, L, FLYDSL_D))
        scale, shift = _mk(ss_shape), _mk(ss_shape)
        y = nss_op(x, weight, None, scale, shift, "rms", FLYDSL_EPS)
        y_ref = _ref_norm_ss(x, weight, None, scale, shift, "rms", FLYDSL_EPS)
        torch.testing.assert_close(
            y, y_ref, atol=1.0, rtol=5e-2, msg=f"L={L} layout={ss}"
        )


def test_flydsl_imports_without_flydsl_source_tree():
    """The module must resolve against the installed FlyDSL wheel alone.

    A FlyDSL source checkout on PYTHONPATH makes its `kernels` package
    importable and would hide an accidental source-tree dependency, so re-import
    in a subprocess with those entries stripped from the path.
    """
    _flydsl_ops()

    clean = [
        d
        for d in sys.path
        if d
        and not os.path.isfile(os.path.join(d, "kernels", "common", "buffer_ops.py"))
    ]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(clean))
    code = (
        "import importlib, sys;"
        f"m = importlib.import_module('{FLYDSL_MODULE}');"
        "assert 'kernels' not in sys.modules, 'leaked FlyDSL source-tree kernels package';"
        "print('OK', m.FLYDSL_NORM_MIN_ALIGNED_DIM)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert f"OK {FLYDSL_D}" in r.stdout, r.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
