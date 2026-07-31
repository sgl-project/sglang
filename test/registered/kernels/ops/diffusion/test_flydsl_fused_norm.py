import os
import subprocess
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"
D = 5120
EPS = 1e-6


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


def _ref_fused_residual_norm_ss(
    residual, x, gate, weight, bias, scale, shift, norm_type, eps
):
    ref_res = residual.float() + x.float() * (gate.float() if gate is not None else 1)
    ref_res_bf16 = ref_res.to(torch.bfloat16)
    base = ref_res_bf16.float()
    if norm_type == "layer":
        mean = base.mean(-1, keepdim=True)
        var = base.var(-1, keepdim=True, unbiased=False)
        normed = (base - mean) * torch.rsqrt(var + eps)
    else:
        normed = _ref_rms_norm(base, eps)
    normed = _apply_affine(normed, weight, bias, norm_type)
    y = (normed * (1.0 + scale.float()) + shift.float()).to(torch.bfloat16)
    return y, ref_res_bf16


def _ref_norm_ss(x, weight, bias, scale, shift, norm_type, eps):
    base = x.float()
    if norm_type == "layer":
        mean = base.mean(-1, keepdim=True)
        var = base.var(-1, keepdim=True, unbiased=False)
        normed = (base - mean) * torch.rsqrt(var + eps)
    else:
        normed = _ref_rms_norm(base, eps)
    normed = _apply_affine(normed, weight, bias, norm_type)
    return (normed * (1.0 + scale.float()) + shift.float()).to(torch.bfloat16)


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not hasattr(torch.version, "hip") or not torch.version.hip:
        pytest.skip("ROCm/HIP required for FlyDSL kernels")
    torch.manual_seed(42)


def _mk(shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device=DEVICE, dtype=dtype)


# norm_type, B, L, has_gate, has_weight, scale/shift layout
FUSED_CASES = [
    ("rms", 1, 16, True, True, "bcast"),
    ("rms", 2, 16, True, True, "bcast"),
    ("layer", 2, 16, True, True, "bcast"),
    ("rms", 1, 90000, True, True, "bcast"),
    # gate-free path: a distinct compiled specialization
    ("rms", 2, 16, False, True, "bcast"),
    ("layer", 2, 16, False, True, "bcast"),
    # no-affine path: the production default, previously untested
    ("rms", 2, 16, True, False, "bcast"),
    ("layer", 2, 16, True, False, "bcast"),
    # per-row scale/shift exercises the non-zero row stride
    ("rms", 2, 16, True, True, "perrow"),
    ("layer", 2, 16, True, True, "perrow"),
]


@pytest.mark.parametrize("norm_type,B,L,has_gate,has_weight,ss", FUSED_CASES)
def test_fused_residual_norm_scale_shift(norm_type, B, L, has_gate, has_weight, ss):
    from sglang.kernels.ops.diffusion.flydsl.fused_residual_norm import (
        flydsl_fused_residual_norm_scale_shift,
    )

    ss_shape = (B, 1, D) if ss == "bcast" else (B, L, D)
    residual = _mk((B, L, D))
    x = _mk((B, L, D))
    gate = _mk((B, 1, D)) if has_gate else None
    weight = _mk((D,), torch.float32) if has_weight else None
    bias = _mk((D,), torch.float32) if (has_weight and norm_type == "layer") else None
    scale = _mk(ss_shape)
    shift = _mk(ss_shape)

    y, res_out = flydsl_fused_residual_norm_scale_shift(
        residual, x, gate, weight, bias, scale, shift, norm_type, EPS
    )
    y_ref, res_ref = _ref_fused_residual_norm_ss(
        residual, x, gate, weight, bias, scale, shift, norm_type, EPS
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
def test_norm_scale_shift(norm_type, B, L, has_weight, ss):
    from sglang.kernels.ops.diffusion.flydsl.fused_residual_norm import (
        flydsl_norm_scale_shift,
    )

    ss_shape = (B, 1, D) if ss == "bcast" else (B, L, D)
    x = _mk((B, L, D))
    weight = _mk((D,), torch.float32) if has_weight else None
    bias = _mk((D,), torch.float32) if (has_weight and norm_type == "layer") else None
    scale = _mk(ss_shape)
    shift = _mk(ss_shape)

    y = flydsl_norm_scale_shift(x, weight, bias, scale, shift, norm_type, EPS)
    y_ref = _ref_norm_ss(x, weight, bias, scale, shift, norm_type, EPS)
    torch.testing.assert_close(y, y_ref, atol=1.0, rtol=5e-2)


def test_fused_frame_gate_4d_scale_shift():
    """4D (B, NF, 1, D) scale/shift is expanded to per-row by _prep_slices."""
    from sglang.kernels.ops.diffusion.flydsl.fused_residual_norm import (
        flydsl_fused_residual_norm_scale_shift,
    )

    B, L, NF = 2, 16, 4
    residual, x = _mk((B, L, D)), _mk((B, L, D))
    gate = _mk((B, 1, D))
    weight = _mk((D,), torch.float32)
    scale4, shift4 = _mk((B, NF, 1, D)), _mk((B, NF, 1, D))

    y, res_out = flydsl_fused_residual_norm_scale_shift(
        residual, x, gate, weight, None, scale4, shift4, "rms", EPS
    )

    expand = lambda t: t.expand(B, NF, L // NF, D).reshape(B, L, D)  # noqa: E731
    y_ref, res_ref = _ref_fused_residual_norm_ss(
        residual, x, gate, weight, None, expand(scale4), expand(shift4), "rms", EPS
    )
    torch.testing.assert_close(res_out, res_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(y, y_ref, atol=1.0, rtol=5e-2)


def test_multi_iteration_dim():
    """D=10240 drives NUM_ITERS=2, exercising more than one register-cached tile."""
    from sglang.kernels.ops.diffusion.flydsl.fused_residual_norm import (
        flydsl_norm_scale_shift,
    )

    big_d, B, L = 10240, 1, 16
    x = _mk((B, L, big_d))
    weight = _mk((big_d,), torch.float32)
    scale, shift = _mk((B, 1, big_d)), _mk((B, 1, big_d))

    y = flydsl_norm_scale_shift(x, weight, None, scale, shift, "rms", EPS)
    y_ref = _ref_norm_ss(x, weight, None, scale, shift, "rms", EPS)
    torch.testing.assert_close(y, y_ref, atol=1.0, rtol=5e-2)


def test_compile_cache_reuse_across_row_counts_and_layouts():
    """Guard for the compile-cache/shape-specialization hazard.

    The cache key is (D, is_rms, has_gate, has_weight) and deliberately excludes
    the row count and whether scale/shift are broadcast. One compiled kernel must
    therefore stay correct when both of those change between calls.
    """
    from sglang.kernels.ops.diffusion.flydsl.fused_residual_norm import (
        flydsl_norm_scale_shift,
    )

    weight = _mk((D,), torch.float32)
    for L, ss in ((16, "bcast"), (90000, "bcast"), (16, "perrow"), (64, "bcast")):
        ss_shape = (1, 1, D) if ss == "bcast" else (1, L, D)
        x = _mk((1, L, D))
        scale, shift = _mk(ss_shape), _mk(ss_shape)
        y = flydsl_norm_scale_shift(x, weight, None, scale, shift, "rms", EPS)
        y_ref = _ref_norm_ss(x, weight, None, scale, shift, "rms", EPS)
        torch.testing.assert_close(
            y, y_ref, atol=1.0, rtol=5e-2, msg=f"L={L} layout={ss}"
        )


def test_imports_without_flydsl_source_tree():
    """The module must resolve against the installed FlyDSL wheel alone.

    A FlyDSL source checkout on PYTHONPATH makes its `kernels` package
    importable and would hide an accidental source-tree dependency, so re-import
    in a subprocess with those entries stripped from the path.
    """
    clean = [
        d
        for d in sys.path
        if d
        and not os.path.isfile(os.path.join(d, "kernels", "common", "buffer_ops.py"))
    ]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(clean))
    code = (
        "import importlib, sys;"
        "m = importlib.import_module("
        "'sglang.kernels.ops.diffusion.flydsl.fused_residual_norm');"
        "assert 'kernels' not in sys.modules, 'leaked FlyDSL source-tree kernels package';"
        "print('OK', m.FLYDSL_NORM_MIN_ALIGNED_DIM)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "OK 5120" in r.stdout, r.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
