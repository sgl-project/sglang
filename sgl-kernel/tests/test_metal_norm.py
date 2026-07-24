"""Correctness tests for the AOT Metal ``rms_norm`` kernel (Apple Silicon).

Mirrors ``test_norm.py``: the reference is computed in fp32 and cast back to the
working dtype (the llama RMSNorm form), and the kernel output is compared with
dtype-appropriate tolerances, ``|y - ref| <= atol + rtol * |ref|``.

The whole module is skipped unless the ``sgl_kernel`` Metal extension is built
and loaded, so it is a no-op on non-Apple platforms and in CI without Mac
runners. Run it directly on Apple Silicon with:

    ~/venvs/sglang/bin/python -m pytest sgl-kernel/tests/test_metal_norm.py -v
"""

import pytest

try:
    import mlx.core as mx
    from sgl_kernel import metal

    _HAVE_METAL = metal._metal is not None
except ImportError:
    _HAVE_METAL = False

pytestmark = pytest.mark.skipif(
    not _HAVE_METAL,
    reason="sgl_kernel Metal extension unavailable (requires an Apple Silicon build)",
)


def llama_rms_norm_ref(x, w, eps):
    """RMSNorm in fp32, cast back to x's dtype. Matches mx.fast.rms_norm's contract."""
    orig = x.dtype
    xf = x.astype(mx.float32)
    var = mx.mean(xf * xf, axis=-1, keepdims=True)
    xf = xf * mx.rsqrt(var + eps)
    return (xf * w.astype(mx.float32)).astype(orig)


# (rtol, atol) per dtype. f16 matches test_norm.py's 1e-3; bf16 is looser for its
# 8-bit mantissa; f32 is near-exact (only fp32 reduction-order noise).
_TOL = {
    "float16": (1e-3, 1e-3),
    "bfloat16": (2e-2, 2e-2),
    "float32": (1e-4, 1e-4),
}


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_rms_norm(batch_size, hidden_size, dtype):
    eps = 1e-6
    dt = getattr(mx, dtype)
    rtol, atol = _TOL[dtype]

    mx.random.seed(0)
    x = mx.random.normal((batch_size, hidden_size)).astype(dt)
    w = mx.random.normal((hidden_size,)).astype(dt)

    y = metal.rms_norm(x, w, eps=eps)
    ref = llama_rms_norm_ref(x, w, eps)
    mx.eval(y, ref)

    diff = mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))
    allowed = atol + rtol * mx.abs(ref.astype(mx.float32))
    max_diff = mx.max(diff).item()
    assert mx.all(diff <= allowed).item(), (
        f"{dtype} B={batch_size} H={hidden_size}: max|diff|={max_diff:.3e} "
        f"exceeds tolerance (rtol={rtol}, atol={atol})"
    )


def test_weight_dtype_must_match():
    # The kernel binds w as the same dtype T as x (float(w[i]) upcast inside),
    # so the wrapper must reject a mismatched weight dtype.
    x = mx.random.normal((4, 1024)).astype(mx.float16)
    w = mx.random.normal((1024,)).astype(mx.float32)
    with pytest.raises(ValueError):
        metal.rms_norm(x, w, eps=1e-6)


def test_input_must_be_2d():
    x = mx.random.normal((4, 8, 1024)).astype(mx.float16)  # 3-D, unsupported
    w = mx.random.normal((1024,)).astype(mx.float16)
    with pytest.raises(ValueError):
        metal.rms_norm(x, w, eps=1e-6)


def test_weight_length_must_match_hidden():
    x = mx.random.normal((4, 1024)).astype(mx.float16)
    w = mx.random.normal((512,)).astype(mx.float16)  # wrong length
    with pytest.raises(ValueError):
        metal.rms_norm(x, w, eps=1e-6)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
