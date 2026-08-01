"""silu_and_mul_triton must handle both fused gate/up layouts: interleaved
([gate[0], up[0], ...], the default w13 layout) and contiguous ([gate || up],
what --enable-lora de-interleaves to at load).

Each shape is checked against a float32 torch reference and -- for the same
logical values fed in both layouts -- for bit-identical output, which pins the
contiguous branch to the interleaved one.
"""

import pytest
import torch

from sglang.kernels.ops.moe.inkling_moe import silu_and_mul_triton
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

# tl.sigmoid and torch.sigmoid can differ in the last fp32 bit, which survives as
# ~1 ULP of the output dtype (a few ULP when the output is itself fp32). Still far
# tighter than any real bug: a wrong half or stride is an O(1) relative error.
TOL = {
    torch.bfloat16: dict(rtol=2**-7, atol=1e-6),
    torch.float16: dict(rtol=2**-10, atol=1e-7),
    torch.float32: dict(rtol=2**-20, atol=1e-6),
}


def _reference(
    gateup: torch.Tensor,
    topk_weights: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    use_interleaved: bool,
) -> torch.Tensor:
    x = gateup.float()
    if use_interleaved:
        gate, up = x[:, 0::2], x[:, 1::2]
    else:
        n = x.shape[1] // 2
        gate, up = x[:, :n], x[:, n:]
    out = gate * torch.sigmoid(gate) * up
    if topk_weights is not None:
        out = out * topk_weights.float()[:, None]
    return out.to(out_dtype or gateup.dtype)


def _interleave(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.stack([gate, up], dim=-1).flatten(1).contiguous()


def _check(M, N, *, with_weights, dtype, out_dtype=None, seed=0):
    """Run both layouts on the same gate/up values; compare to torch and to each other."""
    torch.manual_seed(seed)
    gate = torch.randn(M, N, dtype=dtype, device="cuda")
    up = torch.randn(M, N, dtype=dtype, device="cuda")
    topk_weights = torch.randn(M, dtype=dtype, device="cuda") if with_weights else None

    outs = {}
    for use_interleaved in (True, False):
        gateup = (
            _interleave(gate, up)
            if use_interleaved
            else torch.cat([gate, up], dim=1).contiguous()
        )
        got = silu_and_mul_triton(
            gateup, topk_weights, out_dtype, use_interleaved=use_interleaved
        )
        assert got.shape == (M, N), f"{got.shape=} {use_interleaved=}"
        assert got.dtype == (out_dtype or dtype), f"{got.dtype=} {use_interleaved=}"
        expect = _reference(gateup, topk_weights, out_dtype, use_interleaved)
        torch.testing.assert_close(
            got.float(), expect.float(), **TOL[out_dtype or dtype]
        )
        outs[use_interleaved] = got

    # Same math, different memory layout -> the branches must agree exactly.
    assert torch.equal(outs[True], outs[False]), (
        f"layout mismatch {M=} {N=} maxdiff="
        f"{(outs[True].float() - outs[False].float()).abs().max().item()}"
    )


@requires_cuda
@pytest.mark.parametrize("with_weights", [False, True])
@pytest.mark.parametrize("M", [1, 7, 32, 128, 1024, 4099])
@pytest.mark.parametrize("N", [8, 512, 768, 2048, 48 * 96])
def test_layouts(with_weights, M, N):
    _check(M, N, with_weights=with_weights, dtype=torch.bfloat16)


@requires_cuda
@pytest.mark.parametrize("N", [520, 1032, 3000])
def test_masked_n_tail(N):
    """N not a multiple of BLOCK_SIZE_N takes the masked load/store path."""
    _check(129, N, with_weights=True, dtype=torch.bfloat16)


@requires_cuda
@pytest.mark.parametrize(
    "dtype,out_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float16, None),
        (torch.float32, None),
    ],
)
def test_dtypes(dtype, out_dtype):
    _check(256, 1024, with_weights=True, dtype=dtype, out_dtype=out_dtype)


@requires_cuda
def test_int64_indexing():
    """A >=2GiB input flips the kernel to int64 offsets."""
    M, N = 2**19, 1024  # 2 * M * N * 2 bytes = 2 GiB, exactly at the threshold
    _check(M, N, with_weights=True, dtype=torch.bfloat16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
