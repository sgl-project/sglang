import sys

import pytest
import torch

from sglang.kernels.ops.attention.rope import apply_rope_inplace
from sglang.kernels.ops.attention.table_qk_norm_rope import table_qk_norm_rope_
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def separate(qkv, positions, qw, kw, cache, hq, hk, d, eps):
    q, k, _ = qkv.split([hq * d, hk * d, hk * d], dim=-1)
    q, k = q.view(qkv.shape[0], hq, d), k.view(qkv.shape[0], hk, d)
    fused_inplace_qknorm(q, k, qw, kw, eps)
    apply_rope_inplace(q, k, cache, positions, is_neox=True)


def assert_same_bits(actual, expected):
    # NaN payload bits are not part of the contract; signed zero is.
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual.view(torch.int16)[~nan], expected.view(torch.int16)[~nan])


def inputs(d, batch, hq, hk, seed, padded=False, position_dtype=torch.int64):
    torch.manual_seed(seed)
    width = (hq + 2 * hk) * d
    storage = torch.randn(
        batch, width + 16 * padded, device="cuda", dtype=torch.bfloat16
    )
    x = storage[:, 8:-8] if padded else storage
    qw = (1 + torch.randn(d, device="cuda") * 0.2).bfloat16()
    kw = (1 + torch.randn(d, device="cuda") * 0.2).bfloat16()
    inv_freq = 1 / (10000 ** (torch.arange(0, d, 2, device="cuda").float() / d))
    angles = torch.arange(131072, device="cuda").float()[:, None] * inv_freq[None, :]
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    positions = ((torch.arange(batch, device="cuda") * 7919 + seed) % 131072).to(
        position_dtype
    )
    return storage, x, positions, qw, kw, cache


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize(
    "batch,hq,hk,padded,pos_dtype",
    [
        (0, 8, 2, False, torch.int64),
        (1, 32, 8, False, torch.int64),
        (4, 8, 2, True, torch.int32),
        (32, 32, 8, False, torch.int64),
        (257, 7, 1, True, torch.int64),
    ],
)
def test_cuda_sequence_is_bit_exact(d, batch, hq, hk, padded, pos_dtype):
    """Replacing CUDA norm then RoPE must preserve their bytes and untouched V/padding."""
    storage, x, pos, qw, kw, cache = inputs(d, batch, hq, hk, 19, padded, pos_dtype)
    before = storage.clone()
    expected = x.clone()
    # The table operator's empty-input guard is a no-op; the CUDA reference
    # launcher itself does not accept an empty grid.
    if batch:
        separate(expected, pos, qw, kw, cache, hq, hk, d, 1e-6)
    table_qk_norm_rope_(x, pos, qw, kw, cache, hq, hk, d, 1e-6, match_cuda=True)
    assert_same_bits(x, expected)
    if padded:
        assert_same_bits(storage[:, :8], before[:, :8])
        assert_same_bits(storage[:, -8:], before[:, -8:])


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_extreme_values_and_rounding(d, eps):
    """Reduction ordering and rounding must survive small/large magnitudes and NaNs."""
    _, x, pos, qw, kw, cache = inputs(d, 128, 32, 8, 23)
    magnitudes = torch.logspace(-20, 20, 128, device="cuda")
    x.copy_((x.float() * magnitudes[:, None]).bfloat16())
    x[0].zero_()
    x[1].fill_(-0.0)
    x[2, 0] = float("inf")
    x[3, d] = float("nan")
    x[4, 0] = -float("inf")
    expected = x.clone()
    separate(expected, pos, qw, kw, cache, 32, 8, d, eps)
    table_qk_norm_rope_(x, pos, qw, kw, cache, 32, 8, d, eps, match_cuda=True)
    assert_same_bits(x, expected)


@pytest.mark.parametrize("d", [64, 128, 256])
def test_inplace_repeated_large_grid(d):
    """Repeated launches catch unsynchronized reads/stores across warps."""
    _, x, pos, qw, kw, cache = inputs(d, 513, 32, 8, 19)
    torch.manual_seed(20261023)
    x.normal_()
    qw.copy_((0.1 + 10 * torch.rand(d, device="cuda")).bfloat16())
    kw.copy_((0.1 + 10 * torch.rand(d, device="cuda")).bfloat16())
    pos.random_(0, len(cache))
    expected = x.clone()
    separate(expected, pos, qw, kw, cache, 32, 8, d, 1e-6)
    for _ in range(10):
        actual = x.clone()
        table_qk_norm_rope_(actual, pos, qw, kw, cache, 32, 8, d, 1e-6, match_cuda=True)
        assert_same_bits(actual, expected)


@pytest.mark.parametrize("invalid", ["dtype", "qkv_stride", "table", "positions"])
def test_cuda_mode_rejects_invalid_inputs(invalid):
    """Malformed typed/strided inputs must fail before raw-pointer addressing."""
    _, x, pos, qw, kw, cache = inputs(128, 4, 8, 2, 19)
    if invalid == "dtype":
        qw = qw.float()
    elif invalid == "qkv_stride":
        x = x[:, ::2]
    elif invalid == "table":
        cache = cache.bfloat16()
    else:
        pos = pos.float()
    with pytest.raises(AssertionError):
        table_qk_norm_rope_(x, pos, qw, kw, cache, 8, 2, 128, 1e-6, match_cuda=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
