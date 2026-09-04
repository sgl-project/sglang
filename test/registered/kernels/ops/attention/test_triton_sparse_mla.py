import math
import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
    _FP8_MAX,
    triton_sparse_mla_fwd,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="requires an AMD GPU",
)

SEQ = 3
HEAD_DIM = 576
VALUE_DIM = 512
ROPE_DIM = HEAD_DIM - VALUE_DIM
POOL_SIZE = 64
TOPK = 32


def _require_gfx950() -> None:
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if "gfx950" not in arch:
        pytest.skip(f"the sparse-MLA Triton prefill path requires gfx950, got {arch}")


def _make_indices(
    pattern: str,
    *,
    seq: int = SEQ,
    pool_size: int = POOL_SIZE,
    topk: int = TOPK,
) -> torch.Tensor:
    if pattern == "trailing":
        base = torch.arange(pool_size - topk, pool_size)
    elif pattern == "strided":
        base = torch.arange(0, pool_size, pool_size // topk)
    elif pattern == "head_tail":
        base = torch.cat(
            [
                torch.arange(0, topk // 2),
                torch.arange(pool_size - topk // 2, pool_size),
            ]
        )
    else:
        raise ValueError(f"unknown index pattern: {pattern}")

    rows = [torch.roll(base, shifts=row) for row in range(seq)]
    return torch.stack(rows).to(device="cuda", dtype=torch.int32).unsqueeze(1)


def _make_inputs(
    num_heads: int,
    pattern: str,
    *,
    seq: int = SEQ,
    pool_size: int = POOL_SIZE,
    topk: int = TOPK,
):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260901 + num_heads)
    q_nope = (
        torch.randn(
            seq,
            num_heads,
            VALUE_DIM,
            device="cuda",
            generator=generator,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    q_rope = (
        torch.randn(
            seq,
            num_heads,
            ROPE_DIM,
            device="cuda",
            generator=generator,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    kv = (
        torch.randn(
            pool_size,
            1,
            HEAD_DIM,
            device="cuda",
            generator=generator,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    return (
        q_nope,
        q_rope,
        kv,
        _make_indices(pattern, seq=seq, pool_size=pool_size, topk=topk),
    )


def _reference(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    q = torch.cat([q_nope, q_rope], dim=-1).float()
    selected = kv[:, 0].float()[indices[:, 0].long()]
    scores = torch.einsum("shd,std->sht", q, selected) / math.sqrt(HEAD_DIM)
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = (probabilities * _FP8_MAX).to(q_nope.dtype).float() / _FP8_MAX
    return torch.einsum("sht,std->shd", probabilities, selected[:, :, :VALUE_DIM])


@pytest.mark.parametrize("num_heads", [4, 8, 12, 16])
@pytest.mark.parametrize("pattern", ["trailing", "strided", "head_tail"])
def test_triton_sparse_mla_raw_fp8(num_heads: int, pattern: str) -> None:
    """Padded-head correctness for the gfx950 raw-FP8 sparse-MLA prefill path.

    ``num_heads=4``, ``8`` and ``12`` exercise the BLOCK_H padding: fewer
    real heads than the tile occupy a
    16-row MFMA tile, so a regression in the query load/store masking or in the
    softmax row mask lets padded rows contaminate the denominator or write
    garbage into the output. ``num_heads=16`` is the unpadded control, which
    isolates a padding regression from a general kernel regression. The three
    index patterns cover contiguous, strided and split selections, so a masking
    error that only appears on non-contiguous KV cannot pass unnoticed.
    """
    _require_gfx950()
    q_nope, q_rope, kv, indices = _make_inputs(num_heads, pattern)
    actual = triton_sparse_mla_fwd(
        q_nope,
        q_rope,
        kv,
        indices,
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
        d_v=VALUE_DIM,
    ).squeeze(0)
    expected = _reference(q_nope, q_rope, kv, indices)
    torch.testing.assert_close(actual.float(), expected, atol=0.2, rtol=0.2)


def test_triton_sparse_mla_gfx950_full_topk() -> None:
    """Compile every candidate in the gfx950 padded-head dispatch grid."""
    _require_gfx950()
    q_nope, q_rope, kv, indices = _make_inputs(
        8,
        "trailing",
        seq=1,
        pool_size=2048,
        topk=2048,
    )
    actual = triton_sparse_mla_fwd(
        q_nope,
        q_rope,
        kv,
        indices,
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
        d_v=VALUE_DIM,
    ).squeeze(0)
    expected = _reference(q_nope, q_rope, kv, indices)
    torch.testing.assert_close(actual.float(), expected, atol=0.2, rtol=0.2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
