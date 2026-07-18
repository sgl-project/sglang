"""fused_moe_preprocess must be bit-identical to the torch.sort-based path,
and the grouped GEMM must produce identical results under both block_size_m
configs (the block schedule and kernel config are chosen together).
"""

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.inkling_moe import (
    SMALL_M_BLOCK_SIZE_M,
    compute_grouped_gemm_metadata,
    fused_moe_preprocess,
    get_src2dst,
    grouped_gemm_triton,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

E = 256
TOPK = 6


def _reference(topk_ids_flat: torch.Tensor):
    reorder_topk_ids, reorder_ids = torch.sort(
        topk_ids_flat.to(torch.int16), stable=True
    )
    src2dst = get_src2dst(reorder_ids)
    meta = compute_grouped_gemm_metadata(
        reorder_topk_ids, E, block_size_m=SMALL_M_BLOCK_SIZE_M
    )
    return (src2dst, *meta, reorder_topk_ids)


def _ids(tokens: int, seed: int, skew: bool = False) -> torch.Tensor:
    torch.manual_seed(seed)
    if skew:  # all tokens on few experts (stresses multi-block experts)
        return torch.randint(0, 3, (tokens * TOPK,), dtype=torch.int32, device="cuda")
    return (
        torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(tokens)])
        .view(-1)
        .to(torch.int32)
    )


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 2, 7, 32, 64, 170, 341])  # n = 6*T <= 2048
@pytest.mark.parametrize("skew", [False, True])
def test_matches_sort_path(tokens: int, skew: bool):
    ids = _ids(tokens, seed=tokens, skew=skew)
    ref = _reference(ids)
    got = fused_moe_preprocess(ids, E)
    names = [
        "src2dst",
        "num_tokens_per_expert",
        "expert_token_offs",
        "expert_block_offs",
        "expert_block_schedule",
        "reorder_topk_ids",
    ]
    for tag, g, r in zip(names, got, ref):
        assert g.shape == r.shape, (tag, g.shape, r.shape)
        assert torch.equal(g.long(), r.long()), (
            tag,
            g[: min(16, g.numel())],
            r[: min(16, r.numel())],
        )


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 16, 64])
def test_grouped_gemm_small_config_matches(tokens: int):
    """GEMM output must be identical whichever (block_size_m, config) runs."""
    torch.manual_seed(tokens)
    ids = _ids(tokens, seed=tokens)
    m, k, n = tokens * TOPK, 768, 1024
    a = (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
    b = (torch.randn(E, n, k, device="cuda") * 0.02).to(torch.bfloat16)

    sorted_ids, _ = torch.sort(ids.to(torch.int16), stable=True)
    meta128 = compute_grouped_gemm_metadata(sorted_ids, E)
    out128 = grouped_gemm_triton(a, b, E, *meta128)

    pre = fused_moe_preprocess(ids, E)
    out16 = grouped_gemm_triton(a, b, E, *pre[1:5], block_size_m=SMALL_M_BLOCK_SIZE_M)
    # both are fp32-accumulated bf16 tensor-core dots; BLOCK_K differs so
    # accumulation grouping may differ by a few ulp
    torch.testing.assert_close(out16.float(), out128.float(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
