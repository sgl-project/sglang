import sys

import pytest
import torch

from sglang.kernels.ops.speculative.ngram_mask import (
    build_ngram_full_tree_mask,
    build_ngram_full_tree_mask_ref,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")


def _make_offsets(seq_lens: torch.Tensor, draft_token_num: int) -> torch.Tensor:
    """Exclusive prefix sum of D * (seq + D) — same geometry as the worker."""
    bs = seq_lens.numel()
    seq = seq_lens.to(dtype=torch.int32)
    sizes = draft_token_num * (seq + draft_token_num)
    offsets = torch.zeros(bs, dtype=torch.int32, device=seq_lens.device)
    if bs > 1:
        torch.cumsum(sizes[:-1], dim=0, out=offsets[1:])
    return offsets


def _required_numel(seq_lens_cpu: torch.Tensor, draft_token_num: int) -> int:
    """exact = D * sum(seq) + D * D * bs"""
    bs = seq_lens_cpu.numel()
    D = draft_token_num
    return D * int(seq_lens_cpu.sum().item()) + D * D * bs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bs", [1, 2, 8, 32])
@pytest.mark.parametrize("draft_token_num", [4, 8])
@pytest.mark.parametrize("seq_mode", ["equal", "mixed", "long"])
def test_bit_identical(bs, draft_token_num, seq_mode):
    torch.manual_seed(0)
    D = draft_token_num
    if seq_mode == "equal":
        seq_lens_cpu = torch.full((bs,), 128, dtype=torch.int32)
    elif seq_mode == "mixed":
        base = torch.tensor([1, 7, 16, 64, 128, 257, 512, 1023], dtype=torch.int32)
        seq_lens_cpu = base[torch.arange(bs) % len(base)]
    else:
        seq_lens_cpu = torch.full((bs,), 4096, dtype=torch.int32)

    seq_lens = seq_lens_cpu.cuda()
    draft_mask = torch.randint(0, 2, (bs, D, D), device="cuda", dtype=torch.bool)
    if bs >= 2:
        draft_mask[1] = torch.tril(torch.ones(D, D, device="cuda", dtype=torch.bool))

    offsets = _make_offsets(seq_lens, D)
    required_numel = _required_numel(seq_lens_cpu, D)

    ref = build_ngram_full_tree_mask_ref(draft_mask, seq_lens_cpu, D)
    out = build_ngram_full_tree_mask(
        draft_mask,
        seq_lens,
        offsets,
        D,
        required_numel=required_numel,
        tree_mask_buf=None,
    )

    assert out.dtype == torch.bool
    assert out.numel() == required_numel
    assert out.shape == ref.shape
    assert torch.equal(out, ref), (
        f"mismatch bs={bs} D={D} seq_mode={seq_mode} "
        f"diff={(out != ref).sum().item()} / {out.numel()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_empty_batch():
    D = 8
    draft_mask = torch.empty(0, D, D, dtype=torch.bool, device="cuda")
    seq_lens = torch.empty(0, dtype=torch.int32, device="cuda")
    offsets = torch.empty(0, dtype=torch.int32, device="cuda")
    out = build_ngram_full_tree_mask(
        draft_mask,
        seq_lens,
        offsets,
        D,
        required_numel=0,
        tree_mask_buf=None,
    )
    assert out.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("buf_dtype", [torch.bool, torch.uint8])
def test_preallocated_buffer(buf_dtype):
    """Oversized tree_mask_buf is filled in-place (EAGLE-style)."""
    bs, D = 2, 4
    seq_lens_cpu = torch.tensor([5, 10], dtype=torch.int32)
    seq_lens = seq_lens_cpu.cuda()
    draft_mask = torch.tril(torch.ones(bs, D, D, device="cuda", dtype=torch.bool))
    offsets = _make_offsets(seq_lens, D)
    required_numel = _required_numel(seq_lens_cpu, D)

    buf = torch.empty(required_numel + 128, dtype=buf_dtype, device="cuda")
    out = build_ngram_full_tree_mask(
        draft_mask,
        seq_lens,
        offsets,
        D,
        required_numel=required_numel,
        tree_mask_buf=buf,
    )
    ref = build_ngram_full_tree_mask_ref(draft_mask, seq_lens_cpu, D)

    assert out.data_ptr() == buf.data_ptr()
    # Compare the exact prefix (worker does out[:exact]).
    got = out[:required_numel]
    if got.dtype != torch.bool:
        got = got.to(torch.bool)
    assert torch.equal(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_preallocated_buffer_too_small_asserts():
    bs, D = 1, 4
    seq_lens_cpu = torch.tensor([8], dtype=torch.int32)
    seq_lens = seq_lens_cpu.cuda()
    draft_mask = torch.ones(bs, D, D, device="cuda", dtype=torch.bool)
    offsets = _make_offsets(seq_lens, D)
    required_numel = _required_numel(seq_lens_cpu, D)

    short = torch.empty(max(required_numel - 1, 0), dtype=torch.bool, device="cuda")
    with pytest.raises(AssertionError):
        build_ngram_full_tree_mask(
            draft_mask,
            seq_lens,
            offsets,
            D,
            required_numel=required_numel,
            tree_mask_buf=short,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
