"""Test that sgemm kernels produce identical results with and without SORTED_BY_ADAPTER."""

from typing import Any

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-test-1-gpu-large")


def _make_batch_info(
    bs: int,
    weight_indices: list[int],
    lora_ranks: list[int],
    scalings: list[float],
    device: str = "cuda",
) -> Any:
    """Build a per-sequence LoRABatchInfo (no permutation)."""
    from sglang.srt.lora.utils import LoRABatchInfo

    seg_lens = torch.ones(bs, dtype=torch.int32, device=device)
    seg_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
    return LoRABatchInfo(
        bs=bs,
        use_cuda_graph=False,
        num_segments=bs,
        seg_lens=seg_lens,
        seg_indptr=seg_indptr,
        max_len=1,
        weight_indices=torch.tensor(weight_indices, dtype=torch.int32, device=device),
        lora_ranks=torch.tensor(lora_ranks, dtype=torch.int32, device=device),
        scalings=torch.tensor(scalings, dtype=torch.float, device=device),
        permutation=None,
    )


def _make_sorted_batch_info(
    weight_indices: list[int],
    lora_ranks: list[int],
    scalings: list[float],
    max_loras: int,
    device: str = "cuda",
) -> Any:
    from sglang.srt.lora.utils import LoRABatchInfo

    """Build a merged-by-adapter LoRABatchInfo (with permutation)."""
    wi = torch.tensor(weight_indices, dtype=torch.int32, device=device)
    bs = wi.shape[0]

    perm = torch.argsort(wi, stable=True).to(torch.int32)
    sorted_wi = wi[perm]
    adapter_ids = torch.arange(max_loras, device=device, dtype=torch.int32)
    seg_starts = torch.searchsorted(sorted_wi, adapter_ids)
    seg_ends = torch.searchsorted(sorted_wi, adapter_ids, right=True)
    seg_lens = seg_ends - seg_starts

    seg_indptr = torch.zeros(max_loras + 1, dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

    return LoRABatchInfo(
        bs=max_loras,
        use_cuda_graph=False,
        num_segments=max_loras,
        seg_lens=seg_lens,
        seg_indptr=seg_indptr,
        max_len=bs,
        weight_indices=adapter_ids,
        lora_ranks=torch.tensor(lora_ranks, dtype=torch.int32, device=device),
        scalings=torch.tensor(scalings, dtype=torch.float, device=device),
        permutation=perm,
    )


def _check_close(
    a: torch.Tensor, b: torch.Tensor, name: str, atol: float = 1e-4, rtol: float = 1e-3
) -> None:
    diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, atol=atol, rtol=rtol), f"{name}: max diff = {diff}"


def test_sgemm_lora_a():
    from sglang.srt.lora.triton_ops import sgemm_lora_a_fwd

    torch.manual_seed(42)
    bs, input_dim, rank, num_loras = 8, 256, 16, 3
    x = torch.randn(bs, input_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        num_loras, rank, input_dim, device="cuda", dtype=torch.bfloat16
    )
    wi = [i % num_loras for i in range(bs)]
    lora_ranks = [rank] * num_loras
    scalings = [1.0] * num_loras

    bi_plain = _make_batch_info(bs, wi, lora_ranks, scalings)
    bi_sorted = _make_sorted_batch_info(wi, lora_ranks, scalings, num_loras)

    out_plain = sgemm_lora_a_fwd(x, weights, bi_plain)
    out_sorted = sgemm_lora_a_fwd(x, weights, bi_sorted)
    _check_close(out_plain, out_sorted, "sgemm_lora_a")


def test_sgemm_lora_b():
    from sglang.srt.lora.triton_ops import sgemm_lora_b_fwd

    torch.manual_seed(42)
    bs, output_dim, rank, num_loras = 8, 256, 16, 3
    x = torch.randn(bs, rank, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        num_loras, output_dim, rank, device="cuda", dtype=torch.bfloat16
    )
    wi = [i % num_loras for i in range(bs)]
    lora_ranks = [rank] * num_loras
    scalings = [0.5] * num_loras

    bi_plain = _make_batch_info(bs, wi, lora_ranks, scalings)
    bi_sorted = _make_sorted_batch_info(wi, lora_ranks, scalings, num_loras)

    base_plain = torch.randn(bs, output_dim, device="cuda", dtype=torch.bfloat16)
    base_sorted = base_plain.clone()

    out_plain = sgemm_lora_b_fwd(x, weights, bi_plain, base_plain)
    out_sorted = sgemm_lora_b_fwd(x, weights, bi_sorted, base_sorted)
    _check_close(out_plain, out_sorted, "sgemm_lora_b")


def test_qkv_lora_b():
    from sglang.srt.lora.triton_ops import qkv_lora_b_fwd

    torch.manual_seed(42)
    bs, rank, num_loras = 8, 16, 3
    n_slices = 3
    q_dim, kv_dim = 128, 64
    total_out = q_dim + 2 * kv_dim
    x = torch.randn(bs, n_slices * rank, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        num_loras, total_out, rank, device="cuda", dtype=torch.bfloat16
    )
    output_offset = torch.tensor(
        [0, q_dim, q_dim + kv_dim, total_out], device="cuda", dtype=torch.int32
    )
    wi = [i % num_loras for i in range(bs)]
    lora_ranks = [rank] * num_loras
    scalings = [1.0] * num_loras

    bi_plain = _make_batch_info(bs, wi, lora_ranks, scalings)
    bi_sorted = _make_sorted_batch_info(wi, lora_ranks, scalings, num_loras)

    base_plain = torch.randn(bs, total_out, device="cuda", dtype=torch.bfloat16)
    base_sorted = base_plain.clone()

    max_qkv_out_dim = max(q_dim, kv_dim)
    out_plain = qkv_lora_b_fwd(
        x, weights, bi_plain, output_offset, max_qkv_out_dim, base_plain
    )
    out_sorted = qkv_lora_b_fwd(
        x, weights, bi_sorted, output_offset, max_qkv_out_dim, base_sorted
    )
    _check_close(out_plain, out_sorted, "qkv_lora_b")


def test_gate_up_lora_b():
    from sglang.srt.lora.triton_ops import gate_up_lora_b_fwd

    torch.manual_seed(42)
    bs, rank, num_loras = 8, 16, 3
    output_dim = 128
    x = torch.randn(bs, 2 * rank, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        num_loras, 2 * output_dim, rank, device="cuda", dtype=torch.bfloat16
    )
    wi = [i % num_loras for i in range(bs)]
    lora_ranks = [rank] * num_loras
    scalings = [1.0] * num_loras

    bi_plain = _make_batch_info(bs, wi, lora_ranks, scalings)
    bi_sorted = _make_sorted_batch_info(wi, lora_ranks, scalings, num_loras)

    base_plain = torch.randn(bs, 2 * output_dim, device="cuda", dtype=torch.bfloat16)
    base_sorted = base_plain.clone()

    out_plain = gate_up_lora_b_fwd(x, weights, bi_plain, output_dim, base_plain)
    out_sorted = gate_up_lora_b_fwd(x, weights, bi_sorted, output_dim, base_sorted)
    _check_close(out_plain, out_sorted, "gate_up_lora_b")


def test_mixed_ranks():
    """Test with different LoRA ranks per adapter."""
    from sglang.srt.lora.triton_ops import sgemm_lora_a_fwd

    torch.manual_seed(42)
    bs, input_dim, num_loras = 12, 256, 4
    max_rank = 32
    lora_ranks = [8, 16, 32, 16]
    scalings = [0.25, 0.5, 1.0, 2.0]
    # Use max_rank for weight shape, kernel handles per-adapter rank
    weights = torch.randn(
        num_loras, max_rank, input_dim, device="cuda", dtype=torch.bfloat16
    )
    x = torch.randn(bs, input_dim, device="cuda", dtype=torch.bfloat16)
    wi = [i % num_loras for i in range(bs)]

    bi_plain = _make_batch_info(bs, wi, lora_ranks, scalings)
    bi_sorted = _make_sorted_batch_info(wi, lora_ranks, scalings, num_loras)

    out_plain = sgemm_lora_a_fwd(x, weights, bi_plain)
    out_sorted = sgemm_lora_a_fwd(x, weights, bi_sorted)
    _check_close(out_plain, out_sorted, "sgemm_lora_a_mixed_ranks")


def test_single_adapter():
    """All sequences use the same adapter."""
    from sglang.srt.lora.triton_ops import sgemm_lora_a_fwd

    torch.manual_seed(42)
    bs, input_dim, rank, num_loras = 16, 256, 16, 2
    x = torch.randn(bs, input_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        num_loras, rank, input_dim, device="cuda", dtype=torch.bfloat16
    )
    wi = [0] * bs  # all adapter 0
    lora_ranks = [rank, rank]
    scalings = [1.0, 1.0]

    bi_plain = _make_batch_info(bs, wi, lora_ranks, scalings)
    bi_sorted = _make_sorted_batch_info(wi, lora_ranks, scalings, num_loras)

    out_plain = sgemm_lora_a_fwd(x, weights, bi_plain)
    out_sorted = sgemm_lora_a_fwd(x, weights, bi_sorted)
    _check_close(out_plain, out_sorted, "sgemm_lora_a_single_adapter")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
