import pytest
import torch
from sgl_kernel import max_pooling_1d_varlen


def _ref_varlen(
    score: torch.Tensor,  # [num_heads, total_q, max_k]
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cache_lens: torch.Tensor,
    max_context_len: int,
    local_blocks: int,
    init_blocks: int,
    block_size: int,
    kernel_stride: int,
) -> torch.Tensor:
    """Pure-torch reference mirroring the CUDA kernel exactly (fp32 math)."""
    num_heads, total_q, _ = score.shape
    out_len = (max_context_len + block_size - 1) // block_size
    stride = block_size // kernel_stride
    kernel_size = stride + 1
    padding = 1

    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    cache = cache_lens.tolist()
    batch_size = len(cache)

    out = torch.zeros(num_heads, total_q, out_len, dtype=torch.float32)
    s = score.float().cpu()
    for q in range(total_q):
        b = 0
        for bb in range(batch_size):
            if cu_q[bb] <= q < cu_q[bb + 1]:
                b = bb
                break
        bidq_local = q - cu_q[b]
        seqlen_k = cu_k[b + 1] - cu_k[b]
        off_bq = (bidq_local + cache[b]) // block_size
        for h in range(num_heads):
            for k in range(out_len):
                if (k < init_blocks) or (off_bq >= k and off_bq <= k + local_blocks):
                    out[h, q, k] = float("inf")
                else:
                    start = max(k * stride - padding, 0)
                    end = min(start + kernel_size, seqlen_k)
                    if end > start:
                        out[h, q, k] = s[h, q, start:end].max()
                    else:
                        out[h, q, k] = float("-inf")
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("seq_lens", [[37], [16, 48], [8, 8, 24]])
def test_max_pooling_varlen_matches_reference(dtype, num_heads, seq_lens):
    torch.manual_seed(0)
    block_size = 64
    kernel_stride = 16
    local_blocks = 1
    init_blocks = 1
    max_context_len = 512

    total_q = sum(seq_lens)
    max_k = max_context_len // kernel_stride
    cu = [0]
    for n in seq_lens:
        cu.append(cu[-1] + n)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.tensor(cu, dtype=torch.int32, device="cuda")
    cache_lens = torch.zeros(len(seq_lens), dtype=torch.int32, device="cuda")

    score = torch.randn(num_heads, total_q, max_k, dtype=dtype, device="cuda")

    out = max_pooling_1d_varlen(
        score,
        cu_seqlens_q,
        cu_seqlens_k,
        cache_lens,
        max_seqlen_q=max(seq_lens),
        max_context_len=max_context_len,
        local_blocks=local_blocks,
        init_blocks=init_blocks,
        block_size=block_size,
        stride=kernel_stride,
        total_q=total_q,
    )
    ref = _ref_varlen(
        score,
        cu_seqlens_q,
        cu_seqlens_k,
        cache_lens,
        max_context_len,
        local_blocks,
        init_blocks,
        block_size,
        kernel_stride,
    ).to(out.device)

    assert torch.equal(torch.isinf(out) & (out > 0), torch.isinf(ref) & (ref > 0))
    finite = torch.isfinite(ref)
    torch.testing.assert_close(out[finite].float(), ref[finite], rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
