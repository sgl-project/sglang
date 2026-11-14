"""Check TileLang indexer reads against the production FP8 cache store."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.paged_mqa import assert_paged_mqa_matches_ref

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _gfx950():
    return torch.cuda.is_available() and (
        str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950"
    )


@pytest.mark.skipif(not _gfx950(), reason="gfx950 only")
@pytest.mark.parametrize("preshuffle_tile", [0, 16])
@pytest.mark.parametrize("num_heads", [32, 64])
def test_tilelang_paged_mqa_logits(monkeypatch, preshuffle_tile, num_heads):
    from sglang.kernels.ops.attention.dsa import tilelang_kernel as reader
    from sglang.kernels.ops.kvcache import triton_store_cache as writer

    # Select the reader layout without relying on the machine's AITER flags.
    monkeypatch.setattr(
        reader,
        "aiter_can_use_preshuffle_paged_mqa",
        lambda: preshuffle_tile != 0,
        raising=False,
    )
    monkeypatch.setattr(
        writer, "aiter_can_use_preshuffle_paged_mqa", lambda: preshuffle_tile != 0
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(2026 + num_heads)
    device = "cuda"
    batch, page_size, dim, max_len = 4, 64, 128, 1088
    lengths = torch.tensor([511, 512, 513, 1025], device=device, dtype=torch.int32)
    num_pages = max_len // page_size
    permutation = torch.randperm(num_pages, device=device)
    assert not torch.equal(permutation, torch.arange(num_pages, device=device))
    page_table = permutation.int().expand(batch, -1).contiguous()
    # Exact FP8 integer dot products isolate the cache layout contract.
    q = torch.randint(-4, 5, (batch, 1, num_heads, dim), device=device).to(
        torch.float8_e4m3fn
    )
    keys = (torch.randint(-7, 8, (max_len, dim), device=device) * 64).float()
    # Column 0 at the FP8 max pins each token amax, so the store derives exactly these scales.
    keys[:, 0] = 448
    keys = keys.to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.randint(-8, 1, (max_len, 1), device=device).float())
    # FP8 values times a power of two are exactly representable in BF16.
    # The fused store therefore writes these same keys and per-token scales.
    source = (keys.float() * scales).to(torch.bfloat16)
    weights = torch.rand(batch, num_heads, device=device) / num_heads
    logical = torch.arange(max_len, device=device)
    slots = permutation[logical // page_size] * page_size + logical % page_size
    cache = torch.zeros(
        num_pages, page_size * (dim + 4), device=device, dtype=torch.uint8
    )

    writer.triton_fused_store_indexer(source, cache, slots, page_size)
    logits = reader.tilelang_fp8_paged_mqa_logits(
        q,
        cache.view(num_pages, page_size, 1, dim + 4),
        weights,
        lengths,
        page_table,
        deep_gemm_metadata=None,
        max_seq_len=max_len,
        clean_logits=False,
    )
    # Reference the original logical keys, independently of the cache layout.
    ref = torch.empty(batch, max_len, device=device)
    for row in range(batch):
        scores = torch.relu(q[row, 0].float() @ keys.float().T)
        ref[row] = (scores * weights[row, :, None]).sum(0) * scales[:, 0]
    assert_paged_mqa_matches_ref(logits, ref, lengths, batch, 1, max_len)
    for row, length in enumerate(lengths.tolist()):
        k = min(512, length)
        got = logits[row, :length].topk(k).indices.sort().values
        expected = ref[row, :length].topk(k).indices.sort().values
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
