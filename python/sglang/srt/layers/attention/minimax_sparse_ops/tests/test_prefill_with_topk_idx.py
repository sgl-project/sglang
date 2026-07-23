import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
    flash_prefill_with_topk_index,
)


@pytest.mark.parametrize("score_type", ["max", "lse"])
def test_prefill_stale_max_seqlen_k(score_type):
    """A stale max_seqlen_k must not undersize the score buffer."""
    device = "cuda"
    torch.manual_seed(0)

    seq_lens = torch.tensor([64, 4096], dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    cu_seqlens = torch.tensor([0, 64, 4160], dtype=torch.int32, device=device)
    q = torch.randn(4160, 1, 128, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(8192, 1, 128, dtype=torch.bfloat16, device=device)
    req_to_token = torch.arange(8192, dtype=torch.int32, device=device).view(2, 4096)
    slot_ids = torch.arange(2, dtype=torch.int64, device=device)
    prefix_lens = torch.zeros(2, dtype=torch.int32, device=device)

    def run(max_seqlen_k):
        _, topk_idx = flash_prefill_with_topk_index(
            q=q,
            k_cache=k_cache,
            v_cache=None,
            sink=None,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=4096,
            max_seqlen_k=max_seqlen_k,
            block_size_q=128,
            block_size_k=128,
            topk=16,
            init_blocks=1,
            local_blocks=2,
            score_type=score_type,
            disable_index_value=True,
            seq_lens_cpu=seq_lens_cpu,
        )
        return topk_idx.clone()

    expected = run(4096)
    actual = run(64)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
