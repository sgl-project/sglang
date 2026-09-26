"""ROCm tests for the MiniMax sparse decode score tile."""

import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.decode import flash_with_topk_idx
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")
pytestmark = pytest.mark.skipif(not is_hip(), reason="ROCm only")


@pytest.mark.parametrize("on_gfx95, tiny_batch_tile", [(True, 512), (False, 128)])
def test_score_tile_fits_lds(monkeypatch, on_gfx95, tiny_batch_tile):
    monkeypatch.setattr(flash_with_topk_idx, "_ON_GFX95", on_gfx95)
    pick = flash_with_topk_idx._decode_score_block_n
    assert pick({"batch_size": 1, "block_size": 128}) == tiny_batch_tile
    assert pick({"batch_size": 8, "block_size": 128}) == 128
    assert pick({"batch_size": 1, "block_size": 256}) == max(tiny_batch_tile, 256)


def _topk(q, k_cache, req_to_token, seq_lens, slot_ids):
    _, topk_idx, _ = flash_with_topk_idx.flash_decode_with_topk_idx(
        q,
        None,
        k_cache,
        None,
        req_to_token,
        seq_lens,
        req_to_token.shape[1],
        slot_ids,
        block_size=128,
        topk=16,
        init_blocks=0,
        local_blocks=1,
        disable_index_value=True,
    )
    return topk_idx.sort(dim=-1).values


def test_single_request_matches_batched():
    torch.manual_seed(0)
    batch, heads, head_dim = 8, 8, 128
    seq_lens = torch.randint(3000, 8000, (batch,), dtype=torch.int32, device="cuda")
    max_kv_len = int(seq_lens.max())
    req_to_token = torch.arange(
        batch * max_kv_len, dtype=torch.int32, device="cuda"
    ).view(batch, max_kv_len)
    k_cache = torch.randn(
        batch * max_kv_len, 1, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    q = torch.randn(batch, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    slot_ids = torch.arange(batch, dtype=torch.int32, device="cuda")

    batched = _topk(q, k_cache, req_to_token, seq_lens, slot_ids)
    single = _topk(q[:1], k_cache, req_to_token, seq_lens[:1], slot_ids[:1])
    torch.testing.assert_close(single[:, 0], batched[:, 0], rtol=0, atol=0)
