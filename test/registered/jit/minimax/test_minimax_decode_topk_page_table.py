"""Fused decode top-k + page-table transform.

`minimax_decode_topk_page_table` selects the top-k blocks (same as the block-id
`minimax_decode_topk`) and emits the per-query paged page table consumed by the
dense backend (trtllm_mha), instead of block ids. This checks the fused output
end-to-end: trtllm decode over the emitted page table matches the custom
`_gqa_share_sparse_decode_kernel` fed the block-id selection from the same score.
Only the TP>=4 case (num_kv_heads == 1) is covered.
"""

import random
import sys

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

from sglang.jit_kernel.minimax_decode_topk import (
    minimax_decode_topk,
    minimax_decode_topk_page_table,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

dev = "cuda"


def _effective_kv_from_selection(ti, seq_lens, block):
    # Ground-truth effective KV length: sum of valid tokens over the selected
    # blocks (only the final block can be partial), per query.
    bs = seq_lens.shape[0]
    out = torch.zeros(bs, dtype=torch.int32, device=ti.device)
    for b in range(bs):
        sl = int(seq_lens[b])
        tot = 0
        for c in ti[0, b].tolist():
            if c < 0:
                continue
            tot += min(block, sl - c * block)
        out[b] = tot
    return out


@pytest.mark.parametrize(
    "bs,seq_len",
    [
        (1, 5000),
        (2, 300),
        (3, 160),
        (4, 2048),
        (8, 4096),
        (16, 8000),
        (2, 40000),  # num_blocks=313 -> medium radix path
        (1, 90000),  # num_blocks=704 -> large compaction path
        (1, 480000),  # num_blocks=3750 -> large path near kMaxNumBlocks
    ],
)
@pytest.mark.parametrize("nqh", [8, 16])
def test_fused_page_table_matches_custom(bs, seq_len, nqh):
    arch_major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if arch_major < 10:
        pytest.skip("trtllm-gen decode is Blackwell (sm100)")
    nkv, D, block, topk, ps = 1, 128, 128, 16, 64
    torch.manual_seed(bs * 31 + seq_len + nqh)
    random.seed(bs * 31 + seq_len)

    ppr = (seq_len + ps - 1) // ps
    npages = bs * ppr + 4
    kf = torch.randn(npages * ps, nkv, D, device=dev, dtype=torch.bfloat16) * 0.5
    vf = torch.randn(npages * ps, nkv, D, device=dev, dtype=torch.bfloat16) * 0.5
    q = torch.randn(bs, nqh, D, device=dev, dtype=torch.bfloat16) * 0.5

    r2t = torch.zeros(bs, seq_len, dtype=torch.int32, device=dev)
    sl = torch.full((bs,), seq_len, dtype=torch.int32, device=dev)
    sid = torch.arange(bs, dtype=torch.int64, device=dev)
    idxs = torch.arange(seq_len, device=dev)
    for b in range(bs):
        r2t[b] = ((b * ppr + idxs // ps) * ps + idxs % ps).int()

    nb = (seq_len + block - 1) // block
    score = torch.full((1, bs, nb), -float("inf"), device=dev, dtype=torch.float32)
    score[0, :, :nb] = torch.randn(bs, nb, device=dev)
    score[0, :, nb - 1] = 1e30  # final (local) block always selected

    # block-id selection -> custom kernel (reference)
    ti = minimax_decode_topk(score, sl, block, topk)
    ref = flash_decode_with_gqa_share_sparse(
        q, None, kf, vf, r2t, sl, sid, block, ti, sm_scale=D**-0.5
    )

    # fused page-table + effective KV length -> trtllm (allocated + returned)
    pt, cache = minimax_decode_topk_page_table(score, sl, r2t, sid, block, topk, ps)
    # the kernel's effective KV length must match the actual block selection
    expect_cache = _effective_kv_from_selection(ti, sl, block)
    assert torch.equal(cache, expect_cache), f"{cache} != {expect_cache}"
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=dev)
    kv = (
        kf.view(npages, ps, nkv, D).permute(0, 2, 1, 3),
        vf.view(npages, ps, nkv, D).permute(0, 2, 1, 3),
    )
    o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=kv,
        workspace_buffer=ws,
        block_tables=pt,
        seq_lens=cache,
        max_seq_len=topk * block,
        bmm1_scale=D**-0.5,
        bmm2_scale=1.0,
    )
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), o.float().flatten(), dim=0
    ).item()
    assert cos > 0.999, f"cos={cos}"


@pytest.mark.parametrize("seq_len", [300, 5000, 90000])
@pytest.mark.parametrize("bs", [1, 3])
@pytest.mark.parametrize("nkv", [2, 4])
def test_dp_flattened_page_table(nkv, bs, seq_len):
    """DP attention (num_kv_heads>1): each kv head selects its own blocks, flattened
    into bs*nkv pseudo-requests (row = b*nkv + h). Validate the flattened page table
    + effective KV length against the per-head block-id selection, including the
    head-minor head-encoded page index (base_page*nkv + h, the index into an HND
    cache [num_pages, nkv, ps, D] reshaped to [num_pages*nkv, 1, ps, D])."""
    D, block, topk, ps = 128, 128, 16, 64
    ppb = block // ps
    torch.manual_seed(nkv * 131 + bs * 31 + seq_len)
    ppr = (seq_len + ps - 1) // ps
    max_kv = seq_len  # req_to_token width

    r2t = torch.zeros(bs, seq_len, dtype=torch.int32, device=dev)
    sl = torch.full((bs,), seq_len, dtype=torch.int32, device=dev)
    sid = torch.arange(bs, dtype=torch.int64, device=dev)
    idxs = torch.arange(seq_len, device=dev)
    for b in range(bs):
        r2t[b] = ((b * ppr + idxs // ps) * ps + idxs % ps).int()

    nb = (seq_len + block - 1) // block
    score = torch.full((nkv, bs, nb), -float("inf"), device=dev, dtype=torch.float32)
    score[:, :, :nb] = torch.randn(nkv, bs, nb, device=dev)
    score[:, :, nb - 1] = 1e30  # final (local) block always selected

    # per-head block-id selection is the reference for the flattened page table
    ti = minimax_decode_topk(score, sl, block, topk)  # [nkv, bs, topk]
    pt, cache = minimax_decode_topk_page_table(score, sl, r2t, sid, block, topk, ps)
    msp = topk * ppb
    assert pt.shape == (bs * nkv, msp) and cache.shape == (bs * nkv,)

    r2t_cpu = r2t.cpu()
    for b in range(bs):
        for h in range(nkv):
            blocks = sorted(c for c in ti[h, b].tolist() if c >= 0)
            row = b * nkv + h
            # effective KV length = sum of valid tokens over selected blocks
            exp_kv = sum(min(block, seq_len - c * block) for c in blocks)
            assert (
                int(cache[row]) == exp_kv
            ), f"row {row}: {int(cache[row])} != {exp_kv}"
            # page table: each block -> ppb pages via req_to_token, head-minor encoded
            for e in range(len(blocks) * ppb):
                c = blocks[e // ppb]
                tok = c * block + (e % ppb) * ps
                if tok >= max_kv:
                    tok = max_kv - 1
                exp = int(r2t_cpu[b, tok]) // ps * nkv + h
                assert (
                    int(pt[row, e]) == exp
                ), f"row {row} e {e}: {int(pt[row,e])} != {exp}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
