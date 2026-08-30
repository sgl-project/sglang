"""A -1 (tombstoned) virtual_to_physical entry must never reach the page tables
the fused decode-metadata kernel builds: a negative page index sent to an
attention kernel is a k_buffer[-1]-class illegal access under captured graph
replay. Tombstones must land on the reserved slot-0 sink instead."""

import torch
import triton

from sglang.kernels.ops.attention.metadata import _fused_metadata_kernel_general
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

PAGE_SIZE = 128
SHIFT = 7
BLOCK_COLS = 128


def test_v2p_tombstones_clamped_to_sink():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Request 0: 9 live pages, the first 5 outside the sliding window and
    # SWA-tombstoned. Request 1: 2 live pages, fully in-window.
    seq_lens = torch.tensor([1152, 256], dtype=torch.int64, device=device)
    bs = 2
    max_seq_pages = int((int(seq_lens.max()) + PAGE_SIZE - 1) // PAGE_SIZE)

    req_to_token = torch.zeros(
        (bs, max_seq_pages * PAGE_SIZE), dtype=torch.int64, device=device
    )
    for i in range(bs):
        pages = int((int(seq_lens[i]) + PAGE_SIZE - 1) // PAGE_SIZE)
        for p in range(pages):
            full_page = i * 16 + p
            req_to_token[i, p * PAGE_SIZE : (p + 1) * PAGE_SIZE] = torch.arange(
                full_page * PAGE_SIZE, (full_page + 1) * PAGE_SIZE, device=device
            )
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)

    num_pool_pages = 64
    v2p_full = torch.arange(num_pool_pages, dtype=torch.int32, device=device)
    v2p_full[0] = -1
    swa_v2p = torch.arange(num_pool_pages, dtype=torch.int32, device=device)
    swa_v2p[:5] = -1  # request 0's out-of-window prefix pages

    cache_seqlens = torch.zeros(bs, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    page_table = torch.zeros((bs, max_seq_pages), dtype=torch.int32, device=device)
    swa_page_table = torch.zeros((bs, max_seq_pages), dtype=torch.int32, device=device)

    grid = (bs, triton.cdiv(max_seq_pages, BLOCK_COLS))
    _fused_metadata_kernel_general[grid](
        seq_lens,
        seq_lens.stride(0),
        req_to_token,
        req_to_token.stride(0),
        req_to_token.stride(1),
        req_pool_indices,
        req_pool_indices.stride(0),
        cache_seqlens,
        cache_seqlens.stride(0),
        cu_seqlens_k,
        cu_seqlens_k.stride(0),
        page_table,
        page_table.stride(0),
        page_table.stride(1),
        swa_page_table,
        swa_page_table.stride(0),
        swa_page_table.stride(1),
        swa_v2p,
        0,
        bs,
        max_seq_pages,
        PAGE_SIZE,
        0,
        True,
        SHIFT,
        BLOCK_COLS=BLOCK_COLS,
        v2p_ptr=v2p_full,
        PAGE_MULT=1,
        SWA_MAPPING_IS_V2P=True,
        num_warps=4,
        num_stages=3,
    )
    torch.cuda.synchronize()

    for i in range(bs):
        live = int((int(cache_seqlens[i]) + PAGE_SIZE - 1) // PAGE_SIZE)
        assert (page_table[i, :live] >= 0).all(), page_table[i, :live].tolist()
        assert (swa_page_table[i, :live] >= 0).all(), swa_page_table[i, :live].tolist()
    # In-window pages must keep their real translation, not get clamped away.
    assert swa_page_table[0, 5:9].tolist() == [5, 6, 7, 8]
    assert page_table[1, :2].tolist() == [16, 17]
