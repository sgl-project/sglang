"""Regression tests for the DSA KPool decode JIT launch."""

from collections import defaultdict

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("device", ["cuda"])
def test_kpool_decode_does_not_specialize_block_table_width(device):
    """Different page-table widths must reuse one Triton specialization."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the DSA KPool kernel")

    from sglang.srt.layers.attention.dsa import kpool_fp8_index as kpool

    kernel = kpool._kpool_decode_update_and_maybe_write_cache_kernel
    block_table_cols = next(
        param for param in kernel.params if param.name == "block_table_cols"
    )
    assert not block_table_cols.is_constexpr
    assert block_table_cols.do_not_specialize

    class Pool:
        page_size = 64
        index_page_size = 64
        index_head_dim = 128
        index_kpool = 8
        tail_extra_slots = 0
        slots_per_page = 64

    pool = Pool()
    tail_k = torch.zeros(
        (1, pool.index_kpool, pool.index_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    tail_score = torch.zeros_like(tail_k)
    key = torch.zeros((1, pool.index_head_dim), dtype=torch.bfloat16, device=device)
    slot_score = torch.zeros_like(key)
    ape = torch.zeros(
        (pool.index_kpool, pool.index_head_dim), dtype=torch.float32, device=device
    )
    req_pool_indices = torch.zeros((1,), dtype=torch.int32, device=device)
    positions = torch.full((1,), pool.index_kpool - 1, dtype=torch.int32, device=device)
    seq_lens = torch.full((1,), pool.index_kpool, dtype=torch.int32, device=device)
    out_cache_loc = torch.ones((1,), dtype=torch.int32, device=device)

    # Isolate this check from specializations left by other tests in the same
    # worker. The original constexpr argument creates one entry per width.
    old_device_caches = kernel.device_caches
    kernel.device_caches = defaultdict(kernel.create_binder)
    try:
        block_table_storage = torch.zeros((1, 17), dtype=torch.int32, device=device)
        for width in (1, 2, 11, 17):
            # Keep the pointer and strides stable so the cache count measures
            # the width argument rather than pointer-alignment specialization.
            block_tables = block_table_storage[:, :width]
            buf = torch.empty((1, 8192), dtype=torch.uint8, device=device)
            kpool.kpool_decode_update_and_maybe_write_cache(
                pool,
                buf,
                tail_k,
                tail_score,
                key,
                slot_score,
                ape,
                block_tables,
                req_pool_indices,
                positions,
                seq_lens,
                out_cache_loc,
            )
            torch.cuda.synchronize()

        cache = kernel.device_caches[torch.cuda.current_device()][0]
        assert len(cache) == 1
    finally:
        kernel.device_caches = old_device_caches
