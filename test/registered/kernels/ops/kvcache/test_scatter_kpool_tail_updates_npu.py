"""Accuracy test for scatter_kpool_tail_updates_npu against the GPU reference.

The NPU kernel lives in ``sgl_kernel_npu.operators.mem_cache.scatter_kpool_tail_updates_npu`` and is a
drop-in replacement for ``kpool_fp8_index.scatter_kpool_tail_updates``. This test
calls both implementations on identical inputs and compares the in-place updated
``tail_k`` / ``tail_score`` tensors.
"""

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="full-1-npu-a3", nightly=True)

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.operators.mem_cache import (
    scatter_kpool_tail_updates_npu,
)

from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    scatter_kpool_tail_updates,
)

DEVICE = "npu"


# ============================================================================
# region auxiliary functions
# ============================================================================


def _assert_close_by_dtype(cal, ref):
    """按 dtype 自动选择精度容限（precision.md 标准）。"""
    assert cal.dtype == ref.dtype, f"dtype mismatch: {cal.dtype} vs {ref.dtype}"
    if cal.dtype == torch.float32:
        torch.testing.assert_close(ref, cal, rtol=1e-5, atol=1e-5, equal_nan=True)
    elif cal.dtype == torch.float16:
        torch.testing.assert_close(ref, cal, rtol=1e-3, atol=1e-3, equal_nan=True)
    elif cal.dtype == torch.bfloat16:
        torch.testing.assert_close(ref, cal, rtol=5e-3, atol=5e-3, equal_nan=True)
    elif cal.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
        assert torch.equal(cal, ref), f"Integer tensors not equal for {cal.dtype}"
    elif cal.dtype == torch.bool:
        assert torch.equal(cal, ref), "Boolean tensors not equal"
    else:
        raise ValueError(f"Unsupported dtype: {cal.dtype}")


def _create_test_inputs(
    n_rows,
    total_tokens,
    req_pool_size,
    POOL_SIZE,
    TAIL_SIZE,
    HEAD_DIM,
    dtype,
    seed=20,
):
    """生成随机测试数据, 保证 scatter 写入不越界。"""
    torch.manual_seed(seed)

    chunk_k = torch.empty(total_tokens, HEAD_DIM, dtype=dtype, device=DEVICE).normal_(
        mean=0.0, std=0.5
    )
    chunk_score = torch.empty(
        total_tokens, HEAD_DIM, dtype=dtype, device=DEVICE
    ).normal_(mean=0.0, std=0.5)

    tail_k = torch.empty(
        req_pool_size, TAIL_SIZE, HEAD_DIM, dtype=dtype, device=DEVICE
    ).normal_(mean=0.0, std=0.5)
    tail_score = torch.empty(
        req_pool_size, TAIL_SIZE, HEAD_DIM, dtype=dtype, device=DEVICE
    ).normal_(mean=0.0, std=0.5)

    # 每个 row 分配唯一 req pool index (避免并行写入冲突)
    assert req_pool_size >= n_rows, f"req_pool_size={req_pool_size} < n_rows={n_rows}"
    req_pool_idx = torch.arange(n_rows, dtype=torch.int64, device=DEVICE)
    # dst_logical_start: tail 环形缓冲区逻辑起始 slot
    dst_logical_start = torch.randint(
        0, TAIL_SIZE, (n_rows,), dtype=torch.int32, device=DEVICE
    )
    # n_write: 每行需要写入的 slot 数, 范围 [1, POOL_SIZE]
    n_write_vals = torch.randint(
        1, POOL_SIZE + 1, (n_rows,), dtype=torch.int32, device=DEVICE
    )

    # chunk_src_start: 每行在 chunk 中的起始偏移, 确保不重叠且不越界
    cumsum = torch.cumsum(n_write_vals.to(torch.int64), dim=0)
    chunk_src_start = torch.zeros(n_rows, dtype=torch.int64, device=DEVICE)
    chunk_src_start[1:] = cumsum[:-1]
    assert chunk_src_start[-1] + n_write_vals[-1] <= total_tokens, (
        f"total_tokens={total_tokens} too small, need >={chunk_src_start[-1] + n_write_vals[-1]}"
    )

    return (
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write_vals,
    )


# endregion

# ============================================================================
# region precision test
# ============================================================================


def _check_op(
    n_rows, total_tokens, req_pool_size, POOL_SIZE, TAIL_SIZE, HEAD_DIM, dtype
):
    # 1. 参数合法性检查
    max_slots = n_rows * POOL_SIZE
    if total_tokens < max_slots:
        pytest.skip(f"total_tokens={total_tokens} < n_rows*POOL_SIZE={max_slots}")

    # 2. 生成随机测试数据
    (
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
    ) = _create_test_inputs(
        n_rows=n_rows,
        total_tokens=total_tokens,
        req_pool_size=req_pool_size,
        POOL_SIZE=POOL_SIZE,
        TAIL_SIZE=TAIL_SIZE,
        HEAD_DIM=HEAD_DIM,
        dtype=dtype,
    )

    # 保存 tail 副本用于 golden reference
    tail_k_ref = tail_k.clone()
    tail_score_ref = tail_score.clone()

    # 3. 构造 pool mock (scatter_kpool_tail_updates 签名要求)
    pool = SimpleNamespace(index_kpool=POOL_SIZE)

    # 4. NPU 算子结果 (原地更新)
    scatter_kpool_tail_updates_npu(
        pool,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
    )

    # 5. 参考结果 (golden) — GPU 原始实现
    scatter_kpool_tail_updates(
        pool,
        chunk_k,
        chunk_score,
        tail_k_ref,
        tail_score_ref,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
    )

    # 6. 精度对比
    _assert_close_by_dtype(tail_k, tail_k_ref)
    _assert_close_by_dtype(tail_score, tail_score_ref)
    print(
        f"[PASSED] n_rows={n_rows}, TAIL_SIZE={TAIL_SIZE}, "
        f"POOL_SIZE={POOL_SIZE}, dtype={dtype}"
    )


# endregion

# ============================================================================
# region main
# ============================================================================


@pytest.mark.parametrize(
    "n_rows,total_tokens,req_pool_size,POOL_SIZE,TAIL_SIZE,HEAD_DIM,dtype",
    [
        (1, 16, 4, 4, 4, 128, torch.bfloat16),
        (8, 64, 16, 4, 8, 128, torch.bfloat16),
        (8, 64, 16, 4, 12, 128, torch.bfloat16),
        (64, 320, 128, 4, 8, 128, torch.bfloat16),
        (128, 640, 256, 4, 4, 128, torch.bfloat16),
        (128, 640, 256, 4, 12, 128, torch.bfloat16),
        (4, 32, 8, 4, 8, 128, torch.float16),
    ],
)
def test_scatter_kpool_tail_updates(
    n_rows, total_tokens, req_pool_size, POOL_SIZE, TAIL_SIZE, HEAD_DIM, dtype
):
    _check_op(
        n_rows, total_tokens, req_pool_size, POOL_SIZE, TAIL_SIZE, HEAD_DIM, dtype
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
