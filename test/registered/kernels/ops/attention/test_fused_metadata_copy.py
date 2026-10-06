"""Compare single- and multi-backend metadata copies with PyTorch references."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=100, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=100, suite="nightly-amd-kernel-1-gpu", nightly=True)

# =============================================================================
# Helper Functions
# =============================================================================


def create_test_metadata(
    bs: int,
    max_len: int,
    max_seqlen_k: int,
    seqlens_expanded_size: int,
    has_real_page_table: bool = False,
    has_flashmla: bool = False,
    device: str = "cuda",
):
    """Create test metadata tensors matching DSA backend structure."""
    # Basic tensors (always present)
    cache_seqlens_src = torch.randint(
        1, max_len, (bs,), dtype=torch.int32, device=device
    )
    cu_seqlens_k_src = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cu_seqlens_k_src[1:] = torch.cumsum(cache_seqlens_src, dim=0)

    page_indices_src = torch.randint(
        0, 1000, (bs, max_len), dtype=torch.int32, device=device
    )
    dsa_cache_seqlens_src = torch.randint(
        1, max_len, (seqlens_expanded_size,), dtype=torch.int32, device=device
    )
    seqlens_expanded_src = torch.randint(
        1, max_seqlen_k, (seqlens_expanded_size,), dtype=torch.int32, device=device
    )
    dsa_cu_seqlens_k_src = torch.zeros(
        seqlens_expanded_size + 1, dtype=torch.int32, device=device
    )
    dsa_cu_seqlens_k_src[1:] = torch.cumsum(dsa_cache_seqlens_src, dim=0)

    # Destination tensors
    cache_seqlens_dst = torch.zeros(bs, dtype=torch.int32, device=device)
    cu_seqlens_k_dst = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    page_table_1_dst = torch.zeros((bs, max_len + 16), dtype=torch.int32, device=device)
    dsa_cache_seqlens_dst = torch.zeros(
        seqlens_expanded_size, dtype=torch.int32, device=device
    )
    dsa_seqlens_expanded_dst = torch.zeros(
        seqlens_expanded_size, dtype=torch.int32, device=device
    )
    dsa_cu_seqlens_k_dst = torch.zeros(
        seqlens_expanded_size + 1, dtype=torch.int32, device=device
    )

    # Optional tensors
    real_page_table_src = None
    real_page_table_dst = None
    if has_real_page_table:
        real_page_table_cols = max_len // 2
        real_page_table_src = torch.randint(
            0, 1000, (bs, real_page_table_cols), dtype=torch.int32, device=device
        )
        real_page_table_dst = torch.zeros(
            (bs, real_page_table_cols + 8), dtype=torch.int32, device=device
        )

    flashmla_num_splits_src = None
    flashmla_num_splits_dst = None
    flashmla_metadata_src = None
    flashmla_metadata_dst = None
    if has_flashmla:
        flashmla_num_splits_src = torch.randint(
            1, 10, (seqlens_expanded_size + 1,), dtype=torch.int32, device=device
        )
        flashmla_num_splits_dst = torch.zeros(
            seqlens_expanded_size + 1, dtype=torch.int32, device=device
        )
        # FlashMLA metadata is typically (num_sm_parts, TileSchedulerMetaDataSize)
        # For testing, we use a simplified size
        flashmla_metadata_size = 128
        flashmla_metadata_src = torch.randint(
            0, 100, (flashmla_metadata_size,), dtype=torch.int32, device=device
        )
        flashmla_metadata_dst = torch.zeros(
            flashmla_metadata_size, dtype=torch.int32, device=device
        )

    return {
        "src": {
            "cache_seqlens": cache_seqlens_src,
            "cu_seqlens_k": cu_seqlens_k_src,
            "page_indices": page_indices_src,
            "dsa_cache_seqlens": dsa_cache_seqlens_src,
            "seqlens_expanded": seqlens_expanded_src,
            "dsa_cu_seqlens_k": dsa_cu_seqlens_k_src,
            "real_page_table": real_page_table_src,
            "flashmla_num_splits": flashmla_num_splits_src,
            "flashmla_metadata": flashmla_metadata_src,
        },
        "dst": {
            "cache_seqlens": cache_seqlens_dst,
            "cu_seqlens_k": cu_seqlens_k_dst,
            "page_table_1": page_table_1_dst,
            "dsa_cache_seqlens": dsa_cache_seqlens_dst,
            "dsa_seqlens_expanded": dsa_seqlens_expanded_dst,
            "dsa_cu_seqlens_k": dsa_cu_seqlens_k_dst,
            "real_page_table": real_page_table_dst,
            "flashmla_num_splits": flashmla_num_splits_dst,
            "flashmla_metadata": flashmla_metadata_dst,
        },
    }


def reference_copy_decode(src, dst, max_len):
    """Reference implementation: individual .copy_() for DECODE mode."""
    bs = src["cache_seqlens"].shape[0]
    dst["cache_seqlens"].copy_(src["cache_seqlens"])
    dst["cu_seqlens_k"][1:].copy_(src["cu_seqlens_k"][1:])
    dst["page_table_1"][:, :max_len].copy_(src["page_indices"])
    dst["dsa_cache_seqlens"].copy_(src["dsa_cache_seqlens"])
    dst["dsa_cu_seqlens_k"][1 : bs + 1].copy_(src["dsa_cu_seqlens_k"][1 : bs + 1])

    if src["real_page_table"] is not None:
        rows, cols = src["real_page_table"].shape
        dst["real_page_table"][:rows, :cols].copy_(src["real_page_table"])

    if src["flashmla_num_splits"] is not None:
        flashmla_size = bs + 1
        dst["flashmla_num_splits"][:flashmla_size].copy_(
            src["flashmla_num_splits"][:flashmla_size]
        )

    if src["flashmla_metadata"] is not None:
        dst["flashmla_metadata"].copy_(src["flashmla_metadata"])


# =============================================================================
# Single-Backend Kernel Tests
# =============================================================================


def test_fused_metadata_copy_dtype_validation():
    """Test that dtype validation rejects non-int32 tensors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from sglang.kernels.ops.attention.fused_metadata_copy import (
        fused_metadata_copy_cuda,
    )

    bs = 2
    max_len = 128
    max_seqlen_k = 256
    seqlens_expanded_size = bs
    device = "cuda"

    # Create tensors with WRONG dtype (int64 instead of int32)
    cache_seqlens_src_wrong = torch.randint(
        1, max_len, (bs,), dtype=torch.int64, device=device
    )
    cu_seqlens_k_src = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    page_indices_src = torch.randint(
        0, 1000, (bs, max_len), dtype=torch.int32, device=device
    )
    dsa_cache_seqlens_src = torch.randint(
        1, max_len, (seqlens_expanded_size,), dtype=torch.int32, device=device
    )
    seqlens_expanded_src = torch.randint(
        1, max_seqlen_k, (seqlens_expanded_size,), dtype=torch.int32, device=device
    )
    dsa_cu_seqlens_k_src = torch.zeros(
        seqlens_expanded_size + 1, dtype=torch.int32, device=device
    )

    # Destination tensors (correct dtype)
    cache_seqlens_dst = torch.zeros(bs, dtype=torch.int32, device=device)
    cu_seqlens_k_dst = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    page_table_1_dst = torch.zeros((bs, max_len + 16), dtype=torch.int32, device=device)
    dsa_cache_seqlens_dst = torch.zeros(
        seqlens_expanded_size, dtype=torch.int32, device=device
    )
    dsa_seqlens_expanded_dst = torch.zeros(
        seqlens_expanded_size, dtype=torch.int32, device=device
    )
    dsa_cu_seqlens_k_dst = torch.zeros(
        seqlens_expanded_size + 1, dtype=torch.int32, device=device
    )

    # Test 1: Wrong dtype for source tensor should raise RuntimeError
    with pytest.raises(RuntimeError, match="must have dtype int32"):
        fused_metadata_copy_cuda(
            cache_seqlens_src_wrong,  # Wrong dtype: int64
            cu_seqlens_k_src,
            page_indices_src,
            dsa_cache_seqlens_src,
            seqlens_expanded_src,
            dsa_cu_seqlens_k_src,
            None,  # real_page_table_src
            None,  # flashmla_num_splits_src
            None,  # flashmla_metadata_src
            cache_seqlens_dst,
            cu_seqlens_k_dst,
            page_table_1_dst,
            dsa_cache_seqlens_dst,
            dsa_seqlens_expanded_dst,
            dsa_cu_seqlens_k_dst,
            None,  # real_page_table_dst
            None,  # flashmla_num_splits_dst
            None,  # flashmla_metadata_dst
            0,  # forward_mode
            bs,
            max_len,
            max_seqlen_k,
            seqlens_expanded_size,
        )

    # Test 2: Wrong dtype for destination tensor should also raise RuntimeError
    cache_seqlens_src = torch.randint(
        1, max_len, (bs,), dtype=torch.int32, device=device
    )
    cache_seqlens_dst_wrong = torch.zeros(bs, dtype=torch.int64, device=device)

    with pytest.raises(RuntimeError, match="must have dtype int32"):
        fused_metadata_copy_cuda(
            cache_seqlens_src,
            cu_seqlens_k_src,
            page_indices_src,
            dsa_cache_seqlens_src,
            seqlens_expanded_src,
            dsa_cu_seqlens_k_src,
            None,
            None,
            None,
            cache_seqlens_dst_wrong,  # Wrong dtype: int64
            cu_seqlens_k_dst,
            page_table_1_dst,
            dsa_cache_seqlens_dst,
            dsa_seqlens_expanded_dst,
            dsa_cu_seqlens_k_dst,
            None,
            None,
            None,
            0,
            bs,
            max_len,
            max_seqlen_k,
            seqlens_expanded_size,
        )


@pytest.mark.parametrize(
    "bs,has_real_page_table,has_flashmla",
    [
        (bs, page, mla)
        for bs in (1, 2, 4, 8)
        for page in (False, True)
        for mla in (False, True)
    ]
    + [(16, True, True), (32, True, True)],
)
def test_fused_metadata_copy(bs, has_real_page_table, has_flashmla):
    """Test fused metadata copy kernel against reference implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from sglang.kernels.ops.attention.fused_metadata_copy import (
        fused_metadata_copy_cuda,
    )

    forward_mode = 0  # DECODE
    max_len = 128
    max_seqlen_k = 256
    seqlens_expanded_size = bs

    # Create test data
    data = create_test_metadata(
        bs=bs,
        max_len=max_len,
        max_seqlen_k=max_seqlen_k,
        seqlens_expanded_size=seqlens_expanded_size,
        has_real_page_table=has_real_page_table,
        has_flashmla=has_flashmla,
    )

    # Create separate destination tensors for reference and fused kernel
    dst_ref = {k: v.clone() if v is not None else None for k, v in data["dst"].items()}
    dst_fused = {
        k: v.clone() if v is not None else None for k, v in data["dst"].items()
    }

    reference_copy_decode(data["src"], dst_ref, max_len)

    # Run fused kernel
    fused_metadata_copy_cuda(
        data["src"]["cache_seqlens"],
        data["src"]["cu_seqlens_k"],
        data["src"]["page_indices"],
        data["src"]["dsa_cache_seqlens"],
        data["src"]["seqlens_expanded"],
        data["src"]["dsa_cu_seqlens_k"],
        data["src"]["real_page_table"],
        data["src"]["flashmla_num_splits"],
        data["src"]["flashmla_metadata"],
        dst_fused["cache_seqlens"],
        dst_fused["cu_seqlens_k"],
        dst_fused["page_table_1"],
        dst_fused["dsa_cache_seqlens"],
        dst_fused["dsa_seqlens_expanded"],
        dst_fused["dsa_cu_seqlens_k"],
        dst_fused["real_page_table"],
        dst_fused["flashmla_num_splits"],
        dst_fused["flashmla_metadata"],
        forward_mode,
        bs,
        max_len,
        max_seqlen_k,
        seqlens_expanded_size,
    )

    for key, expected in dst_ref.items():
        if expected is not None:
            torch.testing.assert_close(
                dst_fused[key], expected, rtol=0, atol=0, msg=key
            )


# =============================================================================
# Multi-Backend Kernel Tests
# =============================================================================


def create_test_metadata_multi(
    bs: int,
    max_len: int,
    seqlens_expanded_size: int,
    has_real_page_table: bool = False,
    has_flashmla: bool = False,
    device: str = "cuda",
):
    """Create test metadata tensors for multi-backend testing."""
    # Source tensors (precomputed metadata)
    cache_seqlens_src = torch.randint(
        1, max_len, (bs,), dtype=torch.int32, device=device
    )
    cu_seqlens_k_src = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cu_seqlens_k_src[1:] = torch.cumsum(cache_seqlens_src, dim=0)

    page_indices_src = torch.randint(
        0, 1000, (bs, max_len), dtype=torch.int32, device=device
    )
    dsa_cache_seqlens_src = torch.randint(
        1, max_len, (seqlens_expanded_size,), dtype=torch.int32, device=device
    )
    dsa_cu_seqlens_k_src = torch.zeros(
        seqlens_expanded_size + 1, dtype=torch.int32, device=device
    )
    dsa_cu_seqlens_k_src[1:] = torch.cumsum(dsa_cache_seqlens_src, dim=0)

    # Optional tensors
    real_page_table_src = None
    if has_real_page_table:
        real_page_table_cols = max_len // 2
        real_page_table_src = torch.randint(
            0, 1000, (bs, real_page_table_cols), dtype=torch.int32, device=device
        )

    flashmla_num_splits_src = None
    flashmla_metadata_src = None
    if has_flashmla:
        flashmla_num_splits_src = torch.randint(
            1, 10, (seqlens_expanded_size + 1,), dtype=torch.int32, device=device
        )
        flashmla_metadata_size = 128
        flashmla_metadata_src = torch.randint(
            0, 100, (flashmla_metadata_size,), dtype=torch.int32, device=device
        )

    # Create destination tensors for 3 backends
    def create_dst_tensors():
        cache_seqlens_dst = torch.zeros(bs, dtype=torch.int32, device=device)
        cu_seqlens_k_dst = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        page_table_1_dst = torch.zeros(
            (bs, max_len + 16), dtype=torch.int32, device=device
        )
        dsa_cache_seqlens_dst = torch.zeros(
            seqlens_expanded_size, dtype=torch.int32, device=device
        )
        dsa_cu_seqlens_k_dst = torch.zeros(
            seqlens_expanded_size + 1, dtype=torch.int32, device=device
        )

        real_page_table_dst = None
        if has_real_page_table:
            real_page_table_cols = max_len // 2
            real_page_table_dst = torch.zeros(
                (bs, real_page_table_cols + 8), dtype=torch.int32, device=device
            )

        flashmla_num_splits_dst = None
        flashmla_metadata_dst = None
        if has_flashmla:
            flashmla_num_splits_dst = torch.zeros(
                seqlens_expanded_size + 1, dtype=torch.int32, device=device
            )
            flashmla_metadata_size = 128
            flashmla_metadata_dst = torch.zeros(
                flashmla_metadata_size, dtype=torch.int32, device=device
            )

        return {
            "cache_seqlens_int32": cache_seqlens_dst,
            "cu_seqlens_k": cu_seqlens_k_dst,
            "page_table_1": page_table_1_dst,
            "dsa_cache_seqlens_int32": dsa_cache_seqlens_dst,
            "dsa_cu_seqlens_k": dsa_cu_seqlens_k_dst,
            "real_page_table": real_page_table_dst,
            "flashmla_num_splits": flashmla_num_splits_dst,
            "flashmla_metadata": flashmla_metadata_dst,
        }

    return {
        "src": {
            "cache_seqlens": cache_seqlens_src,
            "cu_seqlens_k": cu_seqlens_k_src,
            "page_indices": page_indices_src,
            "dsa_cache_seqlens": dsa_cache_seqlens_src,
            "dsa_cu_seqlens_k": dsa_cu_seqlens_k_src,
            "real_page_table": real_page_table_src,
            "flashmla_num_splits": flashmla_num_splits_src,
            "flashmla_metadata": flashmla_metadata_src,
        },
        "dst0": create_dst_tensors(),
        "dst1": create_dst_tensors(),
        "dst2": create_dst_tensors(),
    }


def reference_copy_for_loop(src, dst_list, bs, max_len):
    """Reference implementation: for-loop calling copy for each backend."""
    for dst in dst_list:
        # Simulate what init_forward_metadata_replay_cuda_graph_from_precomputed does
        dst["cache_seqlens_int32"].copy_(src["cache_seqlens"])
        dst["cu_seqlens_k"][1:].copy_(src["cu_seqlens_k"][1:])
        dst["page_table_1"][:, :max_len].copy_(src["page_indices"])
        dst["dsa_cache_seqlens_int32"].copy_(src["dsa_cache_seqlens"])
        dst["dsa_cu_seqlens_k"][1 : bs + 1].copy_(src["dsa_cu_seqlens_k"][1 : bs + 1])

        if src["real_page_table"] is not None:
            rows, cols = src["real_page_table"].shape
            dst["real_page_table"][:rows, :cols].copy_(src["real_page_table"])

        if src["flashmla_num_splits"] is not None:
            flashmla_size = bs + 1
            dst["flashmla_num_splits"][:flashmla_size].copy_(
                src["flashmla_num_splits"][:flashmla_size]
            )

        if src["flashmla_metadata"] is not None:
            dst["flashmla_metadata"].copy_(src["flashmla_metadata"])


def test_fused_metadata_copy_multi_dtype_validation():
    """Test that dtype validation rejects non-int32 tensors for multi-backend kernel."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from sglang.kernels.ops.attention.fused_metadata_copy import (
        fused_metadata_copy_multi_cuda,
    )

    bs = 2
    max_len = 128
    seqlens_expanded_size = bs
    device = "cuda"

    # Create source tensors - one with WRONG dtype
    cache_seqlens_src_wrong = torch.randint(
        1,
        max_len,
        (bs,),
        dtype=torch.int64,
        device=device,  # Wrong dtype!
    )
    cu_seqlens_k_src = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    page_indices_src = torch.randint(
        0, 1000, (bs, max_len), dtype=torch.int32, device=device
    )
    dsa_cache_seqlens_src = torch.randint(
        1, max_len, (seqlens_expanded_size,), dtype=torch.int32, device=device
    )
    dsa_cu_seqlens_k_src = torch.zeros(
        seqlens_expanded_size + 1, dtype=torch.int32, device=device
    )

    # Create destination tensors for 3 backends (all correct dtype)
    def create_dst():
        return {
            "cache_seqlens": torch.zeros(bs, dtype=torch.int32, device=device),
            "cu_seqlens_k": torch.zeros(bs + 1, dtype=torch.int32, device=device),
            "page_table_1": torch.zeros(
                (bs, max_len + 16), dtype=torch.int32, device=device
            ),
            "dsa_cache_seqlens": torch.zeros(
                seqlens_expanded_size, dtype=torch.int32, device=device
            ),
            "dsa_cu_seqlens_k": torch.zeros(
                seqlens_expanded_size + 1, dtype=torch.int32, device=device
            ),
        }

    dst0 = create_dst()
    dst1 = create_dst()
    dst2 = create_dst()

    # Test: Wrong dtype for source tensor should raise RuntimeError
    with pytest.raises(RuntimeError, match="must have dtype int32"):
        fused_metadata_copy_multi_cuda(
            cache_seqlens_src_wrong,  # Wrong dtype: int64
            cu_seqlens_k_src,
            page_indices_src,
            dsa_cache_seqlens_src,
            dsa_cu_seqlens_k_src,
            None,  # real_page_table_src
            None,  # flashmla_num_splits_src
            None,  # flashmla_metadata_src
            # Backend 0
            dst0["cache_seqlens"],
            dst0["cu_seqlens_k"],
            dst0["page_table_1"],
            dst0["dsa_cache_seqlens"],
            dst0["dsa_cu_seqlens_k"],
            None,
            None,
            None,
            # Backend 1
            dst1["cache_seqlens"],
            dst1["cu_seqlens_k"],
            dst1["page_table_1"],
            dst1["dsa_cache_seqlens"],
            dst1["dsa_cu_seqlens_k"],
            None,
            None,
            None,
            # Backend 2
            dst2["cache_seqlens"],
            dst2["cu_seqlens_k"],
            dst2["page_table_1"],
            dst2["dsa_cache_seqlens"],
            dst2["dsa_cu_seqlens_k"],
            None,
            None,
            None,
            # Parameters
            bs,
            max_len,
            seqlens_expanded_size,
        )


@pytest.mark.parametrize(
    "bs,has_real_page_table,has_flashmla",
    [
        (bs, page, mla)
        for bs in (1, 2, 4, 8, 16)
        for page in (False, True)
        for mla in (False, True)
    ]
    + [(32, True, True), (64, True, True)],
)
def test_fused_metadata_copy_multi(bs, has_real_page_table, has_flashmla):
    """Test fused multi-backend metadata copy kernel against for-loop version."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from sglang.kernels.ops.attention.fused_metadata_copy import (
        fused_metadata_copy_multi_cuda,
    )

    max_len = 128
    seqlens_expanded_size = bs

    # Create test data
    data = create_test_metadata_multi(
        bs=bs,
        max_len=max_len,
        seqlens_expanded_size=seqlens_expanded_size,
        has_real_page_table=has_real_page_table,
        has_flashmla=has_flashmla,
    )

    dst_ref = [
        {k: v.clone() if v is not None else None for k, v in data[f"dst{i}"].items()}
        for i in range(3)
    ]
    dst_fused = [
        {k: v.clone() if v is not None else None for k, v in data[f"dst{i}"].items()}
        for i in range(3)
    ]
    reference_copy_for_loop(data["src"], dst_ref, bs, max_len)

    fused_metadata_copy_multi_cuda(
        # Source tensors
        data["src"]["cache_seqlens"],
        data["src"]["cu_seqlens_k"],
        data["src"]["page_indices"],
        data["src"]["dsa_cache_seqlens"],
        data["src"]["dsa_cu_seqlens_k"],
        data["src"]["real_page_table"],
        data["src"]["flashmla_num_splits"],
        data["src"]["flashmla_metadata"],
        # Destination tensors for backend 0
        dst_fused[0]["cache_seqlens_int32"],
        dst_fused[0]["cu_seqlens_k"],
        dst_fused[0]["page_table_1"],
        dst_fused[0]["dsa_cache_seqlens_int32"],
        dst_fused[0]["dsa_cu_seqlens_k"],
        dst_fused[0]["real_page_table"],
        dst_fused[0]["flashmla_num_splits"],
        dst_fused[0]["flashmla_metadata"],
        # Destination tensors for backend 1
        dst_fused[1]["cache_seqlens_int32"],
        dst_fused[1]["cu_seqlens_k"],
        dst_fused[1]["page_table_1"],
        dst_fused[1]["dsa_cache_seqlens_int32"],
        dst_fused[1]["dsa_cu_seqlens_k"],
        dst_fused[1]["real_page_table"],
        dst_fused[1]["flashmla_num_splits"],
        dst_fused[1]["flashmla_metadata"],
        # Destination tensors for backend 2
        dst_fused[2]["cache_seqlens_int32"],
        dst_fused[2]["cu_seqlens_k"],
        dst_fused[2]["page_table_1"],
        dst_fused[2]["dsa_cache_seqlens_int32"],
        dst_fused[2]["dsa_cu_seqlens_k"],
        dst_fused[2]["real_page_table"],
        dst_fused[2]["flashmla_num_splits"],
        dst_fused[2]["flashmla_metadata"],
        # Parameters
        bs,
        max_len,
        seqlens_expanded_size,
    )
    for backend_idx, (expected, actual) in enumerate(zip(dst_ref, dst_fused)):
        for key, tensor in expected.items():
            if tensor is not None:
                torch.testing.assert_close(
                    actual[key],
                    tensor,
                    rtol=0,
                    atol=0,
                    msg=f"Backend {backend_idx} {key}",
                )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
