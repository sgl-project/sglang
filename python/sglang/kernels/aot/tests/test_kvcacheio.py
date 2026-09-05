import sys

import pytest
import torch
from sgl_kernel.kvcacheio import (
    transfer_embedding_ranges_direct,
    transfer_kv_all_layer,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_all_layer_lf_pfdhg,
    transfer_kv_all_layer_lf_ph,
    transfer_kv_all_layer_mla,
    transfer_kv_direct,
    transfer_kv_per_layer,
    transfer_kv_per_layer_direct_pf_lf,
    transfer_kv_per_layer_mla,
    transfer_kv_per_layer_pfdhg_lf,
)

from sglang.srt.utils import get_cuda_version, is_hip

# Skip entire module on CUDA 13.x — segfaults in transfer_kv kernel.
# Reference failure: https://github.com/sgl-project/sglang/actions/runs/24600433057/job/71938317621?pr=23119
pytestmark = pytest.mark.skipif(
    get_cuda_version()[0] >= 13,
    reason="test_kvcacheio segfaults on CUDA 13.x (sgl-kernel bug)",
)


def ref_copy_with_indices(src_pool, dst_pool, src_indices, dst_indices):
    dst_pool[dst_indices] = src_pool[src_indices].to(dst_pool.device)


def ref_copy_with_indices_pf_direct(
    src_pool, dst_pool, src_indices, dst_indices, page_size, layer_id, lf_to_pf=False
):
    if lf_to_pf:
        for i in range(0, len(src_indices), page_size):
            dst_pool[dst_indices[i] // page_size][layer_id] = src_pool[layer_id][
                src_indices[i : i + page_size]
            ].to(dst_pool.device)
    else:
        for i in range(0, len(src_indices), page_size):
            dst_pool[layer_id][dst_indices[i : i + page_size]] = src_pool[
                src_indices[i] // page_size
            ][layer_id].to(dst_pool.device)


def ref_copy_with_indices_page_head(
    src_pool,
    dst_pool,
    src_indices,
    dst_indices,
    page_size,
    layer_id,
    head_num,
    lf_to_ph=False,
):
    if lf_to_ph:
        for head_id in range(head_num):
            for i in range(0, len(src_indices)):
                dst_pool[dst_indices[i] // page_size][head_id][
                    dst_indices[i] % page_size
                ][layer_id] = src_pool[layer_id][src_indices[i]][head_id].to(
                    dst_pool.device
                )
    else:
        for head_id in range(head_num):
            for i in range(0, len(src_indices)):
                dst_pool[layer_id][dst_indices[i]][head_id] = src_pool[
                    src_indices[i] // page_size
                ][head_id][src_indices[i] % page_size][layer_id].to(dst_pool.device)


def ref_copy_embedding_ranges(src, dst, src_starts, dst_starts, lengths):
    for src_start, dst_start, length in zip(src_starts, dst_starts, lengths):
        dst[dst_start : dst_start + length].copy_(
            src[src_start : src_start + length], non_blocking=True
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(is_hip(), reason="This test covers the CUDA batch-copy op")
@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_transfer_embedding_ranges_direct(direction: str):
    dtype = torch.bfloat16
    embedding_dim = 37
    page_size = 4
    fragmented_starts = [1, 11, 23]
    contiguous_starts = [2, 6, 10]
    lengths = [page_size, page_size, 2]
    host_rows = 28
    device_rows = 16

    host_values = torch.arange(host_rows * embedding_dim, dtype=torch.float32).reshape(
        host_rows, embedding_dim
    )
    device_values = torch.arange(
        device_rows * embedding_dim, dtype=torch.float32
    ).reshape(device_rows, embedding_dim)

    if direction == "h2d":
        src = host_values.to(dtype).pin_memory()
        direct_dst = torch.full(
            (device_rows, embedding_dim), -1, dtype=dtype, device="cuda"
        )
        reference_dst = torch.full_like(direct_dst, -1)
        src_starts, dst_starts = fragmented_starts, contiguous_starts
    else:
        src = device_values.to(dtype).to("cuda")
        direct_dst = torch.full(
            (host_rows, embedding_dim), -1, dtype=dtype, pin_memory=True
        )
        reference_dst = torch.full(
            (host_rows, embedding_dim), -1, dtype=dtype, pin_memory=True
        )
        src_starts, dst_starts = contiguous_starts, fragmented_starts

    torch.cuda.synchronize()
    copy_stream = torch.cuda.Stream()
    assert copy_stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    with torch.cuda.stream(copy_stream):
        ref_copy_embedding_ranges(src, reference_dst, src_starts, dst_starts, lengths)
        transfer_embedding_ranges_direct(
            src, direct_dst, src_starts, dst_starts, lengths
        )
        completion_event = torch.cuda.Event()
        completion_event.record(copy_stream)

    completion_event.synchronize()
    torch.testing.assert_close(direct_dst, reference_dst)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [1, 128, 1024])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("item_size", [256])
@pytest.mark.parametrize("total_items_in_pool", [10240])
@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("all_layers", [False, True])
def test_transfer_kv(
    dtype: torch.dtype,
    num_items_to_transfer: int,
    item_size: int,
    page_size: int,
    total_items_in_pool: int,
    is_mla: bool,
    all_layers: bool,
):
    """
    Tests the per-layer transfer functions, treating tensors as memory pools.
    """

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    device = "cuda"
    torch.cuda.manual_seed(42)

    num_layers = 4  # A small number of layers for pool creation

    total_pages_in_pool = total_items_in_pool // page_size
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return
    page_indices = torch.randperm(total_pages_in_pool, dtype=torch.int64)
    src_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[:num_pages_to_transfer]
        ]
    )
    src_indices_device = src_indices_host.to(device)
    dst_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[num_pages_to_transfer : 2 * num_pages_to_transfer]
        ]
    )
    dst_indices_device = dst_indices_host.to(device)

    # Prepare memory pools based on whether it's an MLA case.
    if is_mla:
        src_pool_host = torch.randn(
            num_layers, total_items_in_pool, item_size
        ).pin_memory()
        dst_pool_ref = torch.zeros_like(src_pool_host).to(device)
        dst_pool_kernel = torch.zeros_like(dst_pool_ref)
        dst_pool_direct = torch.zeros_like(dst_pool_ref)
    else:
        src_k_pool = torch.randn(
            num_layers, total_items_in_pool, item_size
        ).pin_memory()
        src_v_pool = torch.randn(
            num_layers, total_items_in_pool, item_size
        ).pin_memory()
        dst_k_pool_ref = torch.zeros_like(src_k_pool).to(device)
        dst_v_pool_ref = torch.zeros_like(src_v_pool).to(device)
        dst_k_pool_kernel = torch.zeros_like(dst_k_pool_ref)
        dst_v_pool_kernel = torch.zeros_like(dst_v_pool_ref)
        dst_k_pool_direct = torch.zeros_like(dst_k_pool_ref)
        dst_v_pool_direct = torch.zeros_like(dst_v_pool_ref)

    torch.cuda.synchronize()

    # We will test the per-layer function on the first layer (index 0) of the pool.
    layer_idx_to_test = 0

    if is_mla:
        if not all_layers:
            ref_copy_with_indices(
                src_pool_host[layer_idx_to_test],
                dst_pool_ref[layer_idx_to_test],
                src_indices_host,
                dst_indices_device,
            )
            transfer_kv_per_layer_mla(
                src_pool_host[layer_idx_to_test],
                dst_pool_kernel[layer_idx_to_test],
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
            )
            transfer_kv_direct(
                [src_pool_host[layer_idx_to_test]],
                [dst_pool_direct[layer_idx_to_test]],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        else:
            for layer_id in range(num_layers):
                ref_copy_with_indices(
                    src_pool_host[layer_id],
                    dst_pool_ref[layer_id],
                    src_indices_host,
                    dst_indices_device,
                )
            src_layers_device = torch.tensor(
                [src_pool_host[layer_id].data_ptr() for layer_id in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            dst_layers_device = torch.tensor(
                [
                    dst_pool_kernel[layer_id].data_ptr()
                    for layer_id in range(num_layers)
                ],
                dtype=torch.uint64,
                device=device,
            )
            transfer_kv_all_layer_mla(
                src_layers_device,
                dst_layers_device,
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
                num_layers=num_layers,
            )
            transfer_kv_direct(
                [src_pool_host[layer_id] for layer_id in range(num_layers)],
                [dst_pool_direct[layer_id] for layer_id in range(num_layers)],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_pool_kernel, dst_pool_ref)
        torch.testing.assert_close(dst_pool_direct, dst_pool_ref)
    else:
        if not all_layers:
            ref_copy_with_indices(
                src_k_pool[layer_idx_to_test],
                dst_k_pool_ref[layer_idx_to_test],
                src_indices_host,
                dst_indices_device,
            )
            ref_copy_with_indices(
                src_v_pool[layer_idx_to_test],
                dst_v_pool_ref[layer_idx_to_test],
                src_indices_host,
                dst_indices_device,
            )
            transfer_kv_per_layer(
                src_k_pool[layer_idx_to_test],
                dst_k_pool_kernel[layer_idx_to_test],
                src_v_pool[layer_idx_to_test],
                dst_v_pool_kernel[layer_idx_to_test],
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
            )
            transfer_kv_direct(
                [src_k_pool[layer_idx_to_test], src_v_pool[layer_idx_to_test]],
                [
                    dst_k_pool_direct[layer_idx_to_test],
                    dst_v_pool_direct[layer_idx_to_test],
                ],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        else:
            for layer_id in range(num_layers):
                ref_copy_with_indices(
                    src_k_pool[layer_id],
                    dst_k_pool_ref[layer_id],
                    src_indices_host,
                    dst_indices_device,
                )
                ref_copy_with_indices(
                    src_v_pool[layer_id],
                    dst_v_pool_ref[layer_id],
                    src_indices_host,
                    dst_indices_device,
                )

            src_k_layers_device = torch.tensor(
                [src_k_pool[layer_id].data_ptr() for layer_id in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            src_v_layers_device = torch.tensor(
                [src_v_pool[layer_id].data_ptr() for layer_id in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            dst_k_layers_device = torch.tensor(
                [
                    dst_k_pool_kernel[layer_id].data_ptr()
                    for layer_id in range(num_layers)
                ],
                dtype=torch.uint64,
                device=device,
            )
            dst_v_layers_device = torch.tensor(
                [
                    dst_v_pool_kernel[layer_id].data_ptr()
                    for layer_id in range(num_layers)
                ],
                dtype=torch.uint64,
                device=device,
            )
            transfer_kv_all_layer(
                src_k_layers_device,
                dst_k_layers_device,
                src_v_layers_device,
                dst_v_layers_device,
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
                num_layers=num_layers,
            )
            transfer_kv_direct(
                [src_k_pool[layer_id] for layer_id in range(num_layers)]
                + [src_v_pool[layer_id] for layer_id in range(num_layers)],
                [dst_k_pool_direct[layer_id] for layer_id in range(num_layers)]
                + [dst_v_pool_direct[layer_id] for layer_id in range(num_layers)],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_k_pool_kernel, dst_k_pool_ref)
        torch.testing.assert_close(dst_v_pool_kernel, dst_v_pool_ref)
        torch.testing.assert_close(dst_k_pool_direct, dst_k_pool_ref)
        torch.testing.assert_close(dst_v_pool_direct, dst_v_pool_ref)

    torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [128, 1024, 8192])
@pytest.mark.parametrize("page_size", [16, 64, 128])
@pytest.mark.parametrize("item_size", [256])
@pytest.mark.parametrize("total_items_in_pool", [20480])
@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("lf_to_pf", [False, True])
def test_transfer_kv_pf_direct(
    dtype: torch.dtype,
    num_items_to_transfer: int,
    item_size: int,
    page_size: int,
    total_items_in_pool: int,
    is_mla: bool,
    lf_to_pf: bool,
):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    device = "cuda"
    torch.cuda.manual_seed(42)
    test_stream = torch.cuda.Stream()

    num_layers = 4

    total_pages_in_pool = total_items_in_pool // page_size
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return
    page_indices = torch.randperm(total_pages_in_pool, dtype=torch.int64)
    src_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[:num_pages_to_transfer]
        ]
    )
    src_indices_device = src_indices_host.to(device)
    dst_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[num_pages_to_transfer : 2 * num_pages_to_transfer]
        ]
    )
    dst_indices_device = dst_indices_host.to(device)

    # We will test the per-layer function on the first layer (index 0) of the pool.
    layer_idx_to_test = 0

    if lf_to_pf:
        if is_mla:
            src_pool = torch.randn(num_layers, total_items_in_pool, item_size).to(
                device
            )
            src_pool_ptrs = [src_pool[i] for i in range(num_layers)]
            dst_pool_ref = torch.zeros(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()
            dst_pool_direct = torch.zeros_like(dst_pool_ref)
            torch.cuda.synchronize()

            with torch.cuda.stream(test_stream):
                transfer_kv_all_layer_direct_lf_pf(
                    src_pool_ptrs,
                    [dst_pool_direct],
                    src_indices_host,
                    dst_indices_host,
                    page_size,
                )
            test_stream.synchronize()

            for i in range(num_layers):
                ref_copy_with_indices_pf_direct(
                    src_pool,
                    dst_pool_ref,
                    src_indices_device,
                    dst_indices_host,
                    page_size,
                    i,
                    lf_to_pf=True,
                )
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_pool_direct, dst_pool_ref)

        else:
            src_k_pool = torch.randn(num_layers, total_items_in_pool, item_size).to(
                device
            )
            src_k_pool_ptrs = [src_k_pool[i] for i in range(num_layers)]
            src_v_pool = torch.randn(num_layers, total_items_in_pool, item_size).to(
                device
            )
            src_v_pool_ptrs = [src_v_pool[i] for i in range(num_layers)]
            dst_k_pool_ref = torch.zeros(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()
            dst_v_pool_ref = torch.zeros_like(dst_k_pool_ref)
            dst_k_pool_direct = torch.zeros_like(dst_k_pool_ref)
            dst_v_pool_direct = torch.zeros_like(dst_v_pool_ref)
            torch.cuda.synchronize()

            with torch.cuda.stream(test_stream):
                transfer_kv_all_layer_direct_lf_pf(
                    src_k_pool_ptrs + src_v_pool_ptrs,
                    [dst_k_pool_direct, dst_v_pool_direct],
                    src_indices_host,
                    dst_indices_host,
                    page_size,
                )
            test_stream.synchronize()

            for i in range(num_layers):
                ref_copy_with_indices_pf_direct(
                    src_k_pool,
                    dst_k_pool_ref,
                    src_indices_device,
                    dst_indices_host,
                    page_size,
                    i,
                    lf_to_pf=True,
                )
                ref_copy_with_indices_pf_direct(
                    src_v_pool,
                    dst_v_pool_ref,
                    src_indices_device,
                    dst_indices_host,
                    page_size,
                    i,
                    lf_to_pf=True,
                )
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_k_pool_direct, dst_k_pool_ref)
            torch.testing.assert_close(dst_v_pool_direct, dst_v_pool_ref)
    else:
        if is_mla:
            src_pool = torch.randn(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()

            dst_pool_ref = torch.zeros(num_layers, total_items_in_pool, item_size).to(
                device
            )
            dst_pool_direct = torch.zeros_like(dst_pool_ref)
            dst_pool_direct_ptrs = [dst_pool_direct[i] for i in range(num_layers)]
            torch.cuda.synchronize()

            with torch.cuda.stream(test_stream):
                transfer_kv_per_layer_direct_pf_lf(
                    [src_pool],
                    [dst_pool_direct_ptrs[layer_idx_to_test]],
                    src_indices_host,
                    dst_indices_host,
                    layer_idx_to_test,
                    page_size,
                )
            test_stream.synchronize()

            ref_copy_with_indices_pf_direct(
                src_pool,
                dst_pool_ref,
                src_indices_host,
                dst_indices_device,
                page_size,
                layer_idx_to_test,
                lf_to_pf=False,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_pool_direct, dst_pool_ref)
        else:
            src_k_pool = torch.randn(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()
            src_v_pool = torch.randn(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()

            dst_k_pool_ref = torch.zeros(num_layers, total_items_in_pool, item_size).to(
                device
            )
            dst_k_pool_direct = torch.zeros_like(dst_k_pool_ref)
            dst_k_pool_direct_ptrs = [dst_k_pool_direct[i] for i in range(num_layers)]

            dst_v_pool_ref = torch.zeros_like(dst_k_pool_ref)
            dst_v_pool_direct = torch.zeros_like(dst_v_pool_ref)
            dst_v_pool_direct_ptrs = [dst_v_pool_direct[i] for i in range(num_layers)]
            torch.cuda.synchronize()

            with torch.cuda.stream(test_stream):
                transfer_kv_per_layer_direct_pf_lf(
                    [src_k_pool, src_v_pool],
                    [
                        dst_k_pool_direct_ptrs[layer_idx_to_test],
                        dst_v_pool_direct_ptrs[layer_idx_to_test],
                    ],
                    src_indices_host,
                    dst_indices_host,
                    layer_idx_to_test,
                    page_size,
                )
            test_stream.synchronize()

            ref_copy_with_indices_pf_direct(
                src_k_pool,
                dst_k_pool_ref,
                src_indices_host,
                dst_indices_device,
                page_size,
                layer_idx_to_test,
                lf_to_pf=False,
            )
            ref_copy_with_indices_pf_direct(
                src_v_pool,
                dst_v_pool_ref,
                src_indices_host,
                dst_indices_device,
                page_size,
                layer_idx_to_test,
                lf_to_pf=False,
            )

            torch.cuda.synchronize()
            torch.testing.assert_close(dst_k_pool_direct, dst_k_pool_ref)
            torch.testing.assert_close(dst_v_pool_direct, dst_v_pool_ref)
    torch.set_default_dtype(original_dtype)


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [256, 1024])
@pytest.mark.parametrize("page_size", [16, 64, 128])
@pytest.mark.parametrize("item_size", [1024])
@pytest.mark.parametrize("head_num", [8, 16])
@pytest.mark.parametrize("total_items_in_pool", [4096])
@pytest.mark.parametrize("lf_to_ph", [False, True])
def test_transfer_kv_page_head(
    dtype: torch.dtype,
    num_items_to_transfer: int,
    page_size: int,
    item_size: int,
    head_num: int,
    total_items_in_pool: int,
    lf_to_ph: bool,
):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    device = "cuda"
    torch.cuda.manual_seed(42)

    num_layers = 4

    total_pages_in_pool = total_items_in_pool // page_size
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return

    assert item_size % head_num == 0
    head_dim = item_size // head_num

    page_indices = torch.randperm(total_pages_in_pool, dtype=torch.int64)
    src_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[:num_pages_to_transfer]
        ]
    )
    src_indices_device = src_indices_host.to(device)
    dst_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[num_pages_to_transfer : 2 * num_pages_to_transfer]
        ]
    )
    dst_indices_device = dst_indices_host.to(device)

    # We will test the per-layer function on the first layer (index 0) of the pool.
    layer_idx_to_test = 0

    if lf_to_ph:
        src_k_pool = torch.randn(
            num_layers, total_items_in_pool, head_num, head_dim
        ).to(device)
        src_v_pool = torch.randn(
            num_layers, total_items_in_pool, head_num, head_dim
        ).to(device)
        src_k_pool_ptrs = [src_k_pool[i] for i in range(num_layers)]
        src_k_pool_ptrs = torch.tensor(
            [x.data_ptr() for x in src_k_pool_ptrs],
            dtype=torch.uint64,
            device=device,
        )
        src_v_pool_ptrs = [src_v_pool[i] for i in range(num_layers)]
        src_v_pool_ptrs = torch.tensor(
            [x.data_ptr() for x in src_v_pool_ptrs],
            dtype=torch.uint64,
            device=device,
        )

        dst_k_pool_ref = torch.zeros(
            total_pages_in_pool, head_num, page_size, num_layers, head_dim
        ).pin_memory()
        dst_v_pool_ref = torch.zeros_like(dst_k_pool_ref).pin_memory()

        dst_k_pool_kernel = torch.zeros_like(dst_k_pool_ref).pin_memory()
        dst_v_pool_kernel = torch.zeros_like(dst_v_pool_ref).pin_memory()
        torch.cuda.synchronize()

        transfer_kv_all_layer_lf_ph(
            src_k_pool_ptrs,
            dst_k_pool_kernel,
            src_v_pool_ptrs,
            dst_v_pool_kernel,
            src_indices_device,
            dst_indices_device,
            item_size * dtype.itemsize,
            item_size * num_layers * dtype.itemsize,
            num_layers,
            page_size,
            head_num,
        )
        torch.cuda.synchronize()

        for i in range(num_layers):
            ref_copy_with_indices_page_head(
                src_k_pool,
                dst_k_pool_ref,
                src_indices_device,
                dst_indices_host,
                page_size,
                i,
                head_num,
                lf_to_ph=True,
            )
            ref_copy_with_indices_page_head(
                src_v_pool,
                dst_v_pool_ref,
                src_indices_device,
                dst_indices_host,
                page_size,
                i,
                head_num,
                lf_to_ph=True,
            )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_k_pool_kernel, dst_k_pool_ref)
        torch.testing.assert_close(dst_v_pool_kernel, dst_v_pool_ref)
    else:
        from sgl_kernel.kvcacheio import transfer_kv_per_layer_ph_lf

        src_k_pool = torch.randn(
            total_pages_in_pool, head_num, page_size, num_layers, head_dim
        ).pin_memory()
        src_v_pool = torch.randn(
            total_pages_in_pool, head_num, page_size, num_layers, head_dim
        ).pin_memory()

        dst_k_pool_ref = torch.zeros(
            num_layers, total_items_in_pool, head_num, head_dim
        ).to(device)
        dst_v_pool_ref = torch.zeros_like(dst_k_pool_ref)
        dst_k_pool_kernel = torch.zeros_like(dst_k_pool_ref)
        dst_v_pool_kernel = torch.zeros_like(dst_v_pool_ref)
        dst_k_pool_kernel_ptrs = [dst_k_pool_kernel[i] for i in range(num_layers)]
        dst_v_pool_kernel_ptrs = [dst_v_pool_kernel[i] for i in range(num_layers)]
        torch.cuda.synchronize()

        transfer_kv_per_layer_ph_lf(
            src_k_pool,
            dst_k_pool_kernel_ptrs[layer_idx_to_test],
            src_v_pool,
            dst_v_pool_kernel_ptrs[layer_idx_to_test],
            src_indices_device,
            dst_indices_device,
            layer_idx_to_test,
            item_size * dtype.itemsize,
            item_size * num_layers * dtype.itemsize,
            page_size,
            head_num,
        )

        ref_copy_with_indices_page_head(
            src_k_pool,
            dst_k_pool_ref,
            src_indices_host,
            dst_indices_device,
            page_size,
            layer_idx_to_test,
            head_num,
            lf_to_ph=False,
        )
        ref_copy_with_indices_page_head(
            src_v_pool,
            dst_v_pool_ref,
            src_indices_host,
            dst_indices_device,
            page_size,
            layer_idx_to_test,
            head_num,
            lf_to_ph=False,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(dst_k_pool_kernel, dst_k_pool_ref)
        torch.testing.assert_close(dst_v_pool_kernel, dst_v_pool_ref)
    torch.set_default_dtype(original_dtype)


# Head-group-major page blocks, as a set of unified L3 chunks lands them.
# page_first_direct stores a page as (L, P, H, D); when the unified L3 grid cuts
# the kv-head axis, a page reassembled from those chunks is (HG, L, P, hg, D) --
# the same bytes, permuted. The pfdhg kernels absorb that permutation into the
# transfer they were already running.
#
# (head_num, head_groups, layers, page_size, head_dim)
HEAD_GROUP_CONFIGS = [
    # The motivating cross-TP GQA case: one kv head per group, at
    # DeepSeek-V3's prime layer count.
    (8, 8, 61, 16, 128),
    (8, 4, 4, 8, 64),
    (16, 2, 5, 16, 128),
    (16, 4, 61, 8, 128),
    # head_groups == 1: the block is (page, 1, L, P, H, D), i.e. exactly
    # page_first_direct's own order, so this case pins that the kernel
    # degenerates to a plain token gather.
    (8, 1, 3, 4, 32),
    # Degenerate shapes.
    (4, 4, 1, 4, 64),
]


def _head_group_host(shape, seed, dtype):
    """Seeded noise, NOT a counter.

    A counter taken mod 2048 is periodic and the layer stride
    (page_size * hg * head_dim) is an exact multiple of that period in several
    configs -- adjacent layers would hold identical values and a layer mix-up
    would pass. bf16 also carries 8 mantissa bits, so a counter is not exactly
    representable past 256.
    """
    n = 1
    for d in shape:
        n *= d
    gen = torch.Generator().manual_seed(seed)
    # randn has no fp8 kernel, so generate in fp32 and narrow. The kernels move
    # bytes, so any exactly-representable pattern is a valid oracle.
    return torch.randn(n, generator=gen).to(dtype).reshape(shape).contiguous()


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "head_num, head_groups, layers, page_size, head_dim", HEAD_GROUP_CONFIGS
)
def test_transfer_kv_head_group(
    dtype: torch.dtype,
    head_num: int,
    head_groups: int,
    layers: int,
    page_size: int,
    head_dim: int,
):
    """H2D: device[d] == host[s // P, :, layer, s % P, :, :].reshape(H, D).

    The oracle is built from the *logical* host tensor with torch indexing, so
    it catches drift between the offset functor and the layout it claims to
    read (a Python transcription of the functor would not).
    """
    hg = head_num // head_groups
    num_pages = 3
    rows = num_pages * page_size
    itemsize = torch.tensor([], dtype=dtype).element_size()
    item_size = head_num * head_dim * itemsize

    shape = (num_pages, head_groups, layers, page_size, hg, head_dim)
    host_k = _head_group_host(shape, 0, dtype).pin_memory()
    host_v = _head_group_host(shape, 1024, dtype).pin_memory()
    dev_k = torch.zeros(layers, rows, head_num, head_dim, dtype=dtype, device="cuda")
    dev_v = torch.zeros_like(dev_k)

    # non-identity permutation so the index tensors are actually exercised
    src = torch.arange(rows, dtype=torch.int64)
    dst = torch.flip(src, [0]).contiguous()

    for layer in range(layers):
        transfer_kv_per_layer_pfdhg_lf(
            src_k=host_k.view(-1),
            dst_k=dev_k[layer],
            src_v=host_v.view(-1),
            dst_v=dev_v[layer],
            src_indices=src.cuda(),
            dst_indices=dst.cuda(),
            layer_id=layer,
            item_size=item_size,
            src_layout_dim=item_size * layers,
            page_size=page_size,
            head_num=head_groups,
        )
    torch.cuda.synchronize()

    for layer in range(layers):
        exp_k = torch.empty(rows, head_num, head_dim, dtype=dtype)
        exp_v = torch.empty(rows, head_num, head_dim, dtype=dtype)
        for i in range(rows):
            s, d = int(src[i]), int(dst[i])
            page, tok = s // page_size, s % page_size
            exp_k[d] = host_k[page, :, layer, tok, :, :].reshape(head_num, head_dim)
            exp_v[d] = host_v[page, :, layer, tok, :, :].reshape(head_num, head_dim)
        torch.testing.assert_close(dev_k[layer].cpu(), exp_k, atol=0, rtol=0)
        torch.testing.assert_close(dev_v[layer].cpu(), exp_v, atol=0, rtol=0)


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize("block_quota", [2, 8, 32, 132])
def test_transfer_kv_head_group_quota_does_not_change_results(block_quota: int):
    """block_quota only caps the grid (num_blocks <= quota); it must not change
    the bytes moved. It is purely a bandwidth knob -- measured on B200 + PCIe
    Gen5, H2D goes 11.9 -> 49.9 GB/s from quota 2 to 32 -- so a caller may tune
    it freely, and this pins that doing so stays correct."""
    dtype = torch.float16
    head_num, head_groups, layers, page_size, head_dim = 8, 4, 4, 8, 64
    hg = head_num // head_groups
    num_pages = 3
    rows = num_pages * page_size
    item_size = head_num * head_dim * 2

    shape = (num_pages, head_groups, layers, page_size, hg, head_dim)
    host = _head_group_host(shape, 5, dtype).pin_memory()
    dev = torch.zeros(layers, rows, head_num, head_dim, dtype=dtype, device="cuda")
    src = torch.arange(rows, dtype=torch.int64)
    dst = torch.flip(src, [0]).contiguous()

    for layer in range(layers):
        transfer_kv_per_layer_pfdhg_lf(
            src_k=host.view(-1),
            dst_k=dev[layer],
            src_v=host.view(-1),
            dst_v=dev[layer],
            src_indices=src.cuda(),
            dst_indices=dst.cuda(),
            layer_id=layer,
            item_size=item_size,
            src_layout_dim=item_size * layers,
            page_size=page_size,
            head_num=head_groups,
            block_quota=block_quota,
        )
    torch.cuda.synchronize()

    for layer in range(layers):
        exp = torch.empty(rows, head_num, head_dim, dtype=dtype)
        for i in range(rows):
            s_i, d_i = int(src[i]), int(dst[i])
            page, tok = s_i // page_size, s_i % page_size
            exp[d_i] = host[page, :, layer, tok, :, :].reshape(head_num, head_dim)
        torch.testing.assert_close(dev[layer].cpu(), exp, atol=0, rtol=0)


def _head_group_write_back(head_num, head_groups, layers, page_size, head_dim, dtype):
    hg = head_num // head_groups
    num_pages = 3
    rows = num_pages * page_size
    itemsize = torch.tensor([], dtype=dtype).element_size()
    item_size = head_num * head_dim * itemsize

    dev_shape = (layers, rows, head_num, head_dim)
    dev_k = _head_group_host(dev_shape, 0, dtype).cuda()
    dev_v = _head_group_host(dev_shape, 1024, dtype).cuda()
    host_shape = (num_pages, head_groups, layers, page_size, hg, head_dim)
    host_k = torch.zeros(host_shape, dtype=dtype).pin_memory()
    host_v = torch.zeros_like(host_k)

    src = torch.arange(rows, dtype=torch.int64)
    dst = torch.flip(src, [0]).contiguous()

    def layer_table(pool):
        return torch.tensor(
            [pool[i].data_ptr() for i in range(layers)], dtype=torch.uint64
        ).cuda()

    transfer_kv_all_layer_lf_pfdhg(
        src_k_layers=layer_table(dev_k),
        dst_k=host_k.view(-1),
        src_v_layers=layer_table(dev_v),
        dst_v=host_v.view(-1),
        src_indices=src.cuda(),
        dst_indices=dst.cuda(),
        item_size=item_size,
        dst_layout_dim=item_size * layers,
        num_layers=layers,
        page_size=page_size,
        head_num=head_groups,
    )
    torch.cuda.synchronize()
    return host_k, host_v, dev_k, dev_v, src, dst, hg


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "head_num, head_groups, layers, page_size, head_dim", HEAD_GROUP_CONFIGS
)
def test_transfer_kv_head_group_write_back(
    dtype: torch.dtype,
    head_num: int,
    head_groups: int,
    layers: int,
    page_size: int,
    head_dim: int,
):
    """D2H, the inverse contract:

        host[d // P, g, layer, d % P, j, :] == device[layer][s, g * hg + j, :]

    It exists so the host pool has ONE byte order: if write-back kept
    page_first_direct's natural order while the L3 grid reads head-group-major,
    a page's order would depend on whether it arrived from the device or from
    L3, and the H2D would have no way to tell.
    """
    host_k, host_v, dev_k, dev_v, src, dst, hg = _head_group_write_back(
        head_num, head_groups, layers, page_size, head_dim, dtype
    )
    for host, dev, label in ((host_k, dev_k, "K"), (host_v, dev_v, "V")):
        want = torch.zeros_like(host)
        for i in range(len(src)):
            s, d = int(src[i]), int(dst[i])
            page, tok = d // page_size, d % page_size
            for layer in range(layers):
                row = dev[layer, s].cpu()
                for g in range(head_groups):
                    want[page, g, layer, tok] = row[g * hg : (g + 1) * hg]
        torch.testing.assert_close(host, want, atol=0, rtol=0, msg=f"{label} mismatch")


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize(
    "head_num, head_groups, layers, page_size, head_dim", HEAD_GROUP_CONFIGS
)
def test_transfer_kv_head_group_round_trip(
    head_num: int, head_groups: int, layers: int, page_size: int, head_dim: int
):
    """D2H then H2D over the same slots must return the original bytes.

    This is the property the host pool actually relies on: a page written back
    by the device and later loaded again is unchanged, whatever the head-group
    count.
    """
    dtype = torch.bfloat16
    host_k, host_v, dev_k, dev_v, src, dst, _ = _head_group_write_back(
        head_num, head_groups, layers, page_size, head_dim, dtype
    )
    itemsize = torch.tensor([], dtype=dtype).element_size()
    item_size = head_num * head_dim * itemsize
    out_k, out_v = torch.zeros_like(dev_k), torch.zeros_like(dev_v)

    # read back along the inverse index mapping
    for layer in range(layers):
        transfer_kv_per_layer_pfdhg_lf(
            src_k=host_k.view(-1),
            dst_k=out_k[layer],
            src_v=host_v.view(-1),
            dst_v=out_v[layer],
            src_indices=dst.cuda(),
            dst_indices=src.cuda(),
            layer_id=layer,
            item_size=item_size,
            src_layout_dim=item_size * layers,
            page_size=page_size,
            head_num=head_groups,
        )
    torch.cuda.synchronize()
    torch.testing.assert_close(out_k, dev_k, atol=0, rtol=0)
    torch.testing.assert_close(out_v, dev_v, atol=0, rtol=0)


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"block_quota": 0}, "block_quota"),
        ({"page_size": 0}, "page_size"),
        ({"head_num": 0}, "head_num"),
        ({"head_num": 3}, "item_size"),
        ({"item_size": 24, "src_layout_dim": 24, "head_num": 2}, "Per-head-group"),
        ({"src_layout_dim": 513}, "src_layout_dim"),
    ],
)
def test_transfer_kv_head_group_rejects_invalid_geometry(overrides, error):
    """Reject invalid divisors and copy widths instead of truncating."""
    host = torch.zeros(4 * 4 * 4 * 64, dtype=torch.float16).pin_memory()
    dev = torch.zeros(16, 4, 64, dtype=torch.float16, device="cuda")
    idx = torch.arange(16, dtype=torch.int64, device="cuda")
    args = {
        "item_size": 4 * 64 * 2,
        "src_layout_dim": 4 * 64 * 2,
        "page_size": 4,
        "head_num": 4,
        "block_quota": 2,
        "indices": idx,
    }
    args.update(overrides)

    def launch(**kw):
        torch.ops.sgl_kernel.transfer_kv_per_layer_pfdhg_lf(
            host.view(-1),
            dev,
            host.view(-1),
            dev,
            kw["indices"],
            kw["indices"],
            0,
            kw["item_size"],
            kw["src_layout_dim"],
            kw["page_size"],
            kw["head_num"],
            kw["block_quota"],
            32,
        )

    with pytest.raises(RuntimeError, match=error):
        launch(**args)


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
def test_transfer_kv_head_group_empty_is_a_noop():
    """An empty transfer is a valid no-op, not a grid-size division by zero."""
    host = torch.zeros(4 * 4 * 4 * 64, dtype=torch.float16).pin_memory()
    dev = torch.zeros(16, 4, 64, dtype=torch.float16, device="cuda")
    empty = torch.arange(0, dtype=torch.int64, device="cuda")
    transfer_kv_per_layer_pfdhg_lf(
        src_k=host.view(-1),
        dst_k=dev,
        src_v=host.view(-1),
        dst_v=dev,
        src_indices=empty,
        dst_indices=empty,
        layer_id=0,
        item_size=4 * 64 * 2,
        src_layout_dim=4 * 64 * 2,
        page_size=4,
        head_num=4,
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
