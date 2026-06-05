import sys

import pytest
import torch
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_all_layer_lf_pf,
    transfer_kv_all_layer_lf_ph,
    transfer_kv_all_layer_mla,
    transfer_kv_direct,
    transfer_kv_per_layer,
    transfer_kv_per_layer_direct_pf_lf,
    transfer_kv_per_layer_mla,
    transfer_kv_per_layer_pf_lf,
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


# ---------------------------------------------------------------------------
# Asymmetric K/V (head_dim != v_head_dim) tests for the HiCache transfer
# kernels touched by the MiMo V2 fix. Covers transfer_kv_per_layer_pf_lf and
# transfer_kv_all_layer_lf_pf, each in symmetric (128, 128) and MiMo V2
# asymmetric (192, 128) flavors. Reference is plain PyTorch indexed copy, the
# kernel only memcpys, so equality is byte-exact.
# ---------------------------------------------------------------------------

_ASYM_NUM_LAYERS = 4
_ASYM_HEAD_NUM = 2
_ASYM_PAGE_SIZE = 16
_ASYM_TOTAL_PAGES = 32


def _asym_layout_sizes(head_num, head_dim, num_layers, dtype):
    token_stride = head_num * head_dim * dtype.itemsize
    layout_dim = token_stride * num_layers
    return token_stride, layout_dim


def _asym_make_indices(num_pages_to_transfer, page_size, total_pages, device):
    perm = torch.randperm(total_pages, dtype=torch.int64)
    src_pages = perm[:num_pages_to_transfer]
    dst_pages = perm[num_pages_to_transfer : 2 * num_pages_to_transfer]
    src = torch.cat(
        [torch.arange(int(p) * page_size, (int(p) + 1) * page_size) for p in src_pages]
    )
    dst = torch.cat(
        [torch.arange(int(p) * page_size, (int(p) + 1) * page_size) for p in dst_pages]
    )
    return src.to(device), dst.to(device)


def _asym_data_ptrs(per_layer_tensors, device):
    return torch.tensor(
        [t.data_ptr() for t in per_layer_tensors],
        dtype=torch.uint64,
        device=device,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "head_dim, v_head_dim",
    # (192, 128) matches MiMo-V2-Flash's real qk_head_dim / v_head_dim.
    [(128, 128), (192, 128)],
    ids=["symmetric", "asymmetric_mimo_v2_flash"],
)
def test_transfer_kv_per_layer_pf_lf_asymmetric(dtype, head_dim, v_head_dim):
    """H (page_first, separate K/V) -> D (layer_first, per-layer view)."""
    torch.cuda.manual_seed(42)
    device = "cuda"
    layer_id = 1
    num_pages_to_transfer = 4
    total_tokens = _ASYM_TOTAL_PAGES * _ASYM_PAGE_SIZE

    src_indices, dst_indices = _asym_make_indices(
        num_pages_to_transfer, _ASYM_PAGE_SIZE, _ASYM_TOTAL_PAGES, device
    )

    # Asymmetric K/V is exercised by allocating K and V independently with
    # their own trailing dim. The symmetric case uses the same two-tensor
    # shape so the reference path stays uniform; this matches the production
    # MHATokenToKVPoolHost.k_buffer / .v_buffer accessors which yield K and V
    # slices regardless of whether the underlying allocation is fused.
    src_k_host = torch.randn(
        total_tokens,
        _ASYM_NUM_LAYERS,
        _ASYM_HEAD_NUM,
        head_dim,
        dtype=dtype,
        pin_memory=True,
    )
    src_v_host = torch.randn(
        total_tokens,
        _ASYM_NUM_LAYERS,
        _ASYM_HEAD_NUM,
        v_head_dim,
        dtype=dtype,
        pin_memory=True,
    )
    dst_k_dev = torch.zeros(
        _ASYM_NUM_LAYERS,
        total_tokens,
        _ASYM_HEAD_NUM,
        head_dim,
        dtype=dtype,
        device=device,
    )
    dst_v_dev = torch.zeros(
        _ASYM_NUM_LAYERS,
        total_tokens,
        _ASYM_HEAD_NUM,
        v_head_dim,
        dtype=dtype,
        device=device,
    )

    k_token_stride, k_layout_dim = _asym_layout_sizes(
        _ASYM_HEAD_NUM, head_dim, _ASYM_NUM_LAYERS, dtype
    )
    transfer_kv_per_layer_pf_lf(
        src_k=src_k_host,
        dst_k=dst_k_dev[layer_id],
        src_v=src_v_host,
        dst_v=dst_v_dev[layer_id],
        src_indices=src_indices,
        dst_indices=dst_indices,
        layer_id=layer_id,
        item_size=k_token_stride,
        src_layout_dim=k_layout_dim,
    )
    torch.cuda.synchronize()

    expected_k = src_k_host[src_indices.cpu(), layer_id].to(device)
    expected_v = src_v_host[src_indices.cpu(), layer_id].to(device)
    torch.testing.assert_close(dst_k_dev[layer_id, dst_indices], expected_k)
    torch.testing.assert_close(dst_v_dev[layer_id, dst_indices], expected_v)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "head_dim, v_head_dim",
    [(128, 128), (192, 128)],
    ids=["symmetric", "asymmetric_mimo_v2_flash"],
)
def test_transfer_kv_all_layer_lf_pf_asymmetric(dtype, head_dim, v_head_dim):
    """D (layer_first, layer-ptr table) -> H (page_first, separate K/V)."""
    torch.cuda.manual_seed(42)
    device = "cuda"
    num_pages_to_transfer = 4
    total_tokens = _ASYM_TOTAL_PAGES * _ASYM_PAGE_SIZE

    src_indices, dst_indices = _asym_make_indices(
        num_pages_to_transfer, _ASYM_PAGE_SIZE, _ASYM_TOTAL_PAGES, device
    )

    src_k_layers = [
        torch.randn(total_tokens, _ASYM_HEAD_NUM, head_dim, dtype=dtype, device=device)
        for _ in range(_ASYM_NUM_LAYERS)
    ]
    src_v_layers = [
        torch.randn(
            total_tokens, _ASYM_HEAD_NUM, v_head_dim, dtype=dtype, device=device
        )
        for _ in range(_ASYM_NUM_LAYERS)
    ]
    src_k_ptrs = _asym_data_ptrs(src_k_layers, device)
    src_v_ptrs = _asym_data_ptrs(src_v_layers, device)

    dst_k_host = torch.zeros(
        total_tokens,
        _ASYM_NUM_LAYERS,
        _ASYM_HEAD_NUM,
        head_dim,
        dtype=dtype,
        pin_memory=True,
    )
    dst_v_host = torch.zeros(
        total_tokens,
        _ASYM_NUM_LAYERS,
        _ASYM_HEAD_NUM,
        v_head_dim,
        dtype=dtype,
        pin_memory=True,
    )

    k_token_stride, k_layout_dim = _asym_layout_sizes(
        _ASYM_HEAD_NUM, head_dim, _ASYM_NUM_LAYERS, dtype
    )
    transfer_kv_all_layer_lf_pf(
        src_k_layers=src_k_ptrs,
        dst_k=dst_k_host,
        src_v_layers=src_v_ptrs,
        dst_v=dst_v_host,
        src_indices=src_indices,
        dst_indices=dst_indices,
        item_size=k_token_stride,
        dst_layout_dim=k_layout_dim,
        num_layers=_ASYM_NUM_LAYERS,
    )
    torch.cuda.synchronize()

    dst_idx_cpu = dst_indices.cpu()
    for layer_id in range(_ASYM_NUM_LAYERS):
        torch.testing.assert_close(
            dst_k_host[dst_idx_cpu, layer_id],
            src_k_layers[layer_id][src_indices].cpu(),
        )
        torch.testing.assert_close(
            dst_v_host[dst_idx_cpu, layer_id],
            src_v_layers[layer_id][src_indices].cpu(),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
