"""The page_unified host pool round trip: device -> host -> device.

Write-back (#39606's staged relayout) and load-back (the page_unified JIT
kernel) are two halves of one contract, and the contract is a byte order.
Each kernel has its own unit tests against its own reference; what neither can
catch is the pair disagreeing, because a permutation applied twice in opposite
directions cancels only if both sides read the same geometry off the same pool.

So this drives the real ``MHATokenToKVPoolHost`` / ``MLATokenToKVPoolHost``
transfer arms rather than the kernels directly, and compares the device pool
against itself.
"""

import sys

import pytest
import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or is_npu()
    or is_xpu()
    or not (is_cuda() or is_hip()),
    reason="page_unified host pool tests require CUDA/ROCm.",
)

DEVICE = "cuda"
PAGE_SIZE = 16
LAYERS = 4
HEAD_DIM = 128


def _page_indices(pages, page_size):
    offsets = torch.arange(page_size, dtype=torch.int64)
    return (
        torch.tensor(pages, dtype=torch.int64)[:, None] * page_size + offsets
    ).flatten()


def _pinned_host_pool(host_pool_cls, device_pool, **kwargs):
    """Allocate the host pool with pin_memory rather than cudaHostRegister.

    Registering is what the runtime does, but it is process-global: a second
    pool landing on a freed pool's address fails with
    cudaErrorHostMemoryAlreadyRegistered. Parametrized cases would then pass or
    fail on allocator reuse, which says nothing about the layout.
    """
    from sglang.srt.mem_cache.device_pool_info import MLABufferInfo
    from sglang.srt.mem_cache.pool_host.common import (
        ALLOC_MEMORY_FUNCS,
        alloc_with_pin_memory,
    )

    buffer_info = device_pool.get_device_pool_infos()[0].buffer_info
    if isinstance(buffer_info, MLABufferInfo):
        kwargs.update(
            target_layer_num=device_pool.layer_num,
            override_kv_cache_dim=device_pool.kv_cache_dim,
        )
    original = ALLOC_MEMORY_FUNCS[DEVICE]
    ALLOC_MEMORY_FUNCS[DEVICE] = alloc_with_pin_memory
    try:
        return host_pool_cls(
            buffer_info,
            device_capacity_tokens=device_pool.size,
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout="page_unified",
            pin_memory=True,
            device="cpu",
            **kwargs,
        )
    finally:
        ALLOC_MEMORY_FUNCS[DEVICE] = original


def _check_zero_copy_page_meta(host_pool, host_indices):
    """Zero-copy backends address the same page block the flat page API copies."""
    ptrs, sizes = host_pool.get_page_buffer_meta(host_indices)
    pages = [host_pool.get_data_page(int(index)) for index in host_indices[::PAGE_SIZE]]
    assert ptrs == [page.data_ptr() for page in pages]
    assert sizes == [host_pool.page_unified_layout.bytes_per_page] * len(pages)
    assert all(page.nbytes == size for page, size in zip(pages, sizes))


def _restore_storage_pages(host_pool, host_indices):
    """Exercise the flat page API used by generic file storage at new offsets."""
    stored_pages = [
        host_pool.get_data_page(int(index)).clone()
        for index in host_indices[::PAGE_SIZE]
    ]
    dummy_page = host_pool.get_dummy_flat_data_page()
    assert all(page.shape == dummy_page.shape for page in stored_pages)
    assert dummy_page.nbytes == host_pool.page_unified_layout.bytes_per_page

    host_pool.kv_buffer.zero_()
    restored_indices = _page_indices([7, 3, 5, 1], PAGE_SIZE)
    for index, page in zip(restored_indices[::PAGE_SIZE], stored_pages):
        host_pool.set_from_flat_data_page(int(index), page)
        restored_page = host_pool.get_data_page(int(index), flat=False)
        assert restored_page.is_contiguous()
        assert torch.equal(restored_page.flatten(), page)
    return restored_indices


def _prepare_staging(device_pool, *, is_mla=False, head_group_num=1, layer_num=LAYERS):
    from sglang.srt.mem_cache.pool_host.base import (
        prepare_page_unified_write_back_staging,
    )
    from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout

    geometry = PageUnifiedLayout(
        page_size=PAGE_SIZE,
        layer_num=layer_num,
        head_num=1 if is_mla else device_pool.head_num,
        head_group_num=head_group_num,
        head_dim=device_pool.kv_cache_dim if is_mla else HEAD_DIM,
        itemsize=device_pool.store_dtype.itemsize,
        is_mla=is_mla,
    )
    return prepare_page_unified_write_back_staging(
        geometry, dtype=device_pool.store_dtype, device=torch.device(DEVICE)
    )


@pytest.mark.parametrize("head_num,head_group_num", [(8, 1), (8, 2), (8, 8), (4, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("preallocate", [False, True])
def test_mha_page_unified(head_num, head_group_num, dtype, preallocate):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

    device_pool = MHATokenToKVPool(
        size=PAGE_SIZE * 8,
        page_size=PAGE_SIZE,
        dtype=dtype,
        head_num=head_num,
        head_dim=HEAD_DIM,
        layer_num=LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
        start_layer=0,
        end_layer=LAYERS,
    )
    staging = (
        _prepare_staging(device_pool, head_group_num=head_group_num)
        if preallocate
        else None
    )
    host_pool = _pinned_host_pool(
        MHATokenToKVPoolHost,
        device_pool,
        head_group_num=head_group_num,
        write_back_staging=staging,
    )
    if preallocate:
        assert host_pool.staging_buffer.data_ptr() == staging.data_ptr()
        assert (
            host_pool.staging_buffer.shape[0] == host_pool.page_num < staging.shape[0]
        )

    generator = torch.Generator(device=DEVICE).manual_seed(20260915)
    for layer in range(LAYERS):
        for buffer in (device_pool.k_buffer[layer], device_pool.v_buffer[layer]):
            buffer.copy_(
                torch.randint(
                    0,
                    128,
                    buffer.shape,
                    generator=generator,
                    dtype=torch.int32,
                    device=DEVICE,
                ).to(dtype)
            )
    original = [
        (device_pool.k_buffer[layer].clone(), device_pool.v_buffer[layer].clone())
        for layer in range(LAYERS)
    ]

    # Page order is shuffled and the host pages are not the device pages: an
    # arm that ignored one of the two index vectors would still pass on an
    # identity mapping.
    device_indices = _page_indices([5, 1, 3, 0], PAGE_SIZE).to(DEVICE)
    host_indices_cpu = _page_indices([2, 6, 0, 4], PAGE_SIZE)

    # Write-back takes CPU host page ids, which is what the cache controller
    # hands the staged paths.
    host_pool.backup_from_device_all_layer(
        device_pool, host_indices_cpu, device_indices, "kernel"
    )
    torch.cuda.synchronize()
    _check_zero_copy_page_meta(host_pool, host_indices_cpu)
    host_indices_cpu = _restore_storage_pages(host_pool, host_indices_cpu)

    for layer in range(LAYERS):
        device_pool.k_buffer[layer].zero_()
        device_pool.v_buffer[layer].zero_()

    # Load-back takes device-resident host indices, as move_indices produces.
    host_indices = host_indices_cpu.to(DEVICE)
    for layer in range(LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer, "kernel"
        )
    torch.cuda.synchronize()

    for layer in range(LAYERS):
        expected_k, expected_v = original[layer]
        assert torch.equal(
            device_pool.k_buffer[layer][device_indices], expected_k[device_indices]
        ), f"K mismatch at layer {layer}"
        assert torch.equal(
            device_pool.v_buffer[layer][device_indices], expected_v[device_indices]
        ), f"V mismatch at layer {layer}"


@pytest.mark.parametrize("kv_lora_rank,qk_rope_head_dim", [(512, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("preallocate", [False, True])
def test_mla_page_unified(kv_lora_rank, qk_rope_head_dim, dtype, preallocate):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

    device_pool = MLATokenToKVPool(
        size=PAGE_SIZE * 8,
        page_size=PAGE_SIZE,
        dtype=dtype,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
        start_layer=0,
        end_layer=LAYERS,
    )
    staging = _prepare_staging(device_pool, is_mla=True) if preallocate else None
    host_pool = _pinned_host_pool(
        MLATokenToKVPoolHost, device_pool, write_back_staging=staging
    )
    if preallocate:
        assert host_pool.staging_buffer.data_ptr() == staging.data_ptr()
        assert (
            host_pool.staging_buffer.shape[0] == host_pool.page_num < staging.shape[0]
        )

    generator = torch.Generator(device=DEVICE).manual_seed(20260916)
    for layer in range(LAYERS):
        buffer = device_pool.kv_buffer[layer]
        buffer.copy_(
            torch.randint(
                0,
                128,
                buffer.shape,
                generator=generator,
                dtype=torch.int32,
                device=DEVICE,
            ).to(dtype)
        )
    original = [device_pool.kv_buffer[layer].clone() for layer in range(LAYERS)]

    device_indices = _page_indices([5, 1, 3, 0], PAGE_SIZE).to(DEVICE)
    host_indices_cpu = _page_indices([2, 6, 0, 4], PAGE_SIZE)

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices_cpu, device_indices, "kernel"
    )
    torch.cuda.synchronize()
    _check_zero_copy_page_meta(host_pool, host_indices_cpu)
    host_indices_cpu = _restore_storage_pages(host_pool, host_indices_cpu)
    for layer in range(LAYERS):
        device_pool.kv_buffer[layer].zero_()

    host_indices = host_indices_cpu.to(DEVICE)
    for layer in range(LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer, "kernel"
        )
    torch.cuda.synchronize()

    for layer in range(LAYERS):
        assert torch.equal(
            device_pool.kv_buffer[layer][device_indices],
            original[layer][device_indices],
        ), f"latent KV mismatch at layer {layer}"


# Packed draft layers (NEXTN-style MTP): each one-layer draft pool is appended
# to the target host pool's layer axis. The round trip above cannot tell where
# the draft rows sit in a page, because whatever write-back does, load-back
# undoes. These cases pin the position: the bytes of the draft range of a
# page block are compared with the draft device rows, and zeroing that range
# on the host must wipe the draft rows and nothing else.


def _fill(buffers, generator, dtype):
    for buffer in buffers:
        buffer.copy_(
            torch.randint(
                0, 128, buffer.shape, generator=generator, dtype=torch.int32
            ).to(device=DEVICE, dtype=dtype)
        )


def _flat_page(host_pool, index):
    return host_pool.get_data_page(int(index), flat=True).clone().view(torch.uint8)


def _zero_span(host_pool, host_indices, spans):
    for index in host_indices[::PAGE_SIZE]:
        page = _flat_page(host_pool, index)
        for offset, length in spans:
            page[offset : offset + length] = 0
        host_pool.set_from_flat_data_page(
            int(index), page.view(host_pool.kv_buffer.dtype)
        )


def _load_all(host_pool, pools, host_indices, device_indices, target_layers):
    """Target layers, then each draft depth, the way the controller loads them."""
    for layer in range(target_layers):
        host_pool.load_to_device_per_layer(
            pools[0], host_indices, device_indices, layer, "kernel"
        )
    for depth, draft in enumerate(pools[1:]):
        host_pool.load_to_device_per_layer(
            draft,
            host_indices,
            device_indices,
            target_layers + depth,
            "kernel",
            is_draft=True,
        )
    torch.cuda.synchronize()


@pytest.mark.parametrize("head_group_num", [1, 2, 8])
@pytest.mark.parametrize("draft_num", [1, 3])
def test_mha_page_unified_packed_drafts(head_group_num, draft_num):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

    head_num, dtype = 8, torch.bfloat16

    def make_pool(layer_num):
        return MHATokenToKVPool(
            size=PAGE_SIZE * 8,
            page_size=PAGE_SIZE,
            dtype=dtype,
            head_num=head_num,
            head_dim=HEAD_DIM,
            layer_num=layer_num,
            device=DEVICE,
            enable_memory_saver=False,
            start_layer=0,
            end_layer=layer_num,
        )

    target = make_pool(LAYERS)
    drafts = [make_pool(1) for _ in range(draft_num)]
    pools = [target, *drafts]
    total_layers = LAYERS + draft_num
    host_pool = _pinned_host_pool(
        MHATokenToKVPoolHost,
        target,
        head_group_num=head_group_num,
        draft_buffer_infos=[
            draft.get_device_pool_infos()[0].buffer_info for draft in drafts
        ],
        write_back_staging=_prepare_staging(
            target, head_group_num=head_group_num, layer_num=total_layers
        ),
    )
    layout = host_pool.page_unified_layout
    assert (host_pool.target_layer_num, host_pool.layer_num) == (LAYERS, total_layers)
    assert layout.layer_num == total_layers

    generator = torch.Generator().manual_seed(20261116)
    k_buffers = [
        pool.k_buffer[layer] for pool in pools for layer in range(pool.layer_num)
    ]
    v_buffers = [
        pool.v_buffer[layer] for pool in pools for layer in range(pool.layer_num)
    ]
    _fill([*k_buffers, *v_buffers], generator, dtype)
    original_k = [buffer.clone() for buffer in k_buffers]
    original_v = [buffer.clone() for buffer in v_buffers]

    device_pages, host_pages = [5, 1, 3, 0], [2, 6, 0, 4]
    device_indices = _page_indices(device_pages, PAGE_SIZE).to(DEVICE)
    host_indices_cpu = _page_indices(host_pages, PAGE_SIZE)
    host_pool.backup_from_device_all_layer(
        target, host_indices_cpu, device_indices, "kernel"
    )
    torch.cuda.synchronize()

    # Content: a (head group, layer range) rectangle of the page block holds,
    # layer by layer, K then V of the page's tokens for that group's heads.
    heads_per_group = head_num // head_group_num

    def expected_block(device_page, group, layers):
        rows = slice(device_page * PAGE_SIZE, (device_page + 1) * PAGE_SIZE)
        heads = slice(group * heads_per_group, (group + 1) * heads_per_group)
        return torch.stack(
            [
                torch.stack(
                    [original_k[layer][rows, heads], original_v[layer][rows, heads]]
                )
                for layer in layers
            ]
        ).cpu()

    draft_spans, target_spans = [], []
    for group in range(head_group_num):
        draft_spans.append(layout.chunk_span((LAYERS, total_layers), group))
        target_spans.append(layout.chunk_span((0, LAYERS), group))
        for host_page, device_page in zip(host_pages, device_pages):
            page = _flat_page(host_pool, host_page * PAGE_SIZE)
            for (offset, length), layers in (
                (draft_spans[-1], range(LAYERS, total_layers)),
                (target_spans[-1], range(LAYERS)),
            ):
                expected = expected_block(device_page, group, layers)
                assert expected.numel() * expected.element_size() == length
                assert torch.equal(
                    page[offset : offset + length],
                    expected.contiguous().view(torch.uint8).flatten(),
                ), f"group {group}, layers {list(layers)}"

    def reload(zeroed_spans):
        saved = host_pool.kv_buffer.clone()
        _zero_span(host_pool, host_indices_cpu, zeroed_spans)
        for buffer in (*k_buffers, *v_buffers):
            buffer.zero_()
        _load_all(host_pool, pools, host_indices_cpu.to(DEVICE), device_indices, LAYERS)
        host_pool.kv_buffer.copy_(saved)

    def assert_rows(layers, *, restored):
        for layer in layers:
            for buffers, originals in (
                (k_buffers, original_k),
                (v_buffers, original_v),
            ):
                got = buffers[layer][device_indices]
                want = originals[layer][device_indices]
                assert torch.equal(got, want if restored else torch.zeros_like(want)), (
                    f"layer {layer}, restored={restored}"
                )

    # Knock-out: the draft range holds the draft rows and nothing else.
    reload(draft_spans)
    assert_rows(range(LAYERS), restored=True)
    assert_rows(range(LAYERS, total_layers), restored=False)
    reload(target_spans)
    assert_rows(range(LAYERS), restored=False)
    assert_rows(range(LAYERS, total_layers), restored=True)
    reload([])
    assert_rows(range(total_layers), restored=True)


@pytest.mark.parametrize("draft_num", [1, 3])
def test_mla_page_unified_packed_drafts(draft_num):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

    dtype = torch.bfloat16

    def make_pool(layer_num):
        return MLATokenToKVPool(
            size=PAGE_SIZE * 8,
            page_size=PAGE_SIZE,
            dtype=dtype,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=layer_num,
            device=DEVICE,
            enable_memory_saver=False,
            start_layer=0,
            end_layer=layer_num,
        )

    target = make_pool(LAYERS)
    drafts = [make_pool(1) for _ in range(draft_num)]
    pools = [target, *drafts]
    total_layers = LAYERS + draft_num
    host_pool = _pinned_host_pool(
        MLATokenToKVPoolHost,
        target,
        draft_buffer_infos=[
            draft.get_device_pool_infos()[0].buffer_info for draft in drafts
        ],
        write_back_staging=_prepare_staging(
            target, is_mla=True, layer_num=total_layers
        ),
    )
    layout = host_pool.page_unified_layout
    assert (host_pool.target_layer_num, host_pool.layer_num) == (LAYERS, total_layers)
    assert layout.layer_num == total_layers

    generator = torch.Generator().manual_seed(20261117)
    buffers = [
        pool.kv_buffer[layer] for pool in pools for layer in range(pool.layer_num)
    ]
    _fill(buffers, generator, dtype)
    original = [buffer.clone() for buffer in buffers]

    device_pages, host_pages = [5, 1, 3, 0], [2, 6, 0, 4]
    device_indices = _page_indices(device_pages, PAGE_SIZE).to(DEVICE)
    host_indices_cpu = _page_indices(host_pages, PAGE_SIZE)
    host_pool.backup_from_device_all_layer(
        target, host_indices_cpu, device_indices, "kernel"
    )
    torch.cuda.synchronize()

    # Content: the draft layers are the trailing run of the page block.
    draft_span = layout.chunk_span((LAYERS, total_layers))
    target_span = layout.chunk_span((0, LAYERS))
    for host_page, device_page in zip(host_pages, device_pages):
        rows = slice(device_page * PAGE_SIZE, (device_page + 1) * PAGE_SIZE)
        page = _flat_page(host_pool, host_page * PAGE_SIZE)
        for (offset, length), layers in (
            (draft_span, range(LAYERS, total_layers)),
            (target_span, range(LAYERS)),
        ):
            expected = torch.stack(
                [original[layer][rows].reshape(PAGE_SIZE, -1) for layer in layers]
            ).cpu()
            assert expected.numel() * expected.element_size() == length
            assert torch.equal(
                page[offset : offset + length],
                expected.contiguous().view(torch.uint8).flatten(),
            ), f"layers {list(layers)}"

    def reload(zeroed_spans):
        saved = host_pool.kv_buffer.clone()
        _zero_span(host_pool, host_indices_cpu, zeroed_spans)
        for buffer in buffers:
            buffer.zero_()
        _load_all(host_pool, pools, host_indices_cpu.to(DEVICE), device_indices, LAYERS)
        host_pool.kv_buffer.copy_(saved)

    def assert_rows(layers, *, restored):
        for layer in layers:
            got = buffers[layer][device_indices]
            want = original[layer][device_indices]
            assert torch.equal(got, want if restored else torch.zeros_like(want)), (
                f"layer {layer}, restored={restored}"
            )

    reload([draft_span])
    assert_rows(range(LAYERS), restored=True)
    assert_rows(range(LAYERS, total_layers), restored=False)
    reload([target_span])
    assert_rows(range(LAYERS), restored=False)
    assert_rows(range(LAYERS, total_layers), restored=True)
    reload([])
    assert_rows(range(total_layers), restored=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
