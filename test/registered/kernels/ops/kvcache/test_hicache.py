import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_page_unified_load_back_jit_kernel,
    can_use_write_back_jit_kernel,
    transfer_hicache_one_layer_mla_page_unified_lf,
    transfer_hicache_one_layer_page_unified_lf,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_pin_memory,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=37, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or is_npu()
    or is_xpu()
    or not (is_cuda() or is_hip()),
    reason="HiCache JIT tests require CUDA/ROCm.",
)

DEVICE = "cuda"
PAGE_SIZE = 1 if is_hip() else 16
NUM_LAYERS = 2
POOL_SIZE = PAGE_SIZE * 8
MHA_ELEMENT_DIMS = [128, 256, 512, 1024]
MLA_ELEMENT_DIMS = [576]
LAYOUTS = ["layer_first", "page_first"]
STAGED_WRITE_BACK_PAGE_COUNTS = [1, 63, 64, 65, 67, 128, 129]


def _token_indices_for_pages(
    pages: torch.Tensor,
    page_size: int = PAGE_SIZE,
    device: str = DEVICE,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    parts = [
        torch.arange(
            int(page) * page_size,
            (int(page) + 1) * page_size,
            device=device,
            dtype=dtype,
        )
        for page in pages.tolist()
    ]
    return torch.cat(parts, dim=0)


def _pinned_host_pool(host_pool_cls, **kwargs):
    original_alloc = ALLOC_MEMORY_FUNCS[DEVICE]
    ALLOC_MEMORY_FUNCS[DEVICE] = alloc_with_pin_memory
    try:
        return host_pool_cls(
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            pin_memory=True,
            device="cpu",
            **kwargs,
        )
    finally:
        ALLOC_MEMORY_FUNCS[DEVICE] = original_alloc


def _copy_tensor_with_offset(tensor: torch.Tensor, offset: int) -> None:
    data = torch.arange(
        tensor.numel(), device=tensor.device, dtype=tensor.dtype
    ).view_as(tensor)
    tensor.copy_(data + offset)


def _assert_page_filled(tensor: torch.Tensor, page: int, value: float) -> None:
    page_slice = tensor[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]
    expected = torch.full_like(page_slice, value)
    assert torch.equal(page_slice.cpu(), expected.cpu())


def _run_transfer_roundtrip_mha(layout: str, element_dim: int) -> None:
    device_pool = MHATokenToKVPool(
        size=POOL_SIZE,
        page_size=PAGE_SIZE,
        head_num=element_dim // 128,
        head_dim=128,
        dtype=torch.bfloat16,
        layer_num=NUM_LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
    )
    host_pool = _pinned_host_pool(
        MHATokenToKVPoolHost,
        device_pool=device_pool,
        layout=layout,
    )
    assert host_pool.can_use_jit, (
        f"Expected JIT HiCache kernel for MHA dim={element_dim}"
    )

    for layer_id in range(NUM_LAYERS):
        _copy_tensor_with_offset(device_pool.k_buffer[layer_id], layer_id)
        _copy_tensor_with_offset(device_pool.v_buffer[layer_id], layer_id + 100)

    device_pages = torch.tensor([1, 2, 3], device=DEVICE, dtype=torch.int64)
    host_pages = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int64)
    device_indices = _token_indices_for_pages(device_pages)
    host_indices = _token_indices_for_pages(host_pages)
    host_indices_backup = (
        _token_indices_for_pages(host_pages, device="cpu")
        if layout == "page_first"
        else host_indices
    )

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices_backup, device_indices, "kernel"
    )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        for host_page, device_page in zip(host_pages.tolist(), device_pages.tolist()):
            host_start = host_page * PAGE_SIZE
            device_start = device_page * PAGE_SIZE
            assert torch.equal(
                host_pool.k_data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
                device_pool.k_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
            )
            assert torch.equal(
                host_pool.v_data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
                device_pool.v_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
            )

    for layer_id in range(NUM_LAYERS):
        device_pool.k_buffer[layer_id].zero_()
        device_pool.v_buffer[layer_id].zero_()

    load_pages = torch.tensor([4, 5, 6], device=DEVICE, dtype=torch.int64)
    load_indices = _token_indices_for_pages(load_pages)
    for layer_id in range(NUM_LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices, load_indices, layer_id, "kernel"
        )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        for host_page, device_page in zip(host_pages.tolist(), load_pages.tolist()):
            host_start = host_page * PAGE_SIZE
            device_start = device_page * PAGE_SIZE
            assert torch.equal(
                device_pool.k_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
                host_pool.k_data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
            )
            assert torch.equal(
                device_pool.v_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
                host_pool.v_data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
            )


def _run_transfer_roundtrip_mla(layout: str, element_dim: int) -> None:
    device_pool = MLATokenToKVPool(
        size=POOL_SIZE,
        page_size=PAGE_SIZE,
        kv_lora_rank=element_dim - 64,
        qk_rope_head_dim=64,
        dtype=torch.bfloat16,
        layer_num=NUM_LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
    )
    host_pool = _pinned_host_pool(
        MLATokenToKVPoolHost,
        device_pool=device_pool,
        layout=layout,
    )
    assert host_pool.can_use_jit, (
        f"Expected JIT HiCache kernel for MLA dim={element_dim}"
    )

    for layer_id in range(NUM_LAYERS):
        _copy_tensor_with_offset(device_pool.kv_buffer[layer_id], layer_id)

    device_pages = torch.tensor([1, 2, 3], device=DEVICE, dtype=torch.int64)
    host_pages = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int64)
    device_indices = _token_indices_for_pages(device_pages)
    host_indices = _token_indices_for_pages(host_pages)
    host_indices_backup = (
        _token_indices_for_pages(host_pages, device="cpu")
        if layout == "page_first"
        else host_indices
    )

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices_backup, device_indices, "kernel"
    )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        for host_page, device_page in zip(host_pages.tolist(), device_pages.tolist()):
            host_start = host_page * PAGE_SIZE
            device_start = device_page * PAGE_SIZE
            assert torch.equal(
                host_pool.data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
                device_pool.kv_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
            )

    for layer_id in range(NUM_LAYERS):
        device_pool.kv_buffer[layer_id].zero_()

    load_pages = torch.tensor([4, 5, 6], device=DEVICE, dtype=torch.int64)
    load_indices = _token_indices_for_pages(load_pages)
    for layer_id in range(NUM_LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices, load_indices, layer_id, "kernel"
        )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        for host_page, device_page in zip(host_pages.tolist(), load_pages.tolist()):
            host_start = host_page * PAGE_SIZE
            device_start = device_page * PAGE_SIZE
            assert torch.equal(
                device_pool.kv_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
                host_pool.data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
            )


def _run_page_first_staged_write_back_mha(
    layout: str, element_dim: int, page_count: int
) -> None:
    pool_size = PAGE_SIZE * (page_count + 8)
    head_num = (
        element_dim // 128 if element_dim >= 128 and element_dim % 128 == 0 else 1
    )
    head_dim = element_dim // head_num
    device_pool = MHATokenToKVPool(
        size=pool_size,
        page_size=PAGE_SIZE,
        head_num=head_num,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        layer_num=NUM_LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
    )
    host_pool = _pinned_host_pool(
        MHATokenToKVPoolHost,
        device_pool=device_pool,
        layout=layout,
    )
    assert can_use_write_back_jit_kernel(
        element_size=element_dim * host_pool.dtype.itemsize,
    )
    assert host_pool.can_use_write_back_jit
    if element_dim * host_pool.dtype.itemsize % 128 != 0:
        assert not host_pool.can_use_jit
    assert host_pool.staging_page_capacity > 0
    if page_count > 64:
        assert host_pool.staging_page_capacity < page_count

    for layer_id in range(NUM_LAYERS):
        _copy_tensor_with_offset(device_pool.k_buffer[layer_id], layer_id)
        _copy_tensor_with_offset(device_pool.v_buffer[layer_id], layer_id + 100)
    host_pool.k_buffer.fill_(-7)
    host_pool.v_buffer.fill_(-11)

    device_pages = torch.arange(
        2,
        2 + page_count,
        device=DEVICE,
        dtype=torch.int64,
    )
    host_pages = torch.arange(
        page_count,
        0,
        -1,
        dtype=torch.int64,
    )
    src_index_dtype = torch.int32 if page_count == 64 else torch.int64
    device_indices = _token_indices_for_pages(device_pages, dtype=src_index_dtype)
    host_indices = _token_indices_for_pages(host_pages, device="cpu")
    assert not host_indices.is_cuda
    expected_k = [
        device_pool.k_buffer[layer_id][device_indices.to(dtype=torch.int64)].cpu()
        for layer_id in range(NUM_LAYERS)
    ]
    expected_v = [
        device_pool.v_buffer[layer_id][device_indices.to(dtype=torch.int64)].cpu()
        for layer_id in range(NUM_LAYERS)
    ]

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, "kernel"
    )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        for host_page, device_page in zip(host_pages.tolist(), device_pages.tolist()):
            host_start = host_page * PAGE_SIZE
            device_start = device_page * PAGE_SIZE
            assert torch.equal(
                host_pool.k_data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
                device_pool.k_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
            )
            assert torch.equal(
                host_pool.v_data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
                device_pool.v_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
            )

    for layer_id in range(NUM_LAYERS):
        for untouched_page in [0, page_count + 1]:
            _assert_page_filled(host_pool.k_data_refs[layer_id], untouched_page, -7)
            _assert_page_filled(host_pool.v_data_refs[layer_id], untouched_page, -11)

    for layer_id in range(NUM_LAYERS):
        device_pool.k_buffer[layer_id].zero_()
        device_pool.v_buffer[layer_id].zero_()
    load_indices = device_indices.to(dtype=torch.int64)
    host_indices_load = _token_indices_for_pages(host_pages)
    for layer_id in range(NUM_LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices_load, load_indices, layer_id, "kernel"
        )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        assert torch.equal(
            device_pool.k_buffer[layer_id][load_indices].cpu(), expected_k[layer_id]
        )
        assert torch.equal(
            device_pool.v_buffer[layer_id][load_indices].cpu(), expected_v[layer_id]
        )


def _run_page_first_staged_write_back_mla(
    layout: str, element_dim: int, page_count: int
) -> None:
    pool_size = PAGE_SIZE * (page_count + 8)
    device_pool = MLATokenToKVPool(
        size=pool_size,
        page_size=PAGE_SIZE,
        kv_lora_rank=element_dim - 64,
        qk_rope_head_dim=64,
        dtype=torch.bfloat16,
        layer_num=NUM_LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
    )
    host_pool = _pinned_host_pool(
        MLATokenToKVPoolHost,
        device_pool=device_pool,
        layout=layout,
    )
    assert can_use_write_back_jit_kernel(
        element_size=element_dim * host_pool.dtype.itemsize,
    )
    assert host_pool.can_use_write_back_jit
    if element_dim * host_pool.dtype.itemsize % 128 != 0:
        assert not host_pool.can_use_jit
    assert host_pool.staging_page_capacity > 0
    if page_count > 64:
        assert host_pool.staging_page_capacity < page_count

    for layer_id in range(NUM_LAYERS):
        _copy_tensor_with_offset(device_pool.kv_buffer[layer_id], layer_id)
    host_pool.kv_buffer.fill_(-13)

    device_pages = torch.arange(
        2,
        2 + page_count,
        device=DEVICE,
        dtype=torch.int64,
    )
    host_pages = torch.arange(
        page_count,
        0,
        -1,
        dtype=torch.int64,
    )
    src_index_dtype = torch.int32 if page_count == 64 else torch.int64
    device_indices = _token_indices_for_pages(device_pages, dtype=src_index_dtype)
    host_indices = _token_indices_for_pages(host_pages, device="cpu")
    assert not host_indices.is_cuda
    expected = [
        device_pool.kv_buffer[layer_id][device_indices.to(dtype=torch.int64)].cpu()
        for layer_id in range(NUM_LAYERS)
    ]

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, "kernel"
    )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        for host_page, device_page in zip(host_pages.tolist(), device_pages.tolist()):
            host_start = host_page * PAGE_SIZE
            device_start = device_page * PAGE_SIZE
            assert torch.equal(
                host_pool.data_refs[layer_id][
                    host_start : host_start + PAGE_SIZE
                ].cpu(),
                device_pool.kv_buffer[layer_id][
                    device_start : device_start + PAGE_SIZE
                ].cpu(),
            )

    for layer_id in range(NUM_LAYERS):
        for untouched_page in [0, page_count + 1]:
            _assert_page_filled(host_pool.data_refs[layer_id], untouched_page, -13)

    for layer_id in range(NUM_LAYERS):
        device_pool.kv_buffer[layer_id].zero_()
    load_indices = device_indices.to(dtype=torch.int64)
    host_indices_load = _token_indices_for_pages(host_pages)
    for layer_id in range(NUM_LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices_load, load_indices, layer_id, "kernel"
        )
    torch.cuda.synchronize()

    for layer_id in range(NUM_LAYERS):
        assert torch.equal(
            device_pool.kv_buffer[layer_id][load_indices].cpu(), expected[layer_id]
        )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("element_dim", MHA_ELEMENT_DIMS)
def test_hicache_transfer_mha(layout: str, element_dim: int) -> None:
    _run_transfer_roundtrip_mha(layout, element_dim)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("element_dim", MLA_ELEMENT_DIMS)
def test_hicache_transfer_mla(layout: str, element_dim: int) -> None:
    _run_transfer_roundtrip_mla(layout, element_dim)


@pytest.mark.parametrize("layout", ["page_first"])
@pytest.mark.parametrize("element_dim", MHA_ELEMENT_DIMS)
@pytest.mark.parametrize("page_count", STAGED_WRITE_BACK_PAGE_COUNTS)
def test_hicache_page_first_staged_write_back_mha(
    layout: str, element_dim: int, page_count: int
) -> None:
    _run_page_first_staged_write_back_mha(layout, element_dim, page_count)


@pytest.mark.parametrize("layout", ["page_first"])
@pytest.mark.parametrize("element_dim", MLA_ELEMENT_DIMS)
@pytest.mark.parametrize("page_count", STAGED_WRITE_BACK_PAGE_COUNTS)
def test_hicache_page_first_staged_write_back_mla(
    layout: str, element_dim: int, page_count: int
) -> None:
    _run_page_first_staged_write_back_mla(layout, element_dim, page_count)


def test_hicache_page_first_staged_write_back_mha_staged_only_alignment() -> None:
    _run_page_first_staged_write_back_mha("page_first", 72, 65)


def test_hicache_page_first_staged_write_back_mla_staged_only_alignment() -> None:
    _run_page_first_staged_write_back_mla("page_first", 72, 65)


# ---------------------------------------------------------------------------
# page_unified load-back (host -> device), HiCacheKernel::run_one_page_unified
#
# Unlike the cases above, these drive the op directly rather than through a host
# pool: the layout has no pool wrapper yet. The reference is built by plain
# PyTorch indexing of the host tensor rather than by re-deriving the kernel's
# offsets, and the payload is seeded noise rather than a counter -- a counter is
# periodic in the layer stride, so a layer mix-up would compare equal.
# ---------------------------------------------------------------------------

PAGE_UNIFIED_NUM_PAGES = 8
PAGE_UNIFIED_SENTINEL = 255
# Repeats a source page and is unordered; reading the same host token into two
# device slots is a real prefix-sharing case.
PAGE_UNIFIED_SRC_PATTERN = [5, 0, 3, 5, 1, 2, 4, 3]


def _page_unified_host_pages(shape, seed):
    """Pinned host pages of exactly representable noise."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 128, shape, generator=generator, dtype=torch.int32)


def _page_unified_index_pairs(token_count, host_tokens, device_tokens):
    """Host token ids (repeats allowed, unordered) paired with distinct device ids.

    Device ids must be distinct: two items writing the same device slot would
    make the expected result depend on which block happens to land last.
    """
    src_ids = [i % host_tokens for i in PAGE_UNIFIED_SRC_PATTERN[:token_count]]
    dst_ids = list(reversed(range(token_count)))
    assert token_count <= device_tokens
    return src_ids, dst_ids


@pytest.mark.parametrize(
    "groups,layers,page_size,heads_per_group,dim",
    [(1, 1, 1, 1, 16), (3, 5, 3, 2, 24), (8, 3, 64, 1, 128)],
)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("token_count", [0, 1, 8])
def test_page_unified_load_back(
    groups, layers, page_size, heads_per_group, dim, dtype, index_dtype, token_count
):
    heads = groups * heads_per_group
    host_tokens = PAGE_UNIFIED_NUM_PAGES * page_size
    device_tokens = host_tokens + page_size
    src = (
        _page_unified_host_pages(
            (
                PAGE_UNIFIED_NUM_PAGES,
                groups,
                layers,
                2,
                page_size,
                heads_per_group,
                dim,
            ),
            1234,
        )
        .to(dtype)
        .pin_memory()
    )
    src_ids, dst_ids = _page_unified_index_pairs(
        token_count, host_tokens, device_tokens
    )

    src_pages = torch.tensor(src_ids, dtype=index_dtype, device=DEVICE)
    dst_pages = torch.tensor(dst_ids, dtype=index_dtype, device=DEVICE)
    k_dst = torch.full(
        (device_tokens, heads, dim), PAGE_UNIFIED_SENTINEL, dtype=dtype, device=DEVICE
    )
    v_dst = torch.full_like(k_dst, PAGE_UNIFIED_SENTINEL)

    # Every layer is loaded separately: the layer stride is the axis a wrong
    # permutation is most likely to get wrong, and only sweeping it catches that.
    for layer_id in range(layers):
        k_dst.fill_(PAGE_UNIFIED_SENTINEL)
        v_dst.fill_(PAGE_UNIFIED_SENTINEL)
        expected_k = torch.full_like(k_dst.cpu(), PAGE_UNIFIED_SENTINEL)
        expected_v = torch.full_like(expected_k, PAGE_UNIFIED_SENTINEL)
        for src_token, dst_token in zip(src_ids, dst_ids):
            page, token = divmod(src_token, page_size)
            for group in range(groups):
                head_slice = slice(
                    group * heads_per_group, (group + 1) * heads_per_group
                )
                expected_k[dst_token, head_slice] = src[page, group, layer_id, 0, token]
                expected_v[dst_token, head_slice] = src[page, group, layer_id, 1, token]

        transfer_hicache_one_layer_page_unified_lf(
            k_dst, v_dst, src, src_pages, dst_pages, layer_id
        )
        torch.cuda.synchronize()
        # Compare every element, including rows that should retain the sentinel.
        assert torch.equal(k_dst.cpu(), expected_k)
        assert torch.equal(v_dst.cpu(), expected_v)


@pytest.mark.parametrize(
    "layers,page_size,dim", [(1, 1, 16), (5, 3, 512), (3, 64, 576)]
)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("token_count", [0, 1, 8])
def test_page_unified_mla_load_back(
    layers, page_size, dim, dtype, index_dtype, token_count
):
    host_tokens = PAGE_UNIFIED_NUM_PAGES * page_size
    device_tokens = host_tokens + page_size
    src = (
        _page_unified_host_pages((PAGE_UNIFIED_NUM_PAGES, layers, page_size, dim), 5678)
        .to(dtype)
        .pin_memory()
    )
    src_ids, dst_ids = _page_unified_index_pairs(
        token_count, host_tokens, device_tokens
    )

    src_pages = torch.tensor(src_ids, dtype=index_dtype, device=DEVICE)
    dst_pages = torch.tensor(dst_ids, dtype=index_dtype, device=DEVICE)
    dst = torch.full(
        (device_tokens, 1, dim), PAGE_UNIFIED_SENTINEL, dtype=dtype, device=DEVICE
    )

    for layer_id in range(layers):
        dst.fill_(PAGE_UNIFIED_SENTINEL)
        expected = torch.full_like(dst.cpu(), PAGE_UNIFIED_SENTINEL)
        for src_token, dst_token in zip(src_ids, dst_ids):
            page, token = divmod(src_token, page_size)
            expected[dst_token, 0] = src[page, layer_id, token]

        transfer_hicache_one_layer_mla_page_unified_lf(
            dst, src, src_pages, dst_pages, layer_id
        )
        torch.cuda.synchronize()
        assert torch.equal(dst.cpu(), expected)


def test_page_unified_load_back_non_default_stream():
    """All work must land on the caller's stream, including the host reads."""
    groups, layers, page_size, heads_per_group, dim = 2, 3, 8, 2, 64
    heads = groups * heads_per_group
    src = (
        _page_unified_host_pages(
            (
                PAGE_UNIFIED_NUM_PAGES,
                groups,
                layers,
                2,
                page_size,
                heads_per_group,
                dim,
            ),
            99,
        )
        .to(torch.bfloat16)
        .pin_memory()
    )
    src_ids = list(range(0, PAGE_UNIFIED_NUM_PAGES * page_size, 3))
    dst_ids = list(range(len(src_ids)))
    layer_id = layers - 1

    expected_k = torch.full(
        (len(dst_ids), heads, dim), PAGE_UNIFIED_SENTINEL, dtype=torch.bfloat16
    )
    expected_v = torch.full_like(expected_k, PAGE_UNIFIED_SENTINEL)
    for src_token, dst_token in zip(src_ids, dst_ids):
        page, token = divmod(src_token, page_size)
        for group in range(groups):
            head_slice = slice(group * heads_per_group, (group + 1) * heads_per_group)
            expected_k[dst_token, head_slice] = src[page, group, layer_id, 0, token]
            expected_v[dst_token, head_slice] = src[page, group, layer_id, 1, token]

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        src_pages = torch.tensor(src_ids, dtype=torch.int64, device=DEVICE)
        dst_pages = torch.tensor(dst_ids, dtype=torch.int64, device=DEVICE)
        k_dst = torch.full(
            (len(dst_ids), heads, dim),
            PAGE_UNIFIED_SENTINEL,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        v_dst = torch.full_like(k_dst, PAGE_UNIFIED_SENTINEL)
        transfer_hicache_one_layer_page_unified_lf(
            k_dst, v_dst, src, src_pages, dst_pages, layer_id
        )
    stream.synchronize()
    assert torch.equal(k_dst.cpu(), expected_k)
    assert torch.equal(v_dst.cpu(), expected_v)


# The C++ launcher also checks the page and device-row byte sizes against the
# compiled kElementSize, for callers that reach the module directly. Those are
# unreachable here by construction: the entry points derive every geometry
# argument from the tensor shapes they are handed.
@pytest.mark.parametrize(
    "invalid,match",
    [
        ("rank", "Expected \\(page"),
        ("kv_axis", "K/V dimension"),
        ("group_alignment", "16-byte aligned"),
        ("noncontiguous", "must be contiguous"),
        ("layer_id", "layer id out of range"),
        ("device_heads", "device row byte size mismatch"),
        ("index_length", "indices length"),
    ],
)
def test_page_unified_load_back_invalid_input(invalid, match):
    groups, layers, page_size, heads_per_group, dim = 2, 3, 4, 1, 16
    shape = [PAGE_UNIFIED_NUM_PAGES, groups, layers, 2, page_size, heads_per_group, dim]
    if invalid == "kv_axis":
        shape[3] = 3
    elif invalid == "group_alignment":
        shape[6] = 7
    elif invalid == "rank":
        shape.pop()
    src = torch.zeros(shape, dtype=torch.float16).pin_memory()
    if invalid == "noncontiguous":
        src = src.transpose(1, 2).contiguous().transpose(1, 2)

    heads = groups * heads_per_group
    if invalid == "device_heads":
        heads += 1
    k_dst = torch.zeros((8, heads, dim), dtype=torch.float16, device=DEVICE)
    v_dst = torch.zeros_like(k_dst)
    src_pages = torch.zeros(2, dtype=torch.int64, device=DEVICE)
    dst_pages = torch.zeros(2, dtype=torch.int64, device=DEVICE)
    if invalid == "index_length":
        dst_pages = dst_pages[:1]
    layer_id = layers if invalid == "layer_id" else 0

    # TVM FFI uses its own exception type for C++ RuntimeCheck failures.
    with pytest.raises(Exception, match=match):
        transfer_hicache_one_layer_page_unified_lf(
            k_dst, v_dst, src, src_pages, dst_pages, layer_id
        )


@pytest.mark.parametrize(
    "invalid,match",
    [
        ("rank", "Expected MLA"),
        ("dimension", "must be positive"),
        ("alignment", "16-byte aligned"),
        ("layer_id", "layer id out of range"),
    ],
)
def test_page_unified_mla_load_back_invalid_input(invalid, match):
    layers, page_size, dim = 3, 4, 16
    if invalid == "dimension":
        dim = 0
    elif invalid == "alignment":
        dim = 7
    src = torch.zeros(
        (PAGE_UNIFIED_NUM_PAGES, layers, page_size, dim), dtype=torch.float16
    )
    if dim > 0:
        src = src.pin_memory()
    if invalid == "rank":
        src = src.unsqueeze(1)
    dst = torch.zeros((8, 1, max(dim, 1)), dtype=torch.float16, device=DEVICE)
    src_pages = torch.zeros(2, dtype=torch.int64, device=DEVICE)
    dst_pages = torch.zeros(2, dtype=torch.int64, device=DEVICE)
    layer_id = layers if invalid == "layer_id" else 0

    with pytest.raises(Exception, match=match):
        transfer_hicache_one_layer_mla_page_unified_lf(
            dst, src, src_pages, dst_pages, layer_id
        )


@pytest.mark.parametrize("group_bytes,expected", [(0, False), (8, False), (256, True)])
def test_can_use_page_unified_load_back_jit_kernel(group_bytes, expected):
    """The probe owns the alignment precondition and compiles the specialisation."""
    assert (
        can_use_page_unified_load_back_jit_kernel(group_bytes=group_bytes) is expected
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
