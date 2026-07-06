import sys

import pytest
import torch

from sglang.jit_kernel.hisparse import (
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
    transfer_cache_dsv4_mla,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd")
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or is_npu()
    or is_xpu()
    or not (is_cuda() or is_hip()),
    reason="HiSparse JIT tests require CUDA/ROCm.",
)

DEVICE = "cuda"
DTYPE = torch.float32
KV_DIM = 8
HOT_BUFFER_SIZE = 4
PADDED_BUFFER_SIZE = HOT_BUFFER_SIZE + 1
HOST_CACHE_SIZE = 16
DEVICE_CACHE_SIZE = 16
ITEM_SIZE_BYTES = KV_DIM * torch.empty((), dtype=DTYPE).element_size()
DSV4_PAGE_SIZE = 64
DSV4_VALUE_BYTES = 576
DSV4_SCALE_BYTES = 8
DSV4_ITEM_BYTES = DSV4_VALUE_BYTES + DSV4_SCALE_BYTES
DSV4_PAGE_BYTES = ((DSV4_ITEM_BYTES * DSV4_PAGE_SIZE + 575) // 576) * 576
DSV4_SCALE_OFFSET = DSV4_VALUE_BYTES * DSV4_PAGE_SIZE


def _host_cache() -> torch.Tensor:
    host_cache = torch.empty(
        (HOST_CACHE_SIZE, 1, KV_DIM), dtype=DTYPE, device="cpu", pin_memory=True
    )
    host_cache.copy_(torch.arange(host_cache.numel(), dtype=DTYPE).view_as(host_cache))
    return host_cache


def _dsv4_token_pattern(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    value = (
        (torch.arange(DSV4_VALUE_BYTES, dtype=torch.int16) + seed)
        .remainder(256)
        .to(torch.uint8)
    )
    scale = (
        (torch.arange(DSV4_SCALE_BYTES, dtype=torch.int16) + seed + 17)
        .remainder(256)
        .to(torch.uint8)
    )
    return value, scale


def _write_dsv4_token(cache: torch.Tensor, loc: int, seed: int) -> None:
    page = loc // DSV4_PAGE_SIZE
    offset = loc % DSV4_PAGE_SIZE
    value, scale = _dsv4_token_pattern(seed)
    cache[page, offset * DSV4_VALUE_BYTES : (offset + 1) * DSV4_VALUE_BYTES].copy_(
        value.to(cache.device)
    )
    scale_start = DSV4_SCALE_OFFSET + offset * DSV4_SCALE_BYTES
    cache[page, scale_start : scale_start + DSV4_SCALE_BYTES].copy_(
        scale.to(cache.device)
    )


def _read_dsv4_token(cache: torch.Tensor, loc: int) -> torch.Tensor:
    page = loc // DSV4_PAGE_SIZE
    offset = loc % DSV4_PAGE_SIZE
    value = cache[page, offset * DSV4_VALUE_BYTES : (offset + 1) * DSV4_VALUE_BYTES]
    scale_start = DSV4_SCALE_OFFSET + offset * DSV4_SCALE_BYTES
    scale = cache[page, scale_start : scale_start + DSV4_SCALE_BYTES]
    return torch.cat([value, scale])


def _dsv4_ptrs(cache: torch.Tensor) -> torch.Tensor:
    return torch.tensor([cache.data_ptr()], dtype=torch.uint64, device=DEVICE)


def _run_kernel(
    *,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    lru_slots: torch.Tensor,
    seq_len: int | None = None,
    seq_lens: torch.Tensor | None = None,
    seq_lens_dtype: torch.dtype = torch.int32,
    req_pool_indices: torch.Tensor | None = None,
    req_to_token: torch.Tensor | None = None,
    full_to_hisparse_device_index_mapping: torch.Tensor | None = None,
    num_real_reqs: int | None = None,
) -> torch.Tensor:
    batch_size = top_k_tokens.shape[0]
    if req_pool_indices is None:
        req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=DEVICE)
    if seq_lens is None:
        seq_lens = torch.full(
            (batch_size,), seq_len, dtype=seq_lens_dtype, device=DEVICE
        )
    if num_real_reqs is None:
        num_real_reqs = batch_size

    out = torch.full_like(top_k_tokens, -1)
    load_cache_to_device_buffer_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=ITEM_SIZE_BYTES,
        num_top_k=top_k_tokens.shape[1],
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=1,
        block_size=256,
        num_real_reqs=torch.tensor([num_real_reqs], dtype=torch.int32, device=DEVICE),
        req_to_token=req_to_token,
        full_to_hisparse_device_index_mapping=full_to_hisparse_device_index_mapping,
    )
    torch.cuda.synchronize()
    return out


def _make_state(
    device_buffer_locs_rows: list[list[int]],
    device_buffer_tokens_rows: list[list[int]],
    newest_tokens: list[int],
):
    host_cache = _host_cache()
    device_buffer = torch.full(
        (DEVICE_CACHE_SIZE, 1, KV_DIM), -1, dtype=DTYPE, device=DEVICE
    )
    device_buffer_locs = torch.tensor(
        device_buffer_locs_rows, dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens = torch.tensor(
        device_buffer_tokens_rows, dtype=torch.int32, device=DEVICE
    )
    lru_slots = (
        torch.arange(HOT_BUFFER_SIZE, dtype=torch.int16, device=DEVICE)
        .view(1, -1)
        .repeat(device_buffer_locs.shape[0], 1)
    )
    host_cache_locs = (
        torch.arange(HOST_CACHE_SIZE, dtype=torch.int64, device=DEVICE)
        .view(1, -1)
        .repeat(device_buffer_locs.shape[0], 1)
    )

    # Slots 0..3 participate in LRU; slot 4 is the reserved newest slot.
    for rid, newest_token in enumerate(newest_tokens):
        for slot, token in enumerate(device_buffer_tokens_rows[rid][:HOT_BUFFER_SIZE]):
            if token >= 0:
                device_buffer[device_buffer_locs[rid, slot]].copy_(
                    host_cache[token].to(DEVICE, non_blocking=True)
                )
        device_buffer[device_buffer_locs[rid, HOT_BUFFER_SIZE]].copy_(
            host_cache[newest_token].to(DEVICE, non_blocking=True)
        )
    torch.cuda.synchronize()

    return {
        "host_cache": host_cache,
        "device_buffer": device_buffer,
        "device_buffer_locs": device_buffer_locs,
        "device_buffer_tokens": device_buffer_tokens,
        "lru_slots": lru_slots,
        "host_cache_locs": host_cache_locs,
    }


@pytest.mark.skipif(is_hip(), reason="DSV4 paged-layout HiSparse test is CUDA-only.")
def test_transfer_cache_dsv4_mla_copies_paged_token() -> None:
    src_cache = torch.zeros((2, DSV4_PAGE_BYTES), dtype=torch.uint8, device=DEVICE)
    dst_cache = torch.zeros(
        (2, DSV4_PAGE_BYTES), dtype=torch.uint8, device="cpu", pin_memory=True
    )
    src_loc = DSV4_PAGE_SIZE + 6
    dst_loc = DSV4_PAGE_SIZE + 1
    _write_dsv4_token(src_cache, src_loc, seed=41)

    transfer_cache_dsv4_mla(
        src_ptrs=_dsv4_ptrs(src_cache),
        dst_ptrs=_dsv4_ptrs(dst_cache),
        src_indices=torch.tensor([src_loc], dtype=torch.int64, device=DEVICE),
        dst_indices=torch.tensor([dst_loc], dtype=torch.int64, device=DEVICE),
    )
    torch.cuda.synchronize()

    assert torch.equal(
        _read_dsv4_token(dst_cache, dst_loc).to(DEVICE),
        _read_dsv4_token(src_cache, src_loc),
    )


@pytest.mark.skipif(is_hip(), reason="DSV4 paged-layout HiSparse test is CUDA-only.")
def test_dsv4_swap_in_reads_paged_host_layout() -> None:
    host_cache = torch.zeros(
        (2, DSV4_PAGE_BYTES), dtype=torch.uint8, device="cpu", pin_memory=True
    )
    device_buffer = torch.zeros((2, DSV4_PAGE_BYTES), dtype=torch.uint8, device=DEVICE)
    host_loc = DSV4_PAGE_SIZE + 1
    swap_loc = DSV4_PAGE_SIZE + 12
    _write_dsv4_token(host_cache, host_loc, seed=41)

    top_k_tokens = torch.tensor([[3]], dtype=torch.int32, device=DEVICE)
    device_buffer_tokens = torch.full(
        (1, PADDED_BUFFER_SIZE), -1, dtype=torch.int32, device=DEVICE
    )
    host_cache_locs = torch.zeros((1, 8), dtype=torch.int64, device=DEVICE)
    host_cache_locs[0, 3] = host_loc
    device_buffer_locs = torch.tensor(
        [[swap_loc, swap_loc + 1, swap_loc + 2, swap_loc + 3, swap_loc + 4]],
        dtype=torch.int32,
        device=DEVICE,
    )
    lru_slots = torch.arange(HOT_BUFFER_SIZE, dtype=torch.int16, device=DEVICE).view(
        1, -1
    )
    out = torch.full_like(top_k_tokens, -1)

    load_cache_to_device_buffer_dsv4_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=DEVICE),
        lru_slots=lru_slots,
        item_size_bytes=DSV4_ITEM_BYTES,
        num_top_k=1,
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=1,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()

    assert out.item() == swap_loc
    assert torch.equal(
        _read_dsv4_token(device_buffer, swap_loc),
        _read_dsv4_token(host_cache, host_loc).to(DEVICE),
    )


def _long_case():
    # One-request baseline used by the stateful cases below:
    # req 0 LRU slots      : [0, 1, 2, 3]
    # req 0 cached tokens  : slot0->1, slot1->4, slot2->2, slot3->5
    # req 0 physical locs  : slot0->9, slot1->7, slot2->3, slot3->5
    # req 0 newest slot    : slot4/newest -> token 7 at physical loc 11
    return _make_state([[9, 7, 3, 5, 11]], [[1, 4, 2, 5, -1]], [7])


@pytest.mark.parametrize("seq_lens_dtype", [torch.int32, torch.int64])
def test_load_cache_to_device_buffer_fast_path(seq_lens_dtype: torch.dtype) -> None:
    host_cache = _host_cache()
    device_buffer = torch.arange(
        DEVICE_CACHE_SIZE * KV_DIM, dtype=DTYPE, device=DEVICE
    ).view(DEVICE_CACHE_SIZE, 1, KV_DIM)
    device_buffer_before = device_buffer.clone()
    device_buffer_locs = torch.tensor(
        [[13, 9, 5, 1, 15]], dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens = torch.tensor(
        [[10, 11, 12, 13, -1]], dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens_before = device_buffer_tokens.clone()
    lru_slots = torch.tensor([[0, 1, 2, 3]], dtype=torch.int16, device=DEVICE)
    lru_slots_before = lru_slots.clone()

    # Short-sequence layout:
    # token position 0 -> physical loc 13
    # token position 1 -> physical loc 9
    # token position 2 -> physical loc 5
    #
    # seq_len <= HOT_BUFFER_SIZE should skip host loads and LRU mutations,
    # so top_k_tokens acts like direct indexing into device_buffer_locs.
    out = _run_kernel(
        top_k_tokens=torch.tensor([[2, 0, 1]], dtype=torch.int32, device=DEVICE),
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=torch.arange(
            HOST_CACHE_SIZE, dtype=torch.int64, device=DEVICE
        ).view(1, -1),
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        lru_slots=lru_slots,
        seq_len=3,
        seq_lens_dtype=seq_lens_dtype,
    )

    assert torch.equal(out.cpu(), torch.tensor([[5, 13, 9]], dtype=torch.int32))
    assert torch.equal(device_buffer_tokens.cpu(), device_buffer_tokens_before.cpu())
    assert torch.equal(lru_slots.cpu(), lru_slots_before.cpu())
    assert torch.equal(device_buffer.cpu(), device_buffer_before.cpu())


def test_load_cache_to_device_buffer_hits_newest_and_updates_lru() -> None:
    state = _long_case()

    # Query [4, 2, 7]:
    # 4 hits slot1 -> loc 7
    # 2 hits slot2 -> loc 3
    # 7 is the newest token -> reserved newest loc 11
    #
    # Hits move to the MRU tail, so [0, 1, 2, 3] becomes [0, 3, 1, 2].
    out = _run_kernel(
        top_k_tokens=torch.tensor([[4, 2, 7]], dtype=torch.int32, device=DEVICE),
        seq_len=8,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[7, 3, 11]], dtype=torch.int32))
    assert torch.equal(
        state["device_buffer_tokens"].cpu(),
        torch.tensor([[1, 4, 2, 5, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["lru_slots"].cpu(), torch.tensor([[0, 3, 1, 2]], dtype=torch.int16)
    )


def test_load_cache_to_device_buffer_miss_uses_updated_lru_slot() -> None:
    state = _long_case()

    # Step 1: touch tokens [4, 2], so LRU becomes [0, 3, 1, 2].
    # Step 2: query token 6, which is a miss.
    # The kernel should reuse the new LRU head slot0, whose physical loc is 9.
    # This round has no regular hits, so the freshly loaded miss slot ends up at the tail.
    _run_kernel(
        top_k_tokens=torch.tensor([[4, 2]], dtype=torch.int32, device=DEVICE),
        seq_len=8,
        **state,
    )
    out = _run_kernel(
        top_k_tokens=torch.tensor([[6]], dtype=torch.int32, device=DEVICE),
        seq_len=8,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[9]], dtype=torch.int32))
    assert torch.equal(
        state["device_buffer_tokens"].cpu(),
        torch.tensor([[6, 4, 2, 5, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["lru_slots"].cpu(), torch.tensor([[3, 1, 2, 0]], dtype=torch.int16)
    )
    assert torch.equal(state["device_buffer"][9].cpu(), state["host_cache"][6])


def test_load_cache_to_device_buffer_resident_row_translates_logical_locs() -> None:
    state = _long_case()
    state["device_buffer_locs"][0, 0] = -1
    device_buffer_before = state["device_buffer"].clone()
    device_buffer_tokens_before = state["device_buffer_tokens"].clone()
    lru_slots_before = state["lru_slots"].clone()

    req_to_token = torch.tensor(
        [[5, 2, 8, 1, 4, 7, 9, 3]], dtype=torch.int32, device=DEVICE
    )
    full_to_hisparse_device_index_mapping = torch.full(
        (16,), -1, dtype=torch.int64, device=DEVICE
    )
    resident_locs = torch.tensor(
        [40, 41, 42, 43, 44, 45, 46, 47], dtype=torch.int64, device=DEVICE
    )
    full_to_hisparse_device_index_mapping[req_to_token[0].long()] = resident_locs

    out = _run_kernel(
        top_k_tokens=torch.tensor([[6, 2, 0, 7]], dtype=torch.int32, device=DEVICE),
        seq_len=8,
        req_to_token=req_to_token,
        full_to_hisparse_device_index_mapping=full_to_hisparse_device_index_mapping,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[46, 42, 40, 47]], dtype=torch.int32))
    assert torch.equal(state["device_buffer"].cpu(), device_buffer_before.cpu())
    assert torch.equal(
        state["device_buffer_tokens"].cpu(), device_buffer_tokens_before.cpu()
    )
    assert torch.equal(state["lru_slots"].cpu(), lru_slots_before.cpu())


def test_load_cache_to_device_buffer_multiple_misses_copy_all_slots() -> None:
    state = _make_state(
        [[9, 7, 3, 5, 11]],
        [[0, 1, 2, 3, -1]],
        [8],
    )

    out = _run_kernel(
        top_k_tokens=torch.tensor([[4, 5, 6, 7]], dtype=torch.int32, device=DEVICE),
        seq_len=9,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[9, 7, 3, 5]], dtype=torch.int32))
    assert torch.equal(
        state["device_buffer_tokens"].cpu(),
        torch.tensor([[4, 5, 6, 7, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["lru_slots"].cpu(), torch.tensor([[0, 1, 2, 3]], dtype=torch.int16)
    )
    for token, loc in zip([4, 5, 6, 7], [9, 7, 3, 5]):
        assert torch.equal(
            state["device_buffer"][loc].cpu(), state["host_cache"][token]
        )


def test_load_cache_to_device_buffer_batched_with_padding() -> None:
    state = _make_state(
        [
            [9, 7, 3, 5, 11],
            [12, 10, 8, 6, 14],
            [15, 4, 2, 1, 13],
        ],
        [
            [1, 4, 2, 5, -1],
            [0, 1, 2, 3, -1],
            [9, 8, 7, 6, -1],
        ],
        [7, 4, 5],
    )
    padded_tokens_before = state["device_buffer_tokens"][2].clone()
    padded_lru_before = state["lru_slots"][2].clone()

    # req 0: long path
    #   cached tokens/locs : 1@9, 4@7, 2@3, 5@5, newest 7@11
    #   query [4, 6, 7]    : hit loc 7, miss into slot0/loc 9, newest loc 11
    #   LRU update         : remaining evictables [2, 3], then miss [0], then hit [1]
    #                      : [0, 1, 2, 3] -> [2, 3, 0, 1]
    #
    # req 1: fast path
    #   seq_len = 3 <= HOT_BUFFER_SIZE, so [2, 1, 0] maps directly to locs [8, 10, 12]
    #
    # req 2: padded block
    #   num_real_reqs = 2 means this row must be ignored entirely.
    out = _run_kernel(
        top_k_tokens=torch.tensor(
            [[4, 6, 7], [2, 1, 0], [9, 8, 7]], dtype=torch.int32, device=DEVICE
        ),
        seq_lens=torch.tensor([8, 3, 8], dtype=torch.int32, device=DEVICE),
        num_real_reqs=2,
        **state,
    )

    assert torch.equal(
        out.cpu(),
        torch.tensor([[7, 9, 11], [8, 10, 12], [-1, -1, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["device_buffer_tokens"][:2].cpu(),
        torch.tensor([[6, 4, 2, 5, -1], [0, 1, 2, 3, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["lru_slots"][:2].cpu(),
        torch.tensor([[2, 3, 0, 1], [0, 1, 2, 3]], dtype=torch.int16),
    )
    assert torch.equal(
        state["device_buffer_tokens"][2].cpu(), padded_tokens_before.cpu()
    )
    assert torch.equal(state["lru_slots"][2].cpu(), padded_lru_before.cpu())
    assert torch.equal(state["device_buffer"][9].cpu(), state["host_cache"][6])


def test_load_cache_to_device_buffer_dsv4_mla_miss_copy_layout() -> None:
    # Both the host cache and the device buffer use the page-padded C4 layout,
    # matching DeepSeekV4PagedHostPool, the backup/write path, and the swap-in
    # kernel on both CUDA and ROCm. The miss copy must read the host source with
    # paged addressing (get_pointer_paged), not a linear per-item stride.
    num_pages = (HOST_CACHE_SIZE + DSV4_PAGE_SIZE - 1) // DSV4_PAGE_SIZE

    state = _long_case()
    host_cache = torch.zeros(
        (num_pages, DSV4_PAGE_BYTES),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    )
    for token in range(HOST_CACHE_SIZE):
        _write_dsv4_token(host_cache, token, seed=token + 1)

    device_buffer = torch.full(
        (num_pages, DSV4_PAGE_BYTES),
        0xFF,
        dtype=torch.uint8,
        device=DEVICE,
    )
    out = torch.full((1, 1), -1, dtype=torch.int32, device=DEVICE)

    # Token 6 is a miss in _long_case(), so it should be copied into evict slot 0,
    # whose physical device loc is 9.
    load_cache_to_device_buffer_dsv4_mla(
        top_k_tokens=torch.tensor([[6]], dtype=torch.int32, device=DEVICE),
        device_buffer_tokens=state["device_buffer_tokens"],
        host_cache_locs=state["host_cache_locs"],
        device_buffer_locs=state["device_buffer_locs"],
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=DEVICE),
        lru_slots=state["lru_slots"],
        item_size_bytes=DSV4_ITEM_BYTES,
        num_top_k=1,
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=DSV4_PAGE_SIZE,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()

    assert torch.equal(out.cpu(), torch.tensor([[9]], dtype=torch.int32))

    # host_cache_locs[token=6] == 6 in _long_case(); evict slot 0 -> device loc 9.
    assert torch.equal(
        _read_dsv4_token(device_buffer, 9).cpu(),
        _read_dsv4_token(host_cache, 6),
    )


@pytest.mark.skipif(
    not is_hip(), reason="Covers a ROCm wavefront64 LRU writeback regression."
)
def test_load_cache_to_device_buffer_rocm_large_lru_writeback() -> None:
    top_k = 2048
    hot_buffer_size = 4096
    seq_len = 7299
    kv_dim = 4
    item_size_bytes = kv_dim * torch.empty((), dtype=DTYPE).element_size()

    top_k_tokens = torch.cat(
        [
            torch.arange(1000, 2000, dtype=torch.int32),
            torch.arange(5000, 6048, dtype=torch.int32),
        ]
    ).view(1, -1)
    device_buffer_tokens = torch.arange(hot_buffer_size, dtype=torch.int32).view(1, -1)
    device_buffer_locs = torch.arange(hot_buffer_size + 1, dtype=torch.int32).view(
        1, -1
    )
    lru_slots = torch.arange(hot_buffer_size, dtype=torch.int16).view(1, -1)
    host_cache_locs = torch.arange(seq_len, dtype=torch.int64).view(1, -1)

    top_k_tokens = top_k_tokens.to(DEVICE)
    device_buffer_tokens = device_buffer_tokens.to(DEVICE)
    device_buffer_locs = device_buffer_locs.to(DEVICE)
    lru_slots = lru_slots.to(DEVICE)
    host_cache_locs = host_cache_locs.to(DEVICE)

    host_cache = torch.empty((seq_len, 1, kv_dim), dtype=DTYPE, pin_memory=True)
    host_cache.zero_()
    device_buffer = torch.empty(
        (hot_buffer_size + 1, 1, kv_dim), dtype=DTYPE, device=DEVICE
    )
    out = torch.full_like(top_k_tokens, -1)

    load_cache_to_device_buffer_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=DEVICE),
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=1,
        block_size=1024,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()

    expected_lru = torch.cat(
        [
            torch.arange(2048, 4096, dtype=torch.int16),
            torch.arange(0, 1000, dtype=torch.int16),
            torch.arange(2000, 2048, dtype=torch.int16),
            torch.arange(1000, 2000, dtype=torch.int16),
        ]
    )
    assert torch.equal(lru_slots.cpu().view(-1), expected_lru)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
