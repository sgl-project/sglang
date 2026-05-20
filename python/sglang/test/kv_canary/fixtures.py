"""Shared helpers for canary srt-integration self-unit tests.

Plain module-level functions and dataclasses; importers use
``from sglang.test.kv_canary.fixtures import ...``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    RealKvHashMode,
    RealKvSource,
)
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.pool_patch.adapters.mha import attach_mha
from sglang.srt.kv_canary.pool_patch.adapters.mla import attach_mla
from sglang.srt.kv_canary.pool_patch.api import register_pool_attacher
from sglang.srt.kv_canary.pool_patch.utils import (
    alloc_canary_buf_pair,
    ensure_swa_lut_int32,
    make_row_source,
    patch_buf_info_method,
    swa_index_lut,
)

CPU_DEVICE: torch.device = torch.device("cpu")


@dataclass
class FakeMHAPool:
    layer_num: int
    k_buffer: List[torch.Tensor]
    v_buffer: List[torch.Tensor]
    page_size: int = 1

    def get_contiguous_buf_infos(self):
        ptrs = [b.data_ptr() for b in self.k_buffer] + [
            b.data_ptr() for b in self.v_buffer
        ]
        lens = [b.nbytes for b in self.k_buffer] + [b.nbytes for b in self.v_buffer]
        item_lens = [b[0].nbytes * self.page_size for b in self.k_buffer] + [
            b[0].nbytes * self.page_size for b in self.v_buffer
        ]
        return ptrs, lens, item_lens


@dataclass
class FakeMLAPool:
    layer_num: int
    kv_buffer: List[torch.Tensor]
    page_size: int = 1

    def get_contiguous_buf_infos(self):
        ptrs = [b.data_ptr() for b in self.kv_buffer]
        lens = [b.nbytes for b in self.kv_buffer]
        item_lens = [b[0].nbytes * self.page_size for b in self.kv_buffer]
        return ptrs, lens, item_lens


@dataclass
class FakeSwaSubPool:
    k_buffer: List[torch.Tensor]
    v_buffer: List[torch.Tensor]


@dataclass
class FakeSWAPool:
    full_kv_pool: object
    swa_kv_pool: object
    full_to_swa_index_mapping: torch.Tensor
    page_size: int = 1

    def get_state_buf_infos(self):
        all_k = self.full_kv_pool.k_buffer + self.swa_kv_pool.k_buffer
        all_v = self.full_kv_pool.v_buffer + self.swa_kv_pool.v_buffer
        ptrs = [b.data_ptr() for b in all_k] + [b.data_ptr() for b in all_v]
        lens = [b.nbytes for b in all_k] + [b.nbytes for b in all_v]
        item_lens = [b[0].nbytes * self.page_size for b in all_k] + [
            b[0].nbytes * self.page_size for b in all_v
        ]
        return ptrs, lens, item_lens


@dataclass
class FakeDsv4Pool:
    """DSV4-style packed pool: MLA (no v-half) + SWA dual + page_size 128."""

    full_kv_pool: object
    swa_kv_pool: object
    full_to_swa_index_mapping: torch.Tensor
    page_size: int = 128

    def get_state_buf_infos(self):
        all_kv = self.full_kv_pool.kv_buffer + self.swa_kv_pool.kv_buffer
        ptrs = [b.data_ptr() for b in all_kv]
        lens = [b.nbytes for b in all_kv]
        item_lens = [b[0].nbytes * self.page_size for b in all_kv]
        return ptrs, lens, item_lens


@dataclass
class FakeDsv4SubPool:
    kv_buffer: List[torch.Tensor]


def make_mha_pool(
    device: torch.device = CPU_DEVICE,
    *,
    num_slots: int = 16,
    dim: int = 8,
    layer_num: int = 2,
) -> FakeMHAPool:
    k_layers = [
        torch.zeros(num_slots, dim, dtype=torch.float16, device=device)
        for _ in range(layer_num)
    ]
    v_layers = [
        torch.zeros(num_slots, dim, dtype=torch.float16, device=device)
        for _ in range(layer_num)
    ]
    return FakeMHAPool(layer_num=layer_num, k_buffer=k_layers, v_buffer=v_layers)


def make_mla_pool(
    device: torch.device = CPU_DEVICE,
    *,
    num_slots: int = 16,
    dim: int = 16,
    layer_num: int = 2,
) -> FakeMLAPool:
    kv_layers = [
        torch.zeros(num_slots, dim, dtype=torch.float16, device=device)
        for _ in range(layer_num)
    ]
    return FakeMLAPool(layer_num=layer_num, kv_buffer=kv_layers)


def make_swa_pool(
    device: torch.device = CPU_DEVICE,
    *,
    full_slots: int = 16,
    swa_slots: int = 8,
    dim: int = 8,
    layer_num: int = 1,
) -> FakeSWAPool:
    full = FakeSwaSubPool(
        k_buffer=[
            torch.zeros(full_slots, dim, dtype=torch.float16, device=device)
            for _ in range(layer_num)
        ],
        v_buffer=[
            torch.zeros(full_slots, dim, dtype=torch.float16, device=device)
            for _ in range(layer_num)
        ],
    )
    swa = FakeSwaSubPool(
        k_buffer=[
            torch.zeros(swa_slots, dim, dtype=torch.float16, device=device)
            for _ in range(layer_num)
        ],
        v_buffer=[
            torch.zeros(swa_slots, dim, dtype=torch.float16, device=device)
            for _ in range(layer_num)
        ],
    )
    lut = torch.full((full_slots + 1,), -1, dtype=torch.int32, device=device)
    lut[:swa_slots] = torch.arange(swa_slots, dtype=torch.int32, device=device)
    return FakeSWAPool(
        full_kv_pool=full, swa_kv_pool=swa, full_to_swa_index_mapping=lut
    )


def make_dsv4_pool(
    device: torch.device = CPU_DEVICE,
    *,
    full_slots: int = 16,
    swa_slots: int = 8,
    dim: int = 32,
    layer_num: int = 1,
    page_size: int = 128,
) -> FakeDsv4Pool:
    full = FakeDsv4SubPool(
        kv_buffer=[
            torch.zeros(full_slots, dim, dtype=torch.float16, device=device)
            for _ in range(layer_num)
        ]
    )
    swa = FakeDsv4SubPool(
        kv_buffer=[
            torch.zeros(swa_slots, dim, dtype=torch.float16, device=device)
            for _ in range(layer_num)
        ]
    )
    lut = torch.full((full_slots + 1,), -1, dtype=torch.int32, device=device)
    lut[:swa_slots] = torch.arange(swa_slots, dtype=torch.int32, device=device)
    return FakeDsv4Pool(
        full_kv_pool=full,
        swa_kv_pool=swa,
        full_to_swa_index_mapping=lut,
        page_size=page_size,
    )


def make_real_kv_source(
    device: torch.device = CPU_DEVICE,
    *,
    num_slots: int = 16,
    num_bytes_per_token: int = 8,
    read_bytes: int = 4,
    page_size: int = 1,
) -> RealKvSource:
    tensor = torch.zeros(
        num_slots, num_bytes_per_token, dtype=torch.uint8, device=device
    )
    return RealKvSource(
        tensor=tensor,
        page_size=page_size,
        num_bytes_per_token=num_bytes_per_token,
        read_bytes=read_bytes,
    )


def make_buffer_group(
    device: torch.device = CPU_DEVICE,
    *,
    kind: PoolKind = PoolKind.FULL,
    num_slots: int = 16,
    has_v_half: bool = True,
    real_kv_sources_k: tuple = (),
    real_kv_sources_v: tuple = (),
    swa_index_lut: Optional[torch.Tensor] = None,
) -> CanaryBufferGroup:
    k_head = torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    k_tail = torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    v_head = (
        torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
        if has_v_half
        else None
    )
    v_tail = (
        torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
        if has_v_half
        else None
    )
    return CanaryBufferGroup(
        kind=kind,
        k_head=k_head,
        k_tail=k_tail,
        v_head=v_head,
        v_tail=v_tail,
        real_kv_sources_k=real_kv_sources_k,
        real_kv_sources_v=real_kv_sources_v,
        swa_index_lut=swa_index_lut,
    )


def make_base_config() -> CanaryConfig:
    return CanaryConfig(mode=CanaryMode.RAISE, real_kv_hash_mode=RealKvHashMode.OFF)


def make_req_to_token_pool(
    device: torch.device = CPU_DEVICE,
    *,
    max_reqs: int = 8,
    max_seq_len: int = 32,
) -> SimpleNamespace:
    table = torch.zeros(max_reqs, max_seq_len, dtype=torch.int32, device=device)
    return SimpleNamespace(req_to_token=table, size=max_reqs)


def make_forward_batch(
    device: torch.device = CPU_DEVICE,
    *,
    req_pool_indices: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    extend_prefix_lens: Optional[torch.Tensor] = None,
    extend_seq_lens: Optional[torch.Tensor] = None,
    is_extend: bool = False,
    input_ids: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    out_cache_loc: Optional[torch.Tensor] = None,
) -> SimpleNamespace:
    mode = SimpleNamespace(
        is_extend=lambda: is_extend,
        is_mixed=lambda: False,
    )
    return SimpleNamespace(
        forward_mode=mode,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
    )


def make_radix_cache(slot_lists: List[List[int]], device: torch.device = CPU_DEVICE):
    """Build a real RadixCache by directly constructing TreeNodes, bypassing the heavy init path.

    slot_lists[0] = root.value (usually empty), [1+] = children chained linearly.
    """
    from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

    cache = RadixCache.__new__(RadixCache)
    cache.device = device
    cache.page_size = 1
    cache.disable = False

    root = TreeNode()
    root.value = torch.tensor(
        slot_lists[0] if slot_lists else [], dtype=torch.int32, device=device
    )
    cache.root_node = root

    current = root
    for child_slots in slot_lists[1:]:
        child = TreeNode()
        child.value = torch.tensor(child_slots, dtype=torch.int32, device=device)
        child.parent = current
        current.children[child.id] = child
        current = child

    return cache


def _attach_fake_swa(
    *,
    pool: FakeSWAPool,
    device: torch.device,
    read_bytes: int,
    allocator: Optional[object] = None,
) -> tuple[CanaryBufferGroup, ...]:
    """FakeSWAPool only exposes ``get_state_buf_infos`` (no separate ``get_contiguous_buf_infos``),
    so both FULL and SWA groups splice through it.
    """
    ensure_swa_lut_int32(pool=pool, allocator=allocator)
    full_group = _build_fake_swa_group(
        sub_pool=pool.full_kv_pool,
        kind=PoolKind.FULL,
        device=device,
        read_bytes=read_bytes,
        swa_lut=None,
    )
    swa_group = _build_fake_swa_group(
        sub_pool=pool.swa_kv_pool,
        kind=PoolKind.SWA,
        device=device,
        read_bytes=read_bytes,
        swa_lut=swa_index_lut(pool),
    )
    patch_buf_info_method(
        pool,
        method_name="get_state_buf_infos",
        group=full_group,
        has_v_half=True,
        page_size=pool.page_size,
    )
    patch_buf_info_method(
        pool,
        method_name="get_state_buf_infos",
        group=swa_group,
        has_v_half=True,
        page_size=pool.page_size,
    )
    return (full_group, swa_group)


def _build_fake_swa_group(
    *,
    sub_pool: FakeSwaSubPool,
    kind: PoolKind,
    device: torch.device,
    read_bytes: int,
    swa_lut: Optional[torch.Tensor],
) -> CanaryBufferGroup:
    num_slots = int(sub_pool.k_buffer[0].shape[0])
    k_head, k_tail = alloc_canary_buf_pair(num_slots=num_slots, device=device)
    v_head, v_tail = alloc_canary_buf_pair(num_slots=num_slots, device=device)
    return CanaryBufferGroup(
        kind=kind,
        k_head=k_head,
        k_tail=k_tail,
        v_head=v_head,
        v_tail=v_tail,
        real_kv_sources_k=make_row_source(
            layer_buffer=sub_pool.k_buffer[0], read_bytes=read_bytes
        ),
        real_kv_sources_v=make_row_source(
            layer_buffer=sub_pool.v_buffer[0], read_bytes=read_bytes
        ),
        swa_index_lut=swa_lut,
    )


def _attach_fake_dsv4(
    *,
    pool: FakeDsv4Pool,
    device: torch.device,
    read_bytes: int,
    allocator: Optional[object] = None,
) -> tuple[CanaryBufferGroup, ...]:
    """FakeDsv4Pool is an MLA-style packed pool (single ``kv_buffer`` per sub-pool, no V half),
    exposing only ``get_state_buf_infos``.
    """
    ensure_swa_lut_int32(pool=pool, allocator=allocator)
    full_group = _build_fake_dsv4_group(
        sub_pool=pool.full_kv_pool,
        kind=PoolKind.FULL,
        device=device,
        read_bytes=read_bytes,
        swa_lut=None,
    )
    swa_group = _build_fake_dsv4_group(
        sub_pool=pool.swa_kv_pool,
        kind=PoolKind.SWA,
        device=device,
        read_bytes=read_bytes,
        swa_lut=swa_index_lut(pool),
    )
    patch_buf_info_method(
        pool,
        method_name="get_state_buf_infos",
        group=full_group,
        has_v_half=False,
        page_size=pool.page_size,
    )
    patch_buf_info_method(
        pool,
        method_name="get_state_buf_infos",
        group=swa_group,
        has_v_half=False,
        page_size=pool.page_size,
    )
    return (full_group, swa_group)


def _build_fake_dsv4_group(
    *,
    sub_pool: FakeDsv4SubPool,
    kind: PoolKind,
    device: torch.device,
    read_bytes: int,
    swa_lut: Optional[torch.Tensor],
) -> CanaryBufferGroup:
    num_slots = int(sub_pool.kv_buffer[0].shape[0])
    k_head, k_tail = alloc_canary_buf_pair(num_slots=num_slots, device=device)
    return CanaryBufferGroup(
        kind=kind,
        k_head=k_head,
        k_tail=k_tail,
        v_head=None,
        v_tail=None,
        real_kv_sources_k=make_row_source(
            layer_buffer=sub_pool.kv_buffer[0], read_bytes=read_bytes
        ),
        real_kv_sources_v=(),
        swa_index_lut=swa_lut,
    )


register_pool_attacher(FakeMHAPool, attach_mha)
register_pool_attacher(FakeMLAPool, attach_mla)
register_pool_attacher(FakeSWAPool, _attach_fake_swa)
register_pool_attacher(FakeDsv4Pool, _attach_fake_dsv4)
