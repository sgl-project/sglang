"""Shared helpers for canary srt-integration self-unit tests.

Helpers are plain module-level functions / dataclasses plus a
patch_fake_pool_helpers() context manager that replaces the legacy autouse
fixture. Importers use ``from sglang.test.kv_canary.fixtures import ...``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterator, List, Literal, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    RealKvHashMode,
    RealKvSource,
)
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.pool_patch.api import register_canary_adapter
from sglang.srt.kv_canary.pool_patch.utils import (
    _make_row_source,
    _patch_buf_info_method,
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


@register_canary_adapter(FakeMHAPool)
class _FakeMHAAdapter:
    def is_swa(self, pool: FakeMHAPool) -> bool:
        return False

    def has_v_half(self, pool: FakeMHAPool) -> bool:
        return True

    def build_real_kv_sources(
        self,
        pool: FakeMHAPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        buf = pool.k_buffer[0] if half == "K" else pool.v_buffer[0]
        return _make_row_source(layer_buffer=buf, read_bytes=read_bytes)

    def install_full_group(self, pool: FakeMHAPool, group: CanaryBufferGroup) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_contiguous_buf_infos",
            group=group,
            has_v_half=True,
            page_size=pool.page_size,
        )

    def install_swa_group(self, pool: FakeMHAPool, group: CanaryBufferGroup) -> None:
        raise NotImplementedError


@register_canary_adapter(FakeMLAPool)
class _FakeMLAAdapter:
    def is_swa(self, pool: FakeMLAPool) -> bool:
        return False

    def has_v_half(self, pool: FakeMLAPool) -> bool:
        return False

    def build_real_kv_sources(
        self,
        pool: FakeMLAPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        if half == "V":
            return ()
        return _make_row_source(layer_buffer=pool.kv_buffer[0], read_bytes=read_bytes)

    def install_full_group(self, pool: FakeMLAPool, group: CanaryBufferGroup) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_contiguous_buf_infos",
            group=group,
            has_v_half=False,
            page_size=pool.page_size,
        )

    def install_swa_group(self, pool: FakeMLAPool, group: CanaryBufferGroup) -> None:
        raise NotImplementedError


@register_canary_adapter(FakeSWAPool)
class _FakeSWAAdapter:
    def is_swa(self, pool: FakeSWAPool) -> bool:
        return True

    def has_v_half(self, pool: FakeSWAPool) -> bool:
        return True

    def build_real_kv_sources(
        self,
        pool: FakeSWAPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        sub_pool = pool.full_kv_pool if kind is PoolKind.FULL else pool.swa_kv_pool
        buf = sub_pool.k_buffer[0] if half == "K" else sub_pool.v_buffer[0]
        return _make_row_source(layer_buffer=buf, read_bytes=read_bytes)

    def install_full_group(self, pool: FakeSWAPool, group: CanaryBufferGroup) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_state_buf_infos",
            group=group,
            has_v_half=True,
            page_size=pool.page_size,
        )

    def install_swa_group(self, pool: FakeSWAPool, group: CanaryBufferGroup) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_state_buf_infos",
            group=group,
            has_v_half=True,
            page_size=pool.page_size,
        )


@register_canary_adapter(FakeDsv4Pool)
class _FakeDsv4Adapter:
    def is_swa(self, pool: FakeDsv4Pool) -> bool:
        return True

    def has_v_half(self, pool: FakeDsv4Pool) -> bool:
        return False

    def build_real_kv_sources(
        self,
        pool: FakeDsv4Pool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        if half == "V":
            return ()
        sub_pool = pool.full_kv_pool if kind is PoolKind.FULL else pool.swa_kv_pool
        return _make_row_source(
            layer_buffer=sub_pool.kv_buffer[0], read_bytes=read_bytes
        )

    def install_full_group(self, pool: FakeDsv4Pool, group: CanaryBufferGroup) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_state_buf_infos",
            group=group,
            has_v_half=False,
            page_size=pool.page_size,
        )

    def install_swa_group(self, pool: FakeDsv4Pool, group: CanaryBufferGroup) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_state_buf_infos",
            group=group,
            has_v_half=False,
            page_size=pool.page_size,
        )


@contextmanager
def patch_fake_pool_helpers() -> Iterator[None]:
    """Patch _slot_count / _swa_index_lut helpers in pool_patch to recognize Fake pools."""
    from sglang.srt.kv_canary.pool_patch import utils as pp

    original_slot_count = pp._slot_count
    original_swa_lut = pp._swa_index_lut

    def patched_slot_count(pool, kind):
        if isinstance(pool, FakeMHAPool):
            return int(pool.k_buffer[0].shape[0])
        if isinstance(pool, FakeMLAPool):
            return int(pool.kv_buffer[0].shape[0])
        if isinstance(pool, FakeSWAPool):
            sub = pool.full_kv_pool if kind is PoolKind.FULL else pool.swa_kv_pool
            return int(sub.k_buffer[0].shape[0])
        if isinstance(pool, FakeDsv4Pool):
            sub = pool.full_kv_pool if kind is PoolKind.FULL else pool.swa_kv_pool
            return int(sub.kv_buffer[0].shape[0])
        return original_slot_count(pool, kind)

    def patched_swa_lut(pool):
        if isinstance(pool, (FakeSWAPool, FakeDsv4Pool)):
            return pool.full_to_swa_index_mapping
        return original_swa_lut(pool)

    pp._slot_count = patched_slot_count
    pp._swa_index_lut = patched_swa_lut
    try:
        yield
    finally:
        pp._slot_count = original_slot_count
        pp._swa_index_lut = original_swa_lut
