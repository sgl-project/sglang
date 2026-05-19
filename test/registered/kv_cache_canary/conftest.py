"""Shared fixtures for canary srt-integration self-unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

from sglang.jit_kernel.kv_cache_canary_verify import (
    CANARY_SLOT_BYTES,
    RealKvHashMode,
    RealKvSource,
)
from sglang.srt.kv_cache_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


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


@pytest.fixture
def make_mha_pool(device):
    def _make(num_slots: int = 16, dim: int = 8, layer_num: int = 2) -> FakeMHAPool:
        k_layers = [
            torch.zeros(num_slots, dim, dtype=torch.float16, device=device)
            for _ in range(layer_num)
        ]
        v_layers = [
            torch.zeros(num_slots, dim, dtype=torch.float16, device=device)
            for _ in range(layer_num)
        ]
        return FakeMHAPool(layer_num=layer_num, k_buffer=k_layers, v_buffer=v_layers)

    return _make


@pytest.fixture
def make_mla_pool(device):
    def _make(num_slots: int = 16, dim: int = 16, layer_num: int = 2) -> FakeMLAPool:
        kv_layers = [
            torch.zeros(num_slots, dim, dtype=torch.float16, device=device)
            for _ in range(layer_num)
        ]
        return FakeMLAPool(layer_num=layer_num, kv_buffer=kv_layers)

    return _make


@pytest.fixture
def make_swa_pool(device):
    def _make(
        full_slots: int = 16, swa_slots: int = 8, dim: int = 8, layer_num: int = 1
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

    return _make


@pytest.fixture
def make_dsv4_pool(device):
    def _make(
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

    return _make


@pytest.fixture
def make_real_kv_source(device):
    def _make(
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

    return _make


@pytest.fixture
def make_buffer_group(device):
    def _make(
        *,
        kind: PoolKind = PoolKind.FULL,
        num_slots: int = 16,
        has_v_half: bool = True,
        real_kv_sources_k: tuple = (),
        real_kv_sources_v: tuple = (),
        swa_index_lut: Optional[torch.Tensor] = None,
    ) -> CanaryBufferGroup:
        k_head = torch.zeros(
            num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
        )
        k_tail = torch.zeros(
            num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
        )
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

    return _make


@pytest.fixture
def base_config():
    return CanaryConfig(mode=CanaryMode.RAISE, real_kv_hash_mode=RealKvHashMode.OFF)


@pytest.fixture
def make_req_to_token_pool(device):
    def _make(max_reqs: int = 8, max_seq_len: int = 32) -> SimpleNamespace:
        table = torch.zeros(max_reqs, max_seq_len, dtype=torch.int32, device=device)
        pool = SimpleNamespace(req_to_token=table, size=max_reqs)
        return pool

    return _make


@pytest.fixture
def make_forward_batch(device):
    def _make(
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

    return _make


@pytest.fixture
def make_radix_cache(device):
    """Build a real RadixCache by directly constructing TreeNodes, bypassing the heavy init path."""

    from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

    def _make(slot_lists: List[List[int]]) -> RadixCache:
        """slot_lists[0] = root.value (usually empty), [1+] = children chained linearly."""
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

    return _make
