# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Save/load cost of the unified L3 KV layout adapter.

The unified key scheme stores every L3 object in one byte order so that
deployments with different TP/PP splits share a keyspace. Host pools do not hold
their KV in that order, so the adapter converts at the L3 boundary:

    save:  pool --gather_unified_chunks--> staging --put--> L3
    load:  L3 --get--> staging --scatter_unified_chunks--> pool

This module measures what that conversion costs and, because the cost is decided
almost entirely by the choice of byte order, also reports the descriptor matrix
that choice implies.

Two numbers per case:

* **staged %** - how much of the payload is memcpy'd through the staging buffer
  (0 = the slab is already pool-contiguous and transfers in place).
* **descriptors** - how many maximal contiguous spans the chunk decomposes into
  *in unified order*. A transport with a scatter/gather put (mooncake's
  ``batch_put_from_multi_buffers`` / ``batch_get_into_multi_buffers``, already
  wired in ``mooncake_store.py``) moves the chunk with zero host copies using
  that many DMA descriptors. 1 descriptor == fully zero-copy. Descriptors only
  pay off well above the transport's per-descriptor overhead, so the span
  *size* matters as much as the count.

Run as a script for the full report:

    python3 test/registered/unit/mem_cache/test_hicache_unified_layout_perf.py --full
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import sys
import time
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_key_scheme import KVCacheLayoutAdapter
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.pool_host import common as pool_common
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

# The host layout the shipped unified byte order targets: (2, page_num, head,
# layer, page_size, dim). Not a production layout - the pools below add it so
# the benchmark can price the zero-copy alternative honestly.
PROJECTED_LAYOUT = "page_head_layer_direct"

# Fabric rates used to project end-to-end save throughput:
# ~one 400 Gb/s NIC, and a multi-NIC node.
NET_GBS = (50, 200)

MHA_LAYOUTS = ("layer_first", "page_first", "page_first_direct", "page_head")
MLA_LAYOUTS = ("layer_first", "page_first", "page_first_direct")

# Candidate unified byte orders for the MHA K/V half. "head_major" is what the
# branch ships; "layer_major" is the same rectangle serialized the other way
# round. MLA has no head axis, so both collapse to (layer, token, dim) there.
HEAD_MAJOR = "head_major"  # (head, layer, token, dim)
LAYER_MAJOR = "layer_major"  # (layer, token, head, dim)


# --------------------------------------------------------------------------- #
# Pools
# --------------------------------------------------------------------------- #


def _plain_alloc(dims, dtype, device, pin_memory, allocator):
    return torch.zeros(dims, dtype=dtype)


@contextlib.contextmanager
def _plain_host_alloc():
    """Run the pools' real ``init_kv_buffer`` without CUDA pinning.

    ``pin_memory=True`` is impossible on a CPU-only host (the default path goes
    through ``torch.cuda.cudart()``), but the shape tables and the derived
    ``token_stride_size`` / ``layout_dim`` are exactly what we want to exercise,
    so patch the allocator rather than reimplement the shapes here.
    """
    with mock.patch.dict(
        pool_common.ALLOC_MEMORY_FUNCS, {"cpu": _plain_alloc}, clear=False
    ):
        yield


@contextlib.contextmanager
def quiet_gc():
    """gc.collect() costs ~200 ms in a torch process - keep it out of the clock."""
    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
        gc.unfreeze()


class _BenchMHAPool(MHATokenToKVPoolHost):
    """Bare host pool: production ``init_kv_buffer``, no device pool, no CUDA.

    Built without ``HostKVCache.__init__`` (same idiom as
    test_hicache_key_scheme.py) because the adapter path only reads
    layout/geometry/kv_buffer - but the buffer itself comes from the real
    method, so the shapes cannot drift from production.
    """

    def __init__(self, *, layout, layers, heads, head_dim, page_size, pages, dtype):
        self.layout = layout
        self.layer_num = layers
        self.head_num = heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.page_num = pages
        self.size = pages * page_size
        self.dtype = dtype
        self.pin_memory = False
        self.allocator = None
        self.device = "cpu"
        self.device_pool = SimpleNamespace(device="cpu")
        if layout == PROJECTED_LAYOUT:
            # Not a production layout: no init_kv_buffer branch exists for it.
            self.token_stride_size = heads * head_dim * dtype.itemsize
            self.layout_dim = self.token_stride_size * layers
            self.kv_buffer = torch.zeros(
                (2, pages, heads, layers, page_size, head_dim), dtype=dtype
            )
        else:
            with _plain_host_alloc():
                self.kv_buffer = self.init_kv_buffer()

    def _page_kv_view_unified(self, index: int):
        if self.layout == PROJECTED_LAYOUT:
            # (2, pn, H, L, P, D) -> the production (layer, token, head, dim).
            return self.kv_buffer[:, index // self.page_size].permute(0, 2, 3, 1, 4)
        return super()._page_kv_view_unified(index)

    def page_view_head_major(self, index: int):
        """The page as (2, head, layer, token, dim) - the order the adapter
        used before the flip to (layer, token, head, dim). Kept so the report
        can still price the rejected order side by side."""
        P = self.page_size
        if self.layout == "layer_first":
            return self.kv_buffer[:, :, index : index + P].permute(0, 3, 1, 2, 4)
        if self.layout == "page_first":
            return self.kv_buffer[:, index : index + P].permute(0, 3, 2, 1, 4)
        if self.layout == "page_first_direct":
            return self.kv_buffer[:, index // P].permute(0, 3, 1, 2, 4)
        if self.layout == "page_head":
            return self.kv_buffer[:, index // P].permute(0, 1, 3, 2, 4)
        return self.kv_buffer[:, index // P]  # page_head_layer_direct: identity

    def page_view_layer_major(self, index: int):
        """The same page as (2, layer, token, head, dim) - the alternative
        unified order, in which ``page_first_direct`` is the identity."""
        P = self.page_size
        if self.layout == "layer_first":
            return self.kv_buffer[:, :, index : index + P]
        if self.layout == "page_first":
            return self.kv_buffer[:, index : index + P].permute(0, 2, 1, 3, 4)
        if self.layout == "page_first_direct":
            return self.kv_buffer[:, index // P]
        if self.layout == "page_head":
            return self.kv_buffer[:, index // P].permute(0, 3, 2, 1, 4)
        return self.kv_buffer[:, index // P].permute(0, 2, 3, 1, 4)

    def fill_from_logical(self, logical):
        """``logical[p]`` is (2, head, layer, token, dim) for page ``p``."""
        for p, page in enumerate(logical):
            lo = p * self.page_size
            if self.layout == "layer_first":
                self.kv_buffer[:, :, lo : lo + self.page_size] = page.permute(
                    0, 2, 3, 1, 4
                )
            elif self.layout == "page_first":
                self.kv_buffer[:, lo : lo + self.page_size] = page.permute(
                    0, 3, 2, 1, 4
                )
            elif self.layout == "page_first_direct":
                self.kv_buffer[:, p] = page.permute(0, 2, 3, 1, 4)
            elif self.layout == "page_head":
                self.kv_buffer[:, p] = page.permute(0, 1, 3, 2, 4)
            else:
                self.kv_buffer[:, p] = page


class _BenchMLAPool(MLATokenToKVPoolHost):
    def __init__(self, *, layout, layers, kv_dim, page_size, pages, dtype):
        self.layout = layout
        self.layer_num = layers
        self.kv_cache_dim = kv_dim
        self.page_size = page_size
        self.page_num = pages
        self.size = pages * page_size
        self.dtype = dtype
        self.pin_memory = False
        self.allocator = None
        self.device = "cpu"
        self.device_pool = SimpleNamespace(device="cpu")
        with _plain_host_alloc():
            self.kv_buffer = self.init_kv_buffer()

    def fill_from_logical(self, logical):
        """``logical[p]`` is (layer, token, 1, dim) for page ``p``."""
        for p, page in enumerate(logical):
            lo = p * self.page_size
            if self.layout == "layer_first":
                self.kv_buffer[:, lo : lo + self.page_size] = page
            elif self.layout == "page_first":
                self.kv_buffer[lo : lo + self.page_size] = page.permute(1, 0, 2, 3)
            else:
                self.kv_buffer[p] = page


# --------------------------------------------------------------------------- #
# Span accounting
# --------------------------------------------------------------------------- #


def descriptor_count(shape, strides) -> tuple[int, int]:
    """(number of contiguous spans, elements per span) walking ``shape`` in its
    own dim order - i.e. in unified byte order.

    Merges dims inward-out exactly like ``is_contiguous``; the first dim whose
    stride does not continue the accumulated block ends the span, and every
    outer dim multiplies the span count. This is the descriptor list a
    scatter/gather transport would build, so it is a regular decomposition:
    accidental adjacency between two outer iterations is not merged (the true
    maximal-run count can be a hair lower, never higher).
    """
    block = 1
    i = len(shape) - 1
    while i >= 0:
        if shape[i] == 1:
            i -= 1
            continue
        if strides[i] != block:
            break
        block *= shape[i]
        i -= 1
    count = 1
    for j in range(i + 1):
        count *= shape[j]
    return count, block


def span_list(tensor) -> tuple[list[int], list[int]]:
    """(ptrs, sizes) of ``tensor``'s contiguous spans, in its own dim order.

    This is the copy-free alternative to staging: the pool hands the transport
    one descriptor per span instead of one pointer into a staged copy, and
    mooncake's ``batch_put_from_multi_buffers`` concatenates them into the
    object. Reference implementation - a production version would build the
    outer-index walk once per grid rather than per chunk.
    """
    shape, strides = list(tensor.shape), list(tensor.stride())
    count, block = descriptor_count(shape, strides)
    split = len(shape)
    acc = 1
    while split > 0:
        d = split - 1
        if shape[d] != 1 and strides[d] != acc:
            break
        acc *= shape[d]
        split -= 1
    offsets = [0]
    for dim in range(split):
        offsets = [
            off + idx * strides[dim] for off in offsets for idx in range(shape[dim])
        ]
    assert len(offsets) == count
    itemsize = tensor.element_size()
    base = tensor.data_ptr()
    return [base + o * itemsize for o in offsets], [block * itemsize] * count


def chunk_view(pool, order, index, l0, l1, h0, h1):
    """One chunk of one page, as a tensor whose dim order IS the byte order."""
    if order == HEAD_MAJOR:
        return pool.page_view_head_major(index)[:, h0:h1, l0:l1]
    return pool._page_kv_view_unified(index)[:, l0:l1, :, h0:h1]


def chunk_descriptors(pool, layer_ranges, head_ranges) -> tuple[int, int]:
    """(descriptors per page, min elements per descriptor) for one page's chunks
    under the shipped (layer-major) order."""
    total, smallest = 0, None
    if head_ranges is None:
        view = pool._page_view_unified(0)
        for l0, l1 in layer_ranges:
            n, block = descriptor_count(*_shape_stride(view[l0:l1]))
            total += n
            smallest = block if smallest is None else min(smallest, block)
        return total, smallest
    for l0, l1 in layer_ranges:
        for h0, h1 in head_ranges:
            for kv in range(2):
                sub = chunk_view(pool, LAYER_MAJOR, 0, l0, l1, h0, h1)[kv]
                n, block = descriptor_count(*_shape_stride(sub))
                total += n
                smallest = block if smallest is None else min(smallest, block)
    return total, smallest


def _shape_stride(t):
    return list(t.shape), list(t.stride())


def staged_fraction(pool, layer_ranges, head_ranges) -> float:
    """Fraction of a page's chunk bytes that go through the staging buffer."""
    if head_ranges is None:
        slabs = pool._slab_schedule(layer_ranges)
        total = sum(s[2] for s in slabs)
        staged = sum(s[2] for s in slabs if not s[3])
    else:
        slabs = pool._slab_schedule(layer_ranges, head_ranges)
        total = sum(s[5] for s in slabs)
        staged = sum(s[5] for s in slabs if not s[6])
    return staged / total


class _NullStore:
    """Transport that does nothing: isolates conversion + adapter framing."""

    def batch_is_exist(self, keys):
        return [0] * len(keys)

    def batch_put_from(self, keys, ptrs, sizes, config=None):
        return [0] * len(keys)

    def batch_get_into(self, keys, ptrs, sizes):
        return list(sizes)


class _LoopbackStore(_NullStore):
    """Keyed byte store - one object per chunk key, the real object model."""

    def __init__(self):
        self.obj = {}

    def batch_is_exist(self, keys):
        return [1 if k in self.obj else 0 for k in keys]

    def batch_put_from(self, keys, ptrs, sizes, config=None):
        for k, ptr, n in zip(keys, ptrs, sizes):
            self.obj[k] = ctypes.string_at(ptr, n)
        return [0] * len(keys)

    def batch_get_into(self, keys, ptrs, sizes):
        out = []
        for k, ptr, n in zip(keys, ptrs, sizes):
            blob = self.obj.get(k)
            if blob is None or len(blob) != n:
                out.append(-1)
                continue
            ctypes.memmove(ptr, blob, n)
            out.append(n)
        return out


def make_store(pool, layer_ranges, head_ranges, backing, *, staging_mb=256):
    """A real MooncakeStore wired to a fake transport.

    Drives the production `_batch_set_adapter` / `_batch_get_adapter` loop - key
    fan-out, staging sub-batching, the exists filter, result post-processing -
    so the save/load numbers cover the adapter, not just the memcpy.
    """
    from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

    suffixes = (
        [f"L{a}-{b}" for a, b in layer_ranges]
        if head_ranges is None
        else [f"L{a}-{b}_H{c}-{d}" for a, b in layer_ranges for c, d in head_ranges]
    )
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=head_ranges is None,
        enable_storage_metrics=False,
        is_page_first_layout=pool.layout.startswith("page_first"),
        model_name="bench",
        extra_config={"staging_buffer_mb": staging_mb},
        unified_suffix=suffixes,
        unified_layer_ranges=layer_ranges,
        unified_head_ranges=head_ranges,
    )
    store = MooncakeStore.__new__(MooncakeStore)
    store.store = backing
    store.config_prefix = None
    store.is_mla_backend = config.is_mla_model
    store.mha_suffix = store.mla_suffix = None
    store.storage_config = config
    store.should_split_heads = False
    store.mem_pool_host = pool
    store.layout_adapter = _UnpinnedAdapter(pool, config)
    store.enable_storage_metrics = False
    store.gb_per_page = 0.0
    store.prefetch_pgs, store.backup_pgs = [], []
    store.prefetch_bandwidth, store.backup_bandwidth = [], []
    store._use_group_semantics = False
    return store


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


def timed(fn: Callable[[], object], warmup: int = 1, iters: int = 3) -> float:
    """Best-of-N wall seconds. Best-of, not mean: we want the machine's
    capability, not the shared-CI-runner noise floor."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def memcpy_baseline(nbytes: int, dtype) -> float:
    """GB/s of a single flat contiguous copy_ of the same payload."""
    n = nbytes // dtype.itemsize
    src = torch.empty(n, dtype=dtype)
    dst = torch.empty_like(src)
    return nbytes / timed(lambda: dst.copy_(src)) / 1e9


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Model:
    name: str
    family: str  # "mha" | "mla"
    layers: int
    heads: int  # local kv heads (mha) - 0 for mla
    dim: int  # head_dim (mha) or kv_cache_dim (mla)
    page_size: int
    pages: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def payload_bytes(self) -> int:
        per_token = self.layers * self.dim * self.dtype.itemsize
        per_token *= 2 * self.heads if self.family == "mha" else 1
        return per_token * self.page_size * self.pages


# Small enough for the CPU pre-flight budget: ~17 MB and ~9 MB payloads.
CI_MODELS = (
    Model("gqa-small", "mha", layers=8, heads=4, dim=128, page_size=64, pages=16),
    Model("mla-small", "mla", layers=8, heads=0, dim=576, page_size=64, pages=16),
)

# Production-shaped: Llama-3-70B-ish GQA at TP2 (8 kv heads -> 4 local) and
# DeepSeek-V3-ish MLA (61 layers - prime, so the tail chunk is short).
FULL_MODELS = (
    Model("gqa-70b@tp2", "mha", layers=80, heads=4, dim=128, page_size=64, pages=64),
    Model("mla-dsv3", "mla", layers=61, heads=0, dim=576, page_size=64, pages=64),
)


def grids(model: Model) -> list[tuple[str, list, Optional[list]]]:
    """(label, layer_ranges, head_ranges) per fan-out shape a fleet grid can
    produce. ``head_ranges`` is None for rank-replicated (MLA) pools."""
    L, H = model.layers, model.heads
    half = max(L // 2, 1)
    layers_split = [(a, min(a + half, L)) for a in range(0, L, half)]
    all_heads = None if model.family == "mla" else [(0, H)]
    out = [("none", [(0, L)], all_heads), ("layer", layers_split, all_heads)]
    if model.family == "mha" and H > 1:
        heads_split = [(i, i + 1) for i in range(H)]
        out.append(("head", [(0, L)], heads_split))
        out.append(("layer+head", layers_split, heads_split))
    return out


def build_pool(model: Model, layout: str):
    if model.family == "mha":
        return _BenchMHAPool(
            layout=layout,
            layers=model.layers,
            heads=model.heads,
            head_dim=model.dim,
            page_size=model.page_size,
            pages=model.pages,
            dtype=model.dtype,
        )
    return _BenchMLAPool(
        layout=layout,
        layers=model.layers,
        kv_dim=model.dim,
        page_size=model.page_size,
        pages=model.pages,
        dtype=model.dtype,
    )


@dataclass(frozen=True)
class Row:
    model: str
    layout: str
    grid: str
    chunk_mb: float
    slab_kb: float
    staged_pct: float
    descriptors: int
    min_desc_bytes: int
    save_gbs: float
    load_gbs: float
    e2e_save_gbs: float
    e2e_load_gbs: float

    @property
    def zero_copy(self) -> bool:
        return self.staged_pct == 0.0


def measure(model: Model, layout: str, grid_label, layer_ranges, head_ranges) -> Row:
    pool = build_pool(model, layout)
    indices = torch.arange(model.pages * model.page_size)
    page_bytes = pool.unified_bytes_per_page(layer_ranges, head_ranges)
    staging = torch.empty(model.pages * page_bytes, dtype=torch.uint8)
    total_bytes = model.pages * page_bytes

    frac = staged_fraction(pool, layer_ranges, head_ranges)
    descs, min_block = chunk_descriptors(pool, layer_ranges, head_ranges)

    save = timed(
        lambda: pool.gather_unified_chunks(indices, layer_ranges, head_ranges, staging)
    )
    page_ok = [True] * model.pages
    load = timed(
        lambda: pool.scatter_unified_chunks(
            indices, layer_ranges, head_ranges, staging, page_ok
        )
    )
    # The dominant cost driver is not the layout but the per-slab copy_ size,
    # i.e. how finely the fleet grid cuts a page. Report it next to the result.
    slabs = (
        pool._slab_schedule(layer_ranges)
        if head_ranges is None
        else pool._slab_schedule(layer_ranges, head_ranges)
    )
    slab_bytes = min(s[2] if head_ranges is None else s[5] for s in slabs)

    # Same payload through the production save/load entry points against a
    # do-nothing transport: adds key fan-out, sub-batching and the exists filter.
    store = make_store(pool, layer_ranges, head_ranges, _NullStore())
    keys = [f"p{i}" for i in range(model.pages)]
    e2e_save = timed(lambda: store.batch_set_v1(keys, indices))
    e2e_load = timed(lambda: store.batch_get_v1(keys, indices))
    # Throughput is quoted over the payload the adapter is responsible for (the
    # chunk bytes), even when part of it moves for free - so a zero-copy row is
    # reporting the rate of emitting pointers, not a memory bandwidth.
    return Row(
        model=model.name,
        layout=layout,
        grid=grid_label,
        chunk_mb=total_bytes / 1e6,
        slab_kb=slab_bytes / 1e3,
        staged_pct=100.0 * frac,
        descriptors=descs * model.pages,
        min_desc_bytes=min_block * model.dtype.itemsize,
        save_gbs=total_bytes / save / 1e9,
        load_gbs=total_bytes / load / 1e9,
        e2e_save_gbs=total_bytes / e2e_save / 1e9,
        e2e_load_gbs=total_bytes / e2e_load / 1e9,
    )


def run_matrix(models, *, include_projected: bool) -> list[Row]:
    rows = []
    with quiet_gc():
        for model in models:
            layouts = list(MHA_LAYOUTS if model.family == "mha" else MLA_LAYOUTS)
            if include_projected and model.family == "mha":
                layouts.append(PROJECTED_LAYOUT)
            for layout in layouts:
                for label, layer_ranges, head_ranges in grids(model):
                    rows.append(
                        measure(model, layout, label, layer_ranges, head_ranges)
                    )
    return rows


def print_table(rows: list[Row], models) -> None:
    by_model = {m.name: m for m in models}
    print(f"torch threads: {torch.get_num_threads()}")
    for name, model in by_model.items():
        base = memcpy_baseline(min(model.payload_bytes, 1 << 28), model.dtype)
        geom = (
            f"kv_heads={model.heads} head_dim={model.dim}"
            if model.family == "mha"
            else f"kv_dim={model.dim}"
        )
        print(
            f"\n=== {name}: layers={model.layers} {geom} page={model.page_size} "
            f"pages={model.pages}   flat memcpy baseline {base:.0f} GB/s"
        )
        print(
            f"{'host layout':<24}{'fan-out':<11}{'MB':>7}{'slab KB':>9}{'stg%':>6}"
            f"{'descs':>9}{'B/desc':>9}"
            f"{'gather':>9}{'scatter':>10}{'save e2e':>10}{'load e2e':>10}"
        )
        for r in rows:
            if r.model != name:
                continue
            mark = "*" if r.zero_copy else " "
            print(
                f"{r.layout:<24}{r.grid:<11}{r.chunk_mb:7.1f}{r.slab_kb:9.1f}"
                f"{r.staged_pct:6.0f}{r.descriptors:9d}{r.min_desc_bytes:9d}"
                f"{r.save_gbs:8.1f}{mark}{r.load_gbs:9.1f}{mark}"
                f"{r.e2e_save_gbs:9.1f}{mark}{r.e2e_load_gbs:9.1f}{mark}"
            )
    print(
        "\nGB/s columns: gather/scatter = the conversion alone; save/load e2e ="
        "\n  batch_set_v1 / batch_get_v1 against a do-nothing transport, i.e. the"
        "\n  conversion plus key fan-out, sub-batching and the exists filter."
        "\n* no bytes copied - transfers resolve to pool addresses in place."
    )
    print(
        "  slab KB is the smallest per-slab copy_ - the dominant cost driver:"
        "\n  conversion throughput tracks slab size, NOT the host layout."
    )
    print(
        "  descs/B-desc = contiguous spans per batch in unified order: what a\n"
        "  scatter/gather put (batch_put_from_multi_buffers) would need, copying\n"
        "  nothing. Small spans make that trade a loss - see the order matrix."
    )


def order_matrix(model: Model) -> list[tuple]:
    """(order, layout, fan-out) -> descriptors per page, bytes per span.

    This is the design table: the byte order, not the host layout, decides
    whether any host pool can serve a chunk without a copy.
    """
    out = []
    layouts = list(MHA_LAYOUTS) + [PROJECTED_LAYOUT]
    for order in (HEAD_MAJOR, LAYER_MAJOR):
        for layout in layouts:
            pool = build_pool(model, layout)
            for label, layer_ranges, head_ranges in grids(model):
                worst, smallest = 0, None
                for l0, l1 in layer_ranges:
                    for h0, h1 in head_ranges:
                        sub = chunk_view(pool, order, 0, l0, l1, h0, h1)[0]
                        n, block = descriptor_count(*_shape_stride(sub))
                        worst = max(worst, n)
                        smallest = block if smallest is None else min(smallest, block)
                out.append(
                    (order, layout, label, worst, smallest * model.dtype.itemsize)
                )
    return out


def print_order_matrix(model: Model, entries=None) -> None:
    print(
        f"\n=== unified byte order matrix - {model.name} "
        f"(spans x bytes/span for the WORST chunk of each grid)"
    )
    entries = order_matrix(model) if entries is None else entries
    labels = [g[0] for g in grids(model)]
    for order in (HEAD_MAJOR, LAYER_MAJOR):
        tag = (
            "(head, layer, token, dim)"
            if order == HEAD_MAJOR
            else "(layer, token, head, dim)"
        )
        print(f"\n  order {tag}{'  <- shipped' if order == HEAD_MAJOR else ''}")
        print(f"  {'host layout':<24}" + "".join(f"{l:>22}" for l in labels))
        for layout in list(MHA_LAYOUTS) + [PROJECTED_LAYOUT]:
            cells = [
                f"{n} x {b}B"
                for o, la, g, n, b in entries
                if o == order and la == layout
            ]
            print(f"  {layout:<24}" + "".join(f"{c:>22}" for c in cells))


@dataclass(frozen=True)
class SplitRow:
    head_group: int
    layer_grid: str
    chunk_kb: float
    slab_kb: float
    span_bytes: int
    staged_pct: float
    gather_gbs: float
    scatter_gbs: float


def head_split_sweep(model: Model, layout: str, order: str = LAYER_MAJOR):
    """Cost of splitting the kv-head axis, as a function of ``head_group``.

    Cross-TP reuse is the only thing that forces a head split, and a head split
    is the only thing that forces a copy under the layer-major order. This
    sweeps head_group from "no split" (= all local kv heads, one chunk) down to
    1 (finest grid, TP-size == kv-head count) so the price of each step is
    visible, with and without a layer split on top.
    """
    L, H = model.layers, model.heads
    pool = build_pool(model, layout)
    indices = torch.arange(model.pages * model.page_size)
    out = []
    unit = 8 if L > 8 else max(L // 2, 1)
    for grid_label, layer_ranges in (
        ("none", [(0, L)]),
        (f"lg={unit}", [(a, min(a + unit, L)) for a in range(0, L, unit)]),
    ):
        hg = H
        while hg >= 1:
            head_ranges = [(i, i + hg) for i in range(0, H, hg)]
            slabs = []
            spans = None
            for l0, l1 in layer_ranges:
                for h0, h1 in head_ranges:
                    sub = chunk_view(pool, order, 0, l0, l1, h0, h1)[0]
                    n, block = descriptor_count(*_shape_stride(sub))
                    slabs.append((n, block * model.dtype.itemsize))
                    spans = n if spans is None else max(spans, n)
            direct = all(n == 1 for n, _ in slabs)
            chunk_bytes = (
                (layer_ranges[0][1] - layer_ranges[0][0])
                * hg
                * model.page_size
                * model.dim
                * model.dtype.itemsize
            )
            total = model.pages * len(layer_ranges) * len(head_ranges) * 2 * chunk_bytes
            staging = torch.empty(total, dtype=torch.uint8)

            def convert(write: bool):
                cursor = 0
                for index in indices.tolist()[:: model.page_size]:
                    for l0, l1 in layer_ranges:
                        for h0, h1 in head_ranges:
                            for kv in range(2):
                                view = chunk_view(pool, order, index, l0, l1, h0, h1)
                                sub = view[kv]
                                if not direct:
                                    buf = (
                                        staging[cursor : cursor + chunk_bytes]
                                        .view(model.dtype)
                                        .view(sub.shape)
                                    )
                                    if write:
                                        buf.copy_(sub)
                                    else:
                                        sub.copy_(buf)
                                cursor += chunk_bytes

            g = timed(lambda: convert(True))
            s = timed(lambda: convert(False))
            out.append(
                SplitRow(
                    head_group=hg,
                    layer_grid=grid_label,
                    chunk_kb=chunk_bytes / 1e3,
                    slab_kb=chunk_bytes / 1e3,
                    span_bytes=min(b for _, b in slabs),
                    staged_pct=0.0 if direct else 100.0,
                    gather_gbs=total / g / 1e9,
                    scatter_gbs=total / s / 1e9,
                )
            )
            if hg == 1:
                break
            hg //= 2
    return out


def print_head_split_sweep(model: Model, layout: str, rows=None) -> None:
    """Report the head-split sweep, including what the conversion does to
    end-to-end save throughput once a fabric is in series with it."""
    rows = head_split_sweep(model, layout) if rows is None else rows
    print(
        f"\n=== head-split penalty - {model.name} on {layout}, "
        f"order (layer, token, head, dim)"
    )
    print(
        f"  {'layer grid':<10}{'head_group':>12}{'chunk KB':>10}{'B/span':>9}"
        f"{'staged%':>9}{'gather GB/s':>13}{'scatter GB/s':>14}"
        + "".join(f"{'e2e@' + str(n) + 'GB/s':>13}" for n in NET_GBS)
    )
    for r in rows:
        tag = (
            f"{r.head_group} (no split)"
            if r.head_group == model.heads
            else str(r.head_group)
        )
        e2e = ""
        for net in NET_GBS:
            # conversion is serial with the transfer: 1/(1/conv + 1/net)
            eff = net if r.staged_pct == 0.0 else 1.0 / (1.0 / r.gather_gbs + 1.0 / net)
            e2e += f"{eff:12.1f}{'*' if r.staged_pct == 0.0 else ' '}"
        print(
            f"  {r.layer_grid:<10}{tag:>12}{r.chunk_kb:10.0f}{r.span_bytes:9d}"
            f"{r.staged_pct:9.1f}{r.gather_gbs:13.1f}{r.scatter_gbs:14.1f}{e2e}"
        )
    print(
        "\n  * no split = nothing is copied, so the fabric is the only limit.\n"
        "  e2e@N = save throughput with an N GB/s fabric in series with the\n"
        "  conversion, 1/(1/gather + 1/N). Splitting heads is what forces the\n"
        "  copy; splitting layers on top multiplies the slab count and costs\n"
        "  again, even though the span size is unchanged."
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestUnifiedLayoutSaveLoadPerf(CustomTestCase):
    """Prices the adapter's save/load conversion and pins the structural facts
    the price depends on. No wall-clock thresholds beyond a catastrophic-
    regression floor - CI runners are shared."""

    @classmethod
    def setUpClass(cls):
        cls.mha_model = next(m for m in CI_MODELS if m.family == "mha")
        cls.rows = run_matrix(CI_MODELS, include_projected=False)
        cls.orders = order_matrix(cls.mha_model)
        cls.split = head_split_sweep(cls.mha_model, "page_first_direct")

    def test_report(self):
        print()
        print_table(self.rows, CI_MODELS)
        print_order_matrix(self.mha_model, self.orders)
        print_head_split_sweep(self.mha_model, "page_first_direct", self.split)

    def test_head_split_is_the_sole_trigger_for_the_copy(self):
        """Under the layer-major order on page_first_direct, no-split chunks
        are zero-copy with or without a layer split, and every split head
        group stages 100 % - so cross-TP, not cross-PP, is what costs."""
        for row in self.split:
            if row.head_group == self.mha_model.heads:
                self.assertEqual(row.staged_pct, 0.0, row.layer_grid)
            else:
                self.assertEqual(row.staged_pct, 100.0, row.layer_grid)
                self.assertEqual(
                    row.span_bytes,
                    row.head_group * self.mha_model.dim * 2,
                    f"{row.layer_grid}/hg={row.head_group}",
                )

    def test_head_major_order_would_fragment_every_host_layout(self):
        """Why the chunk order was flipped: under (head, layer, token, dim) all
        four shipped MHA layouts fragment a chunk into head_dim-sized pieces, so
        no restriction of the layout set could have removed the copy."""
        by_key = {(o, la, g): (n, b) for o, la, g, n, b in self.orders}
        for layout in MHA_LAYOUTS:
            for label, _, _ in grids(self.mha_model):
                spans, span_bytes = by_key[(HEAD_MAJOR, layout, label)]
                self.assertGreater(spans, 1, f"{layout}/{label}")
                self.assertEqual(
                    span_bytes, self.mha_model.dim * 2, f"{layout}/{label}"
                )

    def test_shipped_order_is_zero_copy_on_page_first_direct(self):
        """What the adapter now does: serializing a chunk as (layer, token,
        head, dim) makes any layer-only fan-out a SINGLE span on
        page_first_direct - a layout that already ships - while head fan-out
        stays fragmented."""
        rows = {(o, la, g): (n, b) for o, la, g, n, b in self.orders}
        for grid in ("none", "layer"):
            self.assertEqual(rows[(LAYER_MAJOR, "page_first_direct", grid)][0], 1, grid)
            # ... and the shipped order is not, on the same layout.
            self.assertGreater(
                rows[(HEAD_MAJOR, "page_first_direct", grid)][0], 1, grid
            )
        for grid in ("head", "layer+head"):
            self.assertGreater(
                rows[(LAYER_MAJOR, "page_first_direct", grid)][0], 1, grid
            )

    def test_head_major_order_needs_a_host_layout_that_does_not_exist(self):
        """The alternative to flipping the order was adding
        page_head_layer_direct, which does make the head-major order zero-copy -
        but it is head-outermost, and §2 of ANALYSIS_unified_l3_zero_copy.md
        measures that layout class at 1.6x slower D2H and 4.5x slower H2D on
        every L2 hit. Pinned here so the rejected branch stays documented."""
        by_key = {(o, la, g): (n, b) for o, la, g, n, b in self.orders}
        for grid in ("none", "head"):
            self.assertEqual(by_key[(HEAD_MAJOR, PROJECTED_LAYOUT, grid)][0], 1, grid)

    def test_mla_page_first_direct_is_zero_copy(self):
        """The one shipped combination that already pays nothing, on both
        fan-out axes - MLA's unified order IS page_first_direct's order."""
        seen = 0
        for row in self.rows:
            if row.model.startswith("mla") and row.layout == "page_first_direct":
                self.assertTrue(row.zero_copy, row.grid)
                seen += 1
        self.assertGreater(seen, 0)

    def test_gather_is_layout_independent_under_load(self):
        """Every layout must produce identical unified bytes at benchmark sizes
        too - a perf change that permutes bytes is a corruption bug."""
        model = next(m for m in CI_MODELS if m.family == "mha")
        torch.manual_seed(0)
        logical = [
            torch.randn(
                2,
                model.heads,
                model.layers,
                model.page_size,
                model.dim,
                dtype=torch.bfloat16,
            )
            for _ in range(4)
        ]
        indices = torch.arange(4 * model.page_size)
        layer_ranges = [(0, model.layers // 2), (model.layers // 2, model.layers)]
        head_ranges = [(i, i + 1) for i in range(model.heads)]
        reference = None
        for layout in list(MHA_LAYOUTS) + [PROJECTED_LAYOUT]:
            pool = _BenchMHAPool(
                layout=layout,
                layers=model.layers,
                heads=model.heads,
                head_dim=model.dim,
                page_size=model.page_size,
                pages=4,
                dtype=torch.bfloat16,
            )
            pool.fill_from_logical(logical)
            staging = torch.zeros(
                4 * pool.unified_bytes_per_page(layer_ranges, head_ranges),
                dtype=torch.uint8,
            )
            ptrs, sizes = pool.gather_unified_chunks(
                indices, layer_ranges, head_ranges, staging
            )
            got = b"".join(ctypes.string_at(p, s) for p, s in zip(ptrs, sizes))
            if reference is None:
                reference = got
            else:
                self.assertEqual(got, reference, layout)

    def test_scatter_inverts_gather_at_benchmark_shapes(self):
        model = next(m for m in CI_MODELS if m.family == "mla")
        torch.manual_seed(0)
        logical = [
            torch.randn(
                model.layers, model.page_size, 1, model.dim, dtype=torch.bfloat16
            )
            for _ in range(4)
        ]
        indices = torch.arange(4 * model.page_size)
        layer_ranges = [(0, model.layers // 2), (model.layers // 2, model.layers)]
        writer = _BenchMLAPool(
            layout="layer_first",
            layers=model.layers,
            kv_dim=model.dim,
            page_size=model.page_size,
            pages=4,
            dtype=torch.bfloat16,
        )
        writer.fill_from_logical(logical)
        staging_w = torch.zeros(
            4 * writer.unified_bytes_per_page(layer_ranges), dtype=torch.uint8
        )
        src, sizes = writer.gather_unified_chunks(
            indices, layer_ranges, None, staging_w
        )
        for layout in MLA_LAYOUTS:
            reader = _BenchMLAPool(
                layout=layout,
                layers=model.layers,
                kv_dim=model.dim,
                page_size=model.page_size,
                pages=4,
                dtype=torch.bfloat16,
            )
            staging_r = torch.zeros_like(staging_w)
            dst, dst_sizes = reader.get_unified_chunk_meta(
                indices, layer_ranges, None, staging_r
            )
            self.assertEqual(sizes, dst_sizes)
            for d, s, n in zip(dst, src, sizes):
                ctypes.memmove(d, s, n)
            reader.scatter_unified_chunks(
                indices, layer_ranges, None, staging_r, [True] * 4
            )
            for p, page in enumerate(logical):
                self.assertTrue(
                    torch.equal(reader._page_view_unified(p * model.page_size), page),
                    layout,
                )

    def test_spans_reproduce_the_staged_bytes_without_copying(self):
        """The copy-free alternative, proven rather than asserted: reading each
        chunk's contiguous spans straight out of the pool yields byte-for-byte
        what the staging buffer holds, for every layout. What differs between
        layouts is only how many descriptors that takes."""
        model = next(m for m in CI_MODELS if m.family == "mha")
        torch.manual_seed(0)
        pages = 2
        logical = [
            torch.randn(
                2,
                model.heads,
                model.layers,
                model.page_size,
                model.dim,
                dtype=torch.bfloat16,
            )
            for _ in range(pages)
        ]
        indices = torch.arange(pages * model.page_size)
        layer_ranges = [(0, model.layers // 2), (model.layers // 2, model.layers)]
        head_ranges = [(0, 2), (2, model.heads)]
        for layout in list(MHA_LAYOUTS) + [PROJECTED_LAYOUT]:
            pool = _BenchMHAPool(
                layout=layout,
                layers=model.layers,
                heads=model.heads,
                head_dim=model.dim,
                page_size=model.page_size,
                pages=pages,
                dtype=torch.bfloat16,
            )
            pool.fill_from_logical(logical)
            staging = torch.zeros(
                pages * pool.unified_bytes_per_page(layer_ranges, head_ranges),
                dtype=torch.uint8,
            )
            ptrs, sizes = pool.gather_unified_chunks(
                indices, layer_ranges, head_ranges, staging
            )
            staged = [ctypes.string_at(p, s) for p, s in zip(ptrs, sizes)]
            spanned = []
            for page in range(pages):
                for l0, l1 in layer_ranges:
                    for h0, h1 in head_ranges:
                        for kv in range(2):
                            sub = chunk_view(
                                pool,
                                LAYER_MAJOR,
                                page * model.page_size,
                                l0,
                                l1,
                                h0,
                                h1,
                            )[kv]
                            sp, sz = span_list(sub)
                            spanned.append(
                                b"".join(ctypes.string_at(p, s) for p, s in zip(sp, sz))
                            )
            # assertTrue, not assertEqual: a mismatch here would otherwise
            # render a multi-kilobyte bytes diff per chunk.
            self.assertTrue(spanned == staged, f"{layout} span/stage mismatch")

    def test_save_then_load_across_layouts_through_the_store(self):
        """End-to-end save/load through the production entry points: one layout
        writes with batch_set_v1, every other layout reads the same objects back
        with batch_get_v1. This is the property the whole unified scheme exists
        for - cross-topology reuse is pure key selection - exercised on the same
        code path the benchmark times."""
        model = next(m for m in CI_MODELS if m.family == "mha")
        pages = 4
        torch.manual_seed(0)
        logical = [
            torch.randn(
                2,
                model.heads,
                model.layers,
                model.page_size,
                model.dim,
                dtype=torch.bfloat16,
            )
            for _ in range(pages)
        ]
        indices = torch.arange(pages * model.page_size)
        keys = [f"page{i}" for i in range(pages)]
        layer_ranges = [(0, model.layers // 2), (model.layers // 2, model.layers)]
        head_ranges = [(0, 2), (2, model.heads)]

        def pool_of(layout, data):
            pool = _BenchMHAPool(
                layout=layout,
                layers=model.layers,
                heads=model.heads,
                head_dim=model.dim,
                page_size=model.page_size,
                pages=pages,
                dtype=torch.bfloat16,
            )
            pool.fill_from_logical(data)
            return pool

        for writer_layout in list(MHA_LAYOUTS) + [PROJECTED_LAYOUT]:
            backing = _LoopbackStore()
            writer = pool_of(writer_layout, logical)
            wstore = make_store(writer, layer_ranges, head_ranges, backing)
            self.assertTrue(all(wstore.batch_set_v1(keys, indices)), writer_layout)

            for reader_layout in list(MHA_LAYOUTS) + [PROJECTED_LAYOUT]:
                blank = [torch.zeros_like(page) for page in logical]
                reader = pool_of(reader_layout, blank)
                rstore = make_store(reader, layer_ranges, head_ranges, backing)
                ok = rstore.batch_get_v1(keys, indices)
                self.assertTrue(
                    all(ok), f"{writer_layout} -> {reader_layout} load failed"
                )
                for i, page in enumerate(logical):
                    got = reader._page_kv_view_unified(i * model.page_size)
                    # logical is (kv, head, layer, token, dim); the unified view
                    # is (kv, layer, token, head, dim).
                    want = page.permute(0, 2, 3, 1, 4)
                    self.assertTrue(
                        torch.equal(got, want),
                        f"{writer_layout} -> {reader_layout} page {i} mismatch",
                    )

    def test_adapter_sizes_staging_from_the_grid(self):
        """The production entry point: KVCacheLayoutAdapter allocates staging
        only when the pool is not already unified-contiguous, and sizes the
        sub-batch from staging_buffer_mb."""
        model = next(m for m in CI_MODELS if m.family == "mla")
        layer_ranges = [(0, model.layers)]
        for layout, expect_staging in (
            ("page_first_direct", False),
            ("layer_first", True),
        ):
            pool = build_pool(model, layout)
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                attn_cp_rank=0,
                attn_cp_size=1,
                is_mla_model=True,
                enable_storage_metrics=False,
                is_page_first_layout=layout.startswith("page_first"),
                model_name="bench",
                extra_config={"staging_buffer_mb": 1},
                unified_suffix=[f"ns_L0-{model.layers}"],
                unified_layer_ranges=layer_ranges,
                unified_head_ranges=None,
            )
            adapter = _UnpinnedAdapter(pool, config)
            self.assertEqual(adapter.staging_set is not None, expect_staging, layout)
            if expect_staging:
                page_bytes = pool.unified_bytes_per_page(layer_ranges)
                self.assertEqual(
                    adapter.staging_pages, max((1 << 20) // page_bytes, 1), layout
                )
                keys = [f"k{i}" for i in range(model.pages)]
                indices = torch.arange(model.pages * model.page_size)
                seen = sum(len(sub) for sub, _ in adapter.sub_batches(keys, indices))
                self.assertEqual(seen, len(keys), layout)

    def test_every_case_produced_a_measurement(self):
        """Deliberately not a bandwidth threshold: CI runners are shared, and a
        loaded host was observed here two orders of magnitude below its idle
        rate. The regression guards that matter are the structural ones above
        (staged fraction, descriptor granularity, byte identity); the throughput
        columns are for reading, not gating."""
        self.assertTrue(self.rows)
        for row in self.rows:
            self.assertGreater(row.save_gbs, 0.0, f"{row.layout}/{row.grid} save")
            self.assertGreater(row.load_gbs, 0.0, f"{row.layout}/{row.grid} load")
            self.assertGreater(row.e2e_save_gbs, 0.0, f"{row.layout}/{row.grid} e2e")
            self.assertGreater(row.e2e_load_gbs, 0.0, f"{row.layout}/{row.grid} e2e")


class _UnpinnedAdapter(KVCacheLayoutAdapter):
    """CPU-only hosts cannot pin; the benchmark does not need DMA."""

    def _alloc_staging(self, numel):
        return torch.empty(numel, dtype=torch.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="unified L3 layout save/load cost")
    parser.add_argument(
        "--full",
        action="store_true",
        help="production-shaped models (~1 GB payload) instead of the CI sizes",
    )
    parser.add_argument(
        "--no-projected",
        action="store_true",
        help=f"omit the projected {PROJECTED_LAYOUT} rows",
    )
    args, _ = parser.parse_known_args()
    models = FULL_MODELS if args.full else CI_MODELS
    print_table(run_matrix(models, include_projected=not args.no_projected), models)
    for model in models:
        if model.family == "mha":
            print_order_matrix(model)
            print_head_split_sweep(model, "page_first_direct")


if __name__ == "__main__":
    # The CI runner appends its own flags, so only take the CLI path when one of
    # ours is present (same idiom as test_unified_radix_cache_bench.py).
    if "--full" in sys.argv or "--no-projected" in sys.argv:
        main()
    else:
        unittest.main()
