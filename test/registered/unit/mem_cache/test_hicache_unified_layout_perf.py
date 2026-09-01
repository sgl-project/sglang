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
deployments with different TP/PP splits share a keyspace. The adapter is a
**direct copy** in both directions:

    save:  pool --put-->  L3
    load:  L3   --get-->  pool

Every chunk is ONE contiguous byte range of host pool memory, so a put reads
straight out of the pool and a get lands straight in it. There is no staging
buffer and no CPU repack in either direction, which is why this file no longer
prices a conversion: what is left to measure is the per-object framing.

Only ``page_first_direct`` is supported. Its page block IS the unified byte
order -- (layer, token, head, dim) per K/V half, MLA (layer, token, dim). When
the fleet grid cuts the kv-head axis the block is stored head-group-major,
(head_group, layer, token, head_in_group, dim), which is the same bytes
permuted and keeps every chunk contiguous; the pfdhg transfer kernels absorb
that permutation on both device paths. Every other host layout is rejected.

Two things per case:

* **chunk KB** - the size of the smallest object the grid produces. Finer
  fleet grids do not cost a copy any more, they cost *objects*: more, smaller
  puts and gets through the transport.
* **e2e save/load** - ``batch_set_v1`` / ``batch_get_v1`` through a real
  ``MooncakeStore`` wired to a do-nothing transport, i.e. key fan-out, the
  exists filter and result post-processing, with the transport removed.

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

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.pool_host import common as pool_common
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.unified_layout import KVCacheLayoutAdapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

# Fabric rates used to project end-to-end save throughput:
# ~one 400 Gb/s NIC, and a multi-NIC node.
NET_GBS = (50, 200)

# The only host layout whose page block is the unified byte order, hence the
# only one an L3 chunk can be a direct copy of. The layout is part of the
# namespace identity, so nothing else can even name these objects.
UNIFIED_LAYOUT = "page_first_direct"
SUPPORTED_LAYOUTS = (UNIFIED_LAYOUT,)
# Layouts whose page block stores some other byte order: no chunk of them is
# contiguous, and since nothing stages any more they are refused outright.
REJECTED_LAYOUTS = ("page_first", "layer_first")


# --------------------------------------------------------------------------- #
# Pools
# --------------------------------------------------------------------------- #


def _plain_alloc(dims, dtype, device, pin_memory, allocator, **kwargs):
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
        # The adapter refuses to change the pool's byte order under live pages.
        self.slot_used = torch.zeros(self.size, dtype=torch.bool)
        with _plain_host_alloc():
            self.kv_buffer = self.init_kv_buffer()

    def fill_from_logical(self, logical, head_groups: int = 1):
        """``logical[p]`` is (2, layer, token, head, dim) for page ``p`` -- the
        NATURAL page block, independent of any head grid.

        Written into the pool head-group-major, which is what the pfdhg
        transfer kernels produce when the L3 grid cuts the kv-head axis. With
        one group this is a plain copy of page_first_direct's own order.
        """
        L, P, H, D = self.layer_num, self.page_size, self.head_num, self.head_dim
        per_group = H // head_groups
        for p, page in enumerate(logical):
            block = self.kv_buffer[:, p]
            permuted = (
                page.view(2, L, P, head_groups, per_group, D)
                .permute(0, 3, 1, 2, 4, 5)
                .contiguous()
            )
            block.copy_(permuted.view(block.shape))


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
        self.slot_used = torch.zeros(self.size, dtype=torch.bool)
        with _plain_host_alloc():
            self.kv_buffer = self.init_kv_buffer()

    def fill_from_logical(self, logical, head_groups: int = 1):
        """``logical[p]`` is (layer, token, 1, dim) for page ``p``. MLA has no
        kv-head axis to cut, so its page blocks are never permuted."""
        assert head_groups == 1
        for p, page in enumerate(logical):
            self.kv_buffer[p] = page


def tobytes(tensor) -> bytes:
    """Raw bytes of ``tensor`` in its own dim order."""
    return tensor.contiguous().flatten().view(torch.uint8).numpy().tobytes()


def reference_chunks(logical, layer_ranges, head_ranges) -> list[bytes]:
    """The logical chunk contents in ``chunk_keys`` order (page-major,
    layer-major, head-minor, K then V), by plain indexing of a naturally
    ordered ``(kv, layer, token, head, dim)`` page tensor.

    ``head_ranges is None`` is the MLA case: ``logical[p]`` is then
    ``(layer, token, 1, dim)`` with a single component.
    """
    if head_ranges is None:
        return [tobytes(page[l0:l1]) for page in logical for l0, l1 in layer_ranges]
    return [
        tobytes(page[kv, l0:l1, :, h0:h1, :])
        for page in logical
        for l0, l1 in layer_ranges
        for h0, h1 in head_ranges
        for kv in range(2)
    ]


def assert_chunk_bytes(case, got, want, label=""):
    """Compare chunk byte lists without letting unittest render a diff.

    A chunk is hundreds of kilobytes at benchmark shapes, so ``assertEqual``
    on the lists would spend minutes building a multi-megabyte repr diff on
    failure. Report the first differing chunk by index instead.
    """
    case.assertEqual(len(got), len(want), f"{label}: chunk count")
    for i, (g, w) in enumerate(zip(got, want)):
        if g == w:
            continue
        where = next(
            (j for j, (a, b) in enumerate(zip(g, w)) if a != b), min(len(g), len(w))
        )
        case.fail(
            f"{label}: chunk {i} differs at byte {where} "
            f"({len(g)} vs {len(w)} bytes)"
        )


class _NullStore:
    """Transport that does nothing: isolates the adapter framing."""

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


def default_suffixes(layer_ranges, head_ranges) -> list[str]:
    if head_ranges is None:
        return [f"L{a}-{b}" for a, b in layer_ranges]
    return [f"L{a}-{b}_H{c}-{d}" for a, b in layer_ranges for c, d in head_ranges]


def make_store(pool, layer_ranges, head_ranges, backing, *, suffixes=None):
    """A real MooncakeStore wired to a fake transport.

    Drives the production `_batch_set_adapter` / `_batch_get_adapter` loop - key
    fan-out, the exists filter, result post-processing - so the save/load
    numbers cover the adapter, not just the pointer arithmetic.

    ``suffixes`` may be given explicitly to model a DIFFERENT topology reading
    the same objects: the chunk names are global fleet coordinates, while the
    ranges are this rank's local view.
    """
    from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

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
        unified_suffix=suffixes or default_suffixes(layer_ranges, head_ranges),
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
    store.layout_adapter = KVCacheLayoutAdapter(pool, config)
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
    """GB/s of a single flat contiguous copy_ of the same payload.

    Kept as the counterfactual: this is the rate the *previous*, staged design
    paid on top of the transport for every byte. The direct path pays none of
    it, which is what the e2e columns below should show.
    """
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
    produce. ``head_ranges`` is None for rank-replicated (MLA) pools.

    Every grid must tile the pool's whole layer and head axes: chunks that left
    part of a page block unnamed would be transferred in the other byte order,
    which the layout refuses to compile.
    """
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


def logical_pages(model: Model, pages: int, seed: int = 0):
    """Reference content in NATURAL order, one tensor per page."""
    torch.manual_seed(seed)
    if model.family == "mla":
        return [
            torch.randn(model.layers, model.page_size, 1, model.dim, dtype=model.dtype)
            for _ in range(pages)
        ]
    return [
        torch.randn(
            2, model.layers, model.page_size, model.heads, model.dim, dtype=model.dtype
        )
        for _ in range(pages)
    ]


def build_pool(model: Model, layout: str = UNIFIED_LAYOUT, pages: Optional[int] = None):
    if model.family == "mha":
        return _BenchMHAPool(
            layout=layout,
            layers=model.layers,
            heads=model.heads,
            head_dim=model.dim,
            page_size=model.page_size,
            pages=pages or model.pages,
            dtype=model.dtype,
        )
    return _BenchMLAPool(
        layout=layout,
        layers=model.layers,
        kv_dim=model.dim,
        page_size=model.page_size,
        pages=pages or model.pages,
        dtype=model.dtype,
    )


@dataclass(frozen=True)
class Row:
    model: str
    layout: str
    grid: str
    batch_mb: float
    chunk_kb: float
    chunks: int
    meta_us: float
    e2e_save_gbs: float
    e2e_load_gbs: float


def measure(model: Model, layout: str, grid_label, layer_ranges, head_ranges) -> Row:
    pool = build_pool(model, layout)
    indices = torch.arange(model.pages * model.page_size)

    ptrs, sizes = pool.get_unified_chunk_meta(indices, layer_ranges, head_ranges)
    total_bytes = sum(sizes)
    # There is no conversion left to time; what the grid still costs is the
    # per-chunk framing, so time the pointer emission on its own.
    meta = timed(
        lambda: pool.get_unified_chunk_meta(indices, layer_ranges, head_ranges)
    )

    # Same payload through the production save/load entry points against a
    # do-nothing transport: adds key fan-out and the exists filter.
    store = make_store(pool, layer_ranges, head_ranges, _NullStore())
    keys = [f"p{i}" for i in range(model.pages)]
    e2e_save = timed(lambda: store.batch_set_v1(keys, indices))
    e2e_load = timed(lambda: store.batch_get_v1(keys, indices))
    return Row(
        model=model.name,
        layout=layout,
        grid=grid_label,
        batch_mb=total_bytes / 1e6,
        chunk_kb=min(sizes) / 1e3,
        chunks=len(sizes),
        meta_us=meta * 1e6,
        e2e_save_gbs=total_bytes / e2e_save / 1e9,
        e2e_load_gbs=total_bytes / e2e_load / 1e9,
    )


def run_matrix(models) -> list[Row]:
    rows = []
    with quiet_gc():
        for model in models:
            for layout in SUPPORTED_LAYOUTS:
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
            f"{'host layout':<24}{'fan-out':<11}{'MB':>7}{'chunk KB':>10}"
            f"{'chunks':>8}{'meta us':>10}{'save e2e':>10}{'load e2e':>10}"
        )
        for r in rows:
            if r.model != name:
                continue
            print(
                f"{r.layout:<24}{r.grid:<11}{r.batch_mb:7.1f}{r.chunk_kb:10.1f}"
                f"{r.chunks:8d}{r.meta_us:10.0f}"
                f"{r.e2e_save_gbs:10.1f}{r.e2e_load_gbs:10.1f}"
            )
    print(
        "\nEvery chunk is one contiguous range of pool memory, so nothing is"
        "\n  copied in either direction: the flat memcpy baseline above is what"
        "\n  the previous staged design paid per byte ON TOP of the transport."
        "\n  meta us = emitting the batch's (ptr, size) list; save/load e2e ="
        "\n  batch_set_v1 / batch_get_v1 against a do-nothing transport, i.e."
        "\n  the framing plus key fan-out and the exists filter."
        "\n  chunk KB is the smallest object the grid produces - a finer fleet"
        "\n  grid now costs OBJECTS, not bytes copied."
    )


@dataclass(frozen=True)
class SplitRow:
    head_group: int
    layer_grid: str
    chunk_kb: float
    chunks: int
    meta_us: float
    save_gbs: float
    load_gbs: float


def head_split_sweep(model: Model, layout: str = UNIFIED_LAYOUT):
    """Cost of splitting the kv-head axis, as a function of ``head_group``.

    Cross-TP reuse is the only thing that forces a head split. It used to force
    a CPU repack as well; it no longer does -- the page block is simply stored
    head-group-major. What is left is the object count, so this sweeps
    head_group from "no split" (all local kv heads, one chunk) down to 1
    (finest grid, TP-size == kv-head count), with and without a layer split on
    top, and prices the production save/load path at each step.
    """
    L, H = model.layers, model.heads
    indices = torch.arange(model.pages * model.page_size)
    keys = [f"p{i}" for i in range(model.pages)]
    out = []
    unit = 8 if L > 8 else max(L // 2, 1)
    for grid_label, layer_ranges in (
        ("none", [(0, L)]),
        (f"lg={unit}", [(a, min(a + unit, L)) for a in range(0, L, unit)]),
    ):
        hg = H
        while hg >= 1:
            head_ranges = [(i, i + hg) for i in range(0, H, hg)]
            # A fresh pool per row: the page-block byte order is a property of
            # the grid, and a pool may not switch order under live pages.
            pool = build_pool(model, layout)
            _, sizes = pool.get_unified_chunk_meta(indices, layer_ranges, head_ranges)
            total = sum(sizes)
            meta = timed(
                lambda: pool.get_unified_chunk_meta(indices, layer_ranges, head_ranges)
            )
            store = make_store(pool, layer_ranges, head_ranges, _NullStore())
            save = timed(lambda: store.batch_set_v1(keys, indices))
            load = timed(lambda: store.batch_get_v1(keys, indices))
            out.append(
                SplitRow(
                    head_group=hg,
                    layer_grid=grid_label,
                    chunk_kb=min(sizes) / 1e3,
                    chunks=len(sizes),
                    meta_us=meta * 1e6,
                    save_gbs=total / save / 1e9,
                    load_gbs=total / load / 1e9,
                )
            )
            if hg == 1:
                break
            hg //= 2
    return out


def print_head_split_sweep(model: Model, layout: str, rows=None) -> None:
    """Report the head-split sweep, including what the framing does to
    end-to-end save throughput once a fabric is in series with it."""
    rows = head_split_sweep(model, layout) if rows is None else rows
    print(
        f"\n=== head-split cost - {model.name} on {layout}, "
        f"page blocks head-group-major"
    )
    print(
        f"  {'layer grid':<10}{'head_group':>12}{'chunk KB':>10}{'chunks':>9}"
        f"{'meta us':>10}{'save GB/s':>12}{'load GB/s':>12}"
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
            # The framing is serial with the transfer: 1/(1/framing + 1/net).
            eff = 1.0 / (1.0 / r.save_gbs + 1.0 / net)
            e2e += f"{eff:12.1f} "
        print(
            f"  {r.layer_grid:<10}{tag:>12}{r.chunk_kb:10.0f}{r.chunks:9d}"
            f"{r.meta_us:10.0f}{r.save_gbs:12.1f}{r.load_gbs:12.1f}{e2e}"
        )
    print(
        "\n  Nothing is copied at any head_group: splitting heads multiplies the"
        "\n  OBJECT count, not the byte traffic. e2e@N = save throughput with an"
        "\n  N GB/s fabric in series with the adapter framing, 1/(1/save + 1/N)."
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestUnifiedLayoutSaveLoadPerf(CustomTestCase):
    """Prices the adapter's save/load path and pins the structural facts the
    price depends on. No wall-clock thresholds beyond a catastrophic-
    regression floor - CI runners are shared."""

    @classmethod
    def setUpClass(cls):
        cls.mha_model = next(m for m in CI_MODELS if m.family == "mha")
        cls.mla_model = next(m for m in CI_MODELS if m.family == "mla")
        cls.rows = run_matrix(CI_MODELS)
        cls.split = head_split_sweep(cls.mha_model)

    def test_report(self):
        print()
        print_table(self.rows, CI_MODELS)
        print_head_split_sweep(self.mha_model, UNIFIED_LAYOUT, self.split)

    def test_every_chunk_is_one_contiguous_range_inside_the_pool(self):
        """The load path reaches the GPU without any extra host copy: every
        read target is an address *inside the pool*, so the transport writes L3
        bytes straight where the H2D kernel will read them. Cutting the kv-head
        axis no longer changes that - the page block is stored head-group-major
        instead, so the chunk stays a single range."""
        for model in CI_MODELS:
            for label, layer_ranges, head_ranges in grids(model):
                pool = build_pool(model)
                indices = torch.arange(model.pages * model.page_size)
                base = pool.kv_buffer.data_ptr()
                span = pool.kv_buffer.numel() * pool.kv_buffer.element_size()
                ptrs, sizes = pool.get_unified_chunk_meta(
                    indices, layer_ranges, head_ranges
                )
                components = 1 if model.family == "mla" else 2
                self.assertEqual(
                    len(ptrs),
                    model.pages
                    * len(layer_ranges)
                    * len(head_ranges or [(0, 1)])
                    * components,
                    f"{model.name}/{label}",
                )
                # Disjoint, in-pool, and together they tile the pool exactly:
                # no chunk overlaps another and no byte is left unnamed.
                cursor = base
                for ptr, size in sorted(zip(ptrs, sizes)):
                    self.assertGreaterEqual(ptr, cursor, f"{model.name}/{label}")
                    self.assertLessEqual(
                        ptr + size, base + span, f"{model.name}/{label}"
                    )
                    cursor = ptr + size
                self.assertEqual(cursor, base + span, f"{model.name}/{label}")

    def test_chunk_size_matches_the_documented_formula(self):
        """A chunk is (layer window) x page_size x (heads per group) x dim."""
        for model in CI_MODELS:
            for label, layer_ranges, head_ranges in grids(model):
                pool = build_pool(model)
                indices = torch.arange(model.pages * model.page_size)
                _, sizes = pool.get_unified_chunk_meta(
                    indices, layer_ranges, head_ranges
                )
                heads_per_group = (
                    1 if head_ranges is None else head_ranges[0][1] - head_ranges[0][0]
                )
                want = [
                    (l1 - l0)
                    * model.page_size
                    * heads_per_group
                    * model.dim
                    * model.dtype.itemsize
                    for _ in range(model.pages)
                    for l0, l1 in layer_ranges
                    for _ in head_ranges or [(0, 1)]
                    for _ in range(1 if model.family == "mla" else 2)
                ]
                self.assertEqual(sizes, want, f"{model.name}/{label}")
                self.assertEqual(
                    pool.unified_bytes_per_page(layer_ranges, head_ranges)
                    * model.pages,
                    sum(sizes),
                    f"{model.name}/{label}",
                )

    def test_only_page_first_direct_can_serve_a_chunk(self):
        """page_first used to be supported (staged). It is not: nothing stages,
        and no chunk of a (token, layer, head, dim) block is contiguous."""
        for model in CI_MODELS:
            layer_ranges = [(0, model.layers)]
            head_ranges = None if model.family == "mla" else [(0, model.heads)]
            indices = torch.arange(model.page_size)
            for layout in REJECTED_LAYOUTS:
                pool = build_pool(model, layout)
                with self.assertRaisesRegex(ValueError, "page_first_direct") as ctx:
                    pool.get_unified_chunk_meta(indices, layer_ranges, head_ranges)
                self.assertIn(layout, str(ctx.exception), f"{model.name}/{layout}")

    def test_chunk_bytes_are_the_logical_content_at_benchmark_shapes(self):
        """The bytes named by chunk_metas must be the logical chunk - at
        benchmark sizes too. A perf change that permutes bytes is a corruption
        bug, so this is checked against a plain-indexed reference."""
        model = self.mha_model
        pages = 4
        logical = logical_pages(model, pages)
        indices = torch.arange(pages * model.page_size)
        layer_ranges = [(0, model.layers // 2), (model.layers // 2, model.layers)]
        for head_ranges in (
            [(0, model.heads)],
            [(0, 2), (2, model.heads)],
            [(i, i + 1) for i in range(model.heads)],
        ):
            pool = build_pool(model, pages=pages)
            pool.fill_from_logical(logical, head_groups=len(head_ranges))
            ptrs, sizes = pool.get_unified_chunk_meta(
                indices, layer_ranges, head_ranges
            )
            got = [ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, sizes)]
            assert_chunk_bytes(
                self,
                got,
                reference_chunks(logical, layer_ranges, head_ranges),
                str(head_ranges),
            )

    def test_save_then_load_through_the_store(self):
        """End-to-end through the production entry points: one pool writes with
        batch_set_v1, a blank pool reads the same objects back with
        batch_get_v1 and ends up byte-identical. The get lands directly in the
        reader's pool - there is no scatter step left to run."""
        model = self.mha_model
        pages = 4
        logical = logical_pages(model, pages)
        keys = [f"page{i}" for i in range(pages)]
        indices = torch.arange(pages * model.page_size)
        layer_ranges = [(0, model.layers // 2), (model.layers // 2, model.layers)]
        for head_ranges in ([(0, model.heads)], [(0, 2), (2, model.heads)]):
            backing = _LoopbackStore()
            writer = build_pool(model, pages=pages)
            writer.fill_from_logical(logical, head_groups=len(head_ranges))
            wstore = make_store(writer, layer_ranges, head_ranges, backing)
            self.assertTrue(all(wstore.batch_set_v1(keys, indices)), str(head_ranges))
            self.assertEqual(
                len(backing.obj),
                pages * len(layer_ranges) * len(head_ranges) * 2,
            )

            reader = build_pool(model, pages=pages)
            rstore = make_store(reader, layer_ranges, head_ranges, backing)
            self.assertTrue(all(rstore.batch_get_v1(keys, indices)), str(head_ranges))
            self.assertTrue(
                torch.equal(reader.kv_buffer, writer.kv_buffer), str(head_ranges)
            )
            ptrs, sizes = reader.get_unified_chunk_meta(
                indices, layer_ranges, head_ranges
            )
            assert_chunk_bytes(
                self,
                [ctypes.string_at(p, s) for p, s in zip(ptrs, sizes)],
                reference_chunks(logical, layer_ranges, head_ranges),
                str(head_ranges),
            )

    def test_cross_tp_reuse_is_pure_key_selection(self):
        """What the whole scheme exists for. A TP-N rank (4 local kv heads,
        head_group 2) writes chunks H0/H1; a TP-2N rank holds only 2 kv heads
        and owns exactly the H1 chunk. It reads that object by NAME, with no
        repack, and its pool ends up holding the writer's upper head pair."""
        model = self.mha_model
        pages = 2
        logical = logical_pages(model, pages)
        keys = [f"page{i}" for i in range(pages)]
        layer_ranges = [(0, model.layers)]
        backing = _LoopbackStore()

        writer = build_pool(model, pages=pages)
        writer.fill_from_logical(logical, head_groups=2)
        wstore = make_store(
            writer,
            layer_ranges,
            [(0, 2), (2, model.heads)],
            backing,
            suffixes=[f"L0-{model.layers}_H0", f"L0-{model.layers}_H1"],
        )
        self.assertTrue(
            all(wstore.batch_set_v1(keys, torch.arange(pages * model.page_size)))
        )

        narrow = Model(
            name="gqa-small@2x-tp",
            family="mha",
            layers=model.layers,
            heads=2,
            dim=model.dim,
            page_size=model.page_size,
            pages=pages,
        )
        reader = build_pool(narrow, pages=pages)
        rstore = make_store(
            reader,
            layer_ranges,
            [(0, 2)],
            backing,
            suffixes=[f"L0-{model.layers}_H1"],
        )
        indices = torch.arange(pages * model.page_size)
        self.assertTrue(all(rstore.batch_get_v1(keys, indices)))
        # The reader's whole (unpermuted, single-group) pool must equal the
        # writer's upper head pair, taken by plain indexing.
        for p, page in enumerate(logical):
            self.assertTrue(
                torch.equal(reader.kv_buffer[:, p], page[:, :, :, 2:4, :].contiguous()),
                f"page {p}",
            )

    def test_mla_layer_partition_round_trip(self):
        """MLA's unified order IS page_first_direct's, on every layer grid -
        including the short trailing chunk of a prime layer count."""
        model = self.mla_model
        pages = 4
        logical = logical_pages(model, pages)
        keys = [f"page{i}" for i in range(pages)]
        indices = torch.arange(pages * model.page_size)
        layer_ranges = [(0, model.layers // 2), (model.layers // 2, model.layers)]
        backing = _LoopbackStore()
        writer = build_pool(model, pages=pages)
        writer.fill_from_logical(logical)
        wstore = make_store(writer, layer_ranges, None, backing)
        self.assertTrue(all(wstore.batch_set_v1(keys, indices)))

        reader = build_pool(model, pages=pages)
        rstore = make_store(reader, layer_ranges, None, backing)
        self.assertTrue(all(rstore.batch_get_v1(keys, indices)))
        for p, page in enumerate(logical):
            self.assertTrue(
                torch.equal(reader._unified_page_view(p * model.page_size)[0], page),
                f"page {p}",
            )

    def test_adapter_owns_no_buffers(self):
        """The production entry point: KVCacheLayoutAdapter allocates nothing
        and registers nothing, for any grid - the pool is the only buffer."""
        for model in CI_MODELS:
            layer_ranges = [(0, model.layers)]
            head_ranges = None if model.family == "mla" else [(0, 2), (2, model.heads)]
            pool = build_pool(model)
            registered = []
            config = HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                attn_cp_rank=0,
                attn_cp_size=1,
                is_mla_model=model.family == "mla",
                enable_storage_metrics=False,
                is_page_first_layout=True,
                model_name="bench",
                unified_suffix=default_suffixes(layer_ranges, head_ranges),
                unified_layer_ranges=layer_ranges,
                unified_head_ranges=head_ranges,
            )
            adapter = KVCacheLayoutAdapter(
                pool, config, register_buffer=registered.append
            )
            self.assertEqual(registered, [], model.name)
            for gone in ("staging_set", "staging_get", "staging_pages", "gather"):
                self.assertFalse(hasattr(adapter, gone), f"{model.name}/{gone}")
            # The declared head-group count is what selects the pfdhg transfer
            # kernels; 1 means page_first_direct's natural order.
            self.assertEqual(
                pool.unified_head_groups,
                1 if head_ranges is None else len(head_ranges),
                model.name,
            )
            base = pool.kv_buffer.data_ptr()
            span = pool.kv_buffer.numel() * pool.kv_buffer.element_size()
            ptrs, sizes = adapter.chunk_metas(
                torch.arange(model.pages * model.page_size)
            )
            self.assertEqual(len(ptrs), model.pages * adapter.keys_per_page)
            for ptr, size in zip(ptrs, sizes):
                self.assertTrue(base <= ptr and ptr + size <= base + span, model.name)

    def test_every_case_produced_a_measurement(self):
        """Deliberately not a bandwidth threshold: CI runners are shared, and a
        loaded host was observed here two orders of magnitude below its idle
        rate. The regression guards that matter are the structural ones above
        (contiguity, chunk sizing, byte identity); the throughput columns are
        for reading, not gating."""
        self.assertTrue(self.rows)
        for row in self.rows:
            self.assertGreater(row.chunks, 0, f"{row.layout}/{row.grid}")
            self.assertGreater(row.meta_us, 0.0, f"{row.layout}/{row.grid} meta")
            self.assertGreater(row.e2e_save_gbs, 0.0, f"{row.layout}/{row.grid} e2e")
            self.assertGreater(row.e2e_load_gbs, 0.0, f"{row.layout}/{row.grid} e2e")
        self.assertTrue(self.split)
        for row in self.split:
            self.assertGreater(row.save_gbs, 0.0, f"hg={row.head_group}")
            self.assertGreater(row.load_gbs, 0.0, f"hg={row.head_group}")
        # Finer head grids cost objects, not bytes: the chunk count doubles
        # each time head_group halves, and the payload never changes.
        by_grid = {}
        for row in self.split:
            by_grid.setdefault(row.layer_grid, []).append(row)
        for label, rows in by_grid.items():
            for prev, cur in zip(rows, rows[1:]):
                self.assertEqual(cur.chunks, 2 * prev.chunks, label)
                self.assertAlmostEqual(cur.chunk_kb, prev.chunk_kb / 2, places=6)


def main() -> None:
    parser = argparse.ArgumentParser(description="unified L3 layout save/load cost")
    parser.add_argument(
        "--full",
        action="store_true",
        help="production-shaped models (~1 GB payload) instead of the CI sizes",
    )
    args, _ = parser.parse_known_args()
    models = FULL_MODELS if args.full else CI_MODELS
    print_table(run_matrix(models), models)
    for model in models:
        if model.family == "mha":
            print_head_split_sweep(model, UNIFIED_LAYOUT)


if __name__ == "__main__":
    # The CI runner appends its own flags, so only take the CLI path when one of
    # ours is present (same idiom as test_unified_radix_cache_bench.py).
    if "--full" in sys.argv or "--no-projected" in sys.argv:
        main()
    else:
        unittest.main()
