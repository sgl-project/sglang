"""Unit coverage for the radixshmem L3 backend against an in-memory fake."""

import enum
import hashlib
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 4
TOKEN_BYTES = 8
PAGE_BYTES = PAGE_SIZE * TOKEN_BYTES
NUM_SLOTS = 64


class _Kind(enum.Enum):
    FULL = 0
    SWA = 1
    MAMBA = 2


class _InsertError(enum.Enum):
    OK = 0
    FULL_PATH_MISSING = 7


class _GeometryMismatch(RuntimeError):
    pass


class _FakeGeometry:
    def __init__(self, block_size, full_slot_bytes, **components):
        self.key = (block_size, full_slot_bytes)


class _FakeStore:
    def __init__(self, num_slots, slot_bytes):
        self.slot_bytes = slot_bytes
        self.buf = bytearray(num_slots * slot_bytes)

    def pool(self, kind):
        return SimpleNamespace(slot_bytes=self.slot_bytes)

    def data_view(self):
        return memoryview(self.buf)

    def slot_offset(self, slot, kind):
        return slot * self.slot_bytes


class _FakeIndex:
    """Radix tree keyed by root-anchored uint64 chains; one slot per page."""

    def __init__(self, slot_bytes):
        self.store = _FakeStore(NUM_SLOTS, slot_bytes)
        self.free = list(range(NUM_SLOTS))
        self.paths = {}

    def _hit(self, chain):
        hit = 0
        for i in range(len(chain)):
            if tuple(chain[: i + 1].tolist()) not in self.paths:
                break
            hit += 1
        return hit

    def query(self, chain, mask, local_only, lock, update_meta):
        hit = self._hit(chain)
        slots = np.array(
            [self.paths[tuple(chain[: i + 1].tolist())] for i in range(hit)],
            dtype=np.int32,
        )
        return SimpleNamespace(
            status=0,
            common_hit=hit,
            full_slots=slots,
            full_fragments=[(0, 0, slots)],
            finalize=lambda: None,
        )

    def allocate_slots(self, n, kind):
        taken = self.free[:n]
        del self.free[:n]
        return np.array(taken, dtype=np.int32)

    def recycle_slots(self, slots, kind):
        self.free.extend(int(s) for s in slots)

    def insert(self, chain, slots, start, auto_recycle, component):
        if self._hit(chain[:start]) < start:
            self.recycle_slots(slots, component)
            return SimpleNamespace(error=_InsertError.OK, unused_slots=list(slots))
        unused = []
        for i in range(start, len(chain)):
            key = tuple(chain[: i + 1].tolist())
            if key in self.paths:
                unused.append(slots[i - start])
            else:
                self.paths[key] = int(slots[i - start])
        self.recycle_slots(unused, component)
        return SimpleNamespace(error=_InsertError.OK, unused_slots=unused)

    def is_distributed(self):
        return False

    def flush(self):
        pass

    def reset(self):
        self.paths.clear()
        self.free = list(range(NUM_SLOTS))


class _FakeServer:
    """Waits for a geometry; the first client's configures it."""

    def __init__(self):
        self.geometry = None
        self.index = None

    def configure(self, geometry):
        if self.geometry is None:
            self.geometry = geometry.key
            self.index = _FakeIndex(geometry.key[1])
        elif self.geometry != geometry.key:
            raise _GeometryMismatch(f"{self.geometry} != {geometry.key}")


class _ReadOnlyIndex:
    """What a local_read_only client may do: query only."""

    def __init__(self, index):
        self.query = index.query

    def __getattr__(self, name):
        raise RuntimeError(f"local_read_only client: {name} not allowed")


class _FakeClient:
    servers = {}

    def __init__(
        self,
        name,
        geometry=None,
        *,
        endpoint=None,
        max_outstanding=256,
        local_read_only=False,
    ):
        self.server = _FakeClient.servers.setdefault(name, _FakeServer())
        self.local_read_only = local_read_only
        if geometry is not None:
            self.configure(geometry)

    def configure(self, geometry):
        self.server.configure(geometry)

    def wait_ready(self, timeout_s):
        block_size, slot_bytes = self.server.geometry
        return SimpleNamespace(
            mode="ready",
            data_plane=True,
            geometry={
                "block_size": block_size,
                "pools": {"full": {"slot_bytes": slot_bytes, "num_slots": NUM_SLOTS}},
            },
        )

    @property
    def index(self):
        if self.local_read_only:
            return _ReadOnlyIndex(self.server.index)
        return self.server.index

    @property
    def store(self):
        return self.server.index.store

    def is_distributed(self):
        return False

    def pull_async(self, chain, mask, lock=True, timeout_ms=0, block=True):
        if self.local_read_only:
            raise RuntimeError("local_read_only client: pull_async not allowed")
        result = self.index.query(chain, mask, False, lock, True)
        return SimpleNamespace(wait=lambda timeout=None: result, cancel=lambda: None)

    def close(self):
        pass


_FAKE_RADIXSHMEM = SimpleNamespace(
    RadixClient=_FakeClient,
    Geometry=_FakeGeometry,
    GeometryMismatch=_GeometryMismatch,
    ComponentType=_Kind,
    COMPONENT_MASK_FULL=1,
    COMPONENT_MASK_SWA=2,
    COMPONENT_MASK_MAMBA=4,
    InsertError=_InsertError,
)


class _FakeHostPool:
    """Page-first staging tensor: one contiguous segment per page."""

    def __init__(self, num_pages, token_bytes=TOKEN_BYTES, page_size=PAGE_SIZE):
        self.page_size = page_size
        self.page_bytes = page_size * token_bytes
        self.buf = torch.zeros(num_pages * self.page_bytes, dtype=torch.uint8)

    def get_page_buffer_meta(self, indices):
        starts = indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        ptrs = [self.buf.data_ptr() + int(p) * self.page_bytes for p in starts]
        return ptrs, [self.page_bytes] * len(ptrs)

    def page(self, p):
        return self.buf[p * self.page_bytes : (p + 1) * self.page_bytes]


class _LogicalPool:
    """DeepSeek-V4's KV anchor: slots without bytes."""

    page_size = PAGE_SIZE

    def get_page_buffer_meta(self, indices):
        return None


class _Group:
    """In-process stand-in for a gloo group: one member per thread."""

    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=10)
        self.vals = [None] * size

    def member(self, rank):
        return _Member(self, rank)


class _Member:
    def __init__(self, group, rank):
        self.group, self.rank, self.size = group, rank, group.size

    def bcast(self, tensor):
        g = self.group
        if self.rank == 0:
            g.vals[0] = tensor.clone()
        g.barrier.wait()
        tensor.copy_(g.vals[0])
        g.barrier.wait()

    def min(self, tensor):
        g = self.group
        g.vals[self.rank] = tensor.clone()
        g.barrier.wait()
        out = torch.stack(g.vals).min(dim=0).values
        g.barrier.wait()
        tensor.copy_(out)


def _identity(rank, tp, cp):
    return dict(
        tp_part=rank % tp, tp_parts=tp, cp_rank=rank // tp, cp_parts=cp, leader_rank=0
    )


def _run(fns):
    out, errors = [None] * len(fns), []

    def call(i):
        try:
            out[i] = fns[i]()
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(len(fns))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    if errors:
        raise errors[0]
    return out


def _config(tp_rank=0, is_mla=False, pull_timeout_ms=30000, cp_size=1):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=cp_size,
        is_mla_model=is_mla,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="m",
        extra_config={
            "radixshmem_name": "test",
            "radixshmem_pull_timeout_ms": pull_timeout_ms,
        },
    )


def _keys(n, seed="k"):
    return [hashlib.sha256(f"{seed}{i}".encode()).hexdigest() for i in range(n)]


def _indices(pages, page_size=PAGE_SIZE):
    return torch.arange(pages * page_size, dtype=torch.int64)


def _info(prefix):
    return HiCacheStorageExtraInfo(prefix_keys=prefix)


class TestRadixShmemStorage(CustomTestCase):
    def setUp(self):
        _FakeClient.servers.clear()
        self._patch = mock.patch.dict(sys.modules, {"radixshmem": _FAKE_RADIXSHMEM})
        self._patch.start()
        from sglang.srt.mem_cache.storage.radixshmem import RadixShmemStorage

        self.cls = RadixShmemStorage

    def tearDown(self):
        self._patch.stop()

    def _backend(self, pages=8, anchor=None, **kw):
        pool = anchor or _FakeHostPool(pages)
        storage = self.cls(_config(**kw), pool)
        storage.register_mem_pool_host(pool)
        return storage, pool

    def test_geometry_from_largest_pool(self):
        storage, _ = self._backend()
        wide = _FakeHostPool(2, token_bytes=2 * TOKEN_BYTES)
        storage.register_mem_host_pool_v2(wide, PoolName.INDEXER)
        self.assertNotIn("test", _FakeClient.servers)
        storage.batch_exists(_keys(1), _info([]))
        self.assertEqual(
            _FakeClient.servers["test"].geometry, (PAGE_SIZE, 2 * PAGE_BYTES)
        )
        # Once configured, a pool with larger pages cannot join.
        with self.assertRaises(ValueError):
            storage.register_mem_host_pool_v2(
                _FakeHostPool(2, token_bytes=3 * TOKEN_BYTES), PoolName.SWA
            )
        # Another engine with another page shape is refused by the server.
        other, _ = self._backend(pages=2)
        with self.assertRaises(_GeometryMismatch):
            other.batch_exists(_keys(1), _info([]))

    def test_write_then_read_roundtrip(self):
        storage, pool = self._backend()
        keys = _keys(4)
        for p in range(4):
            pool.page(p).fill_(p + 1)
        self.assertEqual(storage.batch_set_v1(keys, _indices(4), _info([])), [True] * 4)
        self.assertEqual(storage.batch_exists(keys, _info([])), 4)

        dst = _FakeHostPool(8)
        storage.register_mem_pool_host(dst)
        self.assertEqual(storage.batch_get_v1(keys, _indices(4), _info([])), [True] * 4)
        for p in range(4):
            self.assertTrue(torch.equal(dst.page(p), pool.page(p)))

    def test_prefix_chain(self):
        storage, pool = self._backend()
        keys = _keys(6)
        storage.batch_set_v1(keys[:4], _indices(4), _info([]))
        self.assertEqual(
            storage.batch_set_v1(keys[4:], _indices(2), _info(keys[:4])), [True] * 2
        )
        self.assertEqual(storage.batch_exists(keys[4:], _info(keys[:4])), 2)
        self.assertEqual(storage.batch_exists(keys[4:], _info(_keys(4, "x"))), 0)
        self.assertEqual(storage.batch_exists(keys, _info([])), 6)
        self.assertEqual(
            storage.batch_set_v1(_keys(1, "y"), _indices(1), _info(_keys(2, "z"))),
            [False],
        )
        self.assertEqual(
            storage.batch_set_v1(keys[4:], _indices(2), _info(keys[:4])), [True] * 2
        )

    def test_prefix_none_means_root(self):
        storage, _ = self._backend()
        keys = _keys(2)
        storage.batch_set_v1(keys, _indices(2), HiCacheStorageExtraInfo())
        self.assertEqual(storage.batch_exists(keys, _info([])), 2)

    def _group(self, tp=2, cp=1, is_mla=False, pages=8):
        size = tp * cp
        write, read = _Group(size), _Group(size)
        ranks = []
        for r in range(size):
            pool = _FakeHostPool(pages)
            storage = self.cls(_config(tp_rank=r % tp, is_mla=is_mla, cp_size=cp), pool)
            storage.set_shard_groups(
                write.member(r), read.member(r), **_identity(r, tp, cp)
            )
            storage.register_mem_pool_host(pool)
            ranks.append((storage, pool))
        return ranks

    def test_tp_ranks_share_one_slot(self):
        keys = _keys(3)
        ranks = self._group(tp=2)
        for r, (_, pool) in enumerate(ranks):
            for p in range(3):
                pool.page(p).fill_(10 * r + p + 1)
        got = _run(
            [lambda s=s: s.batch_set_v1(keys, _indices(3), _info([])) for s, _ in ranks]
        )
        self.assertEqual(got, [[True] * 3] * 2)
        server = _FakeClient.servers["test"]
        self.assertEqual(server.geometry, (PAGE_SIZE, 2 * PAGE_BYTES))
        self.assertEqual(len(server.index.paths), 3)
        leader, follower = ranks[0][0], ranks[1][0]
        self.assertEqual(leader.batch_exists(keys, _info([])), 3)
        self.assertEqual(follower.batch_exists(_keys(2, "unknown"), _info([])), 2)

        dsts = [_FakeHostPool(8) for _ in ranks]
        for (s, _), dst in zip(ranks, dsts):
            s.register_mem_pool_host(dst)
        got = _run(
            [lambda s=s: s.batch_get_v1(keys, _indices(3), _info([])) for s, _ in ranks]
        )
        self.assertEqual(got, [[True] * 3] * 2)
        for (_, src), dst in zip(ranks, dsts):
            for p in range(3):
                self.assertTrue(torch.equal(dst.page(p), src.page(p)))

    def test_cp_ranks_write_once_read_all(self):
        keys = _keys(2)
        ranks = self._group(tp=2, cp=2)
        for r, (_, pool) in enumerate(ranks):
            pool.buf.fill_(r % 2 + 1 if r < 2 else 99)
        got = _run(
            [lambda s=s: s.batch_set_v1(keys, _indices(2), _info([])) for s, _ in ranks]
        )
        self.assertEqual(got, [[True] * 2] * 4)
        dsts = [_FakeHostPool(8) for _ in ranks]
        for (s, _), dst in zip(ranks, dsts):
            s.register_mem_pool_host(dst)
        got = _run(
            [lambda s=s: s.batch_get_v1(keys, _indices(2), _info([])) for s, _ in ranks]
        )
        self.assertEqual(got, [[True] * 2] * 4)
        for r, dst in enumerate(dsts):
            self.assertTrue(torch.all(dst.page(0) == r % 2 + 1))

    def test_replicated_pool_written_by_leader(self):
        keys = _keys(3)
        ranks = self._group(tp=2, is_mla=True)
        ranks[0][1].buf.fill_(7)
        self.assertEqual(
            ranks[0][0].batch_set_v1(keys, _indices(3), _info([])), [True] * 3
        )
        self.assertEqual(
            ranks[1][0].batch_set_v1(keys, _indices(3), _info([])), [True] * 3
        )
        self.assertEqual(_FakeClient.servers["test"].geometry, (PAGE_SIZE, PAGE_BYTES))
        dsts = [_FakeHostPool(8) for _ in ranks]
        for (s, _), dst in zip(ranks, dsts):
            s.register_mem_pool_host(dst)
        got = _run(
            [lambda s=s: s.batch_get_v1(keys, _indices(3), _info([])) for s, _ in ranks]
        )
        self.assertEqual(got, [[True] * 3] * 2)
        for dst in dsts:
            self.assertTrue(torch.all(dst.buf[: 3 * PAGE_BYTES] == 7))

    def test_group_read_is_all_or_nothing_per_page(self):
        keys = _keys(4)
        ranks = self._group(tp=2)
        _run(
            [
                lambda s=s: s.batch_set_v1(keys[:2], _indices(2), _info([]))
                for s, _ in ranks
            ]
        )
        got = _run(
            [lambda s=s: s.batch_get_v1(keys, _indices(4), _info([])) for s, _ in ranks]
        )
        self.assertEqual(got, [[True, True, False, False]] * 2)

        def broken(storage):
            storage._copy = mock.Mock(side_effect=RuntimeError("copy"))
            return storage.batch_get_v1(keys[:2], _indices(2), _info([]))

        got = _run(
            [
                lambda: ranks[0][0].batch_get_v1(keys[:2], _indices(2), _info([])),
                lambda: broken(ranks[1][0]),
            ]
        )
        self.assertEqual(got, [[False, False]] * 2)

    def test_failed_copy_is_not_published(self):
        keys = _keys(2)
        ranks = self._group(tp=2)
        ranks[1][0]._copy = mock.Mock(side_effect=RuntimeError("copy"))
        got = _run(
            [lambda s=s: s.batch_set_v1(keys, _indices(2), _info([])) for s, _ in ranks]
        )
        self.assertEqual(got, [[False] * 2] * 2)
        index = _FakeClient.servers["test"].index
        self.assertEqual(index.paths, {})
        self.assertEqual(len(index.free), NUM_SLOTS)

    def test_layer_split_cp_ranks_share_one_slot(self):
        keys = _keys(2)
        size, tp, cp = 4, 2, 2
        write, read = _Group(size), _Group(size)
        ranks = []
        for r in range(size):
            pool = _FakeHostPool(8)
            pool.device_pool = SimpleNamespace(layer_shard_enabled=True)
            pool.buf.fill_(10 * (r // tp) + 1)
            storage = self.cls(_config(tp_rank=r % tp, is_mla=True, cp_size=cp), pool)
            storage.set_shard_groups(
                write.member(r), read.member(r), **_identity(r, tp, cp)
            )
            storage.register_mem_pool_host(pool)
            ranks.append((storage, pool))
        got = _run(
            [lambda s=s: s.batch_set_v1(keys, _indices(2), _info([])) for s, _ in ranks]
        )
        self.assertEqual(got, [[True] * 2] * size)
        self.assertEqual(
            _FakeClient.servers["test"].geometry, (PAGE_SIZE, cp * PAGE_BYTES)
        )
        dsts = [_FakeHostPool(8) for _ in ranks]
        for (s, _), dst in zip(ranks, dsts):
            dst.device_pool = SimpleNamespace(layer_shard_enabled=True)
            s.register_mem_pool_host(dst)
        got = _run(
            [lambda s=s: s.batch_get_v1(keys, _indices(2), _info([])) for s, _ in ranks]
        )
        self.assertEqual(got, [[True] * 2] * size)
        for r, dst in enumerate(dsts):
            self.assertTrue(torch.all(dst.buf[: 2 * PAGE_BYTES] == 10 * (r // tp) + 1))

    def test_read_barrier(self):
        ranks = self._group(tp=2)
        got = _run(
            [
                lambda: ranks[0][0].read_barrier(True),
                lambda: ranks[1][0].read_barrier(False),
            ]
        )
        self.assertEqual(got, [False, False])
        got = _run([lambda s=s: s.read_barrier(True) for s, _ in ranks])
        self.assertEqual(got, [True, True])

    def test_kv_derived_sidecar(self):
        storage, pool = self._backend()
        side = _FakeHostPool(8)
        storage.register_mem_host_pool_v2(side, PoolName.INDEXER)
        keys = _keys(3)
        pool.page(0).fill_(1)
        side.page(0).fill_(9)
        transfer = PoolTransfer(
            name=PoolName.INDEXER,
            host_indices=_indices(3),
            keys=keys,
            indices_from_pool=PoolName.KV,
        )
        storage.batch_set_v1(keys, _indices(3), _info([]))
        self.assertEqual(
            storage.batch_set_v2([transfer], _info([])), {PoolName.INDEXER: [True] * 3}
        )
        result = storage.batch_exists_v2(keys, [transfer], _info([]))
        self.assertEqual(result.kv_hit_pages, 3)
        self.assertEqual(
            result.extra_pool_hit_pages, {PoolName.KV: 3, PoolName.INDEXER: 3}
        )
        self.assertEqual(result.restorable_prefix_pages, [1, 2, 3])

        side.buf.zero_()
        storage.batch_get_v2([transfer], _info([]))
        self.assertTrue(torch.all(side.page(0) == 9))

    def test_trailing_pool(self):
        storage, pool = self._backend()
        swa = _FakeHostPool(8, page_size=2)
        storage.register_mem_host_pool_v2(swa, PoolName.SWA)
        keys = _keys(5)
        storage.batch_set_v1(keys, _indices(5), _info([]))
        # The backup keeps a two-page window at the end of the sequence.
        swa.page(0).fill_(7)
        swa.page(1).fill_(8)
        window = PoolTransfer(
            name=PoolName.SWA,
            host_indices=_indices(2, page_size=2),
            keys=keys[3:5],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        self.assertEqual(
            storage.batch_set_v2([window], _info([])), {PoolName.SWA: [True] * 2}
        )
        # Only the prefix ending at page 5 has a complete window.
        result = storage.batch_exists_v2(keys, [window], _info([]))
        self.assertEqual(result.kv_hit_pages, 5)
        self.assertEqual(result.extra_pool_hit_pages, {PoolName.KV: 5, PoolName.SWA: 5})
        self.assertEqual(result.restorable_prefix_pages, [5])

        swa.buf.zero_()
        self.assertEqual(
            storage.batch_get_v2([window], _info([])), {PoolName.SWA: [True] * 2}
        )
        self.assertTrue(torch.all(swa.page(0) == 7))
        self.assertTrue(torch.all(swa.page(1) == 8))
        window.keys = keys[1:3]
        self.assertEqual(
            storage.batch_get_v2([window], _info([])), {PoolName.SWA: [False] * 2}
        )

    def test_logical_anchor(self):
        storage, _ = self._backend(anchor=_LogicalPool())
        side = _FakeHostPool(8)
        storage.register_mem_host_pool_v2(side, PoolName.DEEPSEEK_V4_C4)
        keys = _keys(3)
        side.page(1).fill_(5)
        transfer = PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C4,
            host_indices=_indices(3),
            keys=keys,
            indices_from_pool=PoolName.KV,
        )
        # The anchor holds no bytes: it is reported stored and never copied.
        self.assertEqual(storage.batch_set_v1(keys, _indices(3), _info([])), [True] * 3)
        before = storage.batch_exists_v2(keys, [transfer], _info([]))
        self.assertEqual(before.kv_hit_pages, 0)
        storage.batch_set_v2([transfer], _info([]))
        result = storage.batch_exists_v2(keys, [transfer], _info([]))
        self.assertEqual(result.kv_hit_pages, 3)
        self.assertEqual(result.restorable_prefix_pages, [1, 2, 3])
        self.assertEqual(storage.batch_get_v1(keys, _indices(3), _info([])), [True] * 3)
        side.buf.zero_()
        storage.batch_get_v2([transfer], _info([]))
        self.assertTrue(torch.all(side.page(1) == 5))


class TestRadixShmemConfig(CustomTestCase):
    def test_shard_layout_from_sglang_groups(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        rows = [[9, 3, 12, 0], [5, 1, 14, 7]]
        tp = SimpleNamespace(ranks=rows[1], rank_in_group=2, world_size=4)
        cp = SimpleNamespace(
            ranks=[0, 7], rank_in_group=1, world_size=2, cpu_group="cp"
        )

        def gather(out, obj, group):
            self.assertEqual(group, "cp")
            out[:] = [(1, rows[1]), (0, rows[0])]

        with (
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_attn_tp_group",
                return_value=tp,
            ),
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_attn_cp_group",
                return_value=cp,
            ),
            mock.patch("torch.distributed.all_gather_object", side_effect=gather),
        ):
            members, identity = HiCacheController._shard_layout()
        self.assertEqual(members, [9, 3, 12, 0, 5, 1, 14, 7])
        self.assertEqual(
            identity,
            dict(tp_part=2, tp_parts=4, cp_rank=1, cp_parts=2, leader_rank=9),
        )
        rows[1] = [5, 1, 14, 9]
        with (
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_attn_tp_group",
                return_value=tp,
            ),
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_attn_cp_group",
                return_value=cp,
            ),
            mock.patch("torch.distributed.all_gather_object", side_effect=gather),
            self.assertRaises(RuntimeError),
        ):
            HiCacheController._shard_layout()

    def test_pp_rank_selects_stage_key(self):
        _FakeClient.servers.clear()
        with mock.patch.dict(sys.modules, {"radixshmem": _FAKE_RADIXSHMEM}):
            from sglang.srt.mem_cache.storage.radixshmem import RadixShmemStorage

            stages = []
            for pp in range(2):
                config = _config()
                config.pp_rank, config.pp_size = pp, 2
                pool = _FakeHostPool(8)
                storage = RadixShmemStorage(config, pool)
                storage.register_mem_pool_host(pool)
                stages.append(storage)
            keys = _keys(2)
            stages[1].batch_set_v1(keys, _indices(2), _info([]))
            self.assertEqual(stages[0].batch_exists(keys, _info([])), 0)
            other = HiCacheStorageExtraInfo(prefix_keys=[], extra_info={"pp_rank": 1})
            self.assertEqual(stages[0].batch_exists(keys, other), 2)

    def test_prefix_keys_forced_on(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        parse = HybridCacheController.parse_storage_backend_extra_config
        self.assertFalse(parse("{}", "mooncake")[4])
        with self.assertLogs(level="WARNING"):
            self.assertTrue(parse("{}", "radixshmem")[4])
        self.assertTrue(
            parse('{"hicache_storage_pass_prefix_keys": true}', "radixshmem")[4]
        )

    def test_startup_guard(self):
        from sglang.srt.arg_groups import hicache_hook

        def check(**kw):
            cfg = SimpleNamespace(
                hicache_storage_backend="radixshmem",
                enable_dsa_cache_layer_split=False,
                tp_size=8,
                pp_size=1,
                attn_dp_size=1,
                nnodes=1,
            )
            for k, v in kw.items():
                setattr(cfg, k, v)
            with mock.patch.object(hicache_hook, "resolving_view", lambda _: cfg):
                hicache_hook.validate_radixshmem_storage(None)

        check()
        check(nnodes=2, attn_dp_size=2)
        check(enable_dsa_cache_layer_split=True)
        with self.assertRaises(ValueError):
            check(nnodes=2)


if __name__ == "__main__":
    unittest.main()
