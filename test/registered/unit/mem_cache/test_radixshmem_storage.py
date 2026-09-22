"""Unit coverage for the radixshmem L3 backend against an in-memory fake."""

import enum
import hashlib
import sys
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


class _InsertError(enum.Enum):
    OK = 0
    FULL_PATH_MISSING = 7


class _GeometryMismatch(RuntimeError):
    pass


class _FakeGeometry:
    def __init__(self, block_size, full_slot_bytes):
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
        return SimpleNamespace(
            status=0,
            common_hit=hit,
            full_slots=np.array(
                [self.paths[tuple(chain[: i + 1].tolist())] for i in range(hit)],
                dtype=np.int32,
            ),
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
            return SimpleNamespace(error=_InsertError.FULL_PATH_MISSING)
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


class _FakeClient:
    servers = {}

    def __init__(self, name, geometry=None, *, endpoint=None, max_outstanding=256):
        self.server = _FakeClient.servers.setdefault(name, _FakeServer())
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
        return self.server.index

    @property
    def store(self):
        return self.server.index.store

    def is_distributed(self):
        return False

    def pull_async(self, chain, mask, lock=True, timeout_ms=0, block=True):
        result = self.index.query(chain, mask, False, lock, True)
        return SimpleNamespace(wait=lambda timeout=None: result, cancel=lambda: None)

    def close(self):
        pass


_FAKE_SHMRADIX = SimpleNamespace(
    RadixClient=_FakeClient,
    Geometry=_FakeGeometry,
    GeometryMismatch=_GeometryMismatch,
    ComponentType=_Kind,
    COMPONENT_MASK_FULL=1,
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


def _config(tp_rank=0, is_mla=False):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_mla,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="m",
        extra_config={"radixshmem_name": "test"},
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
        self._patch = mock.patch.dict(sys.modules, {"shmradix": _FAKE_SHMRADIX})
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
        self.assertIsNone(_FakeClient.servers["test"].geometry)
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
        # Inserting under a missing prefix is refused, not stored.
        self.assertEqual(
            storage.batch_set_v1(_keys(1, "y"), _indices(1), _info(_keys(2, "z"))),
            [False],
        )

    def test_prefix_none_means_root(self):
        storage, _ = self._backend()
        keys = _keys(2)
        storage.batch_set_v1(keys, _indices(2), HiCacheStorageExtraInfo())
        self.assertEqual(storage.batch_exists(keys, _info([])), 2)

    def test_rank_salt(self):
        keys = _keys(2)
        s0, p0 = self._backend(tp_rank=0)
        s1, _ = self._backend(tp_rank=1)
        s0.batch_set_v1(keys, _indices(2), _info([]))
        self.assertEqual(s0.batch_exists(keys, _info([])), 2)
        self.assertEqual(s1.batch_exists(keys, _info([])), 0)
        # Rank-replicated pools share one path.
        m0, _ = self._backend(tp_rank=0, is_mla=True)
        m1, _ = self._backend(tp_rank=1, is_mla=True)
        m0.batch_set_v1(keys, _indices(2), _info([]))
        self.assertEqual(m1.batch_exists(keys, _info([])), 2)

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


if __name__ == "__main__":
    unittest.main()
