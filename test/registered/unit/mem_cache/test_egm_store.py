"""CPU unit tests for the EGM HiCache storage backend.

The segment and the host pools are ordinary process memory and the copier is
memmove, so the byte paths, index rules and refusals run as served, minus the
CUDA driver calls.
"""

import ctypes
import hashlib
import json
import os
import random
import tempfile
import threading
import time
import unittest

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.egm import store
from sglang.srt.mem_cache.storage.egm.backend import EgmStorage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_KEEP = []  # buffers live for the whole test run
PS = 4  # tokens per page


def _buf(n: int) -> int:
    b = ctypes.create_string_buffer(n)
    _KEEP.append(b)
    return ctypes.addressof(b)


def _oracle(pool: str, key: str, comp: int, n: int) -> bytes:
    digest = hashlib.blake2b(("%s|%s|%d" % (pool, key, comp)).encode(), digest_size=8)
    return random.Random(int.from_bytes(digest.digest(), "little")).randbytes(n)


def _flip_byte(addr: int) -> None:
    ctypes.memmove(addr, bytes([store.host_view(addr, 1)[0] ^ 0xFF]), 1)


class FakeCopier:
    def __init__(self):
        self.fail = False
        self.gate = None  # (entered, release) events that hold a read inside its copy
        self.issue_ms = []

    def run(self, copies, to_host):
        if self.fail:
            raise RuntimeError("injected copy failure")
        if to_host and self.gate is not None:
            entered, release = self.gate
            entered.set()
            release.wait(5)
        runs = store.coalesce(copies)
        for d, s, n in runs:
            ctypes.memmove(d, s, n)
        return {"ms": 0.01, "issued": len(runs)}


class FakeKVBuffer:
    pass


class FakeHostPool:
    """page_first pool: a K plane then a V plane; one page's K (or V) is contiguous."""

    def __init__(self, tokens, page_size, tok_bytes, layout="page_first"):
        self.size = tokens
        self.page_size = page_size
        self.tok_bytes = tok_bytes
        self.layout = layout
        self.base = _buf(2 * tokens * tok_bytes)
        self.v_offset = tokens * tok_bytes
        self.kv_buffer = FakeKVBuffer()

    def get_page_buffer_meta(self, indices):
        idx = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
        ptrs = []
        for i in range(0, len(idx), self.page_size):
            k = self.base + idx[i] * self.tok_bytes
            ptrs += [k, k + self.v_offset]
        return ptrs, [self.page_size * self.tok_bytes] * len(ptrs)

    def obj_bytes(self):
        return self.page_size * self.tok_bytes

    def page_ptrs(self, page):
        k = self.base + page * self.obj_bytes()
        return [k, k + self.v_offset]

    def fill(self, pool, key, page):
        for j, p in enumerate(self.page_ptrs(page)):
            ctypes.memmove(p, _oracle(pool, key, j, self.obj_bytes()), self.obj_bytes())

    def zero(self, page):
        for p in self.page_ptrs(page):
            ctypes.memset(p, 0, self.obj_bytes())

    def matches(self, pool, key, page):
        return all(
            bytes(store.host_view(p, self.obj_bytes()))
            == _oracle(pool, key, j, self.obj_bytes())
            for j, p in enumerate(self.page_ptrs(page))
        )

    def indices(self, pages):
        return [p * self.page_size + t for p in pages for t in range(self.page_size)]


def _ptrs(pool, pages):
    return [pool.page_ptrs(p) for p in pages]


def _make_store(slots=8, copier=None, max_mismatch=0):
    pool = FakeHostPool(256 * PS, PS, 4096)
    spec = store.PoolSpec(name="kv", comp_sizes=(pool.obj_bytes(),) * 2, weight=1.0)
    data_size = slots * spec.page_bytes + 4 * store.OBJ_ALIGN
    seg = _buf(store.CTRL_BYTES + data_size + store.OBJ_ALIGN)
    regions = store.plan_regions([spec], store.CTRL_BYTES, data_size)
    st = store.Store(
        regions,
        seg_va=seg,
        copier=copier or FakeCopier(),
        host_view=store.host_view,
        max_mismatch=max_mismatch,
    )
    return st, pool, seg


class TestControlBlockWireFormat(CustomTestCase):
    def test_provider_block_bytes(self):
        # The provider and the backend ranks are separate processes;
        # these bytes are their contract.
        b = store.pack_provider_block(
            store.PROV_RUN, 0x1234ABCD, 7, 4096, 1 << 30, 123, wall_ns=555
        )
        self.assertEqual(
            b.hex(),
            "3142434d4745334c0100000001000000cdab34120000000007000000000000002b02"
            "000000000000001000000000000000000040000000007b000000000000003cdb8c05"
            + "00"
            * 28,
        )
        blk = store.unpack_provider_block(b)
        self.assertEqual(
            (blk.state, blk.epoch, blk.heartbeat, blk.data_off, blk.deadline_ns),
            (store.PROV_RUN, 0x1234ABCD, 7, 4096, 123),
        )
        torn = bytearray(b)
        torn[20] ^= 1
        self.assertIsNone(store.unpack_provider_block(bytes(torn)))
        self.assertIsNone(store.unpack_provider_block(bytes(64)))

    def test_rank_block_bytes(self):
        b = store.pack_rank_block(3, store.RANK_ATTACHED, 0x1234ABCD, 9, wall_ns=777)
        self.assertEqual(
            b.hex(),
            "3142504d4745334c0300000001000000cdab3412000000000900000000000000"
            "09030000000000009168d8d2" + "00" * 20,
        )
        self.assertEqual(store.unpack_rank_block(b).rank, 3)
        self.assertEqual(store.rank_slot_offset(0), store.RANK_AREA_OFF)
        with self.assertRaises(ValueError):
            store.rank_slot_offset(1000)


def _handle(node, epoch="1a2b", total=4 << 20, deadline_in=3600.0):
    h = {
        "kind": store.HANDLE_KIND,
        "version": 1,
        "node_index": node,
        "epoch": epoch,
        "gran": 2 << 20,
        "total_size": total,
        "data_off": store.CTRL_BYTES,
        "data_size": total - store.CTRL_BYTES,
        "allocs": [{"numa": 0, "size": total, "off": 0, "fabric": "00" * 64}],
        "deadline": time.time() + deadline_in,
    }
    h["body_sha256"] = store.handle_body_sha(h)
    return h


class TestHandleAndPlanning(CustomTestCase):
    def test_handle_validation(self):
        h = _handle(2)
        self.assertEqual(store.validate_handle(h, node_index=2), [])
        self.assertIn("node_index 2 != 3", store.validate_handle(h, node_index=3)[0])
        self.assertIn(
            "body sha256", store.validate_handle(dict(h, data_size=h["data_size"] + 1))
        )
        self.assertIn(
            "provider deadline passed",
            store.validate_handle(_handle(2, deadline_in=-1)),
        )

    def test_regions_aligned_disjoint_and_proportional(self):
        specs = [
            store.PoolSpec(name="kv", comp_sizes=(512 << 10, 512 << 10), weight=155.0),
            store.PoolSpec(
                name="draft", comp_sizes=(320 << 10, 320 << 10), weight=97.0
            ),
            store.PoolSpec(name="swa", comp_sizes=(1507520, 1507520), weight=45.0),
        ]
        regs = store.plan_regions(specs, 2 << 20, 64 << 30)
        prev_end = 2 << 20
        for r in regs:
            self.assertGreaterEqual(r.start, prev_end)
            self.assertGreater(r.slots, 0)
            for j, b in enumerate(r.comp_base):
                self.assertEqual(b % store.OBJ_ALIGN, 0)
                end = b + r.comp_sizes[j] * r.slots
                limit = r.comp_base[j + 1] if j + 1 < len(r.comp_base) else r.end
                self.assertLessEqual(end, limit)
            prev_end = r.end
        self.assertLessEqual(prev_end, (2 << 20) + (64 << 30))
        share = regs[0].slots * regs[0].page_bytes / (64 << 30)
        self.assertAlmostEqual(share, 155.0 / 297.0, delta=0.01)
        with self.assertRaises(ValueError):
            store.plan_regions(specs, 0, 1 << 30, shares={"kv": 0.5})
        with self.assertRaises(ValueError):
            store.plan_regions(
                specs, 0, 1 << 30, shares={"kv": 0.6, "draft": 0.3, "swa": 0.3}
            )

    def test_coalesce(self):
        self.assertEqual(
            store.coalesce([(0, 100, 10), (10, 110, 5), (20, 115, 5)]),
            [(0, 100, 15), (20, 115, 5)],
        )

    def test_exists_v2_core(self):
        f, hc, rest = store.exists_v2_core(
            6, [("draft", "all_pages", 1, [True, True, True, True, False, True])]
        )
        self.assertEqual((f, hc, rest), (4, {"draft": 4}, [1, 2, 3, 4]))
        f, hc, rest = store.exists_v2_core(
            6, [("swa", "trailing_pages", 2, [False, True, True, False, True, True])]
        )
        self.assertEqual((f, hc["swa"], rest), (6, 6, [3, 6]))
        f, _, rest = store.exists_v2_core(
            6,
            [
                ("draft", "all_pages", 1, [True] * 6),
                ("swa", "trailing_pages", 2, [False, True, True, False, False, False]),
            ],
        )
        self.assertEqual((f, rest), (3, [3]))
        self.assertEqual(store.exists_v2_core(0, []), (0, {}, []))
        with self.assertRaises(ValueError):
            store.exists_v2_core(2, [("x", "block_groups", 1, [True, True])])


class TestStore(CustomTestCase):
    def test_round_trip_skip_and_leading_run(self):
        st, pool, _ = _make_store()
        keys = ["k%d" % i for i in range(5)]
        for i, k in enumerate(keys):
            pool.fill("kv", k, i)
        self.assertEqual(st.put("kv", keys, _ptrs(pool, range(5))), [True] * 5)
        self.assertEqual(st.put("kv", keys[:2], _ptrs(pool, range(2))), [True, True])
        self.assertEqual(st.c["set_pages_skip"], 2)
        self.assertEqual(st.present_prefix("kv", keys + ["zz"]), 5)
        dst = list(range(20, 25))
        for p in dst:
            pool.zero(p)
        self.assertEqual(st.get("kv", keys, _ptrs(pool, dst)), [True] * 5)
        for k, p in zip(keys, dst):
            self.assertTrue(pool.matches("kv", k, p))
        res = st.get("kv", ["k0", "missing", "k2"], _ptrs(pool, [30, 31, 32]))
        self.assertEqual(res, [True, False, False])

    def test_lru_eviction(self):
        st, pool, _ = _make_store(slots=4)
        n = st.regions["kv"].slots
        keys = ["e%d" % i for i in range(n + 1)]
        for i, k in enumerate(keys):
            pool.fill("kv", k, i)
        self.assertEqual(st.put("kv", keys[:n], _ptrs(pool, range(n))), [True] * n)
        self.assertEqual(st.present_prefix("kv", ["e0"]), 1)  # e1 is now the oldest
        self.assertEqual(st.put("kv", [keys[n]], _ptrs(pool, [n])), [True])
        self.assertEqual(st.present_flags("kv", ["e0", "e1"]), [True, False])
        live = [k for k in keys if k != "e1"]
        dst = list(range(40, 40 + len(live)))
        for p in dst:
            pool.zero(p)
        self.assertEqual(st.get("kv", live, _ptrs(pool, dst)), [True] * len(live))
        for k, p in zip(live, dst):
            self.assertTrue(pool.matches("kv", k, p))

    def test_pages_under_read_are_never_handed_out(self):
        """A full region whose pages are all being read gives a writer no slot."""
        cp = FakeCopier()
        st, pool, _ = _make_store(slots=4, copier=cp)
        n = st.regions["kv"].slots
        old = ["o%d" % i for i in range(n)]
        new = ["n%d" % i for i in range(n)]
        for i, k in enumerate(old + new):
            pool.fill("kv", k, i)
        st.put("kv", old, _ptrs(pool, range(n)))
        entered, release = threading.Event(), threading.Event()
        cp.gate = (entered, release)
        dst = list(range(50, 50 + n))
        out = {}
        reader = threading.Thread(
            target=lambda: out.update(res=st.get("kv", old, _ptrs(pool, dst)))
        )
        reader.start()
        self.assertTrue(entered.wait(5))
        cp.gate = None
        wres = st.put("kv", new, _ptrs(pool, range(n, 2 * n)))
        release.set()
        reader.join(5)
        self.assertEqual(wres, [False] * n)
        self.assertGreater(st.c["no_slot"], 0)
        self.assertEqual(out["res"], [True] * n)
        for k, p in zip(old, dst):
            self.assertTrue(pool.matches("kv", k, p))

    def test_interior_corruption_is_a_miss_and_refuses(self):
        st, pool, seg = _make_store()
        pool.fill("kv", "c0", 0)
        pool.fill("kv", "c1", 1)
        st.put("kv", ["c0", "c1"], _ptrs(pool, [0, 1]))
        r = st.regions["kv"]
        _flip_byte(seg + r.addr(r.index["c1"].slot, 1) + pool.obj_bytes() // 2)
        res = st.get("kv", ["c0", "c1"], _ptrs(pool, [10, 11]))
        self.assertEqual(res, [True, False])
        self.assertEqual(st.c["mismatch_pages"], 1)
        self.assertIsNotNone(st.refused)
        self.assertEqual(st.present_prefix("kv", ["c0"]), 0)

    def test_copy_error_refuses_for_the_rest_of_the_run(self):
        cp = FakeCopier()
        st, pool, _ = _make_store(copier=cp)
        pool.fill("kv", "a", 0)
        st.put("kv", ["a"], _ptrs(pool, [0]))
        cp.fail = True
        self.assertEqual(st.get("kv", ["a"], _ptrs(pool, [1])), [False])
        self.assertIn("copy error", st.refused)
        cp.fail = False
        self.assertEqual(st.present_prefix("kv", ["a"]), 0)
        self.assertEqual(st.put("kv", ["b"], _ptrs(pool, [2])), [False])
        self.assertEqual(st.get("kv", ["a"], _ptrs(pool, [3])), [False])
        self.assertTrue(st.drain(1.0))


class FakeSeg:
    def __init__(self, total, close_hold_s=0.0):
        self.total = total
        self.va = _buf(total)
        self.times = {"total_s": 0.0}
        self.closed = 0
        self.close_began = False
        self.close_hold_s = close_hold_s

    def close(self):
        self.close_began = True
        time.sleep(self.close_hold_s)
        self.closed += 1
        return []


class FakeSmall:
    def __init__(self, seg):
        self.seg = seg
        self.late_reads = 0

    def read(self, src, n):
        if self.seg.close_began:
            self.late_reads += 1
        return bytes(store.host_view(src, n))

    def write(self, dst, data):
        ctypes.memmove(dst, data, len(data))


class FakeEgmStorage(EgmStorage):
    close_hold_s = 0.0

    def _open_device(self, handle, streams):
        seg = FakeSeg(int(handle["total_size"]), close_hold_s=self.close_hold_s)
        blk = store.pack_provider_block(
            store.PROV_RUN,
            int(handle["epoch"], 16),
            1,
            handle["data_off"],
            handle["data_size"],
            0,
        )
        ctypes.memmove(seg.va, blk, len(blk))
        copier = store.DirCopier(get_copier=FakeCopier(), set_copier=FakeCopier())
        return seg, copier, FakeSmall(seg)

    def _start_monitor(self):
        pass  # tests drive _monitor_once / _shut by hand


class TestBackend(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write_handle(self, rank, handle):
        with open(os.path.join(self.tmp, "handle_%d.json" % rank), "w") as fh:
            json.dump(handle, fh)

    def _config(self, rank=0, extra=None, mla=False, pp_size=1):
        ec = {
            "egm_handles": os.path.join(self.tmp, "handle_{rank}.json"),
            "egm_crc_threads": 1,
        }
        ec.update(extra or {})
        return HiCacheStorageConfig(
            tp_rank=rank,
            tp_size=4,
            pp_rank=0,
            pp_size=pp_size,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=mla,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="m",
            extra_config=ec,
        )

    def _attached(self, rank=1, pools=None, close_hold_s=0.0):
        self._write_handle(rank, _handle(rank))
        FakeEgmStorage.close_hold_s = close_hold_s
        b = FakeEgmStorage(self._config(rank))
        kv = FakeHostPool(64 * PS, PS, 4096)
        b.register_mem_pool_host(kv)
        for name, p in (pools or {}).items():
            b.register_mem_host_pool_v2(p, name)
        b.on_host_pools_registered()
        return b, kv

    def test_construction_refusals(self):
        self._write_handle(0, _handle(0))
        with self.assertRaisesRegex(ValueError, "rank-replicated"):
            FakeEgmStorage(self._config(0, mla=True))
        with self.assertRaisesRegex(ValueError, "pipeline"):
            FakeEgmStorage(self._config(0, pp_size=2))
        with self.assertRaisesRegex(ValueError, "egm_handles is missing"):
            FakeEgmStorage(self._config(0, extra={"egm_handles": ""}))
        with self.assertRaisesRegex(ValueError, "handle"):
            FakeEgmStorage(self._config(3))  # no handle file for rank 3
        self._write_handle(2, _handle(5))
        with self.assertRaisesRegex(ValueError, "node_index"):
            FakeEgmStorage(self._config(2))
        bad = _handle(0)
        bad["data_size"] += 1
        self._write_handle(0, bad)
        with self.assertRaisesRegex(ValueError, "body sha256"):
            FakeEgmStorage(self._config(0))

    def test_registration_refusals(self):
        self._write_handle(0, _handle(0))
        b = FakeEgmStorage(self._config(0))
        with self.assertRaisesRegex(ValueError, "layout"):
            b.register_mem_pool_host(FakeHostPool(16, PS, 64, layout="layer_first"))
        p = FakeHostPool(16, PS, 64)
        p.kv_buffer = None
        with self.assertRaisesRegex(ValueError, "no KV buffer"):
            b.register_mem_pool_host(p)

    def test_v1_v2_round_trip_and_exists_v2(self):
        draft = FakeHostPool(64 * PS, PS, 2560)
        swa = FakeHostPool(32 * PS, PS, 6144)
        b, kv = self._attached(pools={PoolName.DRAFT: draft, PoolName.SWA: swa})
        keys = ["h%d" % i for i in range(6)]
        for i, k in enumerate(keys):
            kv.fill("kv", k, i)
            draft.fill("draft", k, i)
        self.assertEqual(b.batch_set_v1(keys, kv.indices(range(6))), [True] * 6)
        r = b.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=draft.indices(range(4)),
                    keys=keys[:4],
                )
            ]
        )
        self.assertEqual(r[PoolName.DRAFT], [True] * 4)
        for i, k in zip((3, 4), keys[1:3]):
            swa.fill("swa", k, i)
        r = b.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=swa.indices([3, 4]),
                    keys=keys[1:3],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]
        )
        self.assertEqual(r[PoolName.SWA], [True, True])
        res = b.batch_exists_v2(
            keys,
            [
                PoolTransfer(name=PoolName.DRAFT, indices_from_pool=PoolName.KV),
                PoolTransfer(
                    name=PoolName.SWA,
                    keys=["placeholder"] * 2,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                ),
            ],
        )
        self.assertEqual(res.kv_hit_pages, 3)
        self.assertEqual(res.extra_pool_hit_pages[PoolName.KV], 6)
        self.assertEqual(res.extra_pool_hit_pages[PoolName.DRAFT], 4)
        self.assertEqual(res.restorable_prefix_pages, [3])
        for p in range(40, 43):
            kv.zero(p)
            draft.zero(p)
        res = b.batch_get_v1(keys[:3], kv.indices(range(40, 43)))
        self.assertEqual(res, [True] * 3)
        g = b.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=draft.indices(range(40, 43)),
                    keys=keys[:3],
                )
            ]
        )
        self.assertEqual(g[PoolName.DRAFT], [True] * 3)
        for i, k in enumerate(keys[:3]):
            self.assertTrue(kv.matches("kv", k, 40 + i))
            self.assertTrue(draft.matches("draft", k, 40 + i))
        # A geometry mismatch switches the tier off and reads as a miss.
        self.assertEqual(
            b.batch_get_v1(keys[:2], kv.indices(range(40, 43))), [False, False]
        )
        self.assertIsNotNone(b.store.refused)
        self.assertEqual(b.seg.closed, 1)
        self.assertEqual(b.batch_exists(keys), 0)

    def test_monitor_refusals(self):
        b, _ = self._attached(rank=2)
        raw = b.small.read(b.seg.va + store.rank_slot_offset(2), 64)
        p = store.unpack_rank_block(raw)
        self.assertEqual((p.state, p.epoch_echo), (store.RANK_ATTACHED, b.epoch))
        t = 1000.0
        self.assertIsNone(b._monitor_once(t))
        self.assertIsNone(b._monitor_once(t + 1))
        self.assertIn("heartbeat stale", b._monitor_once(t + 1 + b.hb_timeout + 0.1))

        b2, kv2 = self._attached(rank=3)
        blk = store.pack_provider_block(store.PROV_STOPPING, b2.epoch, 9, 4096, 1, 0)
        ctypes.memmove(b2.seg.va, blk, len(blk))
        self.assertIn("stopping", b2._monitor_once(5.0))
        b2._shut("provider stopping")
        self.assertEqual(b2.seg.closed, 1)
        raw = bytes(store.host_view(b2.seg.va + store.rank_slot_offset(3), 64))
        self.assertEqual(store.unpack_rank_block(raw).state, store.RANK_DETACHED)
        self.assertEqual(b2.batch_set_v1(["z"], kv2.indices([0])), [False])

        b3, _ = self._attached(rank=0)
        blk = store.pack_provider_block(store.PROV_RUN, b3.epoch ^ 1, 9, 4096, 1, 0)
        ctypes.memmove(b3.seg.va, blk, len(blk))
        self.assertIn("epoch", b3._monitor_once(5.0))

        b4, kv4 = self._attached(rank=1)
        kv4.fill("kv", "q", 0)
        b4.batch_set_v1(["q"], kv4.indices([0]))
        self.assertIsNone(b4._monitor_step(time.monotonic()))
        b4.store.refuse("copy error on read: injected")
        self.assertIn("store: copy error", b4._monitor_step(time.monotonic()))

    def test_late_sidecar_registration(self):
        b, kv = self._attached()
        draft = FakeHostPool(64 * PS, PS, 2560)
        b.register_mem_host_pool_v2(draft, PoolName.DRAFT)
        self.assertIsNone(b.store)
        kv.fill("kv", "a", 0)
        draft.fill("draft", "a", 0)
        self.assertEqual(b.batch_set_v1(["a"], kv.indices([0])), [True])
        self.assertIn(PoolName.DRAFT.value, b.store.regions)
        r = b.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT, host_indices=draft.indices([0]), keys=["a"]
                )
            ]
        )
        self.assertEqual(r[PoolName.DRAFT], [True])
        # A pool after the first storage call switches the tier off, without raising.
        b.register_mem_host_pool_v2(FakeHostPool(16 * PS, PS, 64), PoolName.SWA)
        self.assertIn("registered after the regions were fixed", b.store.refused)
        self.assertEqual(b.batch_exists(["a"]), 0)

    def test_restart_over_old_bytes_reads_nothing(self):
        b, kv = self._attached()
        keys = ["s%d" % i for i in range(3)]
        for i, k in enumerate(keys):
            kv.fill("kv", k, i)
        self.assertEqual(b.batch_set_v1(keys, kv.indices(range(3))), [True] * 3)
        data = bytes(store.host_view(b.seg.va, b.seg.total))[store.CTRL_BYTES :]
        b2, kv2 = self._attached()
        ctypes.memmove(b2.seg.va + store.CTRL_BYTES, data, len(data))
        self.assertEqual(b2.batch_exists(keys), 0)
        self.assertEqual(b2.batch_get_v1(keys, kv2.indices(range(3))), [False] * 3)

    def test_concurrent_shuts_unmap_once_without_late_control_reads(self):
        b, _ = self._attached(close_hold_s=0.3)
        go = threading.Barrier(3)
        stop = threading.Event()

        def shut(i):
            go.wait(5)
            b._shut("caller %d" % i)

        def monitor_reads():
            go.wait(5)
            while not stop.is_set():
                b._read_provider()
                time.sleep(0.002)

        threads = [threading.Thread(target=shut, args=(i,)) for i in range(2)]
        threads.append(threading.Thread(target=monitor_reads))
        for t in threads:
            t.start()
        for t in threads[:2]:
            t.join(5)
        stop.set()
        threads[2].join(5)
        self.assertEqual(b.seg.closed, 1)
        self.assertEqual(b.small.late_reads, 0)
        self.assertIsNone(b._read_provider())

    def test_close_waits_for_a_shut_under_way(self):
        b, _ = self._attached(close_hold_s=0.3)
        t = threading.Thread(target=b._shut, args=("monitor latch",))
        t.start()
        while not b.seg.close_began:
            time.sleep(0.005)
        b.close()
        self.assertTrue(b.unmapped)
        self.assertEqual(b.seg.closed, 1)
        t.join(5)


if __name__ == "__main__":
    unittest.main()
