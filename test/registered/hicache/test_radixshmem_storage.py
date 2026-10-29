"""Unit tests for the radixshmem HiCache storage backend.

The backend runs against an in-process radixshmem server (no etcd, no RDMA)
and CPU host pools; the ranks of one attention TP/CP group are threads that
share a barrier-based stand-in for the gloo groups.

Run with:
    python3 -m pytest test/registered/hicache/test_radixshmem_storage.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import hashlib
import threading
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.test.test_utils import CustomTestCase

try:
    import radixshmem
    from radixshmem import RadixServer, ServerConfig

    HAVE_RADIXSHMEM = hasattr(radixshmem, "_data")
except ImportError:
    HAVE_RADIXSHMEM = False

PAGE = 4
LAYERS = 2
HEAD_DIM = 8
KV_HEADS = 4
MODULE = "sglang.srt.mem_cache.storage.radixshmem.radixshmem_storage"


class _Group:
    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=20)
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
        t.join(60)
    if errors:
        raise errors[0]
    return out


def _kv_pool(heads, layout="page_first", pages=8):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

    device = MHATokenToKVPool(
        dtype=torch.bfloat16,
        size=pages * PAGE,
        page_size=PAGE,
        layer_num=LAYERS,
        start_layer=0,
        end_layer=LAYERS,
        head_num=heads,
        head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        device="cpu",
        enable_memory_saver=False,
    )
    return MHATokenToKVPoolHost(
        buffer_info=device.get_device_pool_infos()[0].buffer_info,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PAGE,
        layout=layout,
        pin_memory=False,
        device="cpu",
        device_capacity_tokens=device.size,
        start_layer=device.start_layer,
        end_layer=device.end_layer,
        write_back_staging=device.hicache_write_back_staging,
    )


def _page_view(pool, p):
    """One page of an MHA host pool as (2, layer, token, head, dim)."""
    kv, t = pool.kv_buffer, p * PAGE
    if pool.layout == "page_first":
        return kv[:, t : t + PAGE].permute(0, 2, 1, 3, 4)
    if pool.layout == "layer_first":
        return kv[:, :, t : t + PAGE]
    if pool.layout == "page_first_direct":
        return kv[:, p]
    return kv[:, p].permute(0, 3, 2, 1, 4)


class _FlatPool:
    """A page-first byte pool: one contiguous segment per page."""

    def __init__(self, pages, page_bytes, page_size=PAGE):
        self.page_size, self.page_bytes = page_size, page_bytes
        self.buf = torch.zeros(pages * page_bytes, dtype=torch.uint8)

    def get_page_buffer_meta(self, indices):
        starts = indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        ptrs = [self.buf.data_ptr() + int(p) * self.page_bytes for p in starts]
        return ptrs, [self.page_bytes] * len(ptrs)

    def page(self, p):
        return self.buf[p * self.page_bytes : (p + 1) * self.page_bytes]


class _LogicalPool:
    page_size = PAGE

    def get_page_buffer_meta(self, indices):
        return None


class _MambaPool:
    """temporal + one conv buffer + one side state, page_first like MambaPoolHost."""

    def __init__(self, heads, conv_dim, slots=4):
        self.layout, self.page_size = "page_first", 1
        self.temporal_buffer = torch.zeros(slots, 2, 1, heads, 2, 2)
        self.conv_buffer = [torch.zeros(slots, 2, 1, conv_dim, 3)]
        self.slot_state_buffers = [torch.zeros(slots, 3, 5)]
        self.slot_state_device_tensors = list(self.slot_state_buffers)


def _keys(n, seed="k"):
    return [hashlib.sha256(f"{seed}{i}".encode()).hexdigest() for i in range(n)]


def _idx(pages, page_size=PAGE):
    return torch.arange(pages * page_size, dtype=torch.int64)


def _info(prefix=()):
    return HiCacheStorageExtraInfo(prefix_keys=list(prefix))


@unittest.skipUnless(HAVE_RADIXSHMEM, "radixshmem with a data plane is not installed")
class TestRadixShmemStorage(CustomTestCase):
    def setUp(self):
        from sglang.srt.mem_cache.storage.radixshmem import RadixShmemStorage

        self.cls = RadixShmemStorage
        self._new_server()
        for target, value in (
            (f"{MODULE}._prefix_keys_enabled", lambda: True),
            ("sglang.srt.mem_cache.pool_host.mamba.MambaPoolHost", _MambaPool),
        ):
            patch = mock.patch(target, value)
            patch.start()
            self.addCleanup(patch.stop)

    def _new_server(self, swa_ratio=0.0, mamba_ratio=0.0):
        tag = uuid.uuid4().hex[:8]
        self.name, self.endpoint = f"/rsu_{tag}", f"unix:///tmp/rsu_{tag}.sock"
        server = RadixServer(
            ServerConfig(
                name=self.name,
                endpoint=self.endpoint,
                data_bytes=16 << 20,
                swa_ratio=swa_ratio,
                mamba_ratio=mamba_ratio,
                slot_align=64,
                prefault=False,
                background_evict_ratio=0.0,
            )
        ).start()
        self.addCleanup(server.close)

    def _config(self, tp=1, cp=1, is_mla=False, **extra):
        return HiCacheStorageConfig(
            tp_rank=0,
            tp_size=tp,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=cp,
            is_mla_model=is_mla,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="m",
            extra_config={
                "radixshmem_name": self.name,
                "radixshmem_endpoint": self.endpoint,
                "radixshmem_kv_heads": KV_HEADS,
                **extra,
            },
        )

    def _group(self, make_pool, tp=1, cp=1, is_mla=False, **extra):
        size = tp * cp
        write, read = _Group(size), _Group(size)
        ranks = []
        for r in range(size):
            pool = make_pool(r)
            storage = self.cls(self._config(tp, cp, is_mla, **extra), pool)
            if size > 1:
                storage.set_shard_groups(
                    write.member(r),
                    read.member(r),
                    tp_part=r % tp,
                    tp_parts=tp,
                    cp_rank=r // tp,
                    cp_parts=cp,
                    leader_rank=0,
                )
            storage.register_mem_pool_host(pool)
            ranks.append((storage, pool))
            self.addCleanup(storage.close)
        return ranks

    def _set(self, ranks, keys, prefix=()):
        n = len(keys)
        return _run(
            [lambda s=s: s.batch_set_v1(keys, _idx(n), _info(prefix)) for s, _ in ranks]
        )

    def _get(self, ranks, keys, prefix=()):
        n = len(keys)
        return _run(
            [lambda s=s: s.batch_get_v1(keys, _idx(n), _info(prefix)) for s, _ in ranks]
        )

    def _fill(self, ranks, tp, pages, replicas=1):
        full = torch.randn(pages, 2, LAYERS, PAGE, KV_HEADS, HEAD_DIM)
        full = full.to(torch.bfloat16)
        heads = max(1, KV_HEADS // tp)
        for r, (_, pool) in enumerate(ranks):
            first = (r % tp) // replicas * heads
            for p in range(pages):
                part = full[p, ..., first : first + heads, :]
                if (r % tp) % replicas:
                    part = torch.full_like(part, -1)
                _page_view(pool, p).copy_(part)
        return full

    def _check(self, ranks, tp, full, pages):
        heads = KV_HEADS // tp
        for r, (_, pool) in enumerate(ranks):
            first = (r % tp) * heads
            for p in range(pages):
                want = full[p, ..., first : first + heads, :]
                got = _page_view(pool, p)
                self.assertTrue(torch.equal(got, want), f"rank {r} page {p}")

    def test_round_trip_and_prefix_chain(self):
        ((storage, pool),) = self._group(lambda r: _kv_pool(KV_HEADS))
        keys = _keys(6)
        full = self._fill([(storage, pool)], 1, 6)
        self.assertEqual(storage.batch_set_v1(keys[:4], _idx(4), _info()), [True] * 4)
        self.assertEqual(
            storage.batch_set_v1(keys[4:], _idx(2), _info(keys[:4])), [True] * 2
        )
        self.assertEqual(storage.batch_exists(keys, _info()), 6)
        self.assertEqual(storage.batch_exists(keys[4:], _info(keys[:4])), 2)
        self.assertEqual(storage.batch_exists(keys[4:], _info(_keys(4, "x"))), 0)
        orphan = storage.batch_set_v1(_keys(1, "y"), _idx(1), _info(_keys(2, "z")))
        self.assertEqual(orphan, [False])
        pool.kv_buffer.zero_()
        self.assertEqual(storage.batch_get_v1(keys[:4], _idx(4), _info()), [True] * 4)
        self._check([(storage, pool)], 1, full, 4)
        with self.assertRaises(ValueError):
            storage.batch_set_v1(keys, _idx(6), HiCacheStorageExtraInfo())

    def test_one_slot_shared_across_tp_sizes_and_layouts(self):
        keys = _keys(3)
        tp2 = self._group(lambda r: _kv_pool(KV_HEADS // 2), tp=2)
        full = self._fill(tp2, 2, 3)
        self.assertEqual(self._set(tp2, keys), [[True] * 3] * 2)
        slot = 2 * LAYERS * PAGE * KV_HEADS * HEAD_DIM * 2
        self.assertEqual(tp2[0][0].slot_bytes, slot)
        self.assertEqual([s.leader for s, _ in tp2], [True, False])
        for tp, layout in (
            (4, "page_head"),
            (1, "layer_first"),
            (2, "page_first_direct"),
        ):
            ranks = self._group(lambda r: _kv_pool(KV_HEADS // tp, layout), tp=tp)
            self.assertEqual(self._get(ranks, keys), [[True] * 3] * tp, layout)
            self._check(ranks, tp, full, 3)

    def test_replicated_heads_and_mla_written_once(self):
        keys = _keys(2)
        tp8 = self._group(lambda r: _kv_pool(1), tp=8)
        full = self._fill(tp8, 8, 2, replicas=2)
        self.assertEqual(self._set(tp8, keys), [[True] * 2] * 8)
        tp1 = self._group(lambda r: _kv_pool(KV_HEADS))
        self.assertEqual(self._get(tp1, keys), [[True] * 2])
        self._check(tp1, 1, full, 2)

        self._new_server()
        mla = self._group(lambda r: _FlatPool(4, 64), tp=2, is_mla=True)
        mla[0][1].buf.fill_(7)
        mla[1][1].buf.fill_(9)
        self.assertEqual(self._set(mla, _keys(2, "m")), [[True] * 2] * 2)
        self.assertEqual(mla[0][0].slot_bytes, 64)
        for _, pool in mla:
            pool.buf.zero_()
        self.assertEqual(self._get(mla, _keys(2, "m")), [[True] * 2] * 2)
        for _, pool in mla:
            self.assertTrue(torch.all(pool.buf[:128] == 7))

    def test_cp_ranks_and_layer_split(self):
        keys = _keys(2)
        ranks = self._group(lambda r: _FlatPool(4, 32), cp=2, is_mla=True)
        ranks[0][1].buf.fill_(3)
        ranks[1][1].buf.fill_(99)
        self.assertEqual(self._set(ranks, keys), [[True] * 2] * 2)
        for _, pool in ranks:
            pool.buf.zero_()
        self.assertEqual(self._get(ranks, keys), [[True] * 2] * 2)
        for _, pool in ranks:
            self.assertTrue(torch.all(pool.buf[:64] == 3))

        def split_pool(r):
            pool = _FlatPool(4, 32)
            pool.device_pool = SimpleNamespace(layer_shard_enabled=True)
            pool.buf.fill_(10 * (r // 2) + 1)
            return pool

        self._new_server()
        split = self._group(split_pool, tp=2, cp=2, is_mla=True)
        self.assertEqual(self._set(split, _keys(2, "ls")), [[True] * 2] * 4)
        self.assertEqual(split[0][0].slot_bytes, 64)
        for _, pool in split:
            pool.buf.zero_()
        self.assertEqual(self._get(split, _keys(2, "ls")), [[True] * 2] * 4)
        for r, (_, pool) in enumerate(split):
            self.assertTrue(torch.all(pool.buf[:64] == 10 * (r // 2) + 1))

    def test_group_failures_stay_rank_consistent(self):
        keys = _keys(4)
        ranks = self._group(lambda r: _FlatPool(8, 32), tp=2)
        self.assertEqual(self._set(ranks, keys[:2]), [[True] * 2] * 2)
        self.assertEqual(self._get(ranks, keys), [[True, True, False, False]] * 2)

        broken = mock.Mock(side_effect=RuntimeError("copy"))
        with mock.patch.object(ranks[1][0], "_copy", broken):
            self.assertEqual(self._get(ranks, keys[:2]), [[False] * 2] * 2)
            self.assertEqual(self._set(ranks, _keys(2, "f")), [[False] * 2] * 2)
        self.assertEqual(ranks[0][0].batch_exists(_keys(2, "f"), _info()), 0)

        got = _run(
            [
                lambda: ranks[0][0].read_barrier(True),
                lambda: ranks[1][0].read_barrier(False),
            ]
        )
        self.assertEqual(got, [False, False])

    def test_kv_derived_pool_on_logical_anchor(self):
        ((storage, _),) = self._group(lambda r: _LogicalPool())
        side = _FlatPool(8, 48)
        storage.register_mem_host_pool_v2(side, PoolName.DEEPSEEK_V4_C4)
        storage.describe_sidecar(SidecarPoolSpec(PoolName.DEEPSEEK_V4_C4, PoolName.KV))
        keys = _keys(3)
        side.page(1).fill_(5)
        transfer = PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C4,
            host_indices=_idx(3),
            keys=keys,
            indices_from_pool=PoolName.KV,
        )
        got = storage.batch_set_v2([transfer], _info())
        self.assertEqual(got, {PoolName.DEEPSEEK_V4_C4: [True] * 3})
        self.assertEqual(storage.batch_set_v1(keys, _idx(3), _info()), [True] * 3)
        result = storage.batch_exists_v2(keys, [transfer], _info())
        self.assertEqual(result.kv_hit_pages, 3)
        self.assertEqual(result.restorable_prefix_pages, [1, 2, 3])
        side.buf.zero_()
        self.assertEqual(storage.batch_get_v1(keys, _idx(3), _info()), [True] * 3)
        self.assertTrue(torch.all(side.page(1) == 5))
        side.buf.zero_()
        got = storage.batch_get_v2([transfer], _info())
        self.assertEqual(got, {PoolName.DEEPSEEK_V4_C4: [True] * 3})
        self.assertTrue(torch.all(side.page(1) == 0))

    def test_swa_component_and_unknown_window(self):
        keys = _keys(5)
        window = lambda: PoolTransfer(
            name=PoolName.SWA,
            host_indices=_idx(2),
            keys=keys[3:5],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        self._new_server(swa_ratio=0.2)
        ((storage, _),) = self._group(
            lambda r: _FlatPool(16, 64), radixshmem_swa_window_pages=2
        )
        swa = _FlatPool(16, 32)
        storage.register_mem_host_pool_v2(swa, PoolName.SWA)
        storage.batch_set_v1(keys, _idx(5), _info())
        self.assertNotEqual(storage.pack_of[PoolName.SWA].kind, storage.full)
        swa.page(0).fill_(3)
        swa.page(1).fill_(4)
        got = storage.batch_set_v2([window()], _info(keys[:3]))
        self.assertEqual(got, {PoolName.SWA: [True] * 2})
        result = storage.batch_exists_v2(keys, [window()], _info())
        self.assertEqual(result.restorable_prefix_pages, [5])
        swa.buf.zero_()
        got = storage.batch_get_v2([window()], _info(keys[:3]))
        self.assertEqual(got, {PoolName.SWA: [True] * 2})
        self.assertTrue(torch.all(swa.page(0) == 3))
        self.assertTrue(torch.all(swa.page(1) == 4))

        ((storage, _),) = self._group(lambda r: _FlatPool(16, 64))
        storage.register_mem_host_pool_v2(_FlatPool(16, 32), PoolName.SWA)
        with self.assertRaises(ValueError):
            storage.batch_exists(keys, _info())

    def test_mamba_component_across_tp_with_deferred_publish(self):
        self._new_server(mamba_ratio=0.2)
        groups = [4, 4, 8]
        temporal = torch.randn(2, 1, 4, 2, 2)
        conv = torch.randn(2, 1, 16, 3)
        side = torch.randn(3, 5)

        def shard(t, tp, part, subs=None):
            if subs is None:
                n = t.shape[2] // tp
                return t.narrow(2, part * n, n)
            pieces, at = [], 0
            for full in subs:
                n = full // tp
                pieces.append(t.narrow(2, at + part * n, n))
                at += full
            return torch.cat(pieces, dim=2)

        def build(tp):
            ranks = self._group(lambda r: _kv_pool(KV_HEADS // tp), tp=tp)
            mambas = [_MambaPool(4 // tp, 16 // tp) for _ in range(tp)]
            for (s, _), m in zip(ranks, mambas):
                s.mamba_shape = SimpleNamespace(
                    conv_shard_groups=groups, conv_slice_axis=0
                )
                s.register_mem_host_pool_v2(m, PoolName.MAMBA)
            return ranks, mambas

        keys = _keys(2)
        state = lambda: PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([2]),
            keys=keys[1:],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        writers, mambas = build(2)
        for r, m in enumerate(mambas):
            m.temporal_buffer[2].copy_(shard(temporal, 2, r))
            m.conv_buffer[0][2].copy_(shard(conv, 2, r, groups))
            m.slot_state_buffers[0][2].copy_(side if r == 0 else -side)
        got = _run(
            [lambda s=s: s.batch_set_v2([state()], _info(keys[:1])) for s, _ in writers]
        )
        self.assertEqual(got, [{PoolName.MAMBA: [True]}] * 2)
        self.assertEqual(len(writers[0][0]._deferred), 1)
        self.assertEqual(self._set(writers, keys), [[True] * 2] * 2)
        self.assertEqual(writers[0][0]._deferred, [])
        probe = PoolTransfer(
            name=PoolName.MAMBA, keys=["x"], hit_policy=PoolHitPolicy.TRAILING_PAGES
        )
        hit = writers[0][0].batch_exists_v2(keys, [probe], _info()).kv_hit_pages
        self.assertEqual(hit, 2)

        readers, rmambas = build(1)
        got = readers[0][0].batch_get_v2([state()], _info(keys[:1]))
        self.assertEqual(got, {PoolName.MAMBA: [True]})
        self.assertTrue(torch.equal(rmambas[0].temporal_buffer[2], temporal))
        self.assertTrue(torch.equal(rmambas[0].conv_buffer[0][2], conv))
        self.assertTrue(torch.equal(rmambas[0].slot_state_buffers[0][2], side))

    def test_unavailable_server_degrades_and_recovers(self):
        ((storage, _),) = self._group(lambda r: _FlatPool(8, 64), radixshmem_retry_s=60)
        storage.server_name, storage.endpoint = "/rsu_missing", None
        keys = _keys(2)
        with self.assertLogs(level="WARNING") as logs:
            self.assertEqual(storage.batch_exists(keys, _info()), 0)
            self.assertEqual(storage.batch_set_v1(keys, _idx(2), _info()), [False] * 2)
            self.assertEqual(storage.batch_get_v1(keys, _idx(2), _info()), [False] * 2)
        self.assertEqual(len(logs.output), 1)
        storage.server_name, storage.endpoint = self.name, self.endpoint
        self.assertEqual(storage.batch_exists(keys, _info()), 0)
        storage._retry_at = 0.0
        self.assertEqual(storage.batch_set_v1(keys, _idx(2), _info()), [True] * 2)
        self.assertEqual(storage.batch_exists(keys, _info()), 2)

    def test_shard_layout_from_sglang_groups(self):
        from sglang.srt.mem_cache.storage.radixshmem.radixshmem_storage import (
            _shard_layout,
        )

        rows = {0: [9, 3, 12, 0], 1: [5, 1, 14, 7]}
        tp = SimpleNamespace(ranks=rows[1], rank_in_group=2, world_size=4)
        cp = SimpleNamespace(
            ranks=[0, 7], rank_in_group=1, world_size=2, cpu_group="cp"
        )

        def gather(out, obj, group):
            out[:] = [(1, rows[1]), (0, rows[0])]

        for target, kw in (
            (
                "sglang.srt.distributed.parallel_state.get_attn_tp_group",
                dict(return_value=tp),
            ),
            (
                "sglang.srt.distributed.parallel_state.get_attn_cp_group",
                dict(return_value=cp),
            ),
            ("torch.distributed.all_gather_object", dict(side_effect=gather)),
        ):
            patch = mock.patch(target, **kw)
            patch.start()
            self.addCleanup(patch.stop)
        members, identity = _shard_layout()
        self.assertEqual(members, [9, 3, 12, 0, 5, 1, 14, 7])
        self.assertEqual(
            identity,
            dict(tp_part=2, tp_parts=4, cp_rank=1, cp_parts=2, leader_rank=9),
        )
        rows[1] = [5, 1, 14, 9]
        with self.assertRaises(RuntimeError):
            _shard_layout()

    def test_unsupported_configs_rejected(self):
        config = self._config()
        config.pp_size = 2
        with self.assertRaises(ValueError):
            self.cls(config, _FlatPool(2, 32))
        with mock.patch(f"{MODULE}._prefix_keys_enabled", return_value=False):
            with self.assertRaises(ValueError):
                self.cls(self._config(), _FlatPool(2, 32))


if __name__ == "__main__":
    unittest.main()
