"""radixshmem backend: slots hold whole MHA pages and Mamba states in a
TP-independent layout, so engines with different TP sizes share them."""

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
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

try:
    import radixshmem
    from radixshmem import RadixServer, ServerConfig

    HAVE_RADIXSHMEM = hasattr(radixshmem, "_data")
except ImportError:
    HAVE_RADIXSHMEM = False

PAGE, LAYERS, HEAD_DIM, KV_HEADS = 4, 3, 8, 4


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
        t.join(60)
    if errors:
        raise errors[0]
    return out


class _MHAHost:
    """kv_buffer laid out like MHATokenToKVPoolHost."""

    def __init__(self, layout, heads, pages=8):
        self.layout, self.head_num, self.page_size = layout, heads, PAGE
        self.layer_num, self.head_dim = LAYERS, HEAD_DIM
        size = pages * PAGE
        dims = {
            "page_first": (2, size, LAYERS, heads, HEAD_DIM),
            "layer_first": (2, LAYERS, size, heads, HEAD_DIM),
            "page_first_direct": (2, pages, LAYERS, PAGE, heads, HEAD_DIM),
            "page_head": (2, pages, heads, PAGE, LAYERS, HEAD_DIM),
        }[layout]
        self.kv_buffer = torch.zeros(dims, dtype=torch.float32)

    def page(self, p):
        t = p * PAGE
        kv = self.kv_buffer
        if self.layout == "page_first":
            return kv[:, t : t + PAGE].permute(0, 2, 1, 3, 4)
        if self.layout == "layer_first":
            return kv[:, :, t : t + PAGE]
        if self.layout == "page_first_direct":
            return kv[:, p]
        return kv[:, p].permute(0, 3, 2, 1, 4)

    def get_page_buffer_meta(self, indices):
        assert self.layout == "page_first"
        kv, ptrs = self.kv_buffer, []
        for t in indices[:: self.page_size].tolist():
            ptrs += [kv[0, t].data_ptr(), kv[1, t].data_ptr()]
        size = self.page_size * kv[0, 0].numel() * kv.element_size()
        return ptrs, [size] * len(ptrs)


class _MambaHost:
    """temporal + one conv buffer laid out like MambaPoolHost (page_first)."""

    def __init__(self, heads, conv_dim, slots=4, side=0, layout="page_first"):
        self.layout, self.page_size = layout, 1
        self.temporal_buffer = torch.zeros(slots, 2, 1, heads, 2, 2)
        self.conv_buffer = [torch.zeros(slots, 2, 1, conv_dim, 3)]
        self.slot_state_buffers = [torch.zeros(slots, side, 5)] if side else []
        self.slot_state_device_tensors = list(self.slot_state_buffers)


def _global_kv():
    g = torch.arange(2 * LAYERS * PAGE * KV_HEADS * HEAD_DIM, dtype=torch.float32)
    return g.view(2, LAYERS, PAGE, KV_HEADS, HEAD_DIM)


def _keys(n, seed="k"):
    return [hashlib.sha256(f"{seed}{i}".encode()).hexdigest() for i in range(n)]


def _info(prefix=()):
    return HiCacheStorageExtraInfo(prefix_keys=list(prefix))


@unittest.skipUnless(HAVE_RADIXSHMEM, "radixshmem with a data plane is not installed")
class TestRadixShmemLayout(CustomTestCase):
    def setUp(self):
        from sglang.srt.mem_cache.storage.radixshmem import RadixShmemStorage

        tag = uuid.uuid4().hex[:8]
        self.name, self.endpoint = f"/rsl_{tag}", f"unix:///tmp/rsl_{tag}.sock"
        self.server = RadixServer(
            ServerConfig(
                name=self.name,
                endpoint=self.endpoint,
                data_bytes=8 << 20,
                mamba_ratio=0.25,
                slot_align=64,
                prefault=False,
                background_evict_ratio=0.0,
            )
        ).start()
        self.cls = RadixShmemStorage
        patches = [
            mock.patch(
                "sglang.srt.mem_cache.pool_host.mha.MHATokenToKVPoolHost", _MHAHost
            ),
            mock.patch(
                "sglang.srt.mem_cache.pool_host.mha.AsymmetricMHATokenToKVPoolHost",
                type("_Never", (), {}),
            ),
            mock.patch(
                "sglang.srt.mem_cache.pool_host.mamba.MambaPoolHost", _MambaHost
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.server.close()

    def _ranks(self, tp, make_pool, extra=None, cp=1):
        size = tp * cp
        write, read = _Group(size), _Group(size)
        ranks = []
        for r in range(size):
            config = HiCacheStorageConfig(
                tp_rank=r % tp,
                tp_size=tp,
                pp_rank=0,
                pp_size=1,
                attn_cp_rank=r // tp,
                attn_cp_size=cp,
                is_mla_model=False,
                enable_storage_metrics=False,
                is_page_first_layout=True,
                model_name="m",
                extra_config={
                    "radixshmem_name": self.name,
                    "radixshmem_endpoint": self.endpoint,
                    "radixshmem_kv_heads": KV_HEADS,
                    **(extra or {}),
                },
            )
            pool = make_pool(r % tp)
            storage = self.cls(config, pool)
            if size > 1:
                storage.set_shard_groups(
                    write.member(r), read.member(r), **_identity(r, tp, cp)
                )
            storage.register_mem_pool_host(pool)
            ranks.append((storage, pool))
        return ranks

    def _fill(self, ranks, tp, page):
        heads = KV_HEADS // tp
        g = _global_kv()
        for r, (_, pool) in enumerate(ranks):
            part = r % tp
            pool.page(page).copy_(g[..., part * heads : (part + 1) * heads, :])

    def _check(self, ranks, tp, page):
        heads = KV_HEADS // tp
        g = _global_kv()
        for r, (_, pool) in enumerate(ranks):
            part = r % tp
            want = g[..., part * heads : (part + 1) * heads, :]
            self.assertTrue(torch.equal(pool.page(page), want), f"rank {r}")

    def _set(self, ranks, keys):
        idx = torch.arange(len(keys) * PAGE)
        return _run([lambda s=s: s.batch_set_v1(keys, idx, _info()) for s, _ in ranks])

    def _get(self, ranks, keys):
        idx = torch.arange(len(keys) * PAGE)
        return _run([lambda s=s: s.batch_get_v1(keys, idx, _info()) for s, _ in ranks])

    def test_tp2_writes_tp4_and_tp1_read(self):
        keys = _keys(1)
        tp2 = self._ranks(2, lambda _: _MHAHost("page_first", KV_HEADS // 2))
        self._fill(tp2, 2, 0)
        self.assertEqual(self._set(tp2, keys), [[True]] * 2)
        self.assertEqual(tp2[0][0].slot_bytes, _global_kv().numel() * 4)

        tp4 = self._ranks(4, lambda _: _MHAHost("page_head", KV_HEADS // 4))
        self.assertEqual(tp4[0][0].batch_exists(keys, _info()), 1)
        self.assertEqual(self._get(tp4, keys), [[True]] * 4)
        self._check(tp4, 4, 0)

        tp1 = self._ranks(1, lambda _: _MHAHost("layer_first", KV_HEADS))
        self.assertEqual(self._get(tp1, keys), [[True]])
        self._check(tp1, 1, 0)

    def test_all_host_layouts_round_trip(self):
        for i, layout in enumerate(
            ["page_first", "layer_first", "page_first_direct", "page_head"]
        ):
            keys = _keys(2, seed=layout)
            src = self._ranks(2, lambda _: _MHAHost(layout, KV_HEADS // 2))
            self._fill(src, 2, 1)
            idx = torch.arange(2 * PAGE)
            got = _run([lambda s=s: s.batch_set_v1(keys, idx, _info()) for s, _ in src])
            self.assertEqual(got, [[True, True]] * 2, layout)
            dst = self._ranks(2, lambda _: _MHAHost("page_first", KV_HEADS // 2))
            got = _run([lambda s=s: s.batch_get_v1(keys, idx, _info()) for s, _ in dst])
            self.assertEqual(got, [[True, True]] * 2, layout)
            self._check(dst, 2, 1)

    def test_replicated_kv_heads_written_once(self):
        keys = _keys(1)
        heads_global = 2

        tp4 = self._ranks(
            4,
            lambda _: _MHAHost("page_first", 1),
            extra={"radixshmem_kv_heads": heads_global},
        )
        for r, (_, pool) in enumerate(tp4):
            pool.page(0).fill_(r // 2 + 1 if r % 2 == 0 else -1)
        self.assertEqual(self._set(tp4, keys), [[True]] * 4)
        tp1 = self._ranks(
            1,
            lambda _: _MHAHost("page_first", heads_global),
            extra={"radixshmem_kv_heads": heads_global},
        )
        self.assertEqual(self._get(tp1, keys), [[True]])
        page = tp1[0][1].page(0)
        self.assertTrue(torch.all(page[..., 0, :] == 1))
        self.assertTrue(torch.all(page[..., 1, :] == 2))

    def test_mamba_state_across_tp(self):
        groups = [4, 4, 8]
        keys = _keys(2)

        def kv(_):
            return _MHAHost("page_first", KV_HEADS // 2)

        def full_state():
            temporal = torch.arange(1, 1 + 2 * 4 * 2 * 2, dtype=torch.float32)
            conv = torch.arange(1000, 1000 + 2 * 16 * 3, dtype=torch.float32)
            return temporal.view(2, 1, 4, 2, 2), conv.view(2, 1, 16, 3)

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

        temporal, conv = full_state()
        writers = []
        for tp, readers in ((2, False), (1, True)):
            ranks = self._ranks(
                tp, kv if tp == 2 else (lambda _: _MHAHost("page_first", KV_HEADS))
            )
            mambas = [_MambaHost(4 // tp, 16 // tp) for _ in range(tp)]
            for (s, _), m in zip(ranks, mambas):
                s.describe_pool(
                    PoolName.MAMBA,
                    SimpleNamespace(conv_shard_groups=groups, conv_slice_axis=0),
                )
                s.register_mem_host_pool_v2(m, PoolName.MAMBA)
            writers.append((ranks, mambas, tp))

        ranks, mambas, tp = writers[0]
        for r, m in enumerate(mambas):
            m.temporal_buffer[2].copy_(shard(temporal, tp, r))
            m.conv_buffer[0][2].copy_(shard(conv, tp, r, groups))
        idx = torch.arange(2 * PAGE)
        self.assertEqual(
            _run([lambda s=s: s.batch_set_v1(keys, idx, _info()) for s, _ in ranks]),
            [[True, True]] * 2,
        )
        state = lambda: PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([2]),
            keys=keys[1:],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        got = _run(
            [lambda s=s: s.batch_set_v2([state()], _info(keys[:1])) for s, _ in ranks]
        )
        self.assertEqual(got, [{PoolName.MAMBA: [True]}] * 2)

        readers, rmambas, _ = writers[1]
        readers[0][0].batch_exists(keys, _info())
        self.assertEqual(
            readers[0][0].pools[PoolName.MAMBA].salt(0),
            ranks[0][0].pools[PoolName.MAMBA].salt(0),
        )
        got = readers[0][0].batch_get_v2([state()], _info(keys[:1]))
        self.assertEqual(got, {PoolName.MAMBA: [True]})
        self.assertTrue(torch.equal(rmambas[0].temporal_buffer[2], temporal))
        self.assertTrue(torch.equal(rmambas[0].conv_buffer[0][2], conv))

    def test_mamba_side_state_is_replicated(self):
        keys = _keys(2)
        ranks = self._ranks(2, lambda _: _MHAHost("page_first", KV_HEADS // 2))
        mambas = [_MambaHost(2, 8, side=3) for _ in ranks]
        for r, ((s, _), m) in enumerate(zip(ranks, mambas)):
            s.describe_pool(PoolName.MAMBA, SimpleNamespace(conv_shard_groups=None))
            s.register_mem_host_pool_v2(m, PoolName.MAMBA)
            m.temporal_buffer[1].fill_(r + 1)
            m.slot_state_buffers[0][1].fill_(7 if r == 0 else -1)
        idx = torch.arange(2 * PAGE)
        _run([lambda s=s: s.batch_set_v1(keys, idx, _info()) for s, _ in ranks])
        state = lambda: PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([1]),
            keys=keys[1:],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        got = _run(
            [lambda s=s: s.batch_set_v2([state()], _info(keys[:1])) for s, _ in ranks]
        )
        self.assertEqual(got, [{PoolName.MAMBA: [True]}] * 2)
        for m in mambas:
            m.temporal_buffer.zero_()
            m.slot_state_buffers[0].zero_()
        got = _run(
            [lambda s=s: s.batch_get_v2([state()], _info(keys[:1])) for s, _ in ranks]
        )
        self.assertEqual(got, [{PoolName.MAMBA: [True]}] * 2)
        for r, m in enumerate(mambas):
            self.assertTrue(torch.all(m.temporal_buffer[1] == r + 1))
            self.assertTrue(torch.all(m.slot_state_buffers[0][1] == 7))

    def test_mamba_side_state_unplaceable_layout_rejected(self):
        tp1 = self._ranks(1, lambda _: _MHAHost("page_first", KV_HEADS))
        bad = _MambaHost(4, 16, side=3)
        bad.slot_state_buffers = []
        tp1[0][0].register_mem_host_pool_v2(bad, PoolName.MAMBA)
        with self.assertRaises(ValueError):
            tp1[0][0].batch_exists(_keys(1), _info())

    def test_unknown_shape_keeps_rank_parts(self):
        tp2 = self._ranks(
            2,
            lambda _: _MHAHost("page_first", KV_HEADS // 2),
            extra={"radixshmem_kv_heads": 3},
        )
        tp2[0][0].batch_exists(_keys(1), _info())
        pool = tp2[0][0].pools[PoolName.KV]
        self.assertIsNone(pool.layout)


if __name__ == "__main__":
    unittest.main()
