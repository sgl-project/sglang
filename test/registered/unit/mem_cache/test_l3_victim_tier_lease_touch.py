"""Unit tests for the L3 victim tier's lease touch (SGLANG_OPT_HICACHE_L3_VICTIM_TIER).

``LeaseStore`` below models a store whose objects rank for eviction by their
last lease (an object never checked or read ranks by its put time), where a
live lease protects an object.

Covered here, on the served code (lifted by AST, as in test_l3_victim_tier.py):
* ``MooncakeStore.l3vt_touch`` names exactly the objects the stock put path
  writes (target KV as ``batch_exists`` / ``_get_mha_buffer_meta`` name them,
  sidecars and states via ``_get_hybrid_page_component_keys``); counts are right;
* ``HybridCacheController._l3vt_touch_after_put`` runs after the write and
  never fails the op;
* end to end (victim pass -> put -> touch -> ack): without the touch, fresh
  victims die first when lookups lease older objects; with it they survive
  the store's turnover window; bytes read back equal the ground truth;
* the ``assume_stored`` left-extension is off under the victim tier only; with
  the touch off nothing is touched.
"""

import contextlib
import importlib.util
import os
import random
import sys
import textwrap
import unittest
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # standalone run without an sglang install

    def register_cpu_ci(**kwargs):
        pass


register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The fakes of test_l3_victim_tier.py (host pools with real bytes, radix tree
# with a host LRU, the lifted cache methods); its TestCases stay in that module
# and are not rerun.
B = _load("_l3vt_base_fakes", os.path.join(_HERE, "test_l3_victim_tier.py"))
l3vt = B.l3vt
PoolName, PoolTransfer, PoolHitPolicy = B.PoolName, B.PoolTransfer, B.PoolHitPolicy
PAGE = B.PAGE
_SRT = B._SRT


class _NotThisPool:
    pass


_ms = B._lift(
    os.path.join(_SRT, "mem_cache/storage/mooncake_store/mooncake_store.py"),
    [
        "MooncakeStore.l3vt_touch",
        "MooncakeStore._get_hybrid_page_component_keys",
        "MooncakeStore._get_mha_buffer_meta",
        "MooncakeStore._tag_keys",
        "MooncakeStore.batch_exists",
        "MooncakeStore._unified_object_keys",
    ],
    "_l3vt_lease_mooncake",
    {
        "PoolName": PoolName,
        "PoolTransfer": PoolTransfer,
        "List": List,
        "Tuple": Tuple,
        "Optional": Optional,
        "DevicePoolEntry": _NotThisPool,
        "MLATokenToKVPoolHost": _NotThisPool,
        "HiCacheStorageExtraInfo": object,
    },
)
_hcc = B._lift(
    os.path.join(_SRT, "mem_cache/hybrid_cache/hybrid_cache_controller.py"),
    ["HybridCacheController._l3vt_touch_after_put"],
    "_l3vt_lease_hcc",
    {"logger": __import__("logging").getLogger("l3vt-lease-test")},
)


# ---------------------------------------------------------------------------
# Lease-order eviction model
# ---------------------------------------------------------------------------


class LeaseStore:
    """Objects rank for eviction by ``order`` = their put time, or their last
    lease's expiry once checked / read; an object under a live lease is never
    evicted. At the high watermark a round evicts ``ratio`` of the objects,
    lowest ``order`` first among those without a live lease."""

    def __init__(self, capacity, *, wm=0.90, ratio=0.05, ttl=10.0):
        self.capacity = capacity  # in objects (every object one unit here)
        self.wm, self.ratio, self.ttl = wm, ratio, ttl
        self.now = 0.0
        # key -> [payload, order, lease_until, put_time, last_activity]
        self.objs = {}
        self.evicted = 0
        self.refused = 0
        # An object evicted while an evictable one (no live lease) whose last
        # put / check / read is older stays: the order is not LRU.
        self.lru_violations = 0

    def _evict_round(self):
        # Ties in ``order`` (a lease ending exactly at a put time: the test's
        # clock is coarse) go to the older activity, as continuous time would.
        cands = sorted(
            (o[1], o[4], k) for k, o in self.objs.items() if o[2] <= self.now
        )
        n = max(1, int(self.ratio * len(self.objs)))
        out, rest = cands[:n], cands[n:]
        if rest:
            oldest_left = min(self.objs[k][4] for _, _, k in rest)
            self.lru_violations += sum(
                1 for _, _, k in out if self.objs[k][4] > oldest_left
            )
        for _, _, k in out:
            del self.objs[k]
            self.evicted += 1
        return len(out)

    def put(self, key, payload):
        if key in self.objs:
            return 0
        while len(self.objs) + 1 > self.wm * self.capacity:
            if self._evict_round() == 0:
                self.refused += 1
                return -1
        self.objs[key] = [payload, self.now, 0.0, self.now, self.now]
        return 0

    def exists(self, keys):
        out = []
        for k in keys:
            o = self.objs.get(k)
            if o is None:
                out.append(0)
                continue
            o[2] = self.now + self.ttl
            o[1] = max(o[1], o[2])
            o[4] = self.now
            out.append(1)
        return out

    def probe(self, keys):  # batch_probe_key(policy="none"): no lease
        return [1 if k in self.objs else 0 for k in keys]

    def get(self, key):
        o = self.objs.get(key)
        if o is None:
            return None
        self.exists([key])
        return o[0]


# ---------------------------------------------------------------------------
# A Mooncake-like backend carrying the lifted served methods
# ---------------------------------------------------------------------------


class FakeMooncake:
    def __init__(self, store, prefix="qwen"):
        self.store = store
        self.config_prefix = prefix
        self.is_mla_backend = False
        self.should_split_heads = False
        self.mha_suffix = "0_1"
        self.mla_suffix = "1"
        self.unified_suffixes = None
        self.mem_pool_host = SimpleNamespace(
            kv_buffer=object(),
            get_page_buffer_meta=lambda idx: (
                [0] * (2 * (len(idx) // PAGE)),
                [1] * (2 * (len(idx) // PAGE)),
            ),
        )
        self.registered_pools = {
            PoolName.KV: object(),
            PoolName.DRAFT: object(),
            PoolName.MAMBA: SimpleNamespace(
                conv_buffer=[0], temporal_state_elem_size=1
            ),
        }
        self.exist_calls = []
        self.raise_on_touch = False

    def _batch_exist(self, names):
        if self.raise_on_touch:
            raise RuntimeError("master unreachable")
        self.exist_calls.append(list(names))
        return self.store.exists(names)

    def request_context(self, **kwargs):
        return contextlib.nullcontext()

    def _request_context_from_extra_info(self, extra_info):
        return {}

    # The stock write path's names (what batch_set_v1 / _batch_io_v2 put).
    def put_names_kv(self, keys):
        tagged = self._tag_keys(list(keys))
        names, _, _ = self._get_mha_buffer_meta(
            tagged, torch.arange(len(keys) * PAGE, dtype=torch.int64)
        )
        return names

    def put_names_component(self, keys, transfer):
        names, mult = self._get_hybrid_page_component_keys(list(keys), transfer)
        return self._tag_keys(names), mult


for _n in (
    "l3vt_touch",
    "_get_hybrid_page_component_keys",
    "_get_mha_buffer_meta",
    "_tag_keys",
    "batch_exists",
    "_unified_object_keys",
):
    setattr(FakeMooncake, _n, getattr(_ms, _n))


class MooncakeController(B.FakeController):
    """The fake backup thread writing Mooncake-named objects (stock names)
    with real bytes into a LeaseStore, then the served post-put touch."""

    def __init__(self, group, backend, test):
        super().__init__(group, None, test)
        self.backend = backend
        self.storage_backend = backend
        self.touch_results = []
        self.kv_put_log = []  # (page key, store time) of each fresh KV write

    def write_storage(
        self,
        host_indices,
        token_ids,
        hash_value=None,
        prefix_keys=None,
        extra_pools=None,
        lease_touch=False,
    ):
        op = B.FakeOp(host_indices, token_ids, hash_value, prefix_keys, extra_pools)
        op.l3vt_touch = lease_touch
        self.backup_queue.append(op)
        return op.id

    _l3vt_touch_after_put = _hcc._l3vt_touch_after_put

    def run_backup(self, op):
        g = self.mem_pool_host
        be, st = self.backend, self.backend.store
        results = {}
        for t in op.pool_transfers or ():
            if t.indices_from_pool == PoolName.KV:
                t.host_indices = op.host_indices
                t.keys = op.hash_value
            names, mult = be.put_names_component(t.keys, t)
            ok = []
            if t.name == PoolName.MAMBA:
                for i, (slot, key) in enumerate(zip(t.host_indices.tolist(), t.keys)):
                    payload = g.state.read_rows(slot, 1)
                    self.test.assertEqual(payload, B.page_bytes(key, "state"))
                    r = [st.put(n, payload) for n in names[i * mult : (i + 1) * mult]]
                    ok.append(all(x == 0 for x in r))
            elif t.name == PoolName.DRAFT:
                firsts = t.host_indices.reshape(-1, PAGE)[:, 0].tolist()
                for i, (first, key) in enumerate(zip(firsts, t.keys)):
                    payload = g.draft.read_rows(first, PAGE)
                    self.test.assertEqual(payload, B.page_bytes(key, "draft"))
                    r = [st.put(n, payload) for n in names[i * mult : (i + 1) * mult]]
                    ok.append(all(x == 0 for x in r))
            else:
                raise AssertionError(t.name)
            results[t.name] = ok
        op.pool_storage_result.update_extra_pool_hit_pages(
            B._hs.count_pool_hits(results)
        )
        done = 0
        firsts = (
            op.host_indices.reshape(-1, PAGE)[:, 0].tolist()
            if op.host_indices.numel()
            else []
        )
        for first, key in zip(firsts, op.hash_value):
            payload = g.kv.read_rows(first, PAGE)
            self.test.assertEqual(payload, B.page_bytes(key, "kv"))
            fresh = not all(st.probe(be.put_names_kv([key])))
            if any(st.put(n, payload) != 0 for n in be.put_names_kv([key])):
                break
            if fresh:
                self.kv_put_log.append((key, st.now))
            done += PAGE
        op.completed_tokens = done
        # The served backup thread: the touch runs after the write, before the ack.
        if op.l3vt_touch:
            self._l3vt_touch_after_put(op)
            self.touch_results.append(op.l3vt_touch_result)


def make_mc_cache(test, *, touch, capacity, kv_tokens=512, state_slots=16, margin=0.05):
    store = LeaseStore(capacity)
    backend = FakeMooncake(store)
    c = B.make_cache(
        test, victim=True, kv_tokens=kv_tokens, state_slots=state_slots, margin=margin
    )
    c.cache_controller = MooncakeController(c.host_pool_group, backend, test)
    c.store = None
    c._l3vt_touch = touch
    return c, store, backend


def kv_present(backend, key):
    return all(backend.store.probe(backend.put_names_kv([key])))


def state_present(backend, key):
    names, _ = backend.put_names_component([key], l3vt_mamba())
    return all(backend.store.probe(names))


def l3vt_mamba():
    return PoolTransfer(name=PoolName.MAMBA, host_indices=None, keys=None)


# ---------------------------------------------------------------------------


class TestLeaseModel(unittest.TestCase):
    def _replay(self, scen):
        """Put the keys in batches; "touch" checks each batch after its puts,
        "mixed" checks an early prefix once."""
        st = LeaseStore(400)
        keys = [f"k{i:05d}" for i in range(1200)]
        snaps = {}
        for b in range(0, 1200, 20):
            for i in range(b, b + 20):
                st.put(keys[i], b"x")
                st.now += 1 / 16
            if scen == "touch":
                st.exists(keys[b : b + 20])
            if scen == "mixed" and b < 470 <= b + 20:
                st.now += 1.0
                st.exists(keys[:470])
            snaps[b + 20] = [i for i, r in enumerate(st.probe(keys[: b + 20])) if r]
        return snaps

    def test_lease_model_orders_eviction_by_last_lease(self):
        for scen in ("none", "touch"):
            final = self._replay(scen)[1200]
            # the model keeps a newest suffix (FIFO)
            self.assertEqual(final, list(range(final[0], 1200)), scen)
            self.assertGreater(len(final), 320)
        snaps = self._replay("mixed")
        mid = set(snaps[600])
        self.assertTrue(set(range(130, 470)) <= mid)
        self.assertEqual(len(mid & set(range(470, 590))), 0)
        # after the leases expire the checked objects go first, then FIFO
        self.assertTrue(set(range(640, 660)) <= set(snaps[660]))
        final = snaps[1200]
        self.assertEqual(final, list(range(final[0], 1200)))


class TestTouchNaming(unittest.TestCase):
    def test_touch_names_equal_the_put_names(self):
        st = LeaseStore(10_000)
        be = FakeMooncake(st)
        keys = ["a1", "a2", "a3"]
        draft = PoolTransfer(name=PoolName.DRAFT, host_indices=None, keys=keys)
        state = PoolTransfer(name=PoolName.MAMBA, host_indices=None, keys=["a3"])
        written = be.put_names_kv(keys)
        written += be.put_names_component(keys, draft)[0]
        written += be.put_names_component(["a3"], state)[0]
        for n in written:
            st.put(n, b"x")
        res = be.l3vt_touch(keys, [draft, state])
        self.assertEqual(sorted(be.exist_calls[-1]), sorted(written))
        self.assertEqual(res["objects"], len(written))
        self.assertEqual(res["present"], len(written))
        # an object storage no longer holds is counted missing, not created
        st.objs.pop(written[0])
        res = be.l3vt_touch(keys, [draft, state])
        self.assertEqual(
            (res["objects"], res["present"]), (len(written), len(written) - 1)
        )
        self.assertNotIn(written[0], st.objs)
        st.put(written[0], b"x")
        self.assertTrue(all(n.startswith("qwen_") for n in be.exist_calls[-1]))
        # the lookup path names the KV objects the same way
        be.exist_calls.clear()
        self.assertEqual(be.batch_exists(keys), 3)
        self.assertEqual(be.exist_calls[-1], be.put_names_kv(keys))

    def test_touch_grants_leases_and_reorders(self):
        st = LeaseStore(10_000)
        be = FakeMooncake(st)
        for n in be.put_names_kv(["old"]):
            st.put(n, b"x")
        st.now = 5.0
        for n in be.put_names_kv(["new"]):
            st.put(n, b"x")
        be.l3vt_touch(["old"], [])
        o = st.objs[be.put_names_kv(["old"])[0]]
        self.assertEqual(o[2], 15.0)
        self.assertGreater(o[1], st.objs[be.put_names_kv(["new"])[0]][1])


class TestRetentionEndToEnd(unittest.TestCase):
    """Victim pass -> put -> served touch -> ack, against the lease model,
    while lookups keep re-checking older stored objects."""

    def _run(self, *, touch, seed, steps=260):
        rng = random.Random(seed)
        c, st, be = make_mc_cache(self, touch=touch, capacity=950)
        tc = c.tree_core
        cc = c.cache_controller
        for step in range(steps):
            st.now = float(step)
            n = tc.add_child(
                tc.root.id,
                [rng.randrange(10**6) for _ in range(PAGE * rng.randint(2, 5))],
            )
            B.land(c, n)
            c._l3vt_maintain()
            B.drain_backups(c)
            # lookups: a served lookup exists-checks its whole chain (KV, draft
            # and state objects, batch_exists_v2), and storage-hit retries
            # repeat it; here each stored object older than 4 s is re-checked
            # with probability 0.25 a second (half a second after the puts, so
            # no lease ends exactly at a put time).
            st.now = step + 0.5
            st.exists(
                [
                    k
                    for k, o in st.objs.items()
                    if o[3] < step - 4 and rng.random() < 0.25
                ]
            )
        # bytes: every stored KV / draft / state object equals its ground truth
        for name, o in st.objs.items():
            base = name[len("qwen_") :].split("_")[0]
            if name.endswith("_k") or name.endswith("_v"):
                tag = "draft" if "_draft_" in name else "kv"
            else:
                tag = "state"
            self.assertEqual(o[0], B.page_bytes(base, tag), name)
        self.assertGreater(len(cc.kv_put_log), 200)
        return c, st

    def test_touch_makes_eviction_lru(self):
        """Without the touch, a fresh victim ranks below every object checked
        in the 10 s before its put, so it is evicted while objects with older
        activity stay. With the touch every
        object ranks by its last activity: eviction is LRU."""
        for seed in range(3):
            c_off, st_off = self._run(touch=False, seed=seed)
            c_on, st_on = self._run(touch=True, seed=seed)
            self.assertGreater(st_off.evicted, 1000, seed)
            self.assertGreater(st_on.evicted, 1000, seed)
            self.assertGreater(st_off.lru_violations, 100, seed)
            self.assertEqual(st_on.lru_violations, 0, seed)
            self.assertEqual(st_on.refused, 0, seed)
            vt = c_on._l3vt
            self.assertGreater(vt.stats["touch_ops"], 0)
            self.assertEqual(vt.stats["touch_ops"], vt.stats["acked_ops"])
            # An op's own objects can still be evicted between its put and its
            # touch when everything else is leased; the counter reports it.
            self.assertLess(vt.stats["touch_missing"], vt.stats["touch_objects"] // 4)
            self.assertGreater(vt.stats["touch_present"], 0)
            self.assertEqual(c_off._l3vt.stats["touch_ops"], 0)

    def test_fresh_victims_die_first_while_old_ones_are_leased(self):
        """A 'mixed' run through the served path: victims A stored,
        all of A checked once (a lookup sweep), then victims B put while A is
        leased. Without the touch every B but the newest is evicted inside
        the window; with it B is never evicted ahead of A."""
        out = {}
        for touch in (False, True):
            rng = random.Random(9)
            c, st, be = make_mc_cache(self, touch=touch, capacity=400)
            tc = c.tree_core
            cc = c.cache_controller
            t = 0.0
            while st.evicted == 0:  # A: fill the store to its watermark
                t += 0.25
                st.now = t
                n = tc.add_child(
                    tc.root.id, [rng.randrange(10**6) for _ in range(PAGE * 2)]
                )
                B.land(c, n)
                c._l3vt_maintain()
                B.drain_backups(c)
            st.now = t + 11.0  # every A lease from its own touch has expired
            a_keys = list(st.objs)
            st.exists(a_keys)  # the lookup sweep
            n_a_kv = len(cc.kv_put_log)
            for i in range(24):  # B, inside the sweep's lease window
                st.now = t + 11.0 + 0.25 * (i + 1)
                n = tc.add_child(
                    tc.root.id, [rng.randrange(10**6) for _ in range(PAGE * 2)]
                )
                B.land(c, n)
                c._l3vt_maintain()
                B.drain_backups(c)
            b_keys = [k for k, _ in cc.kv_put_log[n_a_kv:]]
            out[touch] = (
                sum(kv_present(be, k) for k in b_keys),
                len(b_keys),
                sum(1 for k in a_keys if k in st.objs) / len(a_keys),
                st.refused,
                c._l3vt.stats["put_kv_pages_failed"],
            )
        b_off, nb_off, a_off, ref_off, fail_off = out[False]
        b_on, nb_on, a_on, ref_on, fail_on = out[True]
        self.assertGreater(nb_off, 10)
        self.assertLessEqual(b_off, 2)  # B die right after their put ...
        self.assertGreater(a_off, 0.9)  # ... while the leased A stay
        self.assertEqual((ref_off, fail_off), (0, 0))  # and nothing says so
        # With the touch a stored B is leased like A, so B is never evicted
        # ahead of A; with the whole store leased the store refuses the put
        # instead, and the victim tier counts it. (The touch cannot make room
        # in a store that lookups keep entirely leased.)
        self.assertGreater(a_on, 0.9)
        self.assertGreater(ref_on, 0)
        self.assertGreater(fail_on, 0)


class TestTouchFailureAndInertness(unittest.TestCase):
    def test_touch_failure_never_fails_the_put(self):
        _hcc.logger.disabled = True  # the expected warnings, one per op
        self.addCleanup(setattr, _hcc.logger, "disabled", False)
        c, st, be = make_mc_cache(self, touch=True, capacity=10_000)
        be.raise_on_touch = True
        tc = c.tree_core
        rng = random.Random(1)
        for _ in range(60):
            n = tc.add_child(
                tc.root.id, [rng.randrange(10**6) for _ in range(PAGE * 3)]
            )
            B.land(c, n)
            c._l3vt_maintain()
            B.drain_backups(c)
        vt = c._l3vt
        self.assertGreater(vt.stats["put_ops"], 0)
        self.assertEqual(vt.stats["touch_failed"], vt.stats["acked_ops"])
        self.assertEqual(vt.stats["put_kv_pages_failed"], 0)
        self.assertEqual(vt.held_kv_tokens, 0)

    def test_no_touch_when_off(self):
        c, st, be = make_mc_cache(self, touch=False, capacity=10_000)
        tc = c.tree_core
        rng = random.Random(2)
        for _ in range(60):
            n = tc.add_child(
                tc.root.id, [rng.randrange(10**6) for _ in range(PAGE * 3)]
            )
            B.land(c, n)
            c._l3vt_maintain()
            B.drain_backups(c)
        self.assertGreater(c._l3vt.stats["put_ops"], 0)
        self.assertEqual(be.exist_calls, [])
        self.assertTrue(all(r is None for r in c.cache_controller.touch_results))
        self.assertEqual(c.cache_controller.touch_results, [])


def _assume_stored_fn():
    """The served lines of prefetch_from_storage that decide the assume_stored
    left-extension, as a function (lifted from the source text)."""
    path = os.path.join(_SRT, "mem_cache/unified_radix_cache.py")
    src = open(path).read()
    a = src.index("        storage_start = len(matched_prefix_tokens or [])")
    b = src.index("        # Key the span by the request's namespace", a)
    body = textwrap.dedent(src[a:b])
    fn = (
        "def f(self, matched_prefix_tokens, new_input_tokens, storage_hit_end):\n"
        + textwrap.indent(body, "    ")
        + "    return assume_stored, new_input_tokens\n"
    )
    ns = {}
    exec(compile(fn, path, "exec"), ns)
    return ns["f"]


class TestMargins(unittest.TestCase):
    def test_kv_and_state_margins_are_separate(self):
        vt = l3vt.VictimTier(
            page_size=64,
            kv_capacity_tokens=6_400_000,
            state_capacity_slots=256,
            margin=0.02,
            state_margin=0.05,
        )
        self.assertEqual(vt.kv_target, 128_000)
        self.assertEqual(vt.state_target, 12)
        self.assertEqual(vt.held_room(), (128_000, 12))
        # a single margin still works (state margin follows it)
        vt1 = l3vt.VictimTier(
            page_size=64,
            kv_capacity_tokens=6_400_000,
            state_capacity_slots=256,
            margin=0.05,
        )
        self.assertEqual((vt1.kv_target, vt1.state_target), (320_000, 12))


class TestAssumeStored(unittest.TestCase):
    def test_left_extension_off_under_the_victim_tier_only(self):
        f = _assume_stored_fn()
        tokens = list(range(256))

        def cache(vt):
            return SimpleNamespace(
                tree_core=SimpleNamespace(is_eagle=False),
                _l3vt=vt,
                _l3vt_stats={"assume_stored_off": 0},
            )

        stock = cache(None)
        a, toks = f(stock, list(range(64)), tokens, 192)
        self.assertTrue(a)
        self.assertEqual(len(toks), 128)
        vic = cache(object())
        a, toks = f(vic, list(range(64)), tokens, 192)
        self.assertFalse(a)
        self.assertEqual(len(toks), 256)
        self.assertEqual(vic._l3vt_stats["assume_stored_off"], 1)
        # no prior hit: nothing to extend in either arm, no count
        a, _ = f(vic, list(range(64)), tokens, None)
        self.assertFalse(a)
        self.assertEqual(vic._l3vt_stats["assume_stored_off"], 1)


if __name__ == "__main__":
    unittest.main()
