"""Unit tests for the L3 victim tier's chain close (SGLANG_OPT_HICACHE_L3_VICTIM_TIER).

Without chain close a lookup from the root (no state on the request's host
path) can miss page 0: the victim tier writes a page only when it leaves the
host tier, so while the start of a session's path stays in host and its GDN
states (and deepest pages) leave, storage holds the stop point's state and the
deepest pages, not the chain from the root, and the served fold
(``MooncakeStore.batch_exists_v2``) gives 0.

With chain close, the victim put that carries such a state (or such deepest
pages) also writes every host-resident node on the root path through the
stock ``write_backup_storage``. Checked on the served code (lifted by AST, as
in the other victim-tier tests; the ack drain is the served ``_drain_backup``
body, lifted from the source text):

* the miss reproduced without chain close, fixed with it: the lookup from the root
  folds to the deepest stored stop point, and every KV / draft page and the
  state read back equal their ground truth;
* off = inert: no path walk, no extra write, no extra existence call;
* no object is written twice (stored objects are skipped);
* host locks are released at the ack, also when the put fails or the backup
  raises; nothing stays locked or in flight;
* the lease touch rides on the close ops; the victim margins and the
  in-flight cap hold under pressure; refresh re-closes after the window;
* the cap defers and the queue bound drops with counters, never silently;
* split fragments resolve once seen; an unseen one is counted;
* two ranks with different put timing stay identical.
"""

import importlib.util
import os
import random
import sys
import unittest
from collections import defaultdict
from types import SimpleNamespace
from typing import List, Optional

import torch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # standalone run without an sglang install

    def register_cpu_ci(**kwargs):
        pass


register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LT = _load(
    "_l3vt_lease_fakes", os.path.join(_HERE, "test_l3_victim_tier_lease_touch.py")
)
B = LT.B
l3vt = B.l3vt
PoolName, PoolTransfer, PoolHitPolicy = B.PoolName, B.PoolTransfer, B.PoolHitPolicy
PoolTransferResult = B.PoolTransferResult
PAGE = B.PAGE
_SRT = B._SRT
_URC = os.path.join(_SRT, "mem_cache/unified_radix_cache.py")

_CC_METHODS = [
    "UnifiedRadixCache._l3cc_alive",
    "UnifiedRadixCache._l3cc_note_record",
    "UnifiedRadixCache._l3cc_refresh_pages",
    "UnifiedRadixCache._l3cc_after_victims",
    "UnifiedRadixCache._l3cc_close",
    "UnifiedRadixCache._l3cc_note_ack",
    "UnifiedRadixCache._l3cc_build",
]
_cc = B._lift(
    _URC,
    _CC_METHODS,
    "_l3vt_chain_urc",
    {
        "Optional": Optional,
        "NodeId": int,
        "PoolName": PoolName,
        "ChainCloser": l3vt.ChainCloser,
        "L3VictimTierConfig": l3vt.L3VictimTierConfig,
    },
)
for _name in _CC_METHODS:
    _m = _name.split(".", 1)[1]
    setattr(B.FakeCache, _m, getattr(_cc, _m))

# The served read path's fold.
_ms3 = B._lift(
    os.path.join(_SRT, "mem_cache/storage/mooncake_store/mooncake_store.py"),
    ["MooncakeStore.batch_exists_v2"],
    "_l3vt_chain_mooncake",
    {
        "PoolName": PoolName,
        "PoolHitPolicy": PoolHitPolicy,
        "PoolTransferResult": PoolTransferResult,
        "PoolTransfer": PoolTransfer,
        "List": List,
        "Optional": Optional,
        "HiCacheStorageExtraInfo": object,
    },
)
LT.FakeMooncake.batch_exists_v2 = _ms3.batch_exists_v2


_make_drain = B._make_drain
_drain_queue = B._drain_queue


# ---------------------------------------------------------------------------
# Fakes: a tree with paths, liveness and splits; a failing store
# ---------------------------------------------------------------------------


class CCTree(B.FakeTreeCore):
    def __init__(self, cache):
        super().__init__(cache)
        self.calls = defaultdict(int)

    def is_backuped(self, nid):
        self.calls["is_backuped"] += 1
        return self.nodes[nid].host_value is not None  # KeyError once gone

    def collect_path_info(self, ids):
        self.calls["collect_path_info"] += 1
        out = []
        for nid in ids:
            n = self.nodes[nid]
            chain = []
            while n is not self.root:
                chain.append(n)
                n = n.parent
            path, start = [], 0
            for m in reversed(chain):
                end = start + len(m.tokens)
                path.append(
                    SimpleNamespace(
                        node_id=m.id,
                        parent_id=m.parent.id,
                        path_start=start,
                        path_end=end,
                        full_value_len=len(m.tokens) if m.device else None,
                        num_children=len(m.children),
                    )
                )
                start = end
            out.append(tuple(path))
        return tuple(out)

    def split(self, nid, k_pages):
        """Both cores: the new node takes the prefix, the child keeps its id."""
        child = self.nodes[nid]
        assert 0 < k_pages < len(child.keys) and child.host_lock == 0
        k = k_pages * PAGE
        new = B.Node(self.next_id, child.parent, child.tokens[:k], child.keys[:k_pages])
        self.next_id += 1
        new.child_key = new.id
        new.device = child.device
        new.last_access = child.last_access
        if child.host_value is not None:
            new.host_value = child.host_value[:k].clone()
            child.host_value = child.host_value[k:].clone()
        del child.parent.children[child.child_key]
        child.parent.children[new.child_key] = new
        new.children[child.child_key] = child
        child.parent = new
        child.tokens = child.tokens[k:]
        child.keys = child.keys[k_pages:]
        self.nodes[new.id] = new
        return new


class FailingLeaseStore(LT.LeaseStore):
    def __init__(self, capacity, fail_rate=0.0, seed=3):
        super().__init__(capacity)
        self.fail_rate = fail_rate
        self.rng = random.Random(seed)
        self.fresh_puts = defaultdict(int)

    def put(self, key, payload):
        if (
            key not in self.objs
            and self.fail_rate
            and self.rng.random() < self.fail_rate
        ):
            return -1
        fresh = key not in self.objs
        r = super().put(key, payload)
        if r == 0 and fresh:
            self.fresh_puts[key] += 1
        return r


class RaisingController(LT.MooncakeController):
    """Every n-th backup raises mid-way; the served backup thread logs it
    (``_page_backup_or_log``) and acks the op anyway."""

    raise_every = 0

    def run_backup(self, op):
        self._n = getattr(self, "_n", 0) + 1
        if self.raise_every and self._n % self.raise_every == 0:
            op.completed_tokens = 0
            return
        super().run_backup(op)


def make_cc_cache(
    test,
    *,
    chain_close,
    touch=True,
    capacity=100_000,
    kv_tokens=1024,
    state_slots=8,
    margin=0.02,
    state_margin=0.05,
    cap_pages=None,
    refresh=0.25,
    max_pending=None,
    fail_rate=0.0,
    raise_every=0,
):
    c = B.make_cache(
        test, victim=True, kv_tokens=kv_tokens, state_slots=state_slots, margin=margin
    )
    c._l3vt = l3vt.VictimTier(
        page_size=PAGE,
        kv_capacity_tokens=kv_tokens,
        state_capacity_slots=state_slots,
        margin=margin,
        state_margin=state_margin,
        stats_interval_s=1e9,
    )
    store = FailingLeaseStore(capacity, fail_rate=fail_rate)
    backend = LT.FakeMooncake(store)
    cc = RaisingController(c.host_pool_group, backend, test)
    cc.raise_every = raise_every
    cc.ack_backup_queue = cc.ack_queue
    c.cache_controller = cc
    c.store = None
    c.tree_core = CCTree(c)
    c.inc_host_lock_ref = c.tree_core.inc_host_lock_ref
    c.dec_host_lock_ref = c.tree_core.dec_host_lock_ref
    c._l3vt_touch = touch
    c.enable_storage_metrics = False
    c.storage_metrics_collector = None
    if chain_close:
        cfg = l3vt.L3VictimTierConfig(
            victim=True, chain_close=True, chain_close_refresh=refresh
        )
        c._l3cc = c._l3cc_build(cfg, None, None)
        if cap_pages is not None:
            c._l3cc.cap_pages = cap_pages
        if max_pending is not None:
            c._l3cc.max_pending = max_pending
    return c, store, backend


def drain(cache, n=None):
    """Backup thread for up to n queued ops, then the served ack drain."""
    cc = cache.cache_controller
    take = len(cc.backup_queue) if n is None else min(n, len(cc.backup_queue))
    for _ in range(take):
        op = cc.backup_queue.pop(0)
        cc.run_backup(op)
        cc.ack_queue.append(op)
    k = len(cc.ack_queue)
    _make_drain(cache, cc, k, False, _drain_queue)()
    return k


def settle(cache, rounds=50):
    for _ in range(rounds):
        cache._l3vt_maintain()
        if (
            not cache.cache_controller.backup_queue
            and not cache.cache_controller.ack_queue
        ):
            if cache._l3cc is None or (
                not cache._l3cc.queue and not cache._l3cc.inflight
            ):
                break
        drain(cache)


def lookup_pages(backend, keys):
    """The served fold over a lookup from the root: KV prefix, then the draft
    (all pages) and GDN state (trailing page) stop points."""
    res = backend.batch_exists_v2(
        list(keys),
        [
            PoolTransfer(
                name=PoolName.DRAFT, keys=["x"], hit_policy=PoolHitPolicy.ALL_PAGES
            ),
            PoolTransfer(
                name=PoolName.MAMBA, keys=["x"], hit_policy=PoolHitPolicy.TRAILING_PAGES
            ),
        ],
    )
    return res.kv_hit_pages


def path_keys(tc, node):
    keys = []
    while node is not tc.root:
        keys = node.keys + keys
        node = node.parent
    return keys


def check_restore_bytes(test, backend, keys, stop):
    st = backend.store
    for k in keys[:stop]:
        for n in backend.put_names_kv([k]):
            test.assertEqual(st.objs[n][0], B.page_bytes(k, "kv"), n)
        dn, _ = backend.put_names_component(
            [k], PoolTransfer(name=PoolName.DRAFT, host_indices=None, keys=None)
        )
        for n in dn:
            test.assertEqual(st.objs[n][0], B.page_bytes(k, "draft"), n)
    sn, _ = backend.put_names_component([keys[stop - 1]], LT.l3vt_mamba())
    for n in sn:
        test.assertEqual(st.objs[n][0], B.page_bytes(keys[stop - 1], "state"), n)


def host_locks(cache):
    return sum(n.host_lock for n in cache.tree_core.nodes.values())


def add_path(cache, parent, sizes, states):
    tc = cache.tree_core
    out = []
    for size, with_state in zip(sizes, states):
        rng = random.Random(len(tc.nodes) * 7919 + size)
        n = tc.add_child(parent.id, [rng.randrange(10**6) for _ in range(PAGE * size)])
        B.land(cache, n, with_state=with_state)
        out.append(n)
        parent = n
    return out


def touch_path(tc, node):
    while node is not None and node is not tc.root:
        node.last_access = tc.tick()
        node = node.parent


# ---------------------------------------------------------------------------


class TestRootLookupChain(unittest.TestCase):
    """Session S = n1 (6 pages) -> n2 (4, state) -> n3 (3, state). Other
    sessions' landings, each with a state, churn the 8-slot host state pool:
    n2's state is tombstoned (n2 internal: put detached), n3 leaves whole
    (leaf: KV + state put). The KV pool is large, so n1 / n2 stay in host
    with no state: the host match is the root, and the lookup runs from the
    root over n1 + n2 + n3."""

    def _run(self, chain_close):
        c, st, be = make_cc_cache(self, chain_close=chain_close)
        tc = c.tree_core
        n1, n2, n3 = add_path(c, tc.root, [6, 4, 3], [False, True, True])
        settle(c)
        for i in range(24):
            add_path(c, tc.root, [1], [True])
            c._l3vt_maintain()
            drain(c)
        settle(c)
        return c, st, be, (n1, n2, n3)

    def test_root_lookup_miss_reproduced_then_fixed(self):
        c0, st0, be0, (n1, n2, n3) = self._run(chain_close=False)
        tc0 = c0.tree_core
        keys = n1.keys + n2.keys + n3.keys
        # head in host without a state; n3 gone from host
        self.assertIsNotNone(tc0.nodes[n1.id].host_value)
        self.assertIsNotNone(tc0.nodes[n2.id].host_value)
        self.assertIsNone(tc0.nodes[n2.id].state_slot)
        self.assertNotIn(n3.id, tc0.nodes)
        # storage holds the stop points' states and n3's pages, not page 0
        self.assertTrue(LT.state_present(be0, n2.keys[-1]))
        self.assertTrue(LT.state_present(be0, n3.keys[-1]))
        self.assertTrue(LT.kv_present(be0, n3.keys[0]))
        self.assertFalse(LT.kv_present(be0, n1.keys[0]))
        self.assertEqual(lookup_pages(be0, keys), 0)  # the zero read
        self.assertEqual(c0.tree_core.calls["collect_path_info"], 0)

        c1, st1, be1, (m1, m2, m3) = self._run(chain_close=True)
        keys1 = m1.keys + m2.keys + m3.keys
        self.assertEqual(keys1, keys)  # same token stream, same keys
        stop = lookup_pages(be1, keys1)
        self.assertEqual(stop, len(keys1))  # pages 0 .. n3's end + n3's state
        check_restore_bytes(self, be1, keys1, stop)
        # the shallower stop point is restorable too (n2's state + its chain)
        self.assertEqual(
            lookup_pages(be1, m1.keys + m2.keys), len(m1.keys) + len(m2.keys)
        )
        s = c1._l3cc.stats
        self.assertGreaterEqual(s["cc_trig_detached"], 1)
        self.assertGreaterEqual(s["cc_trig_state"], 2)
        self.assertGreaterEqual(
            s["cc_anchor_hint"] + s["cc_anchor_pred"], 1
        )  # n3 left whole
        self.assertGreater(s["cc_ops"], 0)
        self.assertEqual(s["cc_acked_ops"], s["cc_ops"])
        self.assertEqual(s["cc_failed_pages"], 0)
        self.assertEqual(c1._l3cc.inflight_pages, 0)
        self.assertEqual(host_locks(c1), 0)
        # the head stays in host (inclusive copy) and is not written twice
        self.assertIsNotNone(c1.tree_core.nodes[m1.id].host_value)
        self.assertTrue(all(v == 1 for v in st1.fresh_puts.values()))

    def test_deepest_pages_leave_with_the_head_resident(self):
        """n2's state went detached (n1 / n2 closed); later n2's KV leaves
        (n3 gone, n2 a leaf) while n1 stays under a newer branch; the close
        re-runs from n1 and the n2 stop point stays restorable from the root."""
        c, st, be = make_cc_cache(
            self,
            chain_close=True,
            kv_tokens=256,
            state_slots=8,
            margin=0.1,
            refresh=1e-9,
        )
        tc = c.tree_core
        n1, n2, n3 = add_path(c, tc.root, [6, 4, 3], [False, True, True])
        settle(c)
        for i in range(12):  # states churn: n2 detached, n3 leaves
            add_path(c, tc.root, [1], [True])
            c._l3vt_maintain()
            drain(c)
        self.assertGreaterEqual(c._l3cc.stats["cc_trig_detached"], 1)
        branch = add_path(c, n1, [2], [False])[0]  # n1 now shared by a newer path
        for i in range(60):  # KV pressure, stateless landings
            touch_path(tc, branch)
            add_path(c, tc.root, [3], [False])
            c._l3vt_maintain()
            drain(c)
            if n2.id not in tc.nodes:
                break
        settle(c)
        self.assertNotIn(n2.id, tc.nodes)
        self.assertIn(n1.id, tc.nodes)
        self.assertGreaterEqual(c._l3cc.stats["cc_trig_kv"], 1)
        keys = n1.keys + n2.keys
        stop = lookup_pages(be, keys)
        self.assertEqual(stop, len(keys))
        check_restore_bytes(self, be, keys, stop)
        self.assertEqual(host_locks(c), 0)


class TestInertAndConfig(unittest.TestCase):
    def test_off_is_inert(self):
        """Victim tier + touch on, chain close off: no path walk, no
        stock write, no existence call beyond the victim puts' own touches."""
        c, st, be = make_cc_cache(self, chain_close=False)
        tc = c.tree_core
        calls = defaultdict(int)
        orig = c.write_backup_storage

        def counting(*a, **k):
            calls["write_backup_storage"] += 1
            return orig(*a, **k)

        c.write_backup_storage = counting
        rng = random.Random(4)
        for i in range(120):
            parent = tc.root
            cands = [n for n in tc.nodes.values() if n is not tc.root]
            if cands and rng.random() < 0.4:
                parent = rng.choice(cands)
            add_path(c, parent, [rng.randint(1, 4)], [rng.random() < 0.8])
            c._l3vt_maintain()
            drain(c, n=rng.randint(0, 2))
        settle(c)
        self.assertGreater(c._l3vt.stats["put_ops"], 10)
        self.assertEqual(tc.calls["collect_path_info"], 0)
        self.assertEqual(tc.calls["is_backuped"], 0)
        self.assertEqual(calls["write_backup_storage"], 0)
        self.assertEqual(c.ongoing_backup, {})
        # every existence call is a victim op's touch
        self.assertEqual(len(be.exist_calls), c._l3vt.stats["touch_ops"])
        self.assertNotIn("cc_", c._l3vt_stats_suffix())

    def test_config_and_environ(self):
        src = open(os.path.join(_SRT, "environ.py")).read()
        self.assertIn(
            "SGLANG_OPT_HICACHE_L3_VICTIM_TIER_CLOSE_CAP = EnvFloat(0.0)", src
        )
        self.assertIn(
            "SGLANG_OPT_HICACHE_L3_VICTIM_TIER_CLOSE_REFRESH = EnvFloat(0.25)", src
        )
        R = l3vt.L3VictimTierConfig.resolve

        class E:
            def __init__(self, **kv):
                self.kv = kv

            def __getattr__(self, name):
                key = (
                    name.replace("SGLANG_OPT_HICACHE_L3_VICTIM_TIER", "")
                    .lstrip("_")
                    .lower()
                )
                defaults = {
                    "": False,
                    "kv_margin": 0.05,
                    "state_margin": 0.05,
                    "pin_frac": 0.25,
                    "budget_tokens": 0,
                    "wait_slack": 3.0,
                    "close_cap": 0.0,
                    "close_refresh": 0.25,
                }
                return SimpleNamespace(get=lambda: self.kv.get(key, defaults[key]))

        BASE = dict(
            enable_storage=True,
            host_memory_mode="cache",
            has_linker=False,
            is_write_back=True,
            victim_unsupported=None,
            prefetch_stop_policy="timeout",
            has_storage_v2=True,
            has_lease_touch=True,
        )
        on = R(E(**{"": True}), **BASE)
        self.assertTrue(on.victim and on.chain_close and on.any_on)
        wt = R(E(**{"": True}), **{**BASE, "is_write_back": False})
        self.assertFalse(wt.chain_close or wt.any_on)
        bad = R(E(**{"": True, "close_cap": 0.6}), **BASE)
        self.assertTrue(bad.victim)
        self.assertFalse(bad.chain_close)
        self.assertIn("cap", bad.ignored["chain_close"])
        off = R(E(), **BASE)
        self.assertFalse(off.chain_close or off.any_on)
        # The engaged line names the child, then the parts with chain close.
        urc = open(_URC).read()
        self.assertIn('"%s engaged: storage is a victim tier of the host tier "', urc)
        self.assertIn(
            '"(lease_touch=%s chain_close=%s pin=%s budget=%s wait_bound=%s "', urc
        )

    def test_build_sizes(self):
        """Cap = the KV victim margin in pages; refresh = 0.25 x the host pool
        in pages; queue = one anchor a host state slot (>= 64)."""
        c, _, _ = make_cc_cache(self, chain_close=True, kv_tokens=4096, state_slots=200)
        cc = c._l3cc
        self.assertEqual(cc.cap_pages, c._l3vt.kv_target // PAGE)
        self.assertEqual(cc.refresh_pages, int(0.25 * 4096 // PAGE))
        self.assertEqual(cc.max_pending, 200)
        cfg = l3vt.L3VictimTierConfig(
            victim=True, chain_close=True, chain_close_cap=0.1
        )
        self.assertEqual(
            c._l3cc_build(cfg, None, None).cap_pages, int(0.1 * 4096) // PAGE
        )


class TestWorkloads(unittest.TestCase):
    def _workload(
        self, c, rng, steps, *, branch=0.35, split=0.05, drains=(0, 3), check=None
    ):
        tc = c.tree_core
        tips = {s: tc.root for s in range(6)}
        hist = defaultdict(list)
        for step in range(steps):
            s = rng.randrange(6)
            parent = tips[s]
            if hist[s] and rng.random() < branch:
                parent = rng.choice(hist[s])
            if parent is not tc.root and parent.id not in tc.nodes:
                parent = tc.root
            if split and parent is not tc.root and rng.random() < split:
                if (
                    len(parent.keys) > 1
                    and parent.host_lock == 0
                    and parent.host_value is not None
                ):
                    parent = tc.split(parent.id, rng.randrange(1, len(parent.keys)))
            n = add_path(c, parent, [rng.randint(1, 5)], [rng.random() < 0.8])[0]
            touch_path(tc, n)
            tips[s] = n
            hist[s].append(n)
            drain(c, n=rng.randint(*drains))
            c._l3vt_maintain()
            if check:
                check(c, step)
        settle(c)

    def test_no_double_write_and_bytes(self):
        for seed in range(4):
            rng = random.Random(seed)
            c, st, be = make_cc_cache(
                self, chain_close=True, kv_tokens=1024, state_slots=12, margin=0.05
            )
            self._workload(c, rng, 260)
            self.assertGreater(c._l3cc.stats["cc_ops"], 5, seed)
            # unbounded store: every object written once (stored objects skipped)
            self.assertTrue(st.fresh_puts and max(st.fresh_puts.values()) == 1, seed)
            for name, o in st.objs.items():
                base = name[len("qwen_") :].split("_")[0]
                tag = (
                    ("draft" if "_draft_" in name else "kv")
                    if name[-2:] in ("_k", "_v")
                    else "state"
                )
                self.assertEqual(o[0], B.page_bytes(base, tag), name)
            self.assertEqual(host_locks(c), 0)
            self.assertEqual(c.ongoing_backup, {})

    def test_every_stored_stop_point_is_closed_from_the_root(self):
        """With a store that drops nothing, every stored GDN state is
        restorable from the root (KV pages 0..stop and the state, bytes equal),
        except where a page of its chain left the host unput: freed by the
        allocation-driven sync path or dropped at the held-page cap (counted
        as freed_keyed_* / dropped_held_cap_*)."""
        for seed in range(3):
            rng = random.Random(100 + seed)
            c, st, be = make_cc_cache(
                self, chain_close=True, kv_tokens=768, state_slots=10, margin=0.05
            )
            tc = c.tree_core
            paths = {}
            unput = set()  # keys whose host page was freed without a put
            vt = c._l3vt
            clear_kv, release = vt.keymap.clear_kv, vt.release
            in_release = [False]

            def clear_recording(pages):
                if not in_release[0]:
                    unput.update(
                        k for k in map(vt.keymap.kv_key, pages) if k is not None
                    )
                return clear_kv(pages)

            def release_marked(op_id):
                in_release[0] = True
                try:
                    return release(op_id)
                finally:
                    in_release[0] = False

            vt.keymap.clear_kv, vt.release = clear_recording, release_marked

            def remember(cache, step):
                for n in tc.nodes.values():
                    if n is not tc.root and n.host_value is not None:
                        paths[n.keys[-1]] = path_keys(tc, n)

            self._workload(c, rng, 220, split=0.0, check=remember)
            stored = closed = holed = 0
            for last, keys in paths.items():
                if not LT.state_present(be, last):
                    continue
                stored += 1
                if lookup_pages(be, keys) == len(keys):
                    check_restore_bytes(self, be, keys, len(keys))
                    closed += 1
                    continue
                missing = [k for k in keys if not LT.kv_present(be, k)]
                self.assertTrue(missing and set(missing) <= unput, (seed, last[:8]))
                holed += 1
            self.assertGreater(stored, 10, seed)
            self.assertGreater(closed, 3 * holed, seed)
            self.assertEqual(c._l3cc.stats["cc_anchor_none"], 0, seed)

    def test_locks_released_on_ack_and_failure(self):
        rng = random.Random(7)
        c, st, be = make_cc_cache(
            self, chain_close=True, fail_rate=0.15, raise_every=7, margin=0.05
        )
        LT._hcc.logger.disabled = True
        self.addCleanup(setattr, LT._hcc.logger, "disabled", False)
        self._workload(c, rng, 200)
        s = c._l3cc.stats
        self.assertGreater(s["cc_failed_pages"], 0)
        self.assertEqual(s["cc_acked_ops"], s["cc_ops"])
        self.assertEqual(c._l3cc.inflight, {})
        self.assertEqual(c._l3cc.inflight_pages, 0)
        self.assertEqual(c.ongoing_backup, {})
        self.assertEqual(host_locks(c), 0)
        # what was stored is right
        for name, o in st.objs.items():
            base = name[len("qwen_") :].split("_")[0]
            tag = (
                ("draft" if "_draft_" in name else "kv")
                if name[-2:] in ("_k", "_v")
                else "state"
            )
            self.assertEqual(o[0], B.page_bytes(base, tag), name)

    def test_touch_margins_and_cap_under_pressure(self):
        """Margins and the cap hold at every step, the close ops carry the
        lease touch, and against chain close off on the same seeds
        (backup thread keeping up: every queued op written each step) the
        victim tier puts about as much and loses about as much at the sync
        path."""
        tot = {False: [0, 0], True: [0, 0]}
        for seed in range(4):
            for on in (False, True):
                c, st, be = make_cc_cache(
                    self, chain_close=on, kv_tokens=512, state_slots=8, margin=0.1
                )
                vt, cc = c._l3vt, c._l3cc

                def check(cache, step):
                    self.assertLessEqual(vt.held_kv_tokens, vt.kv_target)
                    self.assertLessEqual(vt.held_states, max(vt.state_target, 0))
                    if cc is None:
                        return
                    biggest = max((p for _, p in cc.inflight.values()), default=0)
                    self.assertLessEqual(cc.inflight_pages, max(cc.cap_pages, biggest))
                    locked = sum(
                        len(n.keys)
                        for n in cache.tree_core.nodes.values()
                        if n.host_lock
                    )
                    self.assertLessEqual(locked, max(cc.cap_pages, biggest))

                self._workload(c, random.Random(seed), 240, drains=(8, 8), check=check)
                tot[on][0] += vt.stats["put_kv_pages"]
                tot[on][1] += vt.stats["freed_keyed_kv_pages"]
                if on:
                    s = cc.stats
                    self.assertGreater(s["cc_ops"], 0)
                    self.assertEqual(s["cc_touch_ops"], s["cc_acked_ops"])
                    self.assertEqual(s["cc_touch_failed"], 0)
        self.assertGreaterEqual(tot[True][0], 0.95 * tot[False][0], tot)
        self.assertLessEqual(tot[True][1], 1.10 * tot[False][1], tot)
        # touch off: closes carry no touch
        c2, st2, be2 = make_cc_cache(
            self,
            chain_close=True,
            touch=False,
            kv_tokens=512,
            state_slots=8,
            margin=0.05,
        )
        self._workload(c2, random.Random(11), 120)
        self.assertGreater(c2._l3cc.stats["cc_ops"], 0)
        self.assertEqual(c2._l3cc.stats["cc_touch_ops"], 0)
        self.assertEqual(be2.exist_calls, [])

    def test_refresh_window(self):
        """A closed node is skipped until the victim tier has issued refresh x
        host pool pages since its close, then written again (existence-checked:
        no fresh put)."""
        c, st, be = make_cc_cache(
            self, chain_close=True, kv_tokens=1024, state_slots=8, refresh=0.05
        )
        tc = c.tree_core
        head = add_path(c, tc.root, [8], [False])[0]
        for i in range(80):
            add_path(c, head, [1], [True])
            touch_path(tc, head)
            c._l3vt_maintain()
            drain(c)
        settle(c)
        s = c._l3cc.stats
        self.assertGreater(s["cc_skip_closed"], 0)
        head_issues = s["cc_ops"]
        self.assertGreater(head_issues, 1)
        self.assertTrue(all(v == 1 for v in st.fresh_puts.values()))

    def test_cap_defers_and_queue_drops_are_counted(self):
        rng = random.Random(5)
        c, st, be = make_cc_cache(
            self,
            chain_close=True,
            kv_tokens=1024,
            state_slots=6,
            cap_pages=2,
            max_pending=3,
        )
        self._workload(c, rng, 200, drains=(0, 1))
        s = c._l3cc.stats
        self.assertGreater(s["cc_deferred"] + s["cc_cap_waits"], 0)
        self.assertGreater(s["cc_dropped"], 0)
        # accounting: every queued anchor was closed, dropped or is still queued
        self.assertEqual(
            s["cc_enqueued"], s["cc_closes"] + s["cc_dropped"] + len(c._l3cc.queue)
        )
        self.assertEqual(host_locks(c), 0)
        line = c._l3vt_stats_suffix()
        for k in (
            "cc_dropped=",
            "cc_deferred=",
            "cc_pages=",
            "cc_bytes=",
            "cc_inflight_pages=",
        ):
            self.assertIn(k, line)


class TestFragments(unittest.TestCase):
    def _setup(self, discover):
        c, st, be = make_cc_cache(self, chain_close=True, kv_tokens=1024, state_slots=8)
        tc = c.tree_core
        n1, x = add_path(c, tc.root, [3, 6], [False, True])
        settle(c)
        frag = tc.split(x.id, 2)  # frag = x's first 2 pages (new id); x keeps 4
        branch = tc.add_child(frag.id, [7] * (PAGE * 2))
        if discover:
            B.land(c, branch, with_state=False)  # recorded: its path shows frag
        else:
            branch.device = True  # stays on device, never recorded
        touch_path(tc, branch)
        # x (leaf) leaves host with its state: state pressure
        for i in range(12):
            add_path(c, tc.root, [1], [True])
            c._l3vt_maintain()
            drain(c)
            if x.id not in tc.nodes:
                break
        settle(c)
        return c, st, be, n1, frag, x

    def test_fragment_seen_resolves(self):
        c, st, be, n1, frag, x = self._setup(discover=True)
        self.assertNotIn(x.id, c.tree_core.nodes)
        self.assertIn(frag.id, c.tree_core.nodes)
        s = c._l3cc.stats
        self.assertGreaterEqual(s["cc_refreshed_nodes"], 1)
        self.assertGreaterEqual(s["cc_anchor_pred"], 1)
        self.assertEqual(s["cc_frag_unresolved"], 0)
        keys = n1.keys + frag.keys + x.keys
        self.assertEqual(lookup_pages(be, keys), len(keys))
        check_restore_bytes(self, be, keys, len(keys))

    def test_fragment_unseen_is_counted(self):
        c, st, be, n1, frag, x = self._setup(discover=False)
        self.assertNotIn(x.id, c.tree_core.nodes)
        self.assertGreaterEqual(c._l3cc.stats["cc_frag_unresolved"], 1)


class TestTwoRanks(unittest.TestCase):
    def test_same_decisions(self):
        rng = random.Random(21)
        ranks = [
            make_cc_cache(self, chain_close=True, kv_tokens=768, state_slots=10)[0]
            for _ in range(2)
        ]
        ready = [[], []]
        for step in range(220):
            toks = [rng.randrange(10**6) for _ in range(PAGE * rng.randint(1, 5))]
            choice = rng.random()
            with_state = rng.random() < 0.8
            ids = []
            for c in ranks:
                tc = c.tree_core
                cands = sorted(
                    (n for n in tc.nodes.values() if n is not tc.root),
                    key=lambda n: n.id,
                )
                parent = tc.root
                if cands and choice < 0.35:
                    parent = cands[int(choice * 1000) % len(cands)]
                n = tc.add_child(parent.id, toks)
                touch_path(tc, n)
                B.land(c, n, with_state=with_state)
                ids.append(n.id)
            self.assertEqual(ids[0], ids[1])
            for r, c in enumerate(ranks):
                cc = c.cache_controller
                for _ in range(
                    min(len(cc.backup_queue), [1, 3][r] if step % 2 else [2, 0][r])
                ):
                    op = cc.backup_queue.pop(0)
                    cc.run_backup(op)
                    ready[r].append(op)
            k = min(len(ready[0]), len(ready[1]))
            # The served drain MIN-reduces each op's written prefix over the ranks.
            agreed = torch.tensor(
                [
                    min(
                        min(
                            op.completed_tokens,
                            c._backup_requested_tokens_of(op.id) or 0,
                        )
                        for c, op in ((ranks[r], ready[r][i]) for r in range(2))
                    )
                    for i in range(k)
                ],
                dtype=torch.int64,
            )
            for r, c in enumerate(ranks):
                cc = c.cache_controller
                cc.ack_queue.extend(ready[r][:k])
                ready[r] = ready[r][k:]
                c._all_reduce_storage_drain = lambda data, a=agreed: data.copy_(
                    torch.minimum(data, a)
                )
                _make_drain(c, cc, k, False, _drain_queue)()
            for c in ranks:
                c._l3vt_maintain()
            a, b = ranks
            self.assertEqual(a.tree_core.snapshot(), b.tree_core.snapshot(), step)
            self.assertEqual(list(a._l3cc.queue), list(b._l3cc.queue), step)
            self.assertEqual(
                sorted(a._l3cc.inflight_nodes.items()),
                sorted(b._l3cc.inflight_nodes.items()),
                step,
            )
            self.assertEqual(a._l3cc.stats, b._l3cc.stats, step)
        self.assertGreater(ranks[0]._l3cc.stats["cc_ops"], 0)


class TestChainCloserUnits(unittest.TestCase):
    def test_resolve_pred_hint_none_and_sweep(self):
        cc = l3vt.ChainCloser(
            page_size=4,
            num_kv_pages=16,
            num_state_slots=4,
            cap_pages=8,
            refresh_pages=100,
            max_pending=4,
        )
        live = {0: False, 1: True, 2: True}
        alive = lambda n: live.get(n)
        keymap = {3: "k3", 4: "k4", 5: "k5", 6: "k6"}
        kv_key = keymap.get
        cc.note_node(1, 0)
        cc.note_node(2, 1)
        cc.note_node(9, 2)  # 9: dead, recorded under 2
        cc.note_pages(2, [3, 4], ["k3", "k4"])
        cc.note_pages(9, [5, 6], ["k5", "k6"])
        # 9's pages 5,6 recorded as a continuation of 2's? (separate nodes: 5 is 9's first page)
        self.assertEqual(cc.resolve(9, 5, {5, 6}, kv_key, alive, lambda p: False), 2)
        self.assertEqual(cc.stats["cc_anchor_hint"], 1)
        # a node split after record: page 6's predecessor 5 is held by a live fragment 7
        cc.note_node(7, 2)
        cc.note_pages(7, [5], ["k5"])
        live[7] = True
        self.assertEqual(cc.resolve(9, 6, {6}, kv_key, alive, lambda p: False), 7)
        self.assertEqual(cc.stats["cc_anchor_pred"], 1)
        # predecessor still in host under the dead node: counted, parent walk
        live.pop(7)
        cc.note_pages(9, [5, 6], ["k5", "k6"])
        self.assertEqual(cc.resolve(9, 6, {6}, kv_key, alive, lambda p: False), 2)
        self.assertEqual(cc.stats["cc_frag_unresolved"], 1)
        # unknown node
        self.assertIsNone(cc.resolve(42, None, set(), kv_key, alive, lambda p: False))
        self.assertEqual(cc.stats["cc_anchor_none"], 1)
        # sweep drops dead records only
        cc.SWEEP_FLOOR = 0
        cc._sweep_at = 0
        for _ in range(4):
            cc.sweep(alive)
        self.assertEqual(sorted(cc.nodes), [1, 2])

    def test_queue_bound_and_cap(self):
        cc = l3vt.ChainCloser(
            page_size=4,
            num_kv_pages=4,
            num_state_slots=1,
            cap_pages=5,
            refresh_pages=10,
            max_pending=2,
        )
        cc.enqueue([1, 2, 2, 3])
        self.assertEqual(list(cc.queue), [2, 3])
        self.assertEqual((cc.stats["cc_dup"], cc.stats["cc_dropped"]), (1, 1))
        self.assertTrue(cc.can_issue(9))  # nothing in flight: one node may exceed
        cc.issued(100, 2, 4)
        self.assertFalse(cc.can_issue(2))
        self.assertTrue(cc.can_issue(1))
        self.assertEqual(cc.needs_close(2), "inflight")
        cc.note_ack(100, 4 * 4, 0, 0, None)
        self.assertEqual(cc.needs_close(2), None if 2 not in cc.nodes else "closed")
        cc.clock_pages += 10
        self.assertIsNone(cc.needs_close(2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
