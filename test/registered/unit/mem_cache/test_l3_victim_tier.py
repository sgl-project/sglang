"""Unit tests for the HiCache L3 victim tier (SGLANG_OPT_HICACHE_L3_VICTIM_TIER).

The cache-side code under test is the served code: the victim-tier methods of
UnifiedRadixCache (``_l3vt_*``, ``_l3pin_*``, ``release_l3_pins``,
``_l3vt_recharge_hit``) and the stock methods they ride on
(``_finish_write_through_ack``, ``write_backup_storage``, ``evict_host``,
``_apply_eviction_result``, ``_free_values``, ``_drain_host_frees``,
``_build_sidecar_transfers``, ``_prefetch_occupied_span``, the backup ack
drain ``_drain_backup`` and ``_finish_backup_ack``) are lifted from
unified_radix_cache.py by AST onto a fake cache, and l3_victim_tier.py is loaded as
is. Around them: a radix tree model with a host LRU, host KV / draft / state
pools holding real bytes (double alloc / double free asserted), and a storage
store holding real bytes with exists-skip puts. No GPU or Mooncake is needed;
torch (CPU) is.

Properties checked:
* every object read back from storage holds exactly the bytes of the content
  its key names (KV page, its draft page, GDN state), on the stock path and
  under the victim tier, including promoted (storage -> host -> storage)
  copies and delayed puts while freed pages are reused;
* the victim tier writes a page / state only when it leaves the host tier, and
  storage then holds nothing the host tier still holds (exclusive);
* held pages are never reallocated before their put is acked; nothing leaks;
* a key entry exists only for a page / slot the tree (or an in-flight put)
  holds;
* two ranks with different put timings and MIN-synced ack drains take the same
  victim decisions and keep identical trees;
* pins keep a storage hit through host pressure until admit / abandon, within
  the cap; the budget re-charge releases exactly the over-charge;
* with every switch off the stock path runs unchanged (same puts, same order).
"""

import ast
import hashlib
import importlib.util
import os
import random
import sys
import textwrap
import types
import unittest
from collections import OrderedDict, defaultdict
from enum import Enum, IntEnum
from types import SimpleNamespace

import torch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # standalone run without an sglang install

    def register_cpu_ci(**kwargs):
        pass


register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRT = os.path.normpath(os.path.join(_HERE, "../../../../python/sglang/srt"))


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


l3vt = _load_file(
    "_l3vt_module", os.path.join(_SRT, "mem_cache/hybrid_cache/l3_victim_tier.py")
)


def _lift(path, names, module_name, extra_globals):
    """Exec the named top-level classes / functions (or 'Class.method') of a
    file into a fresh module; returns that module."""
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src)
    mod = types.ModuleType(module_name)
    sys.modules[module_name] = mod
    mod.__dict__.update(extra_globals)
    found = set()
    for name in names:
        cls, _, meth = name.partition(".")
        for node in tree.body:
            if getattr(node, "name", None) != cls:
                continue
            if not meth:
                decos = "".join(
                    "@" + ast.get_source_segment(src, d) + "\n"
                    for d in node.decorator_list
                )
                seg = decos + ast.get_source_segment(src, node)
                exec(
                    compile("from __future__ import annotations\n" + seg, path, "exec"),
                    mod.__dict__,
                )
                found.add(name)
            else:
                for sub in node.body:
                    if getattr(sub, "name", None) == meth:
                        seg = textwrap.dedent(ast.get_source_segment(src, sub))
                        decos = "".join(
                            "@" + ast.get_source_segment(src, d) + "\n"
                            for d in sub.decorator_list
                        )
                        exec(
                            compile(
                                "from __future__ import annotations\n" + decos + seg,
                                path,
                                "exec",
                            ),
                            mod.__dict__,
                        )
                        found.add(name)
    missing = set(names) - found
    assert not missing, f"not found in {path}: {missing}"
    return mod


_hs = _lift(
    os.path.join(_SRT, "mem_cache/hicache_storage.py"),
    [
        "PoolName",
        "host_pool_name",
        "PoolHitPolicy",
        "PoolTransfer",
        "PoolTransferResult",
        "SidecarPoolSpec",
        "merge_source_transfers",
        "count_pool_hits",
    ],
    "_l3vt_hicache_storage",
    {
        "HostPoolName": str,
        "Enum": Enum,
        "dataclass": __import__("dataclasses").dataclass,
        "Optional": __import__("typing").Optional,
        "List": __import__("typing").List,
        "Any": object,
        "Iterable": __import__("typing").Iterable,
        "Sequence": __import__("typing").Sequence,
        "Set": __import__("typing").Set,
        "torch": torch,
    },
)
PoolName, PoolHitPolicy, PoolTransfer = (
    _hs.PoolName,
    _hs.PoolHitPolicy,
    _hs.PoolTransfer,
)
PoolTransferResult, SidecarPoolSpec = _hs.PoolTransferResult, _hs.SidecarPoolSpec


class ComponentType(IntEnum):
    FULL = 0
    SWA = 1
    MAMBA = 2


class CacheTransferPhase(Enum):
    BACKUP_HOST = "backup_host"
    BACKUP_STORAGE = "backup_storage"
    PREFETCH = "prefetch"


TREE_STATE = "state"


def _identity_rank_consensus(func=None, **kwargs):
    if func is not None:
        return func
    return lambda f: f


_URC_METHODS = [
    "UnifiedRadixCache._l3vt_pages",
    "UnifiedRadixCache._l3vt_record_node",
    "UnifiedRadixCache._l3vt_forget",
    "UnifiedRadixCache._l3vt_deficits",
    "UnifiedRadixCache._l3vt_maintain",
    "UnifiedRadixCache._l3vt_put_victims",
    "UnifiedRadixCache._l3vt_release",
    "UnifiedRadixCache._l3vt_free_held",
    "UnifiedRadixCache._finish_backup_ack",
    "UnifiedRadixCache._backup_requested_tokens_of",
    "UnifiedRadixCache._l3pin_hold",
    "UnifiedRadixCache._l3pin_release",
    "UnifiedRadixCache.release_l3_pins",
    "UnifiedRadixCache.l3vt_note_refetch",
    "UnifiedRadixCache._l3vt_recharge_hit",
    "UnifiedRadixCache._l3vt_stats_suffix",
    "UnifiedRadixCache._drain_host_frees",
    "UnifiedRadixCache._free_values",
    "UnifiedRadixCache._apply_eviction_result",
    "UnifiedRadixCache._build_sidecar_transfers",
    "UnifiedRadixCache._finish_write_through_ack",
    "UnifiedRadixCache.write_backup_storage",
    "UnifiedRadixCache.evict_host",
    "UnifiedRadixCache._prefetch_occupied_span",
    "UnifiedRadixCache._l3vt_charge",
    "UnifiedRadixCache._l3vt_unsupported",
]
_urc = _lift(
    os.path.join(_SRT, "mem_cache/unified_radix_cache.py"),
    _URC_METHODS,
    "_l3vt_urc",
    {
        "torch": torch,
        "time": __import__("time"),
        "logger": __import__("logging").getLogger("l3vt-test"),
        "Optional": __import__("typing").Optional,
        "NodeId": int,
        "ComponentType": ComponentType,
        "BASE_COMPONENT_TYPE": ComponentType.FULL,
        "PoolName": PoolName,
        "PoolTransfer": PoolTransfer,
        "PoolHitPolicy": PoolHitPolicy,
        "CacheTransferPhase": CacheTransferPhase,
        "HeldVictim": l3vt.HeldVictim,
        "L3VictimTierConfig": l3vt.L3VictimTierConfig,
        "PinEntry": l3vt.PinEntry,
        "TREE_STATE": TREE_STATE,
        "rank_consensus": _identity_rank_consensus,
        "merge_source_transfers": _hs.merge_source_transfers,
        "CacheRequestHandle": object,
        "DecLockRefParams": object,
    },
)

_OngoingWriteThrough = __import__("collections").namedtuple(
    "_OngoingWriteThrough", "node_id lock_params publish_node_ids"
)


def _served_drain_backup_factory():
    """The served ack drain (``_drain_backup``, nested in
    ``_drain_storage_control_queues_impl``), lifted from the source text."""
    path = os.path.join(_SRT, "mem_cache/unified_radix_cache.py")
    src = open(path).read()
    a = src.index("        def _drain_backup():")
    b = src.index("        def _drain_release():", a)
    body = textwrap.dedent(src[a:b])
    fn = (
        "def make(self, cc, n_backup, log_metrics, _drain_queue):\n"
        + textwrap.indent(body, "    ")
        + "    return _drain_backup\n"
    )
    ns = {"torch": torch}
    exec(compile(fn, path, "exec"), ns)
    return ns["make"]


_make_drain = _served_drain_backup_factory()


def _drain_queue(q, n):
    take = len(q) if n is None else min(n, len(q))
    return [q.pop(0) for _ in range(take)]


PAGE = 4  # tokens a page
KV_ROW = 8  # bytes a token row in the fake host KV / draft pools
STATE_BYTES = 64  # bytes a state slot


def page_bytes(key: str, tag: str) -> bytes:
    """Ground-truth content of a page / state named by its storage key."""
    h = hashlib.sha256((tag + key).encode()).digest()
    n = STATE_BYTES if tag == "state" else PAGE * KV_ROW
    return (h * (n // len(h) + 1))[:n]


def chain_keys(tokens, page=PAGE, last=None):
    keys = []
    for i in range(0, len(tokens), page):
        h = hashlib.sha256(
            ((last or "") + "|" + ",".join(map(str, tokens[i : i + page]))).encode()
        ).hexdigest()
        keys.append(h)
        last = h
    return keys


# ---------------------------------------------------------------------------
# Fake pools (real bytes, double alloc / free asserted)
# ---------------------------------------------------------------------------


class FakeHostPool:
    def __init__(self, num_tokens, row_bytes, page_size):
        self.size = num_tokens
        self.logical_size = num_tokens
        self.page_size = page_size
        self.row_bytes = row_bytes
        self.data = bytearray(num_tokens * row_bytes)
        self.used = [False] * num_tokens
        self.free_list = list(range(num_tokens))
        self.shared_allocation_domain = None
        self.dcp_size = 1

    def available_size(self):
        return len(self.free_list)

    def alloc(self, n):
        assert n % self.page_size == 0
        if n > len(self.free_list):
            return None
        idx = self.free_list[:n]
        self.free_list = self.free_list[n:]
        for i in idx:
            assert not self.used[i], f"double alloc {i}"
            self.used[i] = True
        # Clobber: a fresh allocation never shows the previous occupant.
        for i in idx:
            self.data[i * self.row_bytes : (i + 1) * self.row_bytes] = (
                b"\xee" * self.row_bytes
            )
        return torch.tensor(idx, dtype=torch.int64)

    def free(self, indices):
        idx = indices.reshape(-1).tolist()
        for i in idx:
            assert self.used[i], f"double free {i}"
            self.used[i] = False
        self.free_list.extend(idx)
        return len(idx)

    def write_page(self, first_token, payload):
        o = first_token * self.row_bytes
        self.data[o : o + len(payload)] = payload

    def read_rows(self, first_token, n_rows):
        o = first_token * self.row_bytes
        return bytes(self.data[o : o + n_rows * self.row_bytes])


class FakeHostGroup:
    """HostPoolGroup facade: anchor KV pool plus the state and draft pools.
    The draft pool is KV-derived: it shares the KV pool's indices."""

    def __init__(self, kv_tokens, state_slots, page_size):
        self.kv = FakeHostPool(kv_tokens, KV_ROW, page_size)
        self.draft = FakeHostPool(kv_tokens, KV_ROW, page_size)  # aliased indices
        self.state = FakeHostPool(state_slots, STATE_BYTES, 1)
        self.anchor_entry = SimpleNamespace(host_pool=self.kv, host_evict_fn=None)
        self.size = kv_tokens
        self.logical_size = kv_tokens
        self.page_size = page_size

    def get_pool(self, name):
        return self.state if name == PoolName.MAMBA else self.kv

    def available_size(self, pool=None):
        return (
            self.get_pool(pool).available_size() if pool else self.kv.available_size()
        )

    def free(self, indices, pool=None):
        if pool == PoolName.MAMBA:
            return self.state.free(indices)
        return self.kv.free(indices)


# ---------------------------------------------------------------------------
# Fake storage (real bytes) and the backup thread
# ---------------------------------------------------------------------------


class FakeStore:
    def __init__(self, capacity_objects=None):
        self.objs: OrderedDict[str, bytes] = OrderedDict()
        self.capacity = capacity_objects
        self.puts = 0
        self.put_skips = 0
        self.put_log = []
        self.fail_rate = 0.0
        self.rng = random.Random(7)

    def put(self, key, payload):
        if key in self.objs:
            self.put_skips += 1
            self.objs.move_to_end(key)
            return True
        if self.fail_rate and self.rng.random() < self.fail_rate:
            return False
        self.objs[key] = payload
        self.puts += 1
        self.put_log.append(key)
        if self.capacity is not None:
            while len(self.objs) > self.capacity:
                self.objs.popitem(last=False)
        return True

    def get(self, key):
        v = self.objs.get(key)
        if v is not None:
            self.objs.move_to_end(key)
        return v


class FakeOp:
    counter = 0

    def __init__(
        self, host_indices, token_ids, hash_value, prefix_keys, pool_transfers
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.hash_value = list(hash_value or [])
        self.prefix_keys = prefix_keys
        self.pool_transfers = pool_transfers
        self.completed_tokens = 0
        self.pool_storage_result = PoolTransferResult.empty()
        self.l3vt_touch = False
        self.l3vt_touch_result = None
        self.l3vt_charged = None
        self.l3vt_io_t0 = None
        self.id = FakeOp.counter
        FakeOp.counter += 1


class FakeController:
    def __init__(self, group, store, test):
        self.mem_pool_host = group
        self.page_size = PAGE
        self.store = store
        self.backup_queue = []
        self.ack_queue = []
        self.ack_backup_queue = self.ack_queue
        self.backup_skip = False
        self.prefetch_tokens_occupied = 0
        self.prefetch_capacity_limit = group.logical_size // 2
        self.storage_backend = SimpleNamespace(batch_set_v2=lambda *a, **k: None)
        self.test = test

    def write_storage(
        self,
        host_indices,
        token_ids,
        hash_value=None,
        prefix_keys=None,
        extra_pools=None,
        lease_touch=False,
    ):
        op = FakeOp(host_indices, token_ids, hash_value, prefix_keys, extra_pools)
        op.l3vt_touch = lease_touch
        self.backup_queue.append(op)
        return op.id

    def run_backup(self, op):
        """The backup thread's work for one op, on the stock layout: state
        (and other sidecars) first, then the KV pages; the draft sidecar takes
        the op's KV indices and keys."""
        g = self.mem_pool_host
        results = {}
        for t in op.pool_transfers or ():
            if t.indices_from_pool == PoolName.KV:
                t.host_indices = op.host_indices
                t.keys = op.hash_value
            ok = []
            if t.name == PoolName.MAMBA:
                assert len(t.keys) == t.host_indices.numel() > 0
                for slot, key in zip(t.host_indices.tolist(), t.keys):
                    payload = g.state.read_rows(slot, 1)
                    self.test.assertEqual(
                        payload, page_bytes(key, "state"), f"state {key}"
                    )
                    ok.append(self.store.put(key + "#state", payload))
            elif t.name == PoolName.DRAFT:
                assert len(t.keys) > 0
                hi = t.host_indices.reshape(-1, PAGE)[:, 0].tolist()
                for first, key in zip(hi, t.keys):
                    payload = g.draft.read_rows(first, PAGE)
                    self.test.assertEqual(
                        payload, page_bytes(key, "draft"), f"draft {key}"
                    )
                    ok.append(self.store.put(key + "#draft", payload))
            else:
                raise AssertionError(t.name)
            results[t.name] = ok
        op.pool_storage_result.update_extra_pool_hit_pages(_hs.count_pool_hits(results))
        done = 0
        hi = (
            op.host_indices.reshape(-1, PAGE)[:, 0].tolist()
            if op.host_indices.numel()
            else []
        )
        for first, key in zip(hi, op.hash_value):
            payload = g.kv.read_rows(first, PAGE)
            self.test.assertEqual(payload, page_bytes(key, "kv"), f"kv {key}")
            if not self.store.put(key + "#kv", payload):
                break
            done += PAGE
        op.completed_tokens = done


# ---------------------------------------------------------------------------
# Fake radix tree (host tier + LRU), enough for the hooks under test
# ---------------------------------------------------------------------------


class Node:
    def __init__(self, nid, parent, tokens, keys):
        self.id = nid
        self.parent = parent
        self.children = {}
        self.tokens = tokens
        self.keys = keys
        self.device = True  # device copy present (write_back: host only after demote)
        self.host_value = None
        self.state_slot = None
        self.host_lock = 0
        self.last_access = 0.0


class FakeTreeCore:
    def __init__(self, cache):
        self.cache = cache
        self.root = Node(0, None, [], [])
        self.root.device = False
        self.nodes = {0: self.root}
        self.next_id = 1
        self.clock = 0.0

    def tick(self):
        self.clock += 1.0
        return self.clock

    def node(self, nid):
        return self.nodes[nid]

    # tree building
    def add_child(self, parent_id, tokens):
        parent = self.nodes[parent_id]
        keys = chain_keys(tokens, last=parent.keys[-1] if parent.keys else None)
        n = Node(self.next_id, parent, tokens, keys)
        self.next_id += 1
        # Children by id: two siblings may share a first page in this model.
        n.child_key = n.id
        parent.children[n.child_key] = n
        self.nodes[n.id] = n
        n.last_access = self.tick()
        return n

    def path_tokens(self, n):
        out = []
        while n is not None and n is not self.root:
            out = n.tokens + out
            n = n.parent
        return out

    # interface used by the lifted code
    def build_storage_backup_spec(self, node_id, pass_prefix_keys):
        n = self.nodes.get(node_id)
        if n is None or n.host_value is None:
            return None
        comp = {}
        if n.state_slot is not None:
            comp[ComponentType.MAMBA] = [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=torch.tensor([n.state_slot], dtype=torch.int64),
                    keys=[n.keys[-1]],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]
        return SimpleNamespace(
            host_value=n.host_value.clone(),
            token_ids=b"",
            hash_value=list(n.keys),
            prefix_keys=None,
            comp_xfers=comp,
        )

    def get_last_hash_value(self, node_id):
        n = self.nodes[node_id]
        return n.keys[-1] if n.keys else None

    def finish_write_through(self, node_ids, ack_id):
        pass

    def inc_host_lock_ref(self, node_id):
        self.nodes[node_id].host_lock += 1
        return SimpleNamespace(to_dec_params=lambda: ("hl", node_id))

    def dec_host_lock_ref(self, node_id, params):
        assert params == ("hl", node_id)
        n = self.nodes[node_id]
        assert n.host_lock > 0
        n.host_lock -= 1

    def _host_leaf(self, n):
        return (
            n is not self.root
            and n.host_value is not None
            and not n.device
            and not n.children
            and n.host_lock == 0
        )

    def _evict_leaf(self, n, result):
        result.host_frees[ComponentType.FULL].append(n.host_value)
        result.tracker[ComponentType.FULL] += n.host_value.numel()
        if n.state_slot is not None:
            result.host_frees[ComponentType.MAMBA].append(
                torch.tensor([n.state_slot], dtype=torch.int64)
            )
            result.tracker[ComponentType.MAMBA] += 1
        n.host_value = None
        n.state_slot = None
        del n.parent.children[n.child_key]
        del self.nodes[n.id]

    def drive_host_eviction(self, component_type, num):
        result = SimpleNamespace(
            tracker=defaultdict(int),
            device_frees=defaultdict(list),
            host_frees=defaultdict(list),
            cache_actions=[],
        )
        ct = ComponentType(component_type)
        while result.tracker[ct] < num:
            if ct == ComponentType.FULL:
                leaves = [n for n in self.nodes.values() if self._host_leaf(n)]
                if not leaves:
                    break
                self._evict_leaf(min(leaves, key=lambda n: n.last_access), result)
            else:
                holders = [
                    n
                    for n in self.nodes.values()
                    if n.state_slot is not None and not n.device and n.host_lock == 0
                ]
                if not holders:
                    break
                n = min(holders, key=lambda n: n.last_access)
                if self._host_leaf(n):
                    self._evict_leaf(n, result)
                else:  # internal: tombstone the state only
                    result.host_frees[ComponentType.MAMBA].append(
                        torch.tensor([n.state_slot], dtype=torch.int64)
                    )
                    result.tracker[ComponentType.MAMBA] += 1
                    n.state_slot = None
        return result

    def snapshot(self):
        return sorted(
            (
                n.id,
                n.parent.id if n.parent else -1,
                tuple(n.keys),
                n.device,
                n.host_value is not None,
                n.state_slot is not None,
                n.host_lock,
            )
            for n in self.nodes.values()
        )


class FakeComponent:
    def __init__(self, cache, pool_name):
        self.cache = cache
        self.pool_name = pool_name

    def free_host_values(self, values):
        for v in values:
            self.cache.host_pool_group.free(v, pool=self.pool_name)


# ---------------------------------------------------------------------------
# Fake cache carrying the lifted methods
# ---------------------------------------------------------------------------


class FakeCache:
    # UnifiedRadixCache's class-level off default for chain close.
    _l3cc = None

    def _all_reduce_storage_drain(self, data):
        pass  # one rank


for _name in _URC_METHODS:
    _meth = _name.split(".", 1)[1]
    setattr(FakeCache, _meth, getattr(_urc, _meth))


def make_cache(
    test,
    *,
    victim,
    pin=False,
    kv_tokens=4096,
    state_slots=48,
    store=None,
    margin=0.05,
    pin_frac=0.25,
):
    c = FakeCache()
    group = FakeHostGroup(kv_tokens, state_slots, PAGE)
    c.host_pool_group = group
    c.store = store if store is not None else FakeStore()
    c.cache_controller = FakeController(group, c.store, test)
    c.tree_core = FakeTreeCore(c)
    c.page_size = PAGE
    c.enable_storage = True
    c.host_memory_mode = "cache"
    c.buffer_pipeline = None
    c.hicache_storage_pass_prefix_keys = False
    c.ongoing_write_through = {}
    c.ongoing_backup = {}
    c._backup_requested_tokens = {}
    c.enable_storage_metrics = False
    c.storage_metrics_collector = None
    c.sidecar_pool_specs = [
        SidecarPoolSpec(PoolName.DRAFT, indices_from_pool=PoolName.KV)
    ]
    c.components = {
        ComponentType.FULL: FakeComponent(c, PoolName.KV),
        ComponentType.MAMBA: FakeComponent(c, PoolName.MAMBA),
    }
    c.slot_id = lambda family: ComponentType.MAMBA if family == TREE_STATE else None
    c.inc_host_lock_ref = c.tree_core.inc_host_lock_ref
    c.dec_host_lock_ref = c.tree_core.dec_host_lock_ref
    c.dec_lock_ref = lambda *a, **k: None
    c._drain_device_frees = lambda device_frees: device_frees.clear()
    c._apply_cache_actions = lambda actions: actions.clear()
    c._l3vt_cfg = l3vt.L3VictimTierConfig(victim=victim, pin=pin)
    c._l3vt_stats = {
        "budget_recharges": 0,
        "budget_released_tokens": 0,
        "assume_stored_off": 0,
    }
    c._l3vt = (
        l3vt.VictimTier(
            page_size=PAGE,
            kv_capacity_tokens=kv_tokens,
            state_capacity_slots=state_slots,
            margin=margin,
            stats_interval_s=1e9,
        )
        if victim
        else None
    )
    c._l3pin = l3vt.PinLedger(int(pin_frac * kv_tokens)) if pin else None
    c._l3wb = None
    c._l3vt_touch = False
    return c


def land(cache, node, with_state=True):
    """Device eviction under write_back: host pages + state slot written with
    the node's bytes, committed, then the backup ack."""
    g = cache.host_pool_group
    n_tok = len(node.tokens)
    if g.kv.available_size() < n_tok:
        cache.evict_host(n_tok - g.kv.available_size())
    hv = g.kv.alloc(n_tok)
    assert hv is not None, "host full"
    g.draft.used = g.kv.used  # draft rows share the KV indices
    for i, key in enumerate(node.keys):
        first = hv[i * PAGE].item()
        g.kv.write_page(first, page_bytes(key, "kv"))
        g.draft.write_page(first, page_bytes(key, "draft"))
    node.host_value = hv
    if with_state:
        if g.state.available_size() < 1:
            cache.evict_host(1, ComponentType.MAMBA)
        slot = g.state.alloc(1)
        if slot is not None:
            g.state.write_page(slot.item(), page_bytes(node.keys[-1], "state"))
            node.state_slot = slot.item()
    node.device = False
    ack_id = node.id
    cache.ongoing_write_through[ack_id] = _OngoingWriteThrough(node.id, None, [node.id])
    cache._finish_write_through_ack(ack_id)


def drain_backups(cache, n=None):
    """Run the backup thread for up to n queued ops, then the scheduler's
    ack drain (the served _drain_backup hook)."""
    cc = cache.cache_controller
    take = len(cc.backup_queue) if n is None else min(n, len(cc.backup_queue))
    for _ in range(take):
        op = cc.backup_queue.pop(0)
        cc.run_backup(op)
        cc.ack_queue.append(op)
    k = len(cc.ack_queue)
    _make_drain(cache, cc, k, False, _drain_queue)()
    return k


def restorable(store, path_keys):
    """Deepest page count p with KV (+draft) for pages [0, p) and a state at p-1."""
    best = 0
    for p in range(1, len(path_keys) + 1):
        k = path_keys[p - 1]
        if store.get(k + "#kv") is None or store.get(k + "#draft") is None:
            break
        if store.get(k + "#state") is not None:
            best = p
    return best


class TestL3VictimTierUnits(unittest.TestCase):
    def test_config_resolution(self):
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
                    "close_cap": 0.0,
                    "close_refresh": 0.25,
                    "pin_frac": 0.25,
                    "budget_tokens": 0,
                    "wait_slack": 3.0,
                }
                return SimpleNamespace(get=lambda: self.kv.get(key, defaults[key]))

        base = dict(
            enable_storage=True,
            disaggregation_mode="prefill",
            pd_role_switch=False,
            host_memory_mode="cache",
            has_linker=False,
            is_write_back=True,
            tree_core_backend="python",
            victim_unsupported=None,
            prefetch_stop_policy="timeout",
            has_storage_v2=True,
            pass_prefix_keys=False,
            has_lease_touch=True,
        )
        resolve = l3vt.L3VictimTierConfig.resolve
        off = resolve(E(), **base)
        self.assertFalse(off.any_on)
        self.assertIsNone(off.decline)
        self.assertEqual(off.ignored, {})
        on = resolve(E(**{"": True}), **base)
        self.assertTrue(
            on.victim
            and on.lease_touch
            and on.chain_close
            and on.pin
            and on.budget
            and on.wait_bound
        )
        self.assertEqual(on.ignored, {})
        # The served constants are the defaults.
        self.assertEqual(
            (
                on.victim_margin,
                on.state_margin,
                on.chain_close_cap,
                on.chain_close_refresh,
                on.pin_frac,
                on.budget_tokens,
                on.wait_slack,
            ),
            (0.05, 0.05, 0.0, 0.25, 0.25, 0, 3.0),
        )
        # Any tier precondition declines every part.
        for override, reason in (
            ({"is_write_back": False}, "write policy is not write_back"),
            ({"host_memory_mode": "buffer_only"}, "host memory mode buffer_only"),
            ({"enable_storage": False}, "storage off"),
            ({"has_linker": True}, "external cache linker"),
            (
                {"victim_unsupported": "shared or DCP host layout"},
                "shared or DCP host layout",
            ),
            ({"has_storage_v2": False}, "backend has no batch_set_v2"),
            (
                {"disaggregation_mode": "decode"},
                "PD decode role (storage hits restore outside prefill admission)",
            ),
            (
                {"pd_role_switch": True},
                "PD role switch enabled (the role can change after this check)",
            ),
            ({"tree_core_backend": "rust"}, "the rust TreeCore"),
            (
                {"pass_prefix_keys": True},
                "backend takes prefix keys, which a victim put does not carry",
            ),
        ):
            cfg = resolve(E(**{"": True}), **{**base, **override})
            self.assertFalse(
                cfg.victim or cfg.pin or cfg.budget or cfg.wait_bound or cfg.chain_close
            )
            self.assertEqual(cfg.decline, reason)
        self.assertIn(
            "KV margin", resolve(E(**{"": True, "kv_margin": 0.6}), **base).decline
        )
        # Parts with their own precondition stay off, named, with the tier on.
        be = resolve(E(**{"": True}), **{**base, "prefetch_stop_policy": "best_effort"})
        self.assertTrue(be.victim and not be.wait_bound)
        self.assertEqual(be.ignored, {"wait_bound": "prefetch policy best_effort"})
        nt = resolve(E(**{"": True}), **{**base, "has_lease_touch": False})
        self.assertTrue(nt.victim and not nt.lease_touch)
        cc = resolve(E(**{"": True, "close_refresh": 0.0}), **base)
        self.assertTrue(cc.victim and not cc.chain_close)
        pn = resolve(E(**{"": True, "pin_frac": 0.9}), **base)
        self.assertTrue(pn.victim and not pn.pin)
        # An aggregated server engages like a PD prefill.
        self.assertTrue(
            resolve(E(**{"": True}), **{**base, "disaggregation_mode": "null"}).victim
        )

    def test_ranks_agree_on_the_parts(self):
        """Every part changes which storage writes and host evictions a rank
        makes, so the ranks of the lockstep groups take the MIN of their parts:
        a rank that declines (here a sidecar on one PP stage) turns the tier off
        on every rank, and a part off on one rank is off on all of them."""
        resolve = l3vt.L3VictimTierConfig.resolve

        served = {
            "": True,
            "_KV_MARGIN": 0.05,
            "_STATE_MARGIN": 0.05,
            "_CLOSE_CAP": 0.0,
            "_CLOSE_REFRESH": 0.25,
            "_PIN_FRAC": 0.25,
            "_BUDGET_TOKENS": 0,
            "_WAIT_SLACK": 3.0,
        }

        class E:
            def __getattr__(self, name):
                value = served[name.replace("SGLANG_OPT_HICACHE_L3_VICTIM_TIER", "")]
                return SimpleNamespace(get=lambda: value)

        base = dict(
            enable_storage=True,
            disaggregation_mode="prefill",
            pd_role_switch=False,
            host_memory_mode="cache",
            has_linker=False,
            is_write_back=True,
            tree_core_backend="python",
            victim_unsupported=None,
            prefetch_stop_policy="timeout",
            has_storage_v2=True,
            pass_prefix_keys=False,
            has_lease_touch=True,
        )

        def agree(*cfgs):
            parts = [min(col) for col in zip(*(c.parts() for c in cfgs))]
            return [c.agreed(parts) for c in cfgs]

        engaged = resolve(E(), **base)
        self.assertTrue(engaged.victim and engaged.pin and engaged.chain_close)
        self.assertIs(agree(engaged, engaged)[0], engaged)
        pp1 = resolve(
            E(), **{**base, "victim_unsupported": "a sidecar not derived from KV"}
        )
        a, b = agree(engaged, pp1)
        self.assertEqual(a.parts(), [0] * 6)
        self.assertEqual(b.parts(), [0] * 6)
        self.assertEqual(a.decline, "declined on another rank")
        self.assertEqual(b.decline, "a sidecar not derived from KV")
        no_touch = resolve(E(), **{**base, "has_lease_touch": False})
        a, b = agree(engaged, no_touch)
        self.assertEqual(a.parts(), b.parts())
        self.assertTrue(a.victim and not a.lease_touch)
        self.assertEqual(a.ignored, {"lease_touch": "off on another rank"})
        self.assertTrue(engaged.lease_touch, "agreed() must not mutate the input")

    def test_layout_checks(self):
        """The served ``_l3vt_unsupported``: FULL + Mamba with a KV-derived sidecar
        carries the tier. A victim put builds sidecars from its KV pages only, so a
        sidecar indexed by another pool (Mamba here) declines, as do another tree
        component and a shared or DCP host layout."""
        host_pool = SimpleNamespace(shared_allocation_domain=None, dcp_size=1)
        c = FakeCache()
        c.cache_controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(
                anchor_entry=SimpleNamespace(host_pool=host_pool)
            )
        )
        c.tree_components = [ComponentType.FULL, ComponentType.MAMBA]
        c.slot_id = lambda family: ComponentType.MAMBA if family == TREE_STATE else None
        c.sidecar_pool_specs = [
            SidecarPoolSpec(PoolName.DRAFT, indices_from_pool=PoolName.KV)
        ]
        self.assertIsNone(c._l3vt_unsupported())
        c.sidecar_pool_specs.append(
            SidecarPoolSpec(PoolName.DRAFT_SWA, indices_from_pool=PoolName.MAMBA)
        )
        self.assertEqual(c._l3vt_unsupported(), "a sidecar not derived from KV")
        c.sidecar_pool_specs.pop()
        c.tree_components = [ComponentType.FULL, ComponentType.SWA]
        self.assertTrue(c._l3vt_unsupported().startswith("components "))
        c.tree_components = [ComponentType.FULL, ComponentType.MAMBA]
        host_pool.dcp_size = 2
        self.assertEqual(c._l3vt_unsupported(), "shared or DCP host layout")

    def test_env_switch_is_an_opt_in(self):
        """Off unless set: not a child of the NVIDIA runtime master."""
        with open(os.path.join(_SRT, "environ.py")) as f:
            src = f.read()
        self.assertIn("    SGLANG_OPT_HICACHE_L3_VICTIM_TIER = EnvBool(False)\n", src)
        for name, default in (
            ("KV_MARGIN", "EnvFloat(0.05)"),
            ("STATE_MARGIN", "EnvFloat(0.05)"),
            ("CLOSE_CAP", "EnvFloat(0.0)"),
            ("CLOSE_REFRESH", "EnvFloat(0.25)"),
            ("PIN_FRAC", "EnvFloat(0.25)"),
            ("BUDGET_TOKENS", "EnvInt(0)"),
            ("WAIT_SLACK", "EnvFloat(3.0)"),
        ):
            self.assertIn(f"SGLANG_OPT_HICACHE_L3_VICTIM_TIER_{name} = {default}", src)
        # Neither the state tie nor the device anchor exists any more.
        self.assertNotIn("STATE_TIE", src)
        self.assertNotIn("DEVICE_ANCHOR", src)

    def test_keymap_and_margins(self):
        vt = l3vt.VictimTier(
            page_size=4, kv_capacity_tokens=1000, state_capacity_slots=40, margin=0.05
        )
        self.assertEqual(vt.kv_target, 48)  # 50 rounded down to pages
        self.assertEqual(vt.state_target, 2)
        self.assertEqual(vt.kv_deficit(100), 0)
        self.assertEqual(vt.kv_deficit(10), 40)  # 38 rounded up to a page
        vt.keymap.record_kv([1, 2], ["a", "b"])
        self.assertEqual(vt.keymap.split_kv([2, 3, 1]), ([0, 2], ["b", "a"], [1]))
        vt.hold(
            7,
            l3vt.HeldVictim(
                kv_tokens=None,
                state_slots=None,
                kv_pages=[1],
                slots=[],
                num_kv_tokens=4,
                issued_at=0.0,
            ),
        )
        self.assertEqual(vt.kv_deficit(10), 36)
        vt.release(7)
        self.assertIsNone(vt.keymap.kv_key(1))
        self.assertEqual(vt.keymap.kv_key(2), "b")

    def test_wait_bound(self):
        wb = l3vt.WaitBound(slack=3.0)
        self.assertEqual(wb.timeout(1000, 1.0, 9.0), 9.0)  # no measurement yet
        for _ in range(5):
            wb.observe(100_000, 1.0)
        self.assertAlmostEqual(wb.timeout(100_000, 1.0, 30.0), 4.0)
        self.assertEqual(wb.timeout(10_000_000, 1.0, 30.0), 30.0)  # never above stock

    def test_misaligned_pages_are_never_put(self):
        c = make_cache(self, victim=True)
        self.assertIsNone(c._l3vt_pages(torch.tensor([1, 2, 3, 4])))
        self.assertIsNone(c._l3vt_pages(torch.tensor([0, 1, 3, 2])))
        self.assertEqual(
            c._l3vt_pages(torch.tensor([8, 9, 10, 11, 0, 1, 2, 3])), [2, 0]
        )


class TestVictimTierEndToEnd(unittest.TestCase):
    def _workload(
        self, cache, rng, steps, *, sessions=6, turns=6, branch=0.3, check_every=7
    ):
        """Sessions append turns (a node each); a fraction of turns branch off an
        older turn. Each new turn's node lands in host (write_back device
        eviction) right away; the victim pass and the backup drains run every
        step with random delay."""
        tc = cache.tree_core
        tips = {s: tc.root for s in range(sessions)}
        history = defaultdict(list)
        all_paths = []
        for step in range(steps):
            s = rng.randrange(sessions)
            parent = tips[s]
            if history[s] and rng.random() < branch:
                parent = rng.choice(history[s])
            if parent is not tc.root and parent.id not in tc.nodes:
                parent = tc.root
            ntok = PAGE * rng.randint(1, 6)
            tokens = [rng.randrange(1000) for _ in range(ntok)]
            node = tc.add_child(parent.id, tokens)
            # touch the path (match refreshes access times)
            p = node
            while p is not None and p is not tc.root:
                p.last_access = tc.tick()
                p = p.parent
            land(cache, node, with_state=rng.random() < 0.85)
            tips[s] = node
            history[s].append(node)
            all_paths.append(tc.path_tokens(node))
            # scheduler step: drains (random partial), then the victim pass
            drain_backups(cache, n=rng.randint(0, 3))
            if cache._l3vt is not None:
                cache._l3vt_maintain()
            if step % check_every == 0:
                self._check_invariants(cache)
        drain_backups(cache)
        if cache._l3vt is not None:
            cache._l3vt_maintain()
            drain_backups(cache)
        self._check_invariants(cache)
        return all_paths

    def _check_invariants(self, cache):
        vt = cache._l3vt
        g = cache.host_pool_group
        tc = cache.tree_core
        tree_pages = {}
        tree_slots = {}
        for n in tc.nodes.values():
            if n.host_value is not None:
                for i, first in enumerate(
                    n.host_value.reshape(-1, PAGE)[:, 0].tolist()
                ):
                    tree_pages[first // PAGE] = n.keys[i]
            if n.state_slot is not None:
                tree_slots[n.state_slot] = n.keys[-1]
        held_pages = {p for h in (vt.held.values() if vt else ()) for p in h.kv_pages}
        held_slots = {s for h in (vt.held.values() if vt else ()) for s in h.slots}
        # no page is both in the tree and held
        self.assertFalse(set(tree_pages) & held_pages)
        self.assertFalse(set(tree_slots) & held_slots)
        # pool usage == tree + held (+ queued stock puts keep tree pages)
        used_pages = {i // PAGE for i, u in enumerate(g.kv.used) if u}
        self.assertEqual(used_pages, set(tree_pages) | held_pages)
        used_slots = {i for i, u in enumerate(g.state.used) if u}
        self.assertEqual(used_slots, set(tree_slots) | held_slots)
        if vt is None:
            return
        # key entries only for tree-held or in-flight pages, and right
        for p in range(len(vt.keymap._kv)):
            k = vt.keymap.kv_key(p)
            if k is None:
                continue
            self.assertTrue(p in tree_pages or p in held_pages, f"stale entry page {p}")
            if p in tree_pages:
                self.assertEqual(k, tree_pages[p])
        for s in range(len(vt.keymap._st)):
            k = vt.keymap.state_key(s)
            if k is None:
                continue
            self.assertTrue(s in tree_slots or s in held_slots, f"stale entry slot {s}")
            if s in tree_slots:
                self.assertEqual(k, tree_slots[s])
        self.assertEqual(vt.held_kv_tokens, len(held_pages) * PAGE)
        self.assertLessEqual(vt.held_kv_tokens, vt.kv_target)
        self.assertLessEqual(vt.held_states, max(vt.state_target, 0))

    def _check_store_bytes(self, store):
        for key, payload in store.objs.items():
            base, tag = key.rsplit("#", 1)
            self.assertEqual(payload, page_bytes(base, tag), key)

    def test_victim_tier_bytes_and_exclusivity(self):
        for seed in range(6):
            rng = random.Random(seed)
            cache = make_cache(self, victim=True, kv_tokens=1024, state_slots=24)
            self._workload(cache, rng, steps=220)
            store = cache.store
            self._check_store_bytes(store)
            vt = cache._l3vt
            self.assertGreater(vt.stats["put_ops"], 0, seed)
            self.assertGreater(vt.stats["acked_kv_pages_ok"], 0, seed)
            # exclusive: nothing written to storage is still in the host tree
            # unless it came back (none here: no prefetch in this workload)
            tree_keys = {
                k
                for n in cache.tree_core.nodes.values()
                if n.host_value is not None
                for k in n.keys
            }
            stored = {k.rsplit("#", 1)[0] for k in store.objs if k.endswith("#kv")}
            self.assertFalse(stored & tree_keys, seed)
            # a stock put never ran
            self.assertEqual(len(cache.ongoing_backup), 0)

    def test_stock_path_unchanged_when_off(self):
        """Switch off: _finish_write_through_ack issues write_backup_storage
        per node in order, which puts at landing (inclusive)."""
        rng = random.Random(3)
        cache = make_cache(self, victim=False, kv_tokens=1024, state_slots=24)
        tc = cache.tree_core
        nodes = []
        for _ in range(12):
            n = tc.add_child(tc.root.id, [rng.randrange(1000) for _ in range(PAGE * 2)])
            land(cache, n)
            nodes.append(n)
        cc = cache.cache_controller
        self.assertEqual(len(cc.backup_queue), 12)
        self.assertEqual(
            [op.hash_value for op in cc.backup_queue], [n.keys for n in nodes]
        )
        drain_backups(cache)
        self._check_store_bytes(cache.store)
        for n in nodes:
            self.assertEqual(restorable(cache.store, n.keys), len(n.keys))
        # every landed page is in storage while still in host: inclusive
        tree_keys = {k for n in nodes for k in n.keys}
        stored = {k.rsplit("#", 1)[0] for k in cache.store.objs if k.endswith("#kv")}
        self.assertEqual(tree_keys, stored)

    def test_delayed_puts_hold_pages(self):
        """Puts acked late: the freed-but-held pages must keep their bytes while
        new landings allocate around them (the put asserts the bytes)."""
        rng = random.Random(11)
        cache = make_cache(self, victim=True, kv_tokens=512, state_slots=16, margin=0.2)
        tc = cache.tree_core
        for step in range(150):
            n = tc.add_child(
                tc.root.id,
                [rng.randrange(1000) for _ in range(PAGE * rng.randint(1, 4))],
            )
            land(cache, n)
            cache._l3vt_maintain()
            if step % 9 == 0:
                drain_backups(cache)  # acks only every 9 steps
            self._check_invariants(cache)
        drain_backups(cache)
        self._check_store_bytes(cache.store)
        self.assertGreater(cache._l3vt.stats["put_ops"], 0)

    def test_restore_and_promoted_copy(self):
        """A path restored from storage into host (prefetch insert) is recorded;
        evicted again it is put again (skip if still stored); bytes equal on
        every read."""
        rng = random.Random(5)
        store = FakeStore(capacity_objects=None)
        cache = make_cache(
            self, victim=True, kv_tokens=1024, state_slots=24, store=store
        )
        paths = self._workload(cache, rng, steps=150)
        tc = cache.tree_core
        g = cache.host_pool_group
        restored = 0
        for toks in paths[:60]:
            keys = chain_keys(toks)
            p = restorable(store, keys)
            if p == 0:
                continue
            # read back: bytes equal for every page and the stop state
            for k in keys[:p]:
                self.assertEqual(store.get(k + "#kv"), page_bytes(k, "kv"))
                self.assertEqual(store.get(k + "#draft"), page_bytes(k, "draft"))
            self.assertEqual(
                store.get(keys[p - 1] + "#state"), page_bytes(keys[p - 1], "state")
            )
            # insert the restored span as a host-only node under root
            # (fresh namespace via a salt token so it does not collide)
            if g.kv.available_size() < p * PAGE:
                cache.evict_host(p * PAGE - g.kv.available_size())
            hv = g.kv.alloc(p * PAGE)
            slot = g.state.alloc(1)
            if hv is None or slot is None:
                if hv is not None:
                    g.kv.free(hv)
                continue
            node = Node(tc.next_id, tc.root, toks[: p * PAGE], keys[:p])
            tc.next_id += 1
            node.child_key = node.id
            tc.root.children[node.child_key] = node
            tc.nodes[node.id] = node
            node.device = False
            node.last_access = tc.tick()
            for i, k in enumerate(keys[:p]):
                first = hv[i * PAGE].item()
                g.kv.write_page(first, store.get(k + "#kv"))
                g.draft.write_page(first, store.get(k + "#draft"))
            g.state.write_page(slot.item(), store.get(keys[p - 1] + "#state"))
            node.host_value = hv
            node.state_slot = slot.item()
            # the served hook in _handle_prefetch_result
            cache._l3vt_record_node(node.id)
            restored += 1
            cache._l3vt_maintain()
            drain_backups(cache)
            self._check_invariants(cache)
        self.assertGreater(restored, 0)
        # push everything out: promoted copies are re-put (or skipped as stored)
        for _ in range(40):
            n = tc.add_child(tc.root.id, [rng.randrange(1000) for _ in range(PAGE * 4)])
            land(cache, n)
            cache._l3vt_maintain()
            drain_backups(cache)
        self._check_store_bytes(store)
        self.assertGreater(store.put_skips, 0)

    def test_put_failures_free_pages_and_count(self):
        rng = random.Random(9)
        store = FakeStore()
        store.fail_rate = 0.2
        cache = make_cache(
            self, victim=True, kv_tokens=1024, state_slots=24, store=store
        )
        with self.assertLogs("l3vt-test", level="WARNING") as logs:
            self._workload(cache, rng, steps=160)
        vt = cache._l3vt
        self.assertGreater(
            vt.stats["put_kv_pages_failed"] + vt.stats["put_states_failed"], 0
        )
        self._check_store_bytes(store)
        # A short victim put's pages are freed at its ack: its warning must not
        # promise a later write of the tail, as the stock backup's does.
        short = [m for m in logs.output if " wrote " in m]
        self.assertTrue(short)
        for message in short:
            self.assertIn("HiCache L3 victim tier: put", message)
            self.assertNotIn("later write-through", message)

    def test_two_ranks_same_decisions(self):
        """Two ranks: same tree ops, different put timing; acks drained at the
        MIN of the ready counts. Victim decisions and trees stay identical."""
        rng = random.Random(21)
        ranks = [
            make_cache(self, victim=True, kv_tokens=768, state_slots=20)
            for _ in range(2)
        ]
        ready = [[], []]
        for step in range(200):
            toks = [rng.randrange(1000) for _ in range(PAGE * rng.randint(1, 5))]
            parent_choice = rng.random()
            nodes = []
            for r, c in enumerate(ranks):
                tc = c.tree_core
                cands = [n for n in tc.nodes.values() if n is not tc.root]
                cands.sort(key=lambda n: n.id)
                parent = tc.root
                if cands and parent_choice < 0.3:
                    parent = cands[int(parent_choice * 1000) % len(cands)]
                n = tc.add_child(parent.id, toks)
                p = n
                while p is not None and p is not tc.root:
                    p.last_access = tc.tick()
                    p = p.parent
                land(c, n)
                nodes.append(n.id)
            self.assertEqual(nodes[0], nodes[1])
            # each rank's backup thread runs its own number of ops
            for r, c in enumerate(ranks):
                cc = c.cache_controller
                for _ in range(
                    min(len(cc.backup_queue), [1, 3][r] if step % 2 else [2, 0][r])
                ):
                    op = cc.backup_queue.pop(0)
                    cc.run_backup(op)
                    ready[r].append(op)
            n_ack = min(len(ready[0]), len(ready[1]))
            for i in range(n_ack):
                ops = [ready[r][i] for r in range(2)]
                reqs = [
                    c._backup_requested_tokens_of(op.id) for c, op in zip(ranks, ops)
                ]
                self.assertEqual(reqs[0], reqs[1])
                agreed = min(min(op.completed_tokens, reqs[0] or 0) for op in ops)
                for c, op in zip(ranks, ops):
                    c._finish_backup_ack(
                        op,
                        requested_tokens=reqs[0],
                        agreed_tokens=agreed,
                        log_metrics=False,
                    )
            for r in range(2):
                ready[r] = ready[r][n_ack:]
            d = [c._l3vt_deficits() for c in ranks]
            self.assertEqual(d[0], d[1], step)
            for c in ranks:
                c._l3vt_maintain()
            self.assertEqual(
                ranks[0].tree_core.snapshot(), ranks[1].tree_core.snapshot(), step
            )
            self.assertEqual(
                [h.kv_pages for h in ranks[0]._l3vt.held.values()],
                [h.kv_pages for h in ranks[1]._l3vt.held.values()],
            )


Handle = __import__("collections").namedtuple("Handle", "rid generation")


class TestPinAndBudget(unittest.TestCase):
    def test_pin_survives_pressure_until_admit(self):
        rng = random.Random(4)
        cache = make_cache(
            self, victim=True, pin=True, kv_tokens=512, state_slots=16, pin_frac=0.25
        )
        tc = cache.tree_core
        hit = tc.add_child(tc.root.id, [rng.randrange(1000) for _ in range(PAGE * 8)])
        land(cache, hit)
        drain_backups(cache)
        handle = Handle("r1", 0)
        cache._l3pin_hold(handle, hit.id, len(hit.tokens))
        self.assertEqual(hit.host_lock, 1)
        for _ in range(120):  # heavy pressure
            n = tc.add_child(tc.root.id, [rng.randrange(1000) for _ in range(PAGE * 4)])
            land(cache, n)
            cache._l3vt_maintain()
            drain_backups(cache)
        self.assertIn(hit.id, tc.nodes)
        self.assertIsNotNone(tc.nodes[hit.id].host_value)
        cache.release_l3_pins([SimpleNamespace(cache_request_handle=handle)])
        self.assertEqual(hit.host_lock, 0)
        self.assertEqual(cache._l3pin.pinned_tokens, 0)
        self.assertEqual(cache._l3pin.stats["released_admit"], 1)

    def test_pin_cap_replace_abandon(self):
        cache = make_cache(self, victim=False, pin=True, kv_tokens=400, pin_frac=0.1)
        tc = cache.tree_core
        a = tc.add_child(tc.root.id, list(range(PAGE * 5)))
        b = tc.add_child(tc.root.id, list(range(100, 100 + PAGE * 20)))
        land(cache, a)
        land(cache, b)
        drain_backups(cache)  # the stock puts' host locks
        h1, h2 = Handle("1", 0), Handle("2", 0)
        cache._l3pin_hold(h1, a.id, 20)  # cap 40 tokens
        cache._l3pin_hold(h2, b.id, 80)  # over the cap: declined
        self.assertEqual((a.host_lock, b.host_lock), (1, 0))
        self.assertEqual(cache._l3pin.stats["declined_cap"], 1)
        cache._l3pin_hold(h1, b.id, 30)  # refetched: replaces the old pin
        self.assertEqual((a.host_lock, b.host_lock), (0, 1))
        self.assertEqual(cache._l3pin.stats["released_replaced"], 1)
        cache._l3pin_release(h1, "abandon")
        self.assertEqual(b.host_lock, 0)
        self.assertEqual(cache._l3pin.pinned_tokens, 0)

    def test_budget_recharge(self):
        cache = make_cache(self, victim=False)
        cache._l3vt_cfg = l3vt.L3VictimTierConfig(victim=True, budget=True)
        cc = cache.cache_controller
        key = list(range(1000))
        op = SimpleNamespace(l3vt_charged=None)
        info = SimpleNamespace(prefetch_key=key, host_indices=None)
        cc.prefetch_tokens_occupied = len(key) + 50
        self.assertEqual(cache._prefetch_occupied_span(key, None, operation=op), 1000)
        cache._l3vt_recharge_hit(info, op, 256)
        self.assertEqual(cc.prefetch_tokens_occupied, 256 + 50)
        self.assertEqual(cache._prefetch_occupied_span(key, None, operation=op), 256)
        # releasing at insert subtracts the re-charged span: back to the rest
        cc.prefetch_tokens_occupied -= cache._prefetch_occupied_span(
            key, None, operation=op
        )
        self.assertEqual(cc.prefetch_tokens_occupied, 50)
        # Tier off: the stock span, whatever the op carries.
        cache._l3vt_cfg = l3vt.L3VictimTierConfig()
        self.assertEqual(cache._prefetch_occupied_span(key, None, operation=op), 1000)
        self.assertEqual(
            cache._prefetch_occupied_span(key, None, operation=SimpleNamespace()), 1000
        )


class TestRefetchOnLoss(unittest.TestCase):
    """Scheduler._l3vt_refetch_on_loss (lifted from scheduler.py)."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _lift(
            os.path.join(_SRT, "managers/scheduler.py"),
            ["Scheduler._l3vt_refetch_on_loss"],
            "_l3vt_sched",
            {
                "get_memory": lambda: SimpleNamespace(
                    hicache_storage_prefetch_retry_max_attempts=8
                ),
                "logger": __import__("logging").getLogger("l3vt-test"),
                "Req": object,
            },
        )

    def _sched(self):
        calls = []
        notes = []
        sched = SimpleNamespace(
            tree_cache=SimpleNamespace(
                prefetch_threshold=256, l3vt_note_refetch=lambda: notes.append(1)
            ),
            _retry_storage_prefetch=lambda req: calls.append(req.rid),
        )
        return sched, calls, notes

    def _req(self, expected, prefix, host, attempts=0):
        return SimpleNamespace(
            rid="r",
            l3vt_expected_match_len=expected,
            prefix_len=prefix,
            host_hit_length=host,
            storage_prefetch_retry_attempts=attempts,
        )

    def test_decisions(self):
        f = self.mod._l3vt_refetch_on_loss
        sched, calls, notes = self._sched()
        # hit intact
        r = self._req(10_000, 8_000, 2_000)
        self.assertFalse(f(sched, r))
        self.assertIsNone(r.l3vt_expected_match_len)
        # lost less than the threshold
        self.assertFalse(f(sched, self._req(10_000, 9_800, 0)))
        # lost: reissue once, and the expectation is consumed
        r = self._req(10_000, 2_000, 0)
        self.assertTrue(f(sched, r))
        self.assertEqual((calls, len(notes)), (["r"], 1))
        self.assertIsNone(r.l3vt_expected_match_len)
        # past the reissue cap: admitted as is
        self.assertFalse(f(sched, self._req(10_000, 0, 0, attempts=8)))
        self.assertEqual(calls, ["r"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
