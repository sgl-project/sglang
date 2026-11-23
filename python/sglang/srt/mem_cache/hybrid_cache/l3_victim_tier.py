"""HiCache L3 victim tier: bookkeeping for the cache-mode storage tier.

Switch (environ.py): ``SGLANG_OPT_HICACHE_L3_VICTIM_TIER``, opt-in: off unless
set, and ``SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION`` does not turn it on. It
pays where the usable storage pool is smaller than the summed host tiers of all
servers that write to it; one worker cannot see that sum, so the operator sets
it. Cache policy only: every key and byte written or read is
the one the stock path writes for that node. All parts below engage together,
each only where its own precondition holds; a part that does not is named in
the engaged line with its reason. The tier runs on a PD prefill or an
aggregated server (``victim_tier_decline``), and every rank of the attention
TP / CP and PP groups takes the same parts (``L3VictimTierConfig.agreed``).

* Victim tier. Under write_back the host tier is backed up to storage when data
  *leaves* it, not when it lands. The stock path backs up every host landing,
  so storage holds the newest part of the host tier and, when it is no larger
  than the host tier, serves nothing the host tier does not already hold. Here
  a host landing (a backup ack, or a storage hit inserted into the host tree)
  only records which storage key each host page / state slot holds; a host
  eviction run ahead of demand (to keep a free margin) writes the evicted pages
  and slots that carry a recorded key, and keeps them allocated until the put
  is acked. Keys and bytes are exactly the ones the stock backup writes for that
  node (the record is taken from the node's storage backup spec), so this
  changes only *when* a put happens.
* Lease touch (backends with ``l3vt_touch``): once a victim put is written, its
  objects get one batch existence check.
* Chain close: the stock read path looks a request up from its deepest host /
  device node that carries a state (else from the root) and needs every KV page
  from there to a stop point, plus the state at that stop point, in storage.
  The victim tier writes a page only when it leaves the host tier, so a stop
  point whose state (or whose deepest pages) left the host while the start of
  its path stayed there had no KV page 0 in storage, and its lookup from the
  root found nothing. With chain close, such a victim put also writes every
  host-resident node on the path from the root to that stop point through the
  stock ``write_backup_storage`` (the node's own spec: KV pages, KV-derived
  sidecars, its host state if any; host-locked until acked; stored objects
  skipped by the backend's existence check), so storage holds the chain from
  the root for every stop point it holds. In-flight closes are capped (pages),
  deferred closes queue (bounded, drops counted), and a node already closed is
  re-issued only after the victim tier has written ``refresh`` x its host pool
  since.
* Pin: a storage hit inserted into the host tier stays host-locked until its
  request is admitted (or abandoned); a hit lost before admission is looked up
  again at admission.
* Budget: a lookup is charged its hit span once known, not the whole unmatched
  tail; an optional cap in tokens.
* Wait bound (timeout prefetch policy): the admission wait is at most
  base + SLACK x span / measured read rate (never above the stock bound).

This module is torch-free: the cache passes plain ints / lists, so the
decisions are unit-tested without a GPU (test_l3_victim_tier.py).
"""

from __future__ import annotations

import logging
import time
from array import array
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import msgspec

logger = logging.getLogger(__name__)

SWITCH = "SGLANG_OPT_HICACHE_L3_VICTIM_TIER"
# The parts of L3VictimTierConfig, the tier first; every rank of the lockstep
# groups must take the same ones (L3VictimTierConfig.agreed).
_PARTS = ("victim", "lease_touch", "chain_close", "pin", "budget", "wait_bound")


# ---------------------------------------------------------------------------
# Switch and constants
# ---------------------------------------------------------------------------


def victim_tier_decline(
    *,
    enable_storage: bool,
    disaggregation_mode: str,
    pd_role_switch: bool,
    host_memory_mode: str,
    has_linker: bool,
    is_write_back: bool,
    tree_core_backend: str,
    unsupported: Optional[str],
    has_storage_v2: bool,
    pass_prefix_keys: bool,
    victim_margin: float,
    state_margin: float,
) -> Optional[str]:
    """Why the victim tier cannot run on this cache, or None when it can.

    It runs where storage hits are admitted through the prefill scheduler's
    admission loop, which releases the pins and looks a lost hit up again: a PD
    prefill or an aggregated server, whose role cannot change at runtime. A PD
    decode restores its storage hits from its transfer queue instead.
    """
    if not enable_storage:
        return "storage off"
    if disaggregation_mode == "decode":
        return "PD decode role (storage hits restore outside prefill admission)"
    if pd_role_switch:
        return "PD role switch enabled (the role can change after this check)"
    if host_memory_mode != "cache":
        return f"host memory mode {host_memory_mode}"
    if has_linker:
        return "external cache linker"
    if not is_write_back:
        return "write policy is not write_back"
    if tree_core_backend != "python":
        return f"the {tree_core_backend} TreeCore"
    if unsupported:
        return unsupported
    if not has_storage_v2:
        return "backend has no batch_set_v2"
    if pass_prefix_keys:
        # A victim put batches the pages of several nodes under one op.
        return "backend takes prefix keys, which a victim put does not carry"
    if not 0.0 < victim_margin < 0.5:
        return f"KV margin {victim_margin} outside (0, 0.5)"
    if not 0.0 < state_margin < 0.5:
        return f"state margin {state_margin} outside (0, 0.5)"
    return None


class L3VictimTierConfig(msgspec.Struct, kw_only=True):
    """The resolved parts; every part is off unless the victim tier is on."""

    victim: bool = False
    lease_touch: bool = False
    chain_close: bool = False
    pin: bool = False
    budget: bool = False
    wait_bound: bool = False
    victim_margin: float = 0.05
    state_margin: float = 0.05
    chain_close_cap: float = 0.0
    chain_close_refresh: float = 0.25
    pin_frac: float = 0.25
    budget_tokens: int = 0
    wait_slack: float = 3.0
    # Why the switch is on but the tier is off (None: engaged, or switch off).
    decline: Optional[str] = None
    # Parts left off with the tier engaged, with the reason.
    ignored: Dict[str, str] = {}

    @property
    def any_on(self) -> bool:
        return self.victim

    def parts(self) -> List[int]:
        """The engaged parts as 0 / 1, in ``_PARTS`` order (for a MIN over ranks)."""
        return [int(getattr(self, part)) for part in _PARTS]

    def agreed(self, parts: Sequence[int]) -> L3VictimTierConfig:
        """This rank's config under the MIN of ``parts()`` over the ranks that move
        in lockstep: every part another rank left off is off here too, and the
        whole tier declines when another rank declined it."""
        if [int(on) for on in parts] == self.parts():
            return self
        if not parts[0]:
            # Declined here (decline set) or on another rank: every part is off.
            return msgspec.structs.replace(
                self,
                **{part: False for part in _PARTS},
                decline=self.decline or "declined on another rank",
                ignored={},
            )
        cfg = msgspec.structs.replace(self, ignored=dict(self.ignored))
        for part, on in zip(_PARTS[1:], parts[1:]):
            if getattr(cfg, part) and not on:
                setattr(cfg, part, False)
                cfg.ignored[part] = "off on another rank"
        return cfg

    @classmethod
    def resolve(
        cls,
        envs: Any,
        *,
        enable_storage: bool,
        disaggregation_mode: str,
        pd_role_switch: bool,
        host_memory_mode: str,
        has_linker: bool,
        is_write_back: bool,
        tree_core_backend: str,
        victim_unsupported: Optional[str],
        prefetch_stop_policy: str,
        has_storage_v2: bool,
        pass_prefix_keys: bool,
        has_lease_touch: bool = False,
    ) -> L3VictimTierConfig:
        """Resolve the switch against the cache's mode; the parts follow the tier."""
        cfg = cls(
            victim_margin=float(envs.SGLANG_OPT_HICACHE_L3_VICTIM_TIER_KV_MARGIN.get()),
            state_margin=float(
                envs.SGLANG_OPT_HICACHE_L3_VICTIM_TIER_STATE_MARGIN.get()
            ),
            chain_close_cap=float(
                envs.SGLANG_OPT_HICACHE_L3_VICTIM_TIER_CLOSE_CAP.get()
            ),
            chain_close_refresh=float(
                envs.SGLANG_OPT_HICACHE_L3_VICTIM_TIER_CLOSE_REFRESH.get()
            ),
            pin_frac=float(envs.SGLANG_OPT_HICACHE_L3_VICTIM_TIER_PIN_FRAC.get()),
            budget_tokens=int(
                envs.SGLANG_OPT_HICACHE_L3_VICTIM_TIER_BUDGET_TOKENS.get()
            ),
            wait_slack=float(envs.SGLANG_OPT_HICACHE_L3_VICTIM_TIER_WAIT_SLACK.get()),
        )
        if not envs.SGLANG_OPT_HICACHE_L3_VICTIM_TIER.get():
            return cfg
        cfg.decline = victim_tier_decline(
            enable_storage=enable_storage,
            disaggregation_mode=disaggregation_mode,
            pd_role_switch=pd_role_switch,
            host_memory_mode=host_memory_mode,
            has_linker=has_linker,
            is_write_back=is_write_back,
            tree_core_backend=tree_core_backend,
            unsupported=victim_unsupported,
            has_storage_v2=has_storage_v2,
            pass_prefix_keys=pass_prefix_keys,
            victim_margin=cfg.victim_margin,
            state_margin=cfg.state_margin,
        )
        if cfg.decline is not None:
            return cfg
        cfg.victim = True
        if has_lease_touch:
            cfg.lease_touch = True
        else:
            cfg.ignored["lease_touch"] = "backend has no lease touch"
        if not 0.0 <= cfg.chain_close_cap < 0.5:
            cfg.ignored["chain_close"] = f"cap {cfg.chain_close_cap} outside [0, 0.5)"
        elif not cfg.chain_close_refresh > 0.0:
            cfg.ignored["chain_close"] = f"refresh {cfg.chain_close_refresh} <= 0"
        else:
            cfg.chain_close = True
        if 0.0 < cfg.pin_frac <= 0.5:
            cfg.pin = True
        else:
            cfg.ignored["pin"] = f"pin share {cfg.pin_frac} outside (0, 0.5]"
        if cfg.budget_tokens >= 0:
            cfg.budget = True
        else:
            cfg.ignored["budget"] = f"budget tokens {cfg.budget_tokens} < 0"
        if prefetch_stop_policy != "timeout":
            cfg.ignored["wait_bound"] = f"prefetch policy {prefetch_stop_policy}"
        elif cfg.wait_slack < 1.0:
            cfg.ignored["wait_bound"] = f"slack {cfg.wait_slack} < 1"
        else:
            cfg.wait_bound = True
        return cfg


# ---------------------------------------------------------------------------
# Victim tier
# ---------------------------------------------------------------------------


class VictimKeyMap:
    """Storage key of the KV page in each host page, and of the state in each
    host state slot, for pages / slots the tree holds.

    Invariant kept by the cache: an entry is set only from a node's storage
    backup spec (exactly what the stock backup would put under that key), and
    is cleared whenever the tree frees the page / slot. A page without an entry
    is never written.
    """

    def __init__(self, num_kv_pages: int, num_state_slots: int):
        self._kv: List[Optional[str]] = [None] * max(0, int(num_kv_pages))
        self._st: List[Optional[str]] = [None] * max(0, int(num_state_slots))
        self.kv_entries = 0
        self.st_entries = 0

    @staticmethod
    def _grow(table: List[Optional[str]], idx: int) -> None:
        if idx >= len(table):
            table.extend([None] * (idx + 1 - len(table)))

    def record_kv(self, pages: Sequence[int], keys: Sequence[str]) -> int:
        if len(pages) != len(keys):
            raise ValueError(f"{len(pages)} pages vs {len(keys)} keys")
        kv = self._kv
        for p, k in zip(pages, keys):
            if k is None:
                continue
            if p >= len(kv):
                self._grow(kv, p)
            if kv[p] is None:
                self.kv_entries += 1
            kv[p] = k
        return len(pages)

    def record_states(self, slots: Sequence[int], keys: Sequence[str]) -> int:
        if len(slots) != len(keys):
            raise ValueError(f"{len(slots)} slots vs {len(keys)} keys")
        st = self._st
        for s, k in zip(slots, keys):
            if k is None:
                continue
            if s >= len(st):
                self._grow(st, s)
            if st[s] is None:
                self.st_entries += 1
            st[s] = k
        return len(slots)

    def kv_key(self, page: int) -> Optional[str]:
        return self._kv[page] if 0 <= page < len(self._kv) else None

    def state_key(self, slot: int) -> Optional[str]:
        return self._st[slot] if 0 <= slot < len(self._st) else None

    def clear_kv(self, pages: Iterable[int]) -> int:
        kv = self._kv
        cleared = 0
        for p in pages:
            if 0 <= p < len(kv) and kv[p] is not None:
                kv[p] = None
                cleared += 1
        self.kv_entries -= cleared
        return cleared

    def clear_states(self, slots: Iterable[int]) -> int:
        st = self._st
        cleared = 0
        for s in slots:
            if 0 <= s < len(st) and st[s] is not None:
                st[s] = None
                cleared += 1
        self.st_entries -= cleared
        return cleared

    def split_kv(self, pages: Sequence[int]) -> Tuple[List[int], List[str], List[int]]:
        """(keyed page positions, their keys, unkeyed page positions), as
        positions into ``pages``."""
        keyed, keys, unkeyed = [], [], []
        for i, p in enumerate(pages):
            k = self.kv_key(p)
            if k is None:
                unkeyed.append(i)
            else:
                keyed.append(i)
                keys.append(k)
        return keyed, keys, unkeyed

    def split_states(
        self, slots: Sequence[int]
    ) -> Tuple[List[int], List[str], List[int]]:
        keyed, keys, unkeyed = [], [], []
        for i, s in enumerate(slots):
            k = self.state_key(s)
            if k is None:
                unkeyed.append(i)
            else:
                keyed.append(i)
                keys.append(k)
        return keyed, keys, unkeyed

    def reset(self) -> None:
        self._kv = [None] * len(self._kv)
        self._st = [None] * len(self._st)
        self.kv_entries = 0
        self.st_entries = 0


class HeldVictim(msgspec.Struct, kw_only=True):
    """A victim put in flight: its pages / slots stay allocated until the ack."""

    kv_tokens: Any  # the cache's token-index tensor (opaque here)
    state_slots: Any  # the cache's slot tensor (opaque here)
    kv_pages: List[int]
    slots: List[int]
    num_kv_tokens: int
    issued_at: float


class VictimTier:
    """Margins, the held set and counters of the victim tier.

    The cache keeps ``kv_target`` host KV tokens and ``state_target`` host state
    slots free or in flight: it evicts the deficit through the tree core's own
    host LRU, writes the evicted pages / slots that carry a recorded key (one
    put), and frees them when that put is acked. ``held`` never exceeds the
    targets, so the tier holds no memory beyond the margin it reserves inside
    the existing host pools.
    """

    def __init__(
        self,
        *,
        page_size: int,
        kv_capacity_tokens: int,
        state_capacity_slots: int,
        margin: float,
        state_margin: Optional[float] = None,
        stats_interval_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.page_size = int(page_size)
        num_pages = kv_capacity_tokens // max(1, self.page_size)
        self.keymap = VictimKeyMap(num_pages, state_capacity_slots)
        self.margin = float(margin)
        self.state_margin = float(margin if state_margin is None else state_margin)
        # Page-aligned, at least one page; at least one state slot when the
        # model has a state pool.
        self.kv_target = max(
            self.page_size,
            (int(kv_capacity_tokens * self.margin) // self.page_size) * self.page_size,
        )
        self.state_target = (
            max(1, int(state_capacity_slots * self.state_margin))
            if state_capacity_slots > 0
            else 0
        )
        self.held: Dict[int, HeldVictim] = {}
        self.held_kv_tokens = 0
        self.held_states = 0
        self._clock = clock
        self._stats_interval_s = stats_interval_s
        self._last_stats = clock()
        self.stats: Dict[str, int] = {
            "recorded_nodes": 0,
            "recorded_pages": 0,
            "recorded_states": 0,
            "record_skipped": 0,
            "passes": 0,
            "evicted_kv_tokens": 0,
            "evicted_states": 0,
            "put_ops": 0,
            "put_kv_pages": 0,
            "put_states": 0,
            "unkeyed_kv_pages": 0,
            "unkeyed_states": 0,
            "acked_ops": 0,
            "acked_kv_pages_ok": 0,
            "acked_states_ok": 0,
            "put_kv_pages_failed": 0,
            "put_states_failed": 0,
            "dropped_held_cap_kv_pages": 0,
            "dropped_held_cap_states": 0,
            "freed_keyed_kv_pages": 0,
            "freed_keyed_states": 0,
            # States put without their stop-point KV page in the same op (the
            # host state pool dropped them while their KV stayed in host).
            "put_states_detached": 0,
            # Lease touch (read from the backup thread's result).
            "touch_ops": 0,
            "touch_objects": 0,
            "touch_present": 0,
            "touch_missing": 0,
            "touch_failed": 0,
        }

    # -- margins --

    def kv_deficit(self, available_tokens: int) -> int:
        """Host KV tokens to evict now (page-aligned, >= 0)."""
        deficit = self.kv_target - int(available_tokens) - self.held_kv_tokens
        if deficit <= 0:
            return 0
        ps = self.page_size
        return ((deficit + ps - 1) // ps) * ps

    def state_deficit(self, available_slots: int) -> int:
        if self.state_target <= 0:
            return 0
        return max(0, self.state_target - int(available_slots) - self.held_states)

    def held_room(self) -> Tuple[int, int]:
        """(KV tokens, state slots) that may still be held."""
        return (
            max(0, self.kv_target - self.held_kv_tokens),
            max(0, self.state_target - self.held_states)
            if self.state_target > 0
            else 0,
        )

    # -- held set --

    def hold(self, op_id: int, held: HeldVictim) -> None:
        assert op_id not in self.held, f"victim op {op_id} held twice"
        self.held[op_id] = held
        self.held_kv_tokens += held.num_kv_tokens
        self.held_states += len(held.slots)

    def note_touch(self, result: Optional[Dict[str, int]]) -> None:
        """Count one victim op's post-put touch (None: the touch raised or the
        op carried none)."""
        if result is None:
            return
        if result.get("failed"):
            self.stats["touch_failed"] += 1
            return
        st = self.stats
        st["touch_ops"] += 1
        st["touch_objects"] += int(result.get("objects", 0))
        st["touch_present"] += int(result.get("present", 0))
        st["touch_missing"] += int(result.get("objects", 0)) - int(
            result.get("present", 0)
        )

    def release(self, op_id: int) -> Optional[HeldVictim]:
        held = self.held.pop(op_id, None)
        if held is None:
            return None
        self.held_kv_tokens -= held.num_kv_tokens
        self.held_states -= len(held.slots)
        self.keymap.clear_kv(held.kv_pages)
        self.keymap.clear_states(held.slots)
        return held

    def reset(self) -> None:
        """Drop all state without freeing (the caller cleared the pools)."""
        self.held.clear()
        self.held_kv_tokens = 0
        self.held_states = 0
        self.keymap.reset()

    # -- stats --

    def maybe_log(self, extra: str = "") -> None:
        now = self._clock()
        if now - self._last_stats < self._stats_interval_s:
            return
        self._last_stats = now
        logger.info(
            "HiCache L3 victim tier: %s held_kv_tokens=%d held_states=%d "
            "kv_target=%d state_target=%d map_kv=%d map_states=%d%s",
            " ".join(f"{k}={v}" for k, v in self.stats.items()),
            self.held_kv_tokens,
            self.held_states,
            self.kv_target,
            self.state_target,
            self.keymap.kv_entries,
            self.keymap.st_entries,
            extra,
        )


# ---------------------------------------------------------------------------
# Chain close (victim tier)
# ---------------------------------------------------------------------------

_NO_NODE = -1


class ChainCloser:
    """Bookkeeping of ``_CHAIN_CLOSE``; the cache does the tree work (path walk,
    ``write_backup_storage``, locks) through callbacks, so this stays
    torch-free and every decision is a function of state that is identical on
    all ranks (the tree, the key map, the victim ops, counters advanced at
    issue and at the MIN-synced ack drain; never a clock).

    Triggers (one victim op):

    * a state put whose node is still in the tree (it left the host state
      pool while its KV stayed: "detached"): anchor = that node;
    * a state put whose node left the host tier in the same op (a host leaf,
      or a tombstone ancestor deleted with it): anchor = the deepest live node
      before its KV run (below);
    * a KV run of a node whose state was put detached earlier: anchor as
      above (the path start may have stayed while the deepest pages left).

    Anchor of a node that left the tree: the node holding the page recorded
    just before the run's first page, if that page still holds the same key
    (a split fragment that stayed, once seen on a recorded path); else the
    parent last seen for the node, walking up past nodes that left too.

    Close of an anchor: every node on its root path that is host-backed, not
    in a write-through, not in flight and not closed within the refresh
    window, root first, through ``write_backup_storage``; pages in flight are
    capped (``cap_pages``), and an anchor that does not fit waits in a bounded
    queue (overflow drops the oldest, counted).
    """

    MAX_WALK = 256
    SWEEP_PER_PASS = 256
    SWEEP_FLOOR = 4096

    def __init__(
        self,
        *,
        page_size: int,
        num_kv_pages: int,
        num_state_slots: int,
        cap_pages: int,
        refresh_pages: int,
        max_pending: int,
        kv_page_bytes: int = 0,
        state_bytes: int = 0,
    ):
        self.page_size = int(page_size)
        n = max(0, int(num_kv_pages))
        # Per host page, as recorded: the node holding it, the page before it
        # in that node (-1 for the node's first page) and that page's key.
        self.kv_node = array("q", [_NO_NODE]) * n
        self.kv_prev = array("q", [-1]) * n
        self.kv_prev_key: List[Optional[str]] = [None] * n
        # Per host state slot, as recorded: the node holding it.
        self.st_node = array("q", [_NO_NODE]) * max(0, int(num_state_slots))
        # node id -> [parent id last seen, refresh clock at its last close (-1)]
        self.nodes: Dict[int, List[int]] = {}
        # Nodes whose state was put while their KV stayed in host.
        self.st_put_nodes: set = set()
        self.queue: deque = deque()
        self.queued: set = set()
        self.inflight: Dict[int, Tuple[int, int]] = {}
        self.inflight_nodes: Dict[int, int] = {}
        self.inflight_pages = 0
        # Pages issued by the victim tier and by chain close: the refresh clock.
        self.clock_pages = 0
        self.cap_pages = max(1, int(cap_pages))
        self.refresh_pages = max(1, int(refresh_pages))
        self.max_pending = max(1, int(max_pending))
        self.kv_page_bytes = max(0, int(kv_page_bytes))
        self.state_bytes = max(0, int(state_bytes))
        self._sweep_keys: List[int] = []
        self._sweep_at = self.SWEEP_FLOOR
        self.stats: Dict[str, int] = {
            "cc_trig_state": 0,
            "cc_trig_detached": 0,
            "cc_trig_kv": 0,
            "cc_state_unknown": 0,
            "cc_anchor_self": 0,
            "cc_anchor_pred": 0,
            "cc_anchor_hint": 0,
            "cc_anchor_none": 0,
            "cc_frag_unresolved": 0,
            "cc_enqueued": 0,
            "cc_dup": 0,
            "cc_closes": 0,
            "cc_close_empty": 0,
            "cc_path_nodes": 0,
            "cc_ops": 0,
            "cc_pages": 0,
            "cc_max_pages_pass": 0,
            "cc_skip_closed": 0,
            "cc_skip_inflight": 0,
            "cc_skip_write_through": 0,
            "cc_skip_not_host": 0,
            "cc_skip_not_host_pages": 0,
            "cc_skip_no_spec": 0,
            "cc_deferred": 0,
            "cc_cap_waits": 0,
            "cc_victim_waits": 0,
            "cc_dropped": 0,
            "cc_acked_ops": 0,
            "cc_acked_pages_ok": 0,
            "cc_failed_pages": 0,
            "cc_states": 0,
            "cc_states_ok": 0,
            "cc_states_failed": 0,
            "cc_bytes": 0,
            "cc_touch_ops": 0,
            "cc_touch_missing": 0,
            "cc_touch_failed": 0,
            "cc_refreshed_nodes": 0,
            "cc_swept": 0,
            "cc_errors": 0,
        }
        self._pass_pages = 0

    # -- per-page / per-slot / per-node records --

    def _ensure_page(self, page: int) -> None:
        if page >= len(self.kv_node):
            grow = page + 1 - len(self.kv_node)
            self.kv_node.extend(array("q", [_NO_NODE]) * grow)
            self.kv_prev.extend(array("q", [-1]) * grow)
            self.kv_prev_key.extend([None] * grow)

    def _ensure_slot(self, slot: int) -> None:
        if slot >= len(self.st_node):
            self.st_node.extend(array("q", [_NO_NODE]) * (slot + 1 - len(self.st_node)))

    def note_node(self, node_id: int, parent_id: int) -> bool:
        """Record or refresh a node's parent; True if the node is new here."""
        entry = self.nodes.get(node_id)
        if entry is None:
            self.nodes[node_id] = [parent_id, -1]
            return True
        entry[0] = parent_id
        return False

    def note_pages(
        self, node_id: int, pages: Sequence[int], keys: Sequence[str]
    ) -> None:
        prev, prev_key = -1, None
        for page, key in zip(pages, keys):
            self._ensure_page(page)
            self.kv_node[page] = node_id
            self.kv_prev[page] = prev
            self.kv_prev_key[page] = prev_key
            prev, prev_key = page, key

    def note_states(self, node_id: int, slots: Sequence[int]) -> None:
        for slot in slots:
            self._ensure_slot(slot)
            self.st_node[slot] = node_id

    def mark_closed(self, node_id: int) -> None:
        entry = self.nodes.get(node_id)
        if entry is not None:
            entry[1] = self.clock_pages

    def needs_close(self, node_id: int) -> Optional[str]:
        """None if the node should be (re)written now, else why not."""
        if node_id in self.inflight_nodes:
            return "inflight"
        entry = self.nodes.get(node_id)
        if (
            entry is not None
            and entry[1] >= 0
            and self.clock_pages - entry[1] < self.refresh_pages
        ):
            return "closed"
        return None

    # -- triggers and anchors --

    def resolve(
        self,
        node_id: int,
        run_first_page: Optional[int],
        op_pages,
        kv_key: Callable[[int], Optional[str]],
        alive: Callable[[int], Optional[bool]],
        held: Callable[[int], bool],
    ) -> Optional[int]:
        """The deepest live node on the root path of a node that left the
        tree (``alive`` returns None for a node no longer in the tree)."""
        st = self.stats
        start = node_id
        if run_first_page is not None and 0 <= run_first_page < len(self.kv_prev):
            q = self.kv_prev[run_first_page]
            want = self.kv_prev_key[run_first_page]
            if q >= 0 and want is not None and kv_key(q) == want:
                cand = self.kv_node[q]
                if cand != _NO_NODE:
                    if alive(cand) is not None:
                        st["cc_anchor_pred"] += 1
                        return cand
                    if q not in op_pages and not held(q):
                        # The page before the run is still in the host tree
                        # under a node never seen on a recorded path (a split
                        # fragment): the parent walk below misses it.
                        st["cc_frag_unresolved"] += 1
                    start = cand
        n = start
        for _ in range(self.MAX_WALK):
            entry = self.nodes.get(n)
            if entry is None:
                break
            n = entry[0]
            if alive(n) is not None:
                st["cc_anchor_hint"] += 1
                return n
        st["cc_anchor_none"] += 1
        return None

    def triggers(
        self,
        kv_pages: Sequence[int],
        slots: Sequence[int],
        kv_key: Callable[[int], Optional[str]],
        alive: Callable[[int], Optional[bool]],
        held: Callable[[int], bool],
    ) -> List[int]:
        """Anchors for one victim op (its keyed KV pages, each node's pages in
        chain order, and its keyed state slots). Call before the op's keys
        are cleared (they are, at its ack)."""
        st = self.stats
        self.clock_pages += len(kv_pages)
        op_pages = set(kv_pages)
        run_start: Dict[int, int] = {}
        start = None
        for p in kv_pages:
            q = self.kv_prev[p] if p < len(self.kv_prev) else -1
            if start is None or q < 0 or q not in op_pages:
                start = p
            n = self.kv_node[p] if p < len(self.kv_node) else _NO_NODE
            if n != _NO_NODE:
                run_start.setdefault(n, start)
        anchors: List[int] = []
        for s in slots:
            n = self.st_node[s] if 0 <= s < len(self.st_node) else _NO_NODE
            if n == _NO_NODE:
                st["cc_state_unknown"] += 1
                continue
            st["cc_trig_state"] += 1
            if alive(n) is not None:
                st["cc_trig_detached"] += 1
                st["cc_anchor_self"] += 1
                self.st_put_nodes.add(n)
                anchors.append(n)
                continue
            a = self.resolve(n, run_start.get(n), op_pages, kv_key, alive, held)
            if a is not None:
                anchors.append(a)
        for n, p in run_start.items():
            if n not in self.st_put_nodes:
                continue
            self.st_put_nodes.discard(n)
            st["cc_trig_kv"] += 1
            if alive(n) is not None:
                st["cc_anchor_self"] += 1
                anchors.append(n)
                continue
            a = self.resolve(n, p, op_pages, kv_key, alive, held)
            if a is not None:
                anchors.append(a)
        return list(dict.fromkeys(anchors))

    # -- queue, cap, issue, ack --

    def enqueue(self, anchors: Iterable[int]) -> None:
        st = self.stats
        for a in anchors:
            if a in self.queued:
                st["cc_dup"] += 1
                continue
            self.queue.append(a)
            self.queued.add(a)
            st["cc_enqueued"] += 1
        while len(self.queue) > self.max_pending:
            self.queued.discard(self.queue.popleft())
            st["cc_dropped"] += 1

    def can_issue(self, pages: int) -> bool:
        return self.inflight_pages == 0 or self.inflight_pages + pages <= self.cap_pages

    def run(self, close: Callable[[int], bool], victim_backlog: bool = False) -> None:
        """Close queued anchors, oldest first, while pages in flight fit the
        cap; ``close(anchor)`` returns False when it stopped at the cap (the
        anchor stays at the head, its issued nodes in flight). With the victim
        puts backed up (``victim_backlog``) nothing new is issued: close ops
        share the backup queue, and a victim put waiting behind them keeps its
        pages out of the host margin."""
        st = self.stats
        self._pass_pages = 0
        if victim_backlog and self.queue:
            st["cc_victim_waits"] += 1
            return
        while self.queue:
            if self.inflight_pages >= self.cap_pages:
                st["cc_cap_waits"] += 1
                break
            a = self.queue[0]
            try:
                done = close(a)
            except Exception:  # noqa: BLE001 - a lost close is not a lost write
                logger.warning("HiCache L3 chain close failed", exc_info=True)
                st["cc_errors"] += 1
                done = True
            if not done:
                st["cc_deferred"] += 1
                break
            self.queue.popleft()
            self.queued.discard(a)
            st["cc_closes"] += 1
        if self._pass_pages > st["cc_max_pages_pass"]:
            st["cc_max_pages_pass"] = self._pass_pages

    def issued(self, op_id: int, node_id: int, pages: int) -> None:
        st = self.stats
        self.inflight[op_id] = (node_id, pages)
        self.inflight_nodes[node_id] = self.inflight_nodes.get(node_id, 0) + 1
        self.inflight_pages += pages
        self.clock_pages += pages
        self._pass_pages += pages
        entry = self.nodes.get(node_id)
        if entry is None:
            self.nodes[node_id] = entry = [_NO_NODE, -1]
        entry[1] = self.clock_pages
        st["cc_ops"] += 1
        st["cc_pages"] += pages

    def note_ack(
        self,
        op_id: int,
        ok_pages: int,
        states: int,
        states_ok: int,
        touch_result: Optional[Dict[str, int]],
    ) -> bool:
        rec = self.inflight.pop(op_id, None)
        if rec is None:
            return False
        node_id, pages = rec
        left = self.inflight_nodes.get(node_id, 1) - 1
        if left > 0:
            self.inflight_nodes[node_id] = left
        else:
            self.inflight_nodes.pop(node_id, None)
        self.inflight_pages -= pages
        st = self.stats
        ok = max(0, min(pages, int(ok_pages)))
        st["cc_acked_ops"] += 1
        st["cc_acked_pages_ok"] += ok
        st["cc_failed_pages"] += pages - ok
        st["cc_states"] += states
        st["cc_states_ok"] += min(states, states_ok)
        st["cc_states_failed"] += states - min(states, states_ok)
        st["cc_bytes"] += pages * self.kv_page_bytes + states * self.state_bytes
        if touch_result is not None:
            if touch_result.get("failed"):
                st["cc_touch_failed"] += 1
            else:
                st["cc_touch_ops"] += 1
                st["cc_touch_missing"] += int(touch_result.get("objects", 0)) - int(
                    touch_result.get("present", 0)
                )
        return True

    # -- housekeeping --

    def sweep(self, alive: Callable[[int], Optional[bool]]) -> None:
        """Drop records of nodes no longer in the tree, a bounded number a
        pass, once the node table has doubled since the last full sweep."""
        if not self._sweep_keys:
            if len(self.nodes) < self._sweep_at:
                return
            self._sweep_keys = list(self.nodes) + [
                n for n in self.st_put_nodes if n not in self.nodes
            ]
        batch = self._sweep_keys[-self.SWEEP_PER_PASS :]
        del self._sweep_keys[-self.SWEEP_PER_PASS :]
        for n in batch:
            if n in self.inflight_nodes or n in self.queued:
                continue
            if alive(n) is None:
                self.nodes.pop(n, None)
                self.st_put_nodes.discard(n)
                self.stats["cc_swept"] += 1
        if not self._sweep_keys:
            self._sweep_at = max(self.SWEEP_FLOOR, 2 * len(self.nodes))

    def reset(self) -> None:
        """Drop all state (the caller cleared the tree, pools and ops)."""
        self.kv_node = array("q", [_NO_NODE]) * len(self.kv_node)
        self.kv_prev = array("q", [-1]) * len(self.kv_prev)
        self.kv_prev_key = [None] * len(self.kv_prev_key)
        self.st_node = array("q", [_NO_NODE]) * len(self.st_node)
        self.nodes.clear()
        self.st_put_nodes.clear()
        self.queue.clear()
        self.queued.clear()
        self.inflight.clear()
        self.inflight_nodes.clear()
        self.inflight_pages = 0
        self._sweep_keys = []
        self._sweep_at = self.SWEEP_FLOOR

    def log_items(self) -> List[Tuple[str, int]]:
        return list(self.stats.items()) + [
            ("cc_inflight_ops", len(self.inflight)),
            ("cc_inflight_pages", self.inflight_pages),
            ("cc_queue", len(self.queue)),
            ("cc_nodes", len(self.nodes)),
            ("cc_st_put_nodes", len(self.st_put_nodes)),
            ("cc_cap_pages", self.cap_pages),
            ("cc_refresh_pages", self.refresh_pages),
        ]


# ---------------------------------------------------------------------------
# Pinned restores
# ---------------------------------------------------------------------------


class PinEntry(msgspec.Struct):
    node_id: Any
    lock_params: Any
    tokens: int


class PinLedger:
    """Storage hits held in the host tier until their request is admitted."""

    def __init__(self, cap_tokens: int):
        self.cap_tokens = max(0, int(cap_tokens))
        self.pins: Dict[Any, PinEntry] = {}
        self.pinned_tokens = 0
        self.stats: Dict[str, int] = {
            "pinned": 0,
            "declined_cap": 0,
            "released_admit": 0,
            "released_abandon": 0,
            "released_replaced": 0,
            "refetch_on_loss": 0,
        }

    def can_pin(self, tokens: int) -> bool:
        return self.pinned_tokens + int(tokens) <= self.cap_tokens

    def add(self, handle: Any, entry: PinEntry) -> None:
        assert handle not in self.pins, f"pin for {handle} exists"
        self.pins[handle] = entry
        self.pinned_tokens += entry.tokens
        self.stats["pinned"] += 1

    def pop(self, handle: Any, reason: str) -> Optional[PinEntry]:
        entry = self.pins.pop(handle, None)
        if entry is None:
            return None
        self.pinned_tokens -= entry.tokens
        self.stats[f"released_{reason}"] += 1
        return entry

    def reset(self) -> None:
        self.pins.clear()
        self.pinned_tokens = 0


# ---------------------------------------------------------------------------
# Wait bound
# ---------------------------------------------------------------------------


class WaitBound:
    """EWMA of the storage read rate (tokens / s, queueing included, so it
    errs long) and the admission wait bound derived from it."""

    MIN_OBS = 4

    def __init__(self, slack: float, alpha: float = 0.2):
        self.slack = float(slack)
        self.alpha = float(alpha)
        self.rate: Optional[float] = None
        self.n_obs = 0
        self.stats: Dict[str, int] = {"bounded": 0, "stock": 0}

    def observe(self, tokens: int, seconds: float) -> None:
        if tokens <= 0 or seconds <= 0:
            return
        r = tokens / seconds
        self.rate = (
            r if self.rate is None else (1 - self.alpha) * self.rate + self.alpha * r
        )
        self.n_obs += 1

    def timeout(self, span_tokens: int, base_s: float, stock_s: float) -> float:
        if self.rate is None or self.n_obs < self.MIN_OBS or span_tokens <= 0:
            self.stats["stock"] += 1
            return stock_s
        bound = base_s + self.slack * span_tokens / self.rate
        if bound < stock_s:
            self.stats["bounded"] += 1
            return bound
        self.stats["stock"] += 1
        return stock_s
