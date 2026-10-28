"""Operation traces over the unified allocators, for comparing implementations.

An operation names a request, never an id (``extend(req, n)``, ``free(req)``),
so one stream replays on any implementation; an adapter maps it to one
revision's allocator API and reads the observables after every operation.
`record_run` draws a seeded stream against a live adapter (generation reads
the outcomes) or replays a scenario file; `replay_run` replays a recorded
stream on another adapter and returns the first divergence.

Canonicalisation of the observables: hole pages and pending-reuse pages are
sorted; the virtual-to-physical tables, the free virtual id list (the rebind
order) and returned ids are compared in order. Each observable is stored as a
short digest, or in full with ``full=True``.
"""

import collections
import hashlib
import json
import random
from array import array
from typing import Callable, Dict, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.tree_slots import tree_slots_of
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    pin_every_band,
    setup_allocator_context,
)

FORMAT_VERSION = 1
CHAINS = ("swa", "mamba", "mamba_mla", "tri")
WINDOW_CHAINS = ("swa", "tri")
STATE_CHAINS = ("mamba", "mamba_mla", "tri")


class TraceConfig(msgspec.Struct, frozen=True, kw_only=True):
    chain: str
    page_size: int
    lazy: bool
    # Paged extend and decode run device kernels; on "cpu" the stream uses
    # plain token allocation instead.
    device: str = "cuda"

    def key(self) -> str:
        return f"{self.chain}/ps{self.page_size}/lazy{int(self.lazy)}/{self.device}"


def trace_geometry(config: TraceConfig) -> UnifiedGeometry:
    """Two layers per kind, 64 pages per token sub-pool, 8 state slots."""
    ps = config.page_size
    chain = "mamba" if config.chain == "mamba_mla" else config.chain
    return UnifiedGeometry(
        chain=chain,
        full_layer_ids=(0, 1),
        swa_layer_ids=(2, 3) if chain in WINDOW_CHAINS else (),
        state_layer_ids=(4, 5) if chain in ("mamba", "tri") else (),
        mla=config.chain == "mamba_mla",
        page_size=ps,
        full_tokens=64 * ps,
        swa_tokens=64 * ps,
        context_len=256 * ps,
        window=32,
        lazy=config.lazy,
        device=config.device,
    )


class FakeEvent:
    """A forward-done event whose completion the trace controls."""

    def __init__(self):
        self.fired = False

    def query(self) -> bool:
        return self.fired

    def synchronize(self) -> None:
        self.fired = True

    def wait(self, stream=None) -> None:
        """Orders later device work after the forward; completion on the host
        still waits for `fire`."""


class _Request(msgspec.Struct, kw_only=True):
    tokens: List[int] = []
    retired: int = 0  # tokens whose window side is released


class HostModel(msgspec.Struct, kw_only=True):
    """What the trace knows about live requests and state slots; ids are the
    adapter's, so two adapters replaying one stream keep separate models."""

    requests: Dict[int, _Request] = {}
    states: List[int] = []
    next_req: int = 0
    cached: int = 0


class BuilderAdapter:
    """The adapter for the allocator the unified pool builder builds."""

    name = "builder"

    def __init__(self, config: TraceConfig):
        setup_allocator_context()
        self.config = config
        self.device = config.device
        self.page_size = config.page_size
        self.bundle = build_unified_bundle(trace_geometry(config))
        self.allocator = self.bundle.token_to_kv_pool_allocator
        self.state_allocator = (
            self.bundle.token_to_kv_pool_allocator.state_slots
            if config.chain in STATE_CHAINS
            else None
        )
        self.gate_open = True
        self.events: List[FakeEvent] = []
        self.tree = self._build_tree() if config.chain == "swa" else None
        if config.lazy:
            pin_every_band(self.allocator, lambda: not self.gate_open)

    def _build_tree(self) -> UnifiedRadixCache:
        return UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=4,
                    max_context_len=256 * self.page_size,
                    device=self.device,
                    enable_memory_saver=False,
                ),
                token_to_kv_pool_allocator=self.allocator,
                page_size=self.page_size,
                sliding_window_size=32,
                tree_slots=tree_slots_of(
                    (ComponentType.FULL, ComponentType.SWA), window=32
                ),
            )
        )

    def members(self) -> List[object]:
        member = self.allocator.full_attn_allocator
        while member.low_peer is not None:
            member = member.low_peer
        out = []
        while member is not None:
            out.append(member)
            member = member.high_peer
        return out

    # -- operations; each returns the ids it produced, or None on refusal --

    def extend(self, tokens: List[int], n: int) -> Optional[List[int]]:
        if self.page_size == 1:
            # Production allocates page-1 pools token by token (`alloc_token_slots`).
            return self.alloc(n)
        prefix = len(tokens)
        out = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([prefix], dtype=torch.int64, device=self.device),
            prefix_lens_cpu=torch.tensor([prefix], dtype=torch.int64),
            seq_lens=torch.tensor([prefix + n], dtype=torch.int64, device=self.device),
            seq_lens_cpu=torch.tensor([prefix + n], dtype=torch.int64),
            last_loc=torch.tensor(
                [tokens[-1] if tokens else -1], dtype=torch.int64, device=self.device
            ),
            extend_num_tokens=n,
        )
        return None if out is None else out.tolist()

    def decode(self, token_lists: List[List[int]]) -> Optional[List[int]]:
        if self.page_size == 1:
            return self.alloc(len(token_lists))
        seq = torch.tensor(
            [len(t) + 1 for t in token_lists], dtype=torch.int64, device=self.device
        )
        out = self.allocator.alloc_decode(
            seq_lens=seq,
            seq_lens_cpu=seq.cpu(),
            last_loc=torch.tensor(
                [t[-1] for t in token_lists], dtype=torch.int64, device=self.device
            ),
        )
        return None if out is None else out.tolist()

    def alloc(self, n: int) -> Optional[List[int]]:
        out = self.allocator.alloc(n)
        return None if out is None else out.tolist()

    def free(self, ids: List[int]) -> None:
        self.allocator.free(torch.tensor(ids, dtype=torch.int64, device=self.device))

    def retire(self, ids: List[int]) -> None:
        self.allocator.free_swa(
            torch.tensor(ids, dtype=torch.int64, device=self.device)
        )

    def free_segment(self, ids: List[int], start: int) -> None:
        self.allocator.free_segment(
            torch.tensor(ids, dtype=torch.int64, device=self.device), start_pos=start
        )

    def group(self, begin: bool) -> None:
        if begin:
            self.allocator.free_group_begin()
        else:
            self.allocator.free_group_end()

    def state_alloc(self, n: int) -> Optional[List[int]]:
        out = self.state_allocator.alloc(n)
        return None if out is None else out.tolist()

    def state_free(self, ids: List[int]) -> None:
        self.state_allocator.free(
            torch.tensor(ids, dtype=torch.int64, device=self.device)
        )

    def flush(self, member: Optional[str]) -> None:
        if member is None:
            self.allocator.flush_opportunistic()
        else:
            self._member(member).flush_for_allocation()

    def ensure(self, full: int, window: int) -> bool:
        return bool(self.allocator.ensure_capacity(full, window))

    def prepare_state(self, n: int) -> None:
        self.allocator.prepare_mamba_allocation(n)

    def gate(self, open_: bool) -> None:
        self.gate_open = open_

    def forward(self, ids: List[int]) -> None:
        event = FakeEvent()
        self.events.append(event)
        self.allocator.set_latest_forward_done_event(event)
        self.allocator.set_inflight_forward(
            event, torch.tensor(ids, dtype=torch.int64, device=self.device)
        )

    def fire(self) -> None:
        for event in self.events:
            event.fired = True
        self.events.clear()

    def reserve(self, member: str, n: int) -> Optional[List[int]]:
        out = self._member(member).alloc_physical(n)
        return None if out is None else out.tolist()

    def cache(self, key: int, ids: List[int]) -> None:
        start = key * 100_000
        self.tree.insert(
            InsertParams(
                key=RadixKey(array("q", range(start, start + len(ids)))),
                value=torch.tensor(ids, dtype=torch.int64, device=self.device),
            )
        )

    def evict(self, n: int) -> bool:
        return bool(self.allocator.evict_to_free_tokens(self.tree, n))

    def _member(self, name: str):
        return next(m for m in self.members() if m.sub_pool_name == name)

    # -- observables --

    def observables(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for m in self.members():
            p = m.sub_pool_name
            out[f"{p}.v2p"] = m.ids.virtual_to_physical.tolist()
            out[f"{p}.p2v"] = m.ids.physical_to_virtual.tolist()
            if m.grow_direction == "float":
                out[f"{p}.edges"] = [m.low_wm_page, m.high_wm_page]
            else:
                out[f"{p}.edges"] = [m.watermark_physical]
            out[f"{p}.allocated_pages"] = m._allocated_pages()
            out[f"{p}.holes"] = sorted(m._free_phys_pages.tolist())
            out[f"{p}.pending"] = sorted(m._pending_reuse_pages_cpu)
            out[f"{p}.free_vids"] = (
                None if m.free_virtual_ids is None else m.free_virtual_ids.tolist()
            )
            out[f"{p}.available"] = m.available_size()
            out[f"{p}.schedulable"] = m.schedulable_available_size()
        a = self.allocator
        out["available"] = a.available_size()
        if self.config.chain in WINDOW_CHAINS:
            out["full_available"] = a.full_available_size()
            out["swa_available"] = a.swa_available_size()
        if self.state_allocator is not None:
            out["state_available"] = self.state_allocator.available_size()
        out["accounting_errors"] = list(a.verify_byte_accounting())
        return out


ADAPTERS: Dict[str, Callable[[TraceConfig], object]] = {
    BuilderAdapter.name: BuilderAdapter
}


def _digest(value) -> str:
    return hashlib.sha1(json.dumps(value).encode()).hexdigest()[:12]


def _record_observables(adapter, full: bool) -> Dict[str, object]:
    obs = adapter.observables()
    return obs if full else {k: _digest(v) for k, v in obs.items()}


def _page_tokens(model: HostModel, req: int, upto: int, ps: int) -> List[int]:
    """The request's tokens whose window side is still bound, up to a page boundary."""
    r = model.requests[req]
    end = upto // ps * ps
    return r.tokens[r.retired : end] if end > r.retired else []


def _fill_state_ops(adapter, model: HostModel):
    # The id-space size bounds successful takes, even when tokens share its bytes.
    for _ in range(adapter.state_allocator.size + 1):
        allocated = len(model.states)
        yield {"op": "state_alloc", "n": 1}
        if len(model.states) == allocated:
            yield {"op": "state_alloc", "n": 1, "expect": "refused"}
            return
    raise AssertionError("state allocation did not refuse within its id-space bound")


def _scenario_ops(adapter, model: HostModel, ops):
    for op in ops:
        if op["op"] == "state_fill":
            yield from _fill_state_ops(adapter, model)
        else:
            yield op


def apply_op(adapter, model: HostModel, op: dict):
    """Apply an operation and check any scenario precondition or recovery claim."""
    outcome = _apply_op(adapter, model, op)
    expected = op.get("expect")
    if expected == "allocated":
        assert outcome is not None, f"expected allocation to succeed: {op}"
    elif expected == "refused":
        assert outcome is None, f"expected allocation to be refused: {op}"
    else:
        assert expected is None, f"unknown expected outcome: {expected!r}"
    return outcome


def _apply_op(adapter, model: HostModel, op: dict):
    kind = op["op"]
    ps = adapter.page_size
    if kind == "extend":
        req = op["req"]
        tokens = model.requests[req].tokens if req in model.requests else []
        out = adapter.extend(tokens, op["n"])
        if out is not None:
            model.requests.setdefault(req, _Request()).tokens = tokens + out
            model.next_req = max(model.next_req, req + 1)
        return out
    if kind == "decode":
        reqs = op["reqs"]
        out = adapter.decode([model.requests[r].tokens for r in reqs])
        if out is not None:
            for r, token in zip(reqs, out):
                model.requests[r].tokens = model.requests[r].tokens + [token]
        return out
    if kind == "alloc":
        out = adapter.alloc(op["n"])
        if out is not None:
            model.requests[op["req"]] = _Request(tokens=out)
            model.next_req = max(model.next_req, op["req"] + 1)
        return out
    if kind == "free":
        r = model.requests.pop(op["req"])
        adapter.free(r.tokens)
        return None
    if kind == "retire":
        ids = _page_tokens(model, op["req"], op["upto"], ps)
        if ids:
            adapter.retire(ids)
            model.requests[op["req"]].retired += len(ids)
        return None
    if kind == "segment":
        r = model.requests[op["req"]]
        start = op["start"] // ps * ps
        if start == 0:
            model.requests.pop(op["req"])
            adapter.free(r.tokens)
            return None
        adapter.free_segment(r.tokens[start:], start)
        r.tokens = r.tokens[:start]
        r.retired = min(r.retired, start)
        return None
    if kind == "group":
        adapter.group(op["begin"])
        return None
    if kind == "state_fill":
        for take in _fill_state_ops(adapter, model):
            apply_op(adapter, model, take)
        return None
    if kind == "state_alloc":
        out = adapter.state_alloc(op["n"])
        if out is not None:
            model.states.extend(out)
        return out
    if kind == "state_free":
        slot = model.states.pop(op["index"])
        adapter.state_free([slot])
        return None
    if kind == "flush":
        adapter.flush(op.get("member"))
        return None
    if kind == "ensure":
        return adapter.ensure(op["full"], op["window"])
    if kind == "prepare_state":
        adapter.prepare_state(op["n"])
        return None
    if kind == "gate":
        adapter.gate(op["open"])
        return None
    if kind == "forward":
        r = model.requests.get(op["req"])
        adapter.forward([] if r is None else r.tokens[-ps:])
        return None
    if kind == "fire":
        adapter.fire()
        return None
    if kind == "reserve":
        return adapter.reserve(op["member"], op["n"])
    if kind == "cache":
        r = model.requests.pop(op["req"])
        whole = len(r.tokens) // ps * ps
        if whole:
            adapter.cache(model.cached, r.tokens[:whole])
            model.cached += 1
        if whole < len(r.tokens):
            adapter.free(r.tokens[whole:])
        return None
    if kind == "evict":
        return adapter.evict(op["n"])
    raise ValueError(f"unknown operation {kind!r}")


def draw_op(rng: random.Random, model: HostModel, config: TraceConfig) -> dict:
    """One valid operation for the current host model (no double free)."""
    ps = config.page_size
    live = sorted(model.requests)
    roll = rng.random()
    paged = config.device != "cpu" or ps == 1
    if (roll < 0.25 or not live) and not paged:
        return {"op": "alloc", "req": model.next_req, "n": rng.randint(1, 6) * ps}
    if roll < 0.25 or not live:
        return {"op": "extend", "req": model.next_req, "n": rng.randint(1, 6) * ps}
    if roll < 0.35 and paged:
        return {"op": "extend", "req": rng.choice(live), "n": rng.randint(1, 3 * ps)}
    if roll < 0.45 and paged:
        k = rng.randint(1, min(4, len(live)))
        return {"op": "decode", "reqs": rng.sample(live, k)}
    if roll < 0.62:
        return {"op": "free", "req": rng.choice(live)}
    if roll < 0.70 and config.chain in WINDOW_CHAINS:
        req = rng.choice(live)
        return {"op": "retire", "req": req, "upto": len(model.requests[req].tokens)}
    if roll < 0.74:
        # `free_segment` requires the segment's window pages to be bound, so a
        # segment never starts inside the retired prefix.
        req = rng.choice(live)
        r = model.requests[req]
        lo = -(-r.retired // ps) * ps
        if lo <= len(r.tokens):
            return {
                "op": "segment",
                "req": req,
                "start": rng.randint(lo, len(r.tokens)),
            }
    if roll < 0.80 and config.chain in STATE_CHAINS:
        if model.states and rng.random() < 0.5:
            return {"op": "state_free", "index": rng.randrange(len(model.states))}
        return {"op": "state_alloc", "n": 1}
    unretired = [r for r in live if model.requests[r].retired == 0]
    if roll < 0.84 and config.chain == "swa" and unretired:
        return {"op": "cache", "req": rng.choice(unretired)}
    if roll < 0.86 and config.chain == "swa" and model.cached:
        return {"op": "evict", "n": rng.randint(1, 8) * ps}
    if roll < 0.90 and config.lazy:
        return {"op": "gate", "open": rng.random() < 0.7}
    if roll < 0.94 and config.lazy:
        if rng.random() < 0.5:
            return {"op": "forward", "req": rng.choice(live)}
        return {"op": "fire"}
    if roll < 0.97 and config.chain in WINDOW_CHAINS:
        n = rng.randint(1, 8) * ps
        return {"op": "ensure", "full": n, "window": n}
    if roll < 0.97 and config.chain in STATE_CHAINS:
        return {"op": "prepare_state", "n": rng.randint(1, 4)}
    return {"op": "flush"}


def record_run(
    config: TraceConfig,
    *,
    seed: Optional[int] = None,
    ops: Optional[List[dict]] = None,
    steps: int = 200,
    adapter_name: str = BuilderAdapter.name,
    full: bool = False,
    coverage: Optional[collections.Counter] = None,
) -> List[dict]:
    """Records of one run: a seeded stream drawn against a live adapter, or
    the given operations (a scenario). ``coverage`` counts the states reached."""
    adapter = ADAPTERS[adapter_name](config)
    model = HostModel()
    rng = random.Random(seed)
    records = []
    stream = _scenario_ops(adapter, model, ops) if ops is not None else [None] * steps
    for step, given in enumerate(stream):
        op = given if given is not None else draw_op(rng, model, config)
        outcome = apply_op(adapter, model, op)
        obs = adapter.observables()
        if coverage is not None:
            _count_coverage(coverage, config, op, outcome, obs)
        records.append(
            {
                "step": step,
                "op": op,
                "out": outcome,
                "obs": obs if full else {k: _digest(v) for k, v in obs.items()},
            }
        )
    return records


def _count_coverage(coverage, config, op, outcome, obs) -> None:
    chain = config.chain
    if outcome is None and op["op"] in ("extend", "decode", "alloc", "state_alloc"):
        coverage[f"{chain}: refused {op['op']}"] += 1
    for key, value in obs.items():
        name = key.rsplit(".", 1)[-1]
        if name in ("holes", "pending") and value:
            coverage[f"{chain}: {key} non-empty"] += 1
        if name == "edges" and len(value) == 2 and value[0] == value[1]:
            coverage[f"{chain}: {key.split('.')[0]} float empty"] += 1
    if op["op"] == "gate" and not op["open"]:
        coverage[f"{chain}: gate closed"] += 1


class Divergence(msgspec.Struct, frozen=True, kw_only=True):
    step: int
    op: dict
    keys: Tuple[str, ...]
    prefix: Tuple[dict, ...]


def replay_run(
    config: TraceConfig,
    records: List[dict],
    *,
    adapter_name: str = BuilderAdapter.name,
    full: bool = False,
) -> Optional[Divergence]:
    """Replay recorded operations on an adapter; the first step whose outcome
    or observables differ, else None. ``full`` says how the records stored
    their observables."""
    adapter = ADAPTERS[adapter_name](config)
    model = HostModel()
    for i, rec in enumerate(records):
        # Recorded outcomes are the oracle; a failed expectation is a divergence.
        outcome = _apply_op(adapter, model, rec["op"])
        obs = _record_observables(adapter, full)
        keys = tuple(
            k
            for k in sorted(set(obs) | set(rec["obs"]))
            if obs.get(k) != rec["obs"].get(k)
        )
        if outcome != rec["out"]:
            keys = ("out",) + keys
        if keys:
            return Divergence(
                step=i,
                op=rec["op"],
                keys=keys,
                prefix=tuple(r["op"] for r in records[max(0, i - 8) : i]),
            )
    return None


# -- corner states for property tests of capacity decisions --


def corner_states(config: TraceConfig) -> Dict[str, List[dict]]:
    """Named operation prefixes that leave the allocator in a corner state;
    replay one with `apply_op` on a fresh adapter to rebuild the state."""
    ps = config.page_size
    fill = [{"op": "extend", "req": i, "n": 4 * ps} for i in range(12)]
    holes = fill + [{"op": "free", "req": i} for i in range(0, 12, 2)]
    corners = {"holes": holes}
    if config.lazy:
        corners["closed_gate_over_holes"] = holes + [{"op": "gate", "open": False}]
        corners["pending_reuse_behind_forward"] = (
            fill
            + [{"op": "forward", "req": 1}]
            + [{"op": "free", "req": i} for i in range(0, 12, 2)]
            + [{"op": "flush"}]
        )
        corners["hicache_reservation"] = holes + [
            {"op": "reserve", "member": "full", "n": 2 * ps}
        ]
    if config.chain == "tri":
        corners["float_interior_holes"] = holes + [
            {"op": "retire", "req": i, "upto": 4 * ps} for i in range(1, 12, 4)
        ]
    if config.chain in STATE_CHAINS:
        # State slots share the buffer, so their count is a byte question: ask
        # until refused.
        corners["state_full"] = [{"op": "state_fill"}]
    corners["allocator_full"] = [
        {"op": "extend", "req": i, "n": 8 * ps} for i in range(64)
    ]
    return corners
