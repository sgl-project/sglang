"""Cache-only fixtures using the real admission runtime and prefix lease.

Pools/tree/requests are CPU doubles; this does not model DMA or real tree
split/host-lock semantics. No buffer pipeline or private staged holds exist.
"""

import __future__

import ast
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import SimpleNamespace as NS

from sglang.srt import environ as environ
from sglang.srt.managers import pp_prefetch_scheduler as m
from sglang.srt.managers.scheduler_pp_mixin import PrefillControl as PrefillControl
from sglang.srt.mem_cache import pp_cache_prefix as lease  # noqa: F401
from sglang.srt.mem_cache import pp_prefetch_config as config  # noqa: F401
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats

ROOT = Path(__file__).resolve().parents[1] / "srt"


@cache
def source_nodes(path):
    # Source is fixed within a test process; only execution namespaces vary.
    return ast.parse((ROOT / path).read_text()).body


def functions(path, names, namespace, *, cls=None):
    """Compile production definitions while substituting runtime dependencies."""
    nodes = source_nodes(path)
    if cls is not None:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    selected = [
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    if {n.name for n in selected} != set(names):
        raise LookupError(
            f"Missing functions in {path}: {set(names) - {n.name for n in selected}}"
        )
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(ROOT / path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


def method(path, cls, name, extra=None):
    return functions(path, (name,), dict(extra or {}), cls=cls)[name]


def bind_prefill_release(scheduler, *, release_kv_cache, release_metadata):
    release = method(
        "disaggregation/prefill.py",
        "SchedulerDisaggregationPrefillMixin",
        "release_aborted_prefill_waiting_req",
        {
            "release_kv_cache": release_kv_cache,
            "maybe_release_metadata_buffer": release_metadata,
        },
    )
    scheduler.release_aborted_prefill_waiting_req = lambda req: release(scheduler, req)


def prefetch_wait_type(clock):
    """Use the production wait timer with a deterministic clock boundary."""
    import msgspec

    scope = dict(msgspec=msgspec, time=NS(perf_counter=clock))
    functions("utils/poll_timer.py", {"PollTimer"}, scope)
    functions("mem_cache/unified_radix_cache.py", {"_StoragePrefetchWait"}, scope)
    return scope["_StoragePrefetchWait"]


class Allocator:
    def __init__(self, size=65536):
        self.size = size
        self.free_indices = set(range(size))

    def available_size(self):
        return len(self.free_indices)

    def alloc(self, n):
        if n > len(self.free_indices):
            return None
        indices = sorted(self.free_indices)[:n]
        self.free_indices.difference_update(indices)
        return indices

    def free(self, indices):
        assert not self.free_indices.intersection(indices), "double free"
        self.free_indices.update(indices)

    def free_segment(self, indices, *, start_pos):
        assert start_pos == 0
        self.free(indices)


class RowPool(Allocator):
    alloc_rows = Allocator.alloc
    free_rows = Allocator.free


class UnifiedRadixCache:
    host_memory_mode = "cache"
    page_size = 64
    prefetch_stop_policy = "best_effort"

    def __init__(self, caps):
        self.caps = dict(caps)
        self.waiting = set()
        self._pp_cache_storage_spans = {}
        self.ongoing_prefetch = {}
        self.finalized = []
        self.refs = {"device": {}, "host": {}}

    def supports_swa(self):
        return False

    def supports_mamba(self):
        return False

    def _all_reduce_attn_groups(self, tensor, op):
        pass  # Single-process TP stand-in; real TP reduction is tested with Gloo.

    def check_hicache_events(self):
        pass

    def full_evictable_size(self):
        # This double has host descriptors, not allocated device tree pages.
        return 0

    def check_prefetch_progress(self, handle):
        return handle.rid not in self.waiting

    def prefetch_ready_to_finalize(self, handles):
        return [handle.rid not in self.waiting for handle in handles]

    def finalize_prefetch(self, handle):
        assert handle.rid not in self.waiting, "finalized before all-stage readiness"
        self.finalized.append(handle.rid)
        return True

    def get_admission_storage_span(self, handle):
        return self._pp_cache_storage_spans.get(handle.rid)

    def release_admission_storage_span(self, handle):
        self._pp_cache_storage_spans.pop(handle.rid, None)

    def finish_storage_prefetch_admission(self, handle, fulfilled_tokens, reason):
        pass  # Native metric lifecycle is exercised by separate boundary tests.

    def _inc(self, kind, node):
        refs = self.refs[kind]
        refs[node] = refs.get(node, 0) + 1
        return NS(to_dec_params=lambda: (kind, node))

    def _dec(self, kind, node, params):
        assert params == (kind, node)
        assert self.refs[kind][node] > 0
        self.refs[kind][node] -= 1

    def inc_lock_ref(self, node):
        return self._inc("device", node)

    def dec_lock_ref(self, node, params):
        self._dec("device", node, params)

    def lock(self, node):
        return NS(node=node, receipt=self.inc_lock_ref(node).to_dec_params())

    def unlock(self, lock):
        if lock is not None:
            self.dec_lock_ref(lock.node, lock.receipt)

    def inc_host_lock_ref(self, node):
        return self._inc("host", node)

    def dec_host_lock_ref(self, node, params):
        self._dec("host", node, params)


@dataclass(frozen=True)
class CacheRequestHandle:
    rid: str
    attempt_id: int = 0


class RequestDouble(NS):
    # Real Req uses object identity/hash and supports weak references.
    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__


def request(rid):
    req = RequestDouble(
        rid=rid,
        cache_request_handle=CacheRequestHandle(rid),
        origin_input_ids=list(range(1025)),
        extra_key=None,
        cache_salt=None,
        return_logprob=False,
        logprob_start_len=-1,
        sampling_params=NS(max_new_tokens=32),
        time_stats=SchedulerReqTimeStats(),
        grammar=None,
        beam_group=None,
        session=None,
        session_id=None,
        pending_bootstrap=False,
        release_mm_inputs_on_abort=lambda: None,
        metadata_buffer_index=-1,
        prefix_len=0,
        extend_end=1025,
        kv=NS(holds_kv=False, holds_mamba=False),
    )

    def match(cache, *, cow_mamba=False, max_cache_prefix_tokens):
        cap = min(cache.caps[rid], max_cache_prefix_tokens)
        req.prefix_len = 0
        req.host_hit_length = cap
        req.last_node = (rid, "device", 0)
        req.best_match_node = (rid, "host", cap)

    req.init_next_round_input = match
    return req


def stage(rank, size, caps, *, prefetch_policy="best_effort"):
    s = NS(
        ps=NS(pp_rank=rank, pp_size=size),
        tree_cache=UnifiedRadixCache(caps),
        waiting_queue=[request(rid) for rid in caps],
        disagg_prefill_bootstrap_queue=NS(queue=[]),
        page_size=64,
        chunked_req=None,
        token_to_kv_pool_allocator=Allocator(),
        req_to_token_pool=RowPool(64),
    )
    s.tree_cache.prefetch_stop_policy = prefetch_policy
    runtime = m.PPAdmissionGate(s, parallel=s.ps)
    s.pp_prefetch_admission = runtime
    return runtime


def select(runtime, selection, *, limit=100):
    reqs = selection.candidates[:limit]
    for req in reqs:
        cap = selection.prepare_request(req, runtime.s.tree_cache)
        req.prefix_len, req.extend_end = cap, len(req.origin_input_ids)
        if getattr(runtime.s, "allocate_native_batch", False):
            page = runtime.s.page_size
            tokens = runtime.s.token_to_kv_pool_allocator.alloc(
                -(-req.extend_end // page) * page
            )
            rows = runtime.s.req_to_token_pool.alloc_rows(1)
            assert tokens is not None and rows is not None, (
                "certified native allocation failed"
            )
            req.test_allocation = (tokens, rows)
    chosen = {id(req) for req in reqs}
    runtime.s.waiting_queue[:] = [
        req for req in runtime.s.waiting_queue if id(req) not in chosen
    ]
    return NS(reqs=reqs) if reqs else None


def tick(stages, *, limit=100):
    envelope, batches = None, []
    for runtime in stages:
        if envelope is not None:
            runtime.receive_forward(envelope)
        selection = runtime.prepare_selection()
        batch = select(runtime, selection, limit=limit)
        batches.append(m.batch_layout(batch))
        envelope = runtime.finish_selection(batch)
    return envelope, batches


def settle(stages, envelope):
    for runtime in stages:
        runtime.consume_return(envelope)


def cycle(stages, *, limit=100):
    envelope, batches = tick(stages, limit=limit)
    settle(stages, envelope)
    return envelope, batches
