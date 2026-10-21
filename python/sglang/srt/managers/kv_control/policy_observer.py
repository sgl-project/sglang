# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned bounded observations; no policy receives these live objects."""

from itertools import chain

from sglang.srt.managers.kv_control.policy import (
    PolicyObservation,
    PoolObservation,
    RequestObservation,
)
from sglang.srt.managers.schedule_policy import (
    CLIP_MAX_NEW_TOKENS,
    running_request_token_offset,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

FULL = ComponentType.FULL


def _cache_commitment(runtime, req):
    """Read the native scheduler's fresh match, without a second token walk.

    FULL native matching runs just before this observation. Count unreferenced
    nodes that admission would protect; they currently appear as reclaimable
    budget, so treating only new allocation as demand would spend them twice.
    """
    executor = runtime.control.executor
    core = executor.core
    if (
        getattr(req, "lora_id", None) is not None
        or getattr(req, "multimodal_inputs", None) is not None
    ):
        return None, None
    cached = req.prefix_len
    if cached == 0:
        return 0, {}
    selected, length = {}, 0
    try:
        node = core.node_by_id(req.last_node)
        visited = 0
        while node is not core.root_node:
            visited += 1
            if visited > executor.config.max_query_nodes:
                return None, None
            cd = node.component_data[FULL]
            if (
                cd.value is None
                or cd.value.numel() != len(node.key)
                or (
                    executor.transfers is not None
                    and executor.transfers._load_pending(node)
                )
            ):
                return None, None
            length += len(node.key)
            if cd.lock_ref == 0:
                selected[node.id] = len(node.key)
            node = node.parent
        return (cached, selected) if length == cached else (None, None)
    except (KeyError, TypeError, AttributeError):
        return None, None


def observe_scheduler(runtime, scheduler, running_batch):
    cache, control = runtime.service.cache, runtime.control
    allocator = cache.token_to_kv_pool_allocator
    free = int(allocator.available_size())
    evictable = int(cache.tree_core.full_evictable_size())
    total = int(allocator.size)
    pool = PoolObservation(
        "full", "gpu", total, free, total - free - evictable, evictable
    )
    candidates = chain(
        ((r, False) for r in running_batch.reqs),
        ((scheduler.chunked_req, False),) if scheduler.chunked_req is not None else (),
        ((r, True) for r in scheduler.waiting_queue),
    )
    now = control._clock()
    records, seen, union = [], set(), {}
    complete, demand_complete = True, True
    new_rows = 0
    for scanned, (req, waiting) in enumerate(candidates):
        if scanned >= runtime.service.config.max_policy_requests:
            complete = False
            break
        if id(req) in seen or req.finished():
            continue
        seen.add(id(req))
        runtime.register_request(req)
        tasks = tuple(
            t
            for t in getattr(req, "kv_control_policy_tasks", ())
            if control.index.task_identity(t.session_id) == t
        )
        task = tasks[0] if len(tasks) == 1 else None
        cached, commitment = (
            _cache_commitment(runtime, req) if waiting else (None, None)
        )
        incremental = None
        growth = None
        if waiting:
            if cached is not None:
                page = cache.page_size
                input_tokens = max(
                    0, len(req.origin_input_ids) + len(req.output_ids) - cached
                )
                remaining = max(
                    0, req.sampling_params.max_new_tokens - len(req.output_ids)
                )
                allocation = (
                    (input_tokens + page - 1) // page * page
                    + min(remaining, CLIP_MAX_NEW_TOKENS)
                    + page
                )
                incremental = allocation + sum(commitment.values())
                union.update(commitment)
                new_rows += allocation
            else:
                demand_complete = False
        else:
            growth = max(
                0,
                int(
                    running_request_token_offset(
                        req, scheduler.new_token_ratio_tracker.current
                    )
                ),
            )
        records.append(
            RequestObservation(
                req.kv_control_policy_identity,
                task,
                waiting,
                max(0.0, now - req.kv_control_policy_arrival),
                req.kv_control_policy_order,
                int(req.kv.kv_allocated_len),
                cached,
                incremental,
                growth,
            )
        )
    return PolicyObservation(
        now,
        control.registry.namespace,
        control.registry.engine_epoch,
        (pool,),
        tuple(records),
        complete,
        estimated_waiting_tokens=new_rows + sum(union.values())
        if complete and demand_complete
        else None,
    )
