from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import os
import queue
import threading
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

_sync_groups: List[dist.ProcessGroup] = []  # Dedicated gloo groups (one per rank-set).
_q: Optional[queue.Queue[str]] = None
_worker_thread: Optional[threading.Thread] = None
_scheduler_thread: Optional[threading.Thread] = None


def rank_consensus(func=None, *, same_params=None, same_results=None, **kwargs):
    """
    Mark a function that should be consensus in PP and TP ranks.  Here consensus means,
    the same order of calling, same parameters and return values optionally.

    The function must be called in the scheduler thread.

    Usages:

    * Assert that the function is called by all ranks.  The parameters or results may not be same.
    @rank_consensus
    def foo():
        pass

    * Assert that all parameters are same in all ranks.
    @rank_consensus(same_params = True)
    def foo(a, b):
        pass

    * Assert that some parameters are same in all ranks.
    @rank_consensus(same_params = ["a", "c"])
    def foo(a, b, c):
        pass

    * Assert that part of the parameters are same in all ranks.
    @rank_consensus(same_params = ["a.req_id"])
    def foo(a):
        pass

    * Assert that results are same in all ranks.
    @rank_consensus(same_results = True)
    def foo():
        return 1

    * Assert for part of the results are same.
    @rank_consensus(same_results = ["result.some_field"]
    def foo():
        return SomeObject()

    @rank_consensus(same_results = ["result.field", "len(result.field2)"]
    def foo():
        return SomeObject()

    * Assert the function is called by all ranks and all parameters and results are the same.
    @rank_consensus(same_params = True, same_results = True)
    def foo():
        return 1
    """
    if kwargs:
        raise TypeError(
            f"rank_consensus() got unexpected keyword argument(s): " f"{list(kwargs)}"
        )

    params_selector = _normalize_selector(same_params, "same_params")
    results_selector = _normalize_selector(same_results, "same_results")

    def decorator(func: Callable) -> Callable:
        # This decorator function called at import time.  So it should be zero runtime overhead
        # when the consensus checker is disabled.
        if not envs.SGLANG_ENABLE_RANK_CONSENSUS_CHECKER.get():
            return func

        # Unwrap static/class-method descriptors so we always operate on the
        # raw function. We remember the descriptor type so we can re-wrap the
        # result and the class-body descriptor protocol keeps working.
        if isinstance(func, (classmethod, staticmethod)):
            raw_func = func.__func__
            descriptor_type = type(func)
        else:
            raw_func = func
            descriptor_type = None
        sig = inspect.signature(raw_func)

        # When calling class method or object method with "same_params=True",
        # skip the first "cls" or "self", as the text format for that
        # may include memory addresses, which are considered divergence.
        skip_name: Optional[str] = None
        if _is_method_with_receiver(func) and len(sig.parameters) > 0:
            skip_name = next(iter(sig.parameters))

        @functools.wraps(raw_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            params_payload = "<no check>"
            if params_selector is not None:
                # Bind once and apply defaults so that name-based selectors work
                # regardless of whether the caller passed positionally or by kw.
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                arguments = dict(bound.arguments)
                params_payload = _build_payload(
                    "call", params_selector, arguments, skip_name
                )
            assert_same("%s called params=%s", raw_func.__name__, params_payload)

            result = raw_func(*args, **kwargs)

            result_payload = "<no check>"
            if results_selector is not None:
                result_scope = {"result": result}
                result_payload = _build_payload(
                    "return",
                    results_selector,
                    result_scope,
                )
            assert_same("%s returns result=%s", raw_func.__name__, result_payload)
            return result

        # Re-wrap into the original descriptor type so class-body access
        # (C.method / instance.method) still binds correctly.
        if descriptor_type is staticmethod:
            return staticmethod(wrapper)
        if descriptor_type is classmethod:
            return classmethod(wrapper)
        return wrapper

    if func is not None:
        # Bare `@rank_consensus` form.
        return decorator(func)
    else:
        # `@rank_consensus(same_params=True, same_results=True)` form.
        return decorator


def _normalize_selector(
    value: None | bool | str | list[str], name: str
) -> None | bool | list[str]:
    """Normalize a selector argument to one of:
    ``None`` (skip), ``True`` (compare everything), or ``list[str]`` (the
    expressions to evaluate). ``False`` is treated as ``None``.
    """
    if value is None or value is False:
        return None
    if value is True:
        return True
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(s, str) for s in value):
        return list(value)
    raise TypeError(f"{name} must be True / False / str / list[str], got {value!r}")


def _is_method_with_receiver(func: Any) -> bool:
    """Return True iff ``func`` is a method whose first parameter is a
    receiver (instance for instance-methods, class for class-methods) that
    should be dropped from the ``same_params=True`` payload.

    Distinguishes:
      * ``staticmethod`` object  -> False (no receiver)
      * ``classmethod``  object  -> True  (receiver is the class)
      * plain ``def`` defined inside a class body (``__qualname__`` has a
        dot before the final segment and is not a ``<locals>`` closure) ->
        True  (instance method)
      * anything else (module-level function, nested function, lambda) ->
        False
    """
    if isinstance(func, staticmethod):
        return False
    if isinstance(func, classmethod):
        return True
    if inspect.isfunction(func):
        qualname = getattr(func, "__qualname__", "")
        # ``C.m`` -> True; ``m`` -> False; ``outer.<locals>.m`` -> False
        # (closures aren't class-body methods).
        if "." in qualname and "<locals>" not in qualname:
            return True
    return False


def _build_payload(
    tag: str,
    selector: bool | list[str],
    scope: dict[str, Any],
    skip_name: Optional[str] = None,
) -> str:
    """Serialize the selected values into a single comparable string.

    ``skip_name`` only applies to the ``True`` (whole-scope) form and is used
    to drop the receiver (``self`` / ``cls``) from method payloads; explicit
    ``list[str]`` selectors honor exactly what the user listed.
    """
    if selector is True:
        # Whole scope is the payload. For the call checkpoint, the scope is
        # the arguments dict; for the return checkpoint, the caller wrapped
        # result into the scope, so we repr ``result`` directly.
        if tag == "call":
            if skip_name is not None:
                scope = {k: v for k, v in scope.items() if k != skip_name}
            return repr(scope)
        return repr(scope["result"])
    parts: list[str] = []
    for expr in selector:
        value = _eval_selector(expr, scope)
        parts.append(f"{expr}={value!r}")
    return " | ".join(parts)


def _eval_selector(expr: str, scope: dict[str, Any]) -> Any:
    """Evaluate a selector expression in a restricted scope.

    Errors (unknown parameter name, missing attribute, bad syntax) propagate
    -- they are caller bugs and must not be silently swallowed or confused
    with cross-rank divergence.
    """
    safe_builtins = {
        "len": len,
        "int": int,
        "str": str,
        "bool": bool,
        "float": float,
        "tuple": tuple,
        "list": list,
        "dict": dict,
        "set": set,
        "sorted": sorted,
        "min": min,
        "max": max,
        "sum": sum,
    }
    return eval(expr, {"__builtins__": safe_builtins}, dict(scope))


def enabled() -> bool:
    """Test that the checker has been enabled and configure() is called."""
    return _q is not None


def assert_same(msg_fmt: str, *args: Any) -> None:
    """Record a decision that every TP/PP rank must make identically.

    Must be called from the scheduler thread. If the env var is set and the
    checker is configured, an assertion guards that the caller is on the
    scheduler thread recorded at configure() time — events from other threads
    would interleave out of order with peer ranks and corrupt the lock-step
    drain.

    When the divergence checker is disabled, this is a zero-overhead no-op.

    Example:
    assert_same("my decision: %s %d", "foo", 100)

    Prefer `@rank_consensus` over this function for code-cleanliness.
    """
    if not enabled():
        return
    # Sanity check: only the scheduler thread is allowed to enqueue. Other
    # callers would race with the worker's min-length drain and desynchronize
    # ranks, since their events would not exist on peer ranks.
    if threading.current_thread() is not _scheduler_thread:
        raise RuntimeError("rdc.assert_same must be called from the scheduler thread")
    # Format eagerly: args may reference mutable state that mutates
    # between now and when the worker thread drains the queue.
    _q.put(msg_fmt % args)


def configure(groups: List[GroupCoordinator]) -> None:
    """Initialize the checker. No-op if SGLANG_ENABLE_RANK_CONSENSUS_CHECKER is not set."""
    global _sync_groups, _q, _worker_thread, _scheduler_thread

    if not envs.SGLANG_ENABLE_RANK_CONSENSUS_CHECKER.get():
        return

    logger.warning(
        "Rank consensus checker is enabled. The server will suicide if rank divergence detected."
    )
    # Build a dedicated sync group.  So our synchronization work will not affect
    # the scheduler thread at all.
    _sync_groups = _create_sync_groups(groups)
    _q = queue.Queue()
    # Assume the calling thread is the schedule thread.
    # We will check assert_same() must be called by the scheduler thread.
    _scheduler_thread = threading.current_thread()
    _worker_thread = threading.Thread(
        target=_worker_loop, name="rank_consensus_checker", daemon=True
    )
    _worker_thread.start()


def _create_sync_groups(
    groups: List[GroupCoordinator],
) -> List[dist.ProcessGroup]:
    """Create duplicated groups, used for background thread"""
    from sglang.srt.distributed.parallel_state import create_custom_parallel_group

    dedicated: List[dist.ProcessGroup] = []
    seen_rank_sets: set[tuple[int, ...]] = set()
    for group in groups:
        if group is None:
            continue
        # Skip single-rank groups: nothing to compare against.
        if torch.distributed.get_world_size(group=group.cpu_group) == 1:
            continue
        group_ranks = tuple(torch.distributed.get_process_group_ranks(group.cpu_group))
        if group_ranks in seen_rank_sets:
            continue
        seen_rank_sets.add(group_ranks)
        pg = create_custom_parallel_group(group_ranks=list(group_ranks), backend="gloo")
        if pg is not None:
            dedicated.append(pg)
    return dedicated


def _destroy_dedicated_groups() -> None:
    for pg in _sync_groups:
        try:
            torch.distributed.destroy_process_group(pg)
        except Exception:
            pass


def shutdown() -> None:
    """Flush the queue, stop the worker thread, and disable assert_same."""
    global _q, _worker_thread, _sync_groups, _scheduler_thread

    q = _q
    if q is None:
        return

    # Put a sentinel value to wake the worker if it is blocked on _q.get().
    q.put(None)
    if _worker_thread is not None:
        _worker_thread.join()
        _worker_thread = None
    # Tear down the dedicated gloo groups BEFORE clearing _groups so the
    # destroy helper can see them. Worker thread is already joined, so there
    # is no concurrent all_reduce on these groups.
    _destroy_dedicated_groups()
    _q = None
    _sync_groups = []
    _scheduler_thread = None


def _worker_loop() -> None:
    """Consume events in lock-step with peer ranks via gloo all-reduce.

    Each iteration:
      1. Determine the items available in _q.
      2. Drain exactly the minimum number of items in all ranks.
      3. Compare all events are identical across ranks.
    """
    while _q is not None:
        # Drain first.  Block waiting for the first event.
        first = _q.get()
        if first is None:
            # shutdown() is called.
            return

        # Drain more whenever available.
        # Every rank should drain the same number.
        count = _all_reduce_min_int(_q.qsize())
        events: List[str] = [first]
        shutdown_signaled = False
        for _ in range(count):
            event = _q.get()
            if event is None:
                # shutdown() sentinel arrived mid-batch: stop draining but
                # still check the events we already hold — they are real
                # decisions every rank must agree on. Then exit, since the
                # sentinel means shutdown() is waiting on worker_thread.join().
                shutdown_signaled = True
                break
            events.append(event)

        # Cross-rank check.
        _check_for_consensus(events)

        if shutdown_signaled:
            return


def _all_reduce_min_int(value: int) -> int:
    """Reduce `value` to its global minimum across every configured group."""
    tensor = torch.tensor([value], dtype=torch.int64)
    for group in _sync_groups:
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
    return int(tensor.item())


def _check_for_consensus(events: list[str]) -> None:
    # Compute sha1 of concatenation of all msgs.
    hasher = hashlib.sha1()
    for msg in events:
        hasher.update(msg.encode("utf-8"))

    # Determine if some rank has a different value.
    value_bytes = hasher.digest()
    min_value = torch.tensor(list(hasher.digest()), dtype=torch.uint8)
    max_value = min_value.clone()
    for group in _sync_groups:
        dist.all_reduce(min_value, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(max_value, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(min_value, max_value):
        # When divergence, all rank should output the following log.
        logger.critical(
            f"Found rank divergence for {len(events)} events(s)! local hash: {value_bytes.hex()}, events = {events}"
        )
        for handler in logger.handlers:
            handler.flush()

        # os._exit instead of sys.exit: this runs in a background thread, where
        # SystemExit would only kill the thread, not the process. os._exit tears
        # down the whole scheduler process so a TP/PP mismatch can never
        # silently keep serving.
        os._exit(1)

    logger.debug(f"Consensus check passed for {len(events)} event(s).")
