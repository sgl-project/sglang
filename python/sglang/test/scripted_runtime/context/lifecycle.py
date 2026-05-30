"""Free functions for request- and engine-lifecycle control verbs.

These drive pause / continue / abort / retract / preempt and engine-wide
actions (flush cache, shutdown). Each takes the facade ``ctx`` first and
reaches the live scheduler through ``ctx._scheduler``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from sglang.srt.managers.io_struct import (
    AbortReq,
    ContinueGenerationReqInput,
    FlushCacheReqInput,
    PauseGenerationReqInput,
)

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext
    from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle


def pause_generation(
    ctx: "ScriptedContext", *, mode: Literal["retract", "in_place"]
) -> None:
    """Scripted-only entry point: bypasses HTTP server / tokenizer manager.

    Drives ``scheduler.pause_generation`` directly with the given mode.
    ``mode="abort"`` is intentionally not supported here: the scheduler
    has no abort branch inside ``pause_generation``; use
    :meth:`abort_all` instead.
    """
    assert ctx._is_driver, "pause_generation is only callable from the driver rank"
    ctx._scheduler.pause_generation(PauseGenerationReqInput(mode=mode))


def continue_generation(ctx: "ScriptedContext", *, torch_empty_cache: bool) -> None:
    """Scripted-only entry point: bypasses HTTP server / tokenizer manager.

    Resume after :meth:`pause_generation`. Defaults to skipping the
    empty_cache call to keep scripted runs deterministic.
    """
    assert ctx._is_driver, "continue_generation is only callable from the driver rank"
    ctx._scheduler.continue_generation(
        ContinueGenerationReqInput(torch_empty_cache=torch_empty_cache)
    )


def abort_all(ctx: "ScriptedContext") -> None:
    """Scripted-only entry point: bypasses HTTP server / tokenizer manager.

    Drives ``scheduler.abort_request(AbortReq(abort_all=True))`` —
    the real abort code path. ``pause_generation(mode="abort")`` does
    not abort in the scheduler today; this is the only way to reach
    the abort branch from a script.
    """
    assert ctx._is_driver, "abort_all is only callable from the driver rank"
    ctx._scheduler.abort_request(AbortReq(abort_all=True))


def flush_cache(ctx: "ScriptedContext") -> None:
    """Inject an engine-wide cache flush (radix tree + memory pools).

    Reaches every rank through the normal request-broadcast path, so it is
    safe under TP/PP. Like ``start_req``, the flush is visible on the next
    ``yield``, and the scheduler only honors it when the engine is idle
    (no in-flight reqs). ``run`` issues one before each script so a sub-
    script starts from a clean cache.
    """
    assert ctx._is_driver, "flush_cache is only callable from the driver rank"
    ctx._tokenizer_recv_proxy.inject(FlushCacheReqInput())


def trigger_abort_on_waiting_timeout(ctx: "ScriptedContext") -> None:
    """Simulate the watchdog firing on a stuck waiting-queue entry.

    Drives the watchdog-fire branch without waiting for the real
    timer; deterministic shortcut for the abort-on-timeout path.

    Consumed by: test_abort_on_waiting_timeout (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: trigger_abort_on_waiting_timeout is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def shutdown(ctx: "ScriptedContext") -> None:
    """Send an engine shutdown signal from inside the script.

    Lets a lifecycle test verify clean shutdown from the scripted
    side without relying on the outer ``launch_scripted_http_server``
    teardown.

    Consumed by: test_engine_shutdown_from_script (lifecycle).
    """
    raise NotImplementedError(
        "scripted_runtime: shutdown is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def abort(ctx: "ScriptedContext", r: "ScriptedReqHandle") -> None:
    """Abort a single in-flight request immediately.

    Drives the engine's abort code path on a deterministic target
    without needing a client-side cancel. The request transitions to
    ``aborted=True``; downstream finalize / KV release must occur
    exactly once.

    Consumed by: test_abort_during_chunked_prefill (abort),
                 test_force_retract_then_abort_same_yield (abort),
                 test_abort_running_decode (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: abort is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def force_retract(ctx: "ScriptedContext", r: "ScriptedReqHandle") -> None:
    """Force a single req to retract immediately, releasing its KV and rolling chunks_done back to 0.

    Independent of KV pressure — this is a deterministic test hook used to drive the retract code
    path without having to engineer a memory exhaustion scenario.

    Consumed by: test_chunked_oscillation_three_force_retracts (kv_pressure),
                 test_force_retract_then_abort_same_yield (abort),
                 test_chunked_retract_at_chunk_first_mid_last (kv_pressure).
    """
    raise NotImplementedError(
        "scripted_runtime: force_retract is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def retract_all(ctx: "ScriptedContext") -> None:
    """Retract every currently-running request in one call.

    Bulk version of :meth:`force_retract`. Triggers the engine's
    engine-wide retract path; useful for stress / regression tests
    that need a clean slate without shutting down the engine.

    Consumed by: test_retract_all_then_resume (regression),
                 test_retract_all_during_chunked_prefill (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: retract_all is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def pause_retract_all(ctx: "ScriptedContext") -> None:
    """Invoke ``pause_generation(retract)`` engine-wide.

    Distinct from :meth:`retract_all`: this exercises the pause-style
    retract entry point used by the engine to handle external pause
    signals, not the per-req force_retract path.

    Consumed by: test_pause_retract_all_then_resume (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: pause_retract_all is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def force_preempt(
    ctx: "ScriptedContext", *, req: "ScriptedReqHandle", by: "ScriptedReqHandle"
) -> None:
    """Manually trigger priority preemption of ``req`` by ``by``.

    Bypasses the priority comparator so tests can drive the preempt
    code path without engineering specific priority value combos.

    Consumed by: test_priority_preempt_during_chunked_prefill (priority),
                 test_priority_preempt_releases_kv (priority).
    """
    raise NotImplementedError(
        "scripted_runtime: force_preempt is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def force_lora_drainer_reject(ctx: "ScriptedContext", *, adapter: str) -> None:
    """Make the LoRA drainer reject ``adapter`` on its next admit attempt.

    Drives the drainer-reject branch without needing to engineer a
    realistic LoRA cache eviction race.

    Consumed by: test_lora_drainer_reject_then_retry (lora).
    """
    raise NotImplementedError(
        "scripted_runtime: force_lora_drainer_reject is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )
