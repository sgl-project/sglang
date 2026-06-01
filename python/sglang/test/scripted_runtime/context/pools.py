from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def exhaust_kv(ctx: "ScriptedContext", *, leave_pages: int) -> None:
    s = ctx._scheduler
    allocator = s.token_to_kv_pool_allocator

    leave_tokens = leave_pages * s.page_size
    need = allocator.available_size() - leave_tokens
    if need <= 0:
        return

    held = allocator.alloc(need)
    assert (
        held is not None
    ), f"exhaust_kv: allocator could not grab {need} tokens to create pressure"
    ctx._held_kv_allocations.append(held)


def exhaust_row_pool(ctx: "ScriptedContext", *, leave_rows: int) -> None:
    pool = ctx._scheduler.req_to_token_pool

    free_slots = pool.free_slots
    assert isinstance(
        free_slots, list
    ), f"exhaust_row_pool expects list-based free_slots, got {type(free_slots)!r}"

    take = len(free_slots) - leave_rows
    if take <= 0:
        return

    ctx._held_row_slots.extend(free_slots[:take])
    pool.free_slots = free_slots[take:]


def release_exhausted(ctx: "ScriptedContext") -> None:
    # Idempotent cleanup run before each script resets the engine: hand every
    # pool slot we held to create pressure back to the real allocators so the
    # next script starts from a full pool.
    for held in ctx._held_kv_allocations:
        ctx._scheduler.token_to_kv_pool_allocator.free(held)
    ctx._held_kv_allocations.clear()

    if ctx._held_row_slots:
        pool = ctx._scheduler.req_to_token_pool
        pool.free_slots = ctx._held_row_slots + pool.free_slots
        ctx._held_row_slots.clear()
