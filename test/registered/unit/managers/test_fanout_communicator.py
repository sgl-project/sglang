import asyncio

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


async def _pump_until(predicate, max_ticks=200):
    for _ in range(max_ticks):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was never reached")


async def _pump_until_lock_waiters(comm, count, max_ticks=200):
    await _pump_until(
        lambda: len(getattr(comm._lock, "_waiters", None) or ()) >= count,
        max_ticks=max_ticks,
    )


def test_queueing_call_preserves_fifo_under_slot_steal_ordering():
    # A fresh caller (C) is scheduled while the previous request's completion is still
    # handing off to the queued waiter (B) -- i.e. before B resumes. The old
    # ready-queue handoff let C observe the momentarily-empty slot and claim it before
    # B, tripping B's `assert _result_event is None` and returning a 500 (see
    # test_get_server_info_concurrent). The lock makes the handoff atomic and FIFO.
    async def run():
        sent = []
        comm = FanOutCommunicator(send=sent.append, fan_out=1, mode="queueing")

        task_a = asyncio.create_task(comm("A"))
        await _pump_until(lambda: sent == ["A"])
        task_b = asyncio.create_task(comm("B"))
        await _pump_until_lock_waiters(comm, 1)

        # Complete A and let C arrive in the same tick, before A's handoff to B settles.
        comm.handle_recv("result-A")
        task_c = asyncio.create_task(comm("C"))

        # Drive to completion, delivering a reply to whoever legitimately holds the slot.
        for _ in range(200):
            await asyncio.sleep(0)
            if comm._result_event is not None and not comm._result_event.is_set():
                comm.handle_recv("result")
            if task_a.done() and task_b.done() and task_c.done():
                break

        results = await asyncio.gather(task_a, task_b, task_c, return_exceptions=True)
        for name, result in zip("ABC", results):
            assert not isinstance(
                result, BaseException
            ), f"caller {name} failed: {result!r}"
        # FIFO preserved: B is served before the later-arriving C.
        assert sent == ["A", "B", "C"], sent

    asyncio.run(run())


def test_queueing_call_cancelled_request_does_not_contaminate_next_caller():
    # If an in-flight caller is cancelled (e.g. client disconnect), its response is
    # still in flight. The call is shielded so it keeps holding the lock until that
    # response drains, so the next caller cannot send early and receive the stale
    # response. Invariant: only one request is in flight at a time.
    async def run():
        sent = []
        comm = FanOutCommunicator(send=sent.append, fan_out=1, mode="queueing")

        task_a = asyncio.create_task(comm("A"))
        await _pump_until(lambda: sent == ["A"])

        task_a.cancel()
        try:
            await task_a
        except asyncio.CancelledError:
            pass

        # The next caller must not send while A's response is still in flight.
        task_b = asyncio.create_task(comm("B"))
        await _pump_until_lock_waiters(comm, 1)
        assert sent == ["A"], f"B sent before A's response drained: {sent}"

        # A's response drains -> lock releases -> B may proceed with its own request.
        comm.handle_recv("result-A")
        await _pump_until(lambda: sent == ["A", "B"])
        comm.handle_recv("result-B")
        assert await task_b == ["result-B"]

    asyncio.run(run())
