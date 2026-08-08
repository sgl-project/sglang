"""Contract tests for the load monitor decorator seam.

All observations go through public API only:
  - enable_load_monitor / bind_load_monitor
  - business return values and exceptions from decorated functions
  - callback invocation results

No registry introspection, no test-only getters, no production state added.
"""

from __future__ import annotations

import asyncio
import gc
import sys
import weakref
from typing import Any, Optional

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

pytest_plugins = ("pytest_asyncio",)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Minimal stand-ins (live only in test/)
# ---------------------------------------------------------------------------


class FakeServerArgs:
    """Minimal ServerArgs stand-in."""

    def __init__(self, port: Optional[int] = 30100):
        self.load_reporter_port = port


class FakeOwner:
    """Minimal owner that has server_args, mimicking TokenizerManager."""

    def __init__(self, port: Optional[int] = 30100):
        self.server_args = FakeServerArgs(port=port)

    @staticmethod
    def make_sync(port: Optional[int] = 30100) -> FakeOwner:
        return FakeOwner(port=port)


# ---------------------------------------------------------------------------
# Helpers to build decorated functions on FakeOwner
# ---------------------------------------------------------------------------


def make_sync_fn(owner: FakeOwner, payload_factory=None):
    """Return a synchronous scheduler-like method bound to *owner*."""
    from sglang.srt.load_reporter.decorator import enable_load_monitor

    @enable_load_monitor("scheduler_message")
    def dispatch(self, obj: Any) -> str:
        return "dispatched"

    return lambda obj: dispatch(owner, obj)


def make_async_gen(owner: FakeOwner, items=(1, 2, 3)):
    """Return an async-generator method bound to *owner* that yields *items*."""
    from sglang.srt.load_reporter.decorator import enable_load_monitor

    @enable_load_monitor("request_lifecycle")
    async def generate(self):
        for item in items:
            yield item

    async def call():
        async for item in generate(owner):
            yield item

    return call


# ---------------------------------------------------------------------------
# Fixture payloads
# ---------------------------------------------------------------------------


def _make_minimal_single():
    """Create a minimal but real TokenizedGenerateReqInput."""
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.sampling.sampling_params import SamplingParams

    return TokenizedGenerateReqInput(
        input_text=None,
        input_ids=None,
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        return_sampling_mask=False,
    )


def make_single_dispatch():
    return _make_minimal_single()


def make_batch_dispatch(n: int = 3):
    from sglang.srt.managers.io_struct import BatchTokenizedGenerateReqInput

    return BatchTokenizedGenerateReqInput(
        batch=[_make_minimal_single() for _ in range(n)]
    )


def make_abort():
    from sglang.srt.managers.io_struct import AbortReq

    return AbortReq(rid="r1", abort_all=False)


# ---------------------------------------------------------------------------
# Group 1: no binding — passthrough
# ---------------------------------------------------------------------------


class TestNoBinding:
    def test_sync_returns_without_callback(self):
        owner = FakeOwner()
        dispatch = make_sync_fn(owner)
        result = dispatch(make_single_dispatch())
        assert result == "dispatched"

    @pytest.mark.asyncio
    async def test_async_gen_yields_all_items(self):
        owner = FakeOwner()
        items = list(range(5))
        call = make_async_gen(owner, items)
        collected = [x async for x in call()]
        assert collected == items

    @pytest.mark.parametrize("port", [None, 30100])
    def test_async_iterator_is_returned_directly_when_inactive(self, port):
        from sglang.srt.load_reporter.decorator import enable_load_monitor

        class DirectAsyncIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        owner = FakeOwner(port=port)
        direct = DirectAsyncIterator()

        @enable_load_monitor("request_lifecycle")
        def generate(self):
            return direct

        assert generate(owner) is direct

    def test_sync_no_grpc_import_when_no_binding(self, monkeypatch):
        """Ensure grpc/protobuf are not imported via decorator when unbound."""
        import sys

        owner = FakeOwner()
        dispatch = make_sync_fn(owner)
        before = set(sys.modules.keys())
        dispatch(make_single_dispatch())
        after = set(sys.modules.keys())
        new_mods = after - before
        assert not any("grpc" in m or "protobuf" in m for m in new_mods)


# ---------------------------------------------------------------------------
# Group 2: two owners, independent callbacks
# ---------------------------------------------------------------------------


class TestMultipleOwners:
    def test_two_owners_independent_callbacks(self):
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )

        events_a: list = []
        events_b: list = []

        @enable_load_monitor("scheduler_message")
        def dispatch(self, obj: Any) -> None:
            pass

        owner_a = FakeOwner()
        owner_b = FakeOwner()
        unbind_a = bind_load_monitor(owner_a, lambda r, c: events_a.append((r, c)))
        bind_load_monitor(owner_b, lambda r, c: events_b.append((r, c)))

        payload = make_single_dispatch()
        dispatch(owner_a, payload)
        dispatch(owner_b, payload)

        assert len(events_a) == 1
        assert len(events_b) == 1

        # Unbind a — b still fires
        unbind_a()
        dispatch(owner_a, payload)
        dispatch(owner_b, payload)

        assert len(events_a) == 1  # unchanged after unbind
        assert len(events_b) == 2  # still increments

    def test_unbind_is_idempotent(self):
        from sglang.srt.load_reporter.decorator import bind_load_monitor

        owner = FakeOwner()
        unbind = bind_load_monitor(owner, lambda r, c: None)
        unbind()
        unbind()  # must not raise

    def test_stale_unbind_does_not_remove_newer_binding(self):
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )

        owner = FakeOwner()
        old_events = []
        new_events = []

        @enable_load_monitor("scheduler_message")
        def dispatch(self, obj):
            return None

        unbind_old = bind_load_monitor(
            owner, lambda reason, count: old_events.append((reason, count))
        )
        bind_load_monitor(
            owner, lambda reason, count: new_events.append((reason, count))
        )

        unbind_old()
        dispatch(owner, make_single_dispatch())

        assert old_events == []
        assert len(new_events) == 1

    def test_owner_gc_removes_registry_entry(self):
        from sglang.srt.load_reporter.decorator import _REGISTRY, bind_load_monitor

        owner = FakeOwner()
        bind_load_monitor(owner, lambda r, c: None)
        owner_ref = weakref.ref(owner)

        del owner
        gc.collect()

        assert owner_ref() is None
        # WeakKeyDictionary must have cleaned up
        assert all(owner_ref() is not k for k in list(_REGISTRY.keys()))


# ---------------------------------------------------------------------------
# Group 3: scheduler_message classification
# ---------------------------------------------------------------------------


class TestSchedulerMessageClassification:
    def _run(self, payload, port=30100):
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )

        events: list = []
        owner = FakeOwner(port=port)

        @enable_load_monitor("scheduler_message")
        def dispatch(self, obj: Any) -> None:
            pass

        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        dispatch(owner, payload)
        return events

    def test_single_dispatch_reason_and_count(self):
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        events = self._run(make_single_dispatch())
        assert events == [(LoadReporterRefreshReason.DISPATCH, 1)]

    def test_batch_dispatch_count_matches_batch_size(self):
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        batch = make_batch_dispatch(n=5)
        events = self._run(batch)
        assert events == [(LoadReporterRefreshReason.DISPATCH, 5)]

    def test_abort_emits_abort_reason(self):
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        events = self._run(make_abort())
        assert events == [(LoadReporterRefreshReason.ABORT, 1)]

    def test_unknown_message_type_not_notified(self):
        events = self._run(object())  # unknown payload
        assert events == []

    def test_no_event_on_sync_exception(self):
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )

        events: list = []
        owner = FakeOwner()

        @enable_load_monitor("scheduler_message")
        def dispatch(self, obj: Any) -> None:
            raise RuntimeError("boom")

        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        with pytest.raises(RuntimeError, match="boom"):
            dispatch(owner, make_single_dispatch())
        assert events == []


# ---------------------------------------------------------------------------
# Group 4: request_lifecycle async generator
# ---------------------------------------------------------------------------


class TestRequestLifecycle:
    @pytest.mark.asyncio
    async def test_normal_exhaustion_emits_one_completion(self):
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        events: list = []
        owner = FakeOwner()

        @enable_load_monitor("request_lifecycle")
        async def generate(self):
            for i in range(3):
                yield i

        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        collected = [x async for x in generate(owner)]
        assert collected == [0, 1, 2]
        assert events == [(LoadReporterRefreshReason.COMPLETION, 1)]

    @pytest.mark.asyncio
    async def test_business_exception_still_emits_completion(self):
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        events: list = []
        owner = FakeOwner()

        @enable_load_monitor("request_lifecycle")
        async def generate(self):
            yield 1
            raise ValueError("fail")

        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        with pytest.raises(ValueError, match="fail"):
            async for _ in generate(owner):
                pass

        assert events == [(LoadReporterRefreshReason.COMPLETION, 1)]

    @pytest.mark.asyncio
    async def test_consumer_aclose_emits_completion(self):
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        events: list = []
        owner = FakeOwner()

        @enable_load_monitor("request_lifecycle")
        async def generate(self):
            for i in range(100):
                yield i

        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        gen = generate(owner)
        await gen.__anext__()  # consume one
        await gen.aclose()  # close early

        assert events == [(LoadReporterRefreshReason.COMPLETION, 1)]

    @pytest.mark.asyncio
    async def test_early_aclose_finalizes_underlying_source(self):
        """I2: closing the wrapper early must close the underlying generator so
        its ``finally`` (request cleanup) runs, not just the COMPLETION event."""
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        events: list = []
        source_finalized: list = []
        owner = FakeOwner()

        @enable_load_monitor("request_lifecycle")
        async def generate(self):
            try:
                for i in range(100):
                    yield i
            finally:
                source_finalized.append(True)

        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        gen = generate(owner)
        await gen.__anext__()  # suspend inside the underlying source
        await gen.aclose()  # close the wrapper early

        assert events == [(LoadReporterRefreshReason.COMPLETION, 1)]
        assert source_finalized == [
            True
        ], "underlying source generator must be finalized on early aclose"

    @pytest.mark.asyncio
    async def test_early_aclose_finalizes_bound_underlying_source(self):
        """I2 (bound style): early aclose finalizes the underlying bound source."""
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        events: list = []
        source_finalized: list = []

        class Owner:
            def __init__(self) -> None:
                self.server_args = FakeServerArgs(port=30100)

            async def generate_request(self, tag: str = "x"):
                try:
                    for i in range(100):
                        yield (tag, i)
                finally:
                    source_finalized.append(True)

        owner = Owner()
        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        # Bound-method style: decorate the instance's bound method (standalone
        # SMG RPC shape).
        decorated = enable_load_monitor("request_lifecycle")(owner.generate_request)
        gen = decorated("t")
        await gen.__anext__()
        await gen.aclose()

        assert events == [(LoadReporterRefreshReason.COMPLETION, 1)]
        assert source_finalized == [True]

    @pytest.mark.asyncio
    async def test_task_cancel_emits_completion(self):
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        events: list = []
        owner = FakeOwner()

        @enable_load_monitor("request_lifecycle")
        async def generate(self):
            for i in range(100):
                yield i
                await asyncio.sleep(0)

        bind_load_monitor(owner, lambda r, c: events.append((r, c)))

        async def consume():
            async for _ in generate(owner):
                pass

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0)  # let it start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert events == [(LoadReporterRefreshReason.COMPLETION, 1)]

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_mask_business_exception(self):
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )

        owner = FakeOwner()

        @enable_load_monitor("request_lifecycle")
        async def generate(self):
            yield 1
            raise ValueError("business error")

        def bad_notify(r, c):
            raise RuntimeError("notify exploded")

        bind_load_monitor(owner, bad_notify)
        with pytest.raises(ValueError, match="business error"):
            async for _ in generate(owner):
                pass

    @pytest.mark.asyncio
    async def test_completion_not_fired_when_port_is_none(self):
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )

        events: list = []
        owner = FakeOwner(port=None)  # reporter disabled

        @enable_load_monitor("request_lifecycle")
        async def generate(self):
            yield 1
            yield 2

        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        collected = [x async for x in generate(owner)]
        assert collected == [1, 2]
        assert events == []  # bypassed entirely


# ---------------------------------------------------------------------------
# Group 5: port=None bypass
# ---------------------------------------------------------------------------


class TestPortBypass:
    def test_sync_no_callback_when_port_none(self):
        from sglang.srt.load_reporter.decorator import (
            bind_load_monitor,
            enable_load_monitor,
        )

        events: list = []
        owner = FakeOwner(port=None)

        @enable_load_monitor("scheduler_message")
        def dispatch(self, obj: Any) -> str:
            return "ok"

        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        result = dispatch(owner, make_single_dispatch())
        assert result == "ok"
        assert events == []


# ---------------------------------------------------------------------------
# Group 6: request_lifecycle applied to a BOUND method (standalone SMG shape)
# ---------------------------------------------------------------------------


class BoundOwner:
    """Owner whose async-generator method stays undecorated in the class body;
    the decorator is applied at runtime to the *bound* method of one instance."""

    def __init__(self, port: Optional[int] = 30100, n: int = 3, fail: bool = False):
        self.server_args = FakeServerArgs(port=port)
        self._n = n
        self._fail = fail

    async def generate_request(self, tag: str = "x"):
        for i in range(self._n):
            yield (tag, i)
            await asyncio.sleep(0)  # realistic suspension point (awaits scheduler)
        if self._fail:
            raise ValueError("business error")


def _install_bound(owner: BoundOwner):
    """Apply enable_load_monitor to the bound method and shadow the instance,
    mirroring the lifecycle installation without importing lifecycle."""
    from sglang.srt.load_reporter.decorator import enable_load_monitor

    original = owner.generate_request
    decorated = enable_load_monitor("request_lifecycle")(original)
    owner.generate_request = decorated
    return original, decorated


class TestBoundRequestLifecycle:
    @pytest.mark.asyncio
    async def test_bound_normal_exhaustion_one_completion(self):
        from sglang.srt.load_reporter.decorator import bind_load_monitor
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        events: list = []
        owner = BoundOwner(n=3)
        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        _install_bound(owner)

        collected = [x async for x in owner.generate_request("t")]
        assert collected == [("t", 0), ("t", 1), ("t", 2)]
        assert events == [(LoadReporterRefreshReason.COMPLETION, 1)]

    @pytest.mark.asyncio
    async def test_bound_business_exception_still_one_completion(self):
        from sglang.srt.load_reporter.decorator import bind_load_monitor
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        events: list = []
        owner = BoundOwner(n=1, fail=True)
        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        _install_bound(owner)

        with pytest.raises(ValueError, match="business error"):
            async for _ in owner.generate_request():
                pass
        assert events == [(LoadReporterRefreshReason.COMPLETION, 1)]

    @pytest.mark.asyncio
    async def test_bound_consumer_aclose_one_completion(self):
        from sglang.srt.load_reporter.decorator import bind_load_monitor
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        events: list = []
        owner = BoundOwner(n=100)
        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        _install_bound(owner)

        gen = owner.generate_request()
        await gen.__anext__()
        await gen.aclose()
        assert events == [(LoadReporterRefreshReason.COMPLETION, 1)]

    @pytest.mark.asyncio
    async def test_bound_cancellation_one_completion(self):
        from sglang.srt.load_reporter.decorator import bind_load_monitor
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason

        events: list = []
        owner = BoundOwner(n=100)
        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        _install_bound(owner)

        async def consume():
            async for _ in owner.generate_request():
                pass  # only the generator awaits, so cancel lands inside it

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == [(LoadReporterRefreshReason.COMPLETION, 1)]

    @pytest.mark.asyncio
    async def test_bound_callback_exception_does_not_mask_business(self):
        from sglang.srt.load_reporter.decorator import bind_load_monitor

        owner = BoundOwner(n=1, fail=True)
        bind_load_monitor(
            owner, lambda r, c: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        _install_bound(owner)

        with pytest.raises(ValueError, match="business error"):
            async for _ in owner.generate_request():
                pass

    @pytest.mark.asyncio
    async def test_class_method_unchanged(self):
        """Shadowing one instance must not alter the class method."""
        from sglang.srt.load_reporter.decorator import bind_load_monitor

        events: list = []
        owner = BoundOwner(n=2)
        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        _install_bound(owner)

        # A second instance resolves the pristine class method (no events).
        other = BoundOwner(n=2)
        other_events: list = []
        bind_load_monitor(other, lambda r, c: other_events.append((r, c)))
        collected = [x async for x in other.generate_request("o")]
        assert collected == [("o", 0), ("o", 1)]
        assert other_events == []  # class method is undecorated

    @pytest.mark.asyncio
    async def test_wraps_preserves_name_and_wrapped(self):
        from sglang.srt.load_reporter.decorator import enable_load_monitor

        owner = BoundOwner()
        original = owner.generate_request
        decorated = enable_load_monitor("request_lifecycle")(original)
        assert decorated.__name__ == "generate_request"
        assert getattr(decorated, "__wrapped__", None) is original

    @pytest.mark.asyncio
    async def test_bound_disabled_when_port_none(self):
        from sglang.srt.load_reporter.decorator import bind_load_monitor

        events: list = []
        owner = BoundOwner(port=None, n=2)
        bind_load_monitor(owner, lambda r, c: events.append((r, c)))
        _install_bound(owner)

        collected = [x async for x in owner.generate_request()]
        assert collected == [("x", 0), ("x", 1)]
        assert events == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
