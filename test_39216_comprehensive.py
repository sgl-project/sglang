"""Comprehensive CPU validation for SGLang #39216 fix — cancelling() version."""

import asyncio
import unittest
from unittest.mock import MagicMock

_cancelling_log = []

async def _is_disconnected(request) -> bool:
    try:
        return await request.is_disconnected()
    except asyncio.CancelledError:
        task = asyncio.current_task()
        c = task.cancelling() if task is not None else 0
        _cancelling_log.append(c)
        if task is not None and task.cancelling():
            raise
        return True


def make_stream_one_response(abort_request):
    async def _stream_one_response(obj, state, request):
        is_stream = getattr(obj, "stream", False)
        while True:
            try:
                if request is None:
                    await state.event.wait()
                else:
                    await asyncio.wait_for(state.event.wait(), timeout=0.01)
            except asyncio.TimeoutError:
                if (request is not None and not obj.background
                        and await _is_disconnected(request)):
                    abort_request(obj.rid)
                    raise ValueError(f"Request is disconnected from the client side (type 1). Abort request {obj.rid=}")
                continue
            out_list = state.out_list
            state.out_list = []
            finished = state.finished
            state.event.clear()
            if finished:
                yield out_list[-1]
                break
            if is_stream:
                yield out_list[-1]
            else:
                if (request is not None and not obj.background
                        and await _is_disconnected(request)):
                    abort_request(obj.rid)
                    raise ValueError(f"Request is disconnected from the client side (type 3). Abort request {obj.rid=}")
    return _stream_one_response


class FakeRequestConnected:
    async def is_disconnected(self): return False

class FakeRequestDisconnected:
    async def is_disconnected(self): return True

class FakeRequestCancelled:
    async def is_disconnected(self): raise asyncio.CancelledError("probe")

class FakeRequestError:
    async def is_disconnected(self): raise RuntimeError("internal")

class FakeRequestBlocking:
    def __init__(self): self._event = asyncio.Event()
    async def is_disconnected(self): await self._event.wait()


class TestHelperDirect(unittest.TestCase):
    def setUp(self): _cancelling_log.clear()
    def test_connected_returns_false(self):
        asyncio.run(self._run())
    async def _run(self):
        self.assertFalse(await _is_disconnected(FakeRequestConnected()))
    def test_disconnected_returns_true(self):
        self.assertTrue(asyncio.run(_is_disconnected(FakeRequestDisconnected())))
    def test_probe_cancelled_returns_true(self):
        self.assertTrue(asyncio.run(_is_disconnected(FakeRequestCancelled())))
    def test_runtime_error_propagates(self):
        async def r(): await _is_disconnected(FakeRequestError())
        with self.assertRaises(RuntimeError): asyncio.run(r())


class TestCancellingValues(unittest.TestCase):
    def setUp(self): _cancelling_log.clear()
    def test_probe_cancelling_is_zero(self):
        asyncio.run(_is_disconnected(FakeRequestCancelled()))
        self.assertEqual(_cancelling_log, [0])
    def test_outer_cancelling_is_positive(self):
        async def r():
            t = asyncio.create_task(_is_disconnected(FakeRequestBlocking()))
            await asyncio.sleep(0.01)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        asyncio.run(r())
        self.assertTrue(any(c > 0 for c in _cancelling_log))


class TestOuterCancellation(unittest.TestCase):
    def setUp(self): _cancelling_log.clear()
    def test_outer_task_cancellation_propagates(self):
        async def r():
            t = asyncio.create_task(_is_disconnected(FakeRequestBlocking()))
            await asyncio.sleep(0.01)
            t.cancel()
            with self.assertRaises(asyncio.CancelledError): await t
        asyncio.run(r())
    def test_probe_not_swallowed_by_outer_cancel_check(self):
        self.assertTrue(asyncio.run(_is_disconnected(FakeRequestCancelled())))


class TestStreamOneResponse(unittest.TestCase):
    def setUp(self):
        self.abort_calls = []; _cancelling_log.clear()
    def _abort(self, r): self.abort_calls.append(r)
    def _state(self, finished=True):
        s = MagicMock(); s.out_list = [{"text": "hello"}]; s.finished = finished
        s.event = asyncio.Event(); return s
    def _obj(self, stream=False, bg=False):
        o = MagicMock(); o.rid = "r"; o.stream = stream; o.background = bg; return o

    def test_site1_disconnected(self):
        gen = make_stream_one_response(self._abort)(obj=self._obj(), state=self._state(False), request=FakeRequestDisconnected())
        with self.assertRaises(ValueError) as cm:
            asyncio.run(gen.__anext__())
        self.assertIn("type 1", str(cm.exception)); self.assertEqual(self.abort_calls, ["r"])

    def test_site1_cancelled(self):
        gen = make_stream_one_response(self._abort)(obj=self._obj(), state=self._state(False), request=FakeRequestCancelled())
        with self.assertRaises(ValueError) as cm:
            asyncio.run(gen.__anext__())
        self.assertIn("type 1", str(cm.exception)); self.assertEqual(self.abort_calls, ["r"])

    def test_site1_outer_cancel(self):
        gen = make_stream_one_response(self._abort)(obj=self._obj(), state=self._state(False), request=FakeRequestBlocking())
        async def r():
            t = asyncio.create_task(gen.__anext__())
            await asyncio.sleep(0.02); t.cancel()
            with self.assertRaises(asyncio.CancelledError): await t
        asyncio.run(r())
        self.assertEqual(self.abort_calls, [])

    def test_site2_disconnected(self):
        s = self._state(False); s.event.set()
        gen = make_stream_one_response(self._abort)(obj=self._obj(), state=s, request=FakeRequestDisconnected())
        with self.assertRaises(ValueError) as cm:
            asyncio.run(gen.__anext__())
        self.assertIn("type 3", str(cm.exception)); self.assertEqual(self.abort_calls, ["r"])

    def test_site2_cancelled(self):
        s = self._state(False); s.event.set()
        gen = make_stream_one_response(self._abort)(obj=self._obj(), state=s, request=FakeRequestCancelled())
        with self.assertRaises(ValueError) as cm:
            asyncio.run(gen.__anext__())
        self.assertIn("type 3", str(cm.exception)); self.assertEqual(self.abort_calls, ["r"])

    def test_site2_outer_cancel(self):
        s = self._state(False); s.event.set()
        gen = make_stream_one_response(self._abort)(obj=self._obj(), state=s, request=FakeRequestBlocking())
        async def r():
            t = asyncio.create_task(gen.__anext__())
            await asyncio.sleep(0.02); t.cancel()
            with self.assertRaises(asyncio.CancelledError): await t
        asyncio.run(r())
        self.assertEqual(self.abort_calls, [])

    def test_site2_connected_no_abort(self):
        s = self._state(False); s.event.set()
        gen = make_stream_one_response(self._abort)(obj=self._obj(), state=s, request=FakeRequestConnected())
        async def r():
            try: await asyncio.wait_for(gen.__anext__(), timeout=0.02)
            except (TimeoutError, StopAsyncIteration, IndexError): pass
            except ValueError: self.fail("ValueError for connected")
        asyncio.run(r())
        self.assertEqual(self.abort_calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)