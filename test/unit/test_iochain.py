"""
Unit tests for the IOChain filter pipeline.

Tests are organised into five classes:
  - TestIOChainPipeline   : core ingress/egress ordering and exception isolation
  - TestStreamingEgress   : _wrap_streaming_egress wires _after_inference correctly
  - TestLoader            : loader discovers entry-points and CLI paths
  - TestServingBaseWiring : set_iochain / _before_inference / _after_inference on base
  - TestRequestLoggingFilter : reference filter handles streaming + non-streaming
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.srt.iochain.base import IOChain, IOContext, IOFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**kwargs) -> IOContext:
    defaults = dict(
        request_id="req-test",
        raw_request=MagicMock(),
        adapted_request=MagicMock(),
    )
    defaults.update(kwargs)
    return IOContext(**defaults)


class _RecordingFilter(IOFilter):
    """Logs every call to a shared list for assertion."""

    blocking = True

    def __init__(self, name: str, log: list):
        self.name = name
        self._log = log

    async def on_request(self, ctx: IOContext) -> None:
        self._log.append(f"{self.name}.on_request")

    async def on_response(self, ctx: IOContext) -> None:
        self._log.append(f"{self.name}.on_response")


class _RaisingFilter(IOFilter):
    """Always raises on ingress."""

    blocking = True

    async def on_request(self, ctx: IOContext) -> None:
        raise ValueError("intentional ingress error")

    async def on_response(self, ctx: IOContext) -> None:  # pragma: no cover
        pass


class _NonBlockingFilter(IOFilter):
    """Non-blocking; records calls."""

    blocking = False

    def __init__(self, log: list):
        self._log = log

    async def on_request(self, ctx: IOContext) -> None:
        self._log.append("nb.on_request")

    async def on_response(self, ctx: IOContext) -> None:
        self._log.append("nb.on_response")


# ---------------------------------------------------------------------------
# TestIOChainPipeline
# ---------------------------------------------------------------------------


class TestIOChainPipeline(unittest.IsolatedAsyncioTestCase):

    async def test_ingress_runs_filters_in_insertion_order(self):
        log = []
        chain = IOChain()
        chain.add(_RecordingFilter("A", log)).add(_RecordingFilter("B", log))
        await chain.run_ingress(_make_ctx())
        self.assertEqual(log, ["A.on_request", "B.on_request"])

    async def test_egress_runs_filters_in_reverse_order(self):
        log = []
        chain = IOChain()
        chain.add(_RecordingFilter("A", log)).add(_RecordingFilter("B", log))
        await chain.run_egress(_make_ctx())
        self.assertEqual(log, ["B.on_response", "A.on_response"])

    async def test_exception_in_blocking_filter_propagates(self):
        chain = IOChain()
        chain.add(_RaisingFilter())
        with self.assertRaises(Exception):
            # _run_safe logs + re-raises for blocking filters
            await chain.run_ingress(_make_ctx())

    async def test_exception_in_non_blocking_filter_is_swallowed(self):
        """Non-blocking filters are fire-and-forget; errors must not crash caller."""
        log = []

        class _NbRaising(IOFilter):
            blocking = False

            async def on_request(self, ctx: IOContext) -> None:
                raise RuntimeError("nb error")

            async def on_response(self, ctx: IOContext) -> None:
                pass

        chain = IOChain()
        chain.add(_NbRaising())
        # Should complete without raising; background task absorbs the error
        await chain.run_ingress(_make_ctx())
        # Drain pending tasks
        await asyncio.sleep(0)

    async def test_make_context_sets_request_id(self):
        chain = IOChain()
        ctx = chain.make_context(MagicMock(), MagicMock())
        self.assertIsNotNone(ctx.request_id)
        self.assertIsInstance(ctx.request_id, str)
        self.assertGreater(len(ctx.request_id), 0)

    async def test_empty_chain_ingress_is_noop(self):
        chain = IOChain()
        # Must not raise
        await chain.run_ingress(_make_ctx())

    async def test_empty_chain_egress_is_noop(self):
        chain = IOChain()
        await chain.run_egress(_make_ctx())


# ---------------------------------------------------------------------------
# TestStreamingEgress
# ---------------------------------------------------------------------------


class TestStreamingEgress(unittest.IsolatedAsyncioTestCase):
    """_wrap_streaming_egress must call _after_inference after last chunk."""

    def _make_serving_base(self, chain: IOChain):
        """Return a minimal OpenAIServingBase-like stub with set_iochain wired."""
        from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase

        class _Stub(OpenAIServingBase):
            def _request_id_prefix(self):
                return "stub-"

            def _convert_to_internal_request(self, request, raw_request=None):
                return MagicMock(), request

        stub = _Stub.__new__(_Stub)
        stub._iochain = None
        stub._iochain = chain
        return stub

    async def test_after_inference_called_after_last_chunk(self):
        from fastapi.responses import StreamingResponse

        fired = []

        async def _gen():
            yield b"chunk1"
            yield b"chunk2"

        chain = IOChain()
        stub = self._make_serving_base(chain)
        stub._after_inference = AsyncMock(side_effect=lambda *_: fired.append(True))

        sr = StreamingResponse(_gen(), media_type="text/plain")
        wrapped = stub._wrap_streaming_egress(sr, MagicMock(), MagicMock())

        # Consume the wrapped generator
        chunks = []
        async for chunk in wrapped.body_iterator:
            chunks.append(chunk)

        self.assertEqual(chunks, [b"chunk1", b"chunk2"])
        self.assertEqual(len(fired), 1)

    async def test_after_inference_called_even_on_generator_error(self):
        from fastapi.responses import StreamingResponse

        fired = []

        async def _failing_gen():
            yield b"first"
            raise RuntimeError("stream error")

        chain = IOChain()
        stub = self._make_serving_base(chain)
        stub._after_inference = AsyncMock(side_effect=lambda *_: fired.append(True))

        sr = StreamingResponse(_failing_gen(), media_type="text/plain")
        wrapped = stub._wrap_streaming_egress(sr, MagicMock(), MagicMock())

        with self.assertRaises(RuntimeError):
            async for _ in wrapped.body_iterator:
                pass

        self.assertEqual(len(fired), 1)

    async def test_non_streaming_response_returned_unchanged(self):
        from fastapi.responses import ORJSONResponse

        chain = IOChain()
        stub = self._make_serving_base(chain)
        resp = ORJSONResponse({"ok": True})
        result = stub._wrap_streaming_egress(resp, MagicMock(), MagicMock())
        self.assertIs(result, resp)

    async def test_response_is_none_in_after_inference_for_streaming(self):
        """ctx.response must be None when _after_inference fires for streaming."""
        from fastapi.responses import StreamingResponse

        captured_response = []

        async def _gen():
            yield b"data"

        log = []

        class _CheckFilter(IOFilter):
            blocking = False

            async def on_request(self, ctx: IOContext) -> None:
                pass

            async def on_response(self, ctx: IOContext) -> None:
                captured_response.append(ctx.response)

        chain = IOChain()
        chain.add(_CheckFilter())

        from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase

        class _Stub(OpenAIServingBase):
            def _request_id_prefix(self):
                return "s-"

            def _convert_to_internal_request(self, r, raw=None):
                return MagicMock(), r

        stub = _Stub.__new__(_Stub)
        stub._iochain = chain

        request = MagicMock()
        adapted = MagicMock()

        # Simulate ingress to create ctx
        await stub._before_inference(request, adapted)

        sr = StreamingResponse(_gen())
        wrapped = stub._wrap_streaming_egress(sr, request, adapted)
        async for _ in wrapped.body_iterator:
            pass
        await asyncio.sleep(0)

        self.assertEqual(len(captured_response), 1)
        self.assertIsNone(captured_response[0])


# ---------------------------------------------------------------------------
# TestLoader
# ---------------------------------------------------------------------------


class TestLoader(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Reset default chain before each test
        import sglang.srt.iochain as pkg
        pkg._default_chain = IOChain()

    def _make_server_args(self, filters=None):
        args = MagicMock()
        args.iochain_filters = filters or []
        return args

    def test_cli_filter_loaded_by_module_path(self):
        """_load_cli_filters should import and instantiate a filter by path."""
        # Register a fake module
        fake_module = types.ModuleType("_test_filters_cli")
        fake_module.MyFilter = type(
            "MyFilter",
            (IOFilter,),
            {
                "blocking": False,
                "on_request": AsyncMock(),
                "on_response": AsyncMock(),
            },
        )
        sys.modules["_test_filters_cli"] = fake_module

        try:
            from sglang.srt.iochain.loader import _load_cli_filters

            chain = IOChain()
            _load_cli_filters(chain, ["_test_filters_cli:MyFilter"])
            self.assertEqual(len(chain._filters), 1)
            self.assertIsInstance(chain._filters[0], fake_module.MyFilter)
        finally:
            del sys.modules["_test_filters_cli"]

    def test_cli_filter_bad_path_raises_value_error(self):
        from sglang.srt.iochain.loader import _load_cli_filters

        with self.assertRaises(ValueError):
            _load_cli_filters(IOChain(), ["no_colon_here"])

    def test_cli_filter_non_iofilter_raises_type_error(self):
        fake_module = types.ModuleType("_test_bad")
        fake_module.NotAFilter = object  # not an IOFilter subclass
        sys.modules["_test_bad"] = fake_module
        try:
            from sglang.srt.iochain.loader import _load_cli_filters

            with self.assertRaises(TypeError):
                _load_cli_filters(IOChain(), ["_test_bad:NotAFilter"])
        finally:
            del sys.modules["_test_bad"]

    def test_entry_point_filter_loaded(self):
        """Entry-point discovery should instantiate filters declared in the group."""
        fake_module = types.ModuleType("_test_ep")

        class _EPFilter(IOFilter):
            blocking = False

            async def on_request(self, ctx):
                pass

            async def on_response(self, ctx):
                pass

        fake_module._EPFilter = _EPFilter
        sys.modules["_test_ep"] = fake_module

        fake_ep = MagicMock()
        fake_ep.name = "ep_filter"
        fake_ep.value = "_test_ep:_EPFilter"
        fake_ep.load.return_value = _EPFilter

        try:
            from sglang.srt.iochain.loader import _load_entry_points

            with patch(
                "sglang.srt.iochain.loader.entry_points",
                return_value=[fake_ep],
            ):
                chain = IOChain()
                _load_entry_points(chain)

            self.assertEqual(len(chain._filters), 1)
            self.assertIsInstance(chain._filters[0], _EPFilter)
        finally:
            del sys.modules["_test_ep"]

    def test_entry_point_non_iofilter_is_skipped(self):
        fake_ep = MagicMock()
        fake_ep.name = "bad_ep"
        fake_ep.value = "not.a.filter"
        fake_ep.load.return_value = object  # not IOFilter

        from sglang.srt.iochain.loader import _load_entry_points

        with patch("sglang.srt.iochain.loader.entry_points", return_value=[fake_ep]):
            chain = IOChain()
            _load_entry_points(chain)

        self.assertEqual(len(chain._filters), 0)

    def test_load_iochain_combines_both_sources(self):
        """load_iochain should apply entry-point filters then CLI filters."""
        fake_module = types.ModuleType("_test_combo")

        class _A(IOFilter):
            blocking = False

            async def on_request(self, ctx):
                pass

            async def on_response(self, ctx):
                pass

        class _B(IOFilter):
            blocking = False

            async def on_request(self, ctx):
                pass

            async def on_response(self, ctx):
                pass

        fake_module._A = _A
        fake_module._B = _B
        sys.modules["_test_combo"] = fake_module

        fake_ep = MagicMock()
        fake_ep.name = "a_filter"
        fake_ep.value = "_test_combo:_A"
        fake_ep.load.return_value = _A

        try:
            from sglang.srt.iochain.loader import load_iochain

            args = self._make_server_args(["_test_combo:_B"])

            with patch(
                "sglang.srt.iochain.loader.entry_points",
                return_value=[fake_ep],
            ):
                chain = load_iochain(args)

            self.assertEqual(len(chain._filters), 2)
            self.assertIsInstance(chain._filters[0], _A)  # entry-point first
            self.assertIsInstance(chain._filters[1], _B)  # CLI second
        finally:
            del sys.modules["_test_combo"]

    def test_load_iochain_empty_when_no_plugins(self):
        from sglang.srt.iochain.loader import load_iochain

        args = self._make_server_args([])
        with patch("sglang.srt.iochain.loader.entry_points", return_value=[]):
            chain = load_iochain(args)
        self.assertEqual(len(chain._filters), 0)


# ---------------------------------------------------------------------------
# TestServingBaseWiring
# ---------------------------------------------------------------------------


class TestServingBaseWiring(unittest.IsolatedAsyncioTestCase):
    """set_iochain / _before_inference / _after_inference on OpenAIServingBase."""

    def _make_stub(self):
        from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase

        class _Stub(OpenAIServingBase):
            def _request_id_prefix(self):
                return "t-"

            def _convert_to_internal_request(self, r, raw=None):
                return MagicMock(), r

        stub = _Stub.__new__(_Stub)
        stub._iochain = None
        return stub

    async def test_no_chain_before_inference_is_noop(self):
        stub = self._make_stub()
        # Must not raise
        await stub._before_inference(MagicMock(), MagicMock())

    async def test_no_chain_after_inference_is_noop(self):
        stub = self._make_stub()
        await stub._after_inference(MagicMock(), MagicMock(), MagicMock())

    async def test_set_iochain_then_filter_called(self):
        log = []
        chain = IOChain()
        chain.add(_RecordingFilter("X", log))

        stub = self._make_stub()
        stub.set_iochain(chain)

        request = MagicMock()
        adapted = MagicMock()
        await stub._before_inference(request, adapted)
        await stub._after_inference(request, adapted, MagicMock())

        self.assertIn("X.on_request", log)
        self.assertIn("X.on_response", log)

    async def test_empty_chain_is_noop_even_after_set_iochain(self):
        """An empty chain (no filters) should add zero overhead."""
        chain = IOChain()
        stub = self._make_stub()
        stub.set_iochain(chain)
        # Must not raise
        await stub._before_inference(MagicMock(), MagicMock())
        await stub._after_inference(MagicMock(), MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# TestRequestLoggingFilter
# ---------------------------------------------------------------------------


class TestRequestLoggingFilter(unittest.IsolatedAsyncioTestCase):

    async def test_on_response_non_streaming_logs_streaming_false(self):
        from sglang.srt.iochain.filters.request_logging import RequestLoggingFilter

        f = RequestLoggingFilter()
        ctx = _make_ctx(response=MagicMock())  # non-streaming: response set

        with self.assertLogs("sglang.srt.iochain.filters.request_logging", level="INFO") as cm:
            await f.on_response(ctx)

        output = " ".join(cm.output)
        self.assertIn("request.complete", output)

    async def test_on_response_streaming_ctx_response_is_none(self):
        """When ctx.response is None (streaming), filter must not raise."""
        from sglang.srt.iochain.filters.request_logging import RequestLoggingFilter

        f = RequestLoggingFilter()
        ctx = _make_ctx(response=None)  # streaming: response is None

        with self.assertLogs("sglang.srt.iochain.filters.request_logging", level="INFO"):
            await f.on_response(ctx)  # must not raise

    async def test_on_request_does_not_raise(self):
        from sglang.srt.iochain.filters.request_logging import RequestLoggingFilter

        f = RequestLoggingFilter()
        ctx = _make_ctx()
        await f.on_request(ctx)  # must not raise


if __name__ == "__main__":
    unittest.main()
