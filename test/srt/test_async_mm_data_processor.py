"""
Unit tests for AsyncMMDataProcessor.

Covers:
  - Async and sync processing paths
  - Concurrency limiting via semaphore
  - Per-call timeout behavior (async and sync)
  - Argument passthrough (images, audios, text/ids, request_obj, kwargs)
  - Error propagation and shutdown behavior
"""

import asyncio
import logging
import threading
import time
from unittest.mock import Mock

import pytest

from sglang.srt.managers.async_mm_data_processor import AsyncMMDataProcessor


class TestAsyncMMDataProcessor:
    """Test suite for AsyncMMDataProcessor."""

    @pytest.fixture
    def async_processor(self):
        """Create a processor exposing an async process_mm_data_async."""

        class AsyncProc:
            async def process_mm_data_async(
                self,
                *,
                image_data=None,
                audio_data=None,
                input_text=None,
                request_obj=None,
                **kwargs,
            ):
                # Allow tests to simulate latency via kwargs
                delay = kwargs.get("delay_s", 0.0)
                if delay:
                    await asyncio.sleep(delay)
                return {
                    "path": "async",
                    "images": image_data,
                    "audios": audio_data,
                    "text": input_text,
                    "request": request_obj,
                    "kwargs": kwargs,
                }

        return AsyncProc()

    @pytest.fixture
    def sync_processor(self):
        """Provide a processor exposing a sync process_mm_data."""

        class SyncProc:
            def process_mm_data(
                self,
                *,
                image_data=None,
                audio_data=None,
                input_text=None,
                request_obj=None,
                **kwargs,
            ):
                delay = kwargs.get("delay_s", 0.0)
                if delay:
                    # Simulate CPU/blocking work
                    time.sleep(delay)
                return {
                    "path": "sync",
                    "images": image_data,
                    "audios": audio_data,
                    "text": input_text,
                    "request": request_obj,
                    "kwargs": kwargs,
                }

        return SyncProc()

    @pytest.mark.asyncio
    async def test_async_path_basic(self, async_processor):
        """Async processor should be awaited directly."""
        proc = AsyncMMDataProcessor(async_processor)
        out = await proc.process(
            image_data=["img1.png"],
            audio_data=["a.wav"],
            input_text_or_ids="hello",
            request_obj={"rid": 1},
            mode="fast",
        )
        assert out["path"] == "async"
        assert out["images"] == ["img1.png"]
        assert out["audios"] == ["a.wav"]
        assert out["text"] == "hello"
        assert out["request"] == {"rid": 1}
        assert out["kwargs"]["mode"] == "fast"

    @pytest.mark.asyncio
    async def test_sync_fallback_basic(self, sync_processor):
        """Sync processor should run in fallback executor."""
        proc = AsyncMMDataProcessor(sync_processor)
        out = await proc.process(
            image_data=[b"\x00\x01"],
            audio_data=None,
            input_text_or_ids=[1, 2, 3],
            request_obj="req-obj",
            role="user",
        )
        assert out["path"] == "sync"
        assert out["images"] == [b"\x00\x01"]
        assert out["audios"] is None
        assert out["text"] == [1, 2, 3]
        assert out["request"] == "req-obj"
        assert out["kwargs"]["role"] == "user"

    @pytest.mark.asyncio
    async def test_timeout_async(self, async_processor):
        """Timeout should raise asyncio.TimeoutError for async path."""
        proc = AsyncMMDataProcessor(async_processor, timeout_s=0.01)
        with pytest.raises(asyncio.TimeoutError):
            await proc.process(
                input_text_or_ids="slow",
                request_obj=None,
                delay_s=0.05,  # longer than timeout
            )

    @pytest.mark.asyncio
    async def test_timeout_sync(self, sync_processor):
        """Timeout should raise asyncio.TimeoutError for sync fallback path."""
        proc = AsyncMMDataProcessor(sync_processor, timeout_s=0.01)
        with pytest.raises(asyncio.TimeoutError):
            await proc.process(
                input_text_or_ids="slow",
                request_obj=None,
                delay_s=0.05,  # longer than timeout
            )

    @pytest.mark.asyncio
    async def test_semaphore_release_after_timeout(self, sync_processor):
        """
        If a call times out, the semaphore should be released so a subsequent call can proceed.
        Use >=2 fallback workers so the timed-out thread doesn't block the next call.
        """
        proc = AsyncMMDataProcessor(
            sync_processor,
            max_concurrent_calls=2,
            timeout_s=0.01,
        )

        # First call will time out
        with pytest.raises(asyncio.TimeoutError):
            await proc.process(
                input_text_or_ids="slow1", request_obj=None, delay_s=0.05
            )

        # Second call should be able to acquire the semaphore and complete
        out = await proc.process(input_text_or_ids="ok", request_obj=None, delay_s=0.0)
        assert out["text"] == "ok"

    @pytest.mark.asyncio
    async def test_concurrency_limit_async(self):
        """Ensure max_concurrent_calls caps concurrency for async path."""
        current = 0
        max_seen = 0

        class AsyncProc:
            async def process_mm_data_async(self, **kwargs):
                nonlocal current, max_seen
                current += 1
                max_seen = max(max_seen, current)
                try:
                    await asyncio.sleep(0.02)
                    return {"ok": True}
                finally:
                    current -= 1

        proc = AsyncMMDataProcessor(AsyncProc(), max_concurrent_calls=2)

        tasks = [
            proc.process(input_text_or_ids=f"t{i}", request_obj=None) for i in range(6)
        ]
        await asyncio.gather(*tasks)

        assert max_seen <= 2

    @pytest.mark.asyncio
    async def test_concurrency_limit_sync(self):
        """Ensure max_concurrent_calls caps concurrency for sync fallback path."""
        current = 0
        max_seen = 0
        lock = threading.Lock()

        class SyncProc:
            def process_mm_data(self, **kwargs):
                nonlocal current, max_seen
                with lock:
                    current += 1
                    max_seen = max(max_seen, current)
                try:
                    time.sleep(0.02)
                    return {"ok": True}
                finally:
                    with lock:
                        current -= 1

        proc = AsyncMMDataProcessor(SyncProc(), max_concurrent_calls=3)

        tasks = [
            proc.process(input_text_or_ids=f"s{i}", request_obj=None) for i in range(9)
        ]
        await asyncio.gather(*tasks)

        assert max_seen <= 3

    @pytest.mark.asyncio
    async def test_error_from_async_processor(self):
        """Exceptions raised by the async processor should propagate."""

        class BadAsync:
            async def process_mm_data_async(self, **_):
                await asyncio.sleep(0)
                raise ValueError("async boom")

        proc = AsyncMMDataProcessor(BadAsync())
        with pytest.raises(ValueError, match="async boom"):
            await proc.process(input_text_or_ids="x", request_obj=None)

    @pytest.mark.asyncio
    async def test_error_from_sync_processor(self):
        """Exceptions raised by the sync processor should propagate."""

        class BadSync:
            def process_mm_data(self, **_):
                raise RuntimeError("sync boom")

        proc = AsyncMMDataProcessor(BadSync())
        with pytest.raises(RuntimeError, match="sync boom"):
            await proc.process(input_text_or_ids="x", request_obj=None)

    @pytest.mark.asyncio
    async def test_missing_both_methods_raises(self):
        """Processor missing both methods should raise at call time."""

        class Empty:
            pass

        proc = AsyncMMDataProcessor(Empty())
        with pytest.raises(
            RuntimeError, match="neither 'process_mm_data_async' nor 'process_mm_data'"
        ):
            await proc.process(input_text_or_ids="x", request_obj=None)

    @pytest.mark.asyncio
    async def test_async_attribute_not_coroutine_uses_sync_fallback(self):
        """
        If `process_mm_data_async` exists but isn't a coroutine function,
        wrapper should treat it as sync and use `process_mm_data`.
        """

        class WeirdProc:
            # Not a coroutine function:
            def process_mm_data_async(self, **_):
                return {"path": "would-be-async"}

            def process_mm_data(self, **_):
                return {"path": "sync"}

        proc = AsyncMMDataProcessor(WeirdProc())
        out = await proc.process(input_text_or_ids="x", request_obj=None)
        assert out["path"] == "sync"

    @pytest.mark.asyncio
    async def test_kwargs_and_request_passthrough_async(self, async_processor):
        """Extra kwargs and request_obj should be forwarded on async path."""
        proc = AsyncMMDataProcessor(async_processor)
        out = await proc.process(
            image_data=["i1", "i2"],
            audio_data=["a1"],
            input_text_or_ids="hello world",
            request_obj={"uid": 42},
            return_meta=True,
            delay_s=0.0,
        )
        assert out["images"] == ["i1", "i2"]
        assert out["audios"] == ["a1"]
        assert out["text"] == "hello world"
        assert out["request"] == {"uid": 42}
        assert out["kwargs"]["return_meta"] is True

    @pytest.mark.asyncio
    async def test_kwargs_and_request_passthrough_sync(self, sync_processor):
        """Extra kwargs and request_obj should be forwarded on sync path."""
        proc = AsyncMMDataProcessor(sync_processor)
        out = await proc.process(
            image_data=None,
            audio_data=[],
            input_text_or_ids=[101, 102],
            request_obj=("r", 7),
            lang="en",
        )
        assert out["images"] is None
        assert out["audios"] == []
        assert out["text"] == [101, 102]
        assert out["request"] == ("r", 7)
        assert out["kwargs"]["lang"] == "en"

    def test_shutdown_on_sync_executor(self, sync_processor):
        """Explicit shutdown should close fallback executor for sync path."""
        proc = AsyncMMDataProcessor(sync_processor)
        # Swap real executor for a mock to assert shutdown behavior
        proc.fallback_exec = Mock()
        proc.shutdown()
        proc.fallback_exec.shutdown.assert_called_once_with(wait=False)

    def test_del_calls_shutdown(self, sync_processor, caplog):
        """__del__ should best-effort shutdown without raising."""
        caplog.set_level(logging.DEBUG)
        proc = AsyncMMDataProcessor(sync_processor)
        proc.fallback_exec = Mock()
        # Simulate object destruction
        proc.__del__()
        proc.fallback_exec.shutdown.assert_called_once_with(wait=False)

    @pytest.mark.asyncio
    async def test_concurrent_mixed_requests(self, async_processor):
        """Mix different payloads and ensure all complete with valid outputs."""
        proc = AsyncMMDataProcessor(async_processor, max_concurrent_calls=4)

        tasks = [
            proc.process(input_text_or_ids="t1", request_obj=1),
            proc.process(image_data=["i.png"], input_text_or_ids=[9, 8], request_obj=2),
            proc.process(
                audio_data=["v.wav"], input_text_or_ids="speech", request_obj=3
            ),
            proc.process(
                image_data=[], audio_data=[], input_text_or_ids=None, request_obj=4
            ),
        ]
        outs = await asyncio.gather(*tasks)
        assert len(outs) == 4
        for out in outs:
            assert "path" in out
            assert out["path"] == "async"

    @pytest.mark.asyncio
    async def test_many_requests_values_match_inputs(self, sync_processor):
        """For sync path, ensure each response corresponds to its specific input."""
        proc = AsyncMMDataProcessor(sync_processor, max_concurrent_calls=8)
        texts = [f"msg-{i}" for i in range(10)]
        tasks = [
            proc.process(input_text_or_ids=t, request_obj=i)
            for i, t in enumerate(texts)
        ]
        outs = await asyncio.gather(*tasks)
        got = [o["text"] for o in outs]
        assert got == texts


if __name__ == "__main__":
    pytest.main([__file__])
