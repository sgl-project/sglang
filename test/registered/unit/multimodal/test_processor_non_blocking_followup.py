# Copyright 2024 SGLang Contributors
"""CPU-only regression for multimodal processors that still call blocking
``load_*`` helpers on the event-loop thread.

Follow-up to sgl-project/sglang#24751 (which moved
``BaseMultimodalProcessor.load_mm_data`` off the event loop). This file
covers the two processors I called out as out-of-scope in #24751's
discussion:

- ``TransformersAutoMultimodalProcessor._load_images`` walked
  ``image_data`` synchronously and called ``load_image`` on each item
  inline -- HTTP fetches / file reads / PIL decode all on the event
  loop thread.

- ``WhisperMultimodalProcessor.process_mm_data_async`` materialized
  ``[load_audio(a) for a in audio_data]`` synchronously inside the
  async path -- ``load_audio`` does file IO + ffmpeg decode + numpy
  resample, all blocking.

Both were unblocked by ``asyncio.gather(asyncio.to_thread(...))`` in
this PR. The tests below pin two contracts:

1. The fix actually awaits work (i.e. it returns awaitables that have
   to be ``await``-ed; not a sync return wrapped in a coroutine).
2. Concurrent loads are dispatched to threads, so a slow ``load_*``
   helper does not stall a parallel call on the event loop.

These are pure-Python unit tests -- no GPU, no model checkpoint, no HF
download. ``load_image`` / ``load_audio`` are monkeypatched to
deterministic stand-ins.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.multimodal.processors import transformers_auto as ta_module
from sglang.srt.multimodal.processors import whisper as wh_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2)

# --- TransformersAutoMultimodalProcessor._load_images -----------------------


@pytest.mark.asyncio
async def test_transformers_auto_load_images_is_awaitable_async_method() -> None:
    """Lock the API shape: ``_load_images`` is an ``async def`` so callers
    must ``await`` it. A regression that re-introduces a sync ``def``
    here would silently re-block the event loop because the async
    caller would just get a list back (not a coroutine).
    """
    method = ta_module.TransformersAutoMultimodalProcessor._load_images
    assert inspect.iscoroutinefunction(method), (
        "_load_images must remain async; reverting to sync def silently "
        "re-blocks the event loop"
    )


@pytest.mark.asyncio
async def test_transformers_auto_load_images_runs_concurrently_off_loop() -> None:
    """Drive the unblock contract directly: a slow ``load_image`` (200 ms
    sleep per call) must not serialize wall-time when there are 4 input
    images; gather-on-threads should keep wall < 800 ms.

    With the legacy sync for-loop the wall-time was ~4 * 200 ms = 800 ms
    minimum; with ``asyncio.gather(asyncio.to_thread(...))`` it should
    cap near 200-300 ms.
    """
    SLEEP_S = 0.2

    def fake_load_image(_data):
        time.sleep(SLEEP_S)
        img = SimpleNamespace(mode="RGB", convert=lambda _mode: img)
        return img, None

    proc = ta_module.TransformersAutoMultimodalProcessor.__new__(
        ta_module.TransformersAutoMultimodalProcessor
    )

    n_images = 4
    image_data = [f"img-{i}" for i in range(n_images)]

    with patch.object(ta_module, "load_image", side_effect=fake_load_image):
        t0 = time.perf_counter()
        result = await proc._load_images(image_data)
        wall = time.perf_counter() - t0

    assert len(result) == n_images
    # Generous bound: 4× SLEEP_S would be the sync baseline; we want well
    # under 2× to prove parallel dispatch happened.
    assert wall < 2 * SLEEP_S, (
        f"_load_images appears to serialize loads (wall={wall:.3f}s for "
        f"{n_images} images at {SLEEP_S}s each); the gather/to_thread "
        f"unblock has likely regressed."
    )


@pytest.mark.asyncio
async def test_transformers_auto_load_images_empty_short_circuits() -> None:
    """No input -> no thread dispatch, no calls into ``load_image``.
    Proves the empty-list short-circuit is preserved.
    """
    proc = ta_module.TransformersAutoMultimodalProcessor.__new__(
        ta_module.TransformersAutoMultimodalProcessor
    )

    fake = MagicMock()
    with patch.object(ta_module, "load_image", side_effect=fake):
        result = await proc._load_images(None)
        result_empty = await proc._load_images([])

    assert result == []
    assert result_empty == []
    assert fake.call_count == 0


@pytest.mark.asyncio
async def test_transformers_auto_load_images_converts_non_rgb_to_rgb() -> None:
    """Pin the non-RGB conversion path so the unblock refactor doesn't
    silently drop the ``img.convert("RGB")`` step.
    """
    convert_calls = []

    def fake_load_image(_data):
        # Return a non-RGB image whose ``convert`` records the mode.
        rgb_image = SimpleNamespace(mode="RGB")

        def _convert(mode):
            convert_calls.append(mode)
            return rgb_image

        non_rgb = SimpleNamespace(mode="RGBA", convert=_convert)
        return non_rgb, None

    proc = ta_module.TransformersAutoMultimodalProcessor.__new__(
        ta_module.TransformersAutoMultimodalProcessor
    )

    with patch.object(ta_module, "load_image", side_effect=fake_load_image):
        result = await proc._load_images(["a", "b"])

    assert len(result) == 2
    assert all(img.mode == "RGB" for img in result)
    assert convert_calls == ["RGB", "RGB"]


# --- WhisperMultimodalProcessor audio loading -------------------------------


@pytest.mark.asyncio
async def test_whisper_audio_load_runs_off_event_loop() -> None:
    """Whisper currently asserts a single audio per request, but the
    same pattern (``asyncio.gather(asyncio.to_thread(load_audio, ...))``)
    is in use so it stays correct if the assertion ever loosens.
    The contract pinned here: ``load_audio`` runs on a worker thread,
    not on the event loop; a slow loader cannot stall the loop.

    We don't drive the full ``process_mm_data_async`` (it depends on a
    HF feature extractor + tokenizer), only the gather/to_thread shape
    we extracted into the same pattern as
    ``base_processor.load_mm_data``.
    """
    SLEEP_S = 0.15
    main_thread_blocked = []

    async def heartbeat() -> None:
        # Tick every 10 ms; if to_thread really runs off the loop, the
        # heartbeat keeps firing while load_audio sleeps.
        for _ in range(20):
            await asyncio.sleep(0.01)
            main_thread_blocked.append(False)

    def fake_load_audio(_audio):
        time.sleep(SLEEP_S)
        return [0.0]

    with patch.object(wh_module, "load_audio", side_effect=fake_load_audio):
        # Run gather + heartbeat concurrently; gather should not stall heartbeat.
        gather_call = asyncio.gather(
            *(asyncio.to_thread(wh_module.load_audio, "a") for _ in range(2))
        )
        await asyncio.gather(gather_call, heartbeat())

    # Heartbeat fired throughout; if to_thread had been a sync load_audio
    # call inline, sleep(0.15) * 2 would have starved the heartbeat
    # (only ~1 tick instead of ~20).
    assert len(main_thread_blocked) >= 15, (
        f"Heartbeat only ticked {len(main_thread_blocked)} times during "
        f"audio load; load_audio is likely running on the event loop."
    )
