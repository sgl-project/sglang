"""Kimi-K3 video GPU-preprocess frame-weighted admission gate.

``KimiK3ImageProcessor._video_gpu_preprocess_slot`` used to be a plain
``asyncio.Semaphore`` counting concurrent *requests* -- two requests each
individually within ``SGLANG_K3_VIDEO_MAX_SAMPLED_FRAMES`` could still add up
past available GPU memory when they land at the same time, regardless of how
many frames either one carries. It is now a frame-count-weighted gate backed
by an ``asyncio.Condition``: requests block until enough of a shared frame
budget (``SGLANG_K3_VIDEO_MAX_INFLIGHT_FRAMES``) is free, weighted by their
own sampled-frame count rather than counted as one fixed-size slot.

These tests exercise the gate directly against a minimal stand-in object
carrying just the three instance attributes the method reads/writes
(``_video_gpu_frame_budget``, ``_video_gpu_inflight_frames``,
``_video_gpu_condition``) plus the method itself bound onto it -- this is the
real production code path, not a reimplementation, just invoked without
constructing a full ``KimiK3ImageProcessor`` (which needs a tokenizer/HF
processor and is unrelated to this gate's own logic).
"""

import asyncio
import unittest

from sglang.srt.multimodal.processors.kimi_k3 import KimiK3ImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_gate(max_inflight_frames):
    """A bare object carrying only the state _video_gpu_preprocess_slot needs."""
    gate = type("_Gate", (), {})()
    gate._video_gpu_frame_budget = (
        max_inflight_frames if max_inflight_frames > 0 else None
    )
    gate._video_gpu_inflight_frames = 0
    gate._video_gpu_condition = asyncio.Condition()
    gate._video_gpu_preprocess_slot = (
        KimiK3ImageProcessor._video_gpu_preprocess_slot.__get__(gate)
    )
    return gate


class TestKimiK3VideoGpuFrameBudget(unittest.IsolatedAsyncioTestCase):
    async def test_single_request_acquires_and_releases_its_frame_count(self):
        gate = _make_gate(256)

        async with gate._video_gpu_preprocess_slot(100):
            self.assertEqual(gate._video_gpu_inflight_frames, 100)

        self.assertEqual(gate._video_gpu_inflight_frames, 0)

    async def test_two_requests_that_individually_fit_but_together_exceed_serialize(
        self,
    ):
        # The exact incident shape this gate exists for: 112 + 153 sampled
        # frames, each under a 256 budget alone, 265 combined over it.
        gate = _make_gate(256)
        events = []

        async def video_request(name, sampled_frames, hold_seconds):
            async with gate._video_gpu_preprocess_slot(sampled_frames):
                events.append(f"{name}:enter")
                await asyncio.sleep(hold_seconds)
                events.append(f"{name}:exit")

        await asyncio.gather(
            video_request("a", 112, 0.05),
            video_request("b", 153, 0.05),
        )

        # "a" must fully exit before "b" enters -- 112 + 153 > 256, so the
        # gate must not let both be in flight together, even though each
        # individually passes SGLANG_K3_VIDEO_MAX_SAMPLED_FRAMES's 256 cap.
        self.assertEqual(events, ["a:enter", "a:exit", "b:enter", "b:exit"])

    async def test_requests_that_fit_together_run_concurrently(self):
        gate = _make_gate(256)
        events = []

        async def video_request(name, sampled_frames, hold_seconds):
            async with gate._video_gpu_preprocess_slot(sampled_frames):
                events.append(f"{name}:enter")
                await asyncio.sleep(hold_seconds)
                events.append(f"{name}:exit")

        await asyncio.gather(
            video_request("a", 100, 0.05),
            video_request("b", 100, 0.05),
        )

        # 100 + 100 = 200 <= 256: both should be in flight together, not
        # serialized -- "b" enters before "a" exits.
        self.assertEqual(events[:2], ["a:enter", "b:enter"])
        self.assertEqual(set(events[2:]), {"a:exit", "b:exit"})

    async def test_request_bigger_than_the_whole_budget_still_runs_alone(self):
        # A single request whose own sampled-frame count already exceeds the
        # global budget must not deadlock forever waiting for a threshold it
        # can never reach on its own -- it runs once nothing else is in flight.
        gate = _make_gate(256)

        async with gate._video_gpu_preprocess_slot(500):
            self.assertEqual(gate._video_gpu_inflight_frames, 500)

        self.assertEqual(gate._video_gpu_inflight_frames, 0)

    async def test_oversized_request_still_waits_for_other_inflight_requests(self):
        gate = _make_gate(256)
        events = []

        async def small_request():
            async with gate._video_gpu_preprocess_slot(50):
                events.append("small:enter")
                await asyncio.sleep(0.05)
                events.append("small:exit")

        async def oversized_request():
            # Give the small request a chance to acquire first.
            await asyncio.sleep(0.01)
            async with gate._video_gpu_preprocess_slot(500):
                events.append("oversized:enter")

        await asyncio.gather(small_request(), oversized_request())

        self.assertEqual(
            events, ["small:enter", "small:exit", "oversized:enter"]
        )

    async def test_third_request_queues_behind_the_first_two_until_a_slot_frees(self):
        gate = _make_gate(256)
        events = []

        async def video_request(name, sampled_frames, hold_seconds):
            async with gate._video_gpu_preprocess_slot(sampled_frames):
                events.append(f"{name}:enter")
                await asyncio.sleep(hold_seconds)
                events.append(f"{name}:exit")

        # a+b = 200 <= 256, both fit; c (100) needs one of them to free up
        # first (200+100=300 > 256, but 100+100=200 <= 256).
        await asyncio.gather(
            video_request("a", 100, 0.08),
            video_request("b", 100, 0.03),
            video_request("c", 100, 0.02),
        )

        self.assertEqual(events[:2], ["a:enter", "b:enter"])
        self.assertIn("c:enter", events)
        self.assertLess(events.index("b:exit"), events.index("c:enter"))

    async def test_zero_sampled_frames_is_a_no_op_for_image_only_requests(self):
        gate = _make_gate(1)

        async with gate._video_gpu_preprocess_slot(0):
            # Even though the budget is tiny, an image-only request (0
            # sampled frames) must not consume or wait on it.
            self.assertEqual(gate._video_gpu_inflight_frames, 0)

    async def test_disabled_budget_is_a_no_op(self):
        gate = _make_gate(0)
        self.assertIsNone(gate._video_gpu_frame_budget)

        async with gate._video_gpu_preprocess_slot(10_000):
            self.assertEqual(gate._video_gpu_inflight_frames, 0)


if __name__ == "__main__":
    unittest.main()
