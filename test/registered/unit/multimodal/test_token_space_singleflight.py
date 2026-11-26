"""CPU contracts for process-local single-flight in TokenSpaceProcessor.

Concurrent misses for one item feature cache key share one load + preprocessing:
  N concurrent requests, same image, cold (N = 1, 4, 16) -> 1 preprocessing, N equal outputs
  one request with the same image twice -> 1 preprocessing, both items filled
  N concurrent requests, different images -> N preprocessings (no over-coalescing)
  [A, B] concurrent with [B, A] -> 2 preprocessings, each output in its own order
  features kept on device -> a joined item lands on its request's device
  the computing request fails -> every joined request gets its error; the key is not stuck
  one item of the computing request fails -> a request joined on another item still succeeds
  loading fails for the computing request -> its claims are released; joined requests retry
  the computation is cancelled before it settles -> its claims are released, nobody hangs
  a sampled cache check fails -> nothing is written or handed to a joined request
  the computing request is cancelled -> joined requests still get the features, cache written
  a joined request is cancelled -> the computing request is unaffected
  computed but not yet persisted -> a later request joins instead of computing again
  flush while computing -> later requests compute again; the in-memory cache drops the
    old result (also when the flush lands inside the write); a disk cache, which a flush
    does not clear, still takes it
  flush between reading the cache generation and claiming -> correct output, no stale entry
  no item feature cache -> unchanged: every request computes

Concurrency is forced with threading Events and a Condition, never with sleeps:
preprocessing blocks on a gate until every request has either entered
preprocessing itself or joined another request's computation.
"""

import asyncio
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from test_mm_token_expansion import (  # noqa: E402
    _CHECKPOINT,
    IMAGE,
    _bare_tokenizer,
    _ImageWidthStrategy,
)

from sglang.srt.multimodal.cache import (  # noqa: E402
    DiskPreprocessCache,
    MultimodalPreprocessCache,
)
from sglang.srt.multimodal.processors.processor_config import (  # noqa: E402
    MultimodalProcessorConfig,
)
from sglang.srt.multimodal.token_space_mm_process.processor import (  # noqa: E402
    TokenSpaceProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Bounds a wait for a broken implementation; a passing run never waits on it.
_DEADLOCK_TIMEOUT_S = 10


class _Progress:
    """Counts requests that reached preprocessing or joined, across threads."""

    def __init__(self):
        self._condition = threading.Condition()
        self.preprocessed = 0
        self.joined = 0

    def add(self, name):
        with self._condition:
            setattr(self, name, getattr(self, name) + 1)
            self._condition.notify_all()

    async def wait_for(self, predicate):
        def wait():
            with self._condition:
                return self._condition.wait_for(
                    lambda: predicate(self), timeout=_DEADLOCK_TIMEOUT_S
                )

        if not await asyncio.to_thread(wait):
            raise AssertionError(
                f"timed out: preprocessed={self.preprocessed} joined={self.joined}"
            )


class _GatedStrategy(_ImageWidthStrategy):
    """Counts preprocessing and holds it until the test opens the gate."""

    progress = None
    gate = None
    fail_first = False
    # Preprocessing an image of this width fails, every time.
    fail_width = None

    def process_image(self, image, processor, preprocess_kwargs, **kwargs):
        cls = type(self)
        cls.progress.add("preprocessed")
        if not cls.gate.wait(timeout=_DEADLOCK_TIMEOUT_S):
            raise AssertionError("gate never opened")
        if cls.fail_first:
            cls.fail_first = False
            raise ValueError("preprocessing failed")
        if image.width == cls.fail_width:
            raise ValueError(f"preprocessing width {image.width} failed")
        features = super().process_image(image, processor, preprocess_kwargs, **kwargs)
        # Like a real strategy, preprocess on the request's device.
        features["pixel_values"] = features["pixel_values"].to(kwargs.get("device"))
        return features


class _BlockingWrites(dict):
    """A map whose writes wait for the test, like a slow disk."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.written = threading.Event()

    def __setitem__(self, key, value):
        self.release.wait(timeout=_DEADLOCK_TIMEOUT_S)
        super().__setitem__(key, value)
        self.written.set()


class TestTokenSpaceSingleFlight(unittest.TestCase):
    def setUp(self):
        self._reset_strategy()

    def _reset_strategy(self):
        """Fresh counters and a closed gate, e.g. per subtest or after seeding a cache."""
        self.progress = _Progress()
        _GatedStrategy.progress = self.progress
        _GatedStrategy.gate = threading.Event()
        _GatedStrategy.fail_first = False
        _GatedStrategy.fail_width = None
        # Open the gate on teardown too, so a failing test never leaves a worker blocked.
        self.addCleanup(_GatedStrategy.gate.set)

    def _processor(
        self,
        item_feature_cache,
        keep_mm_features_on_device=False,
        item_feature_cache_check_rate=None,
    ):
        processor = TokenSpaceProcessor.for_strategy(_GatedStrategy)(
            _CHECKPOINT,
            _bare_tokenizer(),
            processor_config=MultimodalProcessorConfig(
                mm_process_config={},
                mm_processor_worker_num=16,
                keep_mm_features_on_device=keep_mm_features_on_device,
            ),
            item_feature_cache=item_feature_cache,
            item_feature_cache_check_rate=item_feature_cache_check_rate,
        )
        self.addCleanup(processor.shutdown)
        # Every test ends with nothing in flight: no claim outlives its request and write.
        self.addCleanup(
            lambda: self.assertEqual(
                getattr(processor, "_item_feature_flights", {}), {}
            )
        )
        # Observe joins through the processor's one waiting seam; absent before the fix.
        real_wait = getattr(processor, "_wait_for_item_feature_flight", None)

        async def counted_wait(future):
            self.progress.add("joined")
            return await real_wait(future)

        if real_wait is not None:
            patcher = patch.object(
                processor, "_wait_for_item_feature_flight", counted_wait
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        return processor

    async def _run_gated(self, requests, waiting):
        """Start every request, open the gate once `waiting` have arrived, gather."""
        tasks = [asyncio.create_task(request) for request in requests]
        await self.progress.wait_for(lambda p: p.preprocessed + p.joined >= waiting)
        _GatedStrategy.gate.set()
        return await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _expected(images, input_ids):
        _GatedStrategy.gate.set()
        reference = TokenSpaceProcessor.for_strategy(_ImageWidthStrategy)(
            _CHECKPOINT,
            _bare_tokenizer(),
            processor_config=MultimodalProcessorConfig(mm_process_config={}),
        )
        try:
            return asyncio.run(reference.process(images, input_ids=input_ids))
        finally:
            reference.shutdown()

    def _assert_same_output(self, output, expected):
        self.assertNotIsInstance(output, BaseException, output)
        for name in ("input_ids", "pixel_values"):
            self.assertTrue(torch.equal(output[name], expected[name]), name)
        self.assertEqual(output["mm_spans"], expected["mm_spans"])

    def test_concurrent_same_key_misses_preprocess_once(self):
        for n in (1, 4, 16):
            with self.subTest(n=n):
                self._reset_strategy()
                image = Image.new("RGB", (3, 1))
                processor = self._processor({})
                outputs = asyncio.run(
                    self._run_gated(
                        [
                            processor.process([image], input_ids=[1, IMAGE])
                            for _ in range(n)
                        ],
                        waiting=n,
                    )
                )
                self.assertEqual(self.progress.preprocessed, 1)
                expected = self._expected([image], [1, IMAGE])
                for output in outputs:
                    self._assert_same_output(output, expected)

    def test_same_image_twice_in_one_request_preprocesses_once(self):
        image = Image.new("RGB", (2, 1))
        _GatedStrategy.gate.set()
        output = asyncio.run(
            self._processor({}).process([image, image], input_ids=[IMAGE, 2, IMAGE])
        )
        self.assertEqual(self.progress.preprocessed, 1)
        self._assert_same_output(
            output, self._expected([image, image], [IMAGE, 2, IMAGE])
        )

    def test_different_keys_are_not_coalesced(self):
        images = [Image.new("RGB", (width, 1)) for width in range(1, 5)]
        processor = self._processor({})
        outputs = asyncio.run(
            self._run_gated(
                [processor.process([image], input_ids=[IMAGE]) for image in images],
                waiting=len(images),
            )
        )
        self.assertEqual(self.progress.preprocessed, len(images))
        for image, output in zip(images, outputs):
            self._assert_same_output(output, self._expected([image], [IMAGE]))

    def test_crossed_requests_each_keep_their_order(self):
        a, b = Image.new("RGB", (2, 1)), Image.new("RGB", (5, 1))
        processor = self._processor({})
        ab, ba = asyncio.run(
            self._run_gated(
                [
                    processor.process([a, b], input_ids=[IMAGE, 1, IMAGE]),
                    processor.process([b, a], input_ids=[IMAGE, 1, IMAGE]),
                ],
                # Each request computes one image and joins the other.
                waiting=4,
            )
        )
        self.assertEqual(self.progress.preprocessed, 2)
        self._assert_same_output(ab, self._expected([a, b], [IMAGE, 1, IMAGE]))
        self._assert_same_output(ba, self._expected([b, a], [IMAGE, 1, IMAGE]))

    def test_joined_item_lands_on_its_request_device(self):
        image = Image.new("RGB", (2, 1))
        processor = self._processor({}, keep_mm_features_on_device=True)
        outputs = asyncio.run(
            self._run_gated(
                [
                    processor.process([image], input_ids=[IMAGE], image_device="meta")
                    for _ in range(4)
                ],
                waiting=4,
            )
        )
        self.assertEqual(self.progress.preprocessed, 1)
        for output in outputs:
            self.assertEqual(output["pixel_values"].device.type, "meta")

    def test_failure_reaches_every_joined_request_and_releases_the_key(self):
        image = Image.new("RGB", (2, 1))
        _GatedStrategy.fail_first = True
        processor = self._processor({})
        outputs = asyncio.run(
            self._run_gated(
                [processor.process([image], input_ids=[IMAGE]) for _ in range(4)],
                waiting=4,
            )
        )
        self.assertEqual(self.progress.preprocessed, 1)
        for output in outputs:
            self.assertIsInstance(output, ValueError)
        # Not stuck on the failed computation: the next request computes again.
        output = asyncio.run(processor.process([image], input_ids=[IMAGE]))
        self.assertEqual(self.progress.preprocessed, 2)
        self._assert_same_output(output, self._expected([image], [IMAGE]))

    def test_cancelling_the_computing_request_keeps_the_shared_work(self):
        image = Image.new("RGB", (2, 1))
        cache = {}
        processor = self._processor(cache)

        async def scenario():
            owner = asyncio.create_task(processor.process([image], input_ids=[IMAGE]))
            await self.progress.wait_for(lambda p: p.preprocessed == 1)
            joiners = [
                asyncio.create_task(processor.process([image], input_ids=[IMAGE]))
                for _ in range(3)
            ]
            await self.progress.wait_for(lambda p: p.preprocessed + p.joined >= 4)
            owner.cancel()
            _GatedStrategy.gate.set()
            with self.assertRaises(asyncio.CancelledError):
                await owner
            return await asyncio.gather(*joiners, return_exceptions=True)

        outputs = asyncio.run(scenario())
        self.assertEqual(self.progress.preprocessed, 1)
        for output in outputs:
            self._assert_same_output(output, self._expected([image], [IMAGE]))
        self.assertEqual(len(cache), 1)

    def test_cancelling_a_joined_request_leaves_the_computation(self):
        image = Image.new("RGB", (2, 1))
        processor = self._processor({})

        async def scenario():
            owner = asyncio.create_task(processor.process([image], input_ids=[IMAGE]))
            await self.progress.wait_for(lambda p: p.preprocessed == 1)
            joiner = asyncio.create_task(processor.process([image], input_ids=[IMAGE]))
            await self.progress.wait_for(lambda p: p.preprocessed + p.joined >= 2)
            joiner.cancel()
            _GatedStrategy.gate.set()
            with self.assertRaises(asyncio.CancelledError):
                await joiner
            return await owner

        output = asyncio.run(scenario())
        self.assertEqual(self.progress.preprocessed, 1)
        self._assert_same_output(output, self._expected([image], [IMAGE]))

    def test_request_between_completion_and_persistence_joins(self):
        """The key stays in flight until its write lands, so a slow disk cannot reopen it."""
        image = Image.new("RGB", (2, 1))
        cache = _BlockingWrites()
        self.addCleanup(cache.release.set)
        processor = self._processor(cache)
        _GatedStrategy.gate.set()

        async def scenario():
            first = await processor.process([image], input_ids=[IMAGE])
            self.assertFalse(cache.written.is_set())
            second = await processor.process([image], input_ids=[IMAGE])
            cache.release.set()
            return first, second

        first, second = asyncio.run(scenario())
        self.assertTrue(cache.written.wait(_DEADLOCK_TIMEOUT_S))
        self.assertEqual(self.progress.preprocessed, 1)
        self._assert_same_output(second, first)

    def _flush(self, processor, cache):
        """What TokenSpaceSGLangProcessor.clear_preprocess_cache does."""
        cache.clear()
        processor.reset_item_feature_flights()

    def test_flush_starts_a_new_generation(self):
        image = Image.new("RGB", (2, 1))
        accepted = []

        class _CountingPuts(MultimodalPreprocessCache):
            def put(self, *args, **kwargs):
                accepted.append(super().put(*args, **kwargs))
                return accepted[-1]

        cache = _CountingPuts(64 << 20)
        processor = self._processor(cache)

        async def scenario():
            before = asyncio.create_task(processor.process([image], input_ids=[IMAGE]))
            await self.progress.wait_for(lambda p: p.preprocessed == 1)
            self._flush(processor, cache)
            after = asyncio.create_task(processor.process([image], input_ids=[IMAGE]))
            # The request after the flush computes for itself instead of joining.
            await self.progress.wait_for(lambda p: p.preprocessed == 2)
            _GatedStrategy.gate.set()
            return await asyncio.gather(before, after)

        outputs = asyncio.run(scenario())
        self.assertEqual(self.progress.joined, 0)
        for output in outputs:
            self._assert_same_output(output, self._expected([image], [IMAGE]))
        # asyncio.run drained the write-behind executor: only the post-flush result landed.
        self.assertEqual(accepted.count(True), 1)
        self.assertEqual(len(cache), 1)

    def test_flush_between_write_and_insert_leaves_no_stale_entry(self):
        """A flush landing after the write is decided but before the insert must win."""
        image = Image.new("RGB", (2, 1))
        processor = None

        class _FlushDuringInsert(MultimodalPreprocessCache):
            flush_pending = True

            def put(inner, *args, **kwargs):
                if inner.flush_pending:
                    # Deterministic interleaving: the flush runs inside the write itself,
                    # after any check the processor made before calling into the cache.
                    inner.flush_pending = False
                    self._flush(processor, inner)
                return super().put(*args, **kwargs)

        cache = _FlushDuringInsert(64 << 20)
        processor = self._processor(cache)
        _GatedStrategy.gate.set()
        output = asyncio.run(processor.process([image], input_ids=[IMAGE]))
        self._assert_same_output(output, self._expected([image], [IMAGE]))
        self.assertFalse(cache.flush_pending)
        self.assertEqual(len(cache), 0)

    def test_flush_between_reading_the_cache_generation_and_claiming(self):
        """The claim reads the cache generation before taking the flights lock."""
        image = Image.new("RGB", (2, 1))
        processor = None
        reads = []

        class _FlushAfterGenerationRead(MultimodalPreprocessCache):
            @property
            def generation(inner):
                value = MultimodalPreprocessCache.generation.fget(inner)
                reads.append(value)
                if len(reads) == 1:
                    # Deterministic interleaving: the flush lands after the read and
                    # before the claim, so the claim is post-flush but its write is not.
                    self._flush(processor, inner)
                return value

        cache = _FlushAfterGenerationRead(64 << 20)
        processor = self._processor(cache)
        _GatedStrategy.gate.set()
        first = asyncio.run(processor.process([image], input_ids=[IMAGE]))
        self._assert_same_output(first, self._expected([image], [IMAGE]))
        # Safe: the cache drops the write, so nothing outlives the flush. The cost is
        # one lost write, which the next request recomputes and persists.
        self.assertEqual(len(cache), 0)
        second = asyncio.run(processor.process([image], input_ids=[IMAGE]))
        self._assert_same_output(second, first)
        self.assertEqual(self.progress.preprocessed, 2)
        self.assertEqual(len(cache), 1)

    def test_flush_still_writes_the_disk_cache(self):
        """A flush clears memory only: disk entries stay valid, so in-flight work persists."""
        image = Image.new("RGB", (2, 1))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        processor = self._processor(DiskPreprocessCache(directory.name))

        async def scenario():
            task = asyncio.create_task(processor.process([image], input_ids=[IMAGE]))
            await self.progress.wait_for(lambda p: p.preprocessed == 1)
            processor.reset_item_feature_flights()
            _GatedStrategy.gate.set()
            return await task

        asyncio.run(scenario())
        # Another process sharing the directory hits the pre-flush result.
        other = self._processor(DiskPreprocessCache(directory.name))
        output = asyncio.run(other.process([image], input_ids=[IMAGE]))
        self.assertEqual(self.progress.preprocessed, 1)
        self._assert_same_output(output, self._expected([image], [IMAGE]))

    def test_failing_item_does_not_fail_a_request_joined_on_another(self):
        """Owner [A, B] fails on B; a request joined on A gets A; the owner still fails."""
        a, b = Image.new("RGB", (2, 1)), Image.new("RGB", (5, 1))
        _GatedStrategy.fail_width = 5
        cache = {}
        processor = self._processor(cache)

        async def scenario():
            owner = asyncio.create_task(
                processor.process([a, b], input_ids=[IMAGE, IMAGE])
            )
            await self.progress.wait_for(lambda p: p.preprocessed == 2)
            joiner = asyncio.create_task(processor.process([a], input_ids=[IMAGE]))
            await self.progress.wait_for(lambda p: p.joined == 1)
            _GatedStrategy.gate.set()
            return await asyncio.gather(owner, joiner, return_exceptions=True)

        owner, joiner = asyncio.run(scenario())
        self.assertIsInstance(owner, ValueError)
        self.assertIn("width 5", str(owner))
        self._assert_same_output(joiner, self._expected([a], [IMAGE]))
        self.assertEqual(self.progress.preprocessed, 2)
        # Intended change: the failing request still caches A, which succeeded.
        self.assertEqual(len(cache), 1)

    def test_failed_load_does_not_fail_a_request_joined_on_another_item(self):
        """Loading fails for a whole request, so its claims are released and joiners retry."""
        a = Image.new("RGB", (2, 1))
        bad = "data:image/png;base64,bm90IGFuIGltYWdl"
        _GatedStrategy.gate.set()
        processor = self._processor({})
        real_load = processor._load_media_lists
        owner_loading = None

        async def load(image_data, *args, **kwargs):
            if image_data and bad in image_data:
                # The owner has claimed A by now; fail only once a request joined it.
                owner_loading.set()
                await self.progress.wait_for(lambda p: p.joined == 1)
            return await real_load(image_data, *args, **kwargs)

        async def scenario():
            nonlocal owner_loading
            owner_loading = asyncio.Event()
            owner = asyncio.create_task(
                processor.process([a, bad], input_ids=[IMAGE, IMAGE])
            )
            await owner_loading.wait()
            joiner = asyncio.create_task(processor.process([a], input_ids=[IMAGE]))
            return await asyncio.gather(owner, joiner, return_exceptions=True)

        with patch.object(processor, "_load_media_lists", load):
            owner, joiner = asyncio.run(scenario())
        self.assertIsInstance(owner, (ValueError, RuntimeError))
        self._assert_same_output(joiner, self._expected([a], [IMAGE]))
        # The owner never processed A; the joiner computed it after the release.
        self.assertEqual(self.progress.preprocessed, 1)

    def test_a_computation_cancelled_before_settling_releases_its_claims(self):
        """E.g. a loop torn down mid-flight: joined requests must not wait forever."""
        image = Image.new("RGB", (2, 1))
        _GatedStrategy.gate.set()
        processor = self._processor({})
        real_compute = processor._compute_and_publish
        calls, started = [], None

        async def compute_once_hanging(*args, **kwargs):
            calls.append(None)
            if len(calls) == 1:
                # Never settles its claims itself, like a task cancelled before its
                # first step, whose own error handling never runs.
                started.set()
                await asyncio.Event().wait()
            return await real_compute(*args, **kwargs)

        async def scenario():
            nonlocal started
            started = asyncio.Event()
            owner = asyncio.create_task(processor.process([image], input_ids=[IMAGE]))
            await started.wait()
            joiner = asyncio.create_task(processor.process([image], input_ids=[IMAGE]))
            await self.progress.wait_for(lambda p: p.joined == 1)
            for task in processor._item_feature_tasks:
                task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await owner
            return await asyncio.wait_for(joiner, _DEADLOCK_TIMEOUT_S)

        with patch.object(processor, "_compute_and_publish", compute_once_hanging):
            output = asyncio.run(scenario())
        self._assert_same_output(output, self._expected([image], [IMAGE]))

    def _cache_with_corrupt_hit(self, image):
        """A map holding `image` with features that differ from computing it now."""
        cache = {}
        _GatedStrategy.gate.set()
        asyncio.run(self._processor(cache).process([image], input_ids=[IMAGE]))
        ((key, cached),) = cache.items()
        cached["pixel_values"] = cached["pixel_values"] + 1
        # Fresh gate and counters for the request under test.
        self._reset_strategy()
        return cache

    def test_failed_sampled_check_writes_no_other_item(self):
        a, b = Image.new("RGB", (2, 1)), Image.new("RGB", (5, 1))
        cache = self._cache_with_corrupt_hit(a)
        processor = self._processor(cache, item_feature_cache_check_rate=1)
        _GatedStrategy.gate.set()
        with self.assertRaisesRegex(RuntimeError, "other pixel_values"):
            asyncio.run(processor.process([a, b], input_ids=[IMAGE, IMAGE]))
        # Drained by asyncio.run: B, computed by the failed request, was never written.
        self.assertEqual(len(cache), 1)

    def test_failed_sampled_check_hands_no_item_to_a_joined_request(self):
        a, b = Image.new("RGB", (2, 1)), Image.new("RGB", (5, 1))
        cache = self._cache_with_corrupt_hit(a)
        processor = self._processor(cache, item_feature_cache_check_rate=1)

        async def scenario():
            owner = asyncio.create_task(
                processor.process([a, b], input_ids=[IMAGE, IMAGE])
            )
            # A is computed again to be checked, B because it missed.
            await self.progress.wait_for(lambda p: p.preprocessed == 2)
            joiner = asyncio.create_task(processor.process([b], input_ids=[IMAGE]))
            await self.progress.wait_for(lambda p: p.joined == 1)
            _GatedStrategy.gate.set()
            return await asyncio.gather(owner, joiner, return_exceptions=True)

        owner, joiner = asyncio.run(scenario())
        self.assertIsInstance(owner, RuntimeError)
        self._assert_same_output(joiner, self._expected([b], [IMAGE]))
        # The joiner computed B for itself after the owner released it.
        self.assertEqual(self.progress.preprocessed, 3)

    def test_without_a_cache_every_request_computes(self):
        image = Image.new("RGB", (2, 1))
        processor = self._processor(None)
        outputs = asyncio.run(
            self._run_gated(
                [processor.process([image], input_ids=[IMAGE]) for _ in range(4)],
                waiting=4,
            )
        )
        self.assertEqual(self.progress.preprocessed, 4)
        for output in outputs:
            self._assert_same_output(output, self._expected([image], [IMAGE]))


if __name__ == "__main__":
    unittest.main()
