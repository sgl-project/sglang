import asyncio
import concurrent.futures
import threading
import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.multimodal_preprocessing_admission import (
    MultimodalPreprocessingAdmission,
    MultimodalPreprocessingBusy,
    MultimodalPreprocessingRequestTooLarge,
    count_preprocessed_multimodal_items,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMultimodalPreprocessingAdmission(CustomTestCase):
    def test_weighted_try_acquire_is_atomic_and_non_blocking(self):
        admission = MultimodalPreprocessingAdmission(max_inflight_items=3)

        first = admission.try_acquire(2)
        self.assertIsNotNone(first)
        self.assertEqual(admission.inflight_items, 2)

        # The rejected reservation must not consume a partial item budget.
        self.assertIsNone(admission.try_acquire(2))
        self.assertEqual(admission.inflight_items, 2)

        second = admission.try_acquire(1)
        self.assertIsNotNone(second)
        self.assertEqual(admission.inflight_items, 3)

        first.release()
        first.release()
        second.release()
        self.assertEqual(admission.inflight_items, 0)

    def test_counts_all_modalities_in_a_batch(self):
        request = SimpleNamespace(
            is_single=False,
            batch_size=2,
            image_data=[["image-0", "image-1"], ["image-2"]],
            video_data=[["video-0"], None],
            audio_data=[None, ["audio-0", "audio-1"]],
        )

        self.assertEqual(count_preprocessed_multimodal_items(request), 6)

    def test_parallel_sampling_fanout_is_not_double_counted(self):
        request = SimpleNamespace(
            is_single=False,
            batch_size=1,
            # Normalization repeats modality arrays for sampling_params.n, but
            # TokenizerManager preprocesses the base input once and reuses it.
            image_data=[["image-0", "image-1"]] * 4,
            video_data=[None] * 4,
            audio_data=[None] * 4,
        )

        self.assertEqual(count_preprocessed_multimodal_items(request), 2)

    def test_real_generate_request_batch_and_parallel_sampling_count_once(self):
        request = GenerateReqInput(
            text=["first", "second"],
            image_data=[["image-0", "image-1"], ["image-2"]],
            video_data=[["video-0"], None],
            audio_data=[["audio-0"], ["audio-1", "audio-2"]],
            sampling_params={"n": 3},
        )
        request.normalize_batch_and_arguments()

        self.assertEqual(request.batch_size, 2)
        self.assertEqual(request.parallel_sample_num, 3)
        self.assertEqual(count_preprocessed_multimodal_items(request), 7)

    def test_rejects_non_positive_reservations_and_capacity(self):
        with self.assertRaises(ValueError):
            MultimodalPreprocessingAdmission(max_inflight_items=0)

        admission = MultimodalPreprocessingAdmission(max_inflight_items=1)
        with self.assertRaises(ValueError):
            admission.try_acquire(0)

    def test_acquire_distinguishes_permanent_limit_from_transient_pressure(self):
        admission = MultimodalPreprocessingAdmission(max_inflight_items=3)
        existing = admission.acquire(2)
        try:
            with self.assertRaises(MultimodalPreprocessingBusy) as busy:
                admission.acquire(2)
            self.assertEqual(busy.exception.inflight_items, 2)
            self.assertEqual(admission.inflight_items, 2)

            with self.assertRaises(MultimodalPreprocessingRequestTooLarge) as large:
                admission.try_acquire(4)
            self.assertEqual(large.exception.max_inflight_items, 3)
            self.assertEqual(admission.inflight_items, 2)
        finally:
            existing.release()

    def test_owner_release_waits_for_tracked_background_future(self):
        async def drive():
            admission = MultimodalPreprocessingAdmission(max_inflight_items=1)
            lease = admission.try_acquire(1)
            future = asyncio.get_running_loop().create_future()
            lease.track_future(future)

            lease.release()
            self.assertEqual(admission.inflight_items, 1)

            future.set_result(None)
            await asyncio.sleep(0)
            self.assertEqual(admission.inflight_items, 0)

        asyncio.run(drive())

    def test_already_completed_concurrent_future_does_not_deadlock(self):
        admission = MultimodalPreprocessingAdmission(max_inflight_items=1)
        lease = admission.try_acquire(1)
        future = concurrent.futures.Future()
        future.set_result(None)
        tracked = threading.Event()

        def track_completed_future():
            lease.track_future(future)
            tracked.set()

        thread = threading.Thread(target=track_completed_future, daemon=True)
        thread.start()
        self.assertTrue(tracked.wait(timeout=1))
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())

        lease.release()
        self.assertEqual(admission.inflight_items, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
