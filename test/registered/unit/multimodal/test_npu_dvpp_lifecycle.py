"""DVPP resource ownership, failure recovery and Tensor delivery on CPU CI."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from _dvpp_test_utils import encoded_jpeg, load_dvpp_modules, simulated_npu

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNpuDvppLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header, cls.dvpp, cls.jpeg, cls.cache_module = load_dvpp_modules(cls)

    def assert_released(self, runtime):
        self.assertFalse(runtime.allocations)
        self.assertFalse(runtime.descriptors)
        self.assertFalse(runtime.channels)
        self.assertFalse(runtime.channel_descriptors)
        self.assertFalse(runtime.streams)

    @contextmanager
    def cached_decoders(self):
        with patch.object(
            self.cache_module, "cached_decoders", self.cache_module._DvppCache()
        ):
            yield

    def test_lazy_construction_and_close_without_processing(self):
        with simulated_npu() as runtime:
            decoder = self.dvpp.Dvpp()
            self.assertEqual(decoder.device_index, 3)
            self.assertFalse(runtime.events)
            decoder.close()
            decoder.close()
            self.assertFalse(runtime.events)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                decoder.process_jpeg_dec(encoded_jpeg())

    def test_direct_fallback_does_not_create_resources(self):
        for arch, data in (
            (2201, encoded_jpeg(progressive=True)),
            (3510, encoded_jpeg()),
        ):
            with self.subTest(arch=arch), simulated_npu(arch=arch) as runtime:
                decoder = self.dvpp.Dvpp()
                self.assertIsNone(decoder.process_jpeg_dec(data))
                self.assertIsNone(decoder.process_jpeg_dec(data))
                decoder.close()
                self.assertEqual(runtime.calls["create_stream"], 0)
                self.assertEqual(runtime.calls["malloc"], 0)
                self.assertEqual(runtime.calls["architecture"], int(arch == 3510))

    def test_context_reuses_file_and_memory_inputs_and_closes(self):
        with TemporaryDirectory() as directory, simulated_npu() as runtime:
            path = Path(directory) / "image.jpg"
            path.write_bytes(encoded_jpeg())
            with self.dvpp.Dvpp() as decoder:
                self.assertFalse(runtime.events)
                first = decoder.process_jpeg_dec(image_path=path)
                second = decoder.process_jpeg_dec(encoded_image=path.read_bytes())
                self.assertEqual(runtime.calls["create_stream"], 1)
                self.assertEqual(runtime.calls["create_channel"], 2)
                self.assertFalse(runtime.allocations)
            self.assert_released(runtime)
        for frame, tensor in enumerate((first, second), start=1):
            self.assertEqual(tensor.shape, (3, 49, 65))
            self.assertEqual(tensor.dtype, torch.uint8)
            self.assertTrue(tensor.is_contiguous())
            self.assertTrue(torch.all(tensor[0] == frame))
            self.assertTrue(torch.all(tensor[1] == torch.arange(49)[:, None]))
            self.assertTrue(torch.all(tensor[2] == torch.arange(65)))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            with decoder:
                self.fail("A closed decoder must not enter a new session")

    def test_decode_requires_exactly_one_input_before_io_or_allocation(self):
        with simulated_npu() as runtime, self.dvpp.Dvpp() as decoder:
            with patch.object(Path, "read_bytes") as read_bytes:
                for inputs in (
                    {},
                    {"encoded_image": None},
                    {"image_path": None},
                    {"encoded_image": encoded_jpeg(), "image_path": "image.jpg"},
                    {"encoded_image": b"", "image_path": "image.jpg"},
                    {
                        "encoded_image": torch.tensor([1, 2], dtype=torch.uint8),
                        "image_path": "image.jpg",
                    },
                ):
                    with self.subTest(inputs=inputs.keys()):
                        with self.assertRaisesRegex(ValueError, "exactly one"):
                            decoder.process_jpeg_dec(**inputs)
                read_bytes.assert_not_called()
            self.assertFalse(runtime.events)

    def test_decode_does_not_infer_paths_from_encoded_data(self):
        with simulated_npu() as runtime, self.dvpp.Dvpp() as decoder:
            with patch.object(Path, "read_bytes") as read_bytes:
                with self.assertRaises(TypeError):
                    decoder.process_jpeg_dec("image.jpg")
                with self.assertRaises(TypeError):
                    decoder.process_jpeg_dec(encoded_image=Path("image.jpg"))
                with self.assertRaises(TypeError):
                    decoder.process_jpeg_dec(image_path=encoded_jpeg())
                read_bytes.assert_not_called()
            self.assertFalse(runtime.events)

    def test_file_fallback_and_io_error_do_not_allocate(self):
        with TemporaryDirectory() as directory, simulated_npu() as runtime:
            path = Path(directory) / "unsupported.jpg"
            path.write_bytes(encoded_jpeg(progressive=True))
            with self.dvpp.Dvpp() as decoder:
                self.assertIsNone(decoder.process_jpeg_dec(image_path=path))
            self.assertTrue(decoder.closed)
            with self.assertRaises(FileNotFoundError):
                with self.dvpp.Dvpp() as missing_decoder:
                    missing_decoder.process_jpeg_dec(
                        image_path=path.with_name("missing.jpg")
                    )
            self.assertTrue(missing_decoder.closed)
            self.assertFalse(runtime.events)
            self.assert_released(runtime)

    def test_context_closes_on_body_and_decode_errors(self):
        with simulated_npu() as runtime:
            original = ValueError("consumer failed")
            with self.assertRaises(ValueError) as caught:
                with self.dvpp.Dvpp() as decoder:
                    decoder.process_jpeg_dec(encoded_jpeg())
                    raise original
            self.assertIs(caught.exception, original)
            self.assertTrue(decoder.closed)
            self.assert_released(runtime)
        with simulated_npu((("submit_vpc", 1),)) as runtime:
            with self.assertRaises(self.dvpp.ACLFailure):
                with self.dvpp.Dvpp() as decoder:
                    decoder.process_jpeg_dec(encoded_jpeg())
            self.assertTrue(decoder.closed)
            self.assert_released(runtime)

    def test_context_reports_cleanup_failure_without_hiding_body_error(self):
        for original in (None, ValueError("consumer failed"), KeyboardInterrupt()):
            with (
                self.subTest(original=original),
                simulated_npu((("destroy_stream", 1),)) as runtime,
            ):
                expected = self.dvpp.ACLFailure if original is None else type(original)
                with self.assertRaises(expected) as caught:
                    with self.dvpp.Dvpp() as decoder:
                        decoder.process_jpeg_dec(encoded_jpeg())
                        if original is not None:
                            raise original
                if original is not None:
                    self.assertIs(caught.exception, original)
                    self.assertTrue(
                        any("destroy_stream" in note for note in original.__notes__)
                    )
                self.assertTrue(decoder.closed)
                self.assertIsNotNone(decoder.stream)
                decoder.close()
                self.assert_released(runtime)

    def test_initialization_failures_release_owned_resources(self):
        failures = (
            ("create_stream", 1),
            ("create_channel_desc", 1),
            ("create_channel", 1),
            ("create_channel_desc", 2),
            ("channel_mode", 1),
            ("channel_param", 1),
            ("create_channel", 2),
        )
        for failure in failures:
            with self.subTest(failure=failure), simulated_npu((failure,)) as runtime:
                decoder = self.dvpp.Dvpp()
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.process_jpeg_dec(encoded_jpeg())
                self.assertTrue(decoder.closed)
                self.assert_released(runtime)
                self.assertEqual((runtime.current, runtime.context), (3, 777))

    def test_vpc_rejects_invalid_images_and_formats_before_allocation(self):
        with simulated_npu() as runtime:
            decoder, other = self.dvpp.Dvpp(), self.dvpp.Dvpp()
            data, info = self.header.prepare_jpeg(encoded_jpeg())
            with decoder.lock, decoder._use_context():
                decoder._ensure_resources()
                nv12 = decoder._decode_jpeg_to_nv12(data, info=info)
                with self.assertRaisesRegex(ValueError, "RGB888"):
                    nv12.to_tensor()
                rgb = decoder._convert_nv12_to_rgb(nv12)
                event_count = len(runtime.events)
                with self.assertRaisesRegex(ValueError, "same Dvpp"):
                    other._convert_nv12_to_rgb(nv12)
                with self.assertRaisesRegex(ValueError, "same Dvpp"):
                    decoder._convert_nv12_to_rgb(object())
                with self.assertRaisesRegex(ValueError, "NV12 to RGB888"):
                    decoder._convert_nv12_to_rgb(rgb)
                self.assertEqual(len(runtime.events), event_count)
                nv12.close()
                self.assertIn(rgb.pointer, runtime.allocations)
                event_count = len(runtime.events)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    decoder._convert_nv12_to_rgb(nv12)
                self.assertEqual(len(runtime.events), event_count)
            # The owner also reclaims images not explicitly closed by the caller.
            decoder.close()
            other.close()
            self.assertIsNone(rgb.pointer)
            self.assert_released(runtime)

    def test_bound_device_restores_caller_state_on_success_and_failure(self):
        with simulated_npu((("copy_d2d", 2),)) as runtime:
            decoder = self.dvpp.Dvpp()
            runtime.current, runtime.context = 1, 888
            result = decoder.process_jpeg_dec(encoded_jpeg())
            self.assertTrue(torch.all(result[0] == 1))
            self.assertIn(("architecture", 3), runtime.events)
            self.assertEqual((runtime.current, runtime.context), (1, 888))
            with self.assertRaisesRegex(self.dvpp.ACLFailure, "D2D"):
                decoder.process_jpeg_dec(encoded_jpeg())
            self.assertEqual((runtime.current, runtime.context), (1, 888))
            decoder.close()
            self.assert_released(runtime)
            self.assertEqual((runtime.current, runtime.context), (1, 888))

    def test_failed_submissions_are_drained_before_free(self):
        for operation in ("submit_jpeg", "submit_vpc"):
            with (
                self.subTest(operation=operation),
                simulated_npu(((operation, 1),)) as runtime,
            ):
                decoder = self.dvpp.Dvpp()
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.process_jpeg_dec(encoded_jpeg())
                self.assertFalse(decoder.poisoned)
                self.assertIsNone(runtime.inflight)
                self.assertFalse(runtime.allocations)
                self.assertFalse(runtime.descriptors)
                decoder.close()
                self.assert_released(runtime)

    def test_synchronization_failure_keeps_inflight_memory(self):
        for number in (1, 2):
            with (
                self.subTest(stage=number),
                simulated_npu((("synchronize", number),)) as runtime,
            ):
                decoder = self.dvpp.Dvpp()
                with self.assertRaisesRegex(self.dvpp.ACLFailure, "synchronize_stream"):
                    decoder.process_jpeg_dec(encoded_jpeg())
                self.assertTrue(decoder.poisoned)
                self.assertIsNotNone(runtime.inflight)
                for pointer in runtime.inflight[1:]:
                    self.assertIn(pointer, runtime.allocations)
                event_count = len(runtime.events)
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.process_jpeg_dec(encoded_jpeg())
                with self.assertRaisesRegex(self.dvpp.ACLFailure, "poisoned"):
                    decoder.close()
                for image in decoder._images:
                    with self.assertRaisesRegex(self.dvpp.ACLFailure, "poisoned"):
                        image.close()
                    with self.assertRaisesRegex(self.dvpp.ACLFailure, "poisoned"):
                        image.to_tensor()
                    with self.assertRaisesRegex(self.dvpp.ACLFailure, "poisoned"):
                        decoder._convert_nv12_to_rgb(image)
                self.assertEqual(len(runtime.events), event_count)
                self.assertEqual((runtime.current, runtime.context), (3, 777))

    def test_failed_drain_preserves_original_submission_error(self):
        for operation, sync_number in (("submit_jpeg", 1), ("submit_vpc", 2)):
            failures = ((operation, 1), ("synchronize", sync_number))
            with self.subTest(operation=operation), simulated_npu(failures) as runtime:
                decoder = self.dvpp.Dvpp()
                with self.assertRaises(self.dvpp.ACLFailure) as raised:
                    decoder.process_jpeg_dec(encoded_jpeg())
                self.assertNotIn("synchronize_stream", str(raised.exception))
                self.assertTrue(
                    any(
                        "synchronization failed" in note
                        for note in raised.exception.__notes__
                    )
                )
                self.assertTrue(decoder.poisoned)
                for pointer in runtime.inflight[1:]:
                    self.assertIn(pointer, runtime.allocations)

    def test_execution_failures_release_per_image_buffers(self):
        failures = (
            ("malloc", 1),
            ("malloc", 2),
            ("malloc", 3),
            ("copy_h2d", 1),
            ("copy_d2d", 1),
            ("completion", 1),
            ("completion", 2),
            ("create_pic_desc", 1),
            ("create_pic_desc", 2),
            ("create_pic_desc", 3),
        )
        for failure in failures:
            with self.subTest(failure=failure), simulated_npu((failure,)) as runtime:
                decoder = self.dvpp.Dvpp()
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.process_jpeg_dec(encoded_jpeg())
                self.assertFalse(runtime.allocations)
                self.assertFalse(runtime.descriptors)
                self.assertFalse(decoder.poisoned)
                self.assertEqual((runtime.current, runtime.context), (3, 777))
                decoder.close()
                self.assert_released(runtime)

    def test_close_failure_can_be_retried_and_restores_caller_context(self):
        for operation in ("destroy_channel", "destroy_channel_desc", "destroy_stream"):
            with (
                self.subTest(operation=operation),
                simulated_npu(((operation, 1),)) as runtime,
            ):
                decoder = self.dvpp.Dvpp()
                output = decoder.process_jpeg_dec(encoded_jpeg())
                runtime.current, runtime.context = 1, 888
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.close()
                self.assertTrue(decoder.closed)
                self.assertEqual((runtime.current, runtime.context), (1, 888))
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    decoder.process_jpeg_dec(encoded_jpeg())
                decoder.close()
                self.assert_released(runtime)
                self.assertEqual((runtime.current, runtime.context), (1, 888))
                self.assertTrue(torch.all(output[0] == 1))

    def test_image_release_failure_is_retained_for_close_retry(self):
        # JPEG input is freed first, then the RGB image, then the NV12 image.
        formats = self.dvpp.PixelFormat
        for number, pixel_format in ((2, formats.RGB888), (3, formats.NV12)):
            with (
                self.subTest(pixel_format=pixel_format),
                simulated_npu((("free", number),)) as runtime,
            ):
                decoder = self.dvpp.Dvpp()
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.process_jpeg_dec(encoded_jpeg())
                self.assertEqual(len(runtime.allocations), 1)
                self.assertEqual(len(decoder._images), 1)
                image = next(iter(decoder._images))
                self.assertEqual(image.pixel_format, pixel_format)
                self.assertEqual({image.pointer}, set(runtime.allocations))
                self.assertEqual((runtime.current, runtime.context), (3, 777))
                decoder.close()
                self.assertIsNone(image.pointer)
                self.assertFalse(decoder._images)
                self.assert_released(runtime)

    def test_initialization_cleanup_failure_keeps_original_error_and_is_retryable(self):
        failures = (("create_channel", 2), ("destroy_channel", 1))
        with simulated_npu(failures) as runtime:
            decoder = self.dvpp.Dvpp()
            with self.assertRaisesRegex(
                self.dvpp.ACLFailure, "VPC create channel"
            ) as raised:
                decoder.process_jpeg_dec(encoded_jpeg())
            self.assertTrue(decoder.closed)
            self.assertTrue(
                any(
                    "initialization cleanup failed" in note
                    for note in raised.exception.__notes__
                )
            )
            self.assertTrue(runtime.channels)
            self.assertEqual((runtime.current, runtime.context), (3, 777))
            decoder.close()
            self.assert_released(runtime)

    def test_cache_reclaims_sessions_on_device_switch_and_thread_exit(self):
        with simulated_npu() as runtime, self.cached_decoders():
            cache = self.cache_module.cached_decoders
            self.jpeg.decode_jpeg(encoded_jpeg(), device="npu:1")
            first = cache.get(1)
            self.jpeg.decode_jpeg(encoded_jpeg(), device="npu:1")
            self.assertEqual(runtime.calls["create_stream"], 1)
            self.jpeg.decode_jpeg(encoded_jpeg(), device="npu:2")
            self.assertTrue(first.closed)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(self.jpeg.decode_jpeg, encoded_jpeg(), "npu:1").result()
            cache.get(2)
            self.assertEqual(len(runtime.streams), 1)
            cache.close()
            self.assert_released(runtime)

    def test_failed_step_release_retried_on_close(self):
        for failure in (("free", 1), ("destroy_pic_desc", 1)):
            with self.subTest(failure=failure), simulated_npu((failure,)) as runtime:
                decoder = self.dvpp.Dvpp()
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.process_jpeg_dec(encoded_jpeg())
                decoder.close()
                self.assert_released(runtime)

    def test_cache_replaces_object_closed_by_failed_initialization(self):
        with simulated_npu((("create_stream", 1),)) as runtime, self.cached_decoders():
            with self.assertRaises(self.dvpp.ACLFailure):
                self.jpeg.decode_jpeg(encoded_jpeg(), device="npu:1")
            first = self.cache_module.cached_decoders._sessions[
                threading.current_thread()
            ]
            self.assertTrue(first.closed)
            result = self.jpeg.decode_jpeg(encoded_jpeg(), device="npu:1")
            second = self.cache_module.cached_decoders._sessions[
                threading.current_thread()
            ]
            self.assertIsNot(first, second)
            self.assertEqual(second.device_index, 1)
            self.assertEqual(runtime.current, 3)
            self.assertTrue(torch.all(result[0] == 1))
            self.assertEqual(runtime.calls["create_stream"], 2)
            self.cache_module.cached_decoders.close()
            self.assert_released(runtime)

    def test_cache_does_not_replace_poisoned_object(self):
        with simulated_npu((("synchronize", 1),)) as runtime, self.cached_decoders():
            with self.assertRaises(self.dvpp.ACLFailure):
                self.jpeg.decode_jpeg(encoded_jpeg())
            first = self.cache_module.cached_decoders._sessions[
                threading.current_thread()
            ]
            self.assertTrue(first.poisoned)
            with self.assertRaisesRegex(self.dvpp.ACLFailure, "terminate"):
                self.jpeg.decode_jpeg(encoded_jpeg())
            self.assertIs(
                self.cache_module.cached_decoders._sessions[threading.current_thread()],
                first,
            )
            self.assertEqual(runtime.calls["create_stream"], 1)
            for pointer in runtime.inflight[1:]:
                self.assertIn(pointer, runtime.allocations)

    def test_processing_is_thread_bound_but_cleanup_can_change_threads(self):
        with simulated_npu() as runtime:
            decoder = self.dvpp.Dvpp()
            decoder.process_jpeg_dec(encoded_jpeg())
            errors = []

            def cleanup_thread():
                try:
                    decoder.process_jpeg_dec(encoded_jpeg())
                except Exception as error:
                    errors.append(error)
                decoder.close()

            thread = threading.Thread(target=cleanup_thread)
            thread.start()
            thread.join()
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertIn("creating thread", str(errors[0]))
            self.assert_released(runtime)

    def test_forked_object_cannot_process_or_close(self):
        with simulated_npu() as runtime:
            decoder = self.dvpp.Dvpp()
            decoder.pid -= 1
            with self.assertRaisesRegex(RuntimeError, "after fork"):
                decoder.process_jpeg_dec(encoded_jpeg())
            with self.assertRaisesRegex(RuntimeError, "after fork"):
                decoder.close()
            self.assertFalse(runtime.events)


if __name__ == "__main__":
    unittest.main()
