"""CPU lifecycle tests with simulated ACL; these do not validate NPU pixels."""

import atexit
import ctypes
import importlib.util
import io
import sys
import threading
import types
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _load_modules(test_class):
    directory = (
        Path(__file__).resolve().parents[4] / "python/sglang/srt/hardware_backend/npu"
    )
    package = "sglang.srt.hardware_backend.npu"
    loaded = []
    with patch.dict(
        sys.modules,
        {"acl": types.ModuleType("acl"), "torch_npu": types.SimpleNamespace(npu=None)},
    ):
        for name, filename in (
            (package + ".jpeg_header", "jpeg_header.py"),
            (package + ".dvpp", "dvpp.py"),
            (package + ".jpeg", "jpeg/__init__.py"),
            (package + ".jpeg._cache", "jpeg/_cache.py"),
        ):
            spec = importlib.util.spec_from_file_location(name, directory / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
        atexit.unregister(loaded[-1].cached_decoders.close)
    test_class.enterClassContext(
        patch.dict(sys.modules, {module.__name__: module for module in loaded})
    )
    return loaded


def _jpeg(size=(65, 49), **options):
    stream = io.BytesIO()
    Image.new("RGB", size, (25, 80, 160)).save(
        stream, format="JPEG", subsampling=2, **options
    )
    return stream.getvalue()


class _Runtime:
    """Track actual host allocations and reject frees while DMA may be active."""

    def __init__(self, failures=(), arch=2201):
        self.failures = set(failures)
        self.arch = arch
        self.current = 3
        self.context = 777
        self.events = []
        self.calls = Counter()
        self.allocations = {}
        self.descriptors = {}
        self.channel_descriptors = set()
        self.channels = set()
        self.streams = set()
        self.inflight = None
        self.serial = 100
        self.rgb_value = 0
        self.rt = types.SimpleNamespace(
            get_context=self.get_context,
            set_context=self.set_context,
            create_stream=self.create_stream,
            destroy_stream=self.destroy_stream,
            get_device_info=self.get_device_info,
            synchronize_stream=self.synchronize_stream,
            memcpy=self.memcpy,
        )
        self.util = types.SimpleNamespace(bytes_to_ptr=self.bytes_to_ptr)
        self.media = self

    def status(self, name, *arguments):
        self.events.append((name, *arguments))
        self.calls[name] += 1
        return 17 if (name, self.calls[name]) in self.failures else 0

    def handle(self):
        self.serial += 1
        return self.serial

    def get_context(self):
        return self.context, self.status("get_context")

    def set_context(self, context):
        ret = self.status("set_context", context)
        if not ret:
            self.context = context
        return ret

    def get_device_info(self, device, attribute):
        assert attribute == 601
        return self.arch, self.status("architecture", device)

    def create_stream(self):
        ret = self.status("create_stream")
        if ret:
            return None, ret
        stream = self.handle()
        self.streams.add(stream)
        return stream, 0

    def destroy_stream(self, stream):
        assert self.inflight is None
        ret = self.status("destroy_stream", stream)
        if not ret:
            self.streams.remove(stream)
        return ret

    def dvpp_create_channel_desc(self):
        if self.status("create_channel_desc"):
            return None
        channel = self.handle()
        self.channel_descriptors.add(channel)
        return channel

    def dvpp_create_channel(self, channel):
        ret = self.status("create_channel", channel)
        if not ret:
            self.channels.add(channel)
        return ret

    def dvpp_set_channel_desc_mode(self, channel, mode):
        assert mode == 1
        return self.status("channel_mode", channel)

    def dvpp_set_channel_desc_param(self, channel, parameter, value):
        assert (parameter, value) == (0, 0)
        return channel, self.status("channel_param", channel)

    def dvpp_destroy_channel(self, channel):
        ret = self.status("destroy_channel", channel)
        if not ret:
            self.channels.remove(channel)
        return ret

    def dvpp_destroy_channel_desc(self, channel):
        assert channel not in self.channels
        ret = self.status("destroy_channel_desc", channel)
        if not ret:
            self.channel_descriptors.remove(channel)
        return ret

    def dvpp_malloc(self, size):
        ret = self.status("malloc", size)
        if ret:
            return None, ret
        buffer = ctypes.create_string_buffer(size)
        pointer = ctypes.addressof(buffer)
        self.allocations[pointer] = buffer
        return pointer, 0

    def dvpp_free(self, pointer):
        assert self.inflight is None or pointer not in self.inflight[1:]
        ret = self.status("free", pointer)
        if not ret:
            del self.allocations[pointer]
        return ret

    def dvpp_create_pic_desc(self):
        if self.status("create_pic_desc"):
            return None
        descriptor = self.handle()
        self.descriptors[descriptor] = {}
        return descriptor

    def dvpp_destroy_pic_desc(self, descriptor):
        ret = self.status("destroy_pic_desc", descriptor)
        if not ret:
            del self.descriptors[descriptor]
        return ret

    def bytes_to_ptr(self, data):
        pointer = ctypes.pythonapi.PyBytes_AsString
        pointer.argtypes = [ctypes.py_object]
        pointer.restype = ctypes.c_void_p
        return pointer(data)

    def dvpp_jpeg_get_image_info_v2(self, pointer, size):
        with Image.open(io.BytesIO(ctypes.string_at(pointer, size))) as image:
            width, height = image.size
        return width, height, 3, 0, 0

    def dvpp_jpeg_predict_dec_size(self, pointer, size, pixel_format):
        # The implementation must also honor the aligned NV12 layout size.
        return 1, 0

    def dvpp_jpeg_decode_async(self, channel, pointer, size, descriptor, stream):
        assert channel in self.channels and stream in self.streams
        output = self.descriptors[descriptor]["data"]
        self.inflight = ("jpeg", pointer, output)
        return self.status("submit_jpeg")

    def dvpp_vpc_convert_color_async(self, channel, source, output, stream):
        assert channel in self.channels and stream in self.streams
        source, output = self.descriptors[source], self.descriptors[output]
        assert source["format"] == 1 and output["format"] == 12
        assert source["data"] in self.allocations
        self.inflight = ("vpc", source["data"], output["data"])
        self.rgb_value += 1
        ctypes.memset(output["data"], self.rgb_value, output["size"])
        return self.status("submit_vpc")

    def synchronize_stream(self, stream):
        ret = self.status("synchronize", stream)
        if not ret:
            self.inflight = None
        return ret

    def dvpp_get_pic_desc_ret_code(self, descriptor):
        return self.status("completion", descriptor)

    def memcpy(self, destination, capacity, source, size, kind):
        assert capacity == size
        if kind == 3:
            assert self.inflight is None
        ret = self.status("copy_h2d" if kind == 1 else "copy_d2d")
        if not ret:
            ctypes.memmove(destination, source, size)
        return ret

    def __getattr__(self, name):
        set_prefix, get_prefix = "dvpp_set_pic_desc_", "dvpp_get_pic_desc_"
        if name.startswith(set_prefix):
            field = name[len(set_prefix) :]

            def setter(descriptor, value):
                self.descriptors[descriptor][field] = value
                return 0

            return setter
        if name.startswith(get_prefix):
            return lambda descriptor: self.descriptors[descriptor][
                name[len(get_prefix) :]
            ]
        raise AttributeError(name)


@contextmanager
def _runtime(failures=(), arch=2201):
    runtime = _Runtime(failures, arch)
    real_device, real_empty = torch.device, torch.empty

    def device(value):
        if isinstance(value, str) and value.startswith("npu"):
            parts = value.split(":")
            return types.SimpleNamespace(
                type="npu", index=int(parts[1]) if len(parts) == 2 else None
            )
        return real_device(value)

    @contextmanager
    def select(target):
        previous = runtime.current
        runtime.current = target if isinstance(target, int) else target.index
        runtime.context = 1000 + runtime.current
        try:
            yield
        finally:
            runtime.current = previous
            runtime.context = 1000 + previous

    def empty(*args, **kwargs):
        target = kwargs.pop("device", None)
        if target is not None:
            assert target.type == "npu" and target.index == runtime.current
        return real_empty(*args, **kwargs)

    backend = types.SimpleNamespace(
        current_device=lambda: runtime.current,
        device=select,
        synchronize=lambda _: runtime.status("torch_synchronize"),
    )
    with (
        patch.object(torch, "device", side_effect=device),
        patch.object(torch, "empty", side_effect=empty),
        patch.object(torch, "npu", backend, create=True),
        patch.object(
            sys.modules["sglang.srt.hardware_backend.npu.dvpp"], "acl", runtime
        ),
        patch.object(
            sys.modules["sglang.srt.hardware_backend.npu.dvpp"], "npu", backend
        ),
    ):
        yield runtime


class TestNpuDvppLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header, cls.dvpp, cls.jpeg, cls.cache_module = _load_modules(cls)

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
        with _runtime() as runtime:
            decoder = self.dvpp.Dvpp()
            self.assertEqual(decoder.device_index, 3)
            self.assertFalse(runtime.events)
            decoder.close()
            decoder.close()
            self.assertFalse(runtime.events)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                decoder.process_jpeg_dec(_jpeg())

    def test_direct_fallback_does_not_create_resources(self):
        for arch, data in ((2201, _jpeg(progressive=True)), (3510, _jpeg())):
            with self.subTest(arch=arch), _runtime(arch=arch) as runtime:
                decoder = self.dvpp.Dvpp()
                self.assertIsNone(decoder.process_jpeg_dec(data))
                self.assertIsNone(decoder.process_jpeg_dec(data))
                decoder.close()
                self.assertEqual(runtime.calls["create_stream"], 0)
                self.assertEqual(runtime.calls["malloc"], 0)
                self.assertEqual(runtime.calls["architecture"], int(arch == 3510))

    def test_context_reuses_file_and_memory_inputs_and_closes(self):
        with TemporaryDirectory() as directory, _runtime() as runtime:
            path = Path(directory) / "image.jpg"
            path.write_bytes(_jpeg())
            with self.dvpp.Dvpp() as decoder:
                self.assertFalse(runtime.events)
                first = decoder.process_jpeg_dec(image_path=path)
                second = decoder.process_jpeg_dec(image_path=str(path))
                third = decoder.process_jpeg_dec(encoded_image=path.read_bytes())
                fourth = decoder.process_jpeg_dec(path.read_bytes())
                fifth = decoder.process_jpeg_dec(
                    encoded_image=torch.tensor(
                        list(path.read_bytes()), dtype=torch.uint8
                    )
                )
                self.assertEqual(runtime.calls["create_stream"], 1)
                self.assertEqual(runtime.calls["create_channel"], 2)
                self.assertFalse(runtime.allocations)
            self.assertTrue(decoder.closed)
            self.assert_released(runtime)
            for index, tensor in enumerate(
                (first, second, third, fourth, fifth), start=1
            ):
                self.assertEqual(tuple(tensor.shape), (3, 49, 65))
                self.assertEqual(tensor.dtype, torch.uint8)
                self.assertTrue(tensor.is_contiguous())
                self.assertTrue(torch.all(tensor == index))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                with decoder:
                    self.fail("A closed decoder must not enter a new session")
            with self.assertRaisesRegex(RuntimeError, "closed"):
                decoder.process_jpeg_dec(image_path=path)

    def test_decode_requires_exactly_one_input_before_io_or_allocation(self):
        with _runtime() as runtime, self.dvpp.Dvpp() as decoder:
            with patch.object(Path, "read_bytes") as read_bytes:
                for inputs in (
                    {},
                    {"encoded_image": None},
                    {"image_path": None},
                    {"encoded_image": _jpeg(), "image_path": "image.jpg"},
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
        with _runtime() as runtime, self.dvpp.Dvpp() as decoder:
            with patch.object(Path, "read_bytes") as read_bytes:
                with self.assertRaises(TypeError):
                    decoder.process_jpeg_dec("image.jpg")
                with self.assertRaises(TypeError):
                    decoder.process_jpeg_dec(encoded_image=Path("image.jpg"))
                with self.assertRaises(TypeError):
                    decoder.process_jpeg_dec(image_path=_jpeg())
                read_bytes.assert_not_called()
            self.assertFalse(runtime.events)

    def test_file_fallback_and_io_error_do_not_allocate(self):
        with TemporaryDirectory() as directory, _runtime() as runtime:
            path = Path(directory) / "unsupported.jpg"
            path.write_bytes(_jpeg(progressive=True))
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
        with _runtime() as runtime:
            original = ValueError("consumer failed")
            with self.assertRaises(ValueError) as caught:
                with self.dvpp.Dvpp() as decoder:
                    decoder.process_jpeg_dec(_jpeg())
                    raise original
            self.assertIs(caught.exception, original)
            self.assertTrue(decoder.closed)
            self.assert_released(runtime)
        with _runtime((("submit_vpc", 1),)) as runtime:
            with self.assertRaises(self.dvpp.ACLFailure):
                with self.dvpp.Dvpp() as decoder:
                    decoder.process_jpeg_dec(_jpeg())
            self.assertTrue(decoder.closed)
            self.assert_released(runtime)

    def test_context_reports_cleanup_failure_without_hiding_body_error(self):
        for original in (None, ValueError("consumer failed"), KeyboardInterrupt()):
            with (
                self.subTest(original=original),
                _runtime((("destroy_stream", 1),)) as runtime,
            ):
                expected = self.dvpp.ACLFailure if original is None else type(original)
                with self.assertRaises(expected) as caught:
                    with self.dvpp.Dvpp() as decoder:
                        decoder.process_jpeg_dec(_jpeg())
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

    def test_reuses_channels_and_preserves_output_after_close(self):
        with _runtime() as runtime:
            decoder = self.dvpp.Dvpp()
            first = decoder.process_jpeg_dec(_jpeg())
            second = decoder.process_jpeg_dec(_jpeg((64, 48)))
            self.assertEqual(tuple(first.shape), (3, 49, 65))
            self.assertEqual(tuple(second.shape), (3, 48, 64))
            self.assertTrue(first.is_contiguous() and second.is_contiguous())
            self.assertEqual(first.dtype, torch.uint8)
            self.assertNotEqual(first.data_ptr(), second.data_ptr())
            self.assertEqual(runtime.calls["create_stream"], 1)
            self.assertEqual(runtime.calls["create_channel"], 2)
            self.assertEqual(runtime.calls["architecture"], 1)
            self.assertEqual(runtime.calls["synchronize"], 4)
            self.assertEqual(runtime.calls["torch_synchronize"], 4)
            self.assertFalse(runtime.allocations)
            self.assertFalse(runtime.descriptors)
            decoder.close()
            decoder.close()
            self.assert_released(runtime)
            self.assertTrue(torch.all(first == 1))
            self.assertTrue(torch.all(second == 2))
            self.assertEqual((runtime.current, runtime.context), (3, 777))

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
            with self.subTest(failure=failure), _runtime((failure,)) as runtime:
                decoder = self.dvpp.Dvpp()
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.process_jpeg_dec(_jpeg())
                self.assertTrue(decoder.closed)
                self.assert_released(runtime)
                self.assertEqual((runtime.current, runtime.context), (3, 777))

    def test_components_preserve_image_ownership_and_tensor_independence(self):
        with _runtime() as runtime:
            decoder = self.dvpp.Dvpp()
            data, info = self.header.prepare_jpeg(_jpeg())
            with decoder.lock, decoder._use_context():
                decoder._ensure_resources()
                nv12 = decoder._decode_jpeg_to_nv12(data, info=info)
                self.assertEqual(nv12.pixel_format, self.dvpp.PixelFormat.NV12)
                self.assertEqual((nv12.width, nv12.height), (65, 49))
                self.assertEqual((nv12.storage_width, nv12.storage_height), (66, 50))
                self.assertEqual(decoder._images, {nv12})
                self.assertEqual(set(runtime.allocations), {nv12.pointer})
                self.assertEqual(runtime.calls["submit_jpeg"], 1)
                self.assertEqual(runtime.calls["submit_vpc"], 0)
                self.assertEqual(runtime.calls["copy_d2d"], 0)
                with self.assertRaisesRegex(ValueError, "RGB888"):
                    nv12.to_tensor()

                rgb = decoder._convert_nv12_to_rgb(nv12)
                self.assertEqual(rgb.pixel_format, self.dvpp.PixelFormat.RGB888)
                self.assertEqual((rgb.width, rgb.height), (65, 49))
                self.assertEqual(decoder._images, {nv12, rgb})
                self.assertEqual(set(runtime.allocations), {nv12.pointer, rgb.pointer})
                self.assertFalse(runtime.descriptors)
                self.assertEqual(runtime.calls["submit_vpc"], 1)
                self.assertEqual(runtime.calls["copy_d2d"], 0)

                first = rgb.to_tensor()
                ctypes.memset(rgb.pointer, 7, rgb.size)
                second = rgb.to_tensor()
                self.assertEqual(tuple(first.shape), (3, 49, 65))
                self.assertEqual(first.dtype, torch.uint8)
                self.assertTrue(first.is_contiguous() and second.is_contiguous())
                self.assertNotEqual(first.data_ptr(), second.data_ptr())
                self.assertEqual(decoder._images, {nv12, rgb})
                self.assertEqual(runtime.calls["submit_jpeg"], 1)
                self.assertEqual(runtime.calls["submit_vpc"], 1)
                self.assertEqual(runtime.calls["copy_d2d"], 2)
                self.assertEqual(runtime.calls["torch_synchronize"], 4)
                rgb.close()
                rgb.close()
                self.assertIn(nv12.pointer, runtime.allocations)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    rgb.to_tensor()
                nv12.close()
                self.assertFalse(decoder._images)
            decoder.close()
            self.assert_released(runtime)
            self.assertTrue(torch.all(first == 1))
            self.assertTrue(torch.all(second == 7))

    def test_vpc_rejects_invalid_images_and_formats_before_allocation(self):
        with _runtime() as runtime:
            decoder, other = self.dvpp.Dvpp(), self.dvpp.Dvpp()
            data, info = self.header.prepare_jpeg(_jpeg())
            with decoder.lock, decoder._use_context():
                decoder._ensure_resources()
                nv12 = decoder._decode_jpeg_to_nv12(data, info=info)
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
        with _runtime((("copy_d2d", 2),)) as runtime:
            decoder = self.dvpp.Dvpp()
            runtime.current, runtime.context = 1, 888
            result = decoder.process_jpeg_dec(_jpeg())
            self.assertTrue(torch.all(result == 1))
            self.assertIn(("architecture", 3), runtime.events)
            self.assertEqual((runtime.current, runtime.context), (1, 888))
            with self.assertRaisesRegex(self.dvpp.ACLFailure, "D2D"):
                decoder.process_jpeg_dec(_jpeg())
            self.assertEqual((runtime.current, runtime.context), (1, 888))
            decoder.close()
            self.assert_released(runtime)
            self.assertEqual((runtime.current, runtime.context), (1, 888))

    def test_failed_submissions_are_drained_before_free(self):
        for operation in ("submit_jpeg", "submit_vpc"):
            with (
                self.subTest(operation=operation),
                _runtime(((operation, 1),)) as runtime,
            ):
                decoder = self.dvpp.Dvpp()
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.process_jpeg_dec(_jpeg())
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
                _runtime((("synchronize", number),)) as runtime,
            ):
                decoder = self.dvpp.Dvpp()
                with self.assertRaisesRegex(self.dvpp.ACLFailure, "synchronize_stream"):
                    decoder.process_jpeg_dec(_jpeg())
                self.assertTrue(decoder.poisoned)
                self.assertIsNotNone(runtime.inflight)
                for pointer in runtime.inflight[1:]:
                    self.assertIn(pointer, runtime.allocations)
                event_count = len(runtime.events)
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.process_jpeg_dec(_jpeg())
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
            with self.subTest(operation=operation), _runtime(failures) as runtime:
                decoder = self.dvpp.Dvpp()
                with self.assertRaises(self.dvpp.ACLFailure) as raised:
                    decoder.process_jpeg_dec(_jpeg())
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
            with self.subTest(failure=failure), _runtime((failure,)) as runtime:
                decoder = self.dvpp.Dvpp()
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.process_jpeg_dec(_jpeg())
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
                _runtime(((operation, 1),)) as runtime,
            ):
                decoder = self.dvpp.Dvpp()
                output = decoder.process_jpeg_dec(_jpeg())
                runtime.current, runtime.context = 1, 888
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.close()
                self.assertTrue(decoder.closed)
                self.assertEqual((runtime.current, runtime.context), (1, 888))
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    decoder.process_jpeg_dec(_jpeg())
                decoder.close()
                self.assert_released(runtime)
                self.assertEqual((runtime.current, runtime.context), (1, 888))
                self.assertTrue(torch.all(output == 1))

    def test_image_release_failure_is_retained_for_close_retry(self):
        # JPEG input is freed first, then the RGB image, then the NV12 image.
        formats = self.dvpp.PixelFormat
        for number, pixel_format in ((2, formats.RGB888), (3, formats.NV12)):
            with (
                self.subTest(pixel_format=pixel_format),
                _runtime((("free", number),)) as runtime,
            ):
                decoder = self.dvpp.Dvpp()
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.process_jpeg_dec(_jpeg())
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
        with _runtime(failures) as runtime:
            decoder = self.dvpp.Dvpp()
            with self.assertRaisesRegex(
                self.dvpp.ACLFailure, "VPC create channel"
            ) as raised:
                decoder.process_jpeg_dec(_jpeg())
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
        with _runtime() as runtime, self.cached_decoders():
            cache = self.cache_module.cached_decoders
            self.jpeg.decode_jpeg(_jpeg(), device="npu:1")
            first = cache.get(1)
            self.jpeg.decode_jpeg(_jpeg(), device="npu:2")
            self.assertTrue(first.closed)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(self.jpeg.decode_jpeg, _jpeg(), "npu:1").result()
            cache.get(2)
            self.assertEqual(len(runtime.streams), 1)
            cache.close()
            self.assert_released(runtime)

    def test_failed_step_release_retried_on_close(self):
        for failure in (("free", 1), ("destroy_pic_desc", 1)):
            with self.subTest(failure=failure), _runtime((failure,)) as runtime:
                decoder = self.dvpp.Dvpp()
                with self.assertRaises(self.dvpp.ACLFailure):
                    decoder.process_jpeg_dec(_jpeg())
                decoder.close()
                self.assert_released(runtime)

    def test_cache_replaces_object_closed_by_failed_initialization(self):
        with _runtime((("create_stream", 1),)) as runtime, self.cached_decoders():
            with self.assertRaises(self.dvpp.ACLFailure):
                self.jpeg.decode_jpeg(_jpeg(), device="npu:1")
            first = self.cache_module.cached_decoders._sessions[
                threading.current_thread()
            ]
            self.assertTrue(first.closed)
            result = self.jpeg.decode_jpeg(_jpeg(), device="npu:1")
            second = self.cache_module.cached_decoders._sessions[
                threading.current_thread()
            ]
            self.assertIsNot(first, second)
            self.assertEqual(second.device_index, 1)
            self.assertEqual(runtime.current, 3)
            self.assertTrue(torch.all(result == 1))
            self.assertEqual(runtime.calls["create_stream"], 2)
            self.cache_module.cached_decoders.close()
            self.assert_released(runtime)

    def test_cache_does_not_replace_poisoned_object(self):
        with _runtime((("synchronize", 1),)) as runtime, self.cached_decoders():
            with self.assertRaises(self.dvpp.ACLFailure):
                self.jpeg.decode_jpeg(_jpeg())
            first = self.cache_module.cached_decoders._sessions[
                threading.current_thread()
            ]
            self.assertTrue(first.poisoned)
            with self.assertRaisesRegex(self.dvpp.ACLFailure, "terminate"):
                self.jpeg.decode_jpeg(_jpeg())
            self.assertIs(
                self.cache_module.cached_decoders._sessions[threading.current_thread()],
                first,
            )
            self.assertEqual(runtime.calls["create_stream"], 1)
            for pointer in runtime.inflight[1:]:
                self.assertIn(pointer, runtime.allocations)

    def test_processing_is_thread_bound_but_cleanup_can_change_threads(self):
        with _runtime() as runtime:
            decoder = self.dvpp.Dvpp()
            decoder.process_jpeg_dec(_jpeg())
            errors = []

            def cleanup_thread():
                try:
                    decoder.process_jpeg_dec(_jpeg())
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
        with _runtime() as runtime:
            decoder = self.dvpp.Dvpp()
            decoder.pid -= 1
            with self.assertRaisesRegex(RuntimeError, "after fork"):
                decoder.process_jpeg_dec(_jpeg())
            with self.assertRaisesRegex(RuntimeError, "after fork"):
                decoder.close()
            self.assertFalse(runtime.events)


if __name__ == "__main__":
    unittest.main()
