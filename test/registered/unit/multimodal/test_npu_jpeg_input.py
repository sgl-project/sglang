"""CPU checks for encoded JPEG buffers; ACL calls are simulated, not NPU tested."""

import atexit
import ctypes
import gc
import importlib.util
import io
import logging
import os
import sys
import threading
import types
import unittest
import weakref
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _jpeg(size=(64, 48), **options):
    output = io.BytesIO()
    Image.new("RGB", size, (25, 80, 160)).save(
        output, format="JPEG", subsampling=2, **options
    )
    return output.getvalue()


def _encoded_tensor(data):
    # Writable storage avoids frombuffer's read-only warning in the tests.
    return torch.frombuffer(bytearray(data), dtype=torch.uint8)


def _load_modules(test_class):
    repository = Path(__file__).resolve().parents[4]
    directory = repository / "python/sglang/srt/hardware_backend/npu"
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


class _ACL:
    """Exercise the real decode method while reading its actual host pointer."""

    def __init__(self, info, owner=None, memcpy_error=0):
        self.info = info
        self.owner = owner
        self.memcpy_error = memcpy_error
        self.host_pointer = None
        self.input_bytes = None
        self.bytes_to_ptr_calls = 0
        self.synchronized = False
        self.freed = []
        self._allocation = 100
        self.rt = types.SimpleNamespace(
            set_context=lambda _: 0,
            memcpy=self.memcpy,
            synchronize_stream=self.synchronize,
        )
        self.util = types.SimpleNamespace(bytes_to_ptr=self.bytes_to_ptr)
        self.media = self

    def bytes_to_ptr(self, data):
        self.bytes_to_ptr_calls += 1
        function = ctypes.pythonapi.PyBytes_AsString
        function.argtypes = [ctypes.py_object]
        function.restype = ctypes.c_void_p
        return function(data)

    def memcpy(self, destination, capacity, source, length, kind):
        assert kind == 1
        assert capacity == length
        self.host_pointer = source
        self.input_bytes = ctypes.string_at(source, length)
        if self.owner is not None:
            assert self.owner() is not None
        return self.memcpy_error

    def synchronize(self, stream):
        assert (
            ctypes.string_at(self.host_pointer, len(self.input_bytes))
            == self.input_bytes
        )
        if self.owner is not None:
            assert self.owner() is not None
        self.synchronized = True
        return 0

    def dvpp_jpeg_get_image_info_v2(self, pointer, length):
        with Image.open(io.BytesIO(ctypes.string_at(pointer, length))) as image:
            width, height = image.size
        return width, height, 3, 0, 0

    def dvpp_jpeg_predict_dec_size(self, pointer, length, format):
        return self.info.nv12_bytes, 0

    def dvpp_malloc(self, size):
        self._allocation += 1
        return self._allocation, 0

    def dvpp_free(self, pointer):
        self.freed.append(pointer)
        return 0

    def dvpp_create_pic_desc(self):
        return 300

    def dvpp_destroy_pic_desc(self, descriptor):
        return 0

    def dvpp_jpeg_decode_async(self, *args):
        return 0

    def dvpp_get_pic_desc_ret_code(self, descriptor):
        return 0

    def __getattr__(self, name):
        if name.startswith("dvpp_set_pic_desc_"):
            return lambda *args: 0
        values = {
            "width": self.info.decode_width,
            "height": self.info.decode_height,
            "width_stride": self.info.width_stride,
            "height_stride": self.info.height_stride,
        }
        prefix = "dvpp_get_pic_desc_"
        if name.startswith(prefix) and name[len(prefix) :] in values:
            return lambda descriptor: values[name[len(prefix) :]]
        raise AttributeError(name)


class TestNpuJpegInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normalizer, cls.dvpp, cls.jpeg, cls.cache_module = _load_modules(cls)

    def _decoder(self, data, owner=None, memcpy_error=0):
        info = self.normalizer.inspect_jpeg(data)
        acl = _ACL(info, owner, memcpy_error)
        resources = self.dvpp.Dvpp.__new__(self.dvpp.Dvpp)
        self.enterContext(patch.object(self.dvpp, "acl", acl))
        resources.pid = os.getpid()
        resources.owner = threading.current_thread()
        resources.lock = threading.RLock()
        resources.context, resources.stream = 1, 2
        resources.closed = resources.poisoned = False
        resources._images = set()
        resources._jpegd_channel = 3
        return resources, acl

    @contextmanager
    def _decode(self, decoder, data):
        # Exercise JPEGD directly; this test owns the returned NV12 allocation.
        prepared = self.normalizer.prepare_jpeg(data)
        self.assertIsNotNone(prepared)
        data, info = prepared
        image = decoder._decode_jpeg_to_nv12(data, info)
        try:
            yield image
        finally:
            image.close()

    @contextmanager
    def _npu(self, current=3, arch=2201, arch_error=0):
        state = types.SimpleNamespace(current=current)
        real_device = torch.device

        def device(value):
            if isinstance(value, str) and value.split(":")[0] == "npu":
                parts = value.split(":")
                return types.SimpleNamespace(
                    type="npu", index=int(parts[1]) if len(parts) == 2 else None
                )
            return real_device(value)

        @contextmanager
        def select(value):
            previous = state.current
            state.current = value if isinstance(value, int) else value.index
            try:
                yield
            finally:
                state.current = previous

        backend = types.SimpleNamespace(
            current_device=Mock(side_effect=lambda: state.current),
            architecture_query=Mock(return_value=(arch, arch_error)),
            device=select,
        )
        with (
            patch.object(torch, "device", side_effect=device),
            patch.object(torch, "npu", backend, create=True),
            patch.object(
                self.dvpp,
                "acl",
                types.SimpleNamespace(
                    rt=types.SimpleNamespace(get_device_info=backend.architecture_query)
                ),
            ),
        ):
            yield state, backend

    def test_even_tensor_h2d_uses_original_pointer(self):
        encoded = _encoded_tensor(_jpeg())
        for offset in (0, 7):
            with self.subTest(storage_offset=offset):
                storage = torch.cat(
                    (
                        torch.zeros(offset, dtype=torch.uint8),
                        encoded,
                        torch.zeros(11, dtype=torch.uint8),
                    )
                )
                data = storage[offset : offset + encoded.numel()]
                self.assertTrue(data.is_contiguous())
                self.assertEqual(data.storage_offset(), offset)
                decoder, acl = self._decoder(memoryview(data.numpy()))
                with self._decode(decoder, data):
                    self.assertEqual(acl.host_pointer, data.data_ptr())
                    self.assertEqual(acl.input_bytes, bytes(encoded.tolist()))
                    self.assertEqual(len(acl.input_bytes), encoded.numel())
                    self.assertEqual(acl.bytes_to_ptr_calls, 0)
                    self.assertTrue(acl.synchronized)

    def test_odd_tensor_h2d_uses_private_normalized_buffer(self):
        original = _jpeg((65, 49))
        data = _encoded_tensor(original)
        offset = self.normalizer.inspect_jpeg(memoryview(data.numpy())).dimension_offset
        decoder, acl = self._decoder(memoryview(data.numpy()))
        with self._decode(decoder, data):
            self.assertNotEqual(acl.host_pointer, data.data_ptr())
            self.assertEqual(bytes(data.tolist()), original)
            self.assertEqual(acl.input_bytes[:offset], original[:offset])
            self.assertEqual(acl.input_bytes[offset + 4 :], original[offset + 4 :])
            self.assertEqual(acl.input_bytes[offset : offset + 4], b"\x00\x32\x00\x42")
            self.assertEqual(acl.bytes_to_ptr_calls, 0)
            self.assertTrue(acl.synchronized)

    def test_temporary_tensor_owner_lives_through_synchronize(self):
        encoded = _jpeg()
        decoder, acl = self._decoder(encoded)

        def temporary_input():
            tensor = _encoded_tensor(encoded)
            acl.owner = weakref.ref(tensor)
            return tensor

        with self._decode(decoder, temporary_input()):
            self.assertTrue(acl.synchronized)
            self.assertEqual(acl.input_bytes, encoded)
        gc.collect()
        self.assertIsNone(acl.owner())

    def test_bytes_input_preserves_legacy_pointer_path(self):
        data = _jpeg()
        decoder, acl = self._decoder(data)
        with self._decode(decoder, data):
            self.assertEqual(acl.input_bytes, data)
            self.assertEqual(acl.bytes_to_ptr_calls, 1)

    def test_odd_bytes_input_uses_private_normalized_buffer(self):
        data = _jpeg((65, 49))
        original = bytes(bytearray(data))
        offset = self.normalizer.inspect_jpeg(data).dimension_offset
        pointer = ctypes.pythonapi.PyBytes_AsString
        pointer.argtypes = [ctypes.py_object]
        pointer.restype = ctypes.c_void_p
        original_pointer = pointer(data)
        decoder, acl = self._decoder(data)
        with self._decode(decoder, data):
            self.assertNotEqual(acl.host_pointer, original_pointer)
            self.assertEqual(data, original)
            self.assertEqual(acl.input_bytes[:offset], original[:offset])
            self.assertEqual(acl.input_bytes[offset + 4 :], original[offset + 4 :])
            self.assertEqual(acl.input_bytes[offset : offset + 4], b"\x00\x32\x00\x42")
            self.assertEqual(acl.bytes_to_ptr_calls, 0)

    def test_header_buffer_reader_matches_bytesio(self):
        operations = (
            ("read", (2,)),
            ("tell", ()),
            ("seek", (1,)),
            ("read", (3,)),
            ("seek", (-2, 2)),
            ("read", (8,)),
            ("read", (1,)),
            ("seek", (12,)),
            ("read", (1,)),
            ("tell", ()),
            ("seek", (-3, 1)),
            ("read", ()),
            ("seek", (0,)),
            ("read", (0,)),
        )
        data = b"0123456789"
        actual, expected = (
            self.normalizer._BufferReader(memoryview(data)),
            io.BytesIO(data),
        )
        for method, arguments in operations:
            with self.subTest(method=method, arguments=arguments):
                self.assertEqual(
                    getattr(actual, method)(*arguments),
                    getattr(expected, method)(*arguments),
                )
                self.assertEqual(actual.tell(), expected.tell())
        for arguments in ((-1, 0), (0, 9)):
            for stream in (actual, expected):
                with self.assertRaises(ValueError):
                    stream.seek(*arguments)
        self.assertEqual(self.normalizer._BufferReader(memoryview(b"")).read(), b"")

    def test_acl_copy_error_propagates(self):
        data = _encoded_tensor(_jpeg())
        decoder, acl = self._decoder(memoryview(data.numpy()), memcpy_error=17)
        with self.assertRaisesRegex(self.dvpp.ACLFailure, "memcpy H2D JPEG"):
            with self._decode(decoder, data):
                pass
        self.assertFalse(acl.synchronized)
        self.assertEqual(len(acl.freed), 2)

    def test_warning_uses_encoded_pixel_area(self):
        for size, should_warn in (
            ((105, 105), True),
            ((375, 377), True),
            ((376, 376), True),
            ((377, 377), False),
            ((188, 752), True),
            ((752, 188), True),
            ((188, 753), False),
        ):
            with (
                self.subTest(size=size),
                self._npu(),
                patch.object(self.cache_module.cached_decoders, "get") as factory,
                patch.object(self.jpeg.logger, "warning_once") as warning,
            ):
                data = _jpeg(size)
                self.jpeg.decode_jpeg(data)
                self.assertEqual(warning.call_count, int(should_warn))
                factory.return_value._decode_prepared_jpeg.assert_called_once_with(
                    data, self.normalizer.inspect_jpeg(data)
                )

    def test_warns_once_before_attempt_even_if_decode_fails(self):
        data = _jpeg()
        logger = logging.Logger(__name__)
        with (
            self._npu(),
            patch.object(self.jpeg, "logger", logger),
            patch.object(logger, "warning") as warning,
            patch.object(self.cache_module.cached_decoders, "get") as factory,
        ):

            def fail_decode(data, info):
                warning.assert_called_once()
                raise RuntimeError("decoder failure")

            factory.return_value._decode_prepared_jpeg.side_effect = fail_decode
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "decoder failure"):
                    self.jpeg.decode_jpeg(data)
            self.assertIn("141,376", warning.call_args.args[0])

    def test_public_noncontiguous_input_and_device_restore(self):
        encoded = _encoded_tensor(_jpeg())
        storage = torch.empty(encoded.numel() * 2, dtype=torch.uint8)
        storage[::2] = encoded
        data = storage[::2]
        observed = []

        def decode(value, info):
            observed.append(
                (value.is_contiguous(), bytes(value.tolist()), state.current)
            )
            return "decoded"

        with (
            self._npu() as (state, _),
            patch.object(
                self.cache_module.cached_decoders,
                "get",
                return_value=types.SimpleNamespace(_decode_prepared_jpeg=decode),
            ) as factory,
        ):
            self.assertEqual(self.jpeg.decode_jpeg(data, device="npu:1"), "decoded")
            self.assertEqual(state.current, 3)
        factory.assert_called_once_with(1)
        self.assertEqual(observed, [(True, bytes(encoded.tolist()), 1)])

    def test_decoder_factory_and_reuse(self):
        cache = self.cache_module._DvppCache()
        with patch.object(self.cache_module, "Dvpp") as constructor:
            constructor.return_value.closed = False
            constructor.return_value.device_index = 1
            self.assertIs(cache.get(1), cache.get(1))
            constructor.assert_called_once_with()

    def test_public_default_device_and_failure_restore(self):
        data = _encoded_tensor(_jpeg())
        for device, target in (("npu", 3), ("npu:1", 1)):
            with self.subTest(device=device), self._npu() as (state, backend):

                def fail(value, info, expected_device=target):
                    self.assertEqual(state.current, expected_device)
                    raise RuntimeError("decoder failure")

                failing = types.SimpleNamespace(_decode_prepared_jpeg=fail)
                with patch.object(
                    self.cache_module.cached_decoders, "get", return_value=failing
                ) as factory:
                    with self.assertRaisesRegex(RuntimeError, "decoder failure"):
                        self.jpeg.decode_jpeg(data, device=device)
                    self.assertEqual(state.current, 3)
                    factory.assert_called_once_with(target)
                    backend.architecture_query.assert_called_once_with(target, 601)
                    if device == "npu":
                        backend.current_device.assert_called()

    def test_public_metadata_fallback_without_device_construction(self):
        png = io.BytesIO()
        Image.new("RGB", (64, 48)).save(png, format="PNG")
        exif = Image.Exif()
        exif[274] = 6
        cases = (
            png.getvalue(),
            _jpeg(progressive=True),
            _jpeg(exif=exif),
            _jpeg(icc_profile=b"profile"),
            _jpeg((16, 16)),
        )
        with (
            patch.object(self.jpeg.logger, "warning_once") as warning,
            patch.object(
                self.cache_module.cached_decoders,
                "get",
                side_effect=AssertionError("decoder constructed"),
            ),
        ):
            for encoded in cases:
                with self.subTest(length=len(encoded)):
                    self.assertIsNone(self.jpeg.decode_jpeg(_encoded_tensor(encoded)))
            warning.assert_not_called()

    def test_public_rejects_invalid_tensor_contract(self):
        cases = (
            torch.zeros(10, dtype=torch.float32),
            torch.zeros((2, 8), dtype=torch.uint8),
            torch.empty(8, dtype=torch.uint8, device="meta"),
        )
        with patch.object(
            self.cache_module.cached_decoders,
            "get",
            side_effect=AssertionError("decoder constructed"),
        ):
            for data in cases:
                with (
                    self.subTest(
                        dtype=data.dtype, shape=data.shape, device=data.device
                    ),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    self.jpeg.decode_jpeg(data)

    def test_public_unsupported_chip_and_invalid_target(self):
        data = _encoded_tensor(_jpeg())
        for arch in (1001, 3510, 0):
            with (
                self.subTest(arch=arch),
                self._npu(arch=arch) as (state, backend),
                patch.object(self.jpeg.logger, "warning_once") as warning,
                patch.object(self.cache_module.cached_decoders, "get") as factory,
            ):
                self.assertIsNone(self.jpeg.decode_jpeg(data, device="npu:1"))
                backend.architecture_query.assert_called_once_with(1, 601)
                self.assertEqual(state.current, 3)
                factory.assert_not_called()
                warning.assert_not_called()
                with self.assertRaises((TypeError, ValueError)):
                    self.jpeg.decode_jpeg(data, device="cpu")

    def test_public_architecture_query_error_does_not_construct_decoder(self):
        data = _encoded_tensor(_jpeg())
        with (
            self._npu(arch_error=17) as (state, backend),
            patch.object(self.cache_module.cached_decoders, "get") as factory,
        ):
            with self.assertRaisesRegex(self.dvpp.ACLFailure, "query NPU architecture"):
                self.jpeg.decode_jpeg(data, device="npu:1")
            backend.architecture_query.assert_called_once_with(1, 601)
            self.assertEqual(state.current, 3)
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
