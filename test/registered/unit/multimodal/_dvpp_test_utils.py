"""CPU fixtures for exercising DVPP ownership, copies and failures."""

import atexit
import ctypes
import importlib.util
import io
import sys
import types
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=0, suite="base-a-test-cpu", disabled="Shared DVPP test fixtures"
)


def load_dvpp_modules(test_class):
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


def encoded_jpeg(size=(65, 49), **options):
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
        pixels = np.frombuffer(self.allocations[output["data"]], dtype=np.uint8)
        rows = pixels.reshape(output["height_stride"], output["width_stride"] // 3, 3)
        rows.fill(255)
        active = rows[: output["height"], : output["width"]]
        active[:, :, 0] = self.rgb_value
        active[:, :, 1] = np.arange(output["height"])[:, None]
        active[:, :, 2] = np.arange(output["width"])
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
def simulated_npu(failures=(), arch=2201):
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
