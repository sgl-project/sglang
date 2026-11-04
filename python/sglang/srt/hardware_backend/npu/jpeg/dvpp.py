"""Synchronous DVPP JPEGD with caller-owned torch_npu context.

The accepted baseline 4:2:0 path uses device NV12 memory. It never resets or
finalizes the caller's runtime; failures are not converted into CPU success.
"""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass

from PIL import Image, JpegImagePlugin

from sglang.srt.hardware_backend.npu.jpeg.normalize import (
    UnsupportedSOF,
    inspect_420_sof,
    normalize_420_sof,
)


class UnsupportedJPEG(ValueError):
    """Valid media outside the supported NPU JPEG input contract."""


class ACLFailure(RuntimeError):
    pass


def check(ret, operation):
    if ret != 0:
        raise ACLFailure(f"{operation}: ACL error {ret}")


def synchronize_or_poison(decoder, operation):
    try:
        check(decoder.acl.rt.synchronize_stream(decoder.stream), operation)
    except BaseException:
        decoder.poisoned = True
        raise


def drain_submitted(decoder, original):
    """Do not let a failed drain hide an error already being propagated."""
    try:
        synchronize_or_poison(decoder, "synchronize_stream during cleanup")
    except BaseException as error:
        if original is None:
            raise
        original.add_note(f"Device synchronization failed during cleanup: {error}")


def cleanup_resources(resources, original):
    """Attempt every release and preserve an already propagating exception."""
    errors = []
    for function, resource in resources:
        if resource:
            try:
                ret = function(resource)
                if ret:
                    name = getattr(function, "__name__", type(function).__name__)
                    check(ret, f"cleanup {name}")
            except Exception as error:
                errors.append(error)
    if errors:
        if original is not None:
            for error in errors:
                original.add_note(f"Device resource cleanup failed: {error}")
        else:
            for error in errors[1:]:
                errors[0].add_note(f"Additional cleanup failure: {error}")
            raise errors[0]


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class JPEGInfo:
    width: int
    height: int
    width_stride: int
    height_stride: int

    @property
    def decode_width(self):
        return align_up(self.width, 2)

    @property
    def decode_height(self):
        return align_up(self.height, 2)

    @property
    def nv12_bytes(self):
        return self.width_stride * self.height_stride * 3 // 2


class _BufferReader:
    """Let Pillow read encoded headers without copying the entire input."""

    def __init__(self, data):
        self.data = memoryview(data)
        self.position = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid whence")
        position = (0, self.position, len(self.data))[whence] + offset
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        end = len(self.data) if size < 0 else self.position + size
        chunk = self.data[self.position : end].tobytes()
        self.position += len(chunk)
        return chunk


def inspect_jpeg(data: bytes | bytearray | memoryview):
    """Read device eligibility and metadata without CPU pixel decoding."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("encoded input must be a byte buffer")
    if not data:
        raise ValueError("encoded input must not be empty")
    if len(data) > 32 * 1024 * 1024:
        raise UnsupportedJPEG("encoded input exceeds the 32 MiB device-path limit")
    with Image.open(_BufferReader(data)) as im:
        if im.format != "JPEG":
            raise UnsupportedJPEG("not JPEG")
        w, h = im.size
        if im.info.get("progressive") or im.info.get("progression"):
            raise UnsupportedJPEG("progressive JPEG uses CPU fallback")
        if im.mode != "RGB" or JpegImagePlugin.get_sampling(im) != 2:
            raise UnsupportedJPEG("v1 accepts only three-component YCbCr 4:2:0 JPEG")
        # VPC color conversion accepts at most 4096 in either dimension.
        # Its even decode extent and the NV12 strides remain within that limit.
        if not (32 <= w <= 4096 and 32 <= h <= 4096):
            raise UnsupportedJPEG("JPEGD/VPC requires dimensions in [32, 4096]")
        try:
            inspect_420_sof(data, w, h)
        except UnsupportedSOF as error:
            raise UnsupportedJPEG(str(error)) from error
        if im.getexif() or im.info.get("icc_profile") or im.info.get("exif"):
            raise UnsupportedJPEG("EXIF/ICC metadata uses CPU fallback")
    return JPEGInfo(w, h, align_up(w, 64), align_up(h, 16))


class DeviceFrame:
    """Owned device NV12 allocation. Valid only until frame/decoder close."""

    def __init__(self, decoder, ptr, size, info):
        self.decoder, self.ptr, self.size, self.info = decoder, ptr, size, info

    def close(self):
        d = self.decoder
        with d.lock:
            if self.ptr:
                d._activate()
                check(d.acl.media.dvpp_free(self.ptr), "dvpp_free output")
                self.ptr = 0
                d.frames.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None and self.decoder.poisoned:
            # Keep potentially active DMA memory and preserve the original error.
            return
        if exc_value is None:
            self.close()
        else:
            try:
                self.close()
            except Exception as cleanup_error:
                exc_value.add_note(f"Frame cleanup failed: {cleanup_error}")


class JPEGDecoder:
    """Reuse a JPEGD stream/channel within the caller's initialized context."""

    def __init__(self, device: int):
        if not isinstance(device, int) or isinstance(device, bool) or device < 0:
            raise ValueError("an explicit non-negative device ID is required")
        import acl  # provided by the selected CANN environment

        self.acl = acl
        self.pid, self.lock, self.frames = os.getpid(), threading.RLock(), set()
        self.context = self.stream = self.channel = None
        self.channel_created = False
        self.poisoned = False
        try:
            current, ret = acl.rt.get_device()
            check(ret, "acl.rt.get_device")
            if current != device:
                raise ACLFailure(
                    f"current context device {current} != requested {device}"
                )
            self.context, ret = acl.rt.get_context()
            check(ret, "acl.rt.get_context")
            if not self.context:
                raise ACLFailure("caller has not initialized an ACL context")
            self.stream, ret = acl.rt.create_stream()
            check(ret, "acl.rt.create_stream")
            self.channel = acl.media.dvpp_create_channel_desc()
            if not self.channel:
                raise ACLFailure("dvpp_create_channel_desc returned null")
            check(acl.media.dvpp_create_channel(self.channel), "dvpp_create_channel")
            self.channel_created = True
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"JPEG decoder cleanup failed: {cleanup_error}")
            raise

    def _activate(self):
        if self.pid != os.getpid():
            raise RuntimeError("decoder cannot be used after fork")
        if self.poisoned:
            raise ACLFailure(
                "stream synchronization failed; terminate this worker process"
            )
        if self.context is None:
            raise RuntimeError("decoder is closed")
        check(self.acl.rt.set_context(self.context), "acl.rt.set_context")

    def decode(self, data) -> DeviceFrame:
        # Public Tensor callers supply contiguous CPU uint8; bytes callers
        # retain the original direct API. Both owners stay alive through sync.
        encoded = memoryview(data if isinstance(data, bytes) else data.numpy())
        info = inspect_jpeg(encoded)
        normalized = normalize_420_sof(encoded, info.width, info.height)
        size_in = len(normalized)
        with self.lock:
            self._activate()
            a = self.acl
            if normalized is encoded:
                host_ptr = (
                    a.util.bytes_to_ptr(data)
                    if isinstance(data, bytes)
                    else data.data_ptr()
                )
            else:
                host_ptr = ctypes.addressof(ctypes.c_ubyte.from_buffer(normalized))
            w, h, components, fmt, ret = a.media.dvpp_jpeg_get_image_info_v2(
                host_ptr, size_in
            )
            check(ret, "dvpp_jpeg_get_image_info_v2")
            if (w, h, components) != (info.decode_width, info.decode_height, 3):
                raise ACLFailure("PIL header and DVPP header disagree")
            size, ret = a.media.dvpp_jpeg_predict_dec_size(host_ptr, size_in, 1)  # NV12
            check(ret, "dvpp_jpeg_predict_dec_size")
            size = max(size, info.nv12_bytes)
            inp = out = desc = None
            submitted = False
            failure = None
            try:
                inp, ret = a.media.dvpp_malloc(size_in)
                check(ret, "dvpp_malloc input")
                out, ret = a.media.dvpp_malloc(size)
                check(ret, "dvpp_malloc output")
                check(
                    a.rt.memcpy(inp, size_in, host_ptr, size_in, 1),
                    "memcpy H2D JPEG",
                )
                desc = a.media.dvpp_create_pic_desc()
                if not desc:
                    raise ACLFailure("dvpp_create_pic_desc returned null")
                for name, value in (
                    ("data", out),
                    ("size", size),
                    ("format", 1),
                    ("width", w),
                    ("height", h),
                    ("width_stride", info.width_stride),
                    ("height_stride", info.height_stride),
                ):
                    check(
                        getattr(a.media, "dvpp_set_pic_desc_" + name)(desc, value),
                        "dvpp_set_pic_desc_" + name,
                    )
                # Treat even a failed submit conservatively: drain before free.
                submitted = True
                check(
                    a.media.dvpp_jpeg_decode_async(
                        self.channel, inp, size_in, desc, self.stream
                    ),
                    "dvpp_jpeg_decode_async",
                )
                synchronize_or_poison(self, "synchronize_stream")
                submitted = False
                check(
                    a.media.dvpp_get_pic_desc_ret_code(desc), "JPEGD completion status"
                )
                actual = tuple(
                    getattr(a.media, "dvpp_get_pic_desc_" + n)(desc)
                    for n in ("width", "height", "width_stride", "height_stride")
                )
                if actual != (
                    info.decode_width,
                    info.decode_height,
                    info.width_stride,
                    info.height_stride,
                ):
                    raise ACLFailure(f"unexpected JPEGD layout: {actual}")
            except BaseException as error:
                failure = error
                raise
            finally:
                if submitted and not self.poisoned:
                    drain_submitted(self, failure)
                # If synchronization failed, DMA may still use these addresses.
                # Leave them to process teardown rather than freeing live memory.
                if not self.poisoned:
                    try:
                        cleanup_resources(
                            (
                                (a.media.dvpp_destroy_pic_desc, desc),
                                (a.media.dvpp_free, inp),
                            ),
                            failure,
                        )
                    except BaseException as cleanup_error:
                        cleanup_resources(((a.media.dvpp_free, out),), cleanup_error)
                        raise
                    if failure is not None:
                        cleanup_resources(((a.media.dvpp_free, out),), failure)
            # Transfer the output only after intermediate resources are released.
            frame = DeviceFrame(self, out, size, info)
            self.frames.add(frame)
            return frame

    def close(self):
        with self.lock:
            if self.pid != os.getpid():
                raise RuntimeError("decoder cannot be closed after fork")
            if self.poisoned:
                raise ACLFailure(
                    "poisoned stream: terminate worker to reclaim resources"
                )
            if self.context is not None:
                check(self.acl.rt.set_context(self.context), "set_context for close")
            for frame in list(self.frames):
                frame.close()
            if self.channel_created:
                check(
                    self.acl.media.dvpp_destroy_channel(self.channel),
                    "dvpp_destroy_channel",
                )
                self.channel_created = False
            if self.channel:
                check(
                    self.acl.media.dvpp_destroy_channel_desc(self.channel),
                    "dvpp_destroy_channel_desc",
                )
                self.channel = None
            if self.stream is not None:
                check(self.acl.rt.destroy_stream(self.stream), "destroy_stream")
                self.stream = None
            if self.context is not None:
                self.context = None
