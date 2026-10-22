"""Synchronous DVPP JPEGD with caller-owned torch_npu context.

The accepted baseline 4:2:0 path uses device NV12 memory. It never resets or
finalizes the caller's runtime; failures are not converted into CPU success.
"""

from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, JpegImagePlugin


class UnsupportedJPEG(ValueError):
    """Valid media outside this deliberately narrow first prototype."""


class ACLFailure(RuntimeError):
    pass


def check(ret, operation):
    if ret != 0:
        raise ACLFailure(f"{operation}: ACL error {ret}")


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class JPEGInfo:
    width: int
    height: int
    width_stride: int
    height_stride: int

    @property
    def nv12_bytes(self):
        return self.width_stride * self.height_stride * 3 // 2


def inspect_jpeg(data: bytes, *, width_alignment=128, max_pixels=16_777_216):
    """Read headers only; no CPU pixel decode on the accepted path.

    128 is the original Atlas training alignment. Use 64 for A2/A3 after
    confirming the product. These are NV12 strides, not RGB strides.
    """
    if not isinstance(data, bytes):
        raise TypeError("encoded input must be bytes")
    if width_alignment not in (64, 128):
        raise ValueError("width_alignment must be 64 (A2/A3) or 128 (original 910)")
    if not data or len(data) > 32 * 1024 * 1024:
        raise ValueError("encoded input must contain 1..32 MiB")
    with Image.open(BytesIO(data)) as im:
        if im.format != "JPEG":
            raise UnsupportedJPEG("not JPEG")
        w, h = im.size
        if w * h > max_pixels:
            raise ValueError("pixel budget exceeded")
        if im.info.get("progressive") or im.info.get("progression"):
            raise UnsupportedJPEG("progressive JPEG uses CPU fallback")
        if im.mode != "RGB" or JpegImagePlugin.get_sampling(im) != 2:
            raise UnsupportedJPEG("v1 accepts only three-component YCbCr 4:2:0 JPEG")
        if not (32 <= w <= 8192 and 32 <= h <= 8192) or w % 2 or h % 2:
            raise UnsupportedJPEG("v1 requires even dimensions in [32, 8192]")
    return JPEGInfo(w, h, align_up(w, width_alignment), align_up(h, 16))


class DeviceFrame:
    """Owned device NV12 allocation. Valid only until frame/decoder close."""

    def __init__(self, decoder, ptr, size, info):
        self.decoder, self.ptr, self.size, self.info = decoder, ptr, size, info

    def download(self):
        import numpy as np

        d = self.decoder
        with d.lock:
            d._activate()
            if not self.ptr:
                raise RuntimeError("frame is closed")
            host = np.empty(self.info.nv12_bytes, dtype=np.uint8)
            check(
                d.acl.rt.memcpy(
                    host.ctypes.data, host.nbytes, self.ptr, host.nbytes, 2
                ),
                "acl.rt.memcpy D2H",
            )
            return host

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

    def __exit__(self, *args):
        self.close()


class JPEGDecoder:
    """One dedicated process, one explicit device, reusable stream/channel.

    Do not fork after construction. No automatic device selection. No reset or
    finalize: those are process-global operations which this component does not
    own. The dedicated process releases ACL runtime state on normal exit. In
    borrowed-context mode the caller retains ownership of the runtime/context.
    """

    def __init__(
        self, device: int, *, width_alignment=128, borrow_current_context=False
    ):
        if not isinstance(device, int) or isinstance(device, bool) or device < 0:
            raise ValueError("an explicit non-negative device ID is required")
        if width_alignment not in (64, 128):
            raise ValueError("width_alignment must be 64 or 128")
        import acl  # provided by the selected CANN environment

        self.acl, self.device, self.width_alignment = acl, device, width_alignment
        self.pid, self.lock, self.frames = os.getpid(), threading.RLock(), set()
        self.context = self.stream = self.channel = None
        self.channel_created = False
        self.poisoned = False
        self.owns_context = not borrow_current_context
        # An already initialized process is an integration error, not silently
        # accepted: lifecycle sharing with torch_npu is a separate next step.
        if self.owns_context:
            check(acl.init(), "acl.init (run this prototype in a fresh process)")
        try:
            if self.owns_context:
                check(acl.rt.set_device(device), "acl.rt.set_device")
                self.context, ret = acl.rt.create_context(device)
                check(ret, "acl.rt.create_context")
            else:
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
        except Exception:
            self.close()
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

    def decode(self, data: bytes) -> DeviceFrame:
        info = inspect_jpeg(data, width_alignment=self.width_alignment)
        with self.lock:
            self._activate()
            a = self.acl
            host_ptr = a.util.bytes_to_ptr(data)  # keep data alive through sync
            w, h, components, fmt, ret = a.media.dvpp_jpeg_get_image_info_v2(
                host_ptr, len(data)
            )
            check(ret, "dvpp_jpeg_get_image_info_v2")
            if (w, h, components) != (info.width, info.height, 3):
                raise ACLFailure("PIL header and DVPP header disagree")
            size, ret = a.media.dvpp_jpeg_predict_dec_size(
                host_ptr, len(data), 1
            )  # NV12
            check(ret, "dvpp_jpeg_predict_dec_size")
            size = max(size, info.nv12_bytes)
            inp = out = desc = None
            submitted = False
            try:
                inp, ret = a.media.dvpp_malloc(len(data))
                check(ret, "dvpp_malloc input")
                out, ret = a.media.dvpp_malloc(size)
                check(ret, "dvpp_malloc output")
                check(
                    a.rt.memcpy(inp, len(data), host_ptr, len(data), 1),
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
                        self.channel, inp, len(data), desc, self.stream
                    ),
                    "dvpp_jpeg_decode_async",
                )
                ret = a.rt.synchronize_stream(self.stream)
                if ret:
                    self.poisoned = True
                    check(ret, "synchronize_stream")
                submitted = False
                check(
                    a.media.dvpp_get_pic_desc_ret_code(desc), "JPEGD completion status"
                )
                actual = tuple(
                    getattr(a.media, "dvpp_get_pic_desc_" + n)(desc)
                    for n in ("width", "height", "width_stride", "height_stride")
                )
                if actual != (w, h, info.width_stride, info.height_stride):
                    raise ACLFailure(f"unexpected JPEGD layout: {actual}")
                frame = DeviceFrame(self, out, size, info)
                self.frames.add(frame)
                out = None  # ownership moves to frame
                return frame
            finally:
                if submitted and not self.poisoned:
                    if a.rt.synchronize_stream(self.stream) != 0:
                        self.poisoned = True
                # If synchronization failed, DMA may still use these addresses.
                # Leave them to process teardown rather than freeing live memory.
                if not self.poisoned:
                    for function, resource in (
                        (a.media.dvpp_destroy_pic_desc, desc),
                        (a.media.dvpp_free, inp),
                        (a.media.dvpp_free, out),
                    ):
                        if resource:
                            ret = function(resource)
                            if ret:
                                warnings.warn(f"cleanup {function.__name__}: {ret}")

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
                if self.owns_context:
                    check(self.acl.rt.destroy_context(self.context), "destroy_context")
                self.context = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TritonJPEGDecoder:
    """Borrow the current NPU context; return uint8 contiguous CHW RGB."""

    def __init__(self, device: int, *, width_alignment=64):
        import torch
        import torch_npu  # noqa: F401 - registers NPU support

        self.torch = torch
        self.device = torch.device(f"npu:{device}")
        torch.npu.set_device(self.device)
        # Force runtime initialization before borrowing its context.
        torch.empty(1, device=self.device)
        self.decoder = JPEGDecoder(
            device, width_alignment=width_alignment, borrow_current_context=True
        )

    def decode(self, data: bytes):
        from sglang.srt.hardware_backend.npu.jpeg.nv12 import nv12_to_rgb_triton

        t, d = self.torch, self.decoder
        with d.lock:
            t.npu.set_device(self.device)
            with d.decode(data) as frame:
                info = frame.info
                packed = t.empty(info.nv12_bytes, dtype=t.uint8, device=self.device)
                t.npu.synchronize(self.device)
                check(
                    d.acl.rt.memcpy(
                        packed.data_ptr(), packed.numel(), frame.ptr, packed.numel(), 3
                    ),
                    "memcpy NV12 D2D",
                )
            result = nv12_to_rgb_triton(packed, info)
            # Keep packed alive until the kernel has consumed it. Preserve the
            # synchronous output contract of the baseline decoder.
            t.npu.synchronize(self.device)
            return result

    def close(self):
        self.decoder.close()
