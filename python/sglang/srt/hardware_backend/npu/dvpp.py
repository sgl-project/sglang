"""Ascend DVPP media operations in the calling thread's Torch NPU context.

The JPEG operation decodes to NV12 with JPEGD, converts to RGB888 with VPC,
and returns an independent tensor. Channel and stream lifetime is per session.
"""

from __future__ import annotations

import ctypes
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import acl
import torch
from torch_npu import npu

from sglang.srt.hardware_backend.npu.jpeg_header import align_up, prepare_jpeg

ACL_ERROR_RT_CONTEXT_NULL = 107002
ACL_DEV_ATTR_NPU_ARCH = 601
A2_A3_ARCH = 2201


class ACLFailure(RuntimeError):
    pass


def check(ret, operation):
    if ret != 0:
        raise ACLFailure(f"{operation}: ACL error {ret}")


class PixelFormat(IntEnum):
    """The two ACL pixel formats supported by this JPEG pipeline."""

    NV12 = 1
    RGB888 = 12


ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


class Dvpp:
    """A reusable DVPP session bound to the creating thread's current NPU.

    With a local file:
        with Dvpp() as dvpp:
            image = dvpp.process_jpeg_dec(image_path="image.jpg")

    With encoded bytes or a one-dimensional CPU uint8 tensor, use
    dvpp.process_jpeg_dec(encoded_image=data) in the same kind of block.
    """

    def __init__(self):
        self.device_index = npu.current_device()
        self.device = torch.device(f"npu:{self.device_index}")
        self.pid, self.owner = os.getpid(), threading.current_thread()
        self.lock = threading.RLock()
        self.context = self.stream = None
        self._jpegd_channel = self._vpc_channel = None
        self._jpegd_created = self._vpc_created = False
        self.closed = self.poisoned = False
        self._supported = None
        self._images = set()
        self._pending_releases = []

    @staticmethod
    def supports_device(device):
        """Check the requested NPU architecture before allocating a stream."""
        arch, ret = acl.rt.get_device_info(device, ACL_DEV_ATTR_NPU_ARCH)
        check(ret, "query NPU architecture")
        return arch == A2_A3_ARCH

    def process_jpeg_dec(
        self, encoded_image=None, *, image_path: str | os.PathLike[str] | None = None
    ):
        """Decode one local path or encoded JPEG to contiguous RGB uint8 [3,H,W].

        Unsupported images or devices return None. Invalid arguments, file
        errors and device failures propagate. The tensor survives session close.
        """
        self._check_usable()
        if (encoded_image is None) == (image_path is None):
            raise ValueError("Provide exactly one of encoded_image or image_path")
        data = (
            Path(image_path).read_bytes() if image_path is not None else encoded_image
        )
        prepared = prepare_jpeg(data)
        if prepared is None:
            return None
        if self._supported is None:
            self._supported = self.supports_device(self.device_index)
        if not self._supported:
            return None
        return self._decode_prepared_jpeg(*prepared)

    def _decode_prepared_jpeg(self, data, info):
        """Run JPEGD and VPC; release this image's buffers once DMA is complete."""
        with self.lock:
            self._check_usable()
            with self._use_context():
                self._ensure_resources()
                with self._temporary_resources() as release:
                    nv12_image = self._decode_jpeg_to_nv12(data, info)
                    release.append((_DeviceImage.close, nv12_image))
                    rgb_image = self._convert_nv12_to_rgb(nv12_image)
                    release.append((_DeviceImage.close, rgb_image))
                    return rgb_image.to_tensor()

    def _decode_jpeg_to_nv12(self, data, info):
        """JPEGD: decode into an owned NV12 buffer after draining submitted work."""
        # Public Tensor callers supply contiguous CPU uint8; bytes callers
        # retain the original direct API. Both owners stay alive through sync.
        encoded = memoryview(data if isinstance(data, bytes) else data.numpy())
        normalized = info.normalized_header(encoded)
        size_in = len(normalized)
        if normalized is encoded:
            host_ptr = (
                acl.util.bytes_to_ptr(data)
                if isinstance(data, bytes)
                else data.data_ptr()
            )
        else:
            host_ptr = ctypes.addressof(ctypes.c_ubyte.from_buffer(normalized))
        w, h, components, fmt, ret = acl.media.dvpp_jpeg_get_image_info_v2(
            host_ptr, size_in
        )
        check(ret, "dvpp_jpeg_get_image_info_v2")
        if (w, h, components) != (info.decode_width, info.decode_height, 3):
            raise ACLFailure("PIL header and DVPP header disagree")
        size, ret = acl.media.dvpp_jpeg_predict_dec_size(
            host_ptr, size_in, PixelFormat.NV12
        )
        check(ret, "dvpp_jpeg_predict_dec_size")
        size = max(size, info.nv12_bytes)
        image = None
        try:
            with self._temporary_resources() as release:
                input_ptr, ret = acl.media.dvpp_malloc(size_in)
                check(ret, "dvpp_malloc input")
                release.append((acl.media.dvpp_free, input_ptr))
                image = self._allocate_image(
                    info.width,
                    info.height,
                    w,
                    h,
                    info.width_stride,
                    info.height_stride,
                    size,
                    PixelFormat.NV12,
                )
                check(
                    acl.rt.memcpy(
                        input_ptr, size_in, host_ptr, size_in, ACL_MEMCPY_HOST_TO_DEVICE
                    ),
                    "memcpy H2D JPEG",
                )
                descriptor = self._picture_descriptor(image, release)
                with self._submission():
                    check(
                        acl.media.dvpp_jpeg_decode_async(
                            self._jpegd_channel,
                            input_ptr,
                            size_in,
                            descriptor,
                            self.stream,
                        ),
                        "dvpp_jpeg_decode_async",
                    )
                image.check_result(descriptor, "JPEGD")
            return image
        except BaseException as error:
            self._release_resources(((_DeviceImage.close, image),), error)
            raise

    def _convert_nv12_to_rgb(self, image):
        """Convert a live NV12 image to a new RGB888 image without consuming it.

        Only images from this Dvpp owner are accepted. Synchronize and validate
        the hardware result before returning; Torch conversion is a separate
        _DeviceImage.to_tensor() operation.
        """
        self._check_usable()
        if not isinstance(image, _DeviceImage) or image._owner is not self:
            raise ValueError("VPC input must belong to the same Dvpp")
        if image.pointer is None:
            raise RuntimeError("device image is closed")
        if image.pixel_format != PixelFormat.NV12:
            raise ValueError("VPC convert_color currently supports only NV12 to RGB888")
        width_stride = align_up(image.storage_width, 16) * 3
        rgb_image = self._allocate_image(
            image.width,
            image.height,
            image.storage_width,
            image.storage_height,
            width_stride,
            image.storage_height,
            width_stride * image.storage_height,
            PixelFormat.RGB888,
        )
        try:
            with self._temporary_resources() as release:
                source = self._picture_descriptor(image, release)
                output = self._picture_descriptor(rgb_image, release)
                with self._submission():
                    check(
                        acl.media.dvpp_vpc_convert_color_async(
                            self._vpc_channel, source, output, self.stream
                        ),
                        "dvpp_vpc_convert_color_async",
                    )
                rgb_image.check_result(output, "VPC")
            return rgb_image
        except BaseException as error:
            self._release_resources(((_DeviceImage.close, rgb_image),), error)
            raise

    def _allocate_image(
        self,
        width,
        height,
        storage_width,
        storage_height,
        width_stride,
        height_stride,
        size,
        pixel_format,
    ):
        pointer, ret = acl.media.dvpp_malloc(size)
        check(ret, "dvpp_malloc device image")
        return _DeviceImage(
            self,
            pointer=pointer,
            size=size,
            pixel_format=pixel_format,
            width=width,
            height=height,
            storage_width=storage_width,
            storage_height=storage_height,
            width_stride=width_stride,
            height_stride=height_stride,
        )

    def _picture_descriptor(self, image, release):
        """Register the descriptor before setting fields so partial setup is safe."""
        desc = acl.media.dvpp_create_pic_desc()
        if not desc:
            raise ACLFailure("dvpp_create_pic_desc returned null")
        release.append((acl.media.dvpp_destroy_pic_desc, desc))
        for name, value in (
            ("data", image.pointer),
            ("size", image.size),
            ("format", image.pixel_format),
            ("width", image.storage_width),
            ("height", image.storage_height),
            ("width_stride", image.width_stride),
            ("height_stride", image.height_stride),
        ):
            check(
                getattr(acl.media, "dvpp_set_pic_desc_" + name)(desc, value),
                "dvpp_set_pic_desc_" + name,
            )
        return desc

    def _open_channels(self):
        self._jpegd_channel = acl.media.dvpp_create_channel_desc()
        if not self._jpegd_channel:
            raise ACLFailure("dvpp_create_channel_desc returned null")
        check(
            acl.media.dvpp_create_channel(self._jpegd_channel), "JPEGD create channel"
        )
        self._jpegd_created = True

        self._vpc_channel = acl.media.dvpp_create_channel_desc()
        if not self._vpc_channel:
            raise ACLFailure("dvpp_create_channel_desc VPC returned null")
        check(acl.media.dvpp_set_channel_desc_mode(self._vpc_channel, 1), "VPC mode")
        channel, ret = acl.media.dvpp_set_channel_desc_param(self._vpc_channel, 0, 0)
        check(ret, "VPC BT601 full range")
        self._vpc_channel = channel
        check(acl.media.dvpp_create_channel(channel), "VPC create channel")
        self._vpc_created = True

    def _synchronize(self, operation):
        try:
            check(acl.rt.synchronize_stream(self.stream), operation)
        except BaseException:
            self.poisoned = True
            raise

    @contextmanager
    def _submission(self):
        """Even a failed submission may have queued work. Drain before release."""
        try:
            yield
        except BaseException as original:
            try:
                self._synchronize("synchronize_stream during cleanup")
            except BaseException as error:
                original.add_note(
                    f"Device synchronization failed during cleanup: {error}"
                )
            raise
        self._synchronize("synchronize_stream")

    @contextmanager
    def _temporary_resources(self):
        resources = []
        try:
            yield resources
        except BaseException as original:
            self._release_resources(reversed(resources), original)
            raise
        else:
            self._release_resources(reversed(resources), None)

    def _release_resources(self, resources, original):
        """Release safe resources, keeping the operation's original error."""
        if self.poisoned:
            return
        errors = []
        for function, resource in resources:
            if resource:
                try:
                    ret = function(resource)
                    if ret:
                        name = getattr(function, "__name__", type(function).__name__)
                        check(ret, f"cleanup {name}")
                except Exception as error:
                    if function is not _DeviceImage.close:
                        self._pending_releases.append((function, resource))
                    errors.append(error)
        if errors:
            if original is not None:
                for error in errors:
                    original.add_note(f"Device resource cleanup failed: {error}")
            else:
                for error in errors[1:]:
                    errors[0].add_note(f"Additional cleanup failure: {error}")
                raise errors[0]

    def __enter__(self):
        """Enter a reusable decoding session without allocating resources yet."""
        with self.lock:
            self._check_usable()
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the session without hiding an exception from its body."""
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(f"DVPP session cleanup failed: {cleanup_error}")
        return False

    def _ensure_resources(self):
        """Lazily create two channels on one stream in Torch's active context."""
        if self.context is not None:
            return
        try:
            torch.empty(1, device=self.device)
            context, ret = acl.rt.get_context()
            check(ret, "acl.rt.get_context")
            if not context:
                raise ACLFailure("caller has not initialized an ACL context")
            self.context = context
            self.stream, ret = acl.rt.create_stream()
            check(ret, "acl.rt.create_stream")
            self._open_channels()
        except BaseException as error:
            self.closed = True
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"DVPP initialization cleanup failed: {cleanup_error}")
            raise

    @contextmanager
    def _use_context(self):
        """Select the owned resources, restoring caller state even on failure."""
        previous, ret = acl.rt.get_context()
        if ret == ACL_ERROR_RT_CONTEXT_NULL:
            previous = None
            if self.closed:
                npu.set_device(npu.current_device())
        else:
            check(ret, "get caller context")
        failure = None
        try:
            with npu.device(self.device):
                if self.context is not None:
                    check(acl.rt.set_context(self.context), "activate DVPP context")
                yield
        except BaseException as error:
            failure = error
            raise
        finally:
            if previous:
                try:
                    check(acl.rt.set_context(previous), "restore caller context")
                except BaseException as cleanup_error:
                    if failure is None:
                        raise
                    failure.add_note(f"Context restore failed: {cleanup_error}")

    def _check_usable(self):
        """Apply the same execution-state contract to all internal components."""
        if self.pid != os.getpid():
            raise RuntimeError("DVPP cannot be used after fork")
        if threading.current_thread() is not self.owner:
            raise RuntimeError("DVPP processing belongs to its creating thread")
        if self.poisoned:
            raise ACLFailure(
                "poisoned stream synchronization; terminate this worker process"
            )
        if self.closed:
            raise RuntimeError("DVPP is closed")

    def close(self):
        """Release owned resources; never reset the device or destroy its context.

        Successful close is idempotent. A failed release retains its handle so
        cleanup can be retried, but processing remains disabled. A poisoned
        stream requires worker termination because DMA completion is unknown.
        """
        with self.lock:
            if self.pid != os.getpid():
                raise RuntimeError("DVPP cannot be closed after fork")
            if self.poisoned:
                raise ACLFailure(
                    "poisoned stream: terminate worker to reclaim resources"
                )
            self.closed = True
            if self.context is None:
                return
            with self._use_context():
                pending, self._pending_releases = self._pending_releases, []
                self._release_resources(pending, None)
                for image in list(self._images):
                    image.close()
                if self._vpc_created:
                    check(
                        acl.media.dvpp_destroy_channel(self._vpc_channel),
                        "VPC destroy channel",
                    )
                    self._vpc_created = False
                if self._vpc_channel:
                    check(
                        acl.media.dvpp_destroy_channel_desc(self._vpc_channel),
                        "VPC destroy descriptor",
                    )
                    self._vpc_channel = None
                if self._jpegd_created:
                    check(
                        acl.media.dvpp_destroy_channel(self._jpegd_channel),
                        "JPEGD destroy channel",
                    )
                    self._jpegd_created = False
                if self._jpegd_channel:
                    check(
                        acl.media.dvpp_destroy_channel_desc(self._jpegd_channel),
                        "JPEGD destroy descriptor",
                    )
                    self._jpegd_channel = None
                if self.stream is not None:
                    check(acl.rt.destroy_stream(self.stream), "destroy_stream")
                    self.stream = None
            self.context = None


@dataclass(eq=False)
class _DeviceImage:
    """An owned DVPP allocation, used inside its Dvpp's active context.

    width/height are visible pixels; storage_width/storage_height are the
    hardware processing extent (even for JPEGD). Strides describe the allocated
    layout, with width_stride in bytes. The owner tracks images until close()
    succeeds, so a failed free remains available for a later cleanup retry.
    to_tensor() borrows this image; it neither consumes nor changes its pixels.
    """

    _owner: Dvpp
    pointer: int | None
    size: int
    pixel_format: PixelFormat
    width: int
    height: int
    storage_width: int
    storage_height: int
    width_stride: int
    height_stride: int

    def __post_init__(self):
        self._owner._images.add(self)

    def check_result(self, descriptor, operation):
        check(
            acl.media.dvpp_get_pic_desc_ret_code(descriptor),
            f"{operation} completion status",
        )
        actual = tuple(
            getattr(acl.media, "dvpp_get_pic_desc_" + field)(descriptor)
            for field in ("width", "height", "width_stride", "height_stride")
        )
        if actual != (
            self.storage_width,
            self.storage_height,
            self.width_stride,
            self.height_stride,
        ):
            raise ACLFailure(f"unexpected {operation} layout: {actual}")

    def to_tensor(self) -> torch.Tensor:
        """Return an independent, contiguous CHW uint8 tensor on the same NPU.

        Only RGB888 is supported. Remove alignment padding and synchronize
        before returning. The tensor survives closing this image or its Dvpp.
        This does not perform JPEG decoding or color conversion.
        """
        owner = self._owner
        owner._check_usable()
        if self.pointer is None:
            raise RuntimeError("device image is closed")
        if self.pixel_format != PixelFormat.RGB888:
            raise ValueError("to_tensor requires RGB888; convert color with VPC first")
        packed = torch.empty(self.size, dtype=torch.uint8, device=owner.device)
        npu.synchronize(owner.device)
        check(
            acl.rt.memcpy(
                packed.data_ptr(),
                self.size,
                self.pointer,
                self.size,
                ACL_MEMCPY_DEVICE_TO_DEVICE,
            ),
            "memcpy RGB888 D2D",
        )
        result = (
            packed.view(self.height_stride, self.width_stride // 3, 3)[
                : self.height, : self.width
            ]
            .permute(2, 0, 1)
            .contiguous()
        )
        npu.synchronize(owner.device)
        return result

    def close(self):
        """Free once work is complete; retain ownership if release fails."""
        if self._owner.pid != os.getpid():
            raise RuntimeError("device image cannot be closed after fork")
        if self.pointer is None:
            return
        if self._owner.poisoned:
            raise ACLFailure("poisoned stream: terminate worker to reclaim resources")
        check(acl.media.dvpp_free(self.pointer), "dvpp_free device image")
        self.pointer = None
        self._owner._images.discard(self)
