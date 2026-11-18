"""Ascend DVPP image operations, currently JPEGD + VPC decoding to NPU RGB.

Use Dvpp.process_jpeg_dec for synchronized, Torch-owned CHW pixels.
JPEGD produces NV12; VPC converts it to RGB. Dvpp composes Jpegd and Vpc, each owning its channel and borrowing the same
stream/context from Dvpp. DeviceImage owns intermediate memory and delivers
independent Torch storage through to_tensor(). No device reset/finalize occurs.
The shared image loader handles PIL fallback and model-specific preprocessing.

Reading order: Dvpp.process_jpeg_dec -> Jpegd.decode -> Vpc.convert_color ->
DeviceImage.to_tensor. The component/image classes are internal building blocks:
their operations run under the owning Dvpp's lock and active context. They share
only its ACL/Torch runtime, device, stream, poison state and image registry;
neither hardware component accesses the other component's state.

DVPP also provides VPC resize/crop, JPEG encoding, PNG decoding and video
operations. These are vendor capabilities, not methods implemented here.
Before adding an operation, define its input/output and resource lifetime,
check the target chip and CANN API constraints, then add a focused method and
behavior tests. Do not assume a different API version can reuse V1 resources.
See the matching CANN documentation for the full capability/support tables:
https://www.hiascend.com/document
https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/acce/dvpp/aclpythondevg_01_0220.html
VPC color conversion and its product/format constraints:
https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/acce/dvpp/aclpythondevg_01_0307.html
Reference operation/application flow (file/YUV I/O, unlike this Tensor API):
https://github.com/Ascend/samples/tree/master/python/level2_simple_inference/1_classification/vpc_jpeg_resnet50_imagenet_classification/src
"""

from __future__ import annotations

import ctypes
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from sglang.srt.hardware_backend.npu.jpeg_header import (
    align_up,
    normalize_420_sof,
    prepare_jpeg,
)


class ACLFailure(RuntimeError):
    pass


def check(ret, operation):
    if ret != 0:
        raise ACLFailure(f"{operation}: ACL error {ret}")


def synchronize_or_poison(decoder, operation):
    """Wait for device work; disallow reuse if completion cannot be established.

    A failed wait can leave DMA using the buffers. Marking the decoder poisoned
    prevents normal cleanup from freeing them while device work may be active.
    """
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


def supports_device(device):
    """Gate V1 JPEGD/VPC on the requested device, before allocating resources."""
    import acl

    # ACL_DEV_ATTR_NPU_ARCH: this implementation supports A2/A3 (2201).
    arch, ret = acl.rt.get_device_info(device, 601)
    check(ret, "query NPU architecture")
    return arch == 2201


class PixelFormat(IntEnum):
    """The two ACL pixel formats supported by this JPEG pipeline."""

    NV12 = 1
    RGB888 = 12


ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


class Dvpp:
    """Image operations bound to the creating thread's current NPU.

    Construction records the device but allocates no DVPP resources. Reuse the
    object within that thread, preferably in a with block for automatic close;
    a cleanup thread may close it once processing has stopped. Never reuse
    after fork.
    Returned tensors survive later calls and close(). The object owns only its
    stream, channels and intermediate buffers, not the caller's Torch context.

    Independent examples; choose one according to the input source:
        with Dvpp() as dvpp:
            # Local file: image_path="image.jpg".
            # Encoded data: encoded_image=encoded_image.
            image = dvpp.process_jpeg_dec(image_path="image.jpg")
            # None means unsupported input/device; the caller may use PIL.

        with Dvpp() as dvpp:
            # Local file: image_path="image.jpg".
            # Encoded data: encoded_image=encoded_image (bytes/CPU uint8).
            image = dvpp.process_jpeg_dec(encoded_image=encoded_image)
            # None means unsupported input/device; the caller may use PIL.

        with Dvpp() as dvpp:
            for image_path in image_paths:
                # Local file: image_path=image_path.
                # Encoded data: encoded_image=encoded_image.
                # Reuse the stream and channels across local files.
                image = dvpp.process_jpeg_dec(image_path=image_path)
                # Consume image here; returned tensors also survive block exit.

    Block exit calls close(), including when processing raises. Explicit close()
    remains available for callers that manage a longer-lived object themselves.
    """

    def __init__(self):
        import acl
        import torch
        import torch_npu  # noqa: F401 - registers the NPU device backend

        self.acl, self.torch = acl, torch
        self.device_index = torch.npu.current_device()
        self.device = torch.device(f"npu:{self.device_index}")
        self.pid, self.owner = os.getpid(), threading.current_thread()
        self.lock = threading.RLock()
        self.context = self.stream = None
        self.closed = self.poisoned = False
        self._supported = None
        self._images = set()
        self._jpegd = Jpegd(self)
        self._vpc = Vpc(self)

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
        a = self.acl
        try:
            self.torch.empty(1, device=self.device)
            context, ret = a.rt.get_context()
            check(ret, "acl.rt.get_context")
            if not context:
                raise ACLFailure("caller has not initialized an ACL context")
            self.context = context
            self.stream, ret = a.rt.create_stream()
            check(ret, "acl.rt.create_stream")
            self._jpegd._ensure_channel()
            self._vpc._ensure_channel()
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
        previous, ret = self.acl.rt.get_context()
        if ret == 107002:  # ACL_ERROR_RT_CONTEXT_NULL; first use may initialize it.
            previous = None
        else:
            check(ret, "get caller context")
        failure = None
        try:
            with self.torch.npu.device(self.device):
                if self.context is not None:
                    check(
                        self.acl.rt.set_context(self.context), "activate DVPP context"
                    )
                yield
        except BaseException as error:
            failure = error
            raise
        finally:
            if previous:
                try:
                    check(self.acl.rt.set_context(previous), "restore caller context")
                except BaseException as cleanup_error:
                    if failure is None:
                        raise
                    failure.add_note(f"Context restore failed: {cleanup_error}")

    def process_jpeg_dec(
        self, encoded_image=None, *, image_path: str | os.PathLike[str] | None = None
    ):
        """Decode a JPEG to contiguous [3, H, W] uint8 NPU RGB.

        Provide exactly one non-None input: encoded_image (bytes or a 1-D CPU
        uint8 Tensor), or image_path (a local str/PathLike path, not a URL).
        The first positional argument remains encoded_image; paths must use
        image_path=. File I/O errors propagate without becoming a fallback.

        Use image_path= for a local path, or encoded_image= for encoded data:
            # Local path example:
            image = dvpp.process_jpeg_dec(image_path="image.jpg")

            # Encoded data example (an alternative to the local path example):
            image = dvpp.process_jpeg_dec(encoded_image=jpeg_bytes)

        Unsupported images/devices return None for the caller's PIL fallback.
        Invalid input and execution errors propagate. Keep caller input storage
        when possible; return independent, synchronized Torch storage at the
        original size. The creating thread's NPU is fixed for this object's life.
        """
        with self.lock:
            self._check_usable()
            if (encoded_image is None) == (image_path is None):
                raise ValueError("Provide exactly one of encoded_image or image_path")
            data = (
                Path(image_path).read_bytes()
                if image_path is not None
                else encoded_image
            )
            prepared = prepare_jpeg(data)
            if prepared is None:
                return None
            data, info = prepared
            if self._supported is None:
                self._supported = supports_device(self.device_index)
            if not self._supported:
                return None
            with self._use_context():
                self._ensure_resources()
                decoded_image = rgb_image = None
                failure = None
                try:
                    # JPEGD operations: decode the JPEG into an NV12 device image.
                    decoded_image = self._jpegd.decode(data, info=info)
                    # VPC operations: convert the device image from NV12 to RGB.
                    rgb_image = self._vpc.convert_color(
                        image=decoded_image, output_format=PixelFormat.RGB888
                    )
                    return rgb_image.to_tensor()
                except BaseException as error:
                    failure = error
                    raise
                finally:
                    if not self.poisoned:
                        cleanup_resources(
                            (
                                (DeviceImage.close, rgb_image),
                                (DeviceImage.close, decoded_image),
                            ),
                            failure,
                        )

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
            a = self.acl
            with self._use_context():
                for image in list(self._images):
                    image.close()
                self._vpc.close()
                self._jpegd.close()
                if self.stream is not None:
                    check(a.rt.destroy_stream(self.stream), "destroy_stream")
                    self.stream = None
            self.context = None


@dataclass(eq=False)
class DeviceImage:
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
        torch, acl = owner.torch, owner.acl
        packed = torch.empty(self.size, dtype=torch.uint8, device=owner.device)
        torch.npu.synchronize(owner.device)
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
        torch.npu.synchronize(owner.device)
        return result

    def close(self):
        """Free once work is complete; retain ownership if release fails."""
        if self._owner.pid != os.getpid():
            raise RuntimeError("device image cannot be closed after fork")
        if self.pointer is None:
            return
        if self._owner.poisoned:
            raise ACLFailure("poisoned stream: terminate worker to reclaim resources")
        check(self._owner.acl.media.dvpp_free(self.pointer), "dvpp_free device image")
        self.pointer = None
        self._owner._images.discard(self)


class Jpegd:
    """Internal JPEG decoding component; owns a channel, borrows Dvpp's stream.

    decode() accepts the input/metadata validated by prepare_jpeg and returns
    an owned NV12 DeviceImage. Encoded-input memory and descriptors are private
    to each call. Hardware operations require the owner's active context/lock.
    """

    def __init__(self, owner):
        self._owner = owner
        self.channel = None
        self.channel_created = False

    def _ensure_channel(self):
        acl = self._owner.acl
        self.channel = acl.media.dvpp_create_channel_desc()
        if not self.channel:
            raise ACLFailure("dvpp_create_channel_desc returned null")
        check(acl.media.dvpp_create_channel(self.channel), "JPEGD create channel")
        self.channel_created = True

    def decode(self, encoded_image, *, info):
        """JPEGD: decode into an owned NV12 buffer after draining submitted work."""
        data = encoded_image
        owner = self._owner
        owner._check_usable()
        # Public Tensor callers supply contiguous CPU uint8; bytes callers
        # retain the original direct API. Both owners stay alive through sync.
        encoded = memoryview(data if isinstance(data, bytes) else data.numpy())
        normalized = normalize_420_sof(encoded, info.width, info.height)
        size_in = len(normalized)
        a = owner.acl
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
        size, ret = a.media.dvpp_jpeg_predict_dec_size(
            host_ptr, size_in, PixelFormat.NV12
        )
        check(ret, "dvpp_jpeg_predict_dec_size")
        size = max(size, info.nv12_bytes)
        inp = out = desc = image = None
        submitted = False
        failure = None
        try:
            inp, ret = a.media.dvpp_malloc(size_in)
            check(ret, "dvpp_malloc input")
            out, ret = a.media.dvpp_malloc(size)
            check(ret, "dvpp_malloc output")
            image = DeviceImage(
                owner,
                pointer=out,
                size=size,
                pixel_format=PixelFormat.NV12,
                width=info.width,
                height=info.height,
                storage_width=w,
                storage_height=h,
                width_stride=info.width_stride,
                height_stride=info.height_stride,
            )
            check(
                a.rt.memcpy(inp, size_in, host_ptr, size_in, ACL_MEMCPY_HOST_TO_DEVICE),
                "memcpy H2D JPEG",
            )
            desc = a.media.dvpp_create_pic_desc()
            if not desc:
                raise ACLFailure("dvpp_create_pic_desc returned null")
            for name, value in (
                ("data", out),
                ("size", size),
                ("format", PixelFormat.NV12),
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
                    self.channel, inp, size_in, desc, owner.stream
                ),
                "dvpp_jpeg_decode_async",
            )
            synchronize_or_poison(owner, "synchronize_stream")
            submitted = False
            check(a.media.dvpp_get_pic_desc_ret_code(desc), "JPEGD completion status")
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
            if submitted and not owner.poisoned:
                drain_submitted(owner, failure)
            # If synchronization failed, DMA may still use these addresses.
            # Leave them to process teardown rather than freeing live memory.
            if not owner.poisoned:
                try:
                    cleanup_resources(
                        (
                            (a.media.dvpp_destroy_pic_desc, desc),
                            (a.media.dvpp_free, inp),
                        ),
                        failure,
                    )
                except BaseException as cleanup_error:
                    cleanup_resources(((DeviceImage.close, image),), cleanup_error)
                    raise
                if failure is not None:
                    cleanup_resources(((DeviceImage.close, image),), failure)
        # Transfer the output only after intermediate resources are released.
        return image

    def close(self):
        """Release this component's channel; keep failed handles for retry."""
        acl = self._owner.acl
        if self.channel_created:
            check(acl.media.dvpp_destroy_channel(self.channel), "JPEGD destroy channel")
            self.channel_created = False
        if self.channel:
            check(
                acl.media.dvpp_destroy_channel_desc(self.channel),
                "JPEGD destroy descriptor",
            )
            self.channel = None


class Vpc:
    """Internal image-processing component sharing Dvpp's execution context.

    convert_color currently supports only NV12 -> RGB888; it borrows its input
    and returns a new owned DeviceImage. It does not create a Torch tensor.
    Future crop/resize operations belong here, with explicit format, region
    and lifetime contracts verified against the target chip/CANN API.
    """

    def __init__(self, owner):
        self._owner = owner
        self.channel = None
        self.channel_created = False

    def _ensure_channel(self):
        acl = self._owner.acl
        self.channel = acl.media.dvpp_create_channel_desc()
        if not self.channel:
            raise ACLFailure("dvpp_create_channel_desc VPC returned null")
        check(acl.media.dvpp_set_channel_desc_mode(self.channel, 1), "VPC mode")
        channel, ret = acl.media.dvpp_set_channel_desc_param(
            self.channel,
            0,
            0,  # CSC type: BT.601 full range.
        )
        check(ret, "VPC BT601 full range")
        self.channel = channel
        check(acl.media.dvpp_create_channel(self.channel), "VPC create channel")
        self.channel_created = True

    def convert_color(self, image, *, output_format):
        """Convert a live NV12 image to a new RGB888 image without consuming it.

        Only images from this Dvpp owner are accepted. Synchronize and validate
        the hardware result before returning; Torch conversion is a separate
        DeviceImage.to_tensor() operation.
        """
        owner = self._owner
        owner._check_usable()
        if not isinstance(image, DeviceImage) or image._owner is not owner:
            raise ValueError("VPC input must belong to the same Dvpp")
        if image.pointer is None:
            raise RuntimeError("device image is closed")
        if (
            image.pixel_format != PixelFormat.NV12
            or output_format != PixelFormat.RGB888
        ):
            raise ValueError("VPC convert_color currently supports only NV12 to RGB888")
        a = owner.acl
        dw, dh = image.storage_width, image.storage_height
        # RGB888 stride is in bytes, aligned to 16 pixels.
        rgb_ws = align_up(dw, 16) * 3
        rgb_size = rgb_ws * dh
        output = source_desc = output_desc = rgb_image = None
        submitted = False
        failure = None
        try:
            output, ret = a.media.dvpp_malloc(rgb_size)
            check(ret, "dvpp_malloc VPC RGB888")
            rgb_image = DeviceImage(
                owner,
                pointer=output,
                size=rgb_size,
                pixel_format=PixelFormat.RGB888,
                width=image.width,
                height=image.height,
                storage_width=dw,
                storage_height=dh,
                width_stride=rgb_ws,
                height_stride=dh,
            )
            source_desc = self._description(
                image.pointer,
                image.size,
                PixelFormat.NV12,
                dw,
                dh,
                image.width_stride,
                image.height_stride,
            )
            output_desc = self._description(
                output,
                rgb_size,
                PixelFormat.RGB888,
                dw,
                dh,
                rgb_ws,
                dh,
            )
            submitted = True
            check(
                a.media.dvpp_vpc_convert_color_async(
                    self.channel,
                    source_desc,
                    output_desc,
                    owner.stream,
                ),
                "dvpp_vpc_convert_color_async",
            )
            synchronize_or_poison(owner, "synchronize_stream VPC")
            submitted = False
            check(a.media.dvpp_get_pic_desc_ret_code(output_desc), "VPC completion")
            actual = tuple(
                getattr(a.media, "dvpp_get_pic_desc_" + name)(output_desc)
                for name in ("width", "height", "width_stride", "height_stride")
            )
            if actual != (dw, dh, rgb_ws, dh):
                raise ACLFailure(f"unexpected VPC output layout: {actual}")
        except BaseException as error:
            failure = error
            raise
        finally:
            if submitted and not owner.poisoned:
                drain_submitted(owner, failure)
            # Failed synchronization may leave DMA in flight; process teardown
            # must reclaim those allocations, as in the JPEGD stage.
            if not owner.poisoned:
                try:
                    cleanup_resources(
                        (
                            (a.media.dvpp_destroy_pic_desc, output_desc),
                            (a.media.dvpp_destroy_pic_desc, source_desc),
                        ),
                        failure,
                    )
                except BaseException as cleanup_error:
                    cleanup_resources(((DeviceImage.close, rgb_image),), cleanup_error)
                    raise
                if failure is not None:
                    cleanup_resources(((DeviceImage.close, rgb_image),), failure)
        return rgb_image

    def _description(self, pointer, size, fmt, width, height, ws, hs):
        """Describe an existing device buffer for VPC without owning its storage."""
        a = self._owner.acl
        desc = a.media.dvpp_create_pic_desc()
        if not desc:
            raise ACLFailure("dvpp_create_pic_desc returned null")
        try:
            for name, value in (
                ("data", pointer),
                ("size", size),
                ("format", fmt),
                ("width", width),
                ("height", height),
                ("width_stride", ws),
                ("height_stride", hs),
            ):
                check(
                    getattr(a.media, "dvpp_set_pic_desc_" + name)(desc, value),
                    "dvpp_set_pic_desc_" + name,
                )
            return desc
        except Exception as error:
            try:
                check(a.media.dvpp_destroy_pic_desc(desc), "destroy pic descriptor")
            except Exception as cleanup_error:
                error.add_note(f"Picture descriptor cleanup failed: {cleanup_error}")
            raise

    def crop(self, image, region):
        """Reserved VPC crop operation; not implemented or used by JPEG decoding.

        Before implementing, specify region coordinates, alignment and supported
        formats, then test output ownership and device constraints. Removing
        storage padding in DeviceImage.to_tensor is not a VPC crop operation.
        """
        raise NotImplementedError("VPC crop is not implemented yet")

    def close(self):
        """Release this component's channel; keep failed handles for retry."""
        acl = self._owner.acl
        if self.channel_created:
            check(acl.media.dvpp_destroy_channel(self.channel), "VPC destroy channel")
            self.channel_created = False
        if self.channel:
            check(
                acl.media.dvpp_destroy_channel_desc(self.channel),
                "VPC destroy descriptor",
            )
            self.channel = None
