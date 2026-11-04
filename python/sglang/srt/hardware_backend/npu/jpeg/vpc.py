"""DVPP JPEGD and VPC color conversion to an owned NPU RGB tensor."""

from sglang.srt.hardware_backend.npu.jpeg.dvpp import (
    ACLFailure,
    JPEGDecoder,
    align_up,
    check,
    cleanup_resources,
    drain_submitted,
    synchronize_or_poison,
)


class VPCJPEGDecoder:
    """Borrow the caller's context; return synchronous uint8 contiguous CHW RGB."""

    def __init__(self, device: int):
        import torch
        import torch_npu  # noqa: F401 - registers NPU support

        self.torch = torch
        self.device = torch.device(f"npu:{device}")
        torch.npu.set_device(self.device)
        torch.empty(1, device=self.device)  # Initialize the caller's ACL context.
        self.decoder = JPEGDecoder(device)
        self.channel, self.channel_created = None, False
        d, a = self.decoder, self.decoder.acl
        try:
            with d.lock:
                d._activate()
                self.channel = a.media.dvpp_create_channel_desc()
                if not self.channel:
                    raise ACLFailure("dvpp_create_channel_desc VPC returned null")
                # JPEGD and VPC use separate channels on the same owned stream.
                check(a.media.dvpp_set_channel_desc_mode(self.channel, 1), "VPC mode")
                channel, ret = a.media.dvpp_set_channel_desc_param(
                    self.channel,
                    0,
                    0,  # CSC type, BT.601 full range.
                )
                check(ret, "VPC BT601 full range")
                self.channel = channel
                check(a.media.dvpp_create_channel(self.channel), "VPC create channel")
                self.channel_created = True
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"VPC decoder cleanup failed: {cleanup_error}")
            raise

    def _description(self, pointer, size, fmt, width, height, ws, hs):
        a = self.decoder.acl
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

    def _convert_frame(self, frame):
        t, d = self.torch, self.decoder
        a, info = d.acl, frame.info
        dw, dh = info.decode_width, info.decode_height
        # RGB888 stride is in bytes, aligned to 16 pixels.
        rgb_ws = align_up(dw, 16) * 3
        rgb_size = rgb_ws * dh
        output = source_desc = output_desc = None
        submitted = False
        failure = None
        try:
            output, ret = a.media.dvpp_malloc(rgb_size)
            check(ret, "dvpp_malloc VPC RGB888")
            source_desc = self._description(
                frame.ptr,
                frame.size,
                1,
                dw,
                dh,
                info.width_stride,
                info.height_stride,
            )
            output_desc = self._description(
                output,
                rgb_size,
                12,
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
                    d.stream,
                ),
                "dvpp_vpc_convert_color_async",
            )
            synchronize_or_poison(d, "synchronize_stream VPC")
            submitted = False
            check(a.media.dvpp_get_pic_desc_ret_code(output_desc), "VPC completion")
            actual = tuple(
                getattr(a.media, "dvpp_get_pic_desc_" + name)(output_desc)
                for name in ("width", "height", "width_stride", "height_stride")
            )
            if actual != (dw, dh, rgb_ws, dh):
                raise ACLFailure(f"unexpected VPC output layout: {actual}")
            # Transfer ownership from DVPP memory before releasing its buffers.
            packed = t.empty(rgb_size, dtype=t.uint8, device=self.device)
            t.npu.synchronize(self.device)
            check(
                a.rt.memcpy(packed.data_ptr(), rgb_size, output, rgb_size, 3),
                "memcpy RGB888 D2D",
            )
            result = (
                packed.view(dh, rgb_ws // 3, 3)[: info.height, : info.width]
                .permute(2, 0, 1)
                .contiguous()
            )
            t.npu.synchronize(self.device)
            return result
        except BaseException as error:
            failure = error
            raise
        finally:
            if submitted and not d.poisoned:
                drain_submitted(d, failure)
            # Failed synchronization may leave DMA in flight; process teardown
            # must reclaim those allocations, as in JPEGDecoder.
            if not d.poisoned:
                cleanup_resources(
                    (
                        (a.media.dvpp_destroy_pic_desc, output_desc),
                        (a.media.dvpp_destroy_pic_desc, source_desc),
                        (a.media.dvpp_free, output),
                    ),
                    failure,
                )

    def decode(self, data):
        """Decode bytes or a contiguous CPU uint8 tensor through JPEGD/VPC."""
        d = self.decoder
        with d.lock:
            self.torch.npu.set_device(self.device)
            with d.decode(data) as frame:
                return self._convert_frame(frame)

    def close(self):
        d = self.decoder
        with d.lock:
            if not self.channel and d.context is None:
                return
            context, ret = d.acl.rt.get_context()
            check(ret, "get caller context for close")
            if not context:
                raise ACLFailure("decoder cleanup requires an active caller context")
            try:
                if self.channel:
                    d._activate()
                    if self.channel_created:
                        check(
                            d.acl.media.dvpp_destroy_channel(self.channel),
                            "VPC destroy channel",
                        )
                        self.channel_created = False
                    check(
                        d.acl.media.dvpp_destroy_channel_desc(self.channel),
                        "VPC destroy descriptor",
                    )
                    self.channel = None
                d.close()
            except Exception as error:
                try:
                    check(
                        d.acl.rt.set_context(context),
                        "restore caller context after close",
                    )
                except Exception as cleanup_error:
                    error.add_note(f"Context restore failed: {cleanup_error}")
                raise
            else:
                check(
                    d.acl.rt.set_context(context), "restore caller context after close"
                )
