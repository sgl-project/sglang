"""Private, single-receiver Qwen image and video handoffs."""

import asyncio
import math
import re
from urllib.parse import urlsplit

import msgspec

STAGED_VIDEO_MODEL_TYPES = frozenset(
    {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}
)

# Keep one recovery task per ambiguous registered destination until the original
# write completes or E fences it. The receiver's admission limit bounds this set
# for each scheduler, including when an encoder remains unreachable.
_fence_tasks: set[asyncio.Task] = set()


class ImageLayout(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    t: int
    h: int
    w: int
    identity: str


class VideoLayout(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    t: int
    h: int
    w: int
    identities: list[str]
    timestamps: list[float] = msgspec.field(default_factory=list)
    second_per_grid_ts: float | None = None


class EncoderHandoff(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    version: int
    peer: str
    request_id: str
    transfer_id: str
    part_id: str
    part_idx: int
    num_parts: int
    image_count: int
    images: list[ImageLayout]
    videos: list[VideoLayout] = msgspec.field(default_factory=list)

    @classmethod
    def parse(
        cls, data, placeholder_count: int, video_count: int = 0
    ) -> "EncoderHandoff":
        transfer = msgspec.json.decode(data, type=cls)
        if transfer.version != (2 if transfer.videos else 1):
            raise ValueError("unsupported encoder_handoff version")
        peer = urlsplit(transfer.peer)
        if (
            peer.scheme != "grpc"
            or not peer.hostname
            or not peer.port
            or peer.username is not None
            or peer.path
            or peer.query
            or peer.fragment
        ):
            raise ValueError(
                "encoder_handoff requires an explicit grpc://host:port peer"
            )
        if not transfer.request_id or not re.fullmatch(
            "[0-9a-f]{32}", transfer.transfer_id
        ):
            raise ValueError("invalid staged request or transfer ID")
        if (transfer.num_parts, transfer.part_idx, transfer.part_id) != (
            1,
            0,
            transfer.transfer_id + "_local_part_0",
        ):
            raise ValueError("invalid staged media part")
        if (
            (not transfer.images and not transfer.videos)
            or len(transfer.images) != transfer.image_count
            or transfer.image_count != placeholder_count
            or len(transfer.videos) != video_count
        ):
            raise ValueError(
                "staged media counts differ from layout or prompt placeholders"
            )
        for image in transfer.images:
            if min(image.t, image.h, image.w) < 1 or not re.fullmatch(
                "[0-9a-f]{64}", image.identity
            ):
                raise ValueError("invalid staged image grid or identity")
        for video in transfer.videos:
            if (
                min(video.t, video.h, video.w) < 1
                or not video.identities
                or any(
                    not re.fullmatch("[0-9a-f]{64}", identity)
                    for identity in video.identities
                )
            ):
                raise ValueError("invalid staged video grid or identities")
            if any(not math.isfinite(t) or t < 0 for t in video.timestamps) or any(
                a > b for a, b in zip(video.timestamps, video.timestamps[1:])
            ):
                raise ValueError("invalid staged video timestamps")
            interval = video.second_per_grid_ts
            if interval is not None and (not math.isfinite(interval) or interval <= 0):
                raise ValueError("invalid staged video sampling interval")
        return transfer

    def validate_layout(
        self, merge_size: int, prompt_len: int, max_length: int
    ) -> None:
        rows = 0
        for layout in (*self.images, *self.videos):
            if layout.h % merge_size or layout.w % merge_size:
                raise ValueError("staged media grid is incompatible with the model")
            rows += layout.t * (layout.h // merge_size) * (layout.w // merge_size)
        if prompt_len - self.image_count - len(self.videos) + rows >= max_length:
            raise ValueError("staged media exceed the model's input length limit")

    def grids(self):
        return [[image.t, image.h, image.w] for image in self.images]

    def build_layout(self, processor, prompt, max_length):
        import torch

        self.validate_layout(processor.spatial_merge_size, len(prompt), max_length)
        kwargs = {}
        if self.images:
            kwargs["img_grid_thw"] = torch.tensor(self.grids())
        if self.videos:
            if processor.model_type not in STAGED_VIDEO_MODEL_TYPES:
                raise ValueError(
                    f"staged video is unsupported for {processor.model_type}"
                )
            frame_items = processor.model_type in ("qwen3_vl", "qwen3_vl_moe")
            timed_grid = processor.model_type == "qwen2_5_vl"
            for video in self.videos:
                segments = video.t if frame_items else 1
                if (
                    len(video.identities) != segments
                    or len(video.timestamps) != (video.t if frame_items else 0)
                    or (video.second_per_grid_ts is not None) != timed_grid
                ):
                    raise ValueError("staged video metadata differs from model layout")
            kwargs["video_grid_thw"] = torch.tensor(
                [[v.t, v.h, v.w] for v in self.videos]
            )
            if frame_items:
                kwargs["video_timestamps"] = [v.timestamps for v in self.videos]
            if timed_grid:
                kwargs["second_per_grid_ts"] = torch.tensor(
                    [v.second_per_grid_ts for v in self.videos], dtype=torch.float32
                )
        mm = processor.get_staged_media_layout(list(prompt), **kwargs)
        # Qwen3 inserts timestamp text and delimiters for each temporal slice.
        if len(mm.input_ids) >= max_length:
            raise ValueError("staged media exceed the model's input length limit")
        return mm

    def bind_identities(self, mm_items, routing_hashes):
        from sglang.srt.environ import envs
        from sglang.srt.multimodal.cache.digest import structured_digest
        from sglang.srt.multimodal.modality import Modality

        image_index = 0
        media_index = -1
        hints = routing_hashes or ()
        all_media_hints = len(hints) == self.image_count + len(self.videos)
        video_segments = iter(
            (video, index, identity)
            for video in self.videos
            for index, identity in enumerate(video.identities)
        )
        if len(mm_items) != self.image_count + sum(
            len(v.identities) for v in self.videos
        ):
            raise ValueError("staged media count differs from rebuilt prompt")
        for item in mm_items:
            hint = None
            if self.videos and item.modality == Modality.VIDEO:
                segment = next(video_segments, None)
                if segment is None:
                    raise ValueError("too many staged video segments in prompt")
                video, index, raw_identity = segment
                if index == 0:
                    media_index += 1
                identity = bytes.fromhex(raw_identity)
                # Video time and geometry affect M-RoPE even for identical tensors.
                cache_identity = structured_digest(
                    (
                        "staged-video-v2",
                        identity,
                        video.t,
                        video.h,
                        video.w,
                        index,
                        video.timestamps[index] if video.timestamps else None,
                        video.second_per_grid_ts,
                    )
                )
            else:
                if image_index >= self.image_count:
                    raise ValueError("too many staged image placeholders in prompt")
                identity = cache_identity = bytes.fromhex(
                    self.images[image_index].identity
                )
                if len(hints) == self.image_count:
                    hint = hints[image_index]
                image_index += 1
                media_index += 1
            if all_media_hints:
                hint = hints[media_index]
            if (
                item.precomputed_embeddings is not None
                and not envs.SGLANG_MM_SKIP_COMPUTE_HASH.get()
                and item.cache_key != identity
            ):
                raise ValueError("staged media embedding identity mismatch")
            item.set_identity(cache_identity)
            if hint is not None:
                item.hash = hint

    def native_transfer(self):
        from sglang.encoder.v1.encoder_pb2 import StagedImageTransfer

        fields = msgspec.to_builtins(self)
        del fields["peer"]
        if not self.videos:
            del fields["videos"]
        return StagedImageTransfer(**fields)

    async def deliver(self, receive_url: str, timeout: float):
        import grpc

        from sglang.encoder.v1 import encoder_pb2, encoder_pb2_grpc

        async with grpc.aio.insecure_channel(urlsplit(self.peer).netloc) as channel:
            stub = encoder_pb2_grpc.SglangEncoderStub(channel)
            await stub.Deliver(
                encoder_pb2.DeliverRequest(
                    transfer=self.native_transfer(), receive_url=receive_url
                ),
                timeout=timeout,
            )

    async def receive_nixl(self, receiver, shape: tuple[int, int], timeout: float):
        import grpc
        import torch

        from sglang.encoder.v1 import encoder_pb2, encoder_pb2_grpc

        deadline = asyncio.get_running_loop().time() + timeout
        transfer = self.native_transfer()
        async with grpc.aio.insecure_channel(urlsplit(self.peer).netloc) as channel:
            stub = encoder_pb2_grpc.SglangEncoderStub(channel)
            spec = await stub.Describe(
                encoder_pb2.DescribeRequest(transfer=transfer), timeout=timeout
            )
            if tuple(spec.shape) != shape:
                raise ValueError(
                    "NIXL embedding shape differs from staged media layout/model"
                )
            if spec.dtype not in ("bfloat16", "float16", "float32"):
                raise ValueError(f"unsupported staged embedding dtype: {spec.dtype}")
            dtype = getattr(torch, spec.dtype)
            nbytes = shape[0] * shape[1] * dtype.itemsize
            target = receiver.allocate(nbytes)
            try:
                tensor = target.tensor.view(dtype).view(shape)
                destination = encoder_pb2.DeliverRequest(
                    transfer=transfer, nixl_target=target.descriptor
                )
            except BaseException:
                target.close()
                raise
            try:
                await stub.Deliver(
                    destination,
                    timeout=max(0, deadline - asyncio.get_running_loop().time()),
                )
            except BaseException:
                # A lost Deliver response can leave E writing this memory. Release
                # retires the transfer and fences any writer that already claimed it.
                fenced = False
                try:
                    await stub.Release(
                        encoder_pb2.ReleaseRequest(transfer_id=self.transfer_id),
                        timeout=timeout,
                    )
                    fenced = True
                finally:
                    if fenced:
                        target.close()
                    else:
                        target.abandon()
                        task = asyncio.create_task(
                            self._refence_target(target, timeout)
                        )
                        _fence_tasks.add(task)
                        task.add_done_callback(_fence_tasks.discard)
                raise
            target.close()
            return tensor

    async def _refence_target(self, target, timeout: float):
        import grpc

        delay = 0.25
        while not target.reaped():
            await asyncio.sleep(delay)
            if target.reaped():
                return
            try:
                await self.release(timeout)
            except grpc.RpcError:
                delay = min(delay * 2, 30)
            else:
                target.close()
                return

    async def release(self, timeout: float):
        import grpc

        from sglang.encoder.v1 import encoder_pb2, encoder_pb2_grpc

        async with grpc.aio.insecure_channel(urlsplit(self.peer).netloc) as channel:
            stub = encoder_pb2_grpc.SglangEncoderStub(channel)
            await stub.Release(
                encoder_pb2.ReleaseRequest(transfer_id=self.transfer_id),
                timeout=timeout,
            )
