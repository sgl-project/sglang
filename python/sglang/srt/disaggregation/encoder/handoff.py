"""Private, single-receiver multimodal handoffs."""

import asyncio
import logging
import math
import re
from urllib.parse import urlsplit

import msgspec

logger = logging.getLogger(__name__)

# One short-lived refencing task per ambiguous destination. A target whose
# owner cannot be reached stays registered independently of this task.
_fence_tasks: set[asyncio.Task] = set()


class StagedPromptMarkers:
    """Raw media token IDs used at ingress, before the native processor expands them."""

    def __init__(
        self, image, video, alternatives=(), accepts_expanded=False, audio=None
    ):
        self.image = image
        self.video = video
        self.audio = audio
        self.alternatives = alternatives
        self.accepts_expanded = accepts_expanded

    @classmethod
    def from_processor(cls, processor):
        if custom := getattr(processor, "get_staged_prompt_markers", None):
            return custom()
        mm_tokens = getattr(processor, "mm_tokens", None)
        image = getattr(
            processor, "IM_TOKEN_ID", getattr(mm_tokens, "image_token_id", None)
        )
        if image is None:
            image = getattr(processor, "IMAGE_TOKEN_ID", None)
        # GLM uses distinct prompt markers but the same embedding pad for both.
        video = getattr(
            processor, "VIDEO_TOKEN_ID", getattr(mm_tokens, "video_token_id", None)
        )
        if video is None:
            video = getattr(processor, "video_slot_token_id", None)
        audio = getattr(processor, "audio_token_id", None)
        if audio is None:
            audio = getattr(mm_tokens, "audio_token_id", None)
        if image is None and audio is None:
            raise ValueError("staged encoder processor has no media placeholder")
        return cls(
            image,
            video,
            accepts_expanded=getattr(processor, "accepts_staged_expanded_spans", False),
            audio=audio,
        )

    def counts(self, ids):
        if (
            self.accepts_expanded
            or self.alternatives
            or (self.image is not None and self.image == self.video)
        ):
            # Expanded spans can contain many pads. The native layout checks
            # their offsets and counts after resolving the model's delimiters.
            return None, None, None
        videos = ids.count(self.video) if self.video is not None else 0
        images = ids.count(self.image) if self.image is not None else 0
        audios = ids.count(self.audio) if self.audio is not None else 0
        return images, videos, audios

    def contains_media(self, ids):
        if any(
            marker is not None and marker in ids
            for marker in (self.image, self.video, self.audio)
        ):
            return True
        return any(
            ids[i] == tokens[0] and tuple(ids[i : i + len(tokens)]) == tokens
            for tokens in self.alternatives
            if tokens
            for i in range(len(ids) - len(tokens) + 1)
        )


class ImageLayout(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    t: int
    h: int
    w: int
    identity: str
    embedding_rows: int = 0
    original_size: list[int] = msgspec.field(default_factory=list)


class VideoLayout(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    t: int
    h: int
    w: int
    identities: list[str]
    timestamps: list[float] = msgspec.field(default_factory=list)
    second_per_grid_ts: float | None = None
    embedding_rows: int = 0


class AudioLayout(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    identity: str
    feature_length: int
    embedding_rows: int
    embedding_width: int


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

    native_model_type: str = ""
    embedding_width: int = 0
    audios: list[AudioLayout] = msgspec.field(default_factory=list)
    instance_id: str = ""
    signature: str = ""

    @classmethod
    def parse(
        cls,
        data,
        placeholder_count: int | None,
        video_count: int | None = 0,
        audio_count: int | None = 0,
    ) -> "EncoderHandoff":
        transfer = msgspec.json.decode(data, type=cls)
        native = transfer.version in (3, 4)
        if native:
            if (
                bool(transfer.audios) != (transfer.version == 4)
                or not transfer.native_model_type
                or not 0 < transfer.embedding_width <= 0xFFFFFFFF
            ):
                raise ValueError(
                    "invalid native staged model, modality or embedding width"
                )
        elif (
            transfer.version != (2 if transfer.videos else 1)
            or transfer.audios
            or transfer.native_model_type
            or transfer.embedding_width
        ):
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
            (not transfer.images and not transfer.videos and not transfer.audios)
            or len(transfer.images) != transfer.image_count
            or (
                placeholder_count is not None
                and transfer.image_count != placeholder_count
            )
            or (video_count is not None and len(transfer.videos) != video_count)
            or (audio_count is not None and len(transfer.audios) != audio_count)
        ):
            raise ValueError(
                "staged media counts differ from layout or prompt placeholders"
            )
        for layout in (*transfer.images, *transfer.videos):
            if layout.embedding_rows < 0 or (layout.embedding_rows > 0) != native:
                raise ValueError("invalid staged embedding row count")
        for audio in transfer.audios:
            if (
                not re.fullmatch("[0-9a-f]{64}", audio.identity)
                or not 0 < audio.embedding_rows <= 0xFFFFFFFF
                or not 0 < audio.feature_length <= 0xFFFFFFFF
                or not 0 < audio.embedding_width <= transfer.embedding_width
                or (
                    not (transfer.images or transfer.videos)
                    and audio.embedding_width != transfer.embedding_width
                )
            ):
                raise ValueError("invalid staged audio length, width or identity")
        for image in transfer.images:
            if image.original_size and (
                not native
                or len(image.original_size) != 2
                or min(image.original_size) < 1
            ):
                raise ValueError("invalid staged original image dimensions")
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

    @property
    def packed_modality(self):
        from sglang.srt.multimodal.modality import Modality

        if self.images:
            return Modality.IMAGE
        return Modality.VIDEO if self.videos else Modality.AUDIO

    def grids(self):
        return [[image.t, image.h, image.w] for image in self.images]

    def build_layout(self, processor, prompt, max_length, hidden_size=None):
        model = getattr(processor, "model_type", None) or processor.hf_config.model_type
        if self.version >= 3 and model != self.native_model_type:
            raise ValueError("staged encoder and consumer model types differ")
        layout = getattr(processor, "staged_layouts", {}).get(model)
        if layout is None or (
            (self.images and not layout.images)
            or (self.videos and layout.video is None)
            or (self.audios and not layout.audio)
        ):
            raise ValueError(f"staged media unsupported for {model}")
        if self.version < 3 and not layout.legacy_vision:
            raise ValueError("native encoder requires explicit embedding dimensions")
        if self.version >= 3 and hidden_size is not None:
            width = layout.embedding_width(
                processor, hidden_size, bool(self.images or self.videos)
            )
            if self.embedding_width != width or any(
                audio.embedding_width != hidden_size for audio in self.audios
            ):
                raise ValueError(
                    "staged embedding width differs from the language model"
                )
        return layout.build(processor, self, prompt, max_length)

    def bind_identities(self, mm_items, routing_hashes, *, use_routing_hash=True):
        from sglang.srt.environ import envs
        from sglang.srt.managers.req_types import MultimodalDataItem
        from sglang.srt.multimodal.cache.digest import structured_digest
        from sglang.srt.multimodal.modality import Modality

        image_index = 0
        audio_iter = iter(self.audios)
        media_index = -1
        hints = routing_hashes or ()
        all_media_hints = len(hints) == self.image_count + len(self.videos) + len(
            self.audios
        )
        video_segments = iter(
            (video, index, identity)
            for video in self.videos
            for index, identity in enumerate(video.identities)
        )
        if len(mm_items) != self.image_count + len(self.audios) + sum(
            len(v.identities) for v in self.videos
        ):
            raise ValueError("staged media count differs from rebuilt prompt")
        for item in mm_items:
            hint = None
            if self.videos and item.modality == Modality.VIDEO:
                segment = next(video_segments, None)
                if segment is None:
                    raise ValueError("too many staged video segments in prompt")
                layout, index, raw_identity = segment
                if index == 0:
                    media_index += 1
                identity = bytes.fromhex(raw_identity)
            elif item.modality == Modality.AUDIO:
                layout = next(audio_iter, None)
                if layout is None:
                    raise ValueError("too many staged audio placeholders in prompt")
                identity = bytes.fromhex(layout.identity)
                index = 0
                media_index += 1
            else:
                if item.modality != Modality.IMAGE or image_index >= self.image_count:
                    raise ValueError("unexpected staged image placeholder in prompt")
                layout = self.images[image_index]
                identity = bytes.fromhex(layout.identity)
                index = 0
                if len(hints) == self.image_count:
                    hint = hints[image_index]
                image_index += 1
                media_index += 1
            if isinstance(layout, AudioLayout):
                cache_identity = structured_digest(
                    (
                        "staged-audio-v4",
                        self.native_model_type,
                        identity,
                        layout.feature_length,
                        layout.embedding_rows,
                        layout.embedding_width,
                    )
                )
            elif self.version >= 3:
                # The encoder hashes tensor bytes. Both P and D also bind the
                # native layout, which can affect the prompt for identical bytes.
                if isinstance(layout, VideoLayout):
                    timing = (
                        layout.timestamps[index]
                        if len(layout.identities) > 1
                        else tuple(layout.timestamps)
                    )
                else:
                    timing = tuple(layout.original_size)
                cache_identity = structured_digest(
                    (
                        "staged-native-v3",
                        self.native_model_type,
                        item.modality.name,
                        identity,
                        layout.t,
                        layout.h,
                        layout.w,
                        index,
                        timing,
                    )
                )
            elif isinstance(layout, VideoLayout):
                cache_identity = structured_digest(
                    (
                        "staged-video-v2",
                        identity,
                        layout.t,
                        layout.h,
                        layout.w,
                        index,
                        layout.timestamps[index] if layout.timestamps else None,
                        layout.second_per_grid_ts,
                    )
                )
            else:
                cache_identity = identity
            if all_media_hints:
                hint = hints[media_index]
            if (
                item.precomputed_embeddings is not None
                and not envs.SGLANG_MM_SKIP_COMPUTE_HASH.get()
            ):
                payload = item
                if self.version >= 3:
                    payload = MultimodalDataItem(
                        modality=item.modality,
                        precomputed_embeddings=item.precomputed_embeddings,
                    )
                if payload.cache_key != identity:
                    raise ValueError("staged media embedding identity mismatch")
            item.set_identity(cache_identity)
            if hint is not None and use_routing_hash:
                item.hash = hint

    def native_transfer(self):
        from sglang.encoder.v1.encoder_pb2 import StagedImageTransfer

        fields = msgspec.to_builtins(self)
        del fields["peer"], fields["signature"], fields["instance_id"]
        if not self.videos:
            del fields["videos"]
        if not self.audios:
            del fields["audios"]
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
            issuer = spec.instance_id
            if self.instance_id and issuer.hex() != self.instance_id:
                raise ValueError("NIXL Describe came from a different encoder instance")
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
                # An RPC timeout cannot revoke an already submitted NIXL WRITE.
                # Return the request error now; the receiver owns the destination.
                target.abandon()
                task = asyncio.create_task(
                    self._refence_target(target, issuer, timeout)
                )
                _fence_tasks.add(task)
                task.add_done_callback(_fence_tasks.discard)
                raise
            target.close()
            return tensor

    async def _refence_target(self, target, issuer: bytes, timeout: float):
        import grpc

        delay = 0.25
        deadline = asyncio.get_running_loop().time() + timeout
        try:
            async with asyncio.timeout_at(deadline):
                while not target.reaped():
                    try:
                        ack = await self.release(
                            deadline - asyncio.get_running_loop().time()
                        )
                    except grpc.RpcError:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 5)
                        continue
                    # A restarted E also acknowledges unknown transfers. Its
                    # acknowledgement says nothing about the previous E's WRITE.
                    if len(issuer) == 16 and ack.instance_id == issuer:
                        target.close()
                        return
                    break
        except TimeoutError:
            pass
        finally:
            if not target.reaped():
                target.quarantine()
                logger.warning(
                    "Quarantined NIXL target for transfer %s after losing its encoder fence",
                    self.transfer_id,
                )

    async def release(self, timeout: float):
        import grpc

        from sglang.encoder.v1 import encoder_pb2, encoder_pb2_grpc

        async with grpc.aio.insecure_channel(urlsplit(self.peer).netloc) as channel:
            stub = encoder_pb2_grpc.SglangEncoderStub(channel)
            return await stub.Release(
                encoder_pb2.ReleaseRequest(transfer_id=self.transfer_id),
                timeout=timeout,
            )
