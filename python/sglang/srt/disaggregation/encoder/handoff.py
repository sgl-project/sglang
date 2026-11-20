"""Private, single-receiver multimodal handoffs."""

import asyncio
import math
import re
from urllib.parse import urlsplit

import msgspec

from sglang.srt.disaggregation.encoder.native_handoff import (
    FRAME_VIDEO_MODEL_TYPES,
    NATIVE_IMAGE_MODEL_TYPES,
    NATIVE_VIDEO_MODEL_TYPES,
    STAGED_AUDIO_MODEL_TYPES,
    STAGED_VIDEO_MODEL_TYPES,
    TIMED_VIDEO_MODEL_TYPES,
)

# Keep one recovery task per ambiguous registered destination until the original
# write completes or E fences it. The receiver's admission limit bounds this set
# for each scheduler, including when an encoder remains unreachable.
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
        if hasattr(processor, "image_placeholder_ids"):
            # Muse also accepts expanded spans and literal media placeholders.
            return cls(
                processor.patch_id,
                processor.video_id,
                (
                    tuple(processor.image_placeholder_ids),
                    tuple(processor.video_placeholder_ids),
                    tuple(getattr(processor, "audio_placeholder_ids", None) or ()),
                ),
                audio=getattr(processor, "audio_id", None),
            )
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
        if transfer.audios:
            if (
                transfer.version != 4
                or transfer.native_model_type not in STAGED_AUDIO_MODEL_TYPES
                or not 0 < transfer.embedding_width <= 0xFFFFFFFF
                or (
                    transfer.images
                    and transfer.native_model_type
                    not in NATIVE_IMAGE_MODEL_TYPES | STAGED_VIDEO_MODEL_TYPES
                )
                or (
                    transfer.videos
                    and transfer.native_model_type
                    not in NATIVE_VIDEO_MODEL_TYPES | STAGED_VIDEO_MODEL_TYPES
                )
            ):
                raise ValueError("invalid staged audio model or layout version")
        elif transfer.version == 3:
            if (
                transfer.native_model_type not in NATIVE_IMAGE_MODEL_TYPES
                or (
                    transfer.videos
                    and transfer.native_model_type not in NATIVE_VIDEO_MODEL_TYPES
                )
                or transfer.embedding_width <= 0
            ):
                raise ValueError("invalid native staged model or embedding width")
        elif (
            transfer.version != (2 if transfer.videos else 1)
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
            if (
                transfer.native_model_type in {"kimi_k3", "kimi_k31"}
                and not image.original_size
            ):
                raise ValueError("Kimi K3 requires original image dimensions")
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

    @property
    def packed_modality(self):
        from sglang.srt.multimodal.modality import Modality

        if self.images:
            return Modality.IMAGE
        return Modality.VIDEO if self.videos else Modality.AUDIO

    def grids(self):
        return [[image.t, image.h, image.w] for image in self.images]

    def _native_layout(self, processor, prompt, max_length):
        import torch

        model = self.native_model_type
        if getattr(processor, "model_type", processor.hf_config.model_type) != model:
            raise ValueError("staged encoder and consumer model types differ")
        if (
            sum(x.embedding_rows for x in (*self.images, *self.videos, *self.audios))
            >= max_length
        ):
            raise ValueError("staged media exceed the model's input length limit")
        for layout in (*self.images, *self.videos):
            if model in {"kimi_vl", "kimi_k25", "kimi_k3", "kimi_k31"}:
                expected = processor._num_image_tokens_from_grid(
                    [layout.t, layout.h, layout.w]
                )
            elif model in {"inkling2", "muse_spark"}:
                expected = layout.t * layout.h * layout.w
            else:
                if model in {"mimo_v2", "qwen3_omni_moe"}:
                    merge = processor.spatial_merge_size
                elif model == "dots3_note":
                    merge = processor.image_preprocessor.merge_size
                else:
                    component = (
                        processor._processor.video_processor
                        if isinstance(layout, VideoLayout)
                        else processor._processor.image_processor
                    )
                    merge = component.merge_size
                if layout.h % merge or layout.w % merge:
                    raise ValueError("staged media grid is incompatible with the model")
                frames = layout.t
                if model == "mimo_v2" and isinstance(layout, VideoLayout):
                    ratio = processor.mimo_processor.temporal_compression_ratio
                    if frames % ratio:
                        raise ValueError(
                            "staged grid differs from native temporal compression"
                        )
                    frames //= ratio
                expected = frames * (layout.h // merge) * (layout.w // merge)
            if expected != layout.embedding_rows:
                raise ValueError("staged embedding rows differ from model grid")
        kwargs = {}
        if self.audios:
            # Qwen descriptors carry the input mel-frame lengths. Other native
            # processors use the audio encoder's output sequence lengths.
            lengths = torch.tensor([a.feature_length for a in self.audios])
            if model in {"qwen3_asr", "qwen3_omni_moe"}:
                from sglang.srt.multimodal.mrope_rope_index import (
                    _get_feat_extract_output_lengths,
                )

                rows = _get_feat_extract_output_lengths(lengths)
            elif model == "qwen2_audio":
                rows = (lengths + 1) // 4
            else:
                rows = lengths
            if rows.tolist() != [a.embedding_rows for a in self.audios]:
                raise ValueError("audio feature length differs from its staged rows")
            kwargs["audio_feature_lens"] = lengths
        if self.images:
            grids = self.grids()
            if model == "kimi_vl":
                if any(t != 1 for t, _, _ in grids):
                    raise ValueError(
                        "Kimi-VL image grids require a single temporal patch"
                    )
                grids = [grid[1:] for grid in grids]
            kwargs["img_grid_thw"] = torch.tensor(grids)
            if model in {"kimi_k3", "kimi_k31"}:
                kwargs["original_image_sizes"] = [
                    img.original_size for img in self.images
                ]
        if self.videos:
            for video in self.videos:
                if model == "qwen3_omni_moe":
                    if (
                        len(video.identities) != 1
                        or video.timestamps
                        or video.second_per_grid_ts is None
                    ):
                        raise ValueError(
                            "staged video metadata differs from native model layout"
                        )
                    continue
                frames = video.t
                if model == "mimo_v2":
                    frames //= processor.mimo_processor.temporal_compression_ratio
                if (
                    len(video.identities) != (frames if model == "mimo_v2" else 1)
                    or (
                        len(video.timestamps) != frames
                        and (model != "minimax_m3_vl" or bool(video.timestamps))
                    )
                    or video.second_per_grid_ts is not None
                ):
                    raise ValueError(
                        "staged video metadata differs from native model layout"
                    )
            kwargs["video_grid_thw"] = torch.tensor(
                [[v.t, v.h, v.w] for v in self.videos]
            )
            if model == "qwen3_omni_moe":
                kwargs["second_per_grid_ts"] = torch.tensor(
                    [v.second_per_grid_ts for v in self.videos], dtype=torch.float32
                )
            else:
                kwargs["video_timestamps"] = (
                    [t for video in self.videos for t in video.timestamps]
                    if model == "mimo_v2"
                    else [video.timestamps or None for video in self.videos]
                )
        # All native processors consume the same grid/timing inputs as ordinary EPD.
        staged_layout = getattr(processor, "get_staged_media_layout", None)
        if staged_layout is None:
            mm = processor.get_mm_data(list(prompt), embeddings=None, **kwargs)
        else:
            mm = staged_layout(list(prompt), **kwargs)
        from sglang.srt.multimodal.modality import Modality

        for modality, layouts in (
            (Modality.IMAGE, self.images),
            (Modality.VIDEO, self.videos),
            (Modality.AUDIO, self.audios),
        ):
            items = [item for item in mm.mm_items if item.modality == modality]
            rows = (
                [
                    layout.embedding_rows // len(layout.identities)
                    for layout in layouts
                    for _ in layout.identities
                ]
                if modality == Modality.VIDEO
                else [layout.embedding_rows for layout in layouts]
            )
            if len(items) != len(rows) or any(
                not item.offsets
                or sum(end - start + 1 for start, end in item.offsets) != count
                for item, count in zip(items, rows)
            ):
                raise ValueError("staged media rows differ from rebuilt native prompt")
        # Native processors may group items by modality. Routing hints and cache
        # event labels arrive in media order across modalities.
        mm.mm_items.sort(key=lambda item: item.offsets[0][0])
        if len(mm.input_ids) >= max_length:
            raise ValueError("staged media exceed the model's input length limit")
        return mm

    def build_layout(self, processor, prompt, max_length):
        import torch

        if self.version >= 3:
            return self._native_layout(processor, prompt, max_length)
        if (
            getattr(getattr(processor, "hf_config", None), "model_type", None)
            in NATIVE_IMAGE_MODEL_TYPES
        ):
            raise ValueError("native encoder requires a version 3 handoff")
        self.validate_layout(processor.spatial_merge_size, len(prompt), max_length)
        kwargs = {}
        if self.images:
            kwargs["img_grid_thw"] = torch.tensor(self.grids())
        if self.videos:
            if processor.model_type not in STAGED_VIDEO_MODEL_TYPES:
                raise ValueError(
                    f"staged video is unsupported for {processor.model_type}"
                )
            frame_items = processor.model_type in FRAME_VIDEO_MODEL_TYPES
            timed_grid = processor.model_type in TIMED_VIDEO_MODEL_TYPES
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
            if hint is not None and self.native_model_type not in {
                "kimi_k25",
                "kimi_k3",
            }:
                item.hash = hint

    def native_transfer(self):
        from sglang.encoder.v1.encoder_pb2 import StagedImageTransfer

        fields = msgspec.to_builtins(self)
        del fields["peer"]
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
