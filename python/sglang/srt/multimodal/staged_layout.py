"""Processor contract for reconstructing a prompt from externally encoded media."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from itertools import pairwise


class VideoTiming(Enum):
    NONE = "none"
    TIMESTAMPS = "timestamps"
    OPTIONAL_TIMESTAMPS = "optional_timestamps"
    GRID_DURATION = "grid_duration"


@dataclass(frozen=True)
class StagedMediaLayout:
    """Declare supported media and how native encoder metadata maps to processor inputs.

    A processor exports a `staged_layouts` mapping from HF model type to this
    contract. The encoder and the language worker use the same declaration.
    New processors use explicit tensor dimensions on the existing native wire.
    """

    images: bool = True
    video: VideoTiming | None = None
    audio: bool = False
    per_frame_items: bool = False
    original_image_sizes: bool = False
    use_routing_hash: bool = True
    legacy_vision: bool = False

    # Bump when a processor changes the interpretation of an existing wire field.
    wire_revision: int = 1

    def wire_id(self, model_type):
        """Identify the descriptor layout shared by E, P/PD and D at discovery."""
        return json.dumps(
            {
                "model_type": model_type,
                "revision": self.wire_revision,
                "legacy_vision": self.legacy_vision,
                "images": self.images,
                "video": self.video.value if self.video is not None else None,
                "audio": self.audio,
                "per_frame_items": self.per_frame_items,
                "original_image_sizes": self.original_image_sizes,
                "use_routing_hash": self.use_routing_hash,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    def encoder_vision_enabled(self, encoder):
        return True

    def encoder_audio_enabled(self, encoder):
        return self.audio

    def language_audio_enabled(self, scheduler):
        return self.audio

    def vision_merge_size(self, processor, video):
        component = (
            processor._processor.video_processor
            if video
            else processor._processor.image_processor
        )
        return component.merge_size

    def video_compression(self, processor):
        return 1

    def encoder_video_metadata(self, encoder, data):
        """Per-clip timestamps and the number of input frames per output frame."""
        return getattr(data, "video_timestamps", None), 1

    def vision_rows(self, processor, layout, video):
        merge = self.vision_merge_size(processor, video)
        compression = self.video_compression(processor) if video else 1
        if layout.h % merge or layout.w % merge or layout.t % compression:
            raise ValueError("staged media grid is incompatible with the model")
        return (layout.t // compression) * (layout.h // merge) * (layout.w // merge)

    def audio_rows(self, lengths):
        return lengths

    def embedding_width(self, processor, hidden_size, visual):
        return hidden_size

    def processor_image_grids(self, grids):
        return grids

    def encode_video_layouts(
        self, encoder, data, item_embeddings, identity, explicit_rows
    ):
        from sglang.encoder.v1.encoder_pb2 import VideoLayout

        timestamps, compression = self.encoder_video_metadata(encoder, data)
        seconds = getattr(data, "second_per_grid_ts", None)
        count = len(data.get_grid())
        timed = self.video in (VideoTiming.TIMESTAMPS, VideoTiming.OPTIONAL_TIMESTAMPS)
        if timed and (self.video is VideoTiming.TIMESTAMPS or timestamps is not None):
            if timestamps is None or len(timestamps) != count:
                raise ValueError("Missing native timestamps for staged video")
        if self.video is VideoTiming.GRID_DURATION and (
            seconds is None or len(seconds) != count
        ):
            raise ValueError("Missing native grid duration for staged video")
        layouts = []
        for i, (grid, tensor) in enumerate(item_embeddings(data)):
            t, h, w = map(int, grid)
            if min(t, h, w, compression) <= 0 or t % compression:
                raise ValueError("Invalid staged video grid")
            frames = t // compression
            if tensor.shape[0] % frames:
                raise ValueError("Invalid staged video embedding length")
            layout = VideoLayout(t=t, h=h, w=w)
            if explicit_rows:
                layout.embedding_rows = tensor.shape[0]
            if timed and timestamps is not None and timestamps[i] is not None:
                times = [float(x) for x in timestamps[i]]
                if (
                    len(times) != frames
                    or any(not math.isfinite(x) or x < 0 for x in times)
                    or any(a > b for a, b in pairwise(times))
                ):
                    raise ValueError("Invalid staged video frame timestamps")
                layout.timestamps.extend(times)
            elif self.video is VideoTiming.TIMESTAMPS:
                raise ValueError("Missing native timestamps for staged video")
            if self.video is VideoTiming.GRID_DURATION:
                duration = float(seconds[i])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("Invalid staged video grid duration")
                layout.second_per_grid_ts = duration
            chunks = tensor.tensor_split(frames) if self.per_frame_items else (tensor,)
            layout.identities.extend(identity(chunk) for chunk in chunks)
            layouts.append(layout)
        return layouts

    def processor_video_timestamps(self, videos):
        return [v.timestamps or None for v in videos]

    def build(self, processor, handoff, prompt, max_length):
        import torch

        from sglang.srt.multimodal.modality import Modality

        kwargs = {}
        explicit = handoff.version >= 3
        image_rows = [self.vision_rows(processor, i, False) for i in handoff.images]
        video_rows = [self.vision_rows(processor, v, True) for v in handoff.videos]
        audio_rows = []
        if handoff.audios:
            lengths = torch.tensor([a.feature_length for a in handoff.audios])
            audio_rows = self.audio_rows(lengths).tolist()
            if audio_rows != [a.embedding_rows for a in handoff.audios]:
                raise ValueError("audio feature length differs from its staged rows")
            kwargs["audio_feature_lens"] = lengths
        if explicit and (
            image_rows != [i.embedding_rows for i in handoff.images]
            or video_rows != [v.embedding_rows for v in handoff.videos]
        ):
            raise ValueError("staged embedding rows differ from model grid")
        rows = sum(image_rows) + sum(video_rows) + sum(audio_rows)
        min_length = (
            rows if explicit else len(prompt) - len(image_rows) - len(video_rows) + rows
        )
        if min_length >= max_length:
            raise ValueError("staged media exceed the model's input length limit")
        if handoff.images:
            kwargs["img_grid_thw"] = torch.tensor(
                self.processor_image_grids(handoff.grids())
            )
            if self.original_image_sizes:
                if any(not i.original_size for i in handoff.images):
                    raise ValueError(
                        "staged processor requires original image dimensions"
                    )
                kwargs["original_image_sizes"] = [
                    i.original_size for i in handoff.images
                ]
        if handoff.videos:
            for v in handoff.videos:
                frames = v.t // self.video_compression(processor)
                timestamps = self.video in (
                    VideoTiming.TIMESTAMPS,
                    VideoTiming.OPTIONAL_TIMESTAMPS,
                )
                if (
                    len(v.identities) != (frames if self.per_frame_items else 1)
                    or (v.second_per_grid_ts is not None)
                    != (self.video is VideoTiming.GRID_DURATION)
                    or (
                        len(v.timestamps) != (frames if timestamps else 0)
                        and not (
                            self.video is VideoTiming.OPTIONAL_TIMESTAMPS
                            and not v.timestamps
                        )
                    )
                ):
                    raise ValueError("staged video metadata differs from model layout")
            kwargs["video_grid_thw"] = torch.tensor(
                [[v.t, v.h, v.w] for v in handoff.videos]
            )
            if self.video is VideoTiming.GRID_DURATION:
                kwargs["second_per_grid_ts"] = torch.tensor(
                    [v.second_per_grid_ts for v in handoff.videos], dtype=torch.float32
                )
            elif self.video in (
                VideoTiming.TIMESTAMPS,
                VideoTiming.OPTIONAL_TIMESTAMPS,
            ):
                kwargs["video_timestamps"] = self.processor_video_timestamps(
                    handoff.videos
                )

        staged_layout = getattr(processor, "get_staged_media_layout", None)
        if staged_layout is None:
            mm = processor.get_mm_data(list(prompt), embeddings=None, **kwargs)
        else:
            mm = staged_layout(list(prompt), **kwargs)
        if len(mm.input_ids) >= max_length:
            raise ValueError("staged media exceed the model's input length limit")
        if explicit:
            video_items = [
                rows // len(v.identities)
                for v, rows in zip(handoff.videos, video_rows)
                for _ in v.identities
            ]
            for modality, expected in (
                (Modality.IMAGE, image_rows),
                (Modality.VIDEO, video_items),
                (Modality.AUDIO, audio_rows),
            ):
                actual = [
                    sum(end - start + 1 for start, end in item.offsets)
                    if item.offsets
                    else 0
                    for item in mm.mm_items
                    if item.modality == modality
                ]
                if actual != expected:
                    raise ValueError(
                        "staged media rows differ from rebuilt native prompt"
                    )
            mm.mm_items.sort(key=lambda item: item.offsets[0][0])
        return mm


@lru_cache(maxsize=1)
def _registered_layouts(package):
    from sglang.srt.managers.multimodal_processor import (
        PROCESSOR_MAPPING,
        import_processors,
    )

    import_processors("sglang.srt.multimodal.processors")
    if package:
        import_processors(package, overwrite=True)
    layouts = {}
    for cls in PROCESSOR_MAPPING.values():
        layouts.update(getattr(cls, "staged_layouts", {}))
    return layouts


def staged_layout_for(model_type):
    from sglang.srt.environ import envs

    return _registered_layouts(envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get()).get(
        model_type
    )
