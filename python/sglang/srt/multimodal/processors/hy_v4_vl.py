import asyncio
import re
from typing import List, Optional, Union

import torch
from transformers.video_utils import VideoMetadata

from sglang.srt.configs.hy_v4_flash import HYV4FlashConfig
from sglang.srt.configs.hy_v4_vl_processing import (
    vision_token_count,
    vision_token_grid,
)
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.models.hunyuan_v4_flash import HYV4FlashForConditionalGeneration
from sglang.srt.multimodal.modality import Modality, MultimodalInputFormat
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper

VIDEO_SAMPLING_KEYS = ("num_frames", "fps", "min_frames", "max_frames")
IMAGE_MARKER = "<image>"
VIDEO_MARKER = "<video>"
PARTIAL_PLACEHOLDER_ERROR = (
    "HYV4 VL prompt has a partial or already-expanded media placeholder"
)


def get_hy_v4_vl_mrope_positions(
    input_ids: List[int],
    image_grid_thw: List[List[int]],
    video_grid_thw: List[List[int]],
    hf_config: HYV4FlashConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.tensor(input_ids, dtype=torch.long)
    positions = torch.arange(len(input_ids)).repeat(3, 1)
    if not input_ids or not (image_grid_thw or video_grid_thw):
        return positions, torch.zeros((1, 1), dtype=torch.long)
    vision = hf_config.vision_config
    extra = int(bool(vision.cat_extra_token))
    pad_ids = (hf_config.image_token_id, hf_config.video_token_id)
    starts = torch.where(tokens == hf_config.image_start_token_id)[0].tolist()
    ends = torch.where(tokens == hf_config.image_end_token_id)[0].tolist()
    if len(starts) != len(ends):
        raise ValueError("HYV4 image/frame start and end tokens must be paired")
    start_pos = start_index = mm_index = scan = covered = 0
    image_index = video_index = -1
    inside_video = False
    while mm_index < len(starts):
        marker = starts[mm_index]
        pos = marker + 1 + extra
        while scan < marker:
            if input_ids[scan] == hf_config.video_start_token_id:
                inside_video = True
            elif input_ids[scan] == hf_config.video_end_token_id:
                inside_video = False
            scan += 1
        is_video = inside_video or (
            image_index + 1 >= len(image_grid_thw)
            and video_index + 1 < len(video_grid_thw)
        )
        if start_pos < pos:
            positions[:, start_pos:pos] = torch.arange(pos - start_pos) + start_index
            start_index += pos - start_pos
        if is_video:
            video_index += 1
            if video_index >= len(video_grid_thw):
                raise ValueError("Video placeholders do not match video_grid_thw")
            frames, rows, cols = vision_token_grid(
                video_grid_thw[video_index], vision, "video"
            )
        else:
            image_index += 1
            if image_index >= len(image_grid_thw):
                raise ValueError("Image placeholders do not match image_grid_thw")
            frames, rows, cols = vision_token_grid(
                image_grid_thw[image_index], vision, "image"
            )
        if mm_index + frames > len(starts):
            raise ValueError("Missing HYV4 image/frame boundary tokens")
        count = rows * cols
        for frame in range(frames):
            begin = starts[mm_index + frame] + 1 + extra
            end = begin + count
            pad_id = pad_ids[int(is_video)]
            if end + extra != ends[mm_index + frame] or not torch.all(
                tokens[begin - extra : end + extra] == pad_id
            ):
                raise ValueError(
                    "HYV4 visual placeholders do not match frame boundaries and grid"
                )
            covered += count + 2 * extra
            positions[0, begin:end] = start_index
            positions[1, begin:end] = (
                torch.arange(rows).repeat_interleave(cols) + start_index
            )
            positions[2, begin:end] = torch.arange(cols).repeat(rows) + start_index
            start_index += max(rows, cols)
            if is_video and frame + 1 < frames:
                bridge_start = ends[mm_index + frame]
                bridge_end = starts[mm_index + frame + 1] + 1
                positions[:, bridge_start:bridge_end] = (
                    torch.arange(bridge_end - bridge_start) + start_index
                )
                start_index += bridge_end - bridge_start
        start_pos = ends[mm_index + frames - 1] if is_video else pos + count
        mm_index += frames
    if start_pos < len(input_ids):
        positions[:, start_pos:] = (
            torch.arange(len(input_ids) - start_pos) + start_index
        )
    if image_index + 1 != len(image_grid_thw) or video_index + 1 != len(video_grid_thw):
        raise ValueError("Unused HYV4 media grids after position construction")
    if int(torch.isin(tokens, torch.tensor(pad_ids)).sum()) != covered:
        raise ValueError("HYV4 visual placeholders lie outside image/frame boundaries")
    delta = int(positions.max()) + 1 - len(input_ids)
    return positions, torch.tensor([[delta]], dtype=torch.long)


def _collect_grids(
    mm_items: List[MultimodalDataItem], modality: Modality, key: str
) -> List[List[int]]:
    grids = []
    for item in mm_items:
        if item.modality != modality:
            continue
        grid = item.model_specific_data.get(key)
        if grid is None:
            raise ValueError(f"HYV4 VL {modality.name} item is missing {key}")
        grids.extend(torch.as_tensor(grid).reshape(-1, 3).tolist())
    return grids


class HYV4VLMultimodalProcessor(SGLangBaseProcessor):
    models = [HYV4FlashForConditionalGeneration]
    video_preprocessing_device = "cpu"

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_placeholder = _processor.get_placeholder("image")
        self.video_placeholder = _processor.get_placeholder("video")
        # Accept both the chat-template triple and the bare `<image>` marker.
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_placeholder,
            image_token_id=hf_config.image_token_id,
            image_token_regex=re.compile(
                f"{re.escape(self.image_placeholder)}|{re.escape(IMAGE_MARKER)}"
            ),
            video_token=self.video_placeholder,
            video_token_id=hf_config.video_token_id,
            video_token_regex=re.compile(
                f"{re.escape(self.video_placeholder)}|{re.escape(VIDEO_MARKER)}"
            ),
        ).build(_processor)
        image_triple = [
            hf_config.image_start_token_id,
            hf_config.image_token_id,
            hf_config.image_end_token_id,
        ]
        video_triple = [
            hf_config.video_start_token_id,
            hf_config.video_token_id,
            hf_config.video_end_token_id,
        ]
        self.raw_media_token_ids = frozenset(image_triple + video_triple)
        self.raw_media_tokens = frozenset(
            self._tokenizer.convert_ids_to_tokens(sorted(self.raw_media_token_ids))
        )
        self.id_placeholder_patterns = [
            (image_triple, image_triple, Modality.IMAGE),
            (video_triple, video_triple, Modality.VIDEO),
        ]
        for marker, triple, modality in (
            (IMAGE_MARKER, image_triple, Modality.IMAGE),
            (VIDEO_MARKER, video_triple, Modality.VIDEO),
        ):
            marker_ids = self._tokenizer.encode(marker, add_special_tokens=False)
            if marker_ids:
                self.id_placeholder_patterns.append((marker_ids, triple, modality))

    def _placeholder_text(self, modality: Modality) -> str:
        if modality == Modality.IMAGE:
            return self.image_placeholder
        return self.video_placeholder

    def _normalize_text_prompt(self, text: str, rid) -> tuple[str, dict]:
        counts = {Modality.IMAGE: 0, Modality.VIDEO: 0}
        parts = []
        for index, part in enumerate(self.mm_tokens.get_combined_regex().split(text)):
            if index % 2:
                modality = self.mm_tokens.get_modality_of_token(part)
                counts[modality] += 1
                parts.append(self._placeholder_text(modality))
            elif any(token in part for token in self.raw_media_tokens):
                raise ValueError(f"{PARTIAL_PLACEHOLDER_ERROR}, {rid=}")
            else:
                parts.append(part)
        return "".join(parts), counts

    def _normalize_id_prompt(self, input_ids: List[int], rid) -> tuple[str, list, dict]:
        counts = {Modality.IMAGE: 0, Modality.VIDEO: 0}
        canonical_ids, parts, text_run = [], [], []
        index = 0
        while index < len(input_ids):
            for pattern, triple, modality in self.id_placeholder_patterns:
                if input_ids[index : index + len(pattern)] == pattern:
                    break
            else:
                if input_ids[index] in self.raw_media_token_ids:
                    raise ValueError(f"{PARTIAL_PLACEHOLDER_ERROR}, {rid=}")
                text_run.append(input_ids[index])
                canonical_ids.append(input_ids[index])
                index += 1
                continue
            parts.append(self._decode_text_run(text_run))
            parts.append(self._placeholder_text(modality))
            canonical_ids.extend(triple)
            counts[modality] += 1
            text_run = []
            index += len(pattern)
        parts.append(self._decode_text_run(text_run))
        return "".join(parts), canonical_ids, counts

    def _decode_text_run(self, token_ids: List[int]) -> str:
        # Decoded per run: special tokens decoded together can gain separators.
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def _canonicalize_prompt(
        self, input_text, num_images: int, num_videos: int, rid
    ) -> tuple[str, Optional[List[int]]]:
        canonical_ids = None
        if isinstance(input_text, list):
            text, canonical_ids, counts = self._normalize_id_prompt(
                input_ids=input_text, rid=rid
            )
        else:
            text, counts = self._normalize_text_prompt(text=input_text, rid=rid)
        if counts[Modality.IMAGE] != num_images or counts[Modality.VIDEO] != num_videos:
            raise ValueError(
                f"HYV4 VL prompt has {counts[Modality.IMAGE]} image and "
                f"{counts[Modality.VIDEO]} video placeholders for {num_images} "
                f"image(s) and {num_videos} video(s), {rid=}"
            )
        return text, canonical_ids

    def _sample_video(self, video):
        if not isinstance(video, VideoDecoderWrapper):
            return video
        total_frames, fps = len(video), video.avg_fps
        sampling = {
            key: self.video_config[key]
            for key in VIDEO_SAMPLING_KEYS
            if self.video_config.get(key) is not None
        }
        indices = self._processor.video_processor.sample_frames(
            VideoMetadata(total_num_frames=total_frames, fps=fps), **sampling
        ).tolist()
        metadata = {
            "fps": fps,
            "total_num_frames": total_frames,
            "frames_indices": indices,
            "do_sample_frames": False,
        }
        return video.get_frames_as_tensor(indices), metadata

    async def _sample_videos(self, videos: list) -> list:
        loop = asyncio.get_running_loop()
        return list(
            await asyncio.gather(
                *(
                    loop.run_in_executor(self.io_executor, self._sample_video, video)
                    for video in videos
                )
            )
        )

    def resolve_image_token_counts(self, images: List) -> List[int]:
        counts = []
        for image in images:
            height, width = (
                tuple(image.shape[-2:])
                if isinstance(image, torch.Tensor)
                else (image.height, image.width)
            )
            grid_h, grid_w = (
                self._processor.image_processor.get_number_of_image_patches(
                    height, width, self.image_config
                )
            )
            counts.append(
                vision_token_count(
                    (1, grid_h, grid_w), self.hf_config.vision_config, "image"
                )
            )
        return counts

    def _restore_hashed_pads(
        self, input_ids: List[int], mm_items: List[MultimodalDataItem]
    ) -> List[int]:
        # The scheduler fallback sees ids whose visual spans hold per-item pad values.
        input_ids = list(input_ids)
        for item in mm_items:
            pad_id = self.mm_tokens.get_token_id_by_modality(item.modality)
            if item.pad_value is None:
                continue
            pad_values = set(item.padding_values())
            for start, end in item.offsets:
                span = input_ids[start : end + 1]
                if all(token in pad_values for token in span):
                    input_ids[start : end + 1] = [pad_id] * len(span)
        return input_ids

    def compute_mrope_positions(self, input_ids, mm_items):
        return get_hy_v4_vl_mrope_positions(
            input_ids=self._restore_hashed_pads(input_ids=input_ids, mm_items=mm_items),
            image_grid_thw=_collect_grids(
                mm_items=mm_items, modality=Modality.IMAGE, key="image_grid_thw"
            ),
            video_grid_thw=_collect_grids(
                mm_items=mm_items, modality=Modality.VIDEO, key="video_grid_thw"
            ),
            hf_config=self.hf_config,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ) -> MultimodalProcessorOutput:
        if request_obj.audio_data:
            raise ValueError(
                f"HYV4 VL does not support audio inputs, rid={request_obj.rid!r}"
            )
        image_data = image_data or []
        video_data = request_obj.video_data or []
        canonical_ids = None
        if not self._all_mm_data_is_preprocessed(image_data, video_data):
            input_text, canonical_ids = self._canonicalize_prompt(
                input_text=input_text,
                num_images=len(image_data),
                num_videos=len(video_data),
                rid=request_obj.rid,
            )
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_data,
            multimodal_tokens=self.mm_tokens,
        )
        if canonical_ids is not None:
            base_output.input_ids = canonical_ids
        base_output.videos = await self._sample_videos(base_output.videos)
        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )
        if any(
            item.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING
            for item in mm_items
        ):
            raise ValueError(
                "HYV4 VL does not support precomputed image/video embeddings, "
                f"rid={request_obj.rid!r}"
            )
        input_ids = input_ids.flatten().tolist()
        mrope_positions, mrope_position_delta = self.compute_mrope_positions(
            input_ids, mm_items
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_start_id=self.hf_config.image_start_token_id,
            im_end_id=self.hf_config.image_end_token_id,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )
