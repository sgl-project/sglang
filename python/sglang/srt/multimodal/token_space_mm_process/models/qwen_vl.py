from dataclasses import replace
from types import MappingProxyType

import torch
from transformers import BatchFeature
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessorKwargs,
    _get_feat_extract_output_lengths,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)

from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.mrope_rope_index import get_rope_index
from sglang.srt.multimodal.qwen_vl_media_processing import (
    FPS_MAX_FRAMES,
    QWEN_VIDEO_PREPROCESS_CONFIG_KEYS,
    VIDEO_MAX_PIXELS,
    VIDEO_TOTAL_PIXELS,
    preprocess_video_sync,
)
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)


class QwenTokenSpaceProcessStrategy(TokenSpaceProcessStrategy):
    models = (
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForCausalLMMTP",
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen4ExpForConditionalGeneration",
    )
    mm_item_fields = MappingProxyType(
        {
            "pixel_values": Modality.IMAGE,
            "image_grid_thw": Modality.IMAGE,
            "pixel_values_videos": Modality.VIDEO,
            "video_grid_thw": Modality.VIDEO,
            "second_per_grid_ts": Modality.VIDEO,
            "input_features": Modality.AUDIO,
            "feature_attention_mask": Modality.AUDIO,
        }
    )
    mm_feature_fields = frozenset(
        {"pixel_values", "pixel_values_videos", "input_features"}
    )
    processor_class_attributes = MappingProxyType({"auto_mm_io_worker_num": 16})
    media_processor_kwargs_type = None

    # Adapted from sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor.__init__;
    # token IDs go to mm_token_expansion_params instead of MultimodalSpecialTokens.
    def __init__(self, hf_config, processor, **kwargs):
        self.model_type = hf_config.model_type
        if self.model_type == "qwen3_omni_moe":
            self.media_processor_kwargs_type = Qwen3OmniMoeProcessorKwargs
            hf_config = hf_config.thinker_config

        super().__init__(hf_config, processor, **kwargs)

        vision_config = hf_config.vision_config
        self._tokens_per_second = getattr(vision_config, "tokens_per_second", None)
        self._audio_start_token_id = getattr(hf_config, "audio_start_token_id", None)
        self._position_id_per_seconds = getattr(
            hf_config, "position_id_per_seconds", None
        )

        if not isinstance(vision_config.temporal_patch_size, int):
            raise ValueError(
                "Qwen token expansion requires an int temporal_patch_size, "
                f"got {vision_config.temporal_patch_size!r}"
            )
        self.mm_token_expansion_params = {
            "image_token_id": hf_config.image_token_id,
            "video_token_id": hf_config.video_token_id,
            "vision_start_token_id": hf_config.vision_start_token_id,
            "vision_end_token_id": hf_config.vision_end_token_id,
            "audio_token_id": getattr(hf_config, "audio_token_id", None),
            "spatial_merge_size": vision_config.spatial_merge_size,
            "temporal_patch_size": vision_config.temporal_patch_size,
            # Qwen3-VL and later prefix every frame with a timestamp and vision markers.
            "frame_timestamps": self.model_type
            not in {"qwen2_vl", "qwen2_5_vl", "qwen3_omni_moe"},
        }

    def resolve_media_options(
        self, processor, *, image_device=None, video_device=None, **kwargs
    ) -> dict:
        """Split the request kwargs per modality as the HF processor call does."""
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", True)
        kwargs_type = (
            self.media_processor_kwargs_type or processor.valid_processor_kwargs
        )
        processor_kwargs = processor._merge_kwargs(
            kwargs_type,
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
            **kwargs,
        )
        image_kwargs = processor_kwargs["images_kwargs"]
        video_kwargs = processor_kwargs["videos_kwargs"]
        audio_kwargs = processor_kwargs["audio_kwargs"]
        image_kwargs.update(self.image_config)
        video_kwargs.update(self.video_config)
        audio_kwargs.update(self.audio_config)
        if image_device is not None:
            image_kwargs["device"] = image_device
            video_kwargs.setdefault("device", image_device)
            # The legacy call's common `device` reaches audio only where HF accepts it.
            if "device" in kwargs_type.__annotations__["audio_kwargs"].__annotations__:
                audio_kwargs.setdefault("device", image_device)
        if video_device is not None:
            video_kwargs["device"] = video_device
        return {
            Modality.IMAGE: image_kwargs,
            Modality.VIDEO: video_kwargs,
            Modality.AUDIO: audio_kwargs,
        }

    def process_image(self, image, processor, preprocess_kwargs, **kwargs):
        return dict(
            processor.image_processor([image], **{**kwargs, **preprocess_kwargs})
        )

    # Adapted from the video steps of sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor._process_mm_data_uncached;
    # per item, calling the video processor directly and adding HF's per-grid seconds.
    def process_video(
        self, video, processor, preprocess_kwargs, video_metadata=None, **kwargs
    ):
        if kwargs.pop("use_audio_in_video", False):
            raise ValueError("Qwen token expansion does not support use_audio_in_video")
        kwargs.pop("seconds_per_chunk", None)
        kwargs.pop("position_id_per_seconds", None)
        video_config = {
            **self.video_config,
            **{
                key: value
                for key, value in kwargs.items()
                if key in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS
            },
            **self._bound_item_video_options(preprocess_kwargs),
        }
        processor_kwargs = dict(kwargs)
        if isinstance(processor.video_processor, Qwen3VLVideoProcessor):
            processor_kwargs.update(
                {
                    key: value
                    for key, value in video_config.items()
                    if key not in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS
                }
            )
        frames, metadata = preprocess_video_sync(video, video_config=video_config)
        if metadata is None:
            metadata = video_metadata
        if metadata is not None:
            processor_kwargs = {
                key: value
                for key, value in processor_kwargs.items()
                if key not in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS - {"fps"}
            }
            processor_kwargs["do_sample_frames"] = False
        processor_kwargs["return_metadata"] = True
        output = dict(
            processor.video_processor(
                [frames],
                video_metadata=None if metadata is None else [metadata],
                **processor_kwargs,
            )
        )
        if self.model_type == "qwen2_5_vl":
            # From the sampled fps, as qwen-vl-utils computes it; float32 like HF.
            output["second_per_grid_ts"] = torch.tensor(
                [
                    processor.video_processor.temporal_patch_size / item.sampled_fps
                    for item in output["video_metadata"]
                ],
                dtype=torch.float32,
            )
        elif self.model_type == "qwen3_omni_moe":
            # From the requested fps (default 1.0), as the HF Omni processor does.
            output["video_second_per_grid"] = torch.tensor(
                [
                    processor.video_processor.temporal_patch_size
                    / processor_kwargs.get("fps", 1.0)
                ],
                dtype=torch.float32,
            )
        elif self.model_type != "qwen2_vl":
            output["video_metadata"] = [
                replace(item, fps=24) if item.fps is None else item
                for item in output["video_metadata"]
            ]
        return output

    def _bound_item_video_options(self, preprocess_kwargs):
        """Let per-item options lower, never raise, the configured frame and pixel
        limits: in serving they come from the client's video_url."""
        frame_cap = self.video_config.get("max_frames", FPS_MAX_FRAMES)
        pixel_cap = self.video_config.get("max_pixels", VIDEO_MAX_PIXELS)
        caps = {
            "nframes": frame_cap,
            "max_frames": frame_cap,
            "max_pixels": pixel_cap,
            "total_pixels": self.video_config.get("total_pixels", VIDEO_TOTAL_PIXELS),
        }
        options = {
            key: min(value, caps[key]) if key in caps else value
            for key, value in preprocess_kwargs.items()
        }
        # smart_resize applies no pixel limit to an explicit size.
        if (
            options.get("resized_height", 0) * options.get("resized_width", 0)
            > pixel_cap
        ):
            raise ValueError(
                f"resized_height * resized_width exceeds the video max_pixels {pixel_cap}"
            )
        return options

    def process_audio(self, audio, processor, preprocess_kwargs, **kwargs):
        output = dict(
            processor.feature_extractor([audio], **{**kwargs, **preprocess_kwargs})
        )
        output["feature_attention_mask"] = output.pop("attention_mask")
        return output

    def merge_audio_features(self, item_features: list[dict]) -> dict:
        return {
            "input_features": torch.nn.utils.rnn.pad_sequence(
                [features["input_features"][0].T for features in item_features],
                batch_first=True,
            ).transpose(1, 2),
            "feature_attention_mask": torch.nn.utils.rnn.pad_sequence(
                [features["feature_attention_mask"][0] for features in item_features],
                batch_first=True,
            ),
        }

    def build_processor_output(
        self,
        media_features: BatchFeature,
        input_ids: list[int],
        *,
        processor,
        **kwargs,
    ) -> BatchFeature:
        """Return what processor(text, media) returns, from expanded input_ids.

        Text keys follow the processor defaults and the call kwargs, as in
        ProcessorMixin.__call__: mm_token_type_ids only when requested, and
        video_metadata only with return_metadata. A single unpadded request needs
        no attention_mask.
        """
        kwargs_type = (
            self.media_processor_kwargs_type or processor.valid_processor_kwargs
        )
        text_kwargs = processor._merge_kwargs(
            kwargs_type,
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
            **kwargs,
        )["text_kwargs"]
        output = {"input_ids": torch.tensor([input_ids], dtype=torch.long)}
        if text_kwargs.get("return_mm_token_type_ids", False):
            output["mm_token_type_ids"] = torch.tensor(
                processor.create_mm_token_type_ids([input_ids]), dtype=torch.long
            )
        output.update(media_features)
        if not kwargs.get("return_metadata"):
            output.pop("video_metadata", None)
        return BatchFeature(output)

    # Adapted from sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor.build_input_ids_with_timestamps;
    # returns per-media expansions instead of building input_ids and offsets.
    def get_mm_token_expansion_spec(self, media_features, *, encode_expansion_text):
        image_token_id = self.mm_token_expansion_params["image_token_id"]
        video_token_id = self.mm_token_expansion_params["video_token_id"]
        audio_token_id = self.mm_token_expansion_params["audio_token_id"]
        merge_length = self.mm_token_expansion_params["spatial_merge_size"] ** 2
        image_expansions, video_expansions, audio_expansions = [], [], []
        if "image_grid_thw" in media_features:
            image_expansions = [
                [image_token_id] * (int(grid.prod()) // merge_length)
                for grid in media_features["image_grid_thw"]
            ]
        if "video_grid_thw" in media_features:
            for index, grid in enumerate(media_features["video_grid_thw"]):
                if not self.mm_token_expansion_params["frame_timestamps"]:
                    video_expansions.append(
                        [video_token_id] * (int(grid.prod()) // merge_length)
                    )
                    continue
                metadata = media_features["video_metadata"][index]
                frame_count, height, width = grid.tolist()
                timestamps = _frame_timestamps(
                    frame_indices=metadata.frames_indices,
                    fps=metadata.fps,
                    temporal_patch_size=self.mm_token_expansion_params[
                        "temporal_patch_size"
                    ],
                )
                expanded_tokens = []
                for frame_index in range(frame_count):
                    expanded_tokens.extend(
                        encode_expansion_text(
                            f"<{timestamps[frame_index]:.1f} seconds>"
                        )
                        + [self.mm_token_expansion_params["vision_start_token_id"]]
                        + [video_token_id] * (height * width // merge_length)
                        + [self.mm_token_expansion_params["vision_end_token_id"]]
                    )
                video_expansions.append(expanded_tokens)
        if "feature_attention_mask" in media_features:
            audio_token_counts = _get_feat_extract_output_lengths(
                media_features["feature_attention_mask"].sum(-1)
            )
            audio_expansions = [
                [audio_token_id] * int(token_count)
                for token_count in audio_token_counts
            ]
        mm_token_expansion_spec = {
            image_token_id: image_expansions,
            video_token_id: video_expansions,
        }
        if audio_token_id is not None:
            mm_token_expansion_spec[audio_token_id] = audio_expansions
        return mm_token_expansion_spec

    # Adapted from the M-RoPE block of sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor._process_mm_data_uncached;
    # reads the merged media fields instead of precomputed M-RoPE or mm_items.
    def compute_mrope_positions(self, input_ids, media_features):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long).flatten()
        image_grid_thw = media_features.get("image_grid_thw")
        video_grid_thw = media_features.get("video_grid_thw")
        second_per_grid_ts = media_features.get("second_per_grid_ts")
        if second_per_grid_ts is None:
            second_per_grid_ts = media_features.get("video_second_per_grid")
        audio_feature_lengths = None
        if media_features.get("feature_attention_mask") is not None:
            audio_feature_lengths = media_features["feature_attention_mask"].sum(-1)

        mrope_result = None
        if (
            video_grid_thw is None
            and second_per_grid_ts is None
            and audio_feature_lengths is None
        ):
            mrope_result = self._image_only_mrope_positions(input_ids, image_grid_thw)
        if mrope_result is None:
            mrope_result = get_rope_index(
                spatial_merge_size=self.mm_token_expansion_params["spatial_merge_size"],
                image_token_id=self.mm_token_expansion_params["image_token_id"],
                video_token_id=self.mm_token_expansion_params["video_token_id"],
                vision_start_token_id=self.mm_token_expansion_params[
                    "vision_start_token_id"
                ],
                model_type=self.model_type,
                tokens_per_second=self._tokens_per_second,
                input_ids=input_ids.unsqueeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                use_audio_in_video=False,
                audio_seqlens=audio_feature_lengths,
                audio_token_id=self.mm_token_expansion_params["audio_token_id"],
                audio_start_token_id=self._audio_start_token_id,
                position_id_per_seconds=self._position_id_per_seconds,
            )
        mrope_positions, mrope_position_delta = mrope_result
        if mrope_positions.ndim == 3:
            mrope_positions = mrope_positions.squeeze(1)
        return mrope_positions, mrope_position_delta

    # Adapted from sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor._compute_image_only_mrope_positions_from_offsets;
    # spans come from image-token runs in input_ids, not mm_items offsets.
    def _image_only_mrope_positions(self, input_ids, image_grid_thw):
        """get_rope_index for image-only prompts, from each image's token span.

        Returns None when the model or the prompt layout is outside this fast path.
        """
        if image_grid_thw is None or self.model_type not in (
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen4_exp",
        ):
            return None
        spans = _token_spans(
            input_ids, self.mm_token_expansion_params["image_token_id"]
        )
        if len(spans) != image_grid_thw.shape[0]:
            return None

        merge_size = self.mm_token_expansion_params["spatial_merge_size"]
        input_len = input_ids.numel()
        dtype, device = input_ids.dtype, input_ids.device
        position_segments = []
        st = 0
        next_pos = 0
        for (start, end), grid in zip(spans, image_grid_thw):
            text_len = start - st
            if text_len > 0:
                position_segments.append(
                    torch.arange(text_len, dtype=dtype, device=device)
                    .view(1, -1)
                    .expand(3, -1)
                    + next_pos
                )
                next_pos += text_len
            llm_grid_t, h, w = [int(x) for x in grid.tolist()]
            llm_grid_h, llm_grid_w = h // merge_size, w // merge_size
            if llm_grid_t * llm_grid_h * llm_grid_w != end - start + 1:
                return None
            t_index = (
                torch.arange(llm_grid_t, dtype=dtype, device=device)
                .view(-1, 1)
                .expand(llm_grid_t, llm_grid_h * llm_grid_w)
                .reshape(-1)
            )
            h_index = (
                torch.arange(llm_grid_h, dtype=dtype, device=device)
                .view(1, -1, 1)
                .expand(llm_grid_t, llm_grid_h, llm_grid_w)
                .reshape(-1)
            )
            w_index = (
                torch.arange(llm_grid_w, dtype=dtype, device=device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, llm_grid_w)
                .reshape(-1)
            )
            position_segments.append(
                torch.stack([t_index, h_index, w_index]) + next_pos
            )
            next_pos += max(llm_grid_t, llm_grid_h, llm_grid_w)
            st = end + 1
        if st < input_len:
            position_segments.append(
                torch.arange(input_len - st, dtype=dtype, device=device)
                .view(1, -1)
                .expand(3, -1)
                + next_pos
            )
        mrope_positions = torch.cat(position_segments, dim=1).unsqueeze(1)
        mrope_position_delta = (mrope_positions.max() + 1 - input_len).reshape(1, 1)
        return mrope_positions, mrope_position_delta


def _token_spans(input_ids, token_id):
    """Inclusive (start, end) of each contiguous run of token_id."""
    mask = input_ids == token_id
    starts = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]
    ends = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]
    return list(zip(starts.tolist(), ends.tolist()))


# Adapted from sglang.srt.disaggregation.encoder.preprocessor.EncoderPreprocessor._calculate_timestamps;
# keyword-only, and the caller's frame indices are not extended in place.
def _frame_timestamps(*, frame_indices, fps, temporal_patch_size):
    """Seconds per temporal patch: the mean of its first and last frame."""
    frame_indices = list(frame_indices)
    frame_indices += [frame_indices[-1]] * (-len(frame_indices) % temporal_patch_size)
    timestamps = [index / fps for index in frame_indices]
    return [
        (timestamps[start] + timestamps[start + temporal_patch_size - 1]) / 2
        for start in range(0, len(timestamps), temporal_patch_size)
    ]
