from dataclasses import replace

import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessorKwargs,
    _get_feat_extract_output_lengths,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)

from sglang.srt.multimodal.qwen_vl_media_processing import (
    QWEN_VIDEO_PREPROCESS_CONFIG_KEYS,
    preprocess_video_sync,
)
from sglang.srt.multimodal.token_space.process_strategy import (
    TokenSpaceProcessStrategy,
)


class QwenTokenSpaceProcessStrategy(TokenSpaceProcessStrategy):
    _SUPPORTED_MODEL_TYPES = frozenset(
        {
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_omni_moe",
            "qwen4_exp",
        }
    )

    @classmethod
    def supports_token_space_processing(cls, hf_config):
        return hf_config.model_type in cls._SUPPORTED_MODEL_TYPES

    def __init__(self, hf_config, processor, **kwargs):
        self.model_type = hf_config.model_type
        if self.model_type == "qwen3_omni_moe":
            self.media_processor_kwargs_type = Qwen3OmniMoeProcessorKwargs
            hf_config = hf_config.thinker_config

        super().__init__(hf_config, processor, **kwargs)

        self.image_token_id = hf_config.image_token_id
        self.video_token_id = hf_config.video_token_id
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = hf_config.vision_end_token_id
        self.audio_token_id = getattr(hf_config, "audio_token_id", None)

    def process_video(
        self, video, processor, source_config, video_metadata=None, **kwargs
    ):
        if kwargs.pop("use_audio_in_video", False):
            raise ValueError("Qwen token expansion does not support use_audio_in_video")
        kwargs.pop("seconds_per_chunk", None)
        kwargs.pop("position_id_per_seconds", None)
        config = {
            **self.video_config,
            **{
                key: value
                for key, value in kwargs.items()
                if key in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS
            },
            **source_config,
        }
        processor_kwargs = dict(kwargs)
        if isinstance(processor.video_processor, Qwen3VLVideoProcessor):
            processor_kwargs.update(
                {
                    key: value
                    for key, value in config.items()
                    if key not in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS
                }
            )
        frames, metadata = preprocess_video_sync(video, video_config=config)
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

    def process_audio(self, audio, processor, source_config, **kwargs):
        output = dict(
            processor.feature_extractor([audio], **{**kwargs, **source_config})
        )
        output["feature_attention_mask"] = output.pop("attention_mask")
        return output

    def merge_audio_features(self, features: list[dict]) -> dict:
        return {
            "input_features": torch.nn.utils.rnn.pad_sequence(
                [feature["input_features"][0].T for feature in features],
                batch_first=True,
            ).transpose(1, 2),
            "feature_attention_mask": torch.nn.utils.rnn.pad_sequence(
                [feature["feature_attention_mask"][0] for feature in features],
                batch_first=True,
            ),
        }

    def get_mm_token_expansion_spec(self, processor, media_features):
        image_expansions, video_expansions, audio_expansions = [], [], []
        # The processors produced the grids, so their merge sizes are the ones HF uses.
        if "image_grid_thw" in media_features:
            merge_length = processor.image_processor.merge_size**2
            image_expansions = [
                [self.image_token_id] * (int(grid.prod()) // merge_length)
                for grid in media_features["image_grid_thw"]
            ]
        if "video_grid_thw" in media_features:
            merge_length = processor.video_processor.merge_size**2
            for index, grid in enumerate(media_features["video_grid_thw"]):
                if self.model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_omni_moe"}:
                    video_expansions.append(
                        [self.video_token_id] * (int(grid.prod()) // merge_length)
                    )
                    continue
                metadata = media_features["video_metadata"][index]
                frame_count, height, width = grid.tolist()
                timestamps = processor._calculate_timestamps(
                    list(metadata.frames_indices),
                    metadata.fps,
                    processor.video_processor.temporal_patch_size,
                )
                expanded_tokens = []
                for frame_index in range(frame_count):
                    expanded_tokens.extend(
                        processor.tokenizer.encode(
                            f"<{timestamps[frame_index]:.1f} seconds>",
                            add_special_tokens=False,
                        )
                        + [self.vision_start_token_id]
                        + [self.video_token_id] * (height * width // merge_length)
                        + [self.vision_end_token_id]
                    )
                video_expansions.append(expanded_tokens)
        if "feature_attention_mask" in media_features:
            audio_token_counts = _get_feat_extract_output_lengths(
                media_features["feature_attention_mask"].sum(-1)
            )
            audio_expansions = [
                [self.audio_token_id] * int(token_count)
                for token_count in audio_token_counts
            ]
        mm_token_expansion_spec = [
            ([self.image_token_id], image_expansions),
            ([self.video_token_id], video_expansions),
        ]
        if self.audio_token_id is not None:
            mm_token_expansion_spec.append(([self.audio_token_id], audio_expansions))
        return mm_token_expansion_spec
