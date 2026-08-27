# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    MiniMaxH3AudioEncodingStage,
    MiniMaxH3DecodingStage,
    MiniMaxH3DenoisingStage,
    MiniMaxH3LatentPreparationStage,
    MiniMaxH3TextEncodingStage,
    MiniMaxH3TimestepPreparationStage,
    MiniMaxH3VisualEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MiniMaxH3Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MiniMaxH3Pipeline"
    default_model_subfolder = "FL2VA"
    is_video_pipeline = True
    pipeline_config_cls = MiniMaxH3PipelineConfig
    sampling_params_cls = MiniMaxH3SamplingParams
    _required_config_modules = [
        "processor",
        "text_encoder",
        "tokenizer",
        "video_vae",
        "audio_vae",
        # scheduler intentionally absent: model_index carries scheduler=null;
        # per-modality sigma schedules are generated in TimestepPreparation
        # from the task profile, and the loop scheduler math lives in
        # scheduling_minimax_h3_euler_ancestral (stages accept scheduler=None).
        "transformer",
    ]

    @staticmethod
    def model_subfolder_for_variant(variant: str) -> str:
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError("MiniMax H3 model variant must be a non-empty string")
        normalized = variant.strip().lower()
        subfolders = {
            "fl2va": "FL2VA",
            "ref2va": "Ref2VA",
        }
        try:
            return subfolders[normalized]
        except KeyError as exc:
            raise ValueError(
                f"unsupported MiniMax H3 model variant {variant!r}; "
                f"supported: {sorted(subfolders)!r}"
            ) from exc

    def _load_config(self):
        model_variant = self.server_args.model_variant
        if model_variant is not None:
            semantic_subfolder = self.model_subfolder_for_variant(model_variant)
            explicit_subfolder = self.server_args.model_subfolder
            if (
                explicit_subfolder is not None
                and explicit_subfolder.strip().lower() != semantic_subfolder.lower()
            ):
                raise ValueError(
                    "MiniMax H3 --model-variant and --model-subfolder select "
                    f"different weight partitions: variant={model_variant!r} maps to "
                    f"{semantic_subfolder!r}, model_subfolder="
                    f"{explicit_subfolder!r}"
                )
            self.server_args.model_subfolder = semantic_subfolder
        model_index = super()._load_config()
        self.release_metadata = MiniMaxH3ReleaseMetadata.from_model_index(model_index)
        if (
            model_variant is not None
            and self.release_metadata.partition != model_variant.strip().lower()
        ):
            raise ValueError(
                "MiniMax H3 loaded checkpoint partition does not match "
                f"--model-variant {model_variant!r}"
            )
        return model_index

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "MiniMaxH3Pipeline only supports monolithic deployment; "
                f"disaggregation role {role.value!r} is not supported"
            )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        # Per-model sigma override from model_index.json; contract tests
        # construct the pipeline without model_path, hence the guard.
        release_metadata = getattr(self, "release_metadata", None)
        sigma_shift_scales = (
            release_metadata.sigma_shift_scales
            if release_metadata is not None
            else None
        )
        self.add_stage(InputValidationStage())
        if release_metadata is not None:
            self.add_stage(MiniMaxH3PartitionAdmissionStage(release_metadata))
        self.add_stage(
            MiniMaxH3TextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
            )
        )
        self.add_stage(
            MiniMaxH3VisualEncodingStage(
                video_vae=self.get_module("video_vae"),
                vae_arch_config=server_args.pipeline_config.vae_config.arch_config,
            )
        )
        self.add_stage(
            MiniMaxH3AudioEncodingStage(
                audio_vae=self.get_module("audio_vae"),
                vae_arch_config=server_args.pipeline_config.audio_vae_config.arch_config,
            )
        )
        self.add_stage(MiniMaxH3LatentPreparationStage())
        self.add_stage(
            MiniMaxH3TimestepPreparationStage(
                sigma_shift_scales=sigma_shift_scales,
            )
        )
        self.add_stage(
            MiniMaxH3DenoisingStage(
                transformer=self.get_module("transformer"),
                pipeline=self,
            )
        )
        self.add_stage(
            MiniMaxH3DecodingStage(
                video_vae=self.get_module("video_vae"),
                audio_vae=self.get_module("audio_vae"),
            )
        )


EntryClass = MiniMaxH3Pipeline
