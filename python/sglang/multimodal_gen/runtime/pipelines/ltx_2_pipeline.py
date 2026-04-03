import math
import os

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    LTX2AVDecodingStage,
    LTX2AVDenoisingStage,
    LTX2AVLatentPreparationStage,
    LTX2HalveResolutionStage,
    LTX2LoRASwitchStage,
    LTX2RefinementStage,
    LTX2TextConnectorStage,
    LTX2UpsampleStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


def _resolve_ltx2_two_stage_component_paths(
    model_path: str, component_paths: dict[str, str]
) -> dict[str, str]:
    resolved = dict(component_paths)
    auto_resolved = []

    if "spatial_upsampler" not in resolved:
        spatial_candidates = [
            os.path.join(model_path, "latent_upsampler"),
            os.path.join(model_path, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            os.path.join(model_path, "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
            os.path.join(model_path, "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
        ]
        for candidate in spatial_candidates:
            if os.path.exists(candidate):
                resolved["spatial_upsampler"] = candidate
                auto_resolved.append(f"spatial_upsampler={candidate}")
                break

    if "distilled_lora" not in resolved:
        distilled_lora_candidates = [
            os.path.join(model_path, "ltx-2.3-22b-distilled-lora-384.safetensors"),
            os.path.join(model_path, "ltx-2-19b-distilled-lora-384.safetensors"),
        ]
        for distilled_lora in distilled_lora_candidates:
            if os.path.exists(distilled_lora):
                resolved["distilled_lora"] = distilled_lora
                auto_resolved.append(f"distilled_lora={distilled_lora}")
                break

    if auto_resolved:
        logger.info(
            "Auto-resolved LTX2 two-stage components: %s", ", ".join(auto_resolved)
        )

    return resolved


def calculate_ltx2_shift(
    image_seq_len: int,
    base_seq_len: int = BASE_SHIFT_ANCHOR,
    max_seq_len: int = MAX_SHIFT_ANCHOR,
    base_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
    mm = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - mm * base_seq_len
    return image_seq_len * mm + b


def prepare_ltx2_mu(batch: Req, server_args: ServerArgs):
    latent_num_frames = (int(batch.num_frames) - 1) // int(
        server_args.pipeline_config.vae_temporal_compression
    ) + 1
    latent_height = int(batch.height) // int(
        server_args.pipeline_config.vae_scale_factor
    )
    latent_width = int(batch.width) // int(server_args.pipeline_config.vae_scale_factor)
    video_sequence_length = latent_num_frames * latent_height * latent_width
    return "mu", calculate_ltx2_shift(video_sequence_length)


class LTX2SigmaPreparationStage(PipelineStage):
    """Prepare native LTX-2 sigma schedule before timestep setup."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch.extra["ltx2_phase"] = "stage1"
        batch.sigmas = np.linspace(
            1.0,
            1.0 / int(batch.num_inference_steps),
            int(batch.num_inference_steps),
        ).tolist()
        return batch


def _add_ltx2_front_stages(pipeline: ComposedPipelineBase):
    pipeline.add_stages(
        [
            InputValidationStage(),
            TextEncodingStage(
                text_encoders=[pipeline.get_module("text_encoder")],
                tokenizers=[pipeline.get_module("tokenizer")],
            ),
            LTX2TextConnectorStage(connectors=pipeline.get_module("connectors")),
        ]
    )


def _add_ltx2_stage1_generation_stages(pipeline: ComposedPipelineBase):
    pipeline.add_stage(LTX2SigmaPreparationStage())
    pipeline.add_standard_timestep_preparation_stage(
        prepare_extra_kwargs=[prepare_ltx2_mu]
    )
    pipeline.add_stages(
        [
            LTX2AVLatentPreparationStage(
                scheduler=pipeline.get_module("scheduler"),
                transformer=pipeline.get_module("transformer"),
                audio_vae=pipeline.get_module("audio_vae"),
            ),
            LTX2AVDenoisingStage(
                transformer=pipeline.get_module("transformer"),
                scheduler=pipeline.get_module("scheduler"),
                vae=pipeline.get_module("vae"),
                audio_vae=pipeline.get_module("audio_vae"),
                pipeline=pipeline,
            ),
        ]
    )


def _add_ltx2_decoding_stage(pipeline: ComposedPipelineBase):
    pipeline.add_stage(
        LTX2AVDecodingStage(
            vae=pipeline.get_module("vae"),
            audio_vae=pipeline.get_module("audio_vae"),
            vocoder=pipeline.get_module("vocoder"),
            pipeline=pipeline,
        )
    )


class LTX2FlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    """Override ``_time_shift_exponential`` to use torch f32 instead of numpy f64."""

    def set_timesteps(
        self,
        num_inference_steps=None,
        device=None,
        sigmas=None,
        mu=None,
        timesteps=None,
    ):
        if sigmas is not None and timesteps is None and mu is None:
            sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
            self.num_inference_steps = len(timesteps)
            self.timesteps = timesteps
            self.sigmas = sigmas
            self._step_index = None
            self._begin_index = None
            return

        return super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )

    def _time_shift_exponential(self, mu, sigma, t):
        if isinstance(t, np.ndarray):
            t_torch = torch.from_numpy(t).to(torch.float32)
            result = math.exp(mu) / (math.exp(mu) + (1 / t_torch - 1) ** sigma)
            return result.numpy()
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class _BaseLTX2Pipeline(LoRAPipeline):
    _required_config_modules = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "vae",
        "audio_vae",
        "vocoder",
        "connectors",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        orig = self.get_module("scheduler")
        self.modules["scheduler"] = LTX2FlowMatchScheduler.from_config(orig.config)


class LTX2Pipeline(_BaseLTX2Pipeline):
    # Must match model_index.json `_class_name`.
    pipeline_name = "LTX2Pipeline"

    def create_pipeline_stages(self, server_args: ServerArgs):
        _add_ltx2_front_stages(self)
        _add_ltx2_stage1_generation_stages(self)
        _add_ltx2_decoding_stage(self)


class LTX2TwoStagePipeline(_BaseLTX2Pipeline):
    pipeline_name = "LTX2TwoStagePipeline"
    STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]

    def initialize_pipeline(self, server_args: ServerArgs):
        super().initialize_pipeline(server_args)
        server_args.component_paths = _resolve_ltx2_two_stage_component_paths(
            self.model_path, server_args.component_paths
        )

        upsampler_path = server_args.component_paths.get("spatial_upsampler")
        if not upsampler_path:
            raise ValueError(
                "LTX2TwoStagePipeline requires --spatial-upsampler-path "
                "(component_paths['spatial_upsampler'])."
            )
        module, memory_usage = PipelineComponentLoader.load_component(
            component_name="spatial_upsampler",
            component_model_path=upsampler_path,
            transformers_or_diffusers="diffusers",
            server_args=server_args,
        )
        self.modules["spatial_upsampler"] = module
        self.memory_usages["spatial_upsampler"] = memory_usage

        distilled_lora_path = server_args.component_paths.get("distilled_lora")
        if not distilled_lora_path:
            raise ValueError(
                "LTX2TwoStagePipeline requires --distilled-lora-path "
                "(component_paths['distilled_lora'])."
            )
        self._distilled_lora_path = distilled_lora_path
        self._stage1_lora_path = server_args.lora_path
        self._stage1_lora_scale = float(server_args.lora_scale)
        self._active_lora_phase = None

    def switch_lora_phase(self, phase: str) -> None:
        if phase == self._active_lora_phase:
            return

        if phase == "stage1":
            if self._stage1_lora_path:
                self.set_lora(
                    lora_nickname="ltx2_stage1_base",
                    lora_path=self._stage1_lora_path,
                    target="transformer",
                    strength=self._stage1_lora_scale,
                )
            else:
                # Stage 1 must run on the base transformer weights. If stage 2 left the
                # distilled adapter active, stage 1 quality drifts away from the official
                # two-stage pipeline immediately.
                self.deactivate_lora_weights(target="transformer")
        elif phase == "stage2":
            lora_nicknames = []
            lora_paths = []
            lora_strengths = []
            lora_targets = []
            if self._stage1_lora_path:
                lora_nicknames.append("ltx2_stage1_base")
                lora_paths.append(self._stage1_lora_path)
                lora_strengths.append(self._stage1_lora_scale)
                lora_targets.append("transformer")
            lora_nicknames.append("ltx2_stage2_distilled")
            lora_paths.append(self._distilled_lora_path)
            lora_strengths.append(1.0)
            lora_targets.append("transformer")
            self.set_lora(
                lora_nickname=lora_nicknames,
                lora_path=lora_paths,
                target=lora_targets,
                strength=lora_strengths,
                # Keep the distilled adapter unmerged when it is the only active LoRA.
                # Merging it into the base weights makes the subsequent switch back to
                # stage 1 depend on unmerge bookkeeping instead of the original base.
                merge_weights=self._stage1_lora_path is not None,
            )
        else:
            raise ValueError(f"Unknown LTX2 two-stage LoRA phase: {phase}")

        self._active_lora_phase = phase

    def create_pipeline_stages(self, server_args: ServerArgs):
        _add_ltx2_front_stages(self)
        self.add_stage(LTX2HalveResolutionStage())
        self.add_stage(
            LTX2LoRASwitchStage(pipeline=self, phase="stage1"),
        )
        _add_ltx2_stage1_generation_stages(self)
        self.add_stages(
            [
                LTX2UpsampleStage(
                    spatial_upsampler=self.get_module("spatial_upsampler"),
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                ),
                (
                    LTX2LoRASwitchStage(pipeline=self, phase="stage2"),
                    "ltx2_lora_switch_stage2",
                ),
                LTX2RefinementStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    distilled_sigmas=self.STAGE_2_DISTILLED_SIGMA_VALUES,
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                ),
            ]
        )
        _add_ltx2_decoding_stage(self)


EntryClass = [LTX2Pipeline, LTX2TwoStagePipeline]
