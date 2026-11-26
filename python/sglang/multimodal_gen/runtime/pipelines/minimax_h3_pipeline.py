# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import shutil
from dataclasses import replace

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_image import (
    MiniMaxH3ImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.configs.sample.minimax_h3_image import (
    MiniMaxH3ImageSamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
    minimax_h3_cleanup_temp_dirs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
    MINIMAX_H3_PREPARED_REFERENCE_IMAGE_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_DUAL_VARIANT,
    MINIMAX_H3_TASK_PARTITIONS,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,
    verify_model_config_and_directory,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
        # minimax_h3/denoise_loop.py (stages accept scheduler=None).
        "transformer",
    ]

    def __init__(self, *args, **kwargs):
        # TODO: Enable this check on ROCm after adding ffmpeg to the AMD Docker
        # image and CI dependency installer.
        if self.is_video_pipeline and not current_platform.is_rocm():
            missing_media_tools = [
                executable
                for executable in ("ffmpeg", "ffprobe")
                if shutil.which(executable) is None
            ]
            if missing_media_tools:
                raise RuntimeError(
                    "MiniMax H3 requires ffmpeg and ffprobe for media processing "
                    "and validated output delivery; missing executables: "
                    f"{', '.join(missing_media_tools)}. Install the ffmpeg system "
                    "package before starting SGLang."
                )
        # fl2va+ref2va: (nickname, target, path) of each transformer's own
        # release adapter; LoRAPipeline applies one adapter to every transformer
        self.dual_variant_lora_adapters: list[tuple[str, str, str]] = []
        super().__init__(*args, **kwargs)
        if self.dual_variant_lora_adapters:
            self.convert_to_lora_layers(snapshot_base=False)
            for nickname, target, path in self.dual_variant_lora_adapters:
                self.set_lora(
                    nickname,
                    path,
                    target=target,
                    strength=self.server_args.lora_scale,
                    lora_alpha=self.server_args.lora_alpha,
                    cache_merged=True,
                )

    @staticmethod
    def model_subfolder_for_variant(variant: str) -> str:
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError("MiniMax H3 model variant must be a non-empty string")
        normalized = variant.strip().lower()
        subfolders = {
            "fl2va": "FL2VA",
            "ref2va": "Ref2VA",
            "hybrid": "Ref2VA",
            # FL2VA serves the shared components; Ref2VA adds transformer_2
            MINIMAX_H3_DUAL_VARIANT: "FL2VA",
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
            if model_variant.strip().lower() == "hybrid" and not (
                self.server_args.component_weights_paths.get("transformer")
                or self.server_args.transformer_weights_path
            ):
                raise ValueError(
                    "MiniMax H3 --model-variant hybrid requires explicit merged "
                    "weights via --component-weights-paths.transformer"
                )
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
        if self.release_metadata.is_image == self.is_video_pipeline:
            raise ValueError(
                f"{self.pipeline_name} cannot serve the "
                f"{self.release_metadata.partition!r} release partition"
            )
        if (
            model_variant is not None
            and self.release_metadata.partition != semantic_subfolder.lower()
        ):
            raise ValueError(
                "MiniMax H3 loaded checkpoint partition does not match "
                f"--model-variant {model_variant!r}"
            )
        if model_variant is not None and model_variant.strip().lower() == "hybrid":
            # merged checkpoints share the native graph across all three tasks
            # keep the base partition contract strict unless explicitly selected
            self.release_metadata = replace(
                self.release_metadata,
                partition="hybrid",
                tasks=tuple(MINIMAX_H3_TASK_PARTITIONS),
            )
        if (
            model_variant is not None
            and model_variant.strip().lower() == MINIMAX_H3_DUAL_VARIANT
        ):
            self._add_ref2va_transformer(model_index)
            return model_index
        adapter = self.release_metadata.lora_adapter
        if adapter is not None:
            # LoRAPipeline reads server_args.lora_path once the modules are loaded
            self.server_args.lora_path = self._release_lora_adapter_path(
                adapter, self.model_path
            )
        return model_index

    def _add_ref2va_transformer(self, model_index: dict) -> None:
        """fl2va+ref2va: the Ref2VA transformer loads as ``transformer_2`` and
        serves ref2va; the FL2VA one serves t2va / fl2va. Text encoder, VAEs,
        tokenizer and processor are byte-identical between the partitions and
        load once, from FL2VA."""
        subfolder = self.model_subfolder_for_variant("ref2va")
        ref2va_path = os.path.join(os.path.dirname(self.model_path), subfolder)
        if not os.path.isfile(os.path.join(ref2va_path, "model_index.json")):
            # a Hub snapshot holds the FL2VA partition only; the Hub cache keeps
            # one copy of the files both partitions share
            snapshot = maybe_download_model(
                self.server_args.model_path,
                allow_patterns=[f"{subfolder}/**"],
                revision=self.server_args.revision,
                model_subfolder=subfolder,
            )
            ref2va_path = os.path.join(snapshot, subfolder)
        ref2va = MiniMaxH3ReleaseMetadata.from_model_index(
            verify_model_config_and_directory(ref2va_path)
        )
        fl2va = self.release_metadata
        # both transformers run behind the same stages and sampling contract
        same_contract = replace(
            ref2va,
            partition=fl2va.partition,
            tasks=fl2va.tasks,
            task_aliases=fl2va.task_aliases,
        )
        if ref2va.partition != "ref2va" or same_contract != fl2va:
            raise ValueError(
                f"MiniMax H3 --model-variant {MINIMAX_H3_DUAL_VARIANT} needs a "
                "Ref2VA partition with the FL2VA sampling contract; got "
                f"{ref2va} vs {fl2va}"
            )
        self.release_metadata = replace(
            fl2va,
            partition=MINIMAX_H3_DUAL_VARIANT,
            tasks=fl2va.tasks + ref2va.tasks,
            task_aliases={**fl2va.task_aliases, **ref2va.task_aliases},
        )
        model_index["transformer_2"] = model_index["transformer"]
        self.required_config_modules.append("transformer_2")
        self.server_args.component_paths["transformer_2"] = os.path.join(
            ref2va_path, "transformer"
        )
        for nickname, target, metadata, model_dir in (
            ("fl2va", "transformer", fl2va, self.model_path),
            ("ref2va", "transformer_2", ref2va, ref2va_path),
        ):
            if metadata.lora_adapter is not None:
                adapter_path = self._release_lora_adapter_path(
                    metadata.lora_adapter, model_dir
                )
                self.dual_variant_lora_adapters.append((nickname, target, adapter_path))
        logger.info(
            "MiniMax H3 %s: transformer from %s (%s), transformer_2 from %s (%s); "
            "text_encoder, video_vae, audio_vae, tokenizer and processor once, "
            "from %s; release adapters %s",
            MINIMAX_H3_DUAL_VARIANT,
            os.path.join(self.model_path, "transformer"),
            "/".join(fl2va.tasks),
            os.path.join(ref2va_path, "transformer"),
            "/".join(ref2va.tasks),
            self.model_path,
            self.dual_variant_lora_adapters,
        )

    def _release_lora_adapter_path(self, adapter: str, model_dir: str) -> str:
        """A release's bundled LoRA adapter, loaded as the startup adapter."""
        adapter_path = os.path.join(model_dir, adapter)
        if not os.path.isfile(adapter_path):
            raise ValueError(
                f"{self.pipeline_name} release adapter not found: {adapter_path}"
            )
        if self.server_args.lora_path not in (None, adapter_path):
            raise ValueError(
                f"{self.pipeline_name} applies its own release adapter "
                f"({adapter}); --lora-path cannot replace it"
            )
        return adapter_path

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "MiniMaxH3Pipeline only supports monolithic deployment; "
                f"disaggregation role {role.value!r} is not supported"
            )

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        self.pdd_heads = self._resolve_pdd_heads()
        transformer = self.get_module("transformer")
        if (
            self.pdd_heads
            and transformer is not None
            and transformer.final_layer._pdd_heads is None
        ):
            transformer.final_layer.load_pdd_fused_heads(self.pdd_heads)

    def _resolve_pdd_heads(self) -> str | None:
        """The env override, else a PDD release's own fused heads."""
        if envs.SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS:
            return envs.SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS
        release_metadata = getattr(self, "release_metadata", None)
        if release_metadata is None or release_metadata.pdd_fused_heads is None:
            return None
        return os.path.join(self.model_path, release_metadata.pdd_fused_heads)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        # Per-model sigma override from model_index.json; contract tests
        # construct the pipeline without model_path, hence the guard.
        release_metadata = getattr(self, "release_metadata", None)
        sigma_shift_scales = (
            release_metadata.sigma_shift_scales
            if release_metadata is not None
            else None
        )
        dit_arch = server_args.pipeline_config.dit_config.arch_config
        if self.is_video_pipeline and not dit_arch.audio_enabled:
            raise ValueError(
                f"{self.pipeline_name} generates audio; its DiT checkpoint cannot "
                "be audio-free (audio_enabled: false in transformer/config.json)"
            )
        if self.is_video_pipeline:
            # Video only: image releases skip the generic image_path load/resize;
            # Qwen and the VAE apply their own reference preprocessing.
            self.add_stage(InputValidationStage())
        if release_metadata is not None:
            self.add_stage(MiniMaxH3PartitionAdmissionStage(release_metadata))
        text_encoder = self.get_module("text_encoder")
        visual = getattr(getattr(text_encoder, "model", None), "visual", None)
        if not self.is_video_pipeline and visual is not None:
            # CPU initialization computes pow differently from CUDA. Image
            # releases rebuild the small vision frequency table after offload
            # transfers so conditioning matches the GPU-resident encoder, as in
            # the native Qwen-Image path. Video releases keep the table their
            # consistency references were generated with.
            visual.rotary_pos_emb.recompute_on_device_change = True
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
        if self.is_video_pipeline:
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
                dmd_denoising_steps=(
                    release_metadata.dmd_denoising_steps
                    if release_metadata is not None
                    else None
                ),
                pdd_heads=getattr(self, "pdd_heads", None),
            )
        )
        self.add_stage(
            MiniMaxH3DenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2"),
                pipeline=self,
                stochastic_sampling=(
                    release_metadata is not None
                    and release_metadata.stochastic_sampling
                ),
            )
        )
        self.add_stage(
            MiniMaxH3DecodingStage(
                video_vae=self.get_module("video_vae"),
                audio_vae=self.get_module("audio_vae"),
            )
        )

    def create_comfyui_stages(self, server_args: ServerArgs) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.comfyui_step import (
            MiniMaxH3ComfyUIStepStage,
        )

        self.add_stage(
            MiniMaxH3ComfyUIStepStage(transformer=self.get_module("transformer"))
        )


class FastH3Pipeline(MiniMaxH3Pipeline):
    """FastH3: 8-step DMD2-distilled MiniMax-H3 (V2, t2va only), and the
    FastH3 Omni Ref2VA PDD student.

    The flat single-partition repo is materialized into the base-H3 layout by
    its registered model overlay (see runtime/utils/model_overlay.py), so every stage, loader,
    and admission path below is exactly the MiniMax-H3 one. There is no
    FL2VA/Ref2VA partition layout to default into.
    """

    pipeline_name = "FastH3Pipeline"
    default_model_subfolder = None


class PikaH3FlexPipeline(MiniMaxH3Pipeline):
    """Pika H3 Flex: MiniMax-H3 plus Pika's six-step LoRA adapters.

    The release's model overlay links the pinned MiniMax-H3 base into the
    base-H3 layout (root and ``FL2VA/`` for t2va/fl2va, ``Ref2VA/`` for
    ref2va) with the matching adapter next to each partition. Its
    ``_minimax_h3`` block carries the sampling contract (knots
    1000/750/500/250/100/40, video shift 12, audio shift 3, re-noising) and
    the adapter, which loads at startup.
    """

    pipeline_name = "PikaH3FlexPipeline"

    def _load_config(self):
        # The released adapters require FP32 matmuls at FP32 module boundaries.
        torch.set_float32_matmul_precision("highest")
        return super()._load_config()

    def load_lora_adapter(
        self,
        lora_path: str,
        lora_nickname: str,
        rank: int,
        weight_name: str | None = None,
        lora_alpha: int | None = None,
    ) -> None:
        super().load_lora_adapter(
            lora_path, lora_nickname, rank, weight_name, lora_alpha
        )
        # Release files store FP32 masters; inference uses BF16 adapter factors.
        self.lora_adapters[lora_nickname] = {
            name: weight.to(torch.bfloat16)
            for name, weight in self.lora_adapters[lora_nickname].items()
        }


class MiniMaxH3ImagePipeline(MiniMaxH3Pipeline):
    """One image checkpoint for its declared t2i/edit tasks: H3 stages without audio."""

    pipeline_name = "MiniMaxH3ImagePipeline"
    default_model_subfolder = None
    is_video_pipeline = False
    pipeline_config_cls = MiniMaxH3ImagePipelineConfig
    sampling_params_cls = MiniMaxH3ImageSamplingParams
    _required_config_modules = [
        "processor",
        "text_encoder",
        "tokenizer",
        "video_vae",
        "transformer",
    ]
    # The image release names its one VAE `vae` (model_index.json key and
    # directory), as diffusers does; the shared H3 stages call the module
    # `video_vae`, so the pipeline loads that module from `vae/`.
    _extra_config_module_map = {"video_vae": "vae"}

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        # Image releases are PDD checkpoints; there is no image sampler for a
        # checkpoint without its fused per-step output heads.
        if self._resolve_pdd_heads() is None:
            raise ValueError(
                f"{self.pipeline_name} serves PDD releases only: model_index.json "
                "must name the fused heads in _minimax_h3.pdd_fused_heads"
            )
        super().initialize_pipeline(server_args)

    def forward(self, batch, server_args):
        try:
            return super().forward(batch, server_args)
        finally:
            # Never retain request images or downloaded references.
            batch.extra.pop(MINIMAX_H3_PREPARED_REFERENCE_IMAGE_EXTRA_KEY, None)
            minimax_h3_cleanup_temp_dirs(batch)


EntryClass = [
    MiniMaxH3Pipeline,
    FastH3Pipeline,
    PikaH3FlexPipeline,
    MiniMaxH3ImagePipeline,
]
