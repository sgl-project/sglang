# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 image (t2i / edit) pipeline config and its registration."""

import json
import os
import posixpath
from dataclasses import dataclass
from typing import ClassVar

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.configs.task_type import DataType, ModelTaskType
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    RESIDENT,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)

# The assembled image VAE (CNN encoder + 2D ViT decoder) is one standard
# diffusers component weight file, unlike the video VAE's source/ layout.
IMAGE_VAE_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
# model_index.json ``_minimax_h3.tasks`` names -> the task types the HTTP layer resolves.
_IMAGE_TASK_TYPES = {"t2i": ModelTaskType.T2I, "edit": ModelTaskType.I2I}


@dataclass
class MiniMaxH3ImagePipelineConfig(MiniMaxH3PipelineConfig):
    """H3 DiT, Qwen and VAE with the image release's deployment contract.

    One checkpoint serves the t2i/edit tasks its release metadata declares.
    """

    task_type: ModelTaskType = ModelTaskType.T2I
    supported_task_types: ClassVar[tuple[ModelTaskType, ...]] = (
        ModelTaskType.T2I,
        ModelTaskType.I2I,
    )
    # The image decoder runs with BF16 weights under BF16 autocast.
    vae_decode_precision: str = "bf16"
    vae_sp: bool = False
    supports_auto_residency: bool = False
    output_audio_sample_rate: int | None = None
    output_audio_channels: int | None = None
    output_av_drift_tolerance_s: float | None = None
    # Edit references are resized per consumer so that their short edge lies
    # within [min, max]: a shorter edge is scaled up to the minimum, a longer
    # one down to the maximum, keeping the ratio with sides on the nearest 32px
    # multiple. A short edge already inside keeps the original pixels; equal
    # bounds fix the short edge. Qwen's processor still applies its own 32px
    # rounding and pixel limits. Bounds are multiples of 32; override them
    # through --pipeline-config-path.
    edit_reference_qwen_min_short_edge: int = 1024
    edit_reference_qwen_max_short_edge: int = 1024
    edit_reference_vae_min_short_edge: int = 1024
    edit_reference_vae_max_short_edge: int = 2048
    # A request smaller than this many pixels is generated on a larger canvas
    # with the same aspect ratio and resized back to the requested size after
    # decoding (minimax_h3/image_canvas.py); the default keeps every request at
    # 2048x2048 pixels or more. A request may set its own min_generation_area.
    # None disables the canvas: requests then need 32px-aligned sizes and may
    # not set the field.
    min_generation_area: int | None = 2048 * 2048

    def accepts_audio_input(self) -> bool:
        return False

    @property
    def requires_audio_output(self) -> bool:
        return False

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        # Automatic layerwise offload, FSDP and compile policies stay separate
        # from the larger video workload. One positive branch per step leaves
        # nothing for CFG parallelism, so with no other
        # parallel flag `--num-gpus N` runs Ulysses N over the packed sequence.
        return ModelDeploymentConfig(
            auto_auxiliary_layerwise_offload=False,
            # H200 keeps headroom even with AdaLN weights resident. GB10 uses
            # online AdaLN to fit native 4K in its shared pool; validation below
            # refuses a resident GB10 launch that keeps those projection weights.
            keep_resident_min_available_gb=(
                106 if self._uses_gb10_shared_memory() else 130
            ),
            keep_resident_components=("dit", "text_encoder", "vae"),
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=False,
        )

    @staticmethod
    def _uses_gb10_shared_memory() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.device_shares_host_memory()
            and torch.cuda.get_device_name(0).startswith("NVIDIA GB10")
        )

    def get_supported_task_types(self) -> tuple[ModelTaskType, ...]:
        """The tasks the loaded release declares, in model_index.json order.

        Without a model path the class capability applies; with one,
        ``_minimax_h3.tasks`` narrows it so an undeclared task fails in the HTTP
        process (400) instead of inside the scheduler.
        """
        if not self.model_path:
            return super().get_supported_task_types()
        return tuple(_IMAGE_TASK_TYPES[task] for task in self._declared_tasks())

    def _declared_tasks(self) -> tuple[str, ...]:
        return self._release_metadata().tasks

    def _release_metadata(self):
        cached = self.__dict__.get("_release_metadata_cache")
        if cached is not None and cached[0] == self.model_path:
            return cached[1]
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
            MiniMaxH3ReleaseMetadata,
        )
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_model_index,
        )

        metadata = MiniMaxH3ReleaseMetadata.from_model_index(
            maybe_download_model_index(self.model_path)
        )
        if not metadata.is_image:
            raise ValueError(
                f"{self.model_path!r} is a MiniMax H3 {metadata.partition!r} "
                "release, not an image release"
            )
        self.__dict__["_release_metadata_cache"] = (self.model_path, metadata)
        return metadata

    def release_pdd_steps(self) -> int:
        """The release's fused PDD head count, which fixes the step count.

        Image releases are PDD checkpoints with one head per step, so the HTTP
        process resolves each request to this count. The heads are the ones
        the pipeline loads (SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS, else
        model_index.json ``_minimax_h3.pdd_fused_heads``); pdd_config.json
        beside them lists their sigma grid.
        """
        cached = self.__dict__.get("_release_pdd_steps_cache")
        if cached is not None and cached[0] == self.model_path:
            return cached[1]
        from sglang.multimodal_gen import envs
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
            minimax_h3_image_pdd_grid,
        )
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_model,
        )

        if envs.SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS:
            config_path = os.path.join(
                os.path.dirname(envs.SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS),
                "pdd_config.json",
            )
        elif not self.model_path:
            raise ValueError(
                "MiniMax H3 image requires model_path to resolve its PDD heads"
            )
        elif (heads := self._release_metadata().pdd_fused_heads) is not None:
            relative = posixpath.join(posixpath.dirname(heads), "pdd_config.json")
            root = maybe_download_model(self.model_path, allow_patterns=[relative])
            config_path = os.path.join(root, relative)
        else:
            raise ValueError(
                f"{self.model_path!r} has no PDD heads; MiniMax H3 image serves "
                "PDD releases only (model_index.json _minimax_h3.pdd_fused_heads)"
            )
        with open(config_path) as f:
            steps = len(minimax_h3_image_pdd_grid(json.load(f))) - 1
        self.__dict__["_release_pdd_steps_cache"] = (self.model_path, steps)
        return steps

    def resolve_task_type(
        self,
        requested: ModelTaskType | str | None = None,
        *,
        data_type: DataType | None = None,
        has_image: bool = False,
        has_video: bool = False,
    ) -> ModelTaskType:
        """References select edit (I2I), none t2i; the release must declare it."""
        if has_video:
            raise ValueError("MiniMax H3 image accepts image references only")
        task = ModelTaskType.I2I if has_image else ModelTaskType.T2I
        if requested is not None and ModelTaskType.parse(requested) != task:
            raise ValueError(
                f"task_type {ModelTaskType.parse(requested).name} does not match "
                f"this request's references, which select {task.name}"
            )
        if data_type is not None and data_type != DataType.IMAGE:
            raise ValueError(
                f"MiniMax H3 image produces IMAGE output, not {data_type.name}"
            )
        supported = self.get_supported_task_types()
        if task not in supported:
            names = {value: key for key, value in _IMAGE_TASK_TYPES.items()}
            raise ValueError(
                f"task {names[task]!r} is not served by this MiniMax H3 image "
                f"release; declared tasks: {[names[t] for t in supported]!r}"
            )
        return task

    def validate_server_args(self, server_args) -> None:
        # Replaces the parent's checks, which cover video-VAE parallel decode and
        # video attention modes; the image VAE and dense attention have neither.
        if self._server_arg_value(server_args.backend) not in ("auto", "sglang"):
            raise ValueError("MiniMax H3 image requires the native SGLang backend")
        selected_backend = self.resolve_transformer_attention_backend(server_args)
        if selected_backend not in (None, AttentionBackendEnum.FA):
            raise ValueError("MiniMax H3 image requires dense FlashAttention")
        for name in ("nnodes", "dp_size"):
            value = getattr(server_args, name)
            if type(value) is not int or value != 1:
                raise ValueError(f"MiniMax H3 image requires {name}=1")
        # tp_size and ulysses_degree * ring_degree stay with the DiT's own
        # head/alignment admission checks; ring already has the FA it needs
        # because the image release forces FlashAttention below.
        if server_args.kv_gather_degree > 1 and not server_args.sp_split_auto:
            # The auto-assigned sp_degree=2 alias is fine: the H3 DiT exchanges
            # sequence shards with its own Ulysses all-to-all and never takes
            # the K/V-gather path, so an explicit request for it cannot be met.
            raise ValueError(
                "MiniMax H3 image does not implement K/V-gather sequence "
                "parallelism; use --ulysses-degree (or --ring-degree) instead"
            )
        for name in (
            "use_fsdp_inference",
            "comfyui_mode",
        ):
            if getattr(server_args, name):
                raise ValueError(f"MiniMax H3 image does not support {name}")
        if server_args.minimax_h3_adaln_cache_path is not None:
            raise ValueError("MiniMax H3 image does not support an AdaLN cache")
        online_quantizations = (None, "fp8", "mxfp8")
        if server_args.quantization not in online_quantizations or any(
            name not in ("transformer", "text_encoder")
            or value not in online_quantizations
            for name, value in server_args.component_quantizations.items()
        ):
            raise ValueError(
                "MiniMax H3 image supports online fp8/mxfp8 for the transformer "
                "and text_encoder; the image VAE retains native precision"
            )
        if server_args.lora_path is not None:
            raise ValueError("MiniMax H3 image does not support LoRA")
        if server_args.model_variant is not None:
            raise ValueError(
                "MiniMax H3 image selects t2i/edit from release metadata, not model_variant"
            )
        precisions = {
            "transformer": "bf16",
            "text_encoder": "bf16",
            "video_vae": "fp32",
        }
        if any(
            precisions.get(name) != value
            for name, value in server_args.component_precisions.items()
        ) or (
            self.dit_precision,
            self.text_encoder_precisions,
            self.vae_precision,
            self.vae_decode_precision,
            self.vae_decode_precision_high,
            self.vae_sp,
            self.flow_shift,
            self.scheduler_class_override,
        ) != ("bf16", ("bf16",), "fp32", "bf16", None, False, None, None):
            raise ValueError(
                f"MiniMax H3 image requires precisions {precisions!r} with BF16 VAE "
                "decode, local VAE and the image schedule"
            )
        self.vae_config.post_init()
        self._configure_online_adaln(server_args)
        component_backend, _ = server_args.resolve_component_attention_backend(
            "transformer"
        )
        if component_backend is None:
            # Request FA for the transformer through the generic explicit-request
            # path, so the loader refuses a transformer that does not run FA.
            # Explicit global/component/forced choices were checked above.
            server_args.component_attention_backends["transformer"] = "fa"
            server_args._requested_component_attention_backends["transformer"] = "fa"

    def _configure_online_adaln(self, server_args) -> None:
        # Measured H200/B300, GB10 and RTX PRO 6000 Blackwell recipes avoid keeping
        # 24.2 GiB of constant AdaLN projections resident. Explicit choices and
        # custom weights win.
        if (
            server_args.minimax_h3_adaln_online is None
            and server_args.performance_mode == "auto"
            and server_args.transformer_weights_path is None
            and not server_args.component_paths.get("transformer")
            and not server_args.component_weights_paths.get("transformer")
            and not envs.SGLANG_DIFFUSION_MINIMAX_H3_ADALN_FP32
            and current_platform.is_cuda()
        ):
            gpu_ids = server_args.gpu_ids or range(
                server_args.base_gpu_id, server_args.base_gpu_id + server_args.num_gpus
            )
            device_names = [torch.cuda.get_device_name(i) for i in gpu_ids]
            if any(
                all(name.startswith(family) for name in device_names)
                for family in (
                    "NVIDIA B300",
                    "NVIDIA H200",
                    "NVIDIA GB10",
                    "NVIDIA RTX PRO 6000 Blackwell",
                )
            ):
                server_args.minimax_h3_adaln_online = True
        if server_args.minimax_h3_adaln_online is None:
            server_args.minimax_h3_adaln_online = False
        if not server_args.minimax_h3_adaln_online:
            if (
                self._uses_gb10_shared_memory()
                and server_args.residency_mode("transformer") == RESIDENT
                and server_args.residency_mode("text_encoder") == RESIDENT
            ):
                raise ValueError(
                    "MiniMax H3 image resident GB10 requires online AdaLN "
                    "precomputation to fit the shared memory pool"
                )
            return
        if envs.SGLANG_DIFFUSION_MINIMAX_H3_ADALN_FP32:
            raise ValueError("MiniMax H3 image online AdaLN requires match precision")
        if server_args.minimax_h3_adaln_plan_width is None:
            server_args.minimax_h3_adaln_plan_width = 2
        if (
            server_args.minimax_h3_adaln_plan_width < 2
            and ModelTaskType.I2I in self.get_supported_task_types()
        ):
            raise ValueError("MiniMax H3 image editing needs AdaLN plan width >= 2")
        if (
            server_args.minimax_h3_adaln_host_cache_gb is None
            and self.model_path
            and envs.SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS
            >= self.release_pdd_steps() * len(self.get_supported_task_types())
        ):
            # Every request uses the fixed PDD grid. With all task plans
            # resident, none can be evicted and a host copy is unnecessary.
            server_args.minimax_h3_adaln_host_cache_gb = 0.0

    def select_vae_weight_files(
        self, safetensors_list, component_model_path, component_name, vae_precision
    ) -> list[str]:
        # The image VAE is one standard weight file, not the video source/ layout.
        return safetensors_list

    def validate_quality_deployment(self, server_args) -> None:
        raise ValueError("MiniMax H3 image does not support quality='high'")

    def edit_reference_short_edge_bounds(self, consumer: str) -> tuple[int, int]:
        """Validated ``(min, max)`` edit-reference short edge for ``consumer``
        (``"qwen"`` or ``"vae"``)."""
        if consumer not in ("qwen", "vae"):
            raise ValueError(
                f"edit reference consumer must be 'qwen' or 'vae', got {consumer!r}"
            )
        low_name = f"edit_reference_{consumer}_min_short_edge"
        high_name = f"edit_reference_{consumer}_max_short_edge"
        low = getattr(self, low_name)
        high = getattr(self, high_name)
        for name, value in ((low_name, low), (high_name, high)):
            # 32px: one Qwen merged vision token, one 2x2 patch of 16px VAE cells.
            if type(value) is not int or value < 32 or value % 32:
                raise ValueError(
                    f"{name} must be a positive multiple of 32, got {value!r}"
                )
        if low > high:
            raise ValueError(
                f"{low_name} must not exceed {high_name}, got {low} > {high}"
            )
        return low, high

    def check_pipeline_config(self) -> None:
        super().check_pipeline_config()
        for consumer in ("qwen", "vae"):
            self.edit_reference_short_edge_bounds(consumer)
        if self.min_generation_area is not None:
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.image_canvas import (
                minimax_h3_validate_min_generation_area,
            )

            minimax_h3_validate_min_generation_area(
                self.min_generation_area, "min_generation_area"
            )

    def get_cache_config(self) -> dict:
        raise ValueError("MiniMax H3 image does not support Cache-DiT")


__all__ = ["IMAGE_VAE_WEIGHTS_NAME", "MiniMaxH3ImagePipelineConfig"]


def register():
    from sglang.multimodal_gen.configs.sample.minimax_h3_image import (
        MiniMaxH3ImageSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=MiniMaxH3ImageSamplingParams,
        pipeline_config_cls=MiniMaxH3ImagePipelineConfig,
        # Path matching tries the longest registered id first, so the image repo
        # (and local dirs or caches named after it) is not taken for MiniMax-H3.
        hf_model_paths=["MiniMaxAI/MiniMax-H3-Image"],
        model_detectors=[
            lambda model_id: (
                model_id.lower().replace("-", "").replace("_", "")
                == "minimaxh3imagepipeline"
            )
        ],
        pipeline_config_registry_entries={
            "MiniMaxH3ImagePipeline": (
                MiniMaxH3ImagePipelineConfig,
                MiniMaxH3ImageSamplingParams,
            )
        },
    )
