import math
import os

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    is_ltx23_native_variant,
    sync_ltx23_runtime_vae_markers,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import BYTES_PER_GB
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
    LTX2ImageEncodingStage,
    LTX2LoRASwitchStage,
    LTX2RefinementStage,
    LTX2TextConnectorStage,
    LTX2UpsampleStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.platforms import current_platform
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
            os.path.join(model_path, "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
            os.path.join(model_path, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            os.path.join(model_path, "latent_upsampler"),
            os.path.join(model_path, "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
        ]
        for candidate in spatial_candidates:
            if os.path.exists(candidate):
                resolved["spatial_upsampler"] = candidate
                auto_resolved.append(f"spatial_upsampler={candidate}")
                break

    if "distilled_lora" not in resolved:
        distilled_lora_candidates = [
            os.path.join(model_path, "ltx-2.3-20b-distilled-lora-384.safetensors"),
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
    if is_ltx23_native_variant(server_args.pipeline_config.vae_config.arch_config):
        return "mu", None
    latent_num_frames = (int(batch.num_frames) - 1) // int(
        server_args.pipeline_config.vae_temporal_compression
    ) + 1
    latent_height = int(batch.height) // int(
        server_args.pipeline_config.vae_scale_factor
    )
    latent_width = int(batch.width) // int(server_args.pipeline_config.vae_scale_factor)
    video_sequence_length = latent_num_frames * latent_height * latent_width
    return "mu", calculate_ltx2_shift(video_sequence_length)


def build_official_ltx2_sigmas(
    steps: int,
    *,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
    default_number_of_tokens: int = MAX_SHIFT_ANCHOR,
) -> list[float]:
    sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)

    mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
    b = base_shift - mm * BASE_SHIFT_ANCHOR
    sigma_shift = float(default_number_of_tokens) * mm + b

    non_zero_mask = sigmas != 0
    shifted = torch.where(
        non_zero_mask,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1.0 / sigmas - 1.0)),
        torch.zeros_like(sigmas),
    )

    if stretch:
        one_minus_z = 1.0 - shifted[non_zero_mask]
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        shifted[non_zero_mask] = 1.0 - (one_minus_z / scale_factor)

    return shifted[:-1].tolist()


class LTX2SigmaPreparationStage(PipelineStage):
    """Prepare native LTX-2 sigma schedule before timestep setup."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch.extra["ltx2_phase"] = "stage1"
        if is_ltx23_native_variant(server_args.pipeline_config.vae_config.arch_config):
            batch.sigmas = build_official_ltx2_sigmas(int(batch.num_inference_steps))
        else:
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
            LTX2ImageEncodingStage(
                vae=pipeline.get_module("vae"),
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
        sync_ltx23_runtime_vae_markers(
            server_args.pipeline_config.vae_config.arch_config,
            getattr(self.get_module("vae"), "config", None),
        )


class LTX2Pipeline(_BaseLTX2Pipeline):
    # Must match model_index.json `_class_name`.
    pipeline_name = "LTX2Pipeline"

    def create_pipeline_stages(self, server_args: ServerArgs):
        _add_ltx2_front_stages(self)
        _add_ltx2_stage1_generation_stages(self)
        _add_ltx2_decoding_stage(self)


class LTX2TwoStageDeviceManager:
    """
    LTX2TwoStageDeviceManager provides three modes to manage dit weights for LTX2 two stage pipelines.
    The three modes are:
    - `resident`: keep both transformers on GPU and only update phase state. The most performant one, but requires the most VRAM
    - `snapshot`: release the previous DiT by rebinding to CPU snapshots, and lazily H2D the next DiT when needed
    - `legacy`: caller falls back to classic LoRA hot-switch path. The slowest one, but requires no additional VRAM
    """

    VALID_MODES = ("legacy", "snapshot", "resident")

    def __init__(self, pipeline: "LTX2TwoStagePipeline", server_args: ServerArgs):
        self.pipeline = pipeline
        self.server_args = server_args
        self.mode = self._resolve_mode(server_args)
        self._cpu_param_snapshots: dict[str, dict[str, torch.Tensor]] = {}
        self._cpu_buffer_snapshots: dict[str, dict[str, torch.Tensor]] = {}
        self._active_phase: str | None = None

    @classmethod
    def _resolve_mode(cls, server_args: ServerArgs) -> str:
        mode = server_args.ltx2_two_stage_device_mode
        if mode is None:
            env_mode = os.getenv("SGLANG_LTX2_TWO_STAGE_DEVICE_MODE")
            mode = env_mode.lower() if env_mode else "snapshot"
        if mode not in cls.VALID_MODES:
            raise ValueError(
                f"Invalid ltx2_two_stage_device_mode={mode!r}. "
                f"Expected one of {cls.VALID_MODES}."
            )
        return mode

    def should_use_premerged(self) -> bool:
        """Whether to keep a pre-merged stage-2 DiT for LTX-2.3 two-stage.

        We only enable this optimization for native LTX-2.3 two-stage and when
        users did not explicitly provide a stage-1 LoRA path
        """
        return (
            self.mode != "legacy"
            and self.pipeline._should_merge_stage2_distilled_lora(self.server_args)
            and getattr(self.pipeline, "_stage1_lora_path", None) is None
        )

    def initialize(self) -> None:
        if not self.should_use_premerged():
            return

        self.pipeline._initialize_premerged_stage2_transformer(self.server_args)
        if self.mode == "snapshot":
            self._capture_module_cpu_snapshot("transformer")
            self._capture_module_cpu_snapshot("transformer_2")
            self._pin_stage1_transformer_if_beneficial()
        elif self.mode == "resident":
            self._ensure_on_gpu("transformer")
            self._ensure_on_gpu("transformer_2")
            logger.info(
                "Using resident LTX-2.3 two-stage transformers mode (both DiTs stay on GPU)"
            )

        refinement_stage = self.pipeline.get_stage("LTX2RefinementStage")
        if refinement_stage is not None:
            refinement_stage.transformer = self.pipeline.get_module("transformer_2")

    def switch_phase(self, phase: str) -> bool:
        """Switch active two-stage DiT with minimal transfer/sync overhead."""
        if not self.should_use_premerged():
            return False
        if phase == self._active_phase:
            return True

        if self.mode == "resident":
            self._ensure_on_gpu("transformer")
            self._ensure_on_gpu("transformer_2")
            self._active_phase = phase
            return True

        if self.server_args.dit_cpu_offload:
            if phase == "stage1":
                previous_name, next_name = "transformer_2", "transformer"
            else:
                previous_name, next_name = "transformer", "transformer_2"

            previous_module = self.pipeline.get_module(previous_name)
            next_module = self.pipeline.get_module(next_name)
            prev_param = (
                next(previous_module.parameters(), None)
                if previous_module is not None
                else None
            )
            if prev_param is not None and prev_param.device.type == "cuda":
                self._release_module_to_cpu_snapshot(previous_name)
            next_param = (
                next(next_module.parameters(), None)
                if next_module is not None
                else None
            )
            if next_param is not None and next_param.device.type == "cpu":
                next_module.to(get_local_torch_device(), non_blocking=True)

        self._active_phase = phase
        return True

    def release_premerged_transformers(self) -> None:
        if not self.should_use_premerged() or self.mode != "snapshot":
            return
        for module_name in ("transformer", "transformer_2"):
            module = self.pipeline.get_module(module_name)
            param = next(module.parameters(), None) if module is not None else None
            if param is not None and param.device.type == "cuda":
                self._release_module_to_cpu_snapshot(module_name)
        if torch.get_device_module().is_available():
            torch.get_device_module().empty_cache()

    @staticmethod
    def _clone_cpu_tensor_snapshot(
        tensor: torch.Tensor, *, pin_memory: bool
    ) -> torch.Tensor:
        snapshot = tensor.detach()
        if snapshot.device.type == "cpu":
            if pin_memory and not snapshot.is_pinned():
                return snapshot.pin_memory()
            return snapshot

        cpu_tensor = snapshot.to("cpu")
        if pin_memory:
            return cpu_tensor.pin_memory()
        return cpu_tensor

    def _capture_module_cpu_snapshot(self, module_name: str) -> None:
        if module_name in self._cpu_param_snapshots:
            return

        module = self.pipeline.get_module(module_name)
        if module is None:
            raise ValueError(f"Module {module_name} is not available.")

        pin_memory = bool(
            self.server_args.pin_cpu_memory and torch.get_device_module().is_available()
        )
        self._cpu_param_snapshots[module_name] = {
            name: self._clone_cpu_tensor_snapshot(param.data, pin_memory=pin_memory)
            for name, param in module.named_parameters()
        }
        self._cpu_buffer_snapshots[module_name] = {
            name: self._clone_cpu_tensor_snapshot(buffer.data, pin_memory=pin_memory)
            for name, buffer in module.named_buffers()
        }

    def _release_module_to_cpu_snapshot(self, module_name: str) -> None:
        """Replace module tensors with cached CPU snapshots to avoid D2H copies.

        This does not call `module.to("cpu")`. Instead, parameter and buffer storages
        are rebound to pre-captured CPU tensors so CUDA storages can be released by
        the allocator without an explicit D2H transfer.
        """
        module = self.pipeline.get_module(module_name)
        if module is None:
            return

        param_snapshots = self._cpu_param_snapshots.get(module_name)
        buffer_snapshots = self._cpu_buffer_snapshots.get(module_name)
        if param_snapshots is None or buffer_snapshots is None:
            module.to("cpu")
            return

        for name, param in module.named_parameters():
            snapshot = param_snapshots.get(name)
            if snapshot is None:
                raise KeyError(
                    f"Missing CPU parameter snapshot for {module_name}.{name}"
                )
            param.data = snapshot

        for name, buffer in module.named_buffers():
            snapshot = buffer_snapshots.get(name)
            if snapshot is None:
                raise KeyError(f"Missing CPU buffer snapshot for {module_name}.{name}")
            # Preserve runtime-updated buffers (e.g., lazily built caches) when
            # releasing back to CPU snapshots.
            if buffer.device.type == "cuda":
                snapshot.copy_(buffer.detach().to(device="cpu", dtype=snapshot.dtype))
            elif buffer.device.type == "cpu":
                snapshot.copy_(buffer.detach().to(dtype=snapshot.dtype))
            buffer.data = snapshot

    def _ensure_on_gpu(self, module_name: str) -> None:
        module = self.pipeline.get_module(module_name)
        if module is None:
            return
        param = next(module.parameters(), None)
        if param is not None and param.device.type == "cpu":
            module.to(get_local_torch_device(), non_blocking=True)

    def _pin_stage1_transformer_if_beneficial(self) -> None:
        """Optionally pin stage-1 DiT on GPU to remove first-stage cold H2D stall.

        We only do this on high-VRAM CUDA machines with CPU offload enabled and
        without FSDP inference. It trades extra steady-state VRAM for lower
        request latency before the first denoise step.
        """
        if (
            not self.server_args.dit_cpu_offload
            or self.server_args.use_fsdp_inference
            or not current_platform.is_cuda()
            or current_platform.get_device_total_memory() / BYTES_PER_GB < 70
        ):
            return

        transformer = self.pipeline.get_module("transformer")
        param = (
            next(transformer.parameters(), None) if transformer is not None else None
        )
        if transformer is not None and param is not None and param.device.type == "cpu":
            transformer.to(get_local_torch_device(), non_blocking=True)
            logger.info(
                "Pinned stage1 transformer on GPU for LTX-2.3 two-stage startup"
            )
        self._active_phase = "stage1"


class LTX2TwoStagePipeline(_BaseLTX2Pipeline):
    pipeline_name = "LTX2TwoStagePipeline"
    STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device_manager = LTX2TwoStageDeviceManager(self, self.server_args)
        self._use_premerged_stage2_transformer = (
            self._device_manager.should_use_premerged()
        )
        self._device_manager.initialize()

    @staticmethod
    def _should_merge_stage2_distilled_lora(server_args: ServerArgs) -> bool:
        return is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        )

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
        self._use_premerged_stage2_transformer = False

    def _initialize_premerged_stage2_transformer(self, server_args: ServerArgs) -> None:
        transformer_path = self._resolve_component_path(
            server_args, "transformer", "transformer"
        )
        module, memory_usage = PipelineComponentLoader.load_component(
            component_name="transformer_2",
            component_model_path=transformer_path,
            transformers_or_diffusers="diffusers",
            server_args=server_args,
        )
        self.modules["transformer_2"] = module
        self.memory_usages["transformer_2"] = memory_usage

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if (
            self.loaded_adapter_paths.get("ltx2_stage2_distilled")
            != self._distilled_lora_path
        ):
            self.load_lora_adapter(
                self._distilled_lora_path,
                "ltx2_stage2_distilled",
                rank,
            )

        with self._temporarily_disable_offload(
            target="transformer_2", use_module_names_only=True
        ):
            converted_count = self.convert_module_lora_layers(
                self.modules["transformer_2"],
                "transformer_2",
                self.lora_layers_transformer_2,
                check_exclude=True,
            )
            logger.info(
                "Converted %d layers to LoRA layers in transformer_2",
                converted_count,
            )
            self._apply_lora_to_layers(
                self.lora_layers_transformer_2,
                ["ltx2_stage2_distilled"],
                [self._distilled_lora_path],
                rank,
                [1.0],
                clear_existing=True,
                merge_weights=True,
            )

        self.is_lora_merged["transformer_2"] = True
        self.cur_adapter_name["transformer_2"] = "ltx2_stage2_distilled"
        self.cur_adapter_path["transformer_2"] = self._distilled_lora_path
        self.cur_adapter_strength["transformer_2"] = 1.0
        self.cur_adapter_config["transformer_2"] = (
            ["ltx2_stage2_distilled"],
            [1.0],
        )

    def release_premerged_transformers_to_cpu_snapshots(self) -> None:
        """Release inactive premerged DiTs according to the selected device mode."""
        self._device_manager.release_premerged_transformers()

    def switch_lora_phase(self, phase: str) -> None:
        if phase == self._active_lora_phase:
            return

        if self._device_manager.switch_phase(phase):
            self._active_lora_phase = phase
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
                # Official LTX-2.3 two-stage builds stage 2 with distilled LoRA fused
                # into the transformer weights. Legacy LTX-2 should keep the
                # preexisting unmerged behavior to avoid regressing stage 2 quality.
                merge_weights=self._should_merge_stage2_distilled_lora(
                    self.server_args
                ),
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
                (
                    LTX2ImageEncodingStage(
                        vae=self.get_module("vae"),
                    ),
                    "ltx2_image_encoding_stage2",
                ),
                LTX2RefinementStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    distilled_sigmas=self.STAGE_2_DISTILLED_SIGMA_VALUES,
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                    pipeline=self,
                ),
            ]
        )
        _add_ltx2_decoding_stage(self)


EntryClass = [LTX2Pipeline, LTX2TwoStagePipeline]
