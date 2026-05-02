import math
import os

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    LTX2PipelineConfig,
    is_ltx23_native_variant,
    sync_ltx23_runtime_vae_markers,
)
from sglang.multimodal_gen.configs.sample.ltx_2 import LTX23HQSamplingParams
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import BYTES_PER_GB
from sglang.multimodal_gen.runtime.managers.component_manager import (
    ComponentResidencyStrategy,
    ComponentUse,
    ResidencyState,
)
from sglang.multimodal_gen.runtime.managers.component_resident_strategies import (
    SnapshotModuleResidency,
    SnapshotStrategy,
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
    LTX2ImageEncodingStage,
    LTX2LoRASwitchStage,
    LTX2RefinementStage,
    LTX2TextConnectorStage,
    LTX2UpsampleStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import (
    LTX2_RESIDENT_AUTO_ENABLE_MEM_GB,
    ServerArgs,
)
from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var
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
    number_of_tokens: int | None = None,
) -> list[float]:
    sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)

    mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
    b = base_shift - mm * BASE_SHIFT_ANCHOR
    tokens = (
        int(number_of_tokens)
        if number_of_tokens is not None
        else int(default_number_of_tokens)
    )
    sigma_shift = float(tokens) * mm + b

    non_zero_mask = sigmas != 0
    shifted = torch.where(
        non_zero_mask,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1.0 / sigmas - 1.0)),
        torch.zeros_like(sigmas),
    )

    if stretch:
        one_minus_z = 1.0 - shifted[non_zero_mask]
        if bool(torch.any(one_minus_z != 0)):
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            shifted[non_zero_mask] = 1.0 - (one_minus_z / scale_factor)

    return shifted[:-1].tolist()


class LTX2SigmaPreparationStage(PipelineStage):
    """Prepare native LTX-2 sigma schedule before timestep setup."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch.extra["ltx2_phase"] = "stage1"
        if is_ltx23_native_variant(server_args.pipeline_config.vae_config.arch_config):
            # Gate on pipeline class to mirror the three official entry points:
            # - HQ (`ti2vid_two_stages_hq.py:164`) calls
            #   `LTX2Scheduler.execute(latent=empty_latent, ...)` where
            #   `empty_latent` is built from the **half-resolution** stage-1
            #   shape → resolution-aware sigma shift.
            # - Non-HQ two-stage (`ti2vid_two_stages.py:145`) and
            #   one-stage (`ti2vid_one_stage.py:138`) call
            #   `LTX2Scheduler.execute(steps=...)` with no `latent` →
            #   falls back to `default_number_of_tokens = MAX_SHIFT_ANCHOR
            #   = 4096` → constant-anchor sigma shift.
            if server_args.pipeline_class_name == "LTX2TwoStageHQPipeline":
                # batch.height/width have already been halved by
                # LTX2HalveResolutionStage, so these latents are the
                # half-resolution stage-1 shape (matches `empty_latent`).
                latent_num_frames = (int(batch.num_frames) - 1) // int(
                    server_args.pipeline_config.vae_temporal_compression
                ) + 1
                latent_height = int(batch.height) // int(
                    server_args.pipeline_config.vae_scale_factor
                )
                latent_width = int(batch.width) // int(
                    server_args.pipeline_config.vae_scale_factor
                )
                batch.sigmas = build_official_ltx2_sigmas(
                    int(batch.num_inference_steps),
                    number_of_tokens=latent_num_frames * latent_height * latent_width,
                )
            else:
                batch.sigmas = build_official_ltx2_sigmas(
                    int(batch.num_inference_steps)
                )
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


def _add_ltx2_stage1_generation_stages(
    pipeline: ComposedPipelineBase,
    *,
    denoising_sampler_name: str = "euler",
):
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
                sampler_name=denoising_sampler_name,
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


class LTX2TwoStageResidencyStrategy(ComponentResidencyStrategy):
    name = "ltx2_original"

    def __init__(self, manager: "LTX2TwoStageResidencyController") -> None:
        self.manager = manager

    @property
    def pipeline(self) -> "LTX2TwoStagePipeline":
        return self.manager.pipeline

    @property
    def server_args(self) -> ServerArgs:
        return self.manager.server_args

    def _phase(self, use: ComponentUse) -> str:
        if use.phase in ("stage1", "stage2"):
            return use.phase
        return "stage2" if use.component_name == "transformer_2" else "stage1"

    def initialize(self) -> None:
        pass

    def prepare_for_use(
        self,
        module: torch.nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        phase = self._phase(use)
        if phase != self.manager._active_phase:
            self.enter_phase(phase)

    def wait_for_use(
        self,
        module: torch.nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self.ensure_phase_ready(self._phase(use))

    def finish_use(
        self,
        module: torch.nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        self.exit_phase(self._phase(use))

    def prepare_after_request(
        self,
        module: torch.nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        phase = self._phase(use)
        if phase != self.manager._active_phase:
            self.enter_phase(phase)

    def enter_phase(self, phase: str) -> bool:
        return False

    def exit_phase(self, phase: str | None, next_phase: str | None = None) -> None:
        pass

    def ensure_phase_ready(self, phase: str | None) -> None:
        """wait for the preparation to be ready"""
        pass

    def _ensure_on_gpu(self, module_name: str) -> None:
        module = self.pipeline.get_module(module_name)
        if module is None:
            return
        param = next(module.parameters(), None)
        if param is not None and param.device.type == "cpu":
            module.to(get_local_torch_device(), non_blocking=True)

    @staticmethod
    def _module_is_on_gpu(module: torch.nn.Module | None) -> bool:
        return SnapshotModuleResidency.is_on_gpu(module)


class LTX2OriginalResidencyStrategy(LTX2TwoStageResidencyStrategy):
    pass


class LTX2ResidentResidencyStrategy(LTX2TwoStageResidencyStrategy):
    """A residency strategy for ltx two-stage pipeline with pre-merged lora, that keep both dits always resident"""

    name = "ltx2_resident"

    def initialize(self) -> None:
        self._ensure_on_gpu("transformer")
        self._ensure_on_gpu("transformer_2")
        logger.info(
            "Using resident LTX-2.3 two-stage transformers mode (both DiTs stay on GPU)"
        )
        self.manager._active_phase = "stage1"
        self.manager._sync_refinement_stage_transformer("stage1")

    def enter_phase(self, phase: str) -> bool:
        self.manager._sync_refinement_stage_transformer(phase)
        self.manager._active_phase = phase
        return True


class LTX2SnapshotResidencyStrategy(LTX2TwoStageResidencyStrategy):
    """
    Snapshot mode keeps CPU snapshots and prefetches the target DiT with async H2D. (only with pre-merged lora enabled)

    The DiT_1 will always be kept a replica in CPU.
    - default snapshot behavior: allow stage1/stage2 overlap by prefetching
      stage2 while stage1 is still running.
    - snapshot low-VRAM behavior (`_snapshot_low_vram_mode=True`): evict
      stage1 before stage2 prefetch and disable early overlap prefetch to
      reduce peak VRAM, at the cost of higher phase-switch latency.
    - default toggle: low-VRAM auto-enables on H100-like (<130 GiB) CUDA
      GPUs, and stays disabled by default on higher-memory GPUs. It can be
      overridden with `SGLANG_LTX2_SNAPSHOT_LOW_VRAM_MODE`.
    """

    name = "ltx2_snapshot"

    def __init__(self, manager: "LTX2TwoStageResidencyController") -> None:
        super().__init__(manager)
        self._snapshot_strategy = SnapshotStrategy(
            pin_cpu_memory=manager.server_args.pin_cpu_memory,
            enable_async_prefetch=manager.server_args.dit_cpu_offload,
        )
        self._snapshot_low_vram_mode = self._resolve_snapshot_low_vram_mode()
        self._snapshot_release_empty_cache = get_bool_env_var(
            "SGLANG_LTX2_SNAPSHOT_RELEASE_EMPTY_CACHE",
            default="false",
        )

    @staticmethod
    def _module_name_for_phase(phase: str | None) -> str | None:
        if phase == "stage1":
            return "transformer"
        if phase == "stage2":
            return "transformer_2"
        return None

    def _resolve_snapshot_low_vram_mode(self) -> bool:
        if not current_platform.is_cuda():
            return False
        device_name = str(current_platform.get_device_name(0)).upper()
        device_total_memory_gb = (
            current_platform.get_device_total_memory() / BYTES_PER_GB
        )
        # H100-class (<130 GiB) cards are sensitive to stage1/stage2 overlap windows.
        h100_like_memory_class = (
            "H100" in device_name
            or device_total_memory_gb < LTX2_RESIDENT_AUTO_ENABLE_MEM_GB
        )
        default = "true" if h100_like_memory_class else "false"
        enabled = get_bool_env_var(
            "SGLANG_LTX2_SNAPSHOT_LOW_VRAM_MODE",
            default=default,
        )
        if enabled:
            logger.info(
                "Enabled LTX2 snapshot low-VRAM mode "
                "(SGLANG_LTX2_SNAPSHOT_LOW_VRAM_MODE=%s, device=%s, %.2f GiB total)",
                os.getenv("SGLANG_LTX2_SNAPSHOT_LOW_VRAM_MODE", default),
                device_name,
                device_total_memory_gb,
            )
        return enabled

    def initialize(self) -> None:
        # Snapshot mode keeps both DiT CPU snapshots for cheap GPU release
        # and re-hydrates stage-2 with async H2D when stage-1 finishes.
        self._capture_module_cpu_snapshot("transformer")
        self._capture_module_cpu_snapshot("transformer_2")
        self._pin_stage1_transformer_if_beneficial()
        self.manager._sync_refinement_stage_transformer("stage1")
        self._record_component_ready("transformer")

    def enter_phase(self, phase: str) -> bool:
        if self.server_args.dit_cpu_offload:
            target_module_name = self._module_name_for_phase(phase)
            if target_module_name is None:
                return False
            target_module = self.pipeline.get_module(target_module_name)
            if self._snapshot_low_vram_mode:
                # Trade a bit of phase-switch latency for lower peak VRAM:
                # evict stage-1 before stage-2 H2D.
                if phase == "stage2" and not self._snapshot_strategy.is_ready(
                    target_module_name
                ):
                    self._release_stage1_for_low_vram()

            # make sure the component is pre-fetched
            if not self._snapshot_strategy.is_ready(target_module_name):
                if self._module_is_on_gpu(target_module):
                    self._record_component_ready(target_module_name)
                else:
                    self._snapshot_strategy.prefetch_component(
                        target_module_name, target_module
                    )
        else:
            component_name = self._module_name_for_phase(phase)
            if component_name is not None:
                self._record_component_ready(component_name)

        self.manager._sync_refinement_stage_transformer(phase)
        self.manager._active_phase = phase
        return True

    def prepare_after_request(
        self,
        module: torch.nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        phase = self._phase(use)
        if phase != "stage1":
            return
        if self.server_args.dit_cpu_offload:
            target_module = self.pipeline.get_module("transformer")
            if self._module_is_on_gpu(target_module):
                self._record_component_ready("transformer")
            elif not self._snapshot_strategy.is_ready("transformer"):
                self._snapshot_strategy.prefetch_component("transformer", target_module)
        else:
            self._record_component_ready("transformer")
        self.manager._sync_refinement_stage_transformer("stage1")
        self.manager._active_phase = "stage1"

    def finish_use(
        self,
        module: torch.nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> None:
        phase = self._phase(use)
        if self.server_args.dit_cpu_offload:
            # release cuda storage
            self._snapshot_strategy.release_component(use.component_name, module)
        if (
            phase == "stage2"
            and self._snapshot_release_empty_cache
            and torch.get_device_module().is_available()
        ):
            torch.get_device_module().empty_cache()

    def ensure_phase_ready(self, phase: str | None) -> None:
        component_name = self._module_name_for_phase(phase)
        if component_name is None:
            return
        self._snapshot_strategy.wait_component_ready(component_name)

    def _capture_module_cpu_snapshot(self, module_name: str) -> None:
        module = self.pipeline.get_module(module_name)
        if module is None:
            raise ValueError(f"Module {module_name} is not available.")
        self._snapshot_strategy.capture(module_name, module)

    def _release_module_to_cpu_snapshot(self, module_name: str) -> None:
        module = self.pipeline.get_module(module_name)
        if module is None:
            return
        self._snapshot_strategy.release_component(module_name, module)

    def _release_stage1_for_low_vram(self) -> None:
        stage1_module = self.pipeline.get_module("transformer")
        stage1_param = (
            next(stage1_module.parameters(), None)
            if stage1_module is not None
            else None
        )
        if stage1_param is not None and stage1_param.device.type == "cuda":
            self._release_module_to_cpu_snapshot("transformer")

    def _record_component_ready(self, module_name: str) -> None:
        self._snapshot_strategy.record_ready(
            module_name, self.pipeline.get_module(module_name)
        )

    def prefetch_for_use(
        self,
        module: torch.nn.Module,
        use: ComponentUse,
        state: ResidencyState,
    ) -> bool:
        if not self.server_args.dit_cpu_offload:
            return True
        phase = self._phase(use)
        if phase == "stage2":
            if self._snapshot_strategy.is_ready("transformer_2"):
                return True
            if self._snapshot_low_vram_mode and state.current_use is not None:
                return False
            if self._snapshot_low_vram_mode:
                self._release_stage1_for_low_vram()
        self._snapshot_strategy.prefetch_component(use.component_name, module)
        return True

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
        self.manager._active_phase = "stage1"


class LTX2TwoStageResidencyController:
    """
    LTX-2.3 two-stage residency controller.
    It builds the selected LTX2 ComponentResidencyStrategy and keeps the
    thin stage adapter methods that are specific to two-stage LoRA flow.

    Modes:
    - resident: keep both DiTs on GPU; phase switch is pointer rebinding only.
    - snapshot: keep CPU snapshots and prefetch the target DiT.
    - original: official two-stage semantics without premerged stage-2.
    """

    VALID_MODES = ("original", "snapshot", "resident")

    def __init__(self, pipeline: "LTX2TwoStagePipeline", server_args: ServerArgs):
        self.pipeline = pipeline
        self.server_args = server_args
        self.mode = self._resolve_mode(server_args)
        self._active_phase: str | None = None
        self._strategy = self._build_strategy()

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

    def _build_strategy(self) -> LTX2TwoStageResidencyStrategy:
        if self.mode == "snapshot":
            return LTX2SnapshotResidencyStrategy(self)
        if self.mode == "resident":
            return LTX2ResidentResidencyStrategy(self)
        return LTX2OriginalResidencyStrategy(self)

    @property
    def strategy(self) -> ComponentResidencyStrategy:
        return self._strategy

    @property
    def should_use_premerged(self) -> bool:
        """Whether to keep a pre-merged stage-2 DiT for LTX-2.3 two-stage.

        We only enable this optimization for native LTX-2.3 two-stage and when
        users did not explicitly provide a stage-1 LoRA path
        """
        return (
            self.mode != "original"
            and self.pipeline._should_merge_stage2_distilled_lora(self.server_args)
            and self.pipeline._stage1_lora_path is None
        )

    def initialize(self) -> None:
        if not self.should_use_premerged:
            return
        self.pipeline._initialize_premerged_stage2_transformer(self.server_args)
        self._strategy.initialize()

    def enter_phase(self, phase: str) -> bool:
        """Switch active two-stage DiT with minimal transfer/sync overhead."""
        if not self.should_use_premerged:
            return False
        if phase == self._active_phase:
            return True
        return self._strategy.enter_phase(phase)

    def _sync_refinement_stage_transformer(self, phase: str) -> None:
        """Keep stage-2 refinement bound to the expected DiT for current phase."""
        refinement_stage = self.pipeline.get_stage("LTX2RefinementStage")
        if refinement_stage is None:
            return
        target_name = "transformer_2" if phase == "stage2" else "transformer"
        target_transformer = self.pipeline.get_module(target_name)
        if target_transformer is not None:
            refinement_stage.transformer = target_transformer


class LTX2TwoStagePipeline(_BaseLTX2Pipeline):
    pipeline_name = "LTX2TwoStagePipeline"
    STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]
    STAGE_1_DISTILLED_LORA_STRENGTH = 0.0
    STAGE_2_DISTILLED_LORA_STRENGTH = 1.0
    STAGE_1_DENOISING_SAMPLER_NAME = "euler"
    STAGE_2_DENOISING_SAMPLER_NAME = "euler"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ltx2_residency = LTX2TwoStageResidencyController(self, self.server_args)
        self._use_premerged_stage2_transformer = (
            self._ltx2_residency.should_use_premerged
        )
        self._ltx2_residency.initialize()
        if self._use_premerged_stage2_transformer:
            self.component_residency_strategies["transformer"] = (
                self._ltx2_residency.strategy
            )
            self.component_residency_strategies["transformer_2"] = (
                self._ltx2_residency.strategy
            )

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
                f"{self.pipeline_name} requires --spatial-upsampler-path "
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
                f"{self.pipeline_name} requires --distilled-lora-path "
                "(component_paths['distilled_lora'])."
            )
        self._distilled_lora_path = distilled_lora_path
        self._stage1_lora_path = server_args.lora_path
        self._stage1_lora_scale = float(server_args.lora_scale)
        self._active_lora_phase = None
        self._active_lora_signature = None
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

        # Reuse the canonical LoRA path used by legacy switching to reduce
        # precision drift between snapshot mode and origin/main behavior.
        self.set_lora(
            lora_nickname="ltx2_stage2_distilled",
            lora_path=self._distilled_lora_path,
            target="transformer_2",
            strength=self.STAGE_2_DISTILLED_LORA_STRENGTH,
            merge_weights=True,
        )

    def should_skip_ltx2_lora_switch_stage(self) -> bool:
        return self._use_premerged_stage2_transformer and self._ltx2_residency.mode in (
            "snapshot",
            "resident",
        )

    def _get_stage_distilled_lora_strength(
        self, phase: str, batch: Req | None
    ) -> float:
        if phase == "stage1":
            default_strength = self.STAGE_1_DISTILLED_LORA_STRENGTH
            extra_key = "ltx2_distilled_lora_strength_stage_1"
        elif phase == "stage2":
            default_strength = self.STAGE_2_DISTILLED_LORA_STRENGTH
            extra_key = "ltx2_distilled_lora_strength_stage_2"
        else:
            raise ValueError(f"Unknown LTX2 two-stage LoRA phase: {phase}")

        if batch is None:
            return float(default_strength)

        request_strength = batch.extra.get(extra_key)
        if request_strength is None:
            return float(default_strength)
        return float(request_strength)

    def _can_short_circuit_lora_switch(
        self, phase: str, batch: Req | None = None
    ) -> bool:
        distilled_lora_strength = self._get_stage_distilled_lora_strength(phase, batch)
        if phase == "stage1":
            return (
                self._use_premerged_stage2_transformer
                and self._stage1_lora_path is None
                and distilled_lora_strength == 0.0
            )
        if phase == "stage2":
            return (
                self._use_premerged_stage2_transformer
                and self._stage1_lora_path is None
                and distilled_lora_strength == self.STAGE_2_DISTILLED_LORA_STRENGTH
            )
        return False

    def _build_lora_switch_spec(
        self, phase: str, batch: Req | None = None
    ) -> tuple[list[str], list[str], list[float], list[str]]:
        distilled_lora_strength = self._get_stage_distilled_lora_strength(phase, batch)
        lora_nicknames: list[str] = []
        lora_paths: list[str] = []
        lora_strengths: list[float] = []
        lora_targets: list[str] = []

        if phase == "stage1":
            if self._stage1_lora_path:
                lora_nicknames.append("ltx2_stage1_base")
                lora_paths.append(self._stage1_lora_path)
                lora_strengths.append(self._stage1_lora_scale)
                lora_targets.append("transformer")
            if distilled_lora_strength != 0.0:
                lora_nicknames.append("ltx2_stage1_distilled")
                lora_paths.append(self._distilled_lora_path)
                lora_strengths.append(distilled_lora_strength)
                lora_targets.append("transformer")
        elif phase == "stage2":
            if self._stage1_lora_path:
                lora_nicknames.append("ltx2_stage1_base")
                lora_paths.append(self._stage1_lora_path)
                lora_strengths.append(self._stage1_lora_scale)
                lora_targets.append("transformer")
            if distilled_lora_strength != 0.0:
                lora_nicknames.append("ltx2_stage2_distilled")
                lora_paths.append(self._distilled_lora_path)
                lora_strengths.append(distilled_lora_strength)
                lora_targets.append("transformer")
        else:
            raise ValueError(f"Unknown LTX2 two-stage LoRA phase: {phase}")

        return lora_nicknames, lora_paths, lora_strengths, lora_targets

    def switch_lora_phase(self, phase: str, batch: Req | None = None) -> None:
        distilled_lora_strength = self._get_stage_distilled_lora_strength(phase, batch)
        phase_signature = (phase, distilled_lora_strength)
        if phase_signature == self._active_lora_signature:
            return

        if self._ltx2_residency.enter_phase(
            phase
        ) and self._can_short_circuit_lora_switch(phase, batch):
            self._active_lora_phase = phase
            self._active_lora_signature = phase_signature
            return

        lora_nicknames, lora_paths, lora_strengths, lora_targets = (
            self._build_lora_switch_spec(phase, batch)
        )
        if lora_nicknames:
            set_lora_kwargs = dict(
                lora_nickname=lora_nicknames,
                lora_path=lora_paths,
                target=lora_targets,
                strength=lora_strengths,
            )
            if phase == "stage2":
                # Official LTX-2.3 two-stage builds stage 2 with distilled LoRA fused
                # into the transformer weights. Legacy LTX-2 should keep the
                # preexisting unmerged behavior to avoid regressing stage 2 quality.
                set_lora_kwargs["merge_weights"] = (
                    self._should_merge_stage2_distilled_lora(self.server_args)
                )
            self.set_lora(
                **set_lora_kwargs,
            )
        else:
            # Stage 1 must run on the base transformer weights. If stage 2 left the
            # distilled adapter active, stage 1 quality drifts away from the official
            # two-stage pipeline immediately.
            self.deactivate_lora_weights(target="transformer")

        self._active_lora_phase = phase
        self._active_lora_signature = phase_signature

    def create_pipeline_stages(self, server_args: ServerArgs):
        _add_ltx2_front_stages(self)
        self.add_stage(LTX2HalveResolutionStage())
        self.add_stage(
            LTX2LoRASwitchStage(pipeline=self, phase="stage1"),
        )
        _add_ltx2_stage1_generation_stages(
            self,
            denoising_sampler_name=self.STAGE_1_DENOISING_SAMPLER_NAME,
        )
        self.add_stages(
            [
                LTX2UpsampleStage(
                    spatial_upsampler=self.get_module("spatial_upsampler"),
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                    pipeline=self,
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
                    sampler_name=self.STAGE_2_DENOISING_SAMPLER_NAME,
                ),
            ]
        )
        _add_ltx2_decoding_stage(self)


class LTX2TwoStageHQPipeline(LTX2TwoStagePipeline):
    pipeline_name = "LTX2TwoStageHQPipeline"
    pipeline_config_cls = LTX2PipelineConfig
    sampling_params_cls = LTX23HQSamplingParams
    STAGE_1_DISTILLED_LORA_STRENGTH = 0.25
    STAGE_2_DISTILLED_LORA_STRENGTH = 0.5
    STAGE_1_DENOISING_SAMPLER_NAME = "res2s"
    STAGE_2_DENOISING_SAMPLER_NAME = "res2s"


EntryClass = [LTX2Pipeline, LTX2TwoStagePipeline, LTX2TwoStageHQPipeline]
