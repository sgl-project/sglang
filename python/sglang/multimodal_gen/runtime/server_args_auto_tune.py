"""
ServerArgsAutoTuner tunes the ServerArgs based on the desired performance mode
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = init_logger(__name__)

PERFORMANCE_MODES = ("manual", "auto", "speed", "memory")


class ServerArgsAutoTuner:
    """Auto-tunes the server-arg for the given performance-mode, based on practical deployment experience with different model architectures"""

    def __init__(self, server_args: "ServerArgs"):
        self.server_args = server_args
        self._explicit_memory_policy = self._has_explicit_memory_policy()

    def _deployment_config(self) -> ModelDeploymentConfig:
        return self.server_args.pipeline_config.get_model_deployment_config()

    def adjust(self) -> None:
        """Adjust the server args based on the performance mode"""
        args = self.server_args
        args.performance_mode = self._normalize_performance_mode()

        if current_platform.is_cpu():
            return

        if args.performance_mode == "speed":
            logger.info("Applying performance_mode=speed")
            if args.num_gpus >= 2 and self._can_apply_fsdp_policy(
                require_memory_headroom=False
            ):
                self._set_gpu_resident_defaults(use_fsdp=True)
                self._enable_cfg_parallel_if_supported()
            else:
                self._set_gpu_resident_defaults(use_fsdp=False)
            return

        if args.performance_mode == "memory":
            logger.info("Applying performance_mode=memory")
            if args.use_fsdp_inference:
                self._set_gpu_resident_defaults(use_fsdp=True)
                return
            args.use_fsdp_inference = False
            if self._can_apply_dit_layerwise_offload_policy():
                # apply dit layerwise offload to save VRAM during denoising stage
                self._set_layerwise_offload_defaults()
            else:
                self._set_component_offload_defaults()
            return

    def maybe_adjust_auto_component_residency_after_offload(self) -> None:
        args = self.server_args
        if (
            args.performance_mode != "auto"
            or self._explicit_memory_policy
            or current_platform.is_cpu()
        ):
            return

        min_available_gb = self._get_min_available_device_memory_gb()
        disable_threshold_gb = (
            self._deployment_config().auto_disable_component_offload_min_available_memory_gb
        )
        if (
            min_available_gb is not None
            and disable_threshold_gb is not None
            and min_available_gb >= disable_threshold_gb
        ):
            changed = []
            components = (
                self._deployment_config().auto_disable_component_offload_components
            )
            if args._uses_ltx23_snapshot_two_stage_residency():
                # ltx2 snapshot mode uses DiT offload to release/prefetch stage DiTs between phases
                components = tuple(
                    component for component in components if component != "dit"
                )
            if args.dit_cpu_offload and "dit" in components:
                args.dit_cpu_offload = False
                changed.append("dit_cpu_offload=False")
            if args.text_encoder_cpu_offload and "text_encoder" in components:
                args.text_encoder_cpu_offload = False
                changed.append("text_encoder_cpu_offload=False")
            if args.image_encoder_cpu_offload and "image_encoder" in components:
                args.image_encoder_cpu_offload = False
                changed.append("image_encoder_cpu_offload=False")
            if changed:
                logger.info(
                    "Disabling component offload for %s because minimum available memory on selected GPUs is %.2f GiB: %s",
                    args.pipeline_config.__class__.__name__,
                    min_available_gb,
                    ", ".join(changed),
                )

    def maybe_adjust_auto_fsdp_with_offload_enabled(self) -> None:
        args = self.server_args
        if (
            args.performance_mode == "auto"
            and args.num_gpus >= 2
            and not self._explicit_memory_policy
            and self._auto_uses_dit_offload()
            and self._can_apply_fsdp_policy(require_memory_headroom=True)
        ):
            logger.info(
                "Automatically selecting FSDP defaults for multi-GPU %s to replace DiT offload",
                args.pipeline_config.__class__.__name__,
            )
            args.use_fsdp_inference = True
            if args.dit_cpu_offload:
                args.dit_cpu_offload = False
            if args.dit_layerwise_offload:
                args.dit_layerwise_offload = False
            self._enable_cfg_parallel_if_supported()

    def maybe_adjust_auto_dit_layerwise_offload(self) -> None:
        args = self.server_args
        if not self.could_override_server_args():
            return
        if self._explicit_memory_policy:
            return
        deployment_config = self._deployment_config()
        if envs.SGLANG_CACHE_DIT_ENABLED:
            return
        if (
            not deployment_config.auto_dit_layerwise_offload
            or args.dit_layerwise_offload is not None
        ):
            return
        if args.use_fsdp_inference:
            # if fsdp is enabled, layerwise-offload is weakened since the parameter has already been sharded
            args.dit_layerwise_offload = False
            return

        auto_enable_layerwise_offload = (
            current_platform.enable_dit_layerwise_offload_for_wan_by_default()
        )
        disable_threshold_gb = (
            deployment_config.auto_dit_layerwise_offload_high_memory_disable_gb
        )
        if (
            auto_enable_layerwise_offload
            and current_platform.is_cuda()
            and disable_threshold_gb is not None
        ):
            # auto turn off layerwise-offload if we have sufficient VRAM headroom
            device_total_memory_gb = current_platform.get_device_total_memory() / (
                1 << 30
            )
            if device_total_memory_gb >= disable_threshold_gb:
                logger.info(
                    "Skipping automatic dit_layerwise_offload for %s on a high-memory CUDA GPU (e.g. H200/B200/B300-class, %.2f GiB total)",
                    args.pipeline_config.__class__.__name__,
                    device_total_memory_gb,
                )
                auto_enable_layerwise_offload = False
                args.dit_layerwise_offload = False

        if auto_enable_layerwise_offload:
            logger.info(
                "Automatically enable dit_layerwise_offload for %s for low memory and performance balance",
                args.pipeline_config.__class__.__name__,
            )
            args.dit_layerwise_offload = True
            args.dit_cpu_offload = False

    def finalize_auto_flags(self) -> None:
        """if some args are unset after all the adjustment, set them to defaults"""
        if not self.could_override_server_args():
            return
        args = self.server_args
        if args.use_fsdp_inference is None:
            args.use_fsdp_inference = False
        if args.dit_cpu_offload is None:
            args.dit_cpu_offload = False
        if args.dit_layerwise_offload is None:
            args.dit_layerwise_offload = False
        if args.text_encoder_cpu_offload is None:
            args.text_encoder_cpu_offload = False
        if args.image_encoder_cpu_offload is None:
            args.image_encoder_cpu_offload = False

    def _normalize_performance_mode(self) -> str:
        args = self.server_args
        mode = (args.performance_mode or "auto").lower()
        if mode not in PERFORMANCE_MODES:
            valid_modes = PERFORMANCE_MODES
            raise ValueError(
                f"Invalid performance_mode={args.performance_mode!r}. "
                f"Expected one of {valid_modes}."
            )
        return mode

    def could_override_server_args(self) -> bool:
        return self.server_args.performance_mode != "manual"

    def _set_gpu_resident_defaults(self, *, use_fsdp: bool) -> None:
        """set all components to be resident"""
        args = self.server_args
        changed = []
        if args.use_fsdp_inference is None:
            args.use_fsdp_inference = use_fsdp
            changed.append(f"use_fsdp_inference={use_fsdp}")
        if args.dit_cpu_offload is None:
            args.dit_cpu_offload = False
            changed.append("dit_cpu_offload=False")
        if args.dit_layerwise_offload is None:
            args.dit_layerwise_offload = False
            changed.append("dit_layerwise_offload=False")
        if args.text_encoder_cpu_offload is None:
            args.text_encoder_cpu_offload = False
            changed.append("text_encoder_cpu_offload=False")
        if args.image_encoder_cpu_offload is None:
            args.image_encoder_cpu_offload = False
            changed.append("image_encoder_cpu_offload=False")

        if changed:
            logger.debug(
                "Applied GPU-resident performance defaults: %s", ", ".join(changed)
            )

    def _set_component_offload_defaults(self) -> None:
        args = self.server_args
        changed = []
        if args.dit_cpu_offload is None:
            args.dit_cpu_offload = True
            changed.append("dit_cpu_offload=True")
        if args.text_encoder_cpu_offload is None:
            args.text_encoder_cpu_offload = True
            changed.append("text_encoder_cpu_offload=True")
        if args.image_encoder_cpu_offload is None:
            args.image_encoder_cpu_offload = True
            changed.append("image_encoder_cpu_offload=True")
        if args.use_fsdp_inference is None:
            args.use_fsdp_inference = False
            changed.append("use_fsdp_inference=False")

        if changed:
            logger.info(
                "Applied low-memory component offload defaults: %s",
                ", ".join(changed),
            )

    def _set_layerwise_offload_defaults(self) -> None:
        args = self.server_args
        if args.dit_layerwise_offload is None:
            args.dit_layerwise_offload = True
        if args.dit_cpu_offload is None:
            args.dit_cpu_offload = False
        if args.text_encoder_cpu_offload is None:
            args.text_encoder_cpu_offload = True
        if args.image_encoder_cpu_offload is None:
            args.image_encoder_cpu_offload = True

    def _can_apply_dit_layerwise_offload_policy(self) -> bool:
        return (
            self._deployment_config().auto_dit_layerwise_offload
            and not envs.SGLANG_CACHE_DIT_ENABLED
            and current_platform.enable_dit_layerwise_offload_for_wan_by_default()
        )

    def _auto_uses_dit_offload(self) -> bool:
        args = self.server_args
        return bool(args.dit_cpu_offload or args.dit_layerwise_offload)

    def _get_min_available_device_memory_gb(self) -> float | None:
        args = self.server_args
        if current_platform.is_cpu():
            return None

        # Multi-GPU defaults are limited by the least-free selected GPU.
        return min(
            current_platform.get_available_gpu_memory(
                device_id=device_id,
                empty_cache=False,
            )
            for device_id in range(
                args.base_gpu_id, args.base_gpu_id + max(1, args.num_gpus)
            )
        )

    def _has_explicit_memory_policy(self) -> bool:
        args = self.server_args
        return (
            args.use_fsdp_inference is not None
            or args.dit_cpu_offload is not None
            or args.dit_layerwise_offload is not None
            or args.text_encoder_cpu_offload is not None
            or args.image_encoder_cpu_offload is not None
        )

    def _has_explicit_parallel_policy(self) -> bool:
        args = self.server_args
        return (
            args.tp_size is not None
            or args.sp_degree is not None
            or args.ulysses_degree is not None
            or args.ring_degree is not None
            or args.enable_cfg_parallel is not None
        )

    def _enable_cfg_parallel_if_supported(self) -> None:
        args = self.server_args
        if (
            args.enable_cfg_parallel is None
            and not self._has_explicit_parallel_policy()
            and args._model_default_uses_cfg()
        ):
            args.enable_cfg_parallel = True

    def _supports_high_confidence_fsdp(self) -> bool:
        deployment_config = self._deployment_config()
        return deployment_config.fsdp_auto_min_available_memory_gb is not None and (
            not deployment_config.fsdp_auto_requires_cfg
            or self.server_args._model_default_uses_cfg()
        )

    def _has_enough_available_memory_for_fsdp(self) -> bool:
        args = self.server_args
        min_available_gb = self._get_min_available_device_memory_gb()
        if min_available_gb is None:
            return True

        required_gb = self._deployment_config().fsdp_auto_min_available_memory_gb
        if required_gb is None:
            return False
        if min_available_gb < required_gb:
            logger.info(
                "Skipping automatic FSDP defaults: minimum available memory on selected GPUs %.2f GiB is below %.2f GiB for %s",
                min_available_gb,
                required_gb,
                args.pipeline_config.__class__.__name__,
            )
            return False
        return True

    def _can_apply_fsdp_policy(self, *, require_memory_headroom: bool) -> bool:
        args = self.server_args
        deployment_config = self._deployment_config()
        if not self._supports_high_confidence_fsdp():
            return False
        if envs.SGLANG_CACHE_DIT_ENABLED:
            logger.info("Skipping automatic FSDP defaults because cache-dit is enabled")
            return False
        if (
            args.performance_mode == "auto"
            and deployment_config.fsdp_auto_requires_default_parallelism
            and self._has_explicit_parallel_policy()
        ):
            logger.info(
                "Skipping automatic FSDP defaults because an explicit parallel policy is set"
            )
            return False
        return (
            not require_memory_headroom or self._has_enough_available_memory_for_fsdp()
        )
