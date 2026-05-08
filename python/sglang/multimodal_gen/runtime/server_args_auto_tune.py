"""
    ServerArgsAutoTuner tunes the ServerArgs based on the desired performance mode
"""


from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = init_logger(__name__)

PERFORMANCE_MODES = ("auto", "speed", "memory", "balanced")


class ServerArgsAutoTuner:
    def __init__(self, server_args: "ServerArgs"):
        self.server_args = server_args

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
            if args.num_gpus >= 2 and self._can_apply_fsdp_cfg_policy(
                require_memory_headroom=False
            ):
                self._set_gpu_resident_defaults(use_fsdp=True)
                self._enable_cfg_parallel_if_unset()
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

        if args.performance_mode == "balanced":
            logger.info("Applying performance_mode=balanced")
            if args.num_gpus >= 2 and self._can_apply_fsdp_cfg_policy(
                require_memory_headroom=True
            ):
                self._set_gpu_resident_defaults(use_fsdp=True)
                self._enable_cfg_parallel_if_unset()
            return

        if (
            args.num_gpus >= 2
            and not self._has_explicit_memory_policy()
            and not self._has_explicit_parallel_policy()
            and self._can_apply_fsdp_cfg_policy(require_memory_headroom=True)
        ):
            logger.info(
                "Automatically selecting FSDP+CFG defaults for multi-GPU Qwen/Wan CFG model"
            )
            self._set_gpu_resident_defaults(use_fsdp=True)
            args.enable_cfg_parallel = True

    def adjust_auto_dit_layerwise_offload(self) -> None:
        args = self.server_args
        deployment_config = self._deployment_config()
        if envs.SGLANG_CACHE_DIT_ENABLED:
            return
        if (
            not deployment_config.auto_dit_layerwise_offload
            or args.dit_layerwise_offload is not None
        ):
            return
        if args.use_fsdp_inference:
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

    def finalize_auto_flags(self) -> None:
        """if some args are unset after all the adjustment, set them to defaults"""
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

    def _enable_cfg_parallel_if_unset(self) -> None:
        args = self.server_args
        if args.enable_cfg_parallel is None:
            args.enable_cfg_parallel = True

    def _supports_high_confidence_fsdp_cfg(self) -> bool:
        """wheter applying fsdp cfg will very-likely to bring performance gain"""
        args = self.server_args
        return (
            self._deployment_config().fsdp_cfg_auto_min_available_memory_gb is not None
            and args._model_default_uses_cfg()
        )

    def _has_enough_available_memory_for_fsdp_cfg(self) -> bool:
        args = self.server_args
        min_available_gb = self._get_min_available_device_memory_gb()
        if min_available_gb is None:
            return True

        required_gb = self._deployment_config().fsdp_cfg_auto_min_available_memory_gb
        if required_gb is None:
            return False
        if min_available_gb < required_gb:
            logger.info(
                "Skipping automatic FSDP+CFG defaults: minimum available memory on selected GPUs %.2f GiB is below %.2f GiB for %s",
                min_available_gb,
                required_gb,
                args.pipeline_config.__class__.__name__,
            )
            return False
        return True

    def _can_apply_fsdp_cfg_policy(self, *, require_memory_headroom: bool) -> bool:
        if not self._supports_high_confidence_fsdp_cfg():
            return False
        return (
            not require_memory_headroom
            or self._has_enough_available_memory_for_fsdp_cfg()
        )
