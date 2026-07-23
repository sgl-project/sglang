"""
ServerArgsAutoTuner tunes the ServerArgs based on the desired performance mode
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    LAYERWISE_OFFLOAD_DIT_GROUP,
    LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_VAE_GROUP,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs

logger = init_logger(__name__)

PERFORMANCE_MODES = ("manual", "auto", "speed", "memory")

DEFAULT_LAYERWISE_COMPONENT_ARG_NAMES = (
    (LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP, "text_encoder_cpu_offload"),
    (LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP, "image_encoder_cpu_offload"),
    (LAYERWISE_OFFLOAD_VAE_GROUP, "vae_cpu_offload"),
)

# task-type defaults for keep_resident_min_available_gb when a model does not pin
# one: image vae is tiny so any datacenter gpu keeps it resident, video vae is
# larger so it only stays resident on very-high-memory gpus
IMAGE_GEN_KEEP_RESIDENT_MIN_AVAILABLE_GB = 45.0
DEFAULT_KEEP_RESIDENT_MIN_AVAILABLE_GB = 120.0


class ServerArgsAutoTuner:
    """Auto-tunes the server-arg for the given performance-mode, based on practical deployment experience with different model architectures"""

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self._explicit_memory_policy = self._has_explicit_memory_policy()
        self._explicit_layerwise_replacement_policy = (
            self._has_explicit_layerwise_replacement_policy()
        )

    def _deployment_config(self) -> ModelDeploymentConfig:
        return self.server_args.pipeline_config.get_model_deployment_config()

    def _resolve_keep_resident_min_available_gb(
        self, deployment_config: ModelDeploymentConfig
    ) -> float | None:
        # explicit per-model > task-type default > global default
        explicit = deployment_config.keep_resident_min_available_gb
        if explicit is not None:
            return explicit
        if self.server_args.pipeline_config.task_type.is_image_gen():
            return IMAGE_GEN_KEEP_RESIDENT_MIN_AVAILABLE_GB
        return DEFAULT_KEEP_RESIDENT_MIN_AVAILABLE_GB

    def adjust_based_on_performance_mode(self) -> None:
        """Adjust the server args based on the performance mode"""
        args = self.server_args
        args.performance_mode = self._normalize_performance_mode()

        if current_platform.is_cpu():
            return

        if args.performance_mode == "speed":
            logger.info("Applying performance_mode=speed")
            if not args.enable_torch_compile and not args.is_arg_explicitly_set(
                "enable_torch_compile"
            ):
                # speed means fastest: compile by default. An explicit
                # --enable-torch-compile false still wins (e.g. models where
                # compile measures slower, like short-step Z-Image runs).
                args.enable_torch_compile = True
                logger.info(
                    "performance_mode=speed enables torch.compile "
                    "(pass --enable-torch-compile false to opt out)"
                )
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
                if (
                    args.layerwise_offload_components is None
                    and self._can_apply_default_layerwise_offload_policy()
                ):
                    args.layerwise_offload_components = (
                        self._default_layerwise_components_for_unset_placement() or None
                    )
                return
            args.use_fsdp_inference = False
            if self._can_apply_default_layerwise_offload_policy():
                # apply default layerwise offload to save VRAM during denoising stage
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
        deployment_config = self._deployment_config()
        disable_threshold_gb = self._resolve_keep_resident_min_available_gb(
            deployment_config
        )
        if (
            min_available_gb is not None
            and disable_threshold_gb is not None
            and min_available_gb >= disable_threshold_gb
        ):
            changed = []
            components = deployment_config.keep_resident_components
            if (
                args.layerwise_offload_components is not None
                and not args.is_arg_explicitly_set("layerwise_offload_components")
            ):
                layerwise_components = [
                    component_name
                    for component_name in args.layerwise_offload_components
                    if component_name not in components
                ]
                if layerwise_components != args.layerwise_offload_components:
                    args.layerwise_offload_components = layerwise_components or None
                    changed.append(
                        f"layerwise_offload_components={args.layerwise_offload_components}"
                    )
            if (
                args.dit_cpu_offload
                and "dit" in components
                and not args.is_arg_explicitly_set("dit_cpu_offload")
            ):
                args.dit_cpu_offload = False
                changed.append("dit_cpu_offload=False")
            if (
                args.text_encoder_cpu_offload
                and LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP in components
                and not args.is_arg_explicitly_set("text_encoder_cpu_offload")
            ):
                args.text_encoder_cpu_offload = False
                changed.append("text_encoder_cpu_offload=False")
            if (
                args.image_encoder_cpu_offload
                and LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP in components
                and not args.is_arg_explicitly_set("image_encoder_cpu_offload")
            ):
                args.image_encoder_cpu_offload = False
                changed.append("image_encoder_cpu_offload=False")
            if (
                args.vae_cpu_offload
                and LAYERWISE_OFFLOAD_VAE_GROUP in components
                and not args.is_arg_explicitly_set("vae_cpu_offload")
            ):
                args.vae_cpu_offload = False
                changed.append("vae_cpu_offload=False")
            if changed:
                logger.info(
                    "Disabling component offload for %s because minimum available memory on selected GPUs is %.2f GiB: %s",
                    args.pipeline_config.__class__.__name__,
                    min_available_gb,
                    ", ".join(changed),
                )

        self._maybe_keep_ltx23_resident_aux_components_resident()

    def _maybe_keep_ltx23_resident_aux_components_resident(self) -> None:
        args = self.server_args
        if not args._uses_ltx23_high_memory_resident_two_stage_mode():
            return

        changed: list[str] = []
        if (
            args.layerwise_offload_components is not None
            and not args.is_arg_explicitly_set("layerwise_offload_components")
        ):
            args.layerwise_offload_components = None
            changed.append("layerwise_offload_components=None")

        # high-memory resident mode keeps both DiTs on GPU; unset auxiliary
        # placement should stay resident instead of using default layerwise
        for arg_name in (
            "text_encoder_cpu_offload",
            "image_encoder_cpu_offload",
            "vae_cpu_offload",
        ):
            if getattr(args, arg_name) and not args.is_arg_explicitly_set(arg_name):
                setattr(args, arg_name, False)
                changed.append(f"{arg_name}=False")

        if changed:
            logger.info(
                "Keeping LTX2 high-memory two-stage auxiliary components resident: %s",
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

    def maybe_adjust_auto_default_layerwise_offload(self) -> None:
        """Enable verified layerwise defaults for unset component placement."""
        args = self.server_args
        if args.performance_mode != "auto":
            return
        if not self.could_override_server_args():
            return
        if (
            args.layerwise_offload_components is not None
            or args.dit_layerwise_offload is True
        ):
            return
        if not current_platform.is_cuda():
            return

        layerwise_components = self._default_layerwise_components_for_unset_placement()
        if not layerwise_components:
            return

        logger.info(
            "Auto memory policy for %s selected layerwise offload components: %s",
            args.pipeline_config.__class__.__name__,
            ", ".join(layerwise_components),
        )
        args.layerwise_offload_components = layerwise_components

    def maybe_replace_cpu_offloaded_components_with_layerwise(self) -> None:
        args = self.server_args
        if (
            not self.could_override_server_args()
            or self._explicit_layerwise_replacement_policy
            or current_platform.is_cpu()
            or not current_platform.is_cuda()
            or envs.SGLANG_CACHE_DIT_ENABLED
            or args.use_fsdp_inference
            or args.layerwise_offload_components is not None
        ):
            return

        layerwise_components: list[str] = []
        if args.dit_layerwise_offload:
            layerwise_components.append(LAYERWISE_OFFLOAD_DIT_GROUP)

        changed: list[str] = []
        if args.text_encoder_cpu_offload and not args.is_arg_explicitly_set(
            "text_encoder_cpu_offload"
        ):
            layerwise_components.append(LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP)
            changed.append(LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP)
        if args.image_encoder_cpu_offload and not args.is_arg_explicitly_set(
            "image_encoder_cpu_offload"
        ):
            layerwise_components.append(LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP)
            changed.append(LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP)
        if args.vae_cpu_offload and not args.is_arg_explicitly_set("vae_cpu_offload"):
            layerwise_components.append(LAYERWISE_OFFLOAD_VAE_GROUP)
            changed.append(LAYERWISE_OFFLOAD_VAE_GROUP)

        if not changed:
            return

        args.layerwise_offload_components = layerwise_components
        logger.info(
            "Automatically replacing CPU offload with layerwise offload for components: %s",
            ", ".join(changed),
        )

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
        if args.layerwise_offload_components is None:
            args.layerwise_offload_components = (
                self._default_layerwise_components_for_unset_placement() or None
            )
        if args.dit_cpu_offload is None:
            args.dit_cpu_offload = True
        if args.text_encoder_cpu_offload is None:
            args.text_encoder_cpu_offload = False
        if args.image_encoder_cpu_offload is None:
            args.image_encoder_cpu_offload = False

    def _can_apply_default_layerwise_offload_policy(self) -> bool:
        return current_platform.is_cuda()

    def _default_layerwise_components_for_unset_placement(self) -> list[str]:
        args = self.server_args
        if args.pipeline_config.task_type.is_action_gen():
            return []
        if (
            args.is_arg_explicitly_set("layerwise_offload_components")
            or args.dit_layerwise_offload is True
        ):
            # The legacy --dit-layerwise-offload flag is a DiT-only selector.
            # Do not merge implicit defaults into that explicit mode.
            return []

        # `*_cpu_offload` is the component placement knob. If a user explicitly
        # set it to either true or false, keep that component out of default
        # layerwise selection.
        components = [
            component_name
            for component_name, arg_name in DEFAULT_LAYERWISE_COMPONENT_ARG_NAMES
            if not args.is_arg_explicitly_set(arg_name)
        ]
        components = self._filter_high_memory_resident_components(components)
        if self._should_auto_enable_dit_layerwise_offload():
            components.insert(0, LAYERWISE_OFFLOAD_DIT_GROUP)
            self._set_default_wan_dit_offload_prefetch_size()
        return components

    def _filter_high_memory_resident_components(
        self, components: list[str]
    ) -> list[str]:
        args = self.server_args
        if args.performance_mode != "auto" or current_platform.is_cpu():
            return components

        deployment_config = self._deployment_config()
        threshold_gb = self._resolve_keep_resident_min_available_gb(deployment_config)
        if threshold_gb is None:
            return components

        min_available_gb = self._get_min_available_device_memory_gb()
        if min_available_gb is None or min_available_gb < threshold_gb:
            return components

        resident_components = set(deployment_config.keep_resident_components)
        filtered_components = [
            component
            for component in components
            if component not in resident_components
        ]
        skipped_components = [
            component for component in components if component in resident_components
        ]
        if skipped_components:
            logger.info(
                "Keeping default layerwise components resident for %s because minimum available memory on selected GPUs is %.2f GiB: %s",
                args.pipeline_config.__class__.__name__,
                min_available_gb,
                ", ".join(skipped_components),
            )
        return filtered_components

    def _should_auto_enable_dit_layerwise_offload(self) -> bool:
        args = self.server_args

        # only for wan for now
        if not self._is_wan_pipeline_config():
            return False
        if not self._deployment_config().auto_dit_layerwise_offload:
            return False

        if (
            args.pipeline_config.dmd_denoising_steps is not None
            or not current_platform.enable_dit_layerwise_offload_for_wan_by_default()
            or envs.SGLANG_CACHE_DIT_ENABLED
            or args.use_fsdp_inference
            or args.is_arg_explicitly_set("dit_cpu_offload")
        ):
            return False

        # memory mode is memory-first: keep the broad Wan DiT layerwise policy
        # unless a guard above says it conflicts with another placement path
        if args.performance_mode == "memory":
            return True

        # auto mode is performance-first: profiling only showed clear wins for
        # Wan2.2 A14B, where coarse DiT CPU offload creates large step spikes
        return (
            args.performance_mode == "auto" and self._is_wan2_2_a14b_pipeline_config()
        )

    def _is_wan2_2_a14b_pipeline_config(self) -> bool:
        config_name = self.server_args.pipeline_config.__class__.__name__
        return config_name.startswith("Wan2_2_") and "A14B" in config_name

    def _set_default_wan_dit_offload_prefetch_size(self) -> None:
        args = self.server_args
        if (
            args.performance_mode == "auto"
            and self._is_wan2_2_a14b_pipeline_config()
            and not args.is_arg_explicitly_set("dit_offload_prefetch_size")
        ):
            # p2 was the fastest stable default in the Wan2.2 A14B sweep
            args.dit_offload_prefetch_size = 2

    def _is_wan_pipeline_config(self) -> bool:
        return any(
            cls.__module__.endswith(".wan")
            for cls in self.server_args.pipeline_config.__class__.mro()
        )

    def _auto_uses_dit_offload(self) -> bool:
        args = self.server_args
        return bool(
            args.dit_cpu_offload
            or args.dit_layerwise_offload
            or args.is_dit_layerwise_offload_selected
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
        return any(
            args.is_arg_explicitly_set(arg_name)
            for arg_name in (
                "use_fsdp_inference",
                "dit_cpu_offload",
                "dit_layerwise_offload",
                "layerwise_offload_components",
            )
        )

    def _has_explicit_layerwise_replacement_policy(self) -> bool:
        args = self.server_args
        return any(
            args.is_arg_explicitly_set(arg_name)
            for arg_name in (
                "dit_layerwise_offload",
                "layerwise_offload_components",
            )
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
        deployment_config = self._deployment_config()
        if (
            deployment_config.auto_enable_cfg_parallel
            and args.enable_cfg_parallel is None
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
