"""
ServerArgsAutoTuner tunes the ServerArgs based on the desired performance mode
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.configs.quantization.nunchaku import (
    NunchakuSVDQuantArgs,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    has_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    LAYERWISE_OFFLOAD_ALL_COMPONENTS,
    LAYERWISE_OFFLOAD_DIT_GROUP,
    LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_VAE_GROUP,
    is_dit_component_name,
    normalize_layerwise_offload_components,
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


def resolve_keep_resident_min_available_gb(
    server_args: ServerArgs,
    deployment_config: ModelDeploymentConfig | None = None,
) -> float:
    """Resolve the established device-memory admission threshold for residency."""
    deployment_config = (
        deployment_config or server_args.pipeline_config.get_model_deployment_config()
    )
    explicit = deployment_config.keep_resident_min_available_gb
    if explicit is not None:
        return explicit
    if server_args.pipeline_config.task_type.is_image_gen():
        return IMAGE_GEN_KEEP_RESIDENT_MIN_AVAILABLE_GB
    return DEFAULT_KEEP_RESIDENT_MIN_AVAILABLE_GB


def auto_residency_static_skip_reason(server_args: ServerArgs) -> str | None:
    """Return why args cannot use pre-warmup static residency planning."""
    if envs.SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY:
        return "disabled via SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY"
    if server_args.performance_mode != "auto":
        return f"performance_mode={server_args.performance_mode}"
    if server_args.disagg_role != RoleType.MONOLITHIC:
        return "disaggregated role"
    task_type = server_args.pipeline_config.task_type
    if not (task_type.is_visual_gen() or task_type.is_mesh_gen()):
        return "no synthetic server warmup to calibrate from"
    if not server_args.pipeline_config.supports_auto_residency:
        return "pipeline does not support post-warmup residency changes"
    if has_realtime_model_adapter(server_args):
        return "realtime serving has no representative synthetic warmup"
    if server_args.backend == "diffusers":
        return "diffusers backend"
    if server_args.enable_breakable_cuda_graph:
        return "breakable CUDA graph captures during warmup"
    if server_args.enable_torch_compile:
        # Compile warmup temporarily evicts resident auxiliaries and may
        # layerwise-offload the DiT, so its peak is not a serving peak.
        return "torch.compile warmup uses a stripped memory layout"
    if envs.SGLANG_CACHE_DIT_ENABLED:
        return "cache-dit enabled"
    if server_args.batching_max_size > 1:
        return "dynamic batching enabled"
    if not current_platform.is_cuda():
        return "requires CUDA"
    return None


def auto_residency_args_skip_reason(server_args: ServerArgs) -> str | None:
    """Return why args cannot use warmup-calibrated residency refinement."""
    reason = auto_residency_static_skip_reason(server_args)
    if reason is not None:
        return reason
    if (
        server_args.pipeline_class_name == "LTX2TwoStagePipeline"
        and server_args.ltx2_two_stage_device_mode is None
    ):
        return "legacy LTX-2 two-stage placement"
    if server_args.ltx2_two_stage_device_mode == "original":
        return "LTX-2 original two-stage placement"
    if server_args.warmup_mode != "server":
        return "no synthetic server warmup to calibrate from"
    return None


def fixed_loading_residency_components(
    server_args: ServerArgs, component_names: Iterable[str]
) -> set[str]:
    """Components whose selected residency must remain unchanged.

    Explicit/online quantization and direct GPU loading constrain their target
    DiTs. LTX-2 two-stage ``original`` mode also relies on its offloaded DiT
    while swapping the stage LoRA. Keep those components outside auto
    residency while still calibrating unrelated encoders and VAEs.
    """
    fixed = set(server_args.component_quantizations)
    nunchaku = server_args.nunchaku_config
    fixed_dit_loading = (
        server_args.quantization is not None
        or server_args.direct_gpu_weight_loading
        or server_args.ltx2_two_stage_device_mode == "original"
        or (
            nunchaku is not None
            and (
                not isinstance(nunchaku, NunchakuSVDQuantArgs)
                or nunchaku.enable_svdquant
            )
        )
    )
    if fixed_dit_loading:
        fixed.update(name for name in component_names if is_dit_component_name(name))
    return fixed


class ServerArgsAutoTuner:
    """Auto-tunes the server-arg for the given performance-mode, based on practical deployment experience with different model architectures"""

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self._explicit_dit_residency = self._has_explicit_dit_residency()

    def _deployment_config(self) -> ModelDeploymentConfig:
        return self.server_args.pipeline_config.get_model_deployment_config()

    def _uses_auto_residency_planner(self) -> bool:
        """Whether the static planner replaces coarse deployment defaults."""
        return auto_residency_static_skip_reason(self.server_args) is None

    def _resolve_keep_resident_min_available_gb(
        self, deployment_config: ModelDeploymentConfig
    ) -> float:
        return resolve_keep_resident_min_available_gb(
            self.server_args, deployment_config
        )

    def adjust_based_on_performance_mode(self) -> None:
        """Adjust the server args based on the performance mode"""
        args = self.server_args
        args.performance_mode = self._normalize_performance_mode()

        if current_platform.is_cpu():
            return

        if args.performance_mode == "speed":
            logger.info("Applying performance_mode=speed")
            if (
                self._deployment_config().speed_mode_enable_torch_compile_by_default
                and not args.enable_torch_compile
                and not args.is_arg_explicitly_set("enable_torch_compile")
            ):
                # only models with a validated compile win opt in by default
                args.enable_torch_compile = True
                logger.info(
                    "performance_mode=speed enables torch.compile "
                    "(pass --enable-torch-compile false to opt out)"
                )
            elif not args.enable_torch_compile and not args.is_arg_explicitly_set(
                "enable_torch_compile"
            ):
                logger.info(
                    "performance_mode=speed keeps torch.compile disabled for "
                    "this model (pass --enable-torch-compile true to opt in)"
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
        if args.performance_mode != "auto" or current_platform.is_cpu():
            return
        if self._uses_auto_residency_planner():
            # Keep the load-safe placement until the post-load static planner
            # replaces model/card thresholds with component sizes and the
            # default workload. A server warmup may refine that estimate.
            return

        # Explicit placement is component-scoped; unmatched components still
        # receive automatic defaults.

        explicit_layerwise_components = (
            normalize_layerwise_offload_components(args.layerwise_offload_components)
            if args.is_arg_explicitly_set("layerwise_offload_components")
            else None
        )
        explicit_dit_layerwise = bool(
            args.is_arg_explicitly_set("dit_layerwise_offload")
            and args.dit_layerwise_offload
        ) or bool(
            explicit_layerwise_components
            and (
                LAYERWISE_OFFLOAD_DIT_GROUP in explicit_layerwise_components
                or LAYERWISE_OFFLOAD_ALL_COMPONENTS in explicit_layerwise_components
            )
        )

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
            components = set(deployment_config.keep_resident_components)
            if args.pipeline_config.task_type.is_image_gen():
                components.add(LAYERWISE_OFFLOAD_DIT_GROUP)
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
                and args.explicit_residency_mode("transformer") is None
                and not explicit_dit_layerwise
            ):
                args.dit_cpu_offload = False
                changed.append("dit_cpu_offload=False")
            if (
                args.text_encoder_cpu_offload
                and LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP in components
                and args.explicit_residency_mode("text_encoder") is None
            ):
                args.text_encoder_cpu_offload = False
                changed.append("text_encoder_cpu_offload=False")
            if (
                args.image_encoder_cpu_offload
                and LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP in components
                and args.explicit_residency_mode("image_encoder") is None
            ):
                args.image_encoder_cpu_offload = False
                changed.append("image_encoder_cpu_offload=False")
            if (
                args.vae_cpu_offload
                and LAYERWISE_OFFLOAD_VAE_GROUP in components
                and args.explicit_residency_mode("vae") is None
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
        if (
            args.text_encoder_cpu_offload
            and args.explicit_residency_mode("text_encoder") is None
        ):
            args.text_encoder_cpu_offload = False
            changed.append("text_encoder_cpu_offload=False")
        if (
            args.image_encoder_cpu_offload
            and args.explicit_residency_mode("image_encoder") is None
        ):
            args.image_encoder_cpu_offload = False
            changed.append("image_encoder_cpu_offload=False")
        if args.vae_cpu_offload and args.explicit_residency_mode("vae") is None:
            args.vae_cpu_offload = False
            changed.append("vae_cpu_offload=False")

        if changed:
            logger.info(
                "Keeping LTX2 high-memory two-stage auxiliary components resident: %s",
                ", ".join(changed),
            )

    def maybe_adjust_auto_fsdp_with_offload_enabled(self) -> None:
        args = self.server_args
        if (
            args.use_fsdp_inference
            and args.is_arg_explicitly_set("use_fsdp_inference")
            and args.explicit_residency_mode("transformer") is None
        ):
            # An explicit FSDP request must manage the DiT unless the user also
            # supplied a component-scoped placement. Auto's load-safe default
            # otherwise leaves the transformer offloaded and silently turns
            # --use-fsdp-inference into a no-op.
            args.dit_cpu_offload = False

        if (
            args.performance_mode == "auto"
            and not self._uses_auto_residency_planner()
            and args.num_gpus >= 2
            and not self._explicit_dit_residency
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

        min_available_gb = self._get_min_available_device_memory_gb()
        uses_planner = self._uses_auto_residency_planner()
        model_name = args.pipeline_config.__class__.__name__
        available = (
            f"{min_available_gb:.1f} GiB" if min_available_gb is not None else "unknown"
        )
        if uses_planner:
            logger.info(
                "Auto memory planner for %s: layerwise candidates=%s, "
                "free VRAM=%s. Final placement follows component sizing and "
                "warmup calibration; explicit placement flags always win.",
                model_name,
                ", ".join(layerwise_components),
                available,
            )
        else:
            logger.info(
                "Auto memory policy for %s: free VRAM=%s selects layerwise "
                "offload for %s. Explicit placement flags always win.",
                model_name,
                available,
                ", ".join(layerwise_components),
            )
        args.layerwise_offload_components = layerwise_components
        self._warn_if_resident_dit_contradicts_its_own_threshold(layerwise_components)

    def _warn_if_resident_dit_contradicts_its_own_threshold(
        self, layerwise_components: list[str]
    ) -> None:
        """Name the missing declaration when the DiT stays resident on a small card.

        A model that sets `keep_resident_min_available_gb` is saying to keep
        components resident only above that much device memory. If the auto
        policy nonetheless leaves the DiT resident on a card far below it, the
        two statements disagree, and the cause is almost always that the model
        never declared `dit_layerwise_offload_modes` -- its default is an empty
        tuple, so the DiT is never selected in any mode. That surfaces as a
        failure during load, and it is worth naming the missing knob rather than
        leaving the allocator to report it.

        A warning rather than an error: a small model's DiT can legitimately fit
        below the threshold, which is about plenty and not about fit.
        """
        args = self.server_args
        deployment_config = self._deployment_config()
        threshold_gb = deployment_config.keep_resident_min_available_gb
        if threshold_gb is None:
            return
        if LAYERWISE_OFFLOAD_DIT_GROUP in (
            normalize_layerwise_offload_components(layerwise_components) or ()
        ):
            return
        if args.performance_mode in deployment_config.dit_layerwise_offload_modes:
            return
        available_gb = self._get_min_available_device_memory_gb()
        if available_gb is None or available_gb >= threshold_gb:
            return
        logger.warning(
            "%s keeps its DiT resident with %.1f GiB of device memory available, "
            "below the %.1f GiB this model declares as its threshold for keeping "
            "components resident. The DiT is not in the automatic layerwise "
            "selection because the model does not list %r in "
            "dit_layerwise_offload_modes. If the DiT does not fit, pass "
            "--layerwise-offload-components dit,... explicitly, or add the mode "
            "to the model's deployment config.",
            args.pipeline_config.__class__.__name__,
            available_gb,
            threshold_gb,
            args.performance_mode,
        )

    def maybe_replace_cpu_offloaded_components_with_layerwise(self) -> None:
        args = self.server_args
        if (
            not self.could_override_server_args()
            or current_platform.is_cpu()
            or not current_platform.is_cuda()
            or envs.SGLANG_CACHE_DIT_ENABLED
            or args.use_fsdp_inference
            or args.layerwise_offload_components is not None
            or args.dit_layerwise_offload is True
        ):
            return

        layerwise_components: list[str] = []
        if args.dit_layerwise_offload:
            layerwise_components.append(LAYERWISE_OFFLOAD_DIT_GROUP)

        changed: list[str] = []
        if (
            args.text_encoder_cpu_offload
            and args.explicit_residency_mode("text_encoder") is None
        ):
            layerwise_components.append(LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP)
            changed.append(LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP)
        if (
            args.image_encoder_cpu_offload
            and args.explicit_residency_mode("image_encoder") is None
        ):
            layerwise_components.append(LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP)
            changed.append(LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP)
        if args.vae_cpu_offload and args.explicit_residency_mode("vae") is None:
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
        if (
            args.pin_cpu_memory
            and not args.is_arg_explicitly_set("pin_cpu_memory")
            and current_platform.device_shares_host_memory()
        ):
            # The device reads host pages directly on a shared pool, so a
            # pinned copy of a mapped weight is the same bytes held twice.
            args.pin_cpu_memory = False
            logger.info(
                "Host and device share one memory pool: pinned host weight "
                "copies are disabled (pass --pin-cpu-memory true to override)."
            )
        if (
            current_platform.device_shares_host_memory()
            and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
        ):
            # Every byte the caching allocator keeps reserved is a byte the
            # page cache -- the home of every mapped weight here -- cannot
            # hold. Measured on a GB10: ~30 GiB of reserved-but-idle segments
            # forced the encoder and the DiT to take turns being re-read from
            # disk. Expandable segments let the reserve follow the live peak.
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            logger.info(
                "Host and device share one memory pool: PYTORCH_CUDA_ALLOC_CONF="
                "expandable_segments:True so the allocator's reserve does not "
                "crowd out the page cache."
            )
        if current_platform.device_shares_host_memory():
            try:
                import psutil

                swap_total = psutil.swap_memory().total
            except Exception:
                swap_total = 0
            try:
                # A cgroup with swap disabled (memory.swap.max = 0) protects
                # this process whatever the host has mounted.
                with open("/sys/fs/cgroup/memory.swap.max") as handle:
                    if handle.read().strip() == "0":
                        swap_total = 0
            except OSError:
                pass
            try:
                with open("/sys/fs/cgroup/memory.max") as handle:
                    uncapped = handle.read().strip() == "max"
            except OSError:
                uncapped = True
            if uncapped:
                # The driver takes device memory from free pages and does not
                # wait for the kernel to reclaim page cache: with the cache
                # full and MemFree near zero, device growth fails outright
                # (NVRM out-of-memory on a GB10, three runs). A cgroup limit a
                # little under physical memory makes the kernel reclaim this
                # process's cache ahead of its own allocations.
                logger.warning(
                    "Host and device share one memory pool and this process has "
                    "no cgroup memory limit: device allocations may fail while "
                    "the page cache holds the free memory. Run with a limit a few "
                    "GiB under physical memory (for example docker --memory)."
                )
            if swap_total > 0:
                # Under page-cache pressure the kernel prefers swapping idle
                # anonymous memory -- here the DiT's fused weight copies --
                # over dropping cache, and every denoise step then swaps them
                # back in. Measured on a GB10: 128 s first steps and a 54 s
                # text encoder with 143 GiB of swap enabled.
                logger.warning(
                    "Host and device share one memory pool and swap is enabled "
                    "(%.0f GiB): the kernel may swap out weight copies under "
                    "page-cache pressure. Run with swap off for this process "
                    "(container --memory-swap equal to --memory, or "
                    "vm.swappiness=0).",
                    swap_total / 1024**3,
                )
            if args.dit_cpu_offload or args.text_encoder_cpu_offload:
                # Whole-component offload holds a component twice while it
                # moves: the device copy plus a host copy the size of the
                # component. On a shared pool both come out of the same
                # memory. Measured on a GB10: a 57 GiB DiT moving back to
                # the host at the end of a denoise stage exhausted the pool.
                logger.warning(
                    "Host and device share one memory pool and whole-component "
                    "CPU offload is enabled (dit_cpu_offload=%s, "
                    "text_encoder_cpu_offload=%s): moving a component holds it "
                    "twice while it moves. Prefer layerwise offload, where "
                    "residency is armed layer by layer from the checkpoint "
                    "mapping.",
                    bool(args.dit_cpu_offload),
                    bool(args.text_encoder_cpu_offload),
                )

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
            if args.explicit_residency_mode(component_name) is None
        ]
        components = self._filter_high_memory_resident_components(components)
        if self._should_auto_enable_dit_layerwise_offload():
            components.insert(0, LAYERWISE_OFFLOAD_DIT_GROUP)
            self._set_default_dit_offload_prefetch_size()
        return components

    def _filter_high_memory_resident_components(
        self, components: list[str]
    ) -> list[str]:
        args = self.server_args
        if args.performance_mode != "auto" or current_platform.is_cpu():
            return components
        if self._uses_auto_residency_planner():
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
        deployment_config = self._deployment_config()
        if args.performance_mode not in deployment_config.dit_layerwise_offload_modes:
            return False

        if (
            args.pipeline_config.dmd_denoising_steps is not None
            or not current_platform.enable_dit_layerwise_offload_by_default()
            or envs.SGLANG_CACHE_DIT_ENABLED
            or args.use_fsdp_inference
            or args.explicit_residency_mode("transformer") is not None
        ):
            return False

        return True

    def _set_default_dit_offload_prefetch_size(self) -> None:
        args = self.server_args
        prefetch_size = self._deployment_config().auto_dit_offload_prefetch_size
        if (
            args.performance_mode == "auto"
            and prefetch_size is not None
            and not args.is_arg_explicitly_set("dit_offload_prefetch_size")
        ):
            args.dit_offload_prefetch_size = prefetch_size

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

    def _has_explicit_dit_residency(self) -> bool:
        args = self.server_args
        return bool(
            args.is_arg_explicitly_set("use_fsdp_inference")
            or args.explicit_residency_mode("transformer") is not None
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
