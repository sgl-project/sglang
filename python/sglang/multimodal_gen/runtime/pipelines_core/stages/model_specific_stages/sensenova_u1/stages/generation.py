# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.sensenova_u1 import (
    DEFAULT_CFG_INTERVAL,
    DEFAULT_CFG_NORM,
    DEFAULT_ENABLE_TIMESTEP_SHIFT,
    DEFAULT_T_EPS,
    DEFAULT_THINK_MODE,
    DEFAULT_TIMESTEP_SHIFT,
    MIN_INPUT_MAX_PIXELS,
    RESOLUTION_ALIGNMENT,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
    SenseNovaGuidanceProfile,
    _flatten_rgba_to_rgb,
    derive_guidance_profile,
    has_sensenova_u1_explicit_size,
    resolve_sensenova_u1_edit_auto_size,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_sp_group, get_tp_group
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.utils import (
    smart_resize,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.vision import load_image

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
        CacheDitConfig,
    )

logger = init_logger(__name__)

DEFAULT_INPUT_MAX_PIXELS = 2048 * 2048


def _get_cache_dit_attention_type(transformer: torch.nn.Module) -> str:
    """Validate once per mount: the unified wrapper passes one mask to all blocks."""
    attention_types = {
        layer.attention_type
        for layer in transformer.layers[: transformer.config.num_hidden_layers]
    }
    if len(attention_types) != 1:
        raise ValueError(
            "SenseNova-U1 Cache-DiT requires all decoder layers to "
            "use the same attention type."
        )
    attention_type = next(iter(attention_types))
    if attention_type is None:
        raise ValueError("SenseNova-U1 Cache-DiT requires a decoder attention type.")
    return attention_type


def _denorm_sensenova_output(x: torch.Tensor) -> torch.Tensor:
    """Convert SenseNova's normalized image tensor from [-1, 1] to [0, 1]."""
    return ((x.float() + 1.0) * 0.5).clamp(0, 1)


def _auto_input_max_pixels(num_images: int) -> int:
    if num_images <= 0:
        raise ValueError(
            "SenseNova-U1 image editing requires at least one input image."
        )
    full_resolution_image_budget = 2
    if num_images <= full_resolution_image_budget:
        return DEFAULT_INPUT_MAX_PIXELS
    total_budget = full_resolution_image_budget * DEFAULT_INPUT_MAX_PIXELS
    return max(MIN_INPUT_MAX_PIXELS, total_budget // num_images)


def _resize_input_to_budget(
    image: Image.Image,
    *,
    do_resize: bool,
    input_max_pixels: int | None,
) -> Image.Image:
    image = _flatten_rgba_to_rgb(image)
    if not do_resize or input_max_pixels is None:
        return image
    resized_height, resized_width = smart_resize(
        height=image.height,
        width=image.width,
        factor=RESOLUTION_ALIGNMENT,
        min_pixels=input_max_pixels,
        max_pixels=input_max_pixels,
    )
    if (resized_width, resized_height) == image.size:
        return image
    return image.resize((resized_width, resized_height), Image.LANCZOS)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"do_resize must be a boolean value, got {value!r}.")


def _coerce_cfg_interval(value: Any) -> tuple[float, float]:
    start, end = value
    return float(start), float(end)


def _image_input_to_list(image_input: Any) -> list[Image.Image]:
    if image_input is None:
        return []
    if isinstance(image_input, list):
        items = image_input
    else:
        items = [image_input]

    images = []
    for item in items:
        if isinstance(item, Image.Image):
            images.append(item)
        else:
            images.append(load_image(str(item), convert_method=_flatten_rgba_to_rgb))
    return images


def _load_edit_images(batch: Req) -> list[Image.Image]:
    images = _image_input_to_list(getattr(batch, "image_path", None))
    if not images:
        images = _image_input_to_list(getattr(batch, "condition_image", None))
    return images


def _prepare_edit_images(
    images: list[Image.Image], options: SenseNovaU1GenerationOptions
) -> list[Image.Image]:
    input_max_pixels = options.input_max_pixels
    if input_max_pixels is None:
        input_max_pixels = _auto_input_max_pixels(len(images))
    return [
        _resize_input_to_budget(
            image,
            do_resize=options.do_resize,
            input_max_pixels=input_max_pixels,
        )
        for image in images
    ]


def _has_explicit_output_size(batch: Req) -> bool:
    extra = getattr(batch, "extra", {}) or {}
    return has_sensenova_u1_explicit_size(extra.get("explicit_fields", ()))


def _resolve_edit_output_size(
    batch: Req, edit_images: list[Image.Image]
) -> tuple[int, int]:
    """Preserve the first input image's aspect ratio for SenseNova image edits."""
    if not edit_images or _has_explicit_output_size(batch):
        return int(batch.width), int(batch.height)

    return resolve_sensenova_u1_edit_auto_size(
        edit_images[0].width, edit_images[0].height
    )


@dataclass(frozen=True)
class SenseNovaU1GenerationOptions:
    cfg_norm: str = DEFAULT_CFG_NORM
    timestep_shift: float = DEFAULT_TIMESTEP_SHIFT
    enable_timestep_shift: bool = DEFAULT_ENABLE_TIMESTEP_SHIFT
    cfg_interval: tuple[float, float] = DEFAULT_CFG_INTERVAL
    t_eps: float = DEFAULT_T_EPS
    think_mode: bool = DEFAULT_THINK_MODE
    img_cfg_scale: float = 1.0
    input_max_pixels: int | None = None
    do_resize: bool = True

    @classmethod
    def from_batch(cls, batch: Req) -> SenseNovaU1GenerationOptions:
        extra = batch.extra.get(SENSENOVA_U1_REQUEST_EXTRA_KEY, {})
        return cls(
            cfg_norm=extra.get("cfg_norm", DEFAULT_CFG_NORM),
            timestep_shift=float(extra.get("timestep_shift", DEFAULT_TIMESTEP_SHIFT)),
            enable_timestep_shift=bool(
                extra.get("enable_timestep_shift", DEFAULT_ENABLE_TIMESTEP_SHIFT)
            ),
            cfg_interval=_coerce_cfg_interval(
                extra.get("cfg_interval", DEFAULT_CFG_INTERVAL)
            ),
            t_eps=float(extra.get("t_eps", DEFAULT_T_EPS)),
            think_mode=bool(extra.get("think_mode", DEFAULT_THINK_MODE)),
            img_cfg_scale=float(extra.get("img_cfg_scale", 1.0)),
            input_max_pixels=(
                None
                if extra.get("input_max_pixels") is None
                else int(extra.get("input_max_pixels"))
            ),
            do_resize=_coerce_bool(extra.get("do_resize", True)),
        )


class SenseNovaU1GenerationStage(PipelineStage):
    def __init__(self, model: torch.nn.Module, tokenizer: Any):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._cache_dit_enabled = False
        self._cache_dit_active_key: tuple | None = None
        self._cache_dit_active_config = None
        self._cache_dit_cleanup_required = False

    def _cache_dit_requested(self, batch: Req) -> bool:
        """Per-request Cache-DiT switch; the server default applies when unset."""
        enabled = batch.sampling_params.enable_cache_dit
        if enabled is None:
            return envs.SGLANG_CACHE_DIT_ENABLED
        return enabled

    @property
    def _cache_dit_transformer(self) -> torch.nn.Module:
        return self.model.language_model.model

    def _unmount_cache_dit(self, *, force: bool = False) -> None:
        if not force and not (
            self._cache_dit_enabled or self._cache_dit_cleanup_required
        ):
            return
        # Import lazily: SenseNova remains usable without the optional
        # cache-dit dependency when no request enables it.
        from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
            disable_cache_on_transformer,
        )

        transformer = self._cache_dit_transformer
        cleanup_succeeded = False
        try:
            # A failed enable_cache() may already have installed the forward
            # wrapper or cache context without letting the stage mark itself
            # enabled.  ``force`` therefore deliberately calls disable even
            # when _cache_dit_enabled is still False.
            disable_cache_on_transformer(transformer)
            cleanup_succeeded = True
        finally:
            if hasattr(transformer, "_sensenova_cache_dit_native_layers"):
                del transformer._sensenova_cache_dit_native_layers
            if hasattr(transformer, "_sensenova_cache_dit_attention_type"):
                del transformer._sensenova_cache_dit_attention_type
            self._cache_dit_enabled = False
            self._cache_dit_active_key = None
            self._cache_dit_active_config = None
            # If rollback itself failed, do not let a later ordinary request
            # take the early-return path and execute a potentially wrapped
            # transformer without its native-layers escape hatch.
            self._cache_dit_cleanup_required = not cleanup_succeeded

    def _cache_dit_blocked_reason(
        self,
        server_args: ServerArgs,
        *,
        guidance_profile: SenseNovaGuidanceProfile,
        cfg_interval: tuple[float, float],
    ) -> str | None:
        """Why a request that asked for Cache-DiT cannot mount it; None when it can."""
        if server_args.enable_breakable_cuda_graph:
            return "breakable CUDA graphs are enabled"
        if guidance_profile.branch_count > 2:
            return "three conditioning branches are not supported"
        # cache-dit's separate-CFG context expects a stable pair of forwards per
        # diffusion step. T2I and IT2I use different interval predicates
        # (inclusive + cfg_scale > 1 versus strict bounds + a lo == 0 escape),
        # so the only shared, unambiguous multi-branch schedule is (0, 1).
        # Reject other intervals instead of letting a varying call rhythm
        # advance the cache context on the wrong branch.
        if guidance_profile.branch_count > 1 and tuple(cfg_interval) != (0.0, 1.0):
            return "timestep-gated CFG is not supported; cfg_interval must be (0, 1)"
        return None

    def _maybe_enable_cache_dit(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        cfg_interval: tuple[float, float],
        guidance_profile: SenseNovaGuidanceProfile,
    ) -> None:
        """Mount or refresh the pure-image Cache-DiT path for one request."""
        if self._cache_dit_cleanup_required:
            self._unmount_cache_dit(force=True)

        requested = self._cache_dit_requested(batch)
        if requested:
            blocked_reason = self._cache_dit_blocked_reason(
                server_args,
                guidance_profile=guidance_profile,
                cfg_interval=cfg_interval,
            )
            if blocked_reason is not None:
                logger.warning_once(
                    "Cache-DiT was requested but is disabled because %s.",
                    blocked_reason,
                )
                requested = False

        # Disabled requests ignore cache parameters regardless of the previous
        # request. Unmount lazily to keep cache-dit an optional dependency.
        if not requested:
            self._unmount_cache_dit()
            return

        self._mount_or_refresh_cache_dit(
            batch, server_args=server_args, guidance_profile=guidance_profile
        )

    def _mount_or_refresh_cache_dit(
        self,
        batch: Req,
        *,
        server_args: ServerArgs,
        guidance_profile: SenseNovaGuidanceProfile,
    ) -> None:
        """Reuse the mounted Cache-DiT context, or mount one for this request."""
        from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
            CACHE_DIT_DBCACHE_KEYS,
            CacheDitConfig,
            cache_dit_env_defaults,
            cache_dit_overrides_key,
            refresh_context_on_transformer,
            resolve_cache_dit_request_overrides,
        )

        overrides = resolve_cache_dit_request_overrides(
            batch.sampling_params.cache_dit_params
        )
        unsupported = set(overrides) - CACHE_DIT_DBCACHE_KEYS
        if unsupported:
            raise ValueError(
                "SenseNova-U1 Cache-DiT currently supports DBCache knobs only; "
                f"unsupported keys: {sorted(unsupported)}."
            )
        # Compare effective settings so omitted and explicit defaults share a mount.
        effective_config = cache_dit_env_defaults()
        effective_config.update(overrides)
        has_separate_cfg = guidance_profile.has_separate_cfg
        desired_key = (cache_dit_overrides_key(effective_config), has_separate_cfg)
        if self._cache_dit_enabled and desired_key != self._cache_dit_active_key:
            self._unmount_cache_dit()

        transformer = self._cache_dit_transformer
        steps = int(batch.num_inference_steps)
        if self._cache_dit_enabled:
            if self._cache_dit_active_config is None:
                raise RuntimeError(
                    "SenseNova-U1 Cache-DiT is enabled without an active config."
                )
            refresh_context_on_transformer(
                transformer,
                steps,
                config=self._cache_dit_active_config,
            )
            return

        config = CacheDitConfig(
            enabled=True,
            num_inference_steps=steps,
            **effective_config,
        )
        self._mount_cache_dit(
            transformer,
            config=config,
            has_separate_cfg=has_separate_cfg,
            server_args=server_args,
        )
        self._cache_dit_enabled = True
        self._cache_dit_active_key = desired_key
        self._cache_dit_active_config = config
        self._cache_dit_cleanup_required = False

    def _mount_cache_dit(
        self,
        transformer: torch.nn.Module,
        *,
        config: CacheDitConfig,
        has_separate_cfg: bool,
        server_args: ServerArgs,
    ) -> None:
        """Enable Cache-DiT on a transformer, rolling back a partial mount."""
        from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
            enable_cache_on_transformer,
        )

        attention_type = _get_cache_dit_attention_type(transformer)

        # Cache-DiT's forward wrapper replaces ``transformer.layers`` only
        # dynamically. Preserve the genuine ModuleList so both Qwen3 backbones
        # can use it for prefix/Think/text forwards during this mounted session.
        # Bypass nn.Module.__setattr__: registering the same ModuleList under
        # a second name would duplicate it in state_dict/module traversal.
        object.__setattr__(
            transformer, "_sensenova_cache_dit_native_layers", transformer.layers
        )
        transformer._sensenova_cache_dit_attention_type = attention_type
        try:
            sp_group = (
                get_sp_group().device_group
                if (server_args.sp_degree or 1) > 1
                else None
            )
            tp_group = (
                get_tp_group().device_group
                if (server_args.tp_size or 1) > 1
                else None
            )
            enable_cache_on_transformer(
                transformer,
                config,
                model_name="sensenova-qwen3-image",
                sp_group=sp_group,
                tp_group=tp_group,
                # A full-interval two-branch schedule issues a stable pair at
                # every step; timestep-gated and three-branch schedules were
                # rejected before mounting.
                has_separate_cfg=has_separate_cfg,
            )
        except Exception:
            # cache_dit.enable_cache() is multi-stage and may have already
            # installed contexts or a forward wrapper before raising.  Roll
            # back unconditionally; preserve the original mount exception if
            # cleanup also fails.
            try:
                self._unmount_cache_dit(force=True)
            except Exception:
                logger.exception(
                    "Failed to roll back a partial SenseNova-U1 Cache-DiT mount"
                )
            raise

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        options = SenseNovaU1GenerationOptions.from_batch(batch)

        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        prompt_count = len(prompts)
        if prompt_count == 0:
            raise ValueError(
                "SenseNova-U1 dynamic batch must contain at least one prompt"
            )
        if prompt_count > 1 and options.think_mode:
            raise ValueError(
                "SenseNova-U1 dynamic batching does not support think_mode"
            )

        num_outputs = max(1, int(batch.num_outputs_per_prompt or 1))
        if prompt_count > 1 and num_outputs > 1:
            raise ValueError(
                "SenseNova-U1 dynamic batching does not support multiple outputs per prompt"
            )
        batch_size = prompt_count if prompt_count > 1 else num_outputs

        generators = batch.generator
        if isinstance(generators, torch.Generator):
            generators = [generators]
        if generators is None or len(generators) != batch_size:
            raise ValueError(
                f"Expected {batch_size} generators, "
                f"got {0 if generators is None else len(generators)}."
            )

        raw_edit_images = _load_edit_images(batch)
        edit_images = (
            _prepare_edit_images(raw_edit_images, options) if raw_edit_images else []
        )
        image_size = (
            _resolve_edit_output_size(batch, raw_edit_images)
            if edit_images
            else (int(batch.width), int(batch.height))
        )
        if edit_images:
            original_size = (int(batch.width), int(batch.height))
            batch.width, batch.height = image_size
            if image_size != original_size:
                logger.info(
                    "Resolved SenseNova-U1 edit output size from %sx%s to %sx%s",
                    original_size[0],
                    original_size[1],
                    image_size[0],
                    image_size[1],
                )

        if edit_images:
            if options.cfg_norm == "cfg_zero_star":
                raise ValueError(
                    "cfg_zero_star is only supported for SenseNova-U1 text-to-image, "
                    "not image editing."
                )

        guidance_profile = derive_guidance_profile(
            is_edit=bool(edit_images),
            cfg_scale=float(batch.guidance_scale),
            img_cfg_scale=options.img_cfg_scale,
        )
        self._maybe_enable_cache_dit(
            batch,
            server_args,
            guidance_profile=guidance_profile,
            cfg_interval=options.cfg_interval,
        )

        common_kwargs = dict(
            image_size=image_size,
            cfg_scale=float(batch.guidance_scale),
            cfg_norm=options.cfg_norm,
            timestep_shift=options.timestep_shift,
            enable_timestep_shift=options.enable_timestep_shift,
            cfg_interval=options.cfg_interval,
            num_steps=int(batch.num_inference_steps),
            batch_size=batch_size,
            t_eps=options.t_eps,
            think_mode=options.think_mode,
            generators=generators,
        )
        with set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=batch
        ):
            if edit_images:
                out = self.model.it2i_generate(
                    self.tokenizer,
                    batch.prompt,
                    edit_images,
                    img_cfg_scale=options.img_cfg_scale,
                    **common_kwargs,
                )
            else:
                out = self.model.t2i_generate(
                    self.tokenizer,
                    batch.prompt,
                    **common_kwargs,
                )
        think_text = None
        if options.think_mode:
            images, think_text = out
        else:
            images = out

        images = _denorm_sensenova_output(images)
        samples = [sample.contiguous() for sample in images]
        usage = {"think_text": think_text} if think_text is not None else None
        return OutputBatch(
            output=samples,
            metrics=batch.metrics,
            usage=usage,
        )
