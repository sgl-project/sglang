# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.sensenova_u1 import (
    DEFAULT_CFG_INTERVAL,
    DEFAULT_CFG_NORM,
    DEFAULT_ENABLE_TIMESTEP_SHIFT,
    DEFAULT_T_EPS,
    DEFAULT_THINK_MODE,
    DEFAULT_TIMESTEP_SHIFT,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_SENSENOVA_DBCACHE_KEYS = frozenset(
    {
        "Fn_compute_blocks",
        "Bn_compute_blocks",
        "max_warmup_steps",
        "residual_diff_threshold",
        "max_continuous_cached_steps",
    }
)


def _denorm_sensenova_output(x: torch.Tensor) -> torch.Tensor:
    """Convert SenseNova's normalized image tensor from [-1, 1] to [0, 1]."""
    return ((x.float() + 1.0) * 0.5).clamp(0, 1)


@dataclass(frozen=True)
class SenseNovaU1GenerationOptions:
    cfg_norm: str = DEFAULT_CFG_NORM
    timestep_shift: float = DEFAULT_TIMESTEP_SHIFT
    enable_timestep_shift: bool = DEFAULT_ENABLE_TIMESTEP_SHIFT
    cfg_interval: tuple[float, float] = DEFAULT_CFG_INTERVAL
    t_eps: float = DEFAULT_T_EPS
    think_mode: bool = DEFAULT_THINK_MODE

    @classmethod
    def from_batch(cls, batch: Req) -> SenseNovaU1GenerationOptions:
        extra = getattr(batch, "extra", {}).get(SENSENOVA_U1_REQUEST_EXTRA_KEY, {})
        return cls(
            cfg_norm=extra.get("cfg_norm", DEFAULT_CFG_NORM),
            timestep_shift=float(extra.get("timestep_shift", DEFAULT_TIMESTEP_SHIFT)),
            enable_timestep_shift=bool(
                extra.get("enable_timestep_shift", DEFAULT_ENABLE_TIMESTEP_SHIFT)
            ),
            cfg_interval=tuple(extra.get("cfg_interval", DEFAULT_CFG_INTERVAL)),
            t_eps=float(extra.get("t_eps", DEFAULT_T_EPS)),
            think_mode=bool(extra.get("think_mode", DEFAULT_THINK_MODE)),
        )


class SenseNovaU1GenerationStage(PipelineStage):
    def __init__(self, model: torch.nn.Module, tokenizer: Any):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._cache_dit_enabled = False
        self._cache_dit_active_key: tuple | None = None

    def _cache_dit_requested(self, batch: Req) -> bool:
        sampling_params = getattr(batch, "sampling_params", None)
        enabled = getattr(sampling_params, "enable_cache_dit", None)
        return envs.SGLANG_CACHE_DIT_ENABLED if enabled is None else enabled

    @property
    def _cache_dit_transformer(self) -> torch.nn.Module:
        return self.model.language_model.model

    def _unmount_cache_dit(self) -> None:
        if not self._cache_dit_enabled:
            return
        # Import lazily: SenseNova remains usable without the optional
        # cache-dit dependency when no request enables it.
        from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
            disable_cache_on_transformer,
        )

        transformer = self._cache_dit_transformer
        disable_cache_on_transformer(transformer)
        if hasattr(transformer, "_sensenova_cache_dit_native_layers"):
            del transformer._sensenova_cache_dit_native_layers
        self._cache_dit_enabled = False
        self._cache_dit_active_key = None

    def _maybe_enable_cache_dit(self, batch: Req, server_args: ServerArgs) -> None:
        """Mount or refresh the pure-image Cache-DiT path for one request."""
        requested = self._cache_dit_requested(batch)
        if getattr(server_args, "enable_breakable_cuda_graph", False):
            if requested:
                logger.warning_once(
                    "Cache-DiT was requested but is disabled because breakable "
                    "CUDA graphs are enabled."
                )
            requested = False

        # cache-dit's separate-CFG context expects a stable pair of forwards
        # per diffusion step.  SenseNova can gate CFG by timestep, producing
        # a 1 -> 2 -> 1 call rhythm; do not let that rhythm advance a generic
        # cache context incorrectly.  Full-interval CFG has a stable pair and
        # is supported.  A future branch-aware adapter can lift this guard.
        options = SenseNovaU1GenerationOptions.from_batch(batch)
        has_separate_cfg = float(batch.guidance_scale) > 1.0
        has_partial_cfg = has_separate_cfg and tuple(options.cfg_interval) != (
            0.0,
            1.0,
        )
        if requested and has_partial_cfg:
            logger.warning_once(
                "SenseNova-U1 Cache-DiT is disabled for timestep-gated CFG; "
                "only cfg_interval=(0, 1) is currently safe."
            )
            requested = False

        # Keep cache-dit an optional dependency for ordinary SenseNova
        # requests.  Import it only when a prior request must be unmounted or
        # the current one actually enables it.
        if not requested and not self._cache_dit_enabled:
            return

        from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
            CacheDitConfig,
            cache_dit_overrides_key,
            enable_cache_on_transformer,
            refresh_context_on_transformer,
            resolve_cache_dit_request_overrides,
        )

        overrides = resolve_cache_dit_request_overrides(
            getattr(batch.sampling_params, "cache_dit_params", None)
        )
        unsupported = set(overrides) - _SENSENOVA_DBCACHE_KEYS
        if unsupported:
            raise ValueError(
                "SenseNova-U1 Cache-DiT currently supports DBCache knobs only; "
                f"unsupported keys: {sorted(unsupported)}."
            )
        desired_key = (
            (cache_dit_overrides_key(overrides), has_separate_cfg)
            if requested
            else None
        )
        if self._cache_dit_enabled and desired_key != self._cache_dit_active_key:
            self._unmount_cache_dit()

        transformer = self._cache_dit_transformer
        steps = int(batch.num_inference_steps)
        if self._cache_dit_enabled:
            refresh_context_on_transformer(transformer, steps)
            return
        if not requested:
            return

        # Cache-DiT's forward wrapper replaces ``transformer.layers`` only
        # dynamically. Preserve the genuine ModuleList so both Qwen3 backbones
        # can use it for prefix/Think/text forwards during this mounted session.
        # Bypass nn.Module.__setattr__: registering the same ModuleList under
        # a second name would duplicate it in state_dict/module traversal.
        object.__setattr__(
            transformer, "_sensenova_cache_dit_native_layers", transformer.layers
        )
        try:
            config = CacheDitConfig(
                enabled=True,
                Fn_compute_blocks=overrides.get(
                    "Fn_compute_blocks", envs.SGLANG_CACHE_DIT_FN
                ),
                Bn_compute_blocks=overrides.get(
                    "Bn_compute_blocks", envs.SGLANG_CACHE_DIT_BN
                ),
                max_warmup_steps=overrides.get(
                    "max_warmup_steps", envs.SGLANG_CACHE_DIT_WARMUP
                ),
                residual_diff_threshold=overrides.get(
                    "residual_diff_threshold", envs.SGLANG_CACHE_DIT_RDT
                ),
                max_continuous_cached_steps=overrides.get(
                    "max_continuous_cached_steps", envs.SGLANG_CACHE_DIT_MC
                ),
                num_inference_steps=steps,
            )
            enable_cache_on_transformer(
                transformer,
                config,
                model_name="sensenova-qwen3-image",
                # Full-interval CFG issues a stable cond/uncond pair at every
                # step; timestep-gated CFG was rejected above.
                has_separate_cfg=has_separate_cfg,
            )
        except Exception:
            del transformer._sensenova_cache_dit_native_layers
            raise
        self._cache_dit_enabled = True
        self._cache_dit_active_key = desired_key

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        self._maybe_enable_cache_dit(batch, server_args)
        options = SenseNovaU1GenerationOptions.from_batch(batch)
        if int(batch.num_outputs_per_prompt) != 1:
            raise ValueError(
                "SenseNova-U1 expects output expansion before generation; "
                f"got num_outputs_per_prompt={batch.num_outputs_per_prompt}."
            )
        seed = batch.seed[0] if isinstance(batch.seed, list) else int(batch.seed)

        out = self.model.t2i_generate(
            self.tokenizer,
            batch.prompt,
            image_size=(int(batch.width), int(batch.height)),
            cfg_scale=float(batch.guidance_scale),
            cfg_norm=options.cfg_norm,
            timestep_shift=options.timestep_shift,
            enable_timestep_shift=options.enable_timestep_shift,
            cfg_interval=options.cfg_interval,
            num_steps=int(batch.num_inference_steps),
            batch_size=1,
            t_eps=options.t_eps,
            think_mode=options.think_mode,
            seed=seed,
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
