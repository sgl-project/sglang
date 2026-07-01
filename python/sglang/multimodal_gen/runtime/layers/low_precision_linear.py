from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

TE_NVFP4_LINEAR_TARGETS_ENV = "SGLANG_DIFFUSION_TE_NVFP4_LINEAR_TARGETS"

_TE_NVFP4_RECIPE = None
_TE_NVFP4_LINEAR_CLS = None
_TE_NVFP4_FP8_AUTOCAST = None
_TE_NVFP4_IMPORT_FAILED = False
_TE_NVFP4_RUNTIME_DISABLED = False


def _target_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return {
        token.strip().lower()
        for token in value.replace(";", ",").split(",")
        if token.strip()
    }


def te_nvfp4_linear_target_enabled(target: str) -> bool:
    tokens = _target_tokens(os.getenv(TE_NVFP4_LINEAR_TARGETS_ENV))
    normalized_target = target.strip().lower()
    return "*" in tokens or "all" in tokens or normalized_target in tokens


def maybe_get_te_nvfp4_linear_runner(
    target: str,
    *,
    pad_m_to: int = 16,
) -> TeNvfp4LinearRunner | None:
    if not te_nvfp4_linear_target_enabled(target):
        return None
    return TeNvfp4LinearRunner(target=target, pad_m_to=pad_m_to)


def _linear_base_for_runtime_policy(layer: nn.Module) -> nn.Module | None:
    base_layer = getattr(layer, "base_layer", layer)
    if base_layer is not layer and not (
        getattr(layer, "merged", False) or getattr(layer, "disable_lora", False)
    ):
        return None
    return base_layer


def _get_te_nvfp4_context():
    global _TE_NVFP4_RECIPE
    global _TE_NVFP4_LINEAR_CLS
    global _TE_NVFP4_FP8_AUTOCAST
    global _TE_NVFP4_IMPORT_FAILED
    global _TE_NVFP4_RUNTIME_DISABLED

    if _TE_NVFP4_IMPORT_FAILED or _TE_NVFP4_RUNTIME_DISABLED:
        return None

    try:
        if (
            _TE_NVFP4_RECIPE is None
            or _TE_NVFP4_LINEAR_CLS is None
            or _TE_NVFP4_FP8_AUTOCAST is None
        ):
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import NVFP4BlockScaling
            from transformer_engine.pytorch import fp8_autocast

            _TE_NVFP4_LINEAR_CLS = te.Linear
            _TE_NVFP4_FP8_AUTOCAST = fp8_autocast
            _TE_NVFP4_RECIPE = NVFP4BlockScaling(
                disable_rht=True,
                disable_stochastic_rounding=True,
                disable_2d_quantization=True,
            )
    except Exception as exc:
        _TE_NVFP4_IMPORT_FAILED = True
        logger.warning_once(f"Disabling TE NVFP4 linear runtime path: {exc}")
        return None

    return (_TE_NVFP4_LINEAR_CLS, _TE_NVFP4_FP8_AUTOCAST, _TE_NVFP4_RECIPE)


def _disable_te_nvfp4_runtime(target: str, exc: Exception) -> None:
    global _TE_NVFP4_RUNTIME_DISABLED
    _TE_NVFP4_RUNTIME_DISABLED = True
    logger.warning_once(f"Disabling TE NVFP4 linear runtime path for {target}: {exc}")


class TeNvfp4LinearRunner:
    def __init__(self, *, target: str, pad_m_to: int = 16) -> None:
        self.target = target
        self.pad_m_to = max(1, int(pad_m_to))
        self._cache: dict[str, nn.Module] = {}
        self._weight_cache_ready: set[str] = set()

    def _weight_cache_arg(self, cache_key: str, *, training: bool) -> bool | None:
        if training:
            return None
        return cache_key not in self._weight_cache_ready

    def _mark_weight_cache_ready(
        self,
        cache_key: str,
        *,
        training: bool,
        is_first_microbatch: bool | None,
    ) -> None:
        if not training and is_first_microbatch:
            self._weight_cache_ready.add(cache_key)

    def _get_linear_context(
        self,
        cache_key: str,
        layer: nn.Module,
        *,
        training: bool,
    ) -> tuple[nn.Module, object, object] | None:
        if get_tp_world_size() != 1:
            return None

        base = _linear_base_for_runtime_policy(layer)
        if base is None:
            return None
        if (
            getattr(base, "quant_method", None).__class__.__name__
            != "UnquantizedLinearMethod"
        ):
            return None

        weight = getattr(base, "weight", None)
        bias = getattr(base, "bias", None)
        if (
            weight is None
            or weight.ndim != 2
            or not weight.is_cuda
            or weight.dtype not in (torch.float16, torch.bfloat16)
            or getattr(base, "tp_size", 1) != 1
            or getattr(base, "skip_bias_add", False)
        ):
            return None
        if bias is not None and (
            bias.device != weight.device or bias.dtype != weight.dtype
        ):
            return None

        context = _get_te_nvfp4_context()
        if context is None:
            return None
        te_linear_cls, fp8_autocast, recipe = context

        input_size = int(getattr(base, "input_size_per_partition", weight.shape[1]))
        output_size = int(getattr(base, "output_size_per_partition", weight.shape[0]))
        cached = self._cache.get(cache_key)
        if (
            cached is None
            or getattr(cached, "weight", None) is not weight
            or getattr(cached, "bias", None) is not bias
        ):
            te_layer = te_linear_cls(
                input_size,
                output_size,
                bias=bias is not None,
                params_dtype=weight.dtype,
                device=weight.device,
            )
            te_layer.weight = weight
            if bias is not None:
                te_layer.bias = bias
            te_layer.train(training)
            self._cache[cache_key] = te_layer
            self._weight_cache_ready.discard(cache_key)
            cached = te_layer
        else:
            cached.train(training)

        return cached, fp8_autocast, recipe

    def try_apply(
        self,
        cache_key: str,
        layer: nn.Module,
        x: torch.Tensor,
        *,
        training: bool,
    ) -> torch.Tensor | None:
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            return None

        context = self._get_linear_context(cache_key, layer, training=training)
        if context is None:
            return None
        te_layer, fp8_autocast, recipe = context

        input_shape = tuple(x.shape)
        if not input_shape or input_shape[-1] != int(te_layer.weight.shape[1]):
            return None

        x_2d = x.reshape(-1, input_shape[-1])
        original_m = int(x_2d.shape[0])
        pad_rows = (-original_m) % self.pad_m_to
        if pad_rows:
            x_2d = F.pad(x_2d, (0, 0, 0, pad_rows))

        is_first_microbatch = self._weight_cache_arg(cache_key, training=training)
        try:
            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                out = te_layer(x_2d, is_first_microbatch=is_first_microbatch)
        except Exception as exc:
            _disable_te_nvfp4_runtime(self.target, exc)
            return None
        self._mark_weight_cache_ready(
            cache_key,
            training=training,
            is_first_microbatch=is_first_microbatch,
        )

        if int(out.shape[0]) != original_m:
            out = out[:original_m]
        return out.reshape(*input_shape[:-1], int(out.shape[-1]))
