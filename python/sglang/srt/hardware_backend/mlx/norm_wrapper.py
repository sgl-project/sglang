"""Custom Metal RMSNorm kernel wrapper and the model patch pass that installs it.

Unlike RoPE (``aot.py`` + ``attention_wrapper.py``), RMSNorm is not part of the
attention/decode path, so it does not go through ``MlxAOTKernelRegistry``. Norm
modules are called directly by the model forward, so we wrap each ``nn.RMSNorm``
module and dispatch to the custom kernel (or fall back to ``mx.fast.rms_norm``)
on each call. Gated by ``SGLANG_MLX_USE_CUSTOM_RMSNORM``; only installed when the
flag is set and the kernel is importable, so the default path is unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _load_custom_rms_norm() -> Optional[Callable]:
    """Return the raw AOT Metal ``rms_norm`` entry point, or None if unavailable.

    This is the nanobind ``_metal.rms_norm``, NOT the validating ``metal.rms_norm``
    wrapper: ``MLXRMSNormWrapper._supported`` already guarantees the shape/dtype
    contract, so the wrapper's duplicate validation and its two per-call
    ``mx.contiguous`` graph nodes are skipped. In exchange, contiguity of both
    arguments is this module's responsibility (the raw entry reads flat
    row-major buffers with no checks).
    """
    try:
        from sgl_kernel import metal
    except ImportError:
        return None
    if getattr(metal, "_metal", None) is None:
        return None
    if getattr(metal, "_IMPORT_ERROR", None) is not None:
        return None
    return metal._metal.rms_norm


def is_plain_rms_norm(module: Any) -> bool:
    """True for the plain Llama RMSNorm: ``y = x * rsqrt(mean(x^2)+eps) * w``.

    The custom kernel implements exactly this form, and ``mlx.nn.RMSNorm`` is it
    (its ``__call__`` is ``mx.fast.rms_norm(x, weight, eps)``). The Gemma
    ``(1 + w)`` variant is a different class and is intentionally NOT matched, so
    it keeps its own forward.

    NOTE: if a target model defines its own plain-RMSNorm class instead of using
    ``mlx.nn.RMSNorm``, broaden this predicate, but only after confirming that
    class computes the plain form and is not the Gemma variant. The patch count
    logged by ``patch_model_norms`` on a real model load tells you whether this
    predicate is firing.
    """
    return isinstance(module, nn.RMSNorm)


class MLXRMSNormWrapper(nn.Module):
    """Wraps one norm module and dispatches to the custom kernel or the fallback."""

    def __init__(self, inner: nn.Module, kernel: Callable):
        super().__init__()
        # Bypass nn.Module.__setattr__ so these are plain attributes, not
        # re-registered submodules/params (mirrors MLXAttentionWrapper).
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_kernel", kernel)
        object.__setattr__(self, "_eps", float(inner.eps))
        self._refresh_weight_cache(inner.weight)

    def _refresh_weight_cache(self, w):
        object.__setattr__(self, "_w_is_1D", w.ndim == 1)
        object.__setattr__(self, "_w_dtype", w.dtype)
        object.__setattr__(self, "_w_shape_0", w.shape[0])
        object.__setattr__(self, "_w_contig", mx.contiguous(w))
        object.__setattr__(self, "_w_ref", w)

    def _supported(self, x: mx.array) -> bool:
        """Whether the custom kernel can handle this call; else we fall back."""
        return (
            x.dtype in (mx.float16, mx.bfloat16, mx.float32)
            and self._w_dtype == x.dtype
            and self._w_is_1D
            and x.shape[-1] == self._w_shape_0
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self._inner.weight is not self._w_ref:
            self._refresh_weight_cache(self._inner.weight)
        if not self._supported(x):
            return self._inner(x)
        w = self._w_contig
        if x.ndim == 2:
            return self._kernel(mx.contiguous(x), w, self._eps)
        else:
            rows = mx.contiguous(x.reshape(-1, x.shape[-1]))
            y = self._kernel(rows, w, self._eps)
            return y.reshape(x.shape)


def _patch_norms(module: nn.Module, wrap: Callable[[nn.Module], nn.Module]) -> int:
    """Recursively replace plain RMSNorm modules with wrappers; returns the count.

    ``mlx.nn.Module`` is dict-like: its entries are params (arrays), submodules,
    and lists/dicts of submodules. We walk those and replace norm leaves in
    place. Idempotent: already-wrapped modules are skipped.
    """
    count = 0
    for name, value in list(module.items()):
        if isinstance(value, MLXRMSNormWrapper):
            continue
        if is_plain_rms_norm(value):
            setattr(module, name, wrap(value))
            count += 1
        elif isinstance(value, nn.Module):
            count += _patch_norms(value, wrap)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if is_plain_rms_norm(item):
                    value[i] = wrap(item)
                    count += 1
                elif isinstance(item, nn.Module) and not isinstance(
                    item, MLXRMSNormWrapper
                ):
                    count += _patch_norms(item, wrap)
        elif isinstance(value, dict):
            for k, item in value.items():
                if is_plain_rms_norm(item):
                    value[k] = wrap(item)
                    count += 1
                elif isinstance(item, nn.Module) and not isinstance(
                    item, MLXRMSNormWrapper
                ):
                    count += _patch_norms(item, wrap)
    return count


def patch_model_norms(model: nn.Module) -> int:
    """Install the custom RMSNorm kernel on plain norm modules (idempotent).

    No-op unless ``SGLANG_MLX_USE_CUSTOM_RMSNORM`` is set and the kernel is
    importable, so the default path is unchanged (norms keep using
    ``mx.fast.rms_norm``).
    """
    if not envs.SGLANG_MLX_USE_CUSTOM_RMSNORM.get():
        return 0
    kernel = _load_custom_rms_norm()
    if kernel is None:
        logger.warning(
            "SGLANG_MLX_USE_CUSTOM_RMSNORM is set but the Metal rms_norm kernel "
            "is unavailable; norms fall back to mx.fast.rms_norm."
        )
        return 0
    count = _patch_norms(model, lambda norm: MLXRMSNormWrapper(norm, kernel))
    if count:
        logger.info("MLX custom RMSNorm kernel installed on %d norm module(s).", count)
    else:
        logger.warning(
            "SGLANG_MLX_USE_CUSTOM_RMSNORM is set but no plain nn.RMSNorm modules "
            "were found to patch; check is_plain_rms_norm() against the model."
        )
    return count
