# SPDX-License-Identifier: Apache-2.0
"""LiteAttention backend (MoonMath temporal sparse attention on FA3)."""

from __future__ import annotations

import os
import weakref
from typing import Any

import torch

try:
    from lite_attention import LiteAttention, LiteAttentionRegistry, SeqParallelLiteAttention
except ImportError:
    LiteAttention = None
    SeqParallelLiteAttention = None
    LiteAttentionRegistry = None

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_LITE_ATTN_IMPL_REFS: list[weakref.ReferenceType[LiteAttentionImpl]] = []

# Lazily-built calibration registry shared across all LiteAttention modules in
# the model. Only created when a registry ``mode`` is configured.
_LITE_ATTN_REGISTRY: Any = None
_REGISTRY_MODE: str | None = None

# Modes that route threshold selection through the calibration registry rather
# than a per-module instance threshold.
_REGISTRY_MODES = frozenset({"calib", "load", "const", "disable"})


def _register_impl(impl: "LiteAttentionImpl") -> None:
    _LITE_ATTN_IMPL_REFS.append(weakref.ref(impl))


def reset_all_lite_attention_skip_states() -> None:
    """Reset temporal skip masks at the start of each denoising run."""
    for ref in _LITE_ATTN_IMPL_REFS:
        impl = ref()
        if impl is None:
            continue
        impl.reset_skip_state()


def _get_lite_attention_config() -> dict[str, Any]:
    server_args = get_global_server_args()
    cfg = server_args.attention_backend_config
    if cfg is None:
        return {}
    return dict(cfg)


class _NamedModulesView:
    """Minimal ``model``-like shim exposing ``named_modules()`` for the registry.

    ``LiteAttentionRegistry.from_model`` only consumes ``model.named_modules()``,
    so we hand it the LiteAttention modules tracked in ``_LITE_ATTN_IMPL_REFS``
    (which live inside the attention impls, not the DiT module tree).
    """

    def __init__(self, pairs: list[tuple[str, Any]]) -> None:
        self._pairs = pairs

    def named_modules(self):
        return iter(self._pairs)


def _iter_calibratable_modules():
    """Yield ``(stable_name, LiteAttention)`` for every skip-enabled module.

    Cross-/text-attention impls (``enable_skipping=False``) are excluded so the
    calibrator never tries to find thresholds for dense layers. Names are derived
    from the layer prefix when available so calibrate/load runs stay aligned.
    """
    seen: set[int] = set()
    fallback_idx = 0
    for ref in _LITE_ATTN_IMPL_REFS:
        impl = ref()
        if impl is None or not getattr(impl, "enable_skipping", False):
            continue
        la = impl._la
        base = getattr(impl, "prefix", "") or f"lite_attn_{fallback_idx}"
        fallback_idx += 1
        if LiteAttention is not None and isinstance(la, LiteAttention):
            candidates = [(base, la)]
        elif SeqParallelLiteAttention is not None and isinstance(
            la, SeqParallelLiteAttention
        ):
            inner = getattr(la, "lite_attention", []) or []
            candidates = [(f"{base}.node{j}", m) for j, m in enumerate(inner)]
        else:
            candidates = []
        for name, module in candidates:
            if id(module) in seen:
                continue
            seen.add(id(module))
            yield name, module


def ensure_lite_attention_registry() -> None:
    """Build the calibration registry once, if a registry ``mode`` is configured.

    No-op for the legacy static-threshold path (no ``mode`` key), so existing
    ``threshold=...`` runs are unaffected.
    """
    global _LITE_ATTN_REGISTRY, _REGISTRY_MODE

    if _LITE_ATTN_REGISTRY is not None:
        return

    cfg = _get_lite_attention_config()
    mode = cfg.get("mode")
    if mode is None:
        return
    if mode not in _REGISTRY_MODES:
        raise ValueError(
            f"Unknown LiteAttention registry mode {mode!r}. "
            f"Expected one of {sorted(_REGISTRY_MODES)}."
        )
    if LiteAttentionRegistry is None:
        raise ImportError(
            "LiteAttention is not installed; cannot use calibration registry "
            "mode. Build/install from https://github.com/moonmath-ai/LiteAttention."
        )

    pairs = list(_iter_calibratable_modules())
    if not pairs:
        logger.warning(
            "LiteAttention registry mode=%s requested but no skip-enabled "
            "modules were found; skipping registry setup.",
            mode,
        )
        return

    calib_file = cfg.get("calib_file") or cfg.get("filename")
    calib_config: dict[str, Any] = {}
    if "target_error" in cfg:
        calib_config["target_error"] = float(cfg["target_error"])
    if "metric" in cfg:
        calib_config["metric"] = str(cfg["metric"])

    threshold = cfg.get("threshold")
    _LITE_ATTN_REGISTRY = LiteAttentionRegistry.from_model(
        _NamedModulesView(pairs),
        mode=mode,
        threshold=None if threshold is None else float(threshold),
        filename=calib_file,
        calib_config=calib_config or None,
        force=bool(cfg.get("force", False)),
        disabled_steps=int(cfg.get("disabled_steps", 0) or 0),
    )
    _REGISTRY_MODE = mode
    logger.info(
        "LiteAttention registry initialized: mode=%s, modules=%d, file=%s",
        mode,
        len(pairs),
        calib_file,
    )


def save_lite_attention_calibration() -> None:
    """Persist calibrated thresholds to TOML (only when mode='calib')."""
    if _LITE_ATTN_REGISTRY is None or _REGISTRY_MODE != "calib":
        return
    _LITE_ATTN_REGISTRY.save_if_calib()
    logger.info(
        "LiteAttention calibration saved to %s",
        getattr(_LITE_ATTN_REGISTRY, "_filename", None),
    )


class LiteAttentionMetadata(AttentionMetadata):
    pass


class LiteAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(self, **kwargs: dict[str, Any]) -> LiteAttentionMetadata:
        current_timestep = kwargs.get("current_timestep", 0)
        return LiteAttentionMetadata(current_timestep=current_timestep)


class LiteAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.LITE_ATTENTION

    @staticmethod
    def get_impl_cls() -> type["LiteAttentionImpl"]:
        return LiteAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return LiteAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return LiteAttentionMetadataBuilder


class LiteAttentionImpl(AttentionImpl):
    """Wraps MoonMath LiteAttention (FA3 + temporal tile skipping)."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if LiteAttention is None or SeqParallelLiteAttention is None:
            raise ImportError(
                "LiteAttention is not installed. Build and install from "
                "https://github.com/moonmath-ai/LiteAttention "
                "(cd hopper && pip install .). Requires Hopper (H100/H200), "
                "CUDA >= 12.8, and FlashAttention 3."
            )

        if causal:
            raise NotImplementedError(
                "LiteAttention does not support causal attention masks. "
                "Use --attention-backend fa or torch_sdpa for causal layers."
            )

        cfg = _get_lite_attention_config()
        self.prefix = prefix
        is_cross_attention = bool(extra_impl_args.get("is_cross_attention", False))
        enable_skipping_cfg = cfg.get("enable_skipping")
        if enable_skipping_cfg is None:
            # Cross-attention and text↔video blocks must not use temporal skips.
            self.enable_skipping = not is_cross_attention
        else:
            self.enable_skipping = bool(enable_skipping_cfg)

        self.softmax_scale = softmax_scale
        max_batch_size = int(cfg.get("max_batch_size", 2))
        reverse_skip_list = bool(cfg.get("reverse_skip_list", True))
        use_int8 = bool(cfg.get("use_int8", False))

        sp_size = get_ulysses_parallel_world_size()
        self._use_seq_parallel = sp_size > 1

        common_kwargs = dict(
            enable_skipping=self.enable_skipping,
            max_batch_size=max_batch_size,
            reverse_skip_list=reverse_skip_list,
            use_int8=use_int8,
        )
        # When a registry ``mode`` is configured, the per-layer threshold is owned
        # by the calibration registry (const/load/calib/disable). Only bake an
        # instance threshold for the legacy static path (no ``mode``).
        if cfg.get("mode") is None:
            common_kwargs["threshold"] = float(cfg.get("threshold", -10.0))

        if self._use_seq_parallel:
            self._la = SeqParallelLiteAttention(num_nodes=sp_size, **common_kwargs)
        else:
            self._la = LiteAttention(**common_kwargs)

        _register_impl(self)

    def reset_skip_state(self) -> None:
        if hasattr(self._la, "reset_skip_state"):
            self._la.reset_skip_state()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        *,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = query.dtype
        q = query
        k = key
        v = value
        if orig_dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        scale = self.softmax_scale
        if self._use_seq_parallel:
            split_idx = get_sp_parallel_rank()
            out = self._la(
                q,
                k,
                v,
                split_idx,
                scale,
                return_softmax_lse=return_softmax_lse,
            )
        else:
            out = self._la(
                q,
                k,
                v,
                scale,
                return_softmax_lse=return_softmax_lse,
            )

        if return_softmax_lse:
            output, lse = out
            if output.dtype != orig_dtype:
                output = output.to(orig_dtype)
            return output, lse

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)

        self._maybe_capture(q, k, v)
        return out

    def _maybe_capture(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Debug-only: dump a slice of (q,k,v) per step for offline skip studies.

        Enabled via SGLANG_LITE_ATTN_CAPTURE_DIR; targets one layer via
        SGLANG_LITE_ATTN_CAPTURE_PREFIX (default ``blocks.20.attn1``). Captures
        the conditional CFG branch (batch slot 0), first N heads, a strided set
        of query rows, full keys/values, as fp16. No-op unless the env is set.
        """
        cap_dir = os.getenv("SGLANG_LITE_ATTN_CAPTURE_DIR")
        if not cap_dir:
            return
        target = os.getenv("SGLANG_LITE_ATTN_CAPTURE_PREFIX", "blocks.20.attn1")
        if target not in self.prefix:
            return
        if q.dim() != 4:
            return
        n_heads = int(os.getenv("SGLANG_LITE_ATTN_CAPTURE_HEADS", "4"))
        n_qrows = int(os.getenv("SGLANG_LITE_ATTN_CAPTURE_QROWS", "512"))
        seq = q.shape[1]
        step = getattr(self, "_capture_step", 0)
        self._capture_step = step + 1
        os.makedirs(cap_dir, exist_ok=True)
        stride = max(1, seq // n_qrows)
        q_idx = torch.arange(0, seq, stride, device=q.device)[:n_qrows]
        payload = {
            "step": step,
            "scale": float(self.softmax_scale),
            "prefix": self.prefix,
            "q": q[0][q_idx][:, :n_heads, :].to(torch.float16).cpu(),
            "k": k[0][:, :n_heads, :].to(torch.float16).cpu(),
            "v": v[0][:, :n_heads, :].to(torch.float16).cpu(),
        }
        fname = f"{target.replace('.', '_')}_step{step:03d}.pt"
        torch.save(payload, os.path.join(cap_dir, fname))
