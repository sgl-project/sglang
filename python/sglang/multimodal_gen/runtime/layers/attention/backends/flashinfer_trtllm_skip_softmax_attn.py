# SPDX-License-Identifier: Apache-2.0
"""FlashInfer trtllm-gen skip-softmax (BLASST) attention backend for sglang-diffusion.

This is the **datacenter Blackwell (SM 100/103, B200/B300)** counterpart of the
``modelopt_skip_softmax`` Triton backend. On SM 100, FlashInfer's ``auto``
dispatch selects the NVIDIA hand-tuned **trtllm-gen** FMHA kernels, which carry a
first-class ``skip_softmax_threshold_scale_factor`` parameter (BLASST,
https://arxiv.org/abs/2512.12087). On SM 120 (consumer Blackwell) the same
public APIs silently fall back to the ``xqa`` backend which *ignores* the
skip-softmax parameter -- hence this backend is gated to SM 10.x.

Kernel entrypoint: :func:`flashinfer.prefill.trtllm_ragged_attention_deepseek`.
Despite the ``deepseek`` name, its documented contract is a generic ragged
(no-KV-cache) MHA prefill taking separate ``q/k/v`` of shape
``[num_tokens, num_heads, head_dim]`` with an ``is_causal`` flag, validated on
B200 for Wan2.2's ``(H=40, D=128, non-causal, bf16)`` self-attention against
torch SDPA (max_diff ~2.4e-4).

Threshold semantics differ from the modelopt Triton path. From the FlashInfer
docstring:

    "The actual threshold value equals the provided threshold_scale_factor
     divided by the context length."

i.e. ``effective_lambda = scale_factor / seq_len_k``. The calibration helper
(:mod:`_modelopt_skip_softmax_calib`) fits

    lambda = a * exp(b * target_sparsity) / seq_len_k

so the value we feed FlashInfer is ``scale_factor = lambda * seq_len_k =
a * exp(b * target_sparsity)``. Calibration JSON / CLI / env are 100% shared
with the modelopt backend (same helper, same schema).

Fallback matrix (each falls back to dense torch SDPA for that launch):
    1. cross-attention (``Q.shape[1] != K.shape[1]``)
    2. dtype not in {bf16, fp16}  (NVFP4/FP8 *models* still hit bf16 here because
       the QKV linear dequantises before attention)
    3. resolved threshold <= 0 (no calibration / out of range / disabled)
    4. any runtime error from the FlashInfer kernel

References:
    - BLASST paper: https://arxiv.org/abs/2512.12087
    - SGLang LLM skip-softmax (same kernel family, KV-cache path): PR #19089,
      ``srt/layers/attention/{trtllm_mha,trtllm_mla,dsa}_backend.py``
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends._modelopt_skip_softmax_calib import (
    CalibEntry,
    component_key_from_prefix,
    compute_threshold,
    load_calibration_file,
    pick_calibration_entry,
    resolve_target_sparsity,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Env-var names (shared with the modelopt backend so users learn one set).
_ENV_THRESHOLD = "SGLANG_DIFFUSION_SKIP_SOFTMAX_THRESHOLD"
_ENV_TARGET_SPARSITY = "SGLANG_DIFFUSION_SKIP_SOFTMAX_TARGET_SPARSITY"
_ENV_CALIB_PATH = "SGLANG_DIFFUSION_SKIP_SOFTMAX_CALIBRATION_PATH"

# attention_backend_config keys (consumed via --attention-backend-config k=v).
_ABC_KEY_TARGET_SPARSITY = "target_sparsity"
_ABC_KEY_CALIB_PATH = "calibration_path"
_ABC_KEY_FIXED_THRESHOLD = "fixed_threshold"
# v2-only: directly supply the FlashInfer-native scale_factor, skip calibration.
_ABC_KEY_FIXED_SCALE_FACTOR = "fixed_scale_factor"

# Workspace buffer size for the trtllm-gen kernel. 256 MiB matches the LLM
# trtllm backends; bump via env if a very long sequence underruns.
_WORKSPACE_BYTES = (
    int(os.environ.get("SGLANG_DIFFUSION_SKIP_SOFTMAX_WORKSPACE_MB", "256"))
    * 1024
    * 1024
)


def _read_float_env(name: str, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        warnings.warn(
            f"[skip_softmax/trtllm] env {name}={raw!r} is not a valid float; "
            f"using default {default!r}.",
            stacklevel=2,
        )
        return default


def _resolve_settings_from_server_args() -> dict[str, Any]:
    """Read overrides from ``server_args.attention_backend_config``."""
    try:
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args
    except ImportError:
        return {}
    try:
        sa = get_global_server_args()
    except Exception:
        return {}
    cfg = getattr(sa, "attention_backend_config", None) or {}
    out: dict[str, Any] = {}
    for key in (
        _ABC_KEY_TARGET_SPARSITY,
        _ABC_KEY_CALIB_PATH,
        _ABC_KEY_FIXED_THRESHOLD,
        _ABC_KEY_FIXED_SCALE_FACTOR,
    ):
        val = cfg.get(key) if hasattr(cfg, "get") else None
        if val not in (None, "", {}):
            out[key] = val
    return out


def _import_trtllm_ragged_attention():
    try:
        from flashinfer.prefill import (  # noqa: PLC0415
            trtllm_ragged_attention_deepseek,
        )
    except ImportError as e:
        raise ImportError(
            "FlashInferTrtllmSkipSoftmaxBackend requires `flashinfer-python` "
            ">= 0.6.4 with the trtllm-gen prefill kernels "
            "(`flashinfer.prefill.trtllm_ragged_attention_deepseek`). Install "
            "via `pip install -U flashinfer-python`."
        ) from e
    return trtllm_ragged_attention_deepseek


# Per-device workspace cache so all impls in a process share one buffer.
_WORKSPACE_BY_DEVICE: dict[torch.device, torch.Tensor] = {}


def _get_workspace(device: torch.device) -> torch.Tensor:
    ws = _WORKSPACE_BY_DEVICE.get(device)
    if ws is None:
        ws = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _WORKSPACE_BY_DEVICE[device] = ws
    return ws


class FlashInferTrtllmSkipSoftmaxBackend(AttentionBackend):
    """Skip-softmax (BLASST) via FlashInfer trtllm-gen FMHA (SM 100/103 only)."""

    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # trtllm-gen FMHA supports 64/128 (and MLA 192/128); Wan2.2 uses 128.
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FLASHINFER_TRTLLM_SKIP_SOFTMAX

    @staticmethod
    def get_impl_cls() -> type[FlashInferTrtllmSkipSoftmaxImpl]:
        return FlashInferTrtllmSkipSoftmaxImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata


class FlashInferTrtllmSkipSoftmaxImpl(AttentionImpl):
    """AttentionImpl calling FlashInfer trtllm-gen ragged FMHA with skip-softmax."""

    # Shared calibration cache keyed by absolute path (parse JSON once).
    _calib_cache_by_path: dict[str, dict[str, CalibEntry]] = {}

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = bool(causal)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.prefix = prefix

        self._trtllm_ragged_attention = _import_trtllm_ragged_attention()

        # ----- resolve config: server_args > env > defaults --------------
        sa_cfg = _resolve_settings_from_server_args()
        self.fixed_threshold = float(
            sa_cfg.get(
                _ABC_KEY_FIXED_THRESHOLD, _read_float_env(_ENV_THRESHOLD, default=0.0)
            )
        )
        # Direct FlashInfer-native scale_factor (skips calibration math). 0 = off.
        self.fixed_scale_factor = float(sa_cfg.get(_ABC_KEY_FIXED_SCALE_FACTOR, 0.0))
        self._target_sparsity_override = float(
            sa_cfg.get(
                _ABC_KEY_TARGET_SPARSITY,
                _read_float_env(_ENV_TARGET_SPARSITY, default=0.0),
            )
        )
        self.calib_path: str | None = (
            sa_cfg.get(_ABC_KEY_CALIB_PATH) or os.environ.get(_ENV_CALIB_PATH) or None
        )

        # ----- load + cache calibration JSON ------------------------------
        self._calibration: dict[str, CalibEntry] = {}
        if self.calib_path:
            cache = FlashInferTrtllmSkipSoftmaxImpl._calib_cache_by_path
            if self.calib_path not in cache:
                try:
                    cache[self.calib_path] = load_calibration_file(self.calib_path)
                    logger.info(
                        f"[skip_softmax/trtllm] loaded calibration from "
                        f"{self.calib_path} ({len(cache[self.calib_path])} entries)"
                    )
                except (
                    OSError,
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as e:
                    cache[self.calib_path] = {}
                    warnings.warn(
                        f"[skip_softmax/trtllm] failed to load calibration "
                        f"{self.calib_path!r}: {e!r}; will use fixed threshold only.",
                        stacklevel=2,
                    )
            self._calibration = cache[self.calib_path]

        # ----- SDPA fallback (cross-attn / disabled / out-of-range) ------
        self._fallback = SDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=self.causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )

        logger.info_once(
            f"[skip_softmax/trtllm] backend initialised: "
            f"fixed_threshold={self.fixed_threshold:.6g} "
            f"fixed_scale_factor={self.fixed_scale_factor:.6g} "
            f"target_sparsity_override={self._target_sparsity_override:.3f} "
            f"calib_path={self.calib_path} calib_entries={len(self._calibration)} "
            f"prefix={prefix!r}"
        )

        component = component_key_from_prefix(self.prefix)
        entry = pick_calibration_entry(self._calibration, component)
        if entry is not None:
            a_p, b_p = entry["phases"]["prefill"]
            tgt_p = entry["target_sparse_ratio"]["prefill"]
            logger.info_once(
                f"[skip_softmax/trtllm] component={component} calibration loaded: "
                f"prefill a={a_p:.6g} b={b_p:.6g} target_sparse_ratio={tgt_p:.3f}"
            )
        elif self._calibration:
            logger.warning_once(
                f"[skip_softmax/trtllm] component={component} has no calibration "
                f"entry (available: {sorted(self._calibration)}); will fall back "
                f"to fixed_threshold/scale_factor or dense SDPA."
            )

    # ------------------------------------------------------------------ #
    #  scale_factor resolution                                           #
    # ------------------------------------------------------------------ #
    def _resolve_scale_factor(
        self, seq_len_k: int, *, is_prefill: bool = True
    ) -> float:
        """Resolve the FlashInfer ``skip_softmax_threshold_scale_factor``.

        Returns ``0.0`` to mean "fall back to dense SDPA for this launch".

        Priority:
            1. ``fixed_scale_factor`` (FlashInfer-native, bypasses calibration)
            2. calibration entry -> lambda = a*exp(b*target)/seq_len_k, then
               scale_factor = lambda * seq_len_k = a*exp(b*target)
            3. ``fixed_threshold`` interpreted as a modelopt-style lambda, then
               scale_factor = lambda * seq_len_k
            4. 0 -> dense
        """
        phase = "prefill" if is_prefill else "decode"
        component = component_key_from_prefix(self.prefix)

        if self.fixed_scale_factor > 0.0:
            self._log_resolution_once(
                component=component,
                phase=phase,
                seq_len_k=seq_len_k,
                scale_factor=self.fixed_scale_factor,
                reason="fixed_scale_factor",
            )
            return self.fixed_scale_factor

        entry = pick_calibration_entry(self._calibration, component)
        if entry is not None:
            ab = entry["phases"].get(phase)
            if ab is not None:
                a, b = ab
                target = resolve_target_sparsity(
                    override=self._target_sparsity_override,
                    calib_entry=entry,
                    phase=phase,
                )
                lam = compute_threshold(
                    a=a,
                    b=b,
                    target_sparsity=target,
                    seq_len_k=seq_len_k,
                )
                if lam is not None and lam > 0.0:
                    scale_factor = lam * seq_len_k
                    self._log_resolution_once(
                        component=component,
                        phase=phase,
                        seq_len_k=seq_len_k,
                        scale_factor=scale_factor,
                        reason="calibration",
                        a=a,
                        b=b,
                        target=target,
                        lam=lam,
                    )
                    return scale_factor

        if 0.0 < self.fixed_threshold < 1.0:
            scale_factor = self.fixed_threshold * seq_len_k
            self._log_resolution_once(
                component=component,
                phase=phase,
                seq_len_k=seq_len_k,
                scale_factor=scale_factor,
                reason="fixed_threshold",
                lam=self.fixed_threshold,
            )
            return scale_factor

        self._log_resolution_once(
            component=component,
            phase=phase,
            seq_len_k=seq_len_k,
            scale_factor=0.0,
            reason=(
                "no calibration entry"
                if entry is None
                else f"no entry for phase={phase}"
            ),
        )
        return 0.0

    @staticmethod
    def _log_resolution_once(
        *,
        component: str,
        phase: str,
        seq_len_k: int,
        scale_factor: float,
        reason: str,
        a: float = 0.0,
        b: float = 0.0,
        target: float = 0.0,
        lam: float = 0.0,
    ) -> None:
        if scale_factor > 0.0:
            logger.info_once(
                f"[skip_softmax/trtllm] resolved component={component} phase={phase} "
                f"seq_len_k={seq_len_k} a={a:.6g} b={b:.6g} target_sparsity={target:.3f} "
                f"lambda={lam:.6g} -> scale_factor={scale_factor:.6g} ({reason})"
            )
        else:
            logger.info_once(
                f"[skip_softmax/trtllm] DENSE component={component} phase={phase} "
                f"seq_len_k={seq_len_k} -> fallback to SDPA ({reason})"
            )

    # ------------------------------------------------------------------ #
    #  Forward                                                           #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Run trtllm-gen ragged FMHA with optional skip-softmax.

        Args:
            query: ``[B, S_q, H_q, D]`` (sglang convention).
            key:   ``[B, S_k, H_kv, D]``.
            value: ``[B, S_k, H_kv, D]``.
        Returns:
            ``[B, S_q, H_q, D]``.
        """
        # 1) cross-attention -> dense SDPA.
        if query.shape[1] != key.shape[1]:
            return self._fallback.forward(query, key, value, attn_metadata)

        # 2) dtype guard. trtllm-gen ragged path validated for bf16/fp16.
        if query.dtype not in (torch.bfloat16, torch.float16):
            return self._fallback.forward(query, key, value, attn_metadata)

        B, S_q, H_q, D = query.shape
        _, S_k, H_kv, _ = key.shape
        if S_q != S_k:
            return self._fallback.forward(query, key, value, attn_metadata)

        # 3) threshold resolution.
        scale_factor = self._resolve_scale_factor(seq_len_k=int(S_k), is_prefill=True)
        if scale_factor <= 0.0:
            return self._fallback.forward(query, key, value, attn_metadata)

        # 4) [B, S, H, D] -> ragged [B*S, H, D].
        q_flat = query.reshape(B * S_q, H_q, D).contiguous()
        k_flat = key.reshape(B * S_k, H_kv, D).contiguous()
        v_flat = value.reshape(B * S_k, H_kv, D).contiguous()

        seq_lens = torch.full((B,), S_k, dtype=torch.int32, device=query.device)
        cum_q = torch.arange(
            0, (B + 1) * S_q, S_q, dtype=torch.int32, device=query.device
        )
        cum_kv = torch.arange(
            0, (B + 1) * S_k, S_k, dtype=torch.int32, device=query.device
        )
        workspace = _get_workspace(query.device)

        try:
            out = self._trtllm_ragged_attention(
                query=q_flat,
                key=k_flat,
                value=v_flat,
                workspace_buffer=workspace,
                seq_lens=seq_lens,
                max_q_len=S_q,
                max_kv_len=S_k,
                bmm1_scale=self.softmax_scale,
                bmm2_scale=1.0,
                o_sf_scale=1.0,
                batch_size=B,
                window_left=-1,
                cum_seq_lens_q=cum_q,
                cum_seq_lens_kv=cum_kv,
                enable_pdl=False,
                is_causal=self.causal,
                return_lse=False,
                skip_softmax_threshold_scale_factor=scale_factor,
            )
        except Exception as e:  # noqa: BLE001 - any kernel error -> safe dense path
            logger.warning_once(
                f"[skip_softmax/trtllm] kernel raised {type(e).__name__}: {e}; "
                f"falling back to dense SDPA for this launch."
            )
            return self._fallback.forward(query, key, value, attn_metadata)

        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(B, S_q, H_q, D)


__all__ = [
    "FlashInferTrtllmSkipSoftmaxBackend",
    "FlashInferTrtllmSkipSoftmaxImpl",
]
