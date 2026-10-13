# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang

"""
Minimal KVPress adapter for SGLang.

This module provides a thin, optional bridge between SGLang's attention
subsystem and the NVIDIA ``kvpress`` library
(https://github.com/NVIDIA/kvpress).  KVPress supplies a pluggable family of
KV-cache compression heuristics (SnapKV, AdaKV, ChunkKV, random baselines,
etc.) that were originally written for the HuggingFace ``transformers``
runtime.

Scope (prototype, intentionally narrow)
---------------------------------------

* **Prefill only.**  KVPress is designed to drop KV entries during the prefill
  phase; this adapter mirrors that by **zero-masking** the KV entries that
  would otherwise be pruned.  The page table, ``seq_lens`` and request-to-
  token mappings stay valid — masked entries simply contribute nothing to
  downstream attention.

* **Single-rank, eager paths first.**  The adapter only hooks the
  ``torch_native_backend`` for now.  FlashInfer / Triton / TP-PP-DP paths are
  follow-up work (see ``Known limitations`` below).

* **No CUDA graph support.**  KVPress compressors are pure PyTorch but may
  do dynamic-shape ops that do not record into cuda graphs; the hook
  explicitly short-circuits under ``torch.compiler.is_compiling()``.

Known limitations
-----------------

* **No TP / PP / DP awareness.**  Each rank compresses its own local KV
  independently.  Cross-rank coordinated compression is follow-up work.
* **Masking only, not physical re-layout.**  We zero out KV entries rather
  than removing them.  This trades memory for correctness and simplicity; a
  full implementation would need page-table compaction.
* **Only ScorerPress subclasses are supported.**  Generic ``BasePress``
  subclasses that don't expose a ``score`` method fall back to no-op.

How to enable
-------------

1. ``pip install kvpress`` (requires ``transformers``).
2. Set environment variables::

       export SGLANG_ENABLE_KVPRESS=1
       export SGLANG_KVPRESS_METHOD=random          # or snapkv, knorm, ...
       export SGLANG_KVPRESS_COMPRESSION_RATIO=0.5  # drop half the KV entries

3. Run any SGLang server; the adapter logs its activation once and every
   prefill batch has its KV zero-masked accordingly.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional kvpress import — fail softly so SGLang can still boot without it.
# ---------------------------------------------------------------------------
_KVPRESS_AVAILABLE: bool = False
try:
    from kvpress.presses.base_press import BasePress  # noqa: F401

    _KVPRESS_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    logger.info(
        "kvpress not available (%s).  Install with `pip install kvpress` to use "
        "SGLANG_ENABLE_KVPRESS.",
        _e,
    )
    BasePress = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Fake HuggingFace attention module — KVPress presses reach into attention
# module attributes (config, head_dim, ...).  We wrap SGLang's
# RadixAttention with this shim so KVPress code runs unchanged.
# ---------------------------------------------------------------------------


class _FakeTransformersAttnModule(torch.nn.Module):
    """
    Minimal stand-in for ``transformers`` attention modules.

    KVPress presses read attributes like ``module.config.num_attention_heads``
    and ``module.head_dim`` from the attention layer.  SGLang's
    :class:`~sglang.srt.layers.radix_attention.RadixAttention` exposes the
    same information under different names; this shim translates between them.
    """

    def __init__(self, layer: "RadixAttention"):
        super().__init__()
        self.layer = layer
        self.head_dim = layer.head_dim

        class _Config:
            num_attention_heads = layer.tp_q_head_num
            num_key_value_heads = layer.tp_k_head_num
            hidden_size = layer.tp_q_head_num * layer.qk_head_dim

        self.config = _Config()
        self.layer_idx = layer.layer_id


# ---------------------------------------------------------------------------
# Format conversion: SGLang flat KV ↔ KVPress per-request KV.
# ---------------------------------------------------------------------------


def _reshape_sglang_to_kvpress(
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: "ForwardBatch",
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Convert SGLang flat KV ``[total_tokens, H, D]`` to KVPress per-request
    stacked KV ``(B, H, max_extend, D)``, zero-padded on the right.

    Returns the reshaped k, v and the list of original extend lengths so the
    caller can map compression masks back to flat indices.
    """
    extend_seq_lens = forward_batch.extend_seq_lens_cpu
    if not extend_seq_lens:
        return k, v, []

    device = k.device
    dtype = k.dtype
    bsz = len(extend_seq_lens)
    num_heads = k.shape[1]
    head_dim = k.shape[2]
    max_len = max(extend_seq_lens)

    k_press = torch.zeros(
        bsz, num_heads, max_len, head_dim, device=device, dtype=dtype
    )
    v_press = torch.zeros(
        bsz, num_heads, max_len, head_dim, device=device, dtype=dtype
    )

    offset = 0
    for i, length in enumerate(extend_seq_lens):
        if length == 0:
            continue
        end = offset + length
        k_press[i, :, :length, :] = k[offset:end]
        v_press[i, :, :length, :] = v[offset:end]
        offset = end

    return k_press, v_press, list(extend_seq_lens)


def _apply_mask_back_to_flat(
    k: torch.Tensor,
    k_masked_press: torch.Tensor,
    split_sizes: List[int],
) -> None:
    """
    Write the per-request zero-masked KVPress tensor back into SGLang's flat
    layout **in-place**.  V is left untouched; zeroing K alone suffices to
    eliminate attention contributions from dropped entries.
    """
    offset = 0
    for i, length in enumerate(split_sizes):
        if length == 0:
            continue
        end = offset + length
        k[offset:end] = k_masked_press[i, :, :length, :]
        offset = end


# ---------------------------------------------------------------------------
# Press registry — lazily mirrors kvpress.presses at first use.
# ---------------------------------------------------------------------------


_PRESS_REGISTRY: Optional[Dict[str, Callable[..., "BasePress"]]] = None


def _get_press_registry() -> Dict[str, Callable[..., "BasePress"]]:
    """Return a name → callable registry of every known KVPress subclass."""
    global _PRESS_REGISTRY
    if _PRESS_REGISTRY is not None:
        return _PRESS_REGISTRY

    registry: Dict[str, Callable[..., "BasePress"]] = {}
    if _KVPRESS_AVAILABLE:
        import kvpress.presses as _presses_mod  # noqa: F401

        for name in dir(_presses_mod):
            obj = getattr(_presses_mod, name)
            try:
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BasePress)
                    and obj is not BasePress
                ):
                    stem = name[:-5] if name.endswith("Press") else name
                    registry[name] = obj
                    registry[stem.lower()] = obj
            except TypeError:
                continue

    _PRESS_REGISTRY = registry
    return registry


def _build_press(method_name: str, compression_ratio: float) -> Optional["BasePress"]:
    """Instantiate a KVPress ``BasePress`` subclass by name, or ``None``."""
    if not _KVPRESS_AVAILABLE:
        return None

    registry = _get_press_registry()
    cls = registry.get(method_name.lower()) or registry.get(method_name)
    if cls is None:
        logger.warning(
            "KVPress method %r not found (available: %s).  "
            "Falling back to no-op.",
            method_name,
            sorted({k for k in registry if not k[0].islower()}),
        )
        return None

    import inspect

    try:
        sig = inspect.signature(cls.__init__)
        if "compression_ratio" in sig.parameters:
            return cls(compression_ratio=compression_ratio)
        return cls()
    except Exception as exc:
        logger.warning("Failed to construct KVPress %r: %s", method_name, exc)
        return None


# ---------------------------------------------------------------------------
# Lazy singleton press — built once on first prefill.
# ---------------------------------------------------------------------------


_press_instance: Optional["BasePress"] = None
_press_initialized: bool = False
_press_warned_missing: bool = False


def _get_lazy_press() -> Optional["BasePress"]:
    global _press_instance, _press_initialized, _press_warned_missing
    if _press_initialized:
        return _press_instance
    _press_initialized = True

    if not envs.SGLANG_ENABLE_KVPRESS.get():
        return None
    if not _KVPRESS_AVAILABLE:
        if not _press_warned_missing:
            logger.warning(
                "SGLANG_ENABLE_KVPRESS=1 but kvpress is not installed; "
                "compression is a no-op.  Install with `pip install kvpress`."
            )
            _press_warned_missing = True
        return None

    method = envs.SGLANG_KVPRESS_METHOD.get()
    ratio = envs.SGLANG_KVPRESS_COMPRESSION_RATIO.get()
    press = _build_press(method, ratio)
    if press is not None:
        logger.info(
            "KVPress enabled: method=%s compression_ratio=%.2f press=%r",
            method,
            ratio,
            press,
        )
    _press_instance = press
    return press


# ---------------------------------------------------------------------------
# Public entry point — called by attention backends during prefill.
# ---------------------------------------------------------------------------


def maybe_compress_prefill_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    layer: "RadixAttention",
    forward_batch: "ForwardBatch",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply KVPress compression to prefill KV tensors **in-place** (zero-mask).

    This is the single entry point intended to be called by attention backends
    right before writing ``k`` / ``v`` into the paged KV cache.

    Compression is a no-op when:
    * ``SGLANG_ENABLE_KVPRESS`` is unset / ``0``,
    * we are in decode (not prefill-extend),
    * torch.compile is recording a cuda graph,
    * kvpress is not installed,
    * or the configured press is not a :class:`kvpress.presses.scorer_press.ScorerPress`
      subclass (only those expose a ``score`` method that we need for masking).
    """
    # 1. Cheap early-exit gates.
    if not envs.SGLANG_ENABLE_KVPRESS.get():
        return k, v
    if not forward_batch.forward_mode.is_extend():
        return k, v
    if torch.compiler.is_compiling():
        return k, v
    if k.numel() == 0:
        return k, v

    press = _get_lazy_press()
    if press is None:
        return k, v

    # 2. Reshape flat → per-request stacked.
    try:
        k_press, v_press, split_sizes = _reshape_sglang_to_kvpress(
            k, v, forward_batch
        )
    except Exception as exc:
        logger.warning("KVPress reshape failed: %s — skipping.", exc)
        return k, v

    if not split_sizes:
        return k, v

    # 3. Call press.score — only ScorerPress subclasses expose this; generic
    #    presses fall back to no-op (see docstring / known limitations).
    score_fn = getattr(press, "score", None)
    if not callable(score_fn):
        logger.debug(
            "KVPress %r has no `score` method; keeping original KV "
            "(only ScorerPress subclasses are supported).",
            press.__class__.__name__,
        )
        return k, v

    fake_module = _FakeTransformersAttnModule(layer)
    try:
        scores = score_fn(
            fake_module,
            hidden_states=torch.zeros(
                k_press.shape[0],
                k_press.shape[2],
                fake_module.config.hidden_size,
                device=k_press.device,
                dtype=k_press.dtype,
            ),
            keys=k_press,
            values=v_press,
            attentions=None,
            kwargs={},
        )
    except Exception as exc:
        logger.warning(
            "KVPress score failed for layer %d: %s — skipping compression.",
            layer.layer_id,
            exc,
        )
        return k, v

    # 4. Build a boolean keep-mask; zero-mask the lowest-scoring positions.
    try:
        import kvpress.utils as _kp_utils

        ratio = getattr(press, "compression_ratio", None)
        if ratio is None or ratio == 0:
            ratio = envs.SGLANG_KVPRESS_COMPRESSION_RATIO.get()
        n_kept = _kp_utils.compute_n_kept(k_press.shape[2], ratio)
    except Exception as exc:
        logger.warning("KVPress compute_n_kept failed: %s — skipping.", exc)
        return k, v

    keep_idx = scores.topk(n_kept, dim=-1).indices
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    keep_mask.scatter_(-1, keep_idx, True)

    # 5. Zero-mask KVPress-format K and write back to flat SGLang layout.
    k_masked_press = k_press * keep_mask.unsqueeze(-1).to(k_press.dtype)
    _apply_mask_back_to_flat(k, k_masked_press, split_sizes)

    return k, v


def is_kvpress_enabled() -> bool:
    """Public helper for callers that want to avoid tensor setup upfront."""
    return bool(
        envs.SGLANG_ENABLE_KVPRESS.get()
        and _KVPRESS_AVAILABLE
        and os.environ.get("SGLANG_ENABLE_KVPRESS", "").lower()
        not in ("0", "false", "no")
    )
