# SPDX-License-Identifier: Apache-2.0
"""LTX-2 BlockAdapter registration for cache-dit.

Registers an SGLang-owned adapter under the ``LTX2`` prefix so it wins the
``cls_name.startswith(name)`` match in ``BlockAdapterRegister`` over upstream
cache-dit's broader ``LTX`` adapter. The upstream LTX adapter imports
Diffusers LTX symbols (``LTX2PatchFunctor`` and its Diffusers transformer
classes); SGLang's transformer is not a Diffusers class and we do not want to
depend on Diffusers LTX being importable in the SGLang runtime.

Hook contract:

* The transformer is ``LTX2VideoTransformer3DModel`` (single transformer with
  two internal residual streams, not a dual-transformer expert pattern).
* Block list: ``transformer.transformer_blocks`` (``nn.ModuleList`` of
  ``LTX2TransformerBlock``).
* Block forward returns ``tuple[Tensor, Tensor]`` = ``(hidden_states,
  audio_hidden_states)``. cache-dit's ``ForwardPattern.Pattern_0`` treats the
  second stream as ``encoder_hidden_states`` and caches the coherent tuple
  residual at block exit, which is what we want — the audio stream must
  survive a cache hit untouched in shape and finiteness.
* No ``patch_functor`` is wired here. The Diffusers-side ``LTX2PatchFunctor``
  is not portable to SGLang's transformer class.

Registration fires at module load. ``cache_dit_integration.py`` imports this
module near the top so ``BlockAdapterRegister.is_supported`` returns True
before ``enable_cache_on_transformer`` ever runs.

Caveats:

* This adapter is single-GPU + single-transformer. The dual-transformer path
  in ``cache_dit_integration.enable_cache_on_dual_transformer`` is Wan2.2-only
  and intentionally NOT generalized for LTX-2.
* SP/TP all-reduce of cache-similarity decisions is handled separately via
  ``_patch_cache_dit_similarity`` in ``cache_dit_integration.py`` and the
  ``_sglang_sp_group`` / ``_sglang_tp_group`` attributes attached to the
  context manager. Nothing about the adapter changes that path.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from cache_dit import BlockAdapter, ForwardPattern
    from cache_dit.caching.block_adapters import BlockAdapterRegister
except ImportError:  # pragma: no cover — cache-dit is optional at runtime
    logger.debug(
        "cache-dit not installed; LTX2 BlockAdapter registration is a no-op."
    )
    BlockAdapter = None  # type: ignore[assignment]
    ForwardPattern = None  # type: ignore[assignment]
    BlockAdapterRegister = None  # type: ignore[assignment]


def _build_ltx2_adapter(pipe, **kwargs):
    """Adapter factory for LTX-2 (LTX2VideoTransformer3DModel).

    Wraps the transformer's ``transformer_blocks`` ModuleList with
    ``ForwardPattern.Pattern_0`` so cache-dit caches the dual-stream
    residual tuple ``(hidden_states, audio_hidden_states)`` at every block
    exit.
    """
    transformer = pipe.transformer
    return BlockAdapter(
        pipe=pipe,
        transformer=transformer,
        blocks=transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_0,
        check_forward_pattern=False,
        **kwargs,
    )


if BlockAdapterRegister is not None:
    # The "LTX2" prefix wins the startswith match against upstream cache-dit's
    # broader "LTX" adapter for any ``LTX2VideoTransformer3DModel`` instance.
    BlockAdapterRegister.register("LTX2")(_build_ltx2_adapter)
    # SGLang wraps the transformer in `FSDPLTX2VideoTransformer3DModel` when
    # `--use-fsdp-inference true` shards the model across GPUs. The wrapper's
    # `__class__.__name__` no longer starts with "LTX2", so the prefix match
    # fails and cache-dit rejects the transformer. Register `FSDPLTX2` too so
    # the same adapter factory is used whether or not FSDP is active.
    BlockAdapterRegister.register("FSDPLTX2")(_build_ltx2_adapter)
    logger.debug(
        "Registered SGLang LTX2 BlockAdapter (forward_pattern=Pattern_0, "
        "blocks=transformer_blocks) under both LTX2 and FSDPLTX2 prefixes."
    )

__all__ = ["_build_ltx2_adapter"]
