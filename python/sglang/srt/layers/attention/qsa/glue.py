"""Assembly helpers shared by models that carry a QSA indexer.

Models stay free of backend-wrapping details: they build their indexer via
``build_qsa_indexer`` and fetch per-forward indexer metadata through
``get_qsa_indexer_metadata``.  Hybrid wrappers expose ``get_indexer_metadata``
themselves; the unwrap here only remains as a compatibility fallback.
"""

from __future__ import annotations

from sglang.srt.layers.attention.qsa.config import (
    QSA_VARIANT_COMPRESSED,
    parse_qsa_profile,
)


def build_qsa_indexer(
    config,
    *,
    layer_id: int,
    quant_config=None,
    prefix: str = "",
    rotary_emb=None,
):
    """Construct the indexer module matching ``config``'s QSA profile.

    Only the compressed (Qwen4-Exp) variant has an indexer in this tree; the
    tokenwise (qsa_0511) indexer arrives with its pool in a later stage.
    """

    profile = parse_qsa_profile(config)
    if profile is None:
        raise ValueError(
            "build_qsa_indexer requires a config with a QSA indexer schema"
        )
    if profile.variant == QSA_VARIANT_COMPRESSED:
        # The compressed indexer reuses the layer's own Qwen4-Exp RoPE
        # (mrope); there is intentionally no plain-rope path for it here.
        from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

        return QSAIndexer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            rotary_emb=rotary_emb,
        )
    # Tokenwise (Qwen3Next-DSA): the Lightning Indexer owns its plain
    # per-token RoPE; a shared layer rotary is neither needed nor accepted.
    from sglang.srt.layers.attention.qsa.dsa_indexer import QwenDSAIndexer

    return QwenDSAIndexer(
        config=config,
        layer_id=layer_id,
        quant_config=quant_config,
        prefix=prefix,
    )


def resolve_qsa_sparse_backend(attn_backend):
    """Backend owning the QSA MTP sparse-selection hooks.

    The draft-extend backend is the runner's hybrid wrapper; the MTP shared
    selection lives on its full-attention side, so hook callers resolve
    through the same unwrap as ``get_qsa_indexer_metadata``.  The sentinel is
    the state setter -- the one hook every owner (per-step backend and
    multi-step container alike) defines.
    """

    if hasattr(attn_backend, "set_mtp_shared_sparse_indices"):
        return attn_backend
    full_attn_backend = getattr(attn_backend, "full_attn_backend", None)
    if full_attn_backend is not None and hasattr(
        full_attn_backend, "set_mtp_shared_sparse_indices"
    ):
        return full_attn_backend
    return attn_backend


def get_qsa_indexer_metadata(attn_backend, layer_id: int, forward_batch):
    """Fetch indexer metadata from a (possibly hybrid-wrapped) backend.

    Hybrid wrappers forward ``get_indexer_metadata`` to their full-attention
    side for full-attention layers; unwrapping here is only a compatibility
    fallback for backends that predate that passthrough.
    """

    metadata = None
    get_metadata = getattr(attn_backend, "get_indexer_metadata", None)
    if get_metadata is not None:
        metadata = get_metadata(layer_id, forward_batch)
    if metadata is None:
        full_attn_backend = getattr(attn_backend, "full_attn_backend", None)
        if full_attn_backend is not None and full_attn_backend is not attn_backend:
            get_metadata = getattr(full_attn_backend, "get_indexer_metadata", None)
            if get_metadata is not None:
                metadata = get_metadata(layer_id, forward_batch)
    if metadata is None:
        raise RuntimeError("QSA backend did not provide indexer metadata")
    return metadata


__all__ = [
    "build_qsa_indexer",
    "get_qsa_indexer_metadata",
    "resolve_qsa_sparse_backend",
]
