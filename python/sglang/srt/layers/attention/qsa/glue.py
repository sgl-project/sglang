"""Assembly helpers shared by models that carry a QSA indexer."""

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
    """Backend owning the QSA MTP sparse-selection hooks;
    a hybrid wrapper keeps them on its full-attention side.
    ``set_mtp_shared_sparse_indices`` is the probe, the one hook all owners define."""

    if hasattr(attn_backend, "set_mtp_shared_sparse_indices"):
        return attn_backend
    full_attn_backend = getattr(attn_backend, "full_attn_backend", None)
    if full_attn_backend is not None and hasattr(
        full_attn_backend, "set_mtp_shared_sparse_indices"
    ):
        return full_attn_backend
    return attn_backend


def get_qsa_indexer_metadata(attn_backend, layer_id: int, forward_batch):
    """Fetch indexer metadata from a (possibly hybrid-wrapped) backend."""

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
