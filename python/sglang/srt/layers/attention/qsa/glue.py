"""Assembly helpers shared by models that carry a QSA indexer."""

from __future__ import annotations

from sglang.srt.layers.attention.qsa.config import parse_qsa_profile


def build_qsa_indexer(
    config,
    *,
    layer_id: int,
    quant_config=None,
    prefix: str = "",
    rotary_emb=None,
    distributed_topk_group=None,
):

    profile = parse_qsa_profile(config)
    if profile is None:
        raise ValueError(
            "build_qsa_indexer requires a config with a QSA indexer schema"
        )
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    from sglang.srt.layers.attention.qsa.cache_sharding import (
        get_qsa_cache_sharding_runtime,
    )

    cache_sharding_runtime = get_qsa_cache_sharding_runtime()
    expected_group = (
        cache_sharding_runtime.group if cache_sharding_runtime.enabled else None
    )
    if distributed_topk_group is not None and distributed_topk_group is not expected_group:
        raise ValueError(
            "QSA indexer collective group does not match cache-sharding runtime"
        )
    kwargs = {
        "config": config,
        "layer_id": layer_id,
        "quant_config": quant_config,
        "prefix": prefix,
        "rotary_emb": rotary_emb,
        "cache_sharding_runtime": cache_sharding_runtime,
    }
    if expected_group is not None:
        kwargs["distributed_topk_group"] = expected_group
    return QSAIndexer(**kwargs)


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
