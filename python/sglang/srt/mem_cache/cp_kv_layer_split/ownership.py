"""Model-agnostic ownership math for CP KV LayerSplit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

CP_KV_LAYER_SPLIT_SUPPORTED_MODEL_ARCHS = (
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForCausalLMNextN",
)


def validate_cp_kv_layer_split_model_arch(
    server_args: "ServerArgs", model_arch: str
) -> None:
    """Reject --enable-cp-kv-layer-split when the loaded model arch is not supported."""
    if not server_args.enable_cp_kv_layer_split:
        return

    if model_arch in CP_KV_LAYER_SPLIT_SUPPORTED_MODEL_ARCHS:
        return

    supported = ", ".join(CP_KV_LAYER_SPLIT_SUPPORTED_MODEL_ARCHS)
    raise ValueError(
        f"--enable-cp-kv-layer-split is not supported for model arch {model_arch!r}. "
        f"Supported architectures: {supported}."
    )


CP_KV_LAYER_SPLIT_HICACHE_STORAGE_BACKENDS = ("file", "mooncake")


def assert_cp_kv_layer_split_hicache_supported(server_args: "ServerArgs") -> None:
    """Gate HiCache storage backends that scope keys per attention-CP rank."""
    if not server_args.enable_cp_kv_layer_split:
        return
    backend = server_args.hicache_storage_backend
    if backend is None or backend in CP_KV_LAYER_SPLIT_HICACHE_STORAGE_BACKENDS:
        return
    supported = ", ".join(CP_KV_LAYER_SPLIT_HICACHE_STORAGE_BACKENDS)
    raise ValueError(
        f"--enable-cp-kv-layer-split + --hicache-storage-backend={backend!r} is "
        f"not yet supported. Supported backends under CP KV LayerSplit: {supported} "
        "(other backends need per-CP-rank scoped storage keys, follow-up PR)."
    )


def should_use_cp_kv_layer_split_pool(
    server_args: Optional["ServerArgs"] = None,
) -> bool:
    """True when prefill CP KV layer split is enabled (pool wiring follows the allowlist)."""
    from sglang.srt.server_args import get_global_server_args

    args = server_args or get_global_server_args()
    return bool(
        args.enable_cp_kv_layer_split
        and args.enable_dsa_prefill_context_parallel
        and args.attn_cp_size > 1
    )


def layers_per_cp_rank(num_layers: int, cp_size: int) -> int:
    """Max local KV buffers per CP rank (contiguous block split)."""
    assert num_layers > 0 and cp_size > 0
    return (num_layers + cp_size - 1) // cp_size


def kv_layer_owner(layer_id: int, cp_size: int, num_layers: int) -> int:
    """CP rank that stores KV cache for global transformer layer ``layer_id``."""
    assert 0 <= layer_id < num_layers
    block = layers_per_cp_rank(num_layers, cp_size)
    return min(layer_id // block, cp_size - 1)


def owns_kv_layer(layer_id: int, cp_rank: int, cp_size: int, num_layers: int) -> bool:
    return kv_layer_owner(layer_id, cp_size, num_layers) == cp_rank


def kv_layer_owner_global_rank(layer_id: int, cp_size: int, num_layers: int) -> int:
    """Global process rank (within the attention-CP group) that owns KV for ``layer_id``."""
    from sglang.srt.layers.dp_attention import get_attention_cp_group

    owner_cp = kv_layer_owner(layer_id, cp_size, num_layers)
    return get_attention_cp_group().ranks[owner_cp]


def owned_kv_layer_range(
    cp_rank: int,
    cp_size: int,
    num_layers: int,
    start_layer: int,
    end_layer_exclusive: int,
) -> tuple[int, int]:
    """Return the local PP slice of transformer layers owned by ``cp_rank``."""
    if start_layer >= end_layer_exclusive:
        return start_layer, start_layer

    owned = [
        layer_id
        for layer_id in range(start_layer, end_layer_exclusive)
        if owns_kv_layer(layer_id, cp_rank, cp_size, num_layers)
    ]
    if not owned:
        return start_layer, start_layer
    return owned[0], owned[-1] + 1


def num_owned_kv_layers(
    cp_rank: int,
    cp_size: int,
    num_layers: int,
    start_layer: int,
    end_layer_exclusive: int,
) -> int:
    """Number of transformer layers whose KV cache is stored on ``cp_rank`` on this GPU."""
    owned_start, owned_end = owned_kv_layer_range(
        cp_rank, cp_size, num_layers, start_layer, end_layer_exclusive
    )
    return owned_end - owned_start


def num_owned_compress_layers(
    cp_rank: int,
    cp_size: int,
    num_layers: int,
    start_layer: int,
    end_layer_exclusive: int,
    compression_ratios: list[int],
    compress_ratio: int,
) -> int:
    """Count owned global layers on this CP rank with the given ``compress_ratio``."""
    owned_start, owned_end = owned_kv_layer_range(
        cp_rank, cp_size, num_layers, start_layer, end_layer_exclusive
    )
    return sum(
        1
        for layer_id in range(owned_start, owned_end)
        if compression_ratios[layer_id] == compress_ratio
    )


def num_stage_compress_layers(
    start_layer: int,
    end_layer_exclusive: int,
    compression_ratios: list[int],
    compress_ratio: int,
) -> int:
    return sum(
        1
        for layer_id in range(start_layer, end_layer_exclusive)
        if compression_ratios[layer_id] == compress_ratio
    )


def _family_layer_count(
    *,
    sharded: bool,
    cp_rank: int,
    cp_size: int,
    model_num_hidden_layers: int,
    start_layer: int,
    end_layer_exclusive: int,
    compression_ratios: list[int],
    compress_ratio: Optional[int] = None,
) -> int:
    """Count SWA-like or compressed-family layers for one layout builder."""
    if compress_ratio is None:
        if not sharded:
            return end_layer_exclusive - start_layer
        return num_owned_kv_layers(
            cp_rank,
            cp_size,
            model_num_hidden_layers,
            start_layer,
            end_layer_exclusive,
        )

    if not sharded:
        return num_stage_compress_layers(
            start_layer,
            end_layer_exclusive,
            compression_ratios,
            compress_ratio,
        )
    return num_owned_compress_layers(
        cp_rank,
        cp_size,
        model_num_hidden_layers,
        start_layer,
        end_layer_exclusive,
        compression_ratios,
        compress_ratio,
    )


def build_owned_layer_local_index_map(
    cp_rank: int,
    cp_size: int,
    num_layers: int,
    start_layer: int,
    end_layer_exclusive: int,
) -> dict[int, int]:
    """Map owned global layer id to local buffer index."""
    owned_start, owned_end = owned_kv_layer_range(
        cp_rank, cp_size, num_layers, start_layer, end_layer_exclusive
    )
    return {
        layer_id: layer_id - owned_start for layer_id in range(owned_start, owned_end)
    }
