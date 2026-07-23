"""KV-cache write/transfer kernels.

This group wraps the Triton ``reshape_and_cache`` launcher, whose implementation
now lives in this package (``sglang.kernels.ops.kvcache.cache_ops``) after being
migrated out of ``sglang.srt.layers.attention.triton_ops`` (RFC #29630).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import FormatSignature, KernelBackend, KernelSpec

if TYPE_CHECKING:
    import torch

register_kernel(
    KernelSpec(
        op="kvcache.reshape_and_cache_flash",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.kvcache.cache_ops:launch_reshape_and_cache_flash",
        format_signature=FormatSignature(
            in_place=True,
            description="write token-major K/V into paged KV cache layout",
        ),
        description="Reshape-and-cache (Triton launcher).",
    )
)


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    swa_slot_mapping: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> None:
    """Write token-major ``key``/``value`` into paged KV cache layout."""
    return get_kernel("kvcache.reshape_and_cache_flash", KernelBackend.TRITON)(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        swa_slot_mapping,
        k_scale,
        v_scale,
    )


__all__ = ["reshape_and_cache_flash"]


# Other Triton kernels migrated into this group (from attention/mem_cache
# triton_ops); registered for inventory. Import them from their modules.
_TRITON_KERNELS = [
    ("cache_ops", "concat_and_cast_mha_k_triton"),
    ("cache_ops", "launch_reshape_and_cache_flash"),
    ("kv_indices", "create_flashinfer_kv_indices_triton"),
    ("kv_indices", "create_flashmla_kv_indices_triton"),
    ("kv_indices", "create_chunked_prefix_cache_kv_indices"),
    ("kv_indices", "get_num_kv_index_blocks_flashmla"),
    ("kv_indices", "get_num_page_per_block_flashmla"),
    ("rope_cache", "fused_qk_rope_reshape_and_cache"),
    ("trtllm_fp8_kv_kernel", "fused_fp8_set_kv_buffer"),
    ("trtllm_mha_page_table", "build_trtllm_mha_page_table"),
    ("trtllm_mha_graph_metadata", "update_trtllm_mha_graph_metadata"),
    ("aiter_unified_attention", "scatter_ragged_to_page_table_kernel"),
    ("aiter_unified_attention", "scatter_req_to_token_to_page_table_kernel"),
    ("cache_move", "store_cache_4d"),
    ("cache_move", "set_kv_buffer_prefix_valid_tiled"),
    ("cache_move", "copy_all_layer_kv_cache_tiled"),
    ("mla_buffer", "set_mla_kv_buffer_triton"),
    ("mla_buffer", "get_mla_kv_buffer_triton"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"kvcache.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.kvcache.{_mod}:{_fn}",
        )
    )
del _mod, _fn
