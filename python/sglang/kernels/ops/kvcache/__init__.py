"""KV-cache write/transfer kernels.

This group wraps the Triton ``reshape_and_cache`` launcher, whose implementation
now lives in this package (``sglang.kernels.ops.kvcache.cache_ops``) after being
migrated out of ``sglang.srt.layers.attention.triton_ops`` (RFC #29630).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec

if TYPE_CHECKING:
    import torch

register_kernel(
    KernelSpec(
        op="kvcache.reshape_and_cache_flash",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.kvcache.cache_ops:launch_reshape_and_cache_flash",
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
    ("shuffle_5d_cache", "launch_reshape_and_cache_shuffle_5d"),
    ("shuffle_5d_cache", "launch_gather_shuffle_5d_to_linear"),
    ("pd_dcp_gather", "copy_mla_rows_into_pack"),
    ("kv_read_table", "build_kv_read_table"),
    ("kv_read_table", "build_kv_read_table_packed"),
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

register_kernel(
    KernelSpec(
        op="kvcache.store_fp4_index_k_cache",
        backend=KernelBackend.TRITON,
        target=(
            "sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache:store_fp4_index_k_cache"
        ),
    )
)
