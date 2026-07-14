"""Memory / KV-slot allocation kernels (Triton).

The Triton kernels migrated here live in this package
(``sglang.kernels.ops.memory.<module>``); import them from there. Their
``KernelSpec`` metadata is registered below for inventory (backend = Triton).
"""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec

# (module, public_fn) migrated from mem_cache/triton_ops.
_TRITON_KERNELS = [
    ("allocator", "alloc_extend_kernel"),
    ("allocator", "alloc_decode_kernel"),
    ("common", "get_last_loc_triton"),
    ("common", "get_last_loc_triton_safe"),
    ("virtual_slot", "alloc_bind_inplace"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"memory.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.memory.{_mod}:{_fn}",
        )
    )
del _mod, _fn

register_kernel(
    KernelSpec(
        op="memory.gather_req_to_token_pool",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.memory.req_to_token_pool:GatherReqToTokenPool.triton",
    )
)
register_kernel(
    KernelSpec(
        op="memory.write_req_to_token_pool",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.memory.req_to_token_pool:WriteReqToTokenPool.triton",
    )
)

__all__ = []


# Migrated from srt/layers (RFC #29630, Phase 2.5).
register_kernel(
    KernelSpec(
        op="memory.gpu_tensor_hash",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.memory.gpu_tensor_hash:gpu_tensor_hash",
    )
)
register_kernel(
    KernelSpec(
        op="memory.memcpy_triton",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.memory.memcpy_triton:memcpy_triton",
    )
)
