"""Speculative-decoding kernels (Triton).

The Triton kernels migrated here live in this package
(``sglang.kernels.ops.speculative.<module>``); import them from there. Their
``KernelSpec`` metadata is registered below for inventory (backend = Triton).
"""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec

# (module, public_fn) migrated from speculative/triton_ops.
_TRITON_KERNELS = [
    ("cache_locs", "assign_extend_cache_locs_func"),
    ("cache_locs", "generate_draft_decode_kv_indices"),
    ("eagle", "fill_bonus_tokens"),
    ("eagle", "fill_accept_out_cache_loc"),
    ("gather_spec_extras", "gather_spec_extras"),
    ("multi_layer_eagle", "rotate_input_ids"),
    ("spec_tree", "sgl_build_tree_kernel_efficient_triton"),
    ("spec_tree", "verify_tree_greedy_kernel_triton"),
    ("topk1", "draft_topk1_postprocess"),
    ("topk1", "target_verify_topk1_postprocess"),
    ("ragged_verify_kernels", "pad_verify_lens_to_bucket"),
    ("ragged_verify_kernels", "build_qo_indptr"),
    ("reject_sampling", "chain_speculative_sampling_triton"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"speculative.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.speculative.{_mod}:{_fn}",
        )
    )
del _mod, _fn

__all__ = []
