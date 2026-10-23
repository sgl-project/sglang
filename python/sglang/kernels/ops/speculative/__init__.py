"""Speculative-decoding kernels (Triton).

The Triton kernels migrated here live in this package
(``sglang.kernels.ops.speculative.<module>``); import them from there. Their
``KernelSpec`` metadata is registered below for inventory (backend = Triton).
"""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import CUDA_ONLY, KernelBackend, KernelSpec

# (module, public_fn) migrated from speculative/triton_ops.
_TRITON_KERNELS = [
    ("cache_locs", "assign_draft_cache_locs_contiguous_func"),
    ("cache_locs", "assign_extend_cache_locs_func"),
    ("cache_locs", "assign_extend_cache_locs_uniform_func"),
    ("cache_locs", "fill_front_out_cache_loc_func"),
    ("cache_locs", "generate_draft_decode_kv_indices"),
    ("cache_locs", "generate_draft_decode_window_kv_indices"),
    ("dflash", "compute_dflash_correct_drafts_and_bonus"),
    ("eagle", "fill_bonus_tokens"),
    ("eagle", "fill_accept_out_cache_loc"),
    ("gather_spec_extras", "gather_spec_extras"),
    ("multi_layer_eagle", "rotate_input_ids"),
    ("spec_tree", "sgl_build_tree_kernel_efficient_triton"),
    ("spec_tree", "verify_tree_greedy_kernel_triton"),
    ("spec_tree", "build_tree_kernel_efficient_func"),
    ("spec_tree", "verify_tree_greedy_func"),
    ("topk1", "draft_topk1_postprocess"),
    ("uno_tree", "build_uno_tree_proposal"),
    ("ragged_verify_kernels", "pad_verify_lens_to_bucket"),
    ("ragged_verify_kernels", "build_qo_indptr"),
    ("lilicorr", "lilicorr_topk_lse"),
    ("lilicorr", "lilicorr_sample_path"),
    ("reject_sampling", "chain_speculative_sampling_triton"),
    ("row_argmax", "row_argmax"),
    ("reconstruct_tree", "reconstruct_indices_from_tree_mask_triton"),
    ("reconstruct_tree", "reconstruct_indices_from_tree_mask_func"),
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

register_kernel(
    KernelSpec(
        op="speculative.div_argmax",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.speculative.row_argmax:div_argmax",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="speculative.pack_verify_mask_into",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.speculative.pack_verify_mask:pack_verify_mask_into",
    )
)

__all__ = []
