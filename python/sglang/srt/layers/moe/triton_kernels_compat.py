"""Compatibility layer for triton_kernels package.

Provides fallback implementations when the triton_kernels package
routing function is not available.
"""

import torch

try:
    from triton_kernels.routing import routing

    _HAS_ROUTING = True
except ImportError:
    _HAS_ROUTING = False

if _HAS_ROUTING:
    from triton_kernels.routing import (
        GatherIndx,
        RoutingData,
        ScatterIndx,
    )
else:
    from triton_kernels.matmul_ogs import (
        GatherIndx,
        RoutingData,
        ScatterIndx,
    )


if not _HAS_ROUTING:
    # Following implementation is adapted from https://github.com/triton-lang/triton/pull/8375
    from triton_kernels.topk import topk
    from triton_kernels.tensor import BIT, SparseMatrix, Bitmatrix, make_ragged_tensor_metadata

    def routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act):
        sparse_logits = SparseMatrix(indx=expt_indx, vals=expt_scal, mask=bitmatrix)
        dispatch_indx = sparse_logits.mask_metadata.col_sorted_indx
        combine_indx = sparse_logits.mask_metadata.row_sorted_indx
        ragged_batch_metadata = make_ragged_tensor_metadata(sparse_logits.mask_metadata.col_sum, dispatch_indx.shape[0])
        gate_scal = sparse_logits.vals.flatten()[combine_indx]
        routing_data = RoutingData(gate_scal, ragged_batch_metadata.slice_sizes, n_expts_tot, n_expts_act,
                                ragged_batch_metadata)
        gather_idx = GatherIndx(combine_indx, dispatch_indx)
        scatter_idx = ScatterIndx(dispatch_indx, combine_indx)
        return routing_data, gather_idx, scatter_idx


    def routing(logits, n_expts_act, sm_first=False, expt_indx=None, n_rows=None):

        print(f"routing: {logits.shape=} {n_expts_act=} {sm_first=} {expt_indx=} {n_rows=}")
        if sm_first:
            logits = torch.softmax(logits, dim=-1)
        sparse_logits = topk(logits, n_expts_act, apply_softmax=not sm_first, y_indx=expt_indx, n_rows=n_rows)
        return routing_from_bitmatrix(
            sparse_logits.mask,
            sparse_logits.vals,
            sparse_logits.indx,
            logits.shape[-1],
            n_expts_act,
        )

__all__ = [
    "GatherIndx",
    "RoutingData",
    "ScatterIndx",
    "routing",
]
