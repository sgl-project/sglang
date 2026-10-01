"""Benchmark-only integration for Qwen3.5-9B, BF16 rank-32 adapters.

Keep large-batch shrink unchanged. No device-to-host metadata reads on dispatch.
"""

import torch
import triton
from experimental_kernels import compact_expand, reduce_shrink, split_shrink


def install(namespace):
    original_shrink = namespace["chunked_sgmv_lora_shrink_forward"]
    original_expand = namespace["chunked_sgmv_lora_expand_forward"]

    def shrink(x, weights, batch_info, num_slices):
        rows, k = x.shape
        n = weights.shape[1]
        rank = n // num_slices
        if (
            rows > 128
            or rank != 32
            or k not in (4096, 12288)
            or batch_info.max_len != 16
        ):
            return original_shrink(x, weights, batch_info, num_slices)
        splits = 16 if k == 12288 else 8
        output = torch.empty((rows, n), dtype=x.dtype, device=x.device)
        partials = torch.empty((splits, rows, n), dtype=torch.float32, device=x.device)
        segment_grid = (
            batch_info.weight_indices.shape[0]
            if batch_info.use_cuda_graph
            else batch_info.num_segments
        )
        split_shrink[(triton.cdiv(n, 16), segment_grid, splits)](
            x=x,
            weights=weights,
            output=partials,
            seg_indptr=batch_info.seg_indptr,
            weight_indices=batch_info.weight_indices,
            lora_ranks=batch_info.lora_ranks,
            permutation=batch_info.permutation,
            num_segs=segment_grid,
            N=n,
            K=k,
            NUM_SLICES=num_slices,
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=256,
            SPLIT_K=splits,
            TOTAL_ROWS=rows,
        )
        reduce_shrink[(triton.cdiv(rows * n, 256),)](
            partials, output, ELEMENTS=rows * n, SPLIT_K=splits, BLOCK=256
        )
        return output

    def expand(x, weights, batch_info, slice_offsets, max_slice_size, base_output):
        n, rank = weights.shape[1:]
        slices = len(slice_offsets) - 1
        # These tuples match the model's packed projection layouts. General
        # integration should pass the existing CPU slice metadata explicitly.
        if (slices, n, max_slice_size) == (3, 10240, 8192):
            offsets = (0, 8192, 9216, 10240)
        elif (slices, n, max_slice_size) == (2, 24576, 12288):
            offsets = (0, 12288, 24576)
        elif slices == 1 and max_slice_size == n:
            offsets = (0, n)
        else:
            return original_expand(
                x, weights, batch_info, slice_offsets, max_slice_size, base_output
            )
        if rank != 32 or batch_info.max_len != 16:
            return original_expand(
                x, weights, batch_info, slice_offsets, max_slice_size, base_output
            )
        output = (
            torch.zeros((len(x), n), device=x.device, dtype=x.dtype)
            if base_output is None
            else base_output
        )
        segment_grid = (
            batch_info.weight_indices.shape[0]
            if batch_info.use_cuda_graph
            else batch_info.num_segments
        )
        grid = (
            sum(triton.cdiv(hi - lo, 64) for lo, hi in zip(offsets, offsets[1:])),
            segment_grid,
        )
        compact_expand[grid](
            x=x,
            weights=weights,
            output=output,
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            seg_indptr=batch_info.seg_indptr,
            weight_indices=batch_info.weight_indices,
            lora_ranks=batch_info.lora_ranks,
            permutation=batch_info.permutation,
            num_segs=segment_grid,
            scalings=batch_info.scalings,
            slice_offsets=slice_offsets,
            NUM_SLICES=slices,
            OUTPUT_DIM=n,
            MAX_RANK=rank,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=16,
            OFFSETS=offsets,
        )
        return output

    namespace["chunked_sgmv_lora_shrink_forward"] = shrink
    namespace["chunked_sgmv_lora_expand_forward"] = expand
    print(
        "CSGMV candidate enabled: compact expand; split-K shrink for <=128 rows, rank32, K4096/12288",
        flush=True,
    )
