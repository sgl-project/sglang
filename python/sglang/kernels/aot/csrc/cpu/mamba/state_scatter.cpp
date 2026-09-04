#include "common.h"

namespace {

constexpr int64_t kMaxEntryDims = 4;

template <typename scalar_t>
void mamba_state_scatter_with_mask_kernel_impl(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const int32_t* __restrict__ dst_indices,
    const int32_t* __restrict__ step_indices,
    const int64_t* __restrict__ entry_sizes,
    const int64_t* __restrict__ src_entry_strides,
    const int64_t* __restrict__ dst_entry_strides,
    int64_t entry_ndim,
    int64_t elem_per_entry,
    bool entry_contiguous,
    int64_t num_layers,
    int64_t num_requests,
    int64_t dst_layer_stride,
    int64_t dst_req_stride,
    int64_t src_layer_stride,
    int64_t src_req_stride,
    int64_t src_step_stride,
    int64_t dst_cache_size,
    int64_t src_req_size,
    int64_t src_step_size) {
  at::parallel_for(0, num_requests * num_layers, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t req = i / num_layers;
      int64_t layer = i % num_layers;

      // A padded request carries a negative step or slot and commits nothing
      int64_t step = step_indices[req];
      int64_t dst_index = dst_indices[req];
      if (step < 0 || step >= src_step_size || req >= src_req_size) {
        continue;
      }
      if (dst_index < 0 || dst_index >= dst_cache_size) {
        continue;
      }

      scalar_t* __restrict__ dst_entry = dst + layer * dst_layer_stride + dst_index * dst_req_stride;
      const scalar_t* __restrict__ src_entry =
          src + layer * src_layer_stride + req * src_req_stride + step * src_step_stride;

      if (entry_contiguous) {
        std::memcpy(dst_entry, src_entry, elem_per_entry * sizeof(scalar_t));
        continue;
      }
      // Odometer over the entry dims, advancing both offsets by their strides
      int64_t coord[kMaxEntryDims] = {0};
      int64_t src_offset = 0;
      int64_t dst_offset = 0;
      for (int64_t e = 0; e < elem_per_entry; ++e) {
        dst_entry[dst_offset] = src_entry[src_offset];
        for (int64_t d = entry_ndim - 1; d >= 0; --d) {
          coord[d]++;
          src_offset += src_entry_strides[d];
          dst_offset += dst_entry_strides[d];
          if (coord[d] < entry_sizes[d]) {
            break;
          }
          coord[d] = 0;
          src_offset -= entry_sizes[d] * src_entry_strides[d];
          dst_offset -= entry_sizes[d] * dst_entry_strides[d];
        }
      }
    }
  });
}

}  // anonymous namespace

// Commit one accepted per-step state per request into the persistent caches,
// for every mamba layer at once:
//
//   dst[layer, dst_indices[i]] = src[layer, i, step_indices[i]]
//
// CPU counterpart of the Triton fused_mamba_state_scatter_with_mask and
// fused_conv_window_scatter_with_mask.
//
// Either entry may be strided - the CPU conv kernel keeps conv_states
// dim-contiguous, and the deduplicated conv-window source overlaps its
// per-step windows - so both sides are indexed through their own strides.
//
//   dst : [num_layers, cache_size, *entry]
//   src : [num_layers, num_requests, steps, *entry]
//   dst_indices / step_indices : [num_requests] int32; a negative entry marks
//       a request with nothing to commit
//
void mamba_state_scatter_with_mask_cpu(
    at::Tensor& dst, const at::Tensor& src, const at::Tensor& dst_indices, const at::Tensor& step_indices) {
  CHECK_CPU(dst);
  CHECK_CPU(src);
  CHECK_GE(dst.dim(), 2);
  CHECK_EQ(src.dim(), dst.dim() + 1);
  CHECK_EQ(dst.scalar_type(), src.scalar_type());
  CHECK_EQ(dst.size(0), src.size(0));

  const int64_t entry_ndim = dst.dim() - 2;
  TORCH_CHECK(
      entry_ndim <= kMaxEntryDims,
      "mamba_state_scatter_with_mask_cpu: expect at most ",
      kMaxEntryDims,
      " entry dims, got ",
      entry_ndim);
  for (int64_t d = 0; d < entry_ndim; ++d) {
    CHECK_EQ(dst.size(d + 2), src.size(d + 3));
  }

  CHECK_DIM(1, dst_indices);
  CHECK_DIM(1, step_indices);
  CHECK_CONTIGUOUS(dst_indices);
  CHECK_CONTIGUOUS(step_indices);
  CHECK_EQ(dst_indices.scalar_type(), at::kInt);
  CHECK_EQ(step_indices.scalar_type(), at::kInt);
  CHECK_EQ(dst_indices.size(0), step_indices.size(0));

  const int64_t num_requests = step_indices.size(0);

  // Layer and slot strides are read off the tensors because an envelope pool
  // view spaces its slots out
  int64_t elem_per_entry = 1;
  int64_t entry_sizes[kMaxEntryDims];
  int64_t src_entry_strides[kMaxEntryDims];
  int64_t dst_entry_strides[kMaxEntryDims];
  bool entry_contiguous = true;
  int64_t expected = 1;
  for (int64_t d = entry_ndim - 1; d >= 0; --d) {
    entry_sizes[d] = dst.size(d + 2);
    src_entry_strides[d] = src.stride(d + 3);
    dst_entry_strides[d] = dst.stride(d + 2);
    if (dst.size(d + 2) != 1) {
      entry_contiguous = entry_contiguous && src.stride(d + 3) == expected && dst.stride(d + 2) == expected;
    }
    expected *= dst.size(d + 2);
    elem_per_entry *= dst.size(d + 2);
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, dst.scalar_type(), "mamba_state_scatter_with_mask_kernel_impl", [&] {
        mamba_state_scatter_with_mask_kernel_impl<scalar_t>(
            dst.data_ptr<scalar_t>(),
            src.data_ptr<scalar_t>(),
            dst_indices.data_ptr<int32_t>(),
            step_indices.data_ptr<int32_t>(),
            entry_sizes,
            src_entry_strides,
            dst_entry_strides,
            entry_ndim,
            elem_per_entry,
            entry_contiguous,
            dst.size(0),
            num_requests,
            dst.stride(0),
            dst.stride(1),
            src.stride(0),
            src.stride(1),
            src.stride(2),
            dst.size(1),
            src.size(1),
            src.size(2));
      });
}
