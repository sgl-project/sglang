#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ dst, const scalar_t* __restrict__ src, int size) {
  int d = 0;
#if defined(CPU_CAPABILITY_AVX512)
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();

  for (; d <= size - kVecSize; d += kVecSize) {
    Vec data = Vec::loadu(src + d);
    data.store(dst + d);
  }
#endif
  for (; d < size; ++d) {
    dst[d] = src[d];
  }
}

template <typename scalar_t, typename index_t>
void store_cache_kernel_impl(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const index_t* __restrict__ indices,
    int64_t batch_size,
    int64_t num_pages,
    int64_t row_dim,
    int64_t k_stride,
    int64_t v_stride,
    int64_t kc_stride,
    int64_t vc_stride) {
  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t bs = begin; bs < end; ++bs) {
      const int64_t idx = static_cast<int64_t>(indices[bs]);
      const scalar_t* k_ptr = k + bs * k_stride;
      const scalar_t* v_ptr = v + bs * v_stride;
      scalar_t* kc_ptr = k_cache + idx * kc_stride;
      scalar_t* vc_ptr = v_cache + idx * vc_stride;
      copy_stub(kc_ptr, k_ptr, row_dim);
      copy_stub(vc_ptr, v_ptr, row_dim);
    }
  });
}

}  // anonymous namespace

// check tensor last two dimensions are contiguous
#define CHECK_LAST2_DIM_CONTIGUOUS(x, ndim)                                                 \
  do {                                                                                      \
    const auto& _x = (x);                                                                   \
    const auto _ndim = _x.dim();                                                            \
    const auto _strides = _x.strides();                                                     \
    const auto _sizes = _x.sizes();                                                         \
    TORCH_CHECK(_ndim == ndim, #x " must have " #ndim " dimensions");                       \
    TORCH_CHECK(                                                                            \
        _ndim >= 2 && _strides[_ndim - 1] == 1 && _strides[_ndim - 2] == _sizes[_ndim - 1], \
        #x " must be contiguous at the last two dimensions");                               \
  } while (0)

// [NB]: store_cache takes 3 dimension tensors,
//   This is to avoid the overhead of creating a new TensorImpl
//   from .view(-1, row_dim)
//
//   k       : [batch_size, num_heads, head_size] -> [batch_size, row_dim]
//   v       : [batch_size, num_heads, head_size] -> [batch_size, row_dim]
//   k_cache : [num_pages, num_heads, head_size] -> [num_pages, row_dim]
//   v_cache : [num_pages, num_heads, head_size] -> [num_pages, row_dim]
//   indices : [batch_size]
//
void store_cache_cpu(
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& indices,
    std::optional<int64_t> row_dim) {
  CHECK_LAST2_DIM_CONTIGUOUS(k, 3);
  CHECK_LAST2_DIM_CONTIGUOUS(v, 3);
  CHECK_LAST2_DIM_CONTIGUOUS(k_cache, 3);
  CHECK_LAST2_DIM_CONTIGUOUS(v_cache, 3);
  CHECK_INPUT(indices);

  int64_t batch_size = k.size(0);
  int64_t num_heads = k.size(1);
  int64_t head_size = k.size(2);
  int64_t num_pages = k_cache.size(0);
  int64_t row_dim_value = num_heads * head_size;
  if (row_dim.has_value()) {
    CHECK_EQ(row_dim.value(), row_dim_value);
  }
  CHECK_EQ(indices.size(0), batch_size);

  // strides: batch dimension (dim 0) stride in elements
  int64_t k_stride = k.stride(0);
  int64_t v_stride = v.stride(0);
  int64_t kc_stride = k_cache.stride(0);
  int64_t vc_stride = v_cache.stride(0);

  const auto dtype = k.scalar_type();
  TORCH_CHECK(
      dtype == v.scalar_type() && dtype == k_cache.scalar_type() && dtype == v_cache.scalar_type(),
      "store_cache_cpu: input tensors must have the same dtype");
  const auto index_dtype = indices.scalar_type();
  TORCH_CHECK(index_dtype == at::kLong || index_dtype == at::kInt, "indices must be int64 or int32");

  // dtype : [bfloat16, float16, uint8] for fp8 KV stored as uint8
  // index_dtype : [int64, int32]
  AT_DISPATCH_REDUCED_FLOATING_TYPES_AND(at::ScalarType::Byte, dtype, "store_cache_cpu", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "store_cache_cpu_index", [&] {
      store_cache_kernel_impl<scalar_t, index_t>(
          k.data_ptr<scalar_t>(),
          v.data_ptr<scalar_t>(),
          k_cache.data_ptr<scalar_t>(),
          v_cache.data_ptr<scalar_t>(),
          indices.data_ptr<index_t>(),
          batch_size,
          num_pages,
          row_dim_value,
          k_stride,
          v_stride,
          kc_stride,
          vc_stride);
    });
  });
}

// CPU counterpart of the Triton kernel `copy_all_layer_kv_cache_tiled`:
// for every K/V buffer b, copy the slot rows `src_loc` to `tgt_loc`:
//   buf_b[tgt_loc[i], :] = buf_b[src_loc[i], :]  for i in [0, num_locs)
//
//   data_ptrs : [2 * layer_num] uint64; base address of each K/V buffer
//   strides   : [2 * layer_num] int64; bytes per slot row of each buffer
//   tgt_loc   : [num_locs] int64/int32 slot indices
//   src_loc   : [num_locs] int64/int32 slot indices
//
// Like the Triton kernel, the copy is safe when tgt_loc and src_loc overlap
// arbitrarily: all source rows of a buffer are staged before any target row
// of that buffer is written (gather then scatter).
void copy_all_layer_kv_cache_cpu(
    const at::Tensor& data_ptrs, const at::Tensor& strides, const at::Tensor& tgt_loc, const at::Tensor& src_loc) {
  CHECK_INPUT(data_ptrs);
  CHECK_INPUT(strides);
  CHECK_INPUT(tgt_loc);
  CHECK_INPUT(src_loc);
  CHECK_EQ(data_ptrs.scalar_type(), at::kUInt64);
  CHECK_EQ(strides.scalar_type(), at::kLong);
  CHECK_EQ(tgt_loc.scalar_type(), src_loc.scalar_type());

  int64_t num_bufs = data_ptrs.numel();
  CHECK_EQ(strides.numel(), num_bufs);
  int64_t num_locs = tgt_loc.numel();
  CHECK_EQ(src_loc.numel(), num_locs);
  if (num_bufs == 0 || num_locs == 0) {
    return;
  }

  const uint64_t* __restrict__ ptrs = reinterpret_cast<const uint64_t*>(data_ptrs.data_ptr());
  const int64_t* __restrict__ stride_ptr = strides.data_ptr<int64_t>();

  AT_DISPATCH_INDEX_TYPES(tgt_loc.scalar_type(), "copy_all_layer_kv_cache_cpu", [&] {
    const index_t* __restrict__ tgt_ptr = tgt_loc.data_ptr<index_t>();
    const index_t* __restrict__ src_ptr = src_loc.data_ptr<index_t>();

    at::parallel_for(0, num_bufs, 0, [&](int64_t begin, int64_t end) {
      std::vector<uint8_t> staging;
      for (int64_t b = begin; b < end; ++b) {
        uint8_t* base = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(ptrs[b]));
        const int64_t stride = stride_ptr[b];
        staging.resize(num_locs * stride);
        for (int64_t i = 0; i < num_locs; ++i) {
          std::memcpy(staging.data() + i * stride, base + src_ptr[i] * stride, stride);
        }
        for (int64_t i = 0; i < num_locs; ++i) {
          std::memcpy(base + tgt_ptr[i] * stride, staging.data() + i * stride, stride);
        }
      }
    });
  });
}
