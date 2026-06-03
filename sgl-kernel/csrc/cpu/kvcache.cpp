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
