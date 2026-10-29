#include <sgl_kernel/tensor.h>  // For SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

/// Index preparation for EAGLE decode, launched from reusable host descriptors.

SGL_DEVICE int64_t load_index(const void* p, int64_t i, int64_t is64) {
  return is64 ? static_cast<const int64_t*>(p)[i] : static_cast<int64_t>(static_cast<const int32_t*>(p)[i]);
}

SGL_DEVICE void store_index(void* p, int64_t i, int64_t v, int64_t is64) {
  if (is64) {
    static_cast<int64_t*>(p)[i] = v;
  } else {
    static_cast<int32_t*>(p)[i] = static_cast<int32_t>(v);
  }
}

struct ChainTreeParams {
  const void* bonus;
  const void* draft;
  const void* seq;
  bool* mask;
  int64_t* positions;
  int64_t* retrieve;
  void* tokens;
  const void* req;
  const void* table;
  int64_t* out_cache_loc;
  int64_t* mrope;
  int64_t bonus_is64, bonus_stride;
  int64_t draft_is64, draft_row, draft_col;
  int64_t seq_is64, seq_stride;
  int64_t tokens_is64;
  int64_t req_is64, req_stride;
  int64_t table_is64, table_row, table_col;
  int64_t bs, width, write_cache, write_mrope;
};

// build_chain_tree's _build_chain_tree_kernel, one thread per offset.
__global__ void chain_tree_kernel(const __grid_constant__ ChainTreeParams p) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t width = p.width;
  const int64_t n = p.bs * width;
  if (offset < n) {
    const int64_t row = offset / width;
    const int64_t col = offset % width;
    const int64_t token = col == 0 ? load_index(p.bonus, row * p.bonus_stride, p.bonus_is64)
                                   : load_index(p.draft, row * p.draft_row + (col - 1) * p.draft_col, p.draft_is64);
    store_index(p.tokens, offset, token, p.tokens_is64);
    const int64_t length = load_index(p.seq, row * p.seq_stride, p.seq_is64);
    p.positions[offset] = length + col;
    if (p.write_cache) {
      const int64_t req = load_index(p.req, row * p.req_stride, p.req_is64);
      p.out_cache_loc[offset] = load_index(p.table, req * p.table_row + (length + col) * p.table_col, p.table_is64);
    }
    if (p.write_mrope) {
      for (int axis = 0; axis < 3; ++axis) {
        p.mrope[axis * n + offset] = length + col;
      }
    }
    p.retrieve[offset] = offset;
    p.retrieve[n + offset] = col + 1 < width ? col + 1 : -1;
    p.retrieve[2 * n + offset] = -1;
  }
  if (offset < n * width) {
    p.mask[offset] = (offset % width) <= ((offset / width) % width);
  }
}

struct ExtendLayoutParams {
  const void* seq;
  int32_t* prefix;
  int32_t* extend;
  void* post;
  int64_t* positions;
  int32_t* starts;
  int64_t* mrope;
  int64_t seq_is64, seq_stride, post_is64;
  int64_t bs, width, write_mrope;
};

// prepare_draft_extend_layout's _prepare_draft_extend_lengths_kernel (FRONT 0, MAKE_POSITIONS).
__global__ void extend_layout_kernel(const __grid_constant__ ExtendLayoutParams p) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t width = p.width;
  if (row < p.bs) {
    const int64_t base = load_index(p.seq, row * p.seq_stride, p.seq_is64);
    p.prefix[row] = static_cast<int32_t>(base > 0 ? base : 0);
    p.extend[row] = static_cast<int32_t>(width);
    store_index(p.post, row, base + width, p.post_is64);
    p.starts[row] = static_cast<int32_t>(row * width);
  }
  if (row < p.bs * width) {
    const int64_t token_base = load_index(p.seq, (row / width) * p.seq_stride, p.seq_is64);
    const int32_t prefix = static_cast<int32_t>(token_base > 0 ? token_base : 0);
    const int32_t position = prefix + static_cast<int32_t>(row % width);
    p.positions[row] = static_cast<int64_t>(position);
    if (p.write_mrope) {
      for (int axis = 0; axis < 3; ++axis) {
        p.mrope[axis * p.bs * width + row] = static_cast<int64_t>(position);
      }
    }
  }
}

struct DraftLocsParams {
  const void* req;
  const void* table;
  const void* seq;
  int64_t* out_cache_loc;
  void* positions;
  void* mrope;
  int64_t req_is64, table_is64, seq_is64, positions_is64, mrope_is64;
  int64_t pool_len, topk, steps, bs, write_positions, write_mrope;
};

// assign_draft_cache_locs_contiguous, one block per request.
__global__ void draft_locs_kernel(const __grid_constant__ DraftLocsParams p) {
  const int64_t pid = blockIdx.x;
  const int64_t copy_len = p.topk * p.steps;
  const int64_t kv_start = load_index(p.seq, pid, p.seq_is64);
  const int64_t base = load_index(p.req, pid, p.req_is64) * p.pool_len;
  for (int64_t i = threadIdx.x; i < copy_len; i += blockDim.x) {
    p.out_cache_loc[pid * copy_len + i] = load_index(p.table, base + kv_start + i, p.table_is64);
  }
  if (p.write_positions) {
    for (int64_t j = threadIdx.x; j < p.topk; j += blockDim.x) {
      store_index(p.positions, pid * p.topk + j, kv_start, p.positions_is64);
      if (p.write_mrope) {
        for (int axis = 0; axis < 3; ++axis) {
          store_index(p.mrope, axis * p.bs * p.topk + pid * p.topk + j, kv_start, p.mrope_is64);
        }
      }
    }
  }
}

struct SpeculativeIndex {
  /// \param anchor Any CUDA tensor on the target device; selects the device and its stream.
  /// \param desc Host int64 descriptor, layout per entry point below.
  static int64_t chain_tree(tvm::ffi::TensorView anchor, tvm::ffi::TensorView desc) {
    const int64_t* d = host_desc(desc, 27);
    ChainTreeParams p{};
    p.bonus = ptr(d[0]);
    p.draft = ptr(d[1]);
    p.seq = ptr(d[2]);
    p.mask = static_cast<bool*>(ptr(d[3]));
    p.positions = static_cast<int64_t*>(ptr(d[4]));
    p.retrieve = static_cast<int64_t*>(ptr(d[5]));
    p.tokens = ptr(d[6]);
    p.req = ptr(d[7]);
    p.table = ptr(d[8]);
    p.out_cache_loc = static_cast<int64_t*>(ptr(d[9]));
    p.mrope = static_cast<int64_t*>(ptr(d[10]));
    p.bonus_is64 = d[11];
    p.bonus_stride = d[12];
    p.draft_is64 = d[13];
    p.draft_row = d[14];
    p.draft_col = d[15];
    p.seq_is64 = d[16];
    p.seq_stride = d[17];
    p.tokens_is64 = d[18];
    p.req_is64 = d[19];
    p.req_stride = d[20];
    p.table_is64 = d[21];
    p.table_row = d[22];
    p.table_col = d[23];
    p.bs = d[24];
    p.width = d[25];
    p.write_cache = d[26] & 1;
    p.write_mrope = (d[26] >> 1) & 1;
    const int64_t total = p.bs * p.width * p.width;
    if (total == 0) return 0;
    host::LaunchKernel(
        dim3(static_cast<uint32_t>(host::div_ceil(total, kBlock))), static_cast<uint32_t>(kBlock), device(anchor))(
        chain_tree_kernel, p);
    return 0;
  }

  static int64_t extend_layout(tvm::ffi::TensorView anchor, tvm::ffi::TensorView desc) {
    const int64_t* d = host_desc(desc, 13);
    ExtendLayoutParams p{};
    p.seq = ptr(d[0]);
    p.prefix = static_cast<int32_t*>(ptr(d[1]));
    p.extend = static_cast<int32_t*>(ptr(d[2]));
    p.post = ptr(d[3]);
    p.positions = static_cast<int64_t*>(ptr(d[4]));
    p.starts = static_cast<int32_t*>(ptr(d[5]));
    p.mrope = static_cast<int64_t*>(ptr(d[6]));
    p.seq_is64 = d[7];
    p.seq_stride = d[8];
    p.post_is64 = d[9];
    p.bs = d[10];
    p.width = d[11];
    p.write_mrope = d[12];
    const int64_t total = p.bs * p.width > p.bs ? p.bs * p.width : p.bs;
    if (total == 0) return 0;
    host::LaunchKernel(
        dim3(static_cast<uint32_t>(host::div_ceil(total, kBlock))), static_cast<uint32_t>(kBlock), device(anchor))(
        extend_layout_kernel, p);
    return 0;
  }

  static int64_t draft_locs(tvm::ffi::TensorView anchor, tvm::ffi::TensorView desc) {
    const int64_t* d = host_desc(desc, 17);
    DraftLocsParams p{};
    p.req = ptr(d[0]);
    p.table = ptr(d[1]);
    p.seq = ptr(d[2]);
    p.out_cache_loc = static_cast<int64_t*>(ptr(d[3]));
    p.positions = ptr(d[4]);
    p.mrope = ptr(d[5]);
    p.req_is64 = d[6];
    p.table_is64 = d[7];
    p.seq_is64 = d[8];
    p.positions_is64 = d[9];
    p.mrope_is64 = d[10];
    p.pool_len = d[11];
    p.topk = d[12];
    p.steps = d[13];
    p.bs = d[14];
    p.write_positions = d[15];
    p.write_mrope = d[16];
    if (p.bs == 0) return 0;
    host::LaunchKernel(dim3(static_cast<uint32_t>(p.bs)), static_cast<uint32_t>(kBlock), device(anchor))(
        draft_locs_kernel, p);
    return 0;
  }

 private:
  static constexpr int64_t kBlock = 128;

  static void* ptr(int64_t v) {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(v));
  }

  static const int64_t* host_desc(tvm::ffi::TensorView desc, int64_t entries) {
    using namespace host;
    CHECK_HOST(
        desc.device().device_type == kDLCPU && is_type<int64_t>(desc.dtype()) && desc.ndim() == 1 &&
        desc.size(0) == entries && (entries <= 1 || desc.stride(0) == 1))
        << "descriptor must be a contiguous host int64 vector of " << entries;
    return static_cast<const int64_t*>(desc.data_ptr());
  }

  static DLDevice device(tvm::ffi::TensorView anchor) {
    host::SymbolicDevice device_;
    device_.set_options<kDLCUDA, kDLROCM>();
    device_.verify(anchor.device());
    return device_.unwrap();
  }
};

}  // namespace sglang
