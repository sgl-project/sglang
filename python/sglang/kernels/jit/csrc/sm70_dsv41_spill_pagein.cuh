// SPDX-License-Identifier: Apache-2.0
// Decode-path UVA page-in of spilled MXFP4 expert rows.
// Assigns misses to landing slots and copies host-mapped rows into a shared
// GPU landing pool. Prefill keeps the Python LRU; this kernel is capturable.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_dsv41 {

// Greedy D4-G uses 6 slots. DSpark verify is T=γ+1=6 → landing 36
// (6*(γ+1)). Bound 48 covers T=6 × topk=8 unique-miss worst case.
// assigned_id[] is one-thread stack; 48×4 B is fine.
inline constexpr int32_t kMaxLanding = 48;
inline constexpr int32_t kMaxTensors = 8;
inline constexpr int32_t kCopyThreads = 256;

/**
 * \brief Map topk logical ids onto kept GPU slots or landing indices.
 *
 * Hits (map_table[id] >= 0) stay on the layer Marlin table. Misses get a
 * landing slot 0..n_landing-1 (reused if the same id appears twice) and
 * topk_ids is set to -1 so the kept Marlin skips them.
 *
 * \param slot_host_row Output [n_landing]; host row for each used slot, else -1.
 */
__global__ void spill_assign_kernel(int32_t* __restrict__ topk_ids,
                                    int32_t* __restrict__ land_ids,
                                    int32_t* __restrict__ slot_host_row,
                                    const int32_t* __restrict__ map_table,
                                    const int32_t* __restrict__ host_map,
                                    int32_t n_tok,
                                    int32_t k,
                                    int32_t n_logical,
                                    int32_t n_landing) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int32_t assigned_id[kMaxLanding];
  int32_t next = 0;
  for (int32_t s = 0; s < n_landing; ++s) {
    assigned_id[s] = -1;
    slot_host_row[s] = -1;
  }
  const int32_t n = n_tok * k;
  for (int32_t i = 0; i < n; ++i) {
    const int32_t id = topk_ids[i];
    if (id < 0 || id >= n_logical) {
      land_ids[i] = -1;
      continue;
    }
    const int32_t phys = map_table[id];
    if (phys >= 0) {
      topk_ids[i] = phys;
      land_ids[i] = -1;
      continue;
    }
    int32_t slot = -1;
    for (int32_t s = 0; s < next; ++s) {
      if (assigned_id[s] == id) {
        slot = s;
        break;
      }
    }
    if (slot < 0) {
      const int32_t hs = host_map[id];
      if (hs < 0 || next >= n_landing) {
        topk_ids[i] = -1;
        land_ids[i] = -1;
        continue;
      }
      slot = next;
      assigned_id[slot] = id;
      slot_host_row[slot] = hs;
      ++next;
    }
    topk_ids[i] = -1;
    land_ids[i] = slot;
  }
}

/**
 * \brief Copy one expert row from a UVA host base into a landing slot.
 *
 * Grid.x is the landing slot. Each block walks every tensor listed in
 * src_ptrs/dst_ptrs/row_bytes. Unused slots (host_row < 0) return
 * immediately. Forcing a UVA dummy copy (host row 0 or the first mapped
 * row) made every launch copy landing-count expert rows over PCIe:
 * relaunch137 ~283 ms/launch, relaunch138 still ~5 s/verify. Marlin never
 * reads unused dest slots. Rank-invariant duration is not worth that.
 */
__global__ __launch_bounds__(kCopyThreads) void spill_copy_kernel(
    const int32_t* __restrict__ slot_host_row,
    const int64_t* __restrict__ src_ptrs,
    const int64_t* __restrict__ dst_ptrs,
    const int64_t* __restrict__ row_bytes,
    int32_t n_tensors,
    int32_t n_landing) {
  const int32_t slot = static_cast<int32_t>(blockIdx.x);
  (void)n_landing;
  const int32_t hs = slot_host_row[slot];
  if (hs < 0) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int nt = static_cast<int>(blockDim.x);
  for (int32_t t = 0; t < n_tensors; ++t) {
    const int64_t nbytes = row_bytes[t];
    if (nbytes <= 0) {
      continue;
    }
    const uint8_t* src =
        reinterpret_cast<const uint8_t*>(src_ptrs[t]) + static_cast<int64_t>(hs) * nbytes;
    uint8_t* dst =
        reinterpret_cast<uint8_t*>(dst_ptrs[t]) + static_cast<int64_t>(slot) * nbytes;
    int64_t off = static_cast<int64_t>(tid) * 16;
    const int64_t stride = static_cast<int64_t>(nt) * 16;
    for (; off + 16 <= nbytes; off += stride) {
      *reinterpret_cast<uint4*>(dst + off) =
          *reinterpret_cast<const uint4*>(src + off);
    }
    for (int64_t b = static_cast<int64_t>(tid) + (nbytes & ~int64_t{15}); b < nbytes;
         b += nt) {
      dst[b] = src[b];
    }
  }
}

/**
 * \brief Decode spill page-in: remap topk_ids and fill landing rows from UVA.
 *
 * \param topk_ids      int32 [T, K] in/out (kept physical or -1)
 * \param land_ids      int32 [T, K] out (landing slot or -1)
 * \param slot_host_row int32 [n_landing] workspace
 * \param map_table     int32 [n_logical] GPU slot or -1
 * \param host_map      int32 [n_logical] host row or -1
 * \param src_ptrs      int64 [n_tensors] UVA bases
 * \param dst_ptrs      int64 [n_tensors] landing bases
 * \param row_bytes     int64 [n_tensors] bytes per expert row
 */
inline void spill_page_in(tvm::ffi::TensorView topk_ids,
                          tvm::ffi::TensorView land_ids,
                          tvm::ffi::TensorView slot_host_row,
                          tvm::ffi::TensorView map_table,
                          tvm::ffi::TensorView host_map,
                          tvm::ffi::TensorView src_ptrs,
                          tvm::ffi::TensorView dst_ptrs,
                          tvm::ffi::TensorView row_bytes) {
  using namespace host;
  SymbolicSize n_tok = {"n_tok"};
  SymbolicSize k = {"k"};
  SymbolicSize n_logical = {"n_logical"};
  SymbolicSize n_landing = {"n_landing"};
  SymbolicSize n_tensors = {"n_tensors"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_tok, k})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(topk_ids)
      .verify(land_ids);
  TensorMatcher({n_landing})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(slot_host_row);
  TensorMatcher({n_logical})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(map_table)
      .verify(host_map);
  TensorMatcher({n_tensors})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(src_ptrs)
      .verify(dst_ptrs)
      .verify(row_bytes);

  const int32_t t = static_cast<int32_t>(n_tok.unwrap());
  const int32_t kk = static_cast<int32_t>(k.unwrap());
  const int32_t nlog = static_cast<int32_t>(n_logical.unwrap());
  const int32_t nland = static_cast<int32_t>(n_landing.unwrap());
  const int32_t ntens = static_cast<int32_t>(n_tensors.unwrap());
  CHECK_HOST(nland > 0 && nland <= kMaxLanding)
      << "sm70_dsv41 spill_page_in: n_landing " << nland;
  CHECK_HOST(ntens > 0 && ntens <= kMaxTensors)
      << "sm70_dsv41 spill_page_in: n_tensors " << ntens;
  CHECK_HOST(t * kk > 0) << "sm70_dsv41 spill_page_in: empty topk";

  const DLDevice dev = device_.unwrap();
  LaunchKernel(1, 32, dev)(
      spill_assign_kernel,
      static_cast<int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(land_ids.data_ptr()),
      static_cast<int32_t*>(slot_host_row.data_ptr()),
      static_cast<const int32_t*>(map_table.data_ptr()),
      static_cast<const int32_t*>(host_map.data_ptr()),
      t,
      kk,
      nlog,
      nland);
  LaunchKernel(nland, kCopyThreads, dev)(
      spill_copy_kernel,
      static_cast<const int32_t*>(slot_host_row.data_ptr()),
      static_cast<const int64_t*>(src_ptrs.data_ptr()),
      static_cast<const int64_t*>(dst_ptrs.data_ptr()),
      static_cast<const int64_t*>(row_bytes.data_ptr()),
      ntens,
      nland);
}

}  // namespace sglang::sm70_dsv41
