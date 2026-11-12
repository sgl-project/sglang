// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// K31HT_DRAFT_AR_LEAN (default off; HT DFlash drafter, with K31HT_DRAFT_SITE_EPI): the draft-layer all-reduce runs the STOCK
// all_reduce_2shot_push_kernel (custom_all_reduce.cuh, included unmodified) on the group's own CustomAllReduceV2 push plane with fewer
// threads per CTA than the stock launcher picks (1024 at the HT draft sizes). Same kernel, same grid (push.num_blocks: the per-block epoch
// counters stay in step with every stock call on the plane), same per-vector arithmetic: bit-identical outputs. Purpose: a 1024-thread CTA
// of this kernel (63 registers / thread) takes a whole SM's register file, so under the HIPRI draft || fold overlap the draft all-reduce can
// only start on drained SMs; a 256-thread CTA needs a quarter of it.
#include "custom_all_reduce.cuh"

#include <algorithm>

namespace sglang {
namespace k31ht_arlean {

template <uint32_t kWorldSize, bool kUsePDL>
struct DraftArLean {
  using Tensor = tvm::ffi::Tensor;

  static void run(const CommunicatorRef comm_ref, const Tensor in, const Tensor out, const int64_t block_size) {
    using namespace host;
    using T = bf16_t;
    const auto& comm = *comm_ref.get();
    CHECK_HOST(comm.get_world_size() == kWorldSize) << comm.get_world_size();
    CHECK_HOST(in.IsContiguous() && out.IsContiguous() && is_type<T>(in.dtype()) && is_type<T>(out.dtype()));
    CHECK_HOST(in.numel() == out.numel() && in.device().device_type == kDLCUDA);
    const auto num_elems = static_cast<uint32_t>(in.numel());
    const auto nbytes = static_cast<int64_t>(in.numel()) * static_cast<int64_t>(sizeof(T));
    CHECK_HOST(static_cast<int64_t>(num_elems) == in.numel());
    CHECK_HOST(nbytes % 16 == 0 && reinterpret_cast<intptr_t>(in.data_ptr()) % 16 == 0);
    const auto num_vecs = static_cast<uint32_t>(num_elems / (16 / sizeof(T)));
    const auto stream = LaunchKernel::resolve_device(in.device());
    const auto& push = comm.get_push_obj();
    const auto chunk_vecs = div_ceil(num_vecs, kWorldSize);
    const auto params = AllReducePushParams<kWorldSize>{
        .input = in.data_ptr(),
        .output = out.data_ptr(),
        .num_vecs = num_vecs,
        .rank = push.rank,
        .ws = push.get_workspace<kWorldSize>(int64_t{2} * chunk_vecs * 16),
    };
    auto bs = static_cast<uint32_t>(block_size);
    bs = std::max(bs, 32u * div_ceil(kWorldSize - 1, push.num_blocks));  // the stock floor: >= 1 warp per peer in all
    CHECK_HOST(bs % 32 == 0 && bs >= 32 && bs <= 1024) << bs;
    LaunchKernel(push.num_blocks, bs, stream).enable_pdl(kUsePDL)(all_reduce_2shot_push_kernel<T, kWorldSize, kUsePDL>, params);
  }
};

}  // namespace k31ht_arlean
}  // namespace sglang
