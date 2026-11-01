// SPDX-License-Identifier: Apache-2.0
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#define VSA_BHSD true
#include "vsa_block_sparse_sm100/block_sparse_launch_sm100a.cuh"

namespace sglang {

/**
 * \brief FastVideo's warp-specialized tcgen05 block-sparse attention forward for 64-token
 * tiles (sm_100a / sm_103a).
 *
 * \param q, k, v, out  [B, H, S, 128] bf16, contiguous; S == num_blocks * 64, num_blocks even.
 * \param q2k_idx       [B * H * num_blocks, max_kv] int32 key-tile lists (valid below q2k_num).
 * \param q2k_num       [B * H * num_blocks] int32 per-query-tile list lengths (0 allowed).
 * \param block_sizes   [num_blocks] int32 valid tokens per key tile.
 */
struct VsaBlockSparseSm100Kernel {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView q2k_idx,
      const tvm::ffi::TensorView q2k_num,
      const tvm::ffi::TensorView block_sizes,
      tvm::ffi::TensorView out,
      double sm_scale) {
    using namespace host;

    auto B = SymbolicSize{"batch"};
    auto H = SymbolicSize{"heads"};
    auto S = SymbolicSize{"seq_pad"};
    auto R = SymbolicSize{"rows"};
    auto N = SymbolicSize{"num_blocks"};
    auto MAXKV = SymbolicSize{"max_kv"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, H, S, HEAD_DIM})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q)
        .verify(k)
        .verify(v)
        .verify(out);
    TensorMatcher({R, MAXKV}).with_dtype<int32_t>().with_device(device).verify(q2k_idx);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device).verify(q2k_num);
    TensorMatcher({N}).with_dtype<int32_t>().with_device(device).verify(block_sizes);

    const int64_t num_blocks = N.unwrap();
    CHECK_HOST(num_blocks % 2 == 0) << "num_blocks must be even, got " << num_blocks;
    CHECK_HOST(S.unwrap() == num_blocks * BLOCK) << "seq_pad " << S.unwrap() << " != num_blocks * " << BLOCK;
    CHECK_HOST(R.unwrap() == B.unwrap() * H.unwrap() * num_blocks) << "q2k rows must be batch * heads * num_blocks";

    BlockSparseVsaArgs a{};
    a.q = static_cast<const __nv_bfloat16*>(q.data_ptr());
    a.k = static_cast<const __nv_bfloat16*>(k.data_ptr());
    a.v = static_cast<const __nv_bfloat16*>(v.data_ptr());
    a.v_t = nullptr;
    a.o = static_cast<__nv_bfloat16*>(out.data_ptr());
    a.lse = nullptr;
    a.q2k_idx = static_cast<const int*>(q2k_idx.data_ptr());
    a.q2k_num = static_cast<const int*>(q2k_num.data_ptr());
    a.variable_block_sizes = static_cast<const int*>(block_sizes.data_ptr());
    a.batch = static_cast<int>(B.unwrap());
    a.num_heads = static_cast<int>(H.unwrap());
    a.seqlen = static_cast<int>(S.unwrap());
    a.head_dim = HEAD_DIM;
    a.num_blocks = static_cast<int>(num_blocks);
    a.max_kv = static_cast<int>(MAXKV.unwrap());
    a.sm_scale = static_cast<float>(sm_scale);

    const DLDevice dev = device.unwrap();
    auto stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dev.device_type, dev.device_id));
    CHECK_CUDA(launch_block_sparse_sm100a(a, stream)) << "vsa_block_sparse_sm100 launch";
  }
};

}  // namespace sglang
