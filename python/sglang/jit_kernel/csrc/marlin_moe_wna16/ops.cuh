
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <optional>

void moe_wna16_marlin_gemm(
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& c,
    const tvm::ffi::TensorView& b_q_weight,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& b_bias_or_none,
    const tvm::ffi::TensorView& b_scales,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& global_scale_or_none,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& b_zeros_or_none,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& g_idx_or_none,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& perm_or_none,
    const tvm::ffi::TensorView& workspace,
    const tvm::ffi::TensorView& sorted_token_ids,
    const tvm::ffi::TensorView& expert_ids,
    const tvm::ffi::TensorView& num_tokens_past_padded,
    const tvm::ffi::TensorView& topk_weights,
    int64_t moe_block_size,
    int64_t top_k,
    bool mul_topk_weights,
    bool is_ep,
    DLDataType b_q_type_id,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {}
