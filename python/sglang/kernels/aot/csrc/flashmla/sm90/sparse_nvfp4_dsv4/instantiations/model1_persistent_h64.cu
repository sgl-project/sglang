#include "../splitkv_mla.cuh"

namespace sm90::decode::sparse_nvfp4_dsv4 {

template void run_flash_splitkv_mla_nvfp4_dsv4_sparse_kernel_impl<64>(
    const SparseAttnDecodeParams& params, const float* kv_global_scale, const float* extra_kv_global_scale);

}  // namespace sm90::decode::sparse_nvfp4_dsv4
