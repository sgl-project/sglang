#include "../splitkv_mla.cuh"

namespace sm90::decode::sparse_fp8 {

template void run_flash_splitkv_mla_int4_sparse_kernel<ModelType::MODEL1, 64>(const SparseAttnDecodeParams& params);

}  // namespace sm90::decode::sparse_fp8
