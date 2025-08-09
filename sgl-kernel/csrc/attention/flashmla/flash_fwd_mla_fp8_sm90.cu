#include "flash_fwd_mla_kernel.h"

namespace flashmla {
template void run_mha_fwd_splitkv_mla<cutlass::float_e4m3_t, cutlass::bfloat16_t, 576>(Flash_fwd_mla_params &params, cudaStream_t stream);
} // flashmla