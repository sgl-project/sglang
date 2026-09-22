#include "../topk_select.cuh"

namespace topk_select_bf16_cluster {

template
void run_topk_select_kernel<
    TopkSelectConfig<nv_bfloat16, int32_t, false, false, false, 1024, 256, 1, 4096, 4096, 16, 512, 16>
>(const TopkSelectArgs &args);

}   // topk_select_bf16_cluster
