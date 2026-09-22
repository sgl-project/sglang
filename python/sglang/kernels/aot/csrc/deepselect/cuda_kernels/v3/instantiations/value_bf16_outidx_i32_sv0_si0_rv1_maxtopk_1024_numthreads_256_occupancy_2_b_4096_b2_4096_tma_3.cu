#include "../topk_select.cuh"

namespace topk_select_bf16_normal {

template
void run_topk_select_kernel<
    TopkSelectConfig<nv_bfloat16, int32_t, false, false, true, 1024, 256, 2, 4096, 4096, 3, 512, 1>
>(const TopkSelectArgs &args);

}   // topk_select_bf16_normal
