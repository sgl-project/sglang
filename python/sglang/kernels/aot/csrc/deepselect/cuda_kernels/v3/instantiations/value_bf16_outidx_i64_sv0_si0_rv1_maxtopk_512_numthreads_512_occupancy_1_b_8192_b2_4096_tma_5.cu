#include "../topk_select.cuh"

namespace topk_select_bf16_normal {

template
void run_topk_select_kernel<
    TopkSelectConfig<nv_bfloat16, int64_t, false, false, true, 512, 512, 1, 8192, 4096, 5, 512, 1>
>(const TopkSelectArgs &args);

}   // topk_select_bf16_normal
