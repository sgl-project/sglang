#include "../topk_select.cuh"

namespace topk_select_fp32 {

template
void run_topk_select_kernel<
    TopkSelectConfig<float, int32_t, false, false, false, 512, 512, 1, 8192, 4096, 3, 512, 1>
>(const TopkSelectArgs &args);

}   // topk_select_fp32
