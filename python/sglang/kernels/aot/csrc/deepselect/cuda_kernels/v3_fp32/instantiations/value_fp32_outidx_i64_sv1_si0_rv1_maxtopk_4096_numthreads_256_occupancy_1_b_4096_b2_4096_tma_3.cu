#include "../topk_select.cuh"

namespace topk_select_fp32 {

template
void run_topk_select_kernel<
    TopkSelectConfig<float, int64_t, true, false, true, 4096, 256, 1, 4096, 4096, 3, 512, 1>
>(const TopkSelectArgs &args);

}   // topk_select_fp32
