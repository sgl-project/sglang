#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#include "structs.h"

namespace topk_select_bf16_cluster {

template<typename Config>
void run_topk_select_kernel(const TopkSelectArgs &args);

}   // topk_select_bf16_cluster
