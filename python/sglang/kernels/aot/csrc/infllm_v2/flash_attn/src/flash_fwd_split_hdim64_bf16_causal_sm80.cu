// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template void
run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 64, true>(Flash_fwd_params& params, cudaStream_t stream);
