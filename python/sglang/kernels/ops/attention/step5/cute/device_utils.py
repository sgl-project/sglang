# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import cutlass
import torch

torch2cute_dtype_map = {
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
}
