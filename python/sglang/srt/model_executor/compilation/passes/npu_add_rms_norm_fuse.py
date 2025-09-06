# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch


class NpuAddRmsNormFuse:
    def pattern(rms_norm_input, residual, rms_norm_weight, scale, offset, v1, v2, v3):
        output = torch.ops.npu.npu_add_rms_norm(
            rms_norm_input,
            residual,
            rms_norm_weight,
            1e-6)
        out0 = output[0]
        out2 = output[2]
        quantized_output = torch.ops.npu.npu_quantize(
            out0,
            scale,
            offset,
            v1,
            v2,
            v3)
        return quantized_output, out2

    def replacement(rms_norm_input, residual, rms_norm_weight, scale, offset, v1, v2, v3):
        output = torch.ops.npu.npu_add_rms_norm_quant(
            rms_norm_input,
            residual,
            rms_norm_weight,
            1. / scale,
            offset,
            epsilon=1e-6)
        quantized_output = output[0]
        out2 = output[2]
        return quantized_output, out2
