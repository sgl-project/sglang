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


class DivFuse:
    def pattern(x):
        y = 1.0 / x
        z = 1.0 / y
        return z

    def replacement(x):
        return x


class EraseCopy:
    def __call__(self, graph_module: torch.fx.GraphModule):
        copy_node = None
        prepare_weight_cache_default_node = None

        results = []
        for module in graph_module.modules():
            for node in list(module.graph.nodes):
                if node.type == torch.nn.parameter.Parameter:
                    continue
                if node.target == "copy_":
                    copy_node = node
                    prepare_weight_cache_default_node = None
                    continue

                if (
                    copy_node
                    and node.target == torch.ops.sglang.prepare_weight_cache.default
                ):
                    prepare_weight_cache_default_node = node
                    continue

                if copy_node and node.target == torch.ops.npu.npu_add_rms_norm_quant:
                    arg = copy_node.args[1]

                    if prepare_weight_cache_default_node is not None:
                        prepare_weight_cache_default_node.args = (
                            arg,
                            prepare_weight_cache_default_node.args[1],
                        )

                    node.args = (
                        node.args[0],
                        arg,
                        node.args[2],
                        node.args[3],
                        node.args[4],
                    )

                    module.graph.erase_node(copy_node)

                    result = (
                        arg,
                        copy_node,
                        prepare_weight_cache_default_node,
                    )
                    results.append(result)

                    copy_node = None
                    prepare_weight_cache_default_node = None

        return results


class NpuAddRmsNormQuantFuse:
    def pattern(rms_norm_input, residual, rms_norm_weight, scale, offset, v1, v2, v3):
        output = torch.ops.npu.npu_add_rms_norm(
            rms_norm_input, residual, rms_norm_weight, 1e-6
        )
        out0 = output[0]
        out2 = output[2]
        quantized_output = torch.ops.npu.npu_quantize(out0, scale, offset, v1, v2, v3)
        return quantized_output, out2

    def replacement(
        rms_norm_input, residual, rms_norm_weight, scale, offset, v1, v2, v3
    ):
        output = torch.ops.npu.npu_add_rms_norm_quant(
            rms_norm_input, residual, rms_norm_weight, 1.0 / scale, offset, epsilon=1e-6
        )
        quantized_output = output[0]
        out2 = output[2]
        return quantized_output, out2


class NpuAddRmsNormDynamicQuantFuse:
    def pattern(rms_norm_input, residual, rms_norm_weight):
        output = torch.ops.npu.npu_add_rms_norm(
            rms_norm_input, residual, rms_norm_weight, 1e-6
        )
        out0 = output[0]
        out2 = output[2]
        quantized_output, dynamic_scale = torch.ops.npu.npu_dynamic_quant(out0)
        return quantized_output, out2, dynamic_scale

    def replacement(rms_norm_input, residual, rms_norm_weight):
        output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
            x1=rms_norm_input,
            x2=residual,
            gamma=rms_norm_weight,
            epsilon=1e-6,
            output_mask=[True, True],
        )
        quantized_output = output[0]
        out2 = output[2]
        dynamic_scale = output[3]
        return quantized_output, out2, dynamic_scale
