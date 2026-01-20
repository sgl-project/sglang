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

from sglang.srt.hardware_backend.npu.graph_runner.compilation.custom_ops import (
    split_qkv_rmsnorm_rope,
)


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

                node_target_str = str(node.target)

                if node_target_str == "copy_":
                    copy_node = node
                    prepare_weight_cache_default_node = None
                    continue

                if copy_node and node_target_str == "sglang.prepare_weight_cache":
                    prepare_weight_cache_default_node = node
                    continue

                if copy_node and node_target_str == "npu.npu_add_rms_norm_quant":
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


class SplitQkvRmsnormRopeFuse:
    instance = None

    def __init__(
        self,
        q_size: int,
        kv_size: int,
        head_dim: int,
        q_shape,
        k_shape,
        variance_epsilon: float,
    ):
        self.q_size = q_size
        self.kv_size = kv_size
        self.head_dim = head_dim
        self.q_shape = q_shape
        self.k_shape = k_shape
        self.variance_epsilon = variance_epsilon

        SplitQkvRmsnormRopeFuse.instance = self

    def pattern(
        output_parallel,
        q_norm_parameters_weight,
        k_norm_parameters_weight,
        positions,
        cos_sin_cache,
    ):
        # pattern matching brokes if make static method as class method
        self = SplitQkvRmsnormRopeFuse.instance

        split = output_parallel.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = split[0]
        k = split[1]
        v = split[2]

        q_by_head = q.reshape(-1, self.head_dim)
        npu_rms_norm_q = torch.ops.npu.npu_rms_norm(
            q_by_head, q_norm_parameters_weight, self.variance_epsilon
        )
        q_by_head_1 = npu_rms_norm_q[0]

        k_by_head = k.reshape(-1, self.head_dim)
        npu_rms_norm_k = torch.ops.npu.npu_rms_norm(
            k_by_head, k_norm_parameters_weight, self.variance_epsilon
        )
        k_by_head_1 = npu_rms_norm_k[0]

        q_1 = q_by_head_1.view(self.q_shape)
        k_1 = k_by_head_1.view(self.k_shape)

        npu_mrope = torch.ops.npu.npu_mrope(
            positions,
            q_1,
            k_1,
            cos_sin_cache,
            self.head_dim,
            mrope_section=[0, 0, 0],
            rotary_mode="half",
        )
        query_out = npu_mrope[0]
        key_out = npu_mrope[1]

        return v, query_out, key_out

    def replacement(
        output_parallel,
        q_norm_parameters_weight,
        k_norm_parameters_weight,
        positions,
        cos_sin_cache,
    ):
        # pattern matching brokes if make static method as class method
        self = SplitQkvRmsnormRopeFuse.instance

        flatten = positions.flatten()
        cos_sin = cos_sin_cache.index_select(0, flatten)

        reshape = cos_sin.reshape(-1, 2, 64)
        repeat = reshape.repeat(1, 1, 2)
        chunk = repeat.chunk(2, dim=-2)
        cos = chunk[0]
        sin = chunk[1]

        cos_view = cos.view(-1, 1, 1, self.head_dim)
        cos_contiguous = cos_view.contiguous()

        sin_view = sin.view(-1, 1, 1, self.head_dim)
        sin_contiguous = sin_view.contiguous()

        split_qkv_rmsnorm_rope_default = split_qkv_rmsnorm_rope(
            output_parallel,
            sin_contiguous,
            cos_contiguous,
            q_norm_parameters_weight,
            k_norm_parameters_weight,
            self.q_size,
            self.kv_size,
            self.head_dim,
            self.variance_epsilon,
            q_bias=None,
            k_bias=None,
        )

        q = split_qkv_rmsnorm_rope_default[0]
        k = split_qkv_rmsnorm_rope_default[1]
        v = split_qkv_rmsnorm_rope_default[2]

        return v, q, k
