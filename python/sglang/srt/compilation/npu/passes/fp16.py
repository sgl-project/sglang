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

import sgl_kernel_npu
import torch

import sglang.srt.compilation.npu.custom_ops  # noqa


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

        split_qkv_rmsnorm_rope_default = (
            sgl_kernel_npu.norm.split_qkv_rmsnorm_rope.default(
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
        )

        q = split_qkv_rmsnorm_rope_default[0]
        k = split_qkv_rmsnorm_rope_default[1]
        v = split_qkv_rmsnorm_rope_default[2]

        return v, q, k


class SplitQkvRmsnormRopeFuseMoe:
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

        SplitQkvRmsnormRopeFuseMoe.instance = self

    def pattern(
        output_parallel,
        q_norm_parameters_weight,
        k_norm_parameters_weight,
        positions,
        cos_sin_cache,
    ):
        # pattern matching brokes if make static method as class method
        self = SplitQkvRmsnormRopeFuseMoe.instance

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

        return v, query_out, key_out, q, k

    def replacement(
        output_parallel,
        q_norm_parameters_weight,
        k_norm_parameters_weight,
        positions,
        cos_sin_cache,
    ):
        # pattern matching brokes if make static method as class method
        self = SplitQkvRmsnormRopeFuseMoe.instance

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

        split_qkv_rmsnorm_rope_default = (
            torch.ops.sglang.split_qkv_rmsnorm_rope.default(
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
        )

        q = split_qkv_rmsnorm_rope_default[0]
        k = split_qkv_rmsnorm_rope_default[1]
        v = split_qkv_rmsnorm_rope_default[2]

        return v, q, k, cos_contiguous, sin_contiguous
