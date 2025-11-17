# Copyright 2023-2025 SGLang Team
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

import dataclasses
import importlib
import logging
from typing import Any, Callable

import torch

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.npu.compilation_context import CompilationContext
from sglang.srt.compilation.npu.npu_graph_compiler_backend import (
    NpuGraphCompilerBackend,
)
from sglang.srt.distributed import get_tensor_model_parallel_world_size

logger = logging.getLogger(__name__)


class Submodule(torch.nn.Module):
    block_tables = None

    def __init__(self, page_size, model_config):
        self.page_size = page_size
        self.config = model_config

        tp_size = get_tensor_model_parallel_world_size()
        assert self.config.num_attention_heads % tp_size == 0
        self.num_heads = self.config.num_attention_heads // tp_size

        self.total_num_kv_heads = self.config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.hidden_size = self.config.hidden_size
        self.head_dim = getattr(
            self.config, "head_dim", self.hidden_size // self.config.num_attention_heads
        )

        self.scaling = self.head_dim**-0.5

    def forward_with_calculation(
        self,
        l_args_2_req_to_token_pool_req_to_token,
        l_args_2_req_pool_indices,
        l_args_2_seq_lens,
        query_2,
        l_args_2_token_to_kv_pool_k_buffer_0_,
        l_args_2_token_to_kv_pool_v_buffer_0_,
        l_args_2_attn_backend_forward_metadata_block_tables,
        l_args_2_attn_backend_forward_metadata_seq_lens_cpu_int,
        output,
    ):
        Submodule.block_tables = (
            l_args_2_req_to_token_pool_req_to_token[
                l_args_2_req_pool_indices, : l_args_2_seq_lens.max()
            ][:, :: self.page_size]
            // self.page_size
        )
        _npu_paged_attention = torch.ops.atb._npu_paged_attention(
            query=query_2,
            key_cache=l_args_2_token_to_kv_pool_k_buffer_0_,
            value_cache=l_args_2_token_to_kv_pool_v_buffer_0_,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale_value=self.scaling,
            block_table=Submodule.block_tables,
            context_lens=l_args_2_attn_backend_forward_metadata_seq_lens_cpu_int,
            out=output,
        )

    def forward(
        self,
        l_args_2_req_to_token_pool_req_to_token,
        l_args_2_req_pool_indices,
        l_args_2_seq_lens,
        query_2,
        l_args_2_token_to_kv_pool_k_buffer_0_,
        l_args_2_token_to_kv_pool_v_buffer_0_,
        l_args_2_attn_backend_forward_metadata_block_tables,
        l_args_2_attn_backend_forward_metadata_seq_lens_cpu_int,
        output,
    ):
        _npu_paged_attention = torch.ops.atb._npu_paged_attention(
            query=query_2,
            key_cache=l_args_2_token_to_kv_pool_k_buffer_0_,
            value_cache=l_args_2_token_to_kv_pool_v_buffer_0_,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale_value=self.scaling,
            block_table=Submodule.block_tables,
            context_lens=l_args_2_attn_backend_forward_metadata_seq_lens_cpu_int,
            out=output,
        )


def resolve_obj_by_qualname(qualname: str) -> Any:
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_compiled_only: bool
    graph: torch.fx.GraphModule


class NpuAddRmsNormFuse:
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


class PiecewiseNpuGraphCompilerBackend(NpuGraphCompilerBackend):
    graph: torch.fx.GraphModule

    def __init__(
        self,
        model_runner,
        compilation_config: CompilationConfig,
        compilation_context: CompilationContext,
        page_size: int,
    ):
        super().__init__(model_runner.model_config.dtype)

        self.model_runner = model_runner
        self.model_config = model_runner.model.config

        self.compilation_config = compilation_config
        self.page_size = page_size
        self.compilation_context = compilation_context

        self.split_gm = None
        self.piecewise_graphs = None

        self.callables = {}
        self.callables_by_branch = {}

    def __call__(self, graph: torch.fx.GraphModule, example_inputs) -> Callable:
        example_inputs_len = len(example_inputs)
        if example_inputs_len in self.callables:
            callable = self.callables[example_inputs_len]
            return callable

        super().__call__(graph, example_inputs)

        self.graph = graph
        self.split_gm, self.piecewise_graphs = (
            PiecewiseNpuGraphCompilerBackend.split_graph(
                self.graph, self.compilation_config.splitting_ops
            )
        )

        npu_graph_backend = resolve_obj_by_qualname(
            "sglang.srt.model_executor.compilation.npu_graph_backend.NPUGraphBackend"
        )

        self.submod_names_compiled_only = [
            item.submod_name for item in self.piecewise_graphs if item.is_compiled_only
        ]

        named_modules = self.split_gm.named_modules()
        submod = Submodule(self.page_size, self.model_config)
        use_forward = False
        for name, graph_module in named_modules:
            if not name:
                continue

            graph = getattr(self.split_gm, name)
            if name in self.submod_names_compiled_only:
                if use_forward:
                    self.split_gm.__dict__[name] = submod.forward
                else:
                    self.split_gm.__dict__[name] = submod.forward_with_calculation
                use_forward = True
            else:
                self.split_gm.__dict__[name] = npu_graph_backend(
                    self.model_runner, graph, self.compilation_context
                )

        self.split_gm(*example_inputs)
        self.callables[example_inputs_len] = self.split_gm.forward
        return self.split_gm.forward

    def split_graph(
        graph: torch.fx.GraphModule, ops: list[str]
    ) -> tuple[torch.fx.GraphModule, list[SplitItem]]:
        subgraph_id = 0
        node_to_subgraph_id = {}
        graphs_for_compilation = []

        node_index = 0
        node_index_max = len(graph.graph.nodes)

        nodes = list(graph.graph.nodes)

        counter = 1
        ops_count = 3
        ops_step = ops_count + 1
        while node_index < node_index_max:
            if (
                (node_index + ops_count) < node_index_max
                and nodes[node_index + ops_count].op == "call_function"
                and str(nodes[node_index + ops_count].target) in ops
            ):
                subgraph_id += 1
                graphs_for_compilation.append(subgraph_id)

                for submodule_node_index in range(node_index, node_index + ops_step):
                    submodule_node = nodes[submodule_node_index]
                    node_to_subgraph_id[submodule_node] = subgraph_id
                    counter = counter + 1
                node_index += ops_step

                subgraph_id += 1
            else:
                node = nodes[node_index]
                if node.op in ("output", "placeholder"):
                    node_index += 1
                elif node.op == "call_function" and str(node.target) in ops:
                    subgraph_id += 1
                    graphs_for_compilation.append(subgraph_id)

                    node_to_subgraph_id[node] = subgraph_id
                    node_index += 1

                    subgraph_id += 1
                else:
                    node_to_subgraph_id[node] = subgraph_id
                    node_index += 1
            counter += 1

        split_gm = torch.fx.passes.split_module.split_module(
            graph,
            None,
            lambda node: node_to_subgraph_id[node],
            keep_original_order=True,
        )

        names = [name for (name, module) in split_gm.named_modules()]

        outputs = []
        for name in names:
            if "." in name or name == "":
                # recursive child module or the root module
                continue

            module = getattr(split_gm, name)

            graph_id = int(name.replace("submod_", ""))
            outputs.append(
                SplitItem(name, graph_id, (graph_id in graphs_for_compilation), module)
            )

        # sort by intetger graph_id, rather than string name
        outputs.sort(key=lambda x: x.graph_id)

        return split_gm, outputs
