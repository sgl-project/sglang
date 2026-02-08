#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Base class for Epilogue Visitor Emitter
"""

from cutlass_library import DataTypeTag
from cutlass_cppgen.backend.evt.ir import TopoVisitorNode, DAGIR


class FusionCallbacks:
    def __init__(self, dag_ir: DAGIR, cc: int, emit_CD=True) -> None:
        """
        Emit the EVT fusion callbacks
        :param dag_ir: the DAG IR holding the epilogue visitor
        :param cc: compute capability
        :param emit_CD: whether to emit nodes C & D as a part of the fusion callbacks
                        For Sm90, set emit_CD=False, as Tensor C & D are hardcoded in the collective API
                        so that their shared memory can be explicitly reused
                        For Sm89, set emit_CD=True as they are treated as normal AuxLoad & AuxStore nodes.
        """
        self.dag_ir = dag_ir
        self.emit_CD = emit_CD
        self.cc = cc
        self.evt_cc = 90 if cc >= 90 else cc
        if self.cc < 90:
            self.namespace = "threadblock"
        else:
            self.namespace = "fusion"

    #
    # Helper functions
    #

    def get_visitor_name(self, node: str):
        """
        Get the visitor name
        """
        meta = self.dag_ir.get_node_meta(node)
        if not isinstance(meta, TopoVisitorNode) and self.dag_ir.in_degree(node) > 0:
            return f"EVT{meta.name_camel}"
        else:
            return meta.name_camel

    def emit(self):
        node_metas = self.dag_ir.node_metas_topological_order()
        epilogue_str = ""
        # Step 1: emit individual node type decl
        #         emit the EVT & DAG connector
        for meta in node_metas:
            if not meta.disabled:
                epilogue_str += self.emit_node(meta)
            if not self.emit_CD and meta.name == "D":
                continue
            if isinstance(meta, TopoVisitorNode):
                epilogue_str += self.emit_dag(meta)
            else:
                epilogue_str += self.emit_evt(meta)

        # Step 2: post-processing & get callback name
        if not self.emit_CD:
            if not self.dag_ir.has_node("C"):
                epilogue_str += "using ElementC = void;\nusing StrideC = StrideD;\n"
            output_node = self.dag_ir.get_all_inputs("D")[0]
            # The callback is the src of node D
            callback_name = self.get_visitor_name(output_node)
        else:
            # The callback is the last node in the topological order
            callback_name = self.get_visitor_name(node_metas[-1].name)
        return epilogue_str, callback_name

    def emit_evt(self, node):
        if self.dag_ir.in_degree(node.name) == 0:
            return ""

        evt_tmp = f"""
using EVT{node.name_camel} = cutlass::epilogue::{self.namespace}::Sm{self.evt_cc}EVT<
    {node.name_camel},
"""
        sorted_children = self.dag_ir.get_all_inputs(node.name)
        evt_node_strs = [f"    {self.get_visitor_name(child_name)}" for child_name in sorted_children]
        evt_tmp += ",\n".join(evt_node_strs) + ">;\n"

        return evt_tmp

    def emit_dag(self, node):
        subgraph = node.subgraph
        subgraph_nodes = subgraph.nodes_topological_order()
        # Emit the Edge Tuple
        edge_tuples = "cute::tuple<\n"
        for n in subgraph_nodes[:-1]:
            in_edges = subgraph.in_edges(n)
            edge_weights = [subgraph.get_edge_weight(edge[0], edge[1]) for edge in in_edges]
            sorted_children = [edge[0] for _, edge in sorted(zip(edge_weights, in_edges))]
            edge_tuple = "        cute::seq<"
            edge_str = [str(subgraph_nodes.index(child)) for child in sorted_children]
            edge_tuple += ", ".join(edge_str) + ">,\n"

            edge_tuples += edge_tuple
        edge_tuples += "    >"

        # Emit the node list
        dag_nodes = ""
        dag_node_strs = []
        for n in subgraph_nodes[:-1]:
            n_meta = subgraph.get_node_meta(n)
            if n_meta.disabled:
                dag_node_strs.append(f"    {self.get_visitor_name(n)}")
            else:
                dag_node_strs.append(f"    {n_meta.name_camel}")
        dag_nodes = ",\n".join(dag_node_strs)

        return f"""
using {node.name_camel} = cutlass::epilogue::{self.namespace}::Sm{self.evt_cc}TopologicalVisitor<
    {DataTypeTag[node.subgraph.element_compute]},
    {edge_tuples},
{dag_nodes}
>;
"""

    def emit_node(self, node):
        if isinstance(node, TopoVisitorNode):
            emission = ""
            for node in node.subgraph.node_metas_topological_order():
                if not node.disabled:
                    emission += self.emit_node(node)
            return emission
        else:
            return node.underlying_impl.type_decl
