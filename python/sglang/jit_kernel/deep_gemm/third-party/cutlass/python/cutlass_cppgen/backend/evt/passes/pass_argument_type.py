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
Construct the epilogue visitor argument type
"""

from cutlass_cppgen.backend.c_types import visitor_factory
from cutlass_cppgen.backend.evt.ir import TopoVisitorNode
from cutlass_cppgen.backend.evt.passes.pass_dag_2_tree import PassDAG2Tree
from cutlass_cppgen.backend.evt.passes.pass_get_impl import PassGetImpl
from cutlass_cppgen.backend.evt.passes.pass_manager import EVTPassBase
from cutlass_cppgen.backend.evt.passes.pass_shape_type_propagation import PassShapeTypePropagation
from cutlass_cppgen.backend.evt.passes.util import cc_map


class PassGetArgumentType(EVTPassBase):
    """
    Construct the epilogue visitor argument type
    """
    dependencies = [
        PassShapeTypePropagation,     # The Layout of all nodes must be set
        PassDAG2Tree,                 # The type of each node must be set
        PassGetImpl                   # The DAG subgraphs must be set
    ]

    def requires(self) -> None:
        # Check "D" is in the node list
        if cc_map[self.cc] in [90, 100] and (not self.dag_ir.has_node("D")):
            raise SyntaxError(
                "Sm90+ EVT requires the epilogue to have a returned tensor D, "
                "but the variable 'D' is not found in the return values.")

    def call(self):
        nodes = self.dag_ir.nodes_topological_order()
        self.argument_types = {}
        for node in nodes:
            meta = self.dag_ir.get_node_meta(node)
            if not meta.disabled:
                self.argument_types[node] = meta.underlying_impl.argument_type
            if node == "D" and cc_map[self.cc] in [90, 100]:
                continue
            if isinstance(meta, TopoVisitorNode):
                self.get_dag_argument_type(node)
            else:
                self.get_evt_argument_type(node)

        self.cc_specific_method(self.set_argument_type)()

    def get_evt_argument_type(self, node):
        # Sort the input nodes by edge weight
        input_types = [self.argument_types[child] for child in self.dag_ir.get_all_inputs(node)]
        if len(input_types) > 0:
            self.argument_types[node] = visitor_factory(
                input_types + [self.argument_types[node],], self.dag_ir.get_all_inputs(node) + [node,])

    def get_dag_argument_type(self, node):
        meta = self.dag_ir.get_node_meta(node)
        subgraph = meta.subgraph
        subgraph_nodes = subgraph.nodes_topological_order()
        # Visit the unvisited nodes in subgraph
        for n in subgraph_nodes:
            m = subgraph.get_node_meta(n)
            if m.disabled:
                continue
            else:
                self.argument_types[n] = m.underlying_impl.argument_type
        input_types = [self.argument_types[child] for child in subgraph_nodes[:-1]]
        if len(input_types) > 0:
            self.argument_types[node] = visitor_factory(input_types, subgraph_nodes[:-1])

    def set_argument_type(self):
        pass

    def sm90_set_argument_type(self):
        self.dag_ir.epilogue_thread_type = self.argument_types[self.dag_ir.get_all_inputs("D")[0]]
        # Get the tensorD argument type
        self.dag_ir.arg_d_type = self.dag_ir.get_node_meta("D").underlying_impl.argument_type_d

        # Get the tensorC argument type
        if self.dag_ir.has_node("C"):
            self.dag_ir.arg_c_type = self.dag_ir.get_node_meta("C").underlying_impl.argument_type_c
        else:
            self.dag_ir.arg_c_type = self.dag_ir.arg_d_type

    def sm100_set_argument_type(self):
        self.sm90_set_argument_type()

    def sm80_set_argument_type(self):
        nodes = self.dag_ir.nodes_topological_order()
        self.dag_ir.epilogue_thread_type = self.argument_types[nodes[-1]]
