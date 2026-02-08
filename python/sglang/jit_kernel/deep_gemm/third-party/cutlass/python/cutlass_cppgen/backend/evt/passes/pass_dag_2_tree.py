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
Merge non-tree sub-graphs of the DAG IR into a single DAG. The fused DAG will be implemented
by the topological visitor, while the rest of the graph will be implemented with the tree visitor.
"""

from copy import deepcopy

from cutlass_cppgen.backend.evt.ir import DAGIR, TopoVisitorNode
from cutlass_cppgen.backend.evt.passes.pass_get_impl import PassGetImpl
from cutlass_cppgen.backend.evt.passes.pass_manager import EVTPassBase
from cutlass_cppgen.backend.evt.passes.pass_shape_type_propagation import PassShapeTypePropagation


class PassDAG2Tree(EVTPassBase):
    """
    Convert the DAG IR to Tree by fusing subgraphs
    """
    dependencies = [
        PassShapeTypePropagation,
        PassGetImpl
    ]

    def call(self):
        # Step 1: find the nodes that have multiple parents
        multi_parent_nodes = []

        for node in self.dag_ir.nodes_topological_order():
            if self.dag_ir.out_degree(node) > 1:
                multi_parent_nodes.append(node)
        # Step 2: find the lowest common ancestor (LCA) of all its parents
        for node in multi_parent_nodes:
            # A multi-parent node could be already fused by the previous node
            if not self.dag_ir.has_node(node):
                continue
            # A node uncovered by the previous fusions can have out degree change
            # Case 1: it has <= 1 edges to the previously fused subgraph, no degree change
            # Case 2: it has more than one edges to the previously fused subgraph, degree drops
            if self.dag_ir.out_degree(node) <= 1:
                continue

            # Otherwise, the node still
            reachable_nodes = []
            # Complexity: O(Dout*N)
            for parent in self.dag_ir.get_users(node):
                reachable_nodes.append(set(self.dag_ir.all_reachable_nodes(parent)))
            # get the common reachable objects
            common_items = set.intersection(*reachable_nodes)
            node_to_fuse = set.union(*reachable_nodes).difference(common_items)

            lca = None
            # If common ancestor exists, find the lowest one
            if len(common_items) > 0:
                topo_order = self.dag_ir.nodes_topological_order()
                topo_idx = -1
                for item in common_items:
                    if lca is None:
                        lca = item
                        topo_idx = topo_order.index(item)
                    else:
                        if topo_idx > topo_order.index(item):
                            lca = item
                            topo_idx = topo_order.index(item)
            else:
                # there is no common ancestor for all the parents, we pack all the reachable
                # nodes into a single DAG node as a fallback. The lca should be the input node of
                # one of the output nodes with out_degree = 0
                potential_output_nodes = []
                for node in node_to_fuse:
                    if self.dag_ir.out_degree(node) == 0:
                        potential_output_nodes.append(node)
                if len(potential_output_nodes) == 0:
                    raise RuntimeError(f"No output node with out degree = 0 found.")
                
                output_node = None
                if (self.dag_ir.cc >= 90):
                    # For SM90+, the lca should be the input node of D
                    if (not self.dag_ir.has_node("D")):
                        raise RuntimeError(f"D is not a node in the DAG IR.")
                    output_node = "D"
                else:
                    output_node = potential_output_nodes[0]
                
                if (output_node is None):
                    raise RuntimeError(f"No output node found.")
                lca = self.dag_ir.get_all_inputs(output_node)[0]
                node_to_fuse.remove(output_node)

            # The lca is the output node of the DAG node
            # Get the nodes to be fused
            node_to_fuse.add(lca)
            # Get all the input nodes
            all_input_nodes = []
            all_output_nodes = []
            for node in node_to_fuse:
                all_input_nodes.append(set(self.dag_ir.get_all_inputs(node)))
                all_output_nodes.append(set(self.dag_ir.get_users(node)))
            all_input_nodes = set.union(*all_input_nodes)
            all_output_nodes = set.union(*all_output_nodes)

            new_subgraph_nodes = set.union(node_to_fuse, all_input_nodes, all_output_nodes)

            # Create the subgraph
            subgraph_ = self.dag_ir._graph.subgraph(new_subgraph_nodes)
            subgraph = DAGIR(self.dag_ir.cc)
            for node in subgraph_.nodes:
                meta = deepcopy(self.dag_ir.get_node_meta(node))
                if node not in node_to_fuse:
                    meta.disabled = True
                subgraph.add_node(meta)
            for edge in subgraph_.edges:
                subgraph.add_edge(edge[0], edge[1], self.dag_ir.get_edge_weight(edge[0], edge[1]))


            # Create the fused node
            dag_node = TopoVisitorNode(
                name=f"dag_{lca}", subgraph=subgraph,
                output_node=self.dag_ir.get_node_meta(lca))
            self.dag_ir.add_node(dag_node)

            # Add input edges
            for idx, node in enumerate(all_input_nodes):
                self.dag_ir.add_edge(node, dag_node.name, weight=idx)

            # Replace all uses with DAG node (only 1 output node)
            self.dag_ir.replace_all_uses_with(lca, dag_node.name)

            # Remove all fused nodes
            node_to_fuse.remove(lca)
            for node in node_to_fuse:
                self.dag_ir.remove_node(node)

    def ensures(self) -> None:
        # Ensure that after the pass, the resulting DAG becomes a tree
        for node in self.dag_ir.nodes:
            out_degree = self.dag_ir.out_degree(node)
            if out_degree > 1:
                raise RuntimeError(f"PassDAG2Tree failed. Node {node} still have outdegree = {out_degree}")
