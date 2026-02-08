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
DAG IR used by Python EVT
"""

import networkx as nx

from cutlass_library import DataType

from cutlass_cppgen.backend.evt.ir.compute_nodes import ComputeNode
from cutlass_cppgen.backend.evt.ir.node import NodeBase
from cutlass_cppgen.backend.library import ActivationOp
from cutlass_cppgen.backend.utils import device_cc


class DAGIR:
    """
    ``DAGIR`` is the main data structure used in the EVT Intermediate Representation.
    It consists of a series of ``Node`` s, each representing epilogue visitor nodes.

    In the DAGIR, ``node`` is an string of its name. ``node_meta`` is the underlying class of the node
    """
    def __init__(self, cc, element_compute=DataType.f32) -> None:
        # The EVT DAGIR is managed through the nextworkX Digraph class
        self._graph = nx.DiGraph()

        self.element_compute = element_compute

        self.reduction_names = []

        self.cc = cc

        self.identity_counter = 0

    #
    # IR manipulator
    #

    def add_node(self, meta: NodeBase):
        """
        Add a node to dag ir
        """
        if self.has_node(meta.name):
            raise SyntaxError(f"Variable '{meta.name}' cannot be defined twice.")
        self._graph.add_node(meta.name, meta=meta)

    def add_edge(self, src: str, dst: str, weight: int=0):
        """
        Add an edge src -> dst to dag ir with weight
        """
        if not self.has_node(src):
            raise SyntaxError(f"Variable '{src}' is undefined.")
        if not self.has_node(dst):
            raise SyntaxError(f"Variable '{dst}' is undefined.")

        if self._graph.has_edge(src, dst):
            # The DiGraph doesn't support multiple edges between two nodes
            # We insert an identity node in such case as a workaround
            identity_name = f"autogen_identity_{self.identity_counter}"
            self.identity_counter += 1
            compute_node = ComputeNode(
                name=identity_name, fn=ActivationOp.Identity,
                element_output=self.element_compute,
                element_compute=self.element_compute)
            self.add_node(compute_node)
            self.add_edge(src, identity_name, 0)
            self.add_edge(identity_name, dst, weight)
        else:
            self._graph.add_edge(src, dst, weight=weight)

    def remove_node(self, node: str):
        """
        Remove node from dag ir
        """
        self._graph.remove_node(node)

    def remove_edge(self, src: str, dst: str):
        """
        Remove edge src -> dst
        """
        self._graph.remove_edge(src, dst)

    #
    # Helper functions for getting attrs
    #

    def has_node(self, node: str) -> bool:
        """
        Check if the node is in the graph
        """
        return self._graph.has_node(node)

    def in_degree(self, node: str):
        """
        Get the input degree of node
        """
        return self._graph.in_degree(node)

    def in_edges(self, node: str):
        """
        Get the input edges of node
        """
        return [edge for edge in self._graph.in_edges(node)]

    def out_degree(self, node: str):
        """
        Get the output degree of node
        """
        return self._graph.out_degree(node)

    def out_edges(self, node: str):
        """
        Get the output edges of node
        """
        return [edge for edge in self._graph.out_edges(node)]

    def get_node_meta(self, node: str):
        """
        Get the meta data of the node
        """
        return self._graph.nodes[node]["meta"]

    def get_edge_weight(self, src, dst):
        """
        Get the edge weight of edge src->dst
        """
        return self._graph.get_edge_data(src, dst)["weight"]

    #
    # High-level helper functions
    #

    def all_reachable_nodes(self, node: str):
        """
        Get all the nodes reachable from the current node (exclude)
        """
        return list(nx.dfs_preorder_nodes(self._graph, source=node))

    def get_users(self, node: str):
        """
        Get all users of the current node
        """
        return [edge[1] for edge in self.out_edges(node)]

    def get_all_inputs(self, node: str):
        """
        Get all the input nodes sorted by edge weight
        """
        in_edges = self.in_edges(node)
        edge_weights = [self.get_edge_weight(*edge) for edge in in_edges]
        return [edge[0] for _, edge in sorted(zip(edge_weights, in_edges))]

    def get_all_inputs_meta(self, node: str):
        """
        Get all the input node metas sorted by edge weight
        """
        return [self.get_node_meta(input_node) for input_node in self.get_all_inputs(node)]

    def replace_all_uses_with(self, node1, node2):
        """
        Replace all uses of node1 with node2
        """
        for edge in self.out_edges(node1):
            weight = self.get_edge_weight(*edge)
            user = edge[1]
            self.add_edge(node2, user, weight)
            self.remove_edge(node1, user)
        self.remove_node(node1)

    #
    # Node accessor
    #
    def nodes_topological_order(self):
        """
        Get the nodes in the unique lexicographical topological order
        It generates a unique ordering of nodes by first sorting topologically
        and then additionally by sorting lexicographically.

        Although topological_sort alone also works, this generates a unique key
        for each epilogue visitor pattern and ensures the compilation cache can be reused.
        :return: list[str]
        """
        return list(nx.lexicographical_topological_sort(self._graph))

    def node_metas_topological_order(self):
        """
        Get the node metas in topological order
        :return: list[NodeBase]
        """
        return [self.get_node_meta(node) for node in self.nodes_topological_order()]

    @property
    def nodes(self):
        """
        Get all nodes
        :return: list[str]
        """
        return list(self._graph.nodes)

    @property
    def nodes_meta(self):
        """
        Get all node metas
        :return: list[NodeBase]
        """
        return [data[1]['meta'] for data in self._graph.nodes.data()]

    @property
    def edges(self):
        """
        Get all edges
        :return: list[(str, str)]
        """
        return list(self._graph.edges)

    #
    # Path
    #
    def has_path(self, src: str, target: str) -> bool:
        """
        Return True is a path exists from src to target
        """
        return nx.has_path(self._graph, src, target)
