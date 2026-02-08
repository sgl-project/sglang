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
Preprocess the reduction nodes.

The parser treats reduction as Compute(op=(reg_reduce_fn, gmem_reduce_fn)) - Store()
This pass fuses these into a single store node, and then replaces all uses of the
current node with the new store node.
"""

from cutlass_cppgen.backend.evt.ir import ComputeNode, StoreNode
from cutlass_cppgen.backend.evt.passes.pass_manager import EVTPassBase


class PassPreprocessRed(EVTPassBase):
    """
    Preprocess red nodes
    """

    def call(self):
        # Step 1: find the compute nodes with op=red
        red_compute_nodes = []
        for node_meta in self.dag_ir.nodes_meta:
            if isinstance(node_meta, ComputeNode):
                if type(node_meta.fn) == tuple:
                    # To keep the frontend simple, the reduction nodes
                    # are parsed into compute nodes by default
                    # The simple heuristic to distinguish between compute
                    # and reduction node is that compute node is a single function,
                    # while the reduction node is a tuple of functions for
                    # in-register reduction and atomic global memory reduction
                    red_compute_nodes.append(node_meta.name)

        # Step 2: for each compute, merge it with the succeeding store
        for node in red_compute_nodes:
            # Verify
            users = self.dag_ir.get_users(node)
            inputs = self.dag_ir.get_all_inputs(node)
            # Has a single user
            assert len(users) == 1
            assert len(inputs) == 1
            user = users[0]
            input = inputs[0]

            user_meta = self.dag_ir.get_node_meta(user)
            # Must be a store node
            assert isinstance(user_meta, StoreNode)
            # With output degree == 0
            assert self.dag_ir.out_degree(user) == 0
            # Register the reduce op
            node_meta = self.dag_ir.get_node_meta(node)
            user_meta.reg_reduce_fn, user_meta.gmem_reduce_fn = node_meta.fn
            user_meta.element_compute = node_meta.element_compute
            user_meta.round_style = node_meta.round_style

            # Replace all uses
            self.dag_ir.remove_edge(input, node)
            input_users = self.dag_ir.get_users(input)
            for iu in input_users:
                weight = self.dag_ir.get_edge_weight(input, iu)
                self.dag_ir.add_edge(user, iu, weight)
                self.dag_ir.remove_edge(input, iu)
            self.dag_ir.add_edge(input, user)
            self.dag_ir.remove_node(node)

            # Register the reduction name
            self.dag_ir.reduction_names.append(user)
