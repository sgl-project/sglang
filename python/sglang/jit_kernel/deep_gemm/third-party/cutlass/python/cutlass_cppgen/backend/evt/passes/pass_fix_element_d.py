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
Fix the element_output of producer of D.

In Sm90 epilogue visitor, the node writing D to gmem does not have internal
element converter, so the compute node producing D must have element_output = type(D).
"""

from cutlass_cppgen.backend.evt.passes.pass_layout_elimination import PassLayoutManipulateElimination
from cutlass_cppgen.backend.evt.passes.pass_manager import EVTPassBase


class PassFixElementD(EVTPassBase):
    """
    In Sm90 epilogue visitor, the node writing D to gmem does not have internal
    element converter, so the compute node producing D must have
    element_output = type(D)
    """
    dependencies = [
        PassLayoutManipulateElimination
    ]
    def get_producer(self, node, element_D):
        node_meta = self.dag_ir.get_node_meta(node)
        if node_meta.op == "compute":
            node_meta.element_output = element_D
        elif node_meta.op == "store":
            self.get_producer(self.dag_ir.get_all_inputs(node)[0], element_D)

    def call(self):
        if self.dag_ir.has_node("D"):
            node_d_meta = self.dag_ir.get_node_meta("D")
            element_D = node_d_meta.store_tensor.element
            self.get_producer("D", element_D)
