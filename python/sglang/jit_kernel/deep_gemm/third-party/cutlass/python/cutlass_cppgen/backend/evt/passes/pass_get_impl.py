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
Infer the underlying implement of each node.

While the frontend only distinguish between Load/Store/Compute Node,
each of these nodes can have different underlying implementation based
on their layout. For instance, a LoadNode can be AuxLoad, Row/Col/Scalar broadcast, etc.
This pass infers the underlying impl of each node
"""

import cutlass_cppgen.backend.evt.backend as evt_backend
from cutlass_cppgen.backend.evt.ir import DAGIR, LoadNode
from cutlass_cppgen.backend.evt.passes.pass_fix_element_d import PassFixElementD
from cutlass_cppgen.backend.evt.passes.pass_manager import EVTPassBase
from cutlass_cppgen.backend.evt.passes.pass_no_op_elimination import PassNoOpElimination
from cutlass_cppgen.backend.evt.passes.pass_shape_type_propagation import PassShapeTypePropagation
from cutlass_cppgen.backend.evt.passes.util import cc_map


class PassGetImpl(EVTPassBase):
    """
    While the frontend only distinguish between Load/Store/Compute Node,
    each of these nodes can have different underlying implementation based
    on their layout. For instance, a LoadNode can be AuxLoad, Row/Col/Scalar broadcast, etc.
    This pass infers the underlying impl of each node
    """
    dependencies = [
        PassShapeTypePropagation,  # The shape and type info are required for inference
        PassFixElementD
    ]

    def __init__(self, dag_ir: DAGIR) -> None:
        super().__init__(dag_ir)
        self.no_op_elimination = PassNoOpElimination(dag_ir)

    def requires(self) -> None:
        # Verify "accum" is in the arg list
        if not self.dag_ir.has_node("accum"):
            raise SyntaxError("Cannot find 'accum' in the argument list.")

    def call(self):
        # The loop structure of the epilogue is determined by the
        # accumulator shape
        accumulator: LoadNode = self.dag_ir.get_node_meta("accum")
        problem_size = accumulator.tensor.shape

        for node_meta in self.dag_ir.node_metas_topological_order():
            node_meta.get_underlying_impl(problem_size)

    def ensures(self) -> None:
        # Some nodes will be lowered to NoOp, eliminate them
        self.no_op_elimination()
        # Lower to cc-specific impl
        for node_meta in self.dag_ir.nodes_meta:
            node_impl_ccs = getattr(evt_backend, f"sm{cc_map[self.cc]}_nodes")
            node_meta.underlying_impl = getattr(
                node_impl_ccs,
                f"Sm{cc_map[self.cc]}" + node_meta.underlying_impl.__class__.__name__
            )(node_meta)
