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
Python AST frontend that parses input into DAG IR
"""

import ast
import inspect
import textwrap

from cutlass_library import DataType

import cutlass_cppgen
from cutlass_cppgen.backend.evt.frontend.frontend_base import EVTFrontendBase
from cutlass_cppgen.backend.epilogue import identity, relu, tanh, sigmoid, silu, hardswish, gelu
from cutlass_cppgen.backend.library import FunctionalOp


class PythonASTFrontend(EVTFrontendBase, ast.NodeVisitor):
    def __init__(self, cc, element_compute=DataType.f32, **kwargs):
        super().__init__(cc, element_compute, **kwargs)
        # Flags
        # If this state is True, visit_Constant returns values without creating imm node
        self.no_imm = False
        self.visiting_return = False

    def parse(self, example_inputs):
        self.example_inputs = example_inputs
        self.source = textwrap.dedent(inspect.getsource(self.__call__))
        self.ast = ast.parse(self.source)
        self.visit(self.ast)

    #
    # Helper functions
    #
    @staticmethod
    def ast_op_to_bindings(op):
        mapping = {
            ast.Add: FunctionalOp.Plus,
            ast.Sub: FunctionalOp.Minus,
            ast.Mult: FunctionalOp.Multiplies,
            ast.Div: FunctionalOp.Divides,
            "maximum": FunctionalOp.Maximum,
            "minimum": FunctionalOp.Minimum,
            "identity": identity.binding_type,
            "relu": relu.binding_type,
            "tanh": tanh.binding_type,
            "sigmoid": sigmoid.binding_type,
            "silu": silu.binding_type,
            "hardswish": hardswish.binding_type,
            "gelu": gelu.binding_type,
            "multiply_add": FunctionalOp.MultiplyAdd,
            "sum": (FunctionalOp.Plus, FunctionalOp.AtomicAdd),
            "max": (FunctionalOp.Maximum, FunctionalOp.AtomicMaximum),
            "exp": FunctionalOp.Exp
        }
        return mapping[op]

    #
    # Visiting different node types
    #

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Visit args and register load nodes
        for arg in node.args.args:
            self.visit(arg)
        for expr in node.body:
            self.visit(expr)

    def visit_arg(self, node: ast.arg):
        # Name of the argument
        name = node.arg
        try:
            example_tensor = self.example_inputs[name]
        except:
            raise RuntimeError(f"Example input for {name} is not provided.")

        self.add_load_node(name, example_tensor)

    def visit_Name(self, node: ast.Name):
        return node.id

    def visit_Constant(self, node: ast.Constant):
        if self.no_imm:
            return node.value
        else:
            name = self.add_imm(node.value)
            return name

    def visit_Tuple(self, node: ast.Tuple):
        results = []
        for elt in node.elts:
            results.append(self.visit(elt))
        return tuple(results)

    def visit_keyword(self, node: ast.keyword):
        return {node.arg: self.visit(node.value)}

    def visit_BinOp(self, node: ast.BinOp):
        if self.visiting_return:
            raise SyntaxError("Return value cannot be an expression")
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        op = self.ast_op_to_bindings(type(node.op))
        name = self.add_compute_node(op)

        # Add edges
        # The edge weights are used to sort the input args
        self.add_edge(lhs, name, weight=0)
        self.add_edge(rhs, name, weight=1)
        return name

    def visit_Assign(self, node: ast.BinOp):
        target = self.visit(node.targets[0])
        value = self.visit(node.value)
        # Create the assign node
        self.add_store_node(target)

        # Add edges
        self.add_edge(value, target)
        return target

    def visit_Call(self, node: ast.Call):
        if self.visiting_return:
            raise SyntaxError("Return value cannot be an expression")
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]

        if func in self.layout_fns.keys():
            # Parse kwargs
            # By default, visiting imm automatically creates a load node
            # However, in function call, keyword args are used to set
            # specific function attributes such as indices for permute
            # So no_imm is set to True temporarily
            self.no_imm = True
            kwargs = {}
            for kw in node.keywords:
                kwargs.update(self.visit(kw))
            self.no_imm = False
            op = self.layout_fns[func]
            name = self.add_layout_node(op, kwargs)
        else:
            op = self.ast_op_to_bindings(func)
            name = self.add_compute_node(op)

        # Add edges
        for idx, arg in enumerate(args):
            self.add_edge(arg, name, weight=idx)
        return name

    def visit_Return(self, node: ast.Return):
        self.visiting_return = True
        results = self.visit(node.value)
        self.visiting_return = False
        self.return_names = results
        if not isinstance(results, tuple):
            results = (results,)
        for rst in results:
            try:
                example_tensor = self.example_inputs[rst]
            except:
                raise RuntimeError(f"Example input for {rst} is not provided.")
            self.set_store_tensor(rst, example_tensor)
            self.mark_output(rst)
