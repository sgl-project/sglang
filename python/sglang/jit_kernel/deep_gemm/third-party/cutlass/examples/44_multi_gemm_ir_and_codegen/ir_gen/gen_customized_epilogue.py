#################################################################################################
#
# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ast

fuse_gemm_info = [
    {
    'epilogue': {
        'tp': 'LeakyRelu', #'CustomizedLeaky_RELU'
        'bias': {'addbias': False, 'bias_tp': 'mat'}, 
        'args': [('float', 'leaky_alpha', 1.3), ], 
        'func': '''
y = max(leaky_alpha * x, x)
y = y * x
    '''
        }
    },

]
class AnalysisNodeVisitor(ast.NodeVisitor):
    def visit_Import(self,node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_ImportFrom(self,node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self,node):
        print('Node type: Assign and fields: ', node._fields)
        # print('Node type: Assign and targets value: ', node.targets, node.value)

        ast.NodeVisitor.generic_visit(self, node)
    
    def visit_BinOp(self, node):
        print('Node type: BinOp and fields: ', node._fields)
        print('node op: ', type(node.op).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Expr(self, node):
        print('Node type: Expr and fields: ', node._fields)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Num(self,node):
        print('Node type: Num and fields: ', node._fields)
        print('Node type: Num: ', node.n)

    def visit_Name(self,node):
        print('Node type: Name and fields: ', node._fields)
        print('Node type: Name and fields: ', type(node.ctx).__name__, node.id)

        ast.NodeVisitor.generic_visit(self, node)

    def visit_Str(self, node):
        print('Node type: Str and fields: ', node._fields)

class CodeVisitor(ast.NodeVisitor):
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()
            self.generic_visit(node)

    def visit_Assign(self, node):
        print('Assign %s' % node.value)
        self.generic_visit(node)

    def visit_Name(self, node):
        print("Name:", node.id)
        self.generic_visit(node)


    def visit_FunctionDef(self, node):
        print('Function Name:%s'% node.name.op)
        self.generic_visit(node)
        func_log_stmt = ast.Print(
            dest = None,
            values = [ast.Str(s = 'calling func: %s' % node.name, lineno = 0, col_offset = 0)],
            nl = True,
            lineno = 0,
            col_offset = 0,
        )
        node.body.insert(0, func_log_stmt)

visitor = AnalysisNodeVisitor()

code = \
'''

a=max(leaky_alpha * x, x +1)

'''

visitor.visit(ast.parse(code))
