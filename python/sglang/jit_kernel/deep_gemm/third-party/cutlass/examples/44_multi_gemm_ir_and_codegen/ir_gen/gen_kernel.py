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

import gen_ir
import helper
import gen_threadblock as gen_tb


class gen_default_Gemm:
    def __init__(self, template_param, gen_class_name, b2b_num, cutlass_deps_root, project_root):
        self.gen_class_name = "B2bGemm"
        self.template_param = template_param
        self.b2b_num = b2b_num

        self.cutlass_deps_root = cutlass_deps_root
        self.project_root = project_root

    def gen_B2bMma(self, specialized_template_args):
        code = "using B2bMma = typename cutlass::gemm::threadblock::DefaultB2bMma<\n"
        code += specialized_template_args
        code += ">::ThreadblockB2bMma;\n"

        # print(code)
        return code

    def gen_epilogue(self):
        epilogue_code = ""
        epilogue_code += helper.var_idx("static const int kPartitionsK", self.b2b_num - 1) + helper.var_idx(" = ThreadblockShape", self.b2b_num - 1) + helper.var_idx("::kK / WarpShape", self.b2b_num - 1) + "::kK;\n"

        epilogue_code += "using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<\n"
        epilogue_code += "    " + helper.var_idx("ThreadblockShape", self.b2b_num - 1) + ",\n"
        epilogue_code += "    " + helper.var_idx("typename B2bMma::Operator", self.b2b_num - 1) + ",\n"
        epilogue_code += "    " + helper.var_idx("kPartitionsK", self.b2b_num - 1) + ",\n"
        epilogue_code += "    " + helper.var_idx("EpilogueOutputOp", self.b2b_num - 1) + ",\n"
        epilogue_code += "    " + helper.var_idx("EpilogueOutputOp", self.b2b_num - 1) + "::kCount\n"
        epilogue_code += ">::Epilogue;\n"

        epilogue_code += "using B2bGemmKernel = kernel::B2bGemm<B2bMma, Epilogue, ThreadblockSwizzle, SplitKSerial>;\n\n"

        return epilogue_code


    def gen_include_header(self):
        code = '''
/* Auto Generated code - Do not edit.*/

#pragma once
#include \"{cutlass_dir}cutlass/cutlass.h\"

#include \"{cutlass_dir}cutlass/layout/matrix.h\"
#include \"{cutlass_dir}cutlass/numeric_types.h\"

#include \"{cutlass_dir}cutlass/epilogue/threadblock/epilogue.h\"
#include \"{cutlass_dir}cutlass/epilogue/thread/linear_combination.h\"

#include \"{cutlass_dir}cutlass/gemm/gemm.h\"
#include \"{cutlass_dir}cutlass/gemm/kernel/gemm_pipelined.h\"
#include \"{cutlass_dir}cutlass/gemm/threadblock/default_mma_core_sm75.h\"
#include \"{cutlass_dir}cutlass/gemm/threadblock/default_mma_core_sm70.h\"
#include \"{cutlass_dir}cutlass/gemm/threadblock/default_mma_core_sm80.h\"
#include \"{cutlass_dir}cutlass/gemm/threadblock/default_mma_core_simt.h\"
#include \"{cutlass_dir}cutlass/gemm/threadblock/threadblock_swizzle.h\"
#include \"{cutlass_dir}cutlass/epilogue/threadblock/default_epilogue_tensor_op.h\"
#include \"{cutlass_dir}cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h\"
#include \"{cutlass_dir}cutlass/epilogue/threadblock/default_epilogue_simt.h\"

#include \"{cutlass_dir}cutlass/transform/threadblock/predicated_tile_iterator.h\"

#include \"../kernel/b2b_gemm.h\"
#include \"../threadblock/default_b2b_mma.h\"
'''.format(cutlass_dir=self.cutlass_deps_root)
        return code

    def gen_code(self):
        gen_using = ''
        # Generate default template struct
        gen_code = gen_ir.gen_template_struct("Default" + self.gen_class_name, self.template_param,"", speicalized = None, set_default=False)
        

        filter_list = []
        filter_list.append(('Stages', 2))
        filter_list.append(("OperatorClass", "arch::OpClassTensorOp"))
        filter_list.append(("ArchTag", "arch::Sm75"))

        for i in range(self.b2b_num):
            filter_list.append((helper.var_idx("LayoutC", i), "layout::RowMajor"))


        rtn_template_args, speicalized_template_args = gen_ir.filtered_param(self.template_param, filter_list, keep_= True)


        B2bMma_code = self.gen_B2bMma(speicalized_template_args)
        epilogue_and_rest_code = self.gen_epilogue()
       
        gen_special_code = gen_ir.gen_template_struct("Default" + self.gen_class_name, rtn_template_args, B2bMma_code + epilogue_and_rest_code, speicalized = speicalized_template_args, set_default=False)

        code = gen_ir.gen_namespace("cutlass", gen_ir.gen_namespace("gemm", gen_ir.gen_namespace("kernel", gen_code + gen_special_code)))

        return self.gen_include_header() + code


class gen_Kernel:
    def __init__(self, template_param, gen_class_name, b2b_num, cutlass_deps_root, project_root):
        self.gen_class_name = "B2bGemm"
        self.template_param = template_param
        self.b2bnum = b2b_num

        self.cutlass_deps_root = cutlass_deps_root
        self.project_root = project_root

    def gen_include_header(self):
        code = '''
#pragma once

#include \"{cutlass_dir}cutlass/cutlass.h\"
#include \"{cutlass_dir}cutlass/gemm/gemm.h\"
#include \"{cutlass_dir}cutlass/matrix_coord.h\"\n'''.format(cutlass_dir=self.cutlass_deps_root)  
        return code

    def gen_Params(self):
        gen_param = ""
        for i in range(self.b2bnum):
            gen_param += "    " + helper.var_idx("cutlass::gemm::GemmCoord problem_size_", i) + ";\n"
        gen_param += "    " + "cutlass::gemm::GemmCoord grid_tiled_shape;\n" 
        gen_param += "    " + "typename B2bMma::IteratorA0::Params params_A0;\n" 
        gen_param += "    " + "typename B2bMma::IteratorA0::TensorRef ref_A0;\n" 

        for i in range(self.b2bnum):
            gen_param += "    " + helper.var_idx("typename B2bMma::IteratorB", i) + helper.var_idx("::Params params_B", i) + ";\n"
            gen_param += "    " + helper.var_idx("typename B2bMma::IteratorB", i) + helper.var_idx("::TensorRef ref_B", i) + ";\n"
            if i == self.b2bnum - 1:
                gen_param += "    " + helper.var_idx("typename Epilogue::OutputTileIterator::Params params_C", i) + ";\n"
                gen_param += "    " + helper.var_idx("typename Epilogue::OutputTileIterator::TensorRef ref_C", i) + ";\n"

            else:
                gen_param += "    " + helper.var_idx("typename FusedAddBiasEpilogue", i) + helper.var_idx("::OutputTileIterator::Params params_C", i) + ";\n"
                gen_param += "    " + helper.var_idx("typename FusedAddBiasEpilogue", i) + helper.var_idx("::OutputTileIterator::TensorRef ref_C", i) + ";\n"

                


        gen_param += "    " + helper.var_idx("typename Epilogue::OutputTileIterator::Params params_D", self.b2bnum - 1) + ";\n"
        gen_param += "    " + helper.var_idx("typename Epilogue::OutputTileIterator::TensorRef ref_D", self.b2bnum - 1) + ";\n"

        for i in range(self.b2bnum):
            gen_param += "    " + helper.var_idx("typename OutputOp", i) + helper.var_idx("::Params output_op_", i) + ";\n"

        gen_param += "    " + 'int batch_count' + ";\n"
        gen_param += "    " + 'int gemm_k_iterations_0' + ";\n"


        return gen_param

    def gen_Memberfunc(self):
        code_default = "\nCUTLASS_HOST_DEVICE\n"
        code_default += "Params()"

        code_default += " { } \n\n"

        code_construct = "\nCUTLASS_HOST_DEVICE\n"
        code_construct += "Params(\n"

        for i in range(self.b2bnum):
            code_construct += "    " + helper.var_idx("cutlass::gemm::GemmCoord const & problem_size_", i) + ",\n"

        code_construct += "    " + "cutlass::gemm::GemmCoord const & grid_tiled_shape,\n"

        code_construct += "    " + "typename B2bMma::IteratorA0::TensorRef ref_A0,\n" 

        for i in range(self.b2bnum):
            code_construct += "    " + helper.var_idx("typename B2bMma::IteratorB", i) + helper.var_idx("::TensorRef ref_B", i) + ",\n"
            if i == self.b2bnum - 1:
                code_construct += "    " + helper.var_idx("typename Epilogue::OutputTileIterator::TensorRef ref_C", i) + ",\n"
            else:
                code_construct += "    " + helper.var_idx("typename FusedAddBiasEpilogue", i) + helper.var_idx("::OutputTileIterator::TensorRef ref_C", i) + ",\n"

        code_construct += "    " + helper.var_idx("typename Epilogue::OutputTileIterator::TensorRef ref_D", self.b2bnum - 1) + ",\n"
        for i in range(self.b2bnum):
            code_construct += "    " + helper.var_idx("typename OutputOp", i) + helper.var_idx("::Params output_op_", i) + helper.var_idx(" = typename OutputOp", i) + "::Params(),\n"

        code_construct += "    " + "int batch_count = 1\n"

        code_construct += "):\n"
        
        for i in range(self.b2bnum):
            code_construct += "    " + helper.var_idx("problem_size_", i) + helper.var_idx("(problem_size_", i) + "),\n"

        code_construct += "    " + "grid_tiled_shape(grid_tiled_shape),\n"
        code_construct += "    " + "params_A0(ref_A0.layout()),\n"
        code_construct += "    " + "ref_A0(ref_A0),\n"

        for i in range(self.b2bnum):
            code_construct += "    " + helper.var_idx("params_B", i) + helper.var_idx("(ref_B", i) + ".layout()),\n"
            code_construct += "    " + helper.var_idx("ref_B", i) + helper.var_idx("(ref_B", i) + "),\n"
            code_construct += "    " + helper.var_idx("params_C", i) + helper.var_idx("(ref_C", i) + ".layout()),\n"
            code_construct += "    " + helper.var_idx("ref_C", i) + helper.var_idx("(ref_C", i) + "),\n"

        code_construct += "    " + helper.var_idx("params_D", self.b2bnum - 1) + helper.var_idx("(ref_D", self.b2bnum - 1) + ".layout()),\n"
        code_construct += "    " + helper.var_idx("ref_D", self.b2bnum - 1) + helper.var_idx("(ref_D", self.b2bnum - 1) + "),\n"

        for i in range(self.b2bnum):
            code_construct += "    " + helper.var_idx("output_op_", i) + helper.var_idx("(output_op_", i) + "), \n"

        code_construct += "    " + "batch_count(batch_count) {\n"
        code_construct += "    " + helper.var_idx("gemm_k_iterations_", 0) + helper.var_idx(" = (problem_size_", 0) + helper.var_idx(".k() + B2bMma::Shape", 0) + helper.var_idx("::kK - 1) / B2bMma::Shape", 0) + "::kK;\n"

        code_construct += "}\n"

        return code_default + code_construct

    def gen_using(self):
        code_using = ""

        for i in range(self.b2bnum - 1):
            code_using += "    " + helper.var_idx("using OutputOp", i) +  helper.var_idx(" = typename B2bMma::OutputOp", i) + ";\n"

        code_using += "    " + helper.var_idx("using OutputOp", self.b2bnum - 1) + " = typename Epilogue::OutputOp;\n"

        for i in range(self.b2bnum - 1):
            code_using += "    " + helper.var_idx("using FusedAddBiasEpilogue", i) + helper.var_idx(" = typename B2bMma::FusedAddBiasEpilogue", i) +";\n"


        code_using += "    "  + "using WarpCount0 = typename B2bMma::WarpCount0;\n"
        code_using += "    "  + "static int const kThreadCount = 32 * WarpCount0::kCount;\n"

        code_using += gen_ir.gen_struct("Params", self.gen_Params() + self.gen_Memberfunc())

        code_using += "union SharedStorage {\n"
        code_using += "    " + "typename B2bMma::B2bMmaSharedStorage main_loop;\n"
        code_using += "    " + "typename Epilogue::SharedStorage epilogue;\n"
        code_using += "};\n"

        return code_using

    def gen_can_implement(self):
        gen_code = ""
        return gen_code

    def gen_operator_and_constr(self):
        ctr_code = "CUTLASS_HOST_DEVICE\n"
        ctr_code += self.gen_class_name + "() { } \n\n"
        operator_code = "CUTLASS_DEVICE\n"
        operator_code += "void operator()(Params const &params, SharedStorage &shared_storage) {\n"
        operator_code += "    " + "ThreadblockSwizzle threadblock_swizzle;\n"
        operator_code += "    " + "cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);\n"
        operator_code += "    " + "int batch_idx = threadblock_tile_offset.k();\n"
        operator_code += "    " + "if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||\n"
        operator_code += "    " + "params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {\n"
        operator_code += "    " + "    " + "return;\n"
        operator_code += "    " + "}\n"
        
        operator_code += "    " + "cutlass::MatrixCoord tb_offset_A0{\n"
        operator_code += "    " + "    " + "threadblock_tile_offset.m() * B2bMma::Shape0::kM,\n"
        operator_code += "    " + "    " + "0\n"
        operator_code += "    " + "};\n"

        for i in range(self.b2bnum):
            operator_code += "    " + helper.var_idx("cutlass::MatrixCoord tb_offset_B", i) + "{\n"
            operator_code += "    " + "    " + "0,\n"
            operator_code += "    " + "    " + helper.var_idx("threadblock_tile_offset.n() * B2bMma::Shape", i) + "::kN\n"
            operator_code += "    " + "};\n"
       
        operator_code += "    " + "int thread_idx = threadIdx.x;\n\n"

        operator_code += "    " + "MatrixCoord threadblock_offset(\n"
        operator_code += "    " + "    " + helper.var_idx("threadblock_tile_offset.m() * B2bMma::Shape", self.b2bnum - 1) + "::kM,\n"
        operator_code += "    " + "    " + helper.var_idx("threadblock_tile_offset.n() * B2bMma::Shape", self.b2bnum - 1) + "::kN\n"
        operator_code += "    " + ");\n"

        operator_code += "    " + "typename B2bMma::IteratorA0 iterator_A0(\n"
        operator_code += "    " + "    " + "params.params_A0,\n"
        operator_code += "    " + "    " + "params.ref_A0.data(),\n"
        operator_code += "    " + "    " + "params.problem_size_0.mk(),\n"
        operator_code += "    " + "    " + "thread_idx,\n"
        operator_code += "    " + "    " + "tb_offset_A0);\n"

        operator_code += "    " + "iterator_A0.add_pointer_offset(batch_idx * params.problem_size_0.m() * params.problem_size_0.k());\n\n"


        for i in range (self.b2bnum):
            operator_code += "    " + helper.var_idx("typename B2bMma::IteratorB", i ) + helper.var_idx(" iterator_B", i) + "(\n"
            operator_code += "    " + "    " + helper.var_idx("params.params_B", i) + ",\n"
            operator_code += "    " + "    " + helper.var_idx("params.ref_B", i) + ".data(),\n"
            operator_code += "    " + "    " + helper.var_idx("params.problem_size_", i) + ".kn(),\n"
            operator_code += "    " + "    " + "thread_idx,\n"
            operator_code += "    " + "    " + helper.var_idx("tb_offset_B", i) + ");\n"
            operator_code += "    " + helper.var_idx("iterator_B", i) + helper.var_idx(".add_pointer_offset(batch_idx * params.problem_size_", i) + helper.var_idx(".n() * params.problem_size_", i) + ".k());\n\n"


        for i in range (self.b2bnum - 1):
            operator_code += "    " + helper.var_idx("typename FusedAddBiasEpilogue", i ) + helper.var_idx("::OutputTileIterator iterator_C", i) + "(\n"
            operator_code += "    " + "    " + helper.var_idx("params.params_C", i) + ",\n"
            operator_code += "    " + "    " + helper.var_idx("params.ref_C", i) + ".data(),\n"
            operator_code += "    " + "    " + helper.var_idx("params.problem_size_" , i) + ".mn(),\n"
            operator_code += "    " + "    " + "thread_idx,\n"
            operator_code += "    " + "    " + "threadblock_offset" + ");\n"
            operator_code += "    " + helper.var_idx("int ref_C", i) + helper.var_idx("_stride = params.ref_C", i) + ".stride()[0];\n"
            operator_code += "    " + helper.var_idx("iterator_C", i) + helper.var_idx(".add_pointer_offset(batch_idx * params.problem_size_", i) + helper.var_idx(".n() * (ref_C", i) + helper.var_idx("_stride == 0 ? 1 : params.problem_size_", i) + ".m()));\n\n"


        for i in range (self.b2bnum - 1):
            operator_code += "    " + helper.var_idx("FusedAddBiasEpilogue", i ) + helper.var_idx(" epilogue_", i ) + ";\n"


        operator_code += "    " + "int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);\n"
        operator_code += "    " + "int lane_idx = threadIdx.x % 32;\n"

        for i in range (self.b2bnum - 1):
            operator_code += "    " + helper.var_idx("OutputOp", i) + helper.var_idx(" output_op_", i) + helper.var_idx("(params.output_op_", i) + ");\n"

        operator_code += "    " + "B2bMma b2bMma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);\n"
        
        operator_code += "    " + "typename B2bMma::FragmentC0 src_accum;\n"
        operator_code += "    " + helper.var_idx("typename B2bMma::FragmentC", self.b2bnum - 1)+ " accumulators;\n"

        operator_code += "    " + "src_accum.clear();\n"
        operator_code += "    " + "accumulators.clear();\n"
        operator_code += "    " + "b2bMma(params.gemm_k_iterations_0, accumulators, iterator_A0, "

        for i in range(self.b2bnum):
            operator_code += helper.var_idx("iterator_B", i) + ", "
        
        operator_code += "src_accum"
        if self.b2bnum != 1:
            operator_code += ", "
        for i in range(self.b2bnum - 1):
            operator_code += helper.var_idx("output_op_", i) + ", "
        
        for i in range(self.b2bnum - 1):
            operator_code += helper.var_idx("epilogue_", i) + ", "

        for i in range(self.b2bnum - 1):
            final = ", "
            if i == self.b2bnum - 2:
                final =""
            operator_code += helper.var_idx("iterator_C", i) + final
        operator_code += ");\n"

        operator_code += "    " + helper.var_idx("OutputOp", self.b2bnum - 1) + helper.var_idx(" output_op_", self.b2bnum - 1) + helper.var_idx("(params.output_op_", self.b2bnum - 1) + ");\n"
        operator_code += "    " + "threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);\n"

        

        operator_code += "    " + helper.var_idx("typename Epilogue::OutputTileIterator iterator_C", self.b2bnum - 1) + "(\n"
        operator_code += "    " + "    " + helper.var_idx("params.params_C", self.b2bnum - 1) + ",\n"
        operator_code += "    " + "    " + helper.var_idx("params.ref_C", self.b2bnum - 1) + ".data(),\n"
        operator_code += "    " + "    " + helper.var_idx("params.problem_size_", self.b2bnum - 1) + ".mn(),\n"
        operator_code += "    " + "    " + "thread_idx,\n"
        operator_code += "    " + "    " + "threadblock_offset\n"
        operator_code += "    " + ");\n"
        operator_code += "    " + helper.var_idx("int ref_C", self.b2bnum - 1) + helper.var_idx("_stride = params.ref_C", self.b2bnum - 1) + ".stride()[0];\n"

        operator_code += "    " + helper.var_idx("iterator_C", self.b2bnum - 1) + helper.var_idx(".add_pointer_offset(batch_idx * params.problem_size_", self.b2bnum - 1) + helper.var_idx(".n() * (ref_C", self.b2bnum - 1) + helper.var_idx("_stride == 0 ? 1 : params.problem_size_", self.b2bnum - 1) + ".m()));\n\n"

        operator_code += "    " + helper.var_idx("typename Epilogue::OutputTileIterator iterator_D", self.b2bnum - 1) + "(\n"
        operator_code += "    " + "    " + helper.var_idx("params.params_D", self.b2bnum - 1) + ",\n"
        operator_code += "    " + "    " + helper.var_idx("params.ref_D", self.b2bnum - 1) + ".data(),\n"
        operator_code += "    " + "    " + helper.var_idx("params.problem_size_", self.b2bnum - 1) + ".mn(),\n"
        operator_code += "    " + "    " + "thread_idx,\n"
        operator_code += "    " + "    " + "threadblock_offset\n"
        operator_code += "    " + ");\n"
        operator_code += "    " + helper.var_idx("iterator_D", self.b2bnum - 1) + helper.var_idx(".add_pointer_offset(batch_idx * params.problem_size_", self.b2bnum - 1) + helper.var_idx(".n() * params.problem_size_", self.b2bnum - 1) + ".m());\n\n"


        operator_code += "    " + "Epilogue epilogue(\n"
        operator_code += "    " + "    " + "shared_storage.epilogue,\n"
        operator_code += "    " + "    " + "thread_idx,\n"
        operator_code += "    " + "    " + "warp_idx,\n"
        operator_code += "    " + "    " + "lane_idx\n"
        operator_code += "    " + ");\n"

        operator_code += "    " + "epilogue("
        operator_code += helper.var_idx("output_op_", self.b2bnum - 1) + ", "
        operator_code += helper.var_idx("iterator_D", self.b2bnum - 1) + ", "
        operator_code += "accumulators, "
        operator_code += helper.var_idx("iterator_C", self.b2bnum - 1) + ");\n"
        operator_code += "}\n"

        return ctr_code + operator_code

    def gen_include_header(self):
        code = '''
#pragma once

#include \"{cutlass_dir}cutlass/cutlass.h\"

#include \"{cutlass_dir}cutlass/gemm/gemm.h\"
#include \"{cutlass_dir}cutlass/matrix_coord.h\"
#include \"{cutlass_dir}cutlass/semaphore.h\"
'''.format(cutlass_dir=self.cutlass_deps_root)
        return code
    def gen_code(self):
        
        template_param = []
        template_param.append(("typename", "B2bMma"))
        template_param.append(("typename", "Epilogue"))
        template_param.append(("typename", "ThreadblockSwizzle"))
        template_param.append((bool, "SplitKSerial"))

        code_body = ""
        code_body += self.gen_using()
        code_body += self.gen_operator_and_constr()

        struct_code = gen_ir.gen_template_struct(self.gen_class_name, template_param, code_body)
        code = self.gen_include_header()
        code += gen_ir.gen_namespace("cutlass", gen_ir.gen_namespace("gemm", gen_ir.gen_namespace("kernel", struct_code)))

        return self.gen_include_header() + code



class gen_kernel:
    def __init__(self, template_param, gen_class_name, b2b_num, output_dir, cutlass_deps_root, project_root):
        self.template_param = template_param

        self.gen_class_name = "B2bGemm"
        self.gen_kernel_name = gen_class_name + "Kernel"
        self.template_args = []

        self.cutlass_deps_root = cutlass_deps_root
        self.project_root = project_root

        self.gen_default_b2b_gemm = gen_default_Gemm(template_param, gen_class_name, b2b_num, cutlass_deps_root, project_root)
        self.gen_Kerenl = gen_Kernel(template_param, gen_class_name, b2b_num, cutlass_deps_root, project_root)

        # Include gen_threadBlock
        self.gen_threadBlock = gen_tb.gen_threadblock(template_param, gen_class_name, b2b_num, output_dir, cutlass_deps_root, project_root)
    
        self.file_dir = output_dir + "/kernel/"

    def gen_code(self, first_use_1stage):

        default_b2b_gemm = self.gen_default_b2b_gemm.gen_code()
        
        print("[INFO]: Gen kernel code [default_b2b_gemm.h]output Dir: is ", self.file_dir)

        with open(self.file_dir + "default_b2b_gemm.h", "w+") as f:
            f.write(default_b2b_gemm)

        kernel = self.gen_Kerenl.gen_code()
        print("[INFO]: Gen kernel code [b2b_gemm.h]output Dir: is ", self.file_dir)

        with open(self.file_dir + "b2b_gemm.h", "w+") as f:
            f.write(kernel)

        # Call code to gen threadblock
        self.gen_threadBlock.gen_code(first_use_1stage)
