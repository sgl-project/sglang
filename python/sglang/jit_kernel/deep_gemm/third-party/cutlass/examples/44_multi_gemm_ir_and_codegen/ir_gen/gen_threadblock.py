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


class gen_default_b2b_mma:
    def __init__(self, template_param, gen_class_name, b2b_num,cutlass_deps_root, project_root):
        self.gen_class_name = "DefaultB2bMma"
        self.template_param = template_param
        self.b2b_num = b2b_num

        self.cutlass_deps_root = cutlass_deps_root
        self.project_root = project_root

    def gen_include_header(self):
        code = '''
/* Auto Generated code - Do not edit.*/

#pragma once

#include \"{cutlass_dir}cutlass/cutlass.h\"
#include \"{cutlass_dir}cutlass/numeric_types.h\"
#include \"{cutlass_dir}cutlass/arch/arch.h\"

#include \"{cutlass_dir}cutlass/transform/threadblock/predicated_tile_iterator.h\"
#include \"{cutlass_dir}cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h\"
#include \"{cutlass_dir}cutlass/gemm/threadblock/default_mma_core_sm70.h\"
#include \"{cutlass_dir}cutlass/gemm/threadblock/default_mma_core_sm75.h\"
#include \"{cutlass_dir}cutlass/gemm/threadblock/default_mma_core_sm80.h\"

#include \"../threadblock/b2b_mma_pipelined.h\"
#include \"../../fixed_impl/epilogue/threadblock/fused_bias_act_epilogue.h\"
#include \"../../fixed_impl/epilogue/threadblock/default_bias_act_epilogue_tensor_op.h\"
#include \"../../fixed_impl/gemm/warp/mma_tensor_op_fragment_iterator_without_output_op.h\"
'''.format(cutlass_dir=self.cutlass_deps_root)
        return code


    def gen_using_MmaCore(self, stage):
        threadBlockShape = "ThreadblockShape"
        warpShape = "WarpShape"
        instrunctionShape = "InstructionShape"
        Mma_typename = "typename cutlass::gemm::threadblock::DefaultMmaCore"


        gen_code = ""

        for i in range(self.b2b_num):
            code_using = "using MmaCore" + str(i)
            gen_code += code_using + " = " + gen_ir.gen_declare_template_struct(Mma_typename, \
                                                helper.var_idx(threadBlockShape, i), helper.var_idx(warpShape, i), instrunctionShape, \
                                                "ElementA", "LayoutA", \
                                                helper.var_idx("ElementB", i), helper.var_idx("LayoutB", i), \
                                                helper.var_idx("ElementAccumulator", i), "layout::RowMajor", \
                                                "OperatorClass", str(stage), "Operator")
        return gen_code

    def gen_using_FusedAddBiasEpilogue(self):
        gen_code = ""
        for i in range(self.b2b_num - 1):
            code_using = helper.var_idx("using FusedAddBiasEpilogue", i)
            epilogue_name = "typename cutlass::epilogue::threadblock::DefaultFusedBiasActEpilogueTensorOp"
            template_args = helper.var_idx("<ThreadblockShape", i) + helper.var_idx(",typename MmaCore", i) + helper.var_idx("::MmaPolicy::Operator, 1, EpilogueOutputOp", i) + ", 2>::Epilogue"

            gen_code += code_using + " = " + epilogue_name + template_args + ";\n"

        return gen_code        
        

    def gen_using_Iterator(self):
        code_using = "using IteratorA0"
        iterator_typename = "cutlass::transform::threadblock::PredicatedTileIterator"
        MmaCore = "MmaCore0"
        matrix_shape = "cutlass::MatrixShape<" + MmaCore + "::Shape::kM, " + MmaCore + "::Shape::kK>"
        iterator_map = "typename " + MmaCore + "::IteratorThreadMapA"
        gen_code = code_using + " = " + gen_ir.gen_declare_template_struct(iterator_typename, \
                                                matrix_shape, "ElementA", "LayoutA", "1", iterator_map, "AlignmentA_")

        for i in range(self.b2b_num):
            code_using = "using IteratorB" + str(i)
            iterator_typename = "cutlass::transform::threadblock::PredicatedTileIterator"
            MmaCore = "MmaCore" + str(i)
            matrix_shape = "cutlass::MatrixShape<" + MmaCore + "::Shape::kK, " + MmaCore + "::Shape::kN>"
            iterator_map = "typename " + MmaCore + "::IteratorThreadMapB"

            gen_code += code_using + " = " + gen_ir.gen_declare_template_struct(iterator_typename, \
                                                matrix_shape, helper.var_idx("ElementB", i), helper.var_idx("LayoutB", i), "0", iterator_map, "AlignmentB_")
        
        return gen_code

    def gen_fragment_iterator(self):
        gen_code = "using AccumulatorLayout = cutlass::layout::ColumnMajor;\n"
        
        for i in range(1, self.b2b_num):
            code_using = "using FragmentIteratorA" + str(i)
            iterator_typename = "cutlass::gemm::warp::MmaTensorOpPureFragmentIterator"
            curr_MmaCore = "MmaCore" + str(i)
            prev_MmaCore = "MmaCore" + str(i - 1)
            Matrix_shape_curr = "cutlass::MatrixShape<" + curr_MmaCore + "::WarpShape::kM, " + curr_MmaCore + "::InstructionShape::kK>"
            Matrix_shape_prev = "cutlass::MatrixShape<" + prev_MmaCore + "::WarpShape::kM, " + prev_MmaCore + "::WarpShape::kN>"
            Curr_shape_kK = curr_MmaCore + "::Shape::kK"

            gen_code += code_using + " = " + gen_ir.gen_declare_template_struct(iterator_typename, \
                                                Matrix_shape_curr, Matrix_shape_prev, Curr_shape_kK, \
                                                    helper.var_idx("ElementAccumulator", i-1), "ElementA", \
                                                        "AccumulatorLayout", "InstructionShape_", "true")

        return gen_code

    def gen_threadblockmma(self):
        code_using = "using ThreadblockB2bMma"
        iterator_typename = "cutlass::gemm::threadblock::B2bMmaPipelined"

        MmaPipelined_param_Mma0_shape = "typename MmaCore0::Shape"
        MmaPipelined_param_Mma0_iteratorA = "IteratorA0"
        MmaPipelined_param_Mma0_smemIteratorA = "typename MmaCore0::SmemIteratorA"
        MmaPipelined_param_Mma0_iteratorB = "IteratorB0"
        MmaPipelined_param_Mma0_smemIteratorB = "typename MmaCore0::SmemIteratorB"

        MmaPipelined_param_list = MmaPipelined_param_Mma0_shape + ", " + MmaPipelined_param_Mma0_iteratorA + ", " + MmaPipelined_param_Mma0_smemIteratorA + ", " + MmaPipelined_param_Mma0_iteratorB + ", " + MmaPipelined_param_Mma0_smemIteratorB + ", "

        for i in range(1, self.b2b_num):
            MmaPipelined_param_Mma_shape = "typename MmaCore" + str(i) + "::Shape"
            MmaPipelined_param_Mma_iteratorA = "FragmentIteratorA" + str(i)
            MmaPipelined_param_Mma_iteratorB = "IteratorB" + str(i)
            MmaPipelined_param_Mma_smemIteratorB = "typename MmaCore" + str(i) + "::SmemIteratorB"

            MmaPipelined_param_list += MmaPipelined_param_Mma_shape + ", " + MmaPipelined_param_Mma_iteratorA + ", " + MmaPipelined_param_Mma_iteratorB + ", " + MmaPipelined_param_Mma_smemIteratorB + ", "

        MmaPipelined_param_list += "ElementAccumulator0, layout::RowMajor, "

        for i in range(self.b2b_num - 1):
            epilogue_name = "EpilogueOutputOp" + str(i)
            MmaPipelined_param_list += epilogue_name + ", "

        for i in range(self.b2b_num - 1):
            epilogue_name = "FusedAddBiasEpilogue" + str(i)
            MmaPipelined_param_list += epilogue_name + ", "

        for i in range(self.b2b_num):
            MmaPolicy = "typename MmaCore" + str(i) + "::MmaPolicy"
            MmaPipelined_param_list += MmaPolicy + ", "
            
           
        cnt = 0
        for i in range(self.b2b_num):
            MmaStage = helper.var_idx("Stages", i)
            final = ", "
            if cnt == self.b2b_num - 1:
                final = ""
            MmaPipelined_param_list += MmaStage + final
            cnt += 1
        
        gen_code = code_using + " = " + gen_ir.gen_declare_template_struct(iterator_typename, MmaPipelined_param_list)

        return gen_code

      

    def gen_code(self):
        gen_using = ''
        # Generate default template struct
        gen_code = gen_ir.gen_template_struct(self.gen_class_name, self.template_param, "", speicalized = None, set_default=False)

        # Generate specialized template struct

        mmacore_codebody = self.gen_using_MmaCore(2)
        iterator_codebody = self.gen_using_Iterator()
        fragment_iterator_codebody = self.gen_fragment_iterator()
        epilogue_iterator_codebody = self.gen_using_FusedAddBiasEpilogue()
        threadBlockMma = self.gen_threadblockmma()
        specialized_code = mmacore_codebody + iterator_codebody + fragment_iterator_codebody + epilogue_iterator_codebody + threadBlockMma

        # Specialize layout C -> cutlass::layout::RowMajor

        rtn_template_args, speicalized_template_args = gen_ir.filtered_param(self.template_param, [ ('LayoutD', "cutlass::layout::RowMajor")], keep_= True)

        gen_speical_code = gen_ir.gen_template_struct(self.gen_class_name, rtn_template_args, specialized_code, speicalized = speicalized_template_args, set_default=False)
        code = gen_ir.gen_namespace("cutlass", gen_ir.gen_namespace("gemm", gen_ir.gen_namespace("threadblock", gen_code + gen_speical_code)))

        return self.gen_include_header() + code


class gen_b2b_mme_pipelined:
    def __init__(self, template_param, gen_class_name, b2b_num, cutlass_deps_root, project_root):
        self.gen_class_name = "B2bMmaPipelined"
        self.template_param = template_param
        self.b2b_num = b2b_num
        self.cutlass_deps_root = cutlass_deps_root
        self.project_root = project_root


    def gen_include_header(self):
        code = '''
#pragma once

#include \"{cutlass_dir}cutlass/cutlass.h\"
#include \"{cutlass_dir}cutlass/array.h\"
#include \"{cutlass_dir}cutlass/aligned_buffer.h\"
#include \"{cutlass_dir}cutlass/numeric_conversion.h\"

#include \"{cutlass_dir}cutlass/numeric_types.h\"
#include \"{cutlass_dir}cutlass/matrix_shape.h\"

#include \"{cutlass_dir}cutlass/gemm/gemm.h\"
#include \"{cutlass_dir}cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h\"

#include \"../threadblock/b2b_mma_base.h\"\n'''.format(cutlass_dir = self.cutlass_deps_root)
        return code


    def gen_using(self):
        code_using = "using FragmentA0 = typename IteratorA0::Fragment;\n"
        
        code_using += "using Base = B2bMmaBase<"
        for i in range(self.b2b_num):
            code_using += helper.var_idx("Shape", i) + "_, "
        for i in range(self.b2b_num):
            code_using += helper.var_idx("Policy", i) + "_, "
        for i in range(self.b2b_num):
            code_using += helper.var_idx("Stage", i) + "_, "
        code_using = code_using[: -2] + ">;\n"
            

        for i in range(self.b2b_num):
            code_using += helper.var_idx("using FragmentB", i) + helper.var_idx(" = typename IteratorB", i) + "::Fragment;\n"
            code_using += helper.var_idx("using FragmentC", i) + helper.var_idx(" = typename Policy", i) + "::Operator::FragmentC;\n"
            code_using += helper.var_idx("using Operator", i) + helper.var_idx(" = typename Policy", i) + "::Operator;\n"

        for i in range(self.b2b_num - 1):
            code_using += helper.var_idx("using IteratorC", i) + helper.var_idx(" = typename FusedAddBiasEpilogue", i) + "::OutputTileIterator;\n"

        code_using += "using ArchTag = typename Policy0::Operator::ArchTag;\n"
        code_using += "static ComplexTransform const kTransformA0 = Operator0::kTransformA;\n"

        for i in range(self.b2b_num):
            code_using += helper.var_idx("static ComplexTransform const kTransformB", i) + helper.var_idx(" = Operator", i) + "::kTransformB;\n"
        
        code_using += "private:\n"
        code_using += "using WarpFragmentA0 = typename Operator0::FragmentA;\n"
        code_using += "using WarpFragmentB0 = typename Operator0::FragmentB;\n"

        for i in range(1, self.b2b_num):
            code_using += helper.var_idx("using WarpFragmentA", i) + helper.var_idx(" = typename FragmentIteratorA", i) + "::Fragment;\n"
            code_using += helper.var_idx("using WarpFragmentB", i) + helper.var_idx(" = typename Operator", i) + "::FragmentB;\n"

        code_using += "protected:\n"
        
        code_using += "SmemIteratorA0 smem_iterator_A_;\n"
        
        for i in range(self.b2b_num):
            code_using += helper.var_idx("SmemIteratorB", i) +  helper.var_idx(" smem_iterator_B", i) + "_;\n"

        return code_using


    def gen_operator(self, first_use_1stage = False):
        code = ""
        def gen_operator_param(b2b_num):
            param_code = ""
            param_code += "int gemm_k_iterations_0,\n"
            param_code += helper.var_idx("FragmentC", b2b_num-1) +  helper.var_idx(" &accum", b2b_num-1) + ",\n"
            param_code += "IteratorA0 iterator_A,\n"

            for i in range(b2b_num):
                param_code += helper.var_idx("IteratorB", i) + " " + helper.var_idx("iterator_B", i) + ",\n"

            param_code += "FragmentC0 const &src_accum, \n"

            for i in range(b2b_num - 1):
                param_code += helper.var_idx("OutputOp", i) + " " + helper.var_idx("output_op_", i) + ",\n"
            for i in range(b2b_num - 1):
                param_code += helper.var_idx("FusedAddBiasEpilogue", i) + " " + helper.var_idx("epilogue_", i) + ",\n"
            for i in range(b2b_num - 1):
                param_code += helper.var_idx("IteratorC", i) + " " + helper.var_idx("iterator_C", i) + ",\n"


            param_code += "TransformA0 transform_A0 = TransformA0(), \n"

            for i in range(b2b_num):
                final = "(),\n"
                if i == b2b_num - 1:
                    final = "()\n"
                param_code += helper.var_idx("TransformB", i) + " " + helper.var_idx("transform_B", i) + " = " +helper.var_idx("TransformB", i) + final
            
            return param_code
        


        def gen_first_gemm_1stage(b2b_num):
            accu_code = "     FragmentC0 accum0 = src_accum;\n"
            if b2b_num == 1:
                accu_code = "    accum0 = src_accum;\n"
            
            code ="\
\n\
    FragmentA0 tb_frag_A;\n\
    FragmentB0 tb_frag_B0;\n\
\n\
    int smem_write_stage_idx = 1;\n\
\n\
    tb_frag_A.clear();\n\
    tb_frag_B0.clear();\n\
\n\
    // The last kblock is loaded in the prolog\n\
    iterator_A.load(tb_frag_A);\n\
    iterator_B0.load(tb_frag_B0);\n\
\n\
    ++iterator_A;\n\
    ++iterator_B0;\n\
\n\
    WarpFragmentA0 warp_frag_A0;\n\
    WarpFragmentB0 warp_frag_B0;\n\
\n\
    Operator0 warp_mma0;\n\
\n\
    // Avoid reading out of bounds\n\
    if (gemm_k_iterations_0 <= 1) {\n\
      iterator_A.clear_mask();\n\
      iterator_B0.clear_mask();\n\
    }\n\
\n\
    // Issue loads during the first warp-level matrix multiply-add *AFTER* issuing \n\
    // shared memory loads (which have the tightest latency requirement).\n\
\n\
    //\n\
    // Mainloop\n\
    //\n\
\n\
    // Note: The main loop does not support Base::WarpGemmIterations == 2.\n\
    CUTLASS_GEMM_LOOP\n\
    for (; gemm_k_iterations_0 > 0; --gemm_k_iterations_0) {\n\
\n\
      this->smem_iterator_A_.store(tb_frag_A);\n\
      this->smem_iterator_B0_.store(tb_frag_B0);\n\
\n\
      __syncthreads();\n\
      //\n\
      // Loop over GEMM K dimension\n\
      //\n\
\n\
      CUTLASS_PRAGMA_UNROLL\n\
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations0; ++warp_mma_k) {\n\
\n\
        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group\n\
        // as the case may be.\n\
\n\
        this->warp_tile_iterator_A0_.set_kgroup_index(warp_mma_k % Base::kWarpGemmIterations0);\n\
        this->warp_tile_iterator_B0_.set_kgroup_index(warp_mma_k % Base::kWarpGemmIterations0);\n\
\n\
        this->warp_tile_iterator_A0_.load(warp_frag_A0);\n\
        this->warp_tile_iterator_B0_.load(warp_frag_B0);\n\
\n\
        ++this->warp_tile_iterator_A0_;\n\
        ++this->warp_tile_iterator_B0_;\n\
\n\
        warp_mma0(accum0, warp_frag_A0, warp_frag_B0, accum0);\n\
      }\n\
      this->warp_tile_iterator_A0_.add_tile_offset({0, -Policy0::kPartitionsK * Base::kWarpGemmIterations0});\n\
      this->warp_tile_iterator_B0_.add_tile_offset({-Policy0::kPartitionsK * Base::kWarpGemmIterations0, 0});\n\
\n\
      __syncthreads();\n\
      iterator_A.load(tb_frag_A);\n\
      iterator_B0.load(tb_frag_B0);\n\
\n\
      ++iterator_A;\n\
      ++iterator_B0;\n\
\n\
      if(gemm_k_iterations_0 <= 2) {\n\
        iterator_A.clear_mask();\n\
        iterator_B0.clear_mask();\n\
      }\n\
    }\n"

            return accu_code + code


        def gen_first_gemm_2stage(b2b_num):
             
            accu_code = "     FragmentC0 accum0 = src_accum;\n"
            if b2b_num == 1:
                accu_code = "    accum0 = src_accum;\n"

            code ="\
\n\
    FragmentA0 tb_frag_A;\n\
    FragmentB0 tb_frag_B0;\n\
\n\
    tb_frag_A.clear();\n\
    tb_frag_B0.clear();\n\
\n\
    // The last kblock is loaded in the prolog\n\
    iterator_A.load(tb_frag_A);\n\
    iterator_B0.load(tb_frag_B0);\n\
\n\
    ++iterator_A;\n\
    ++iterator_B0;\n\
\n\
    this->smem_iterator_A_.store(tb_frag_A);\n\
    this->smem_iterator_B0_.store(tb_frag_B0);\n\
\n\
    ++this->smem_iterator_A_;\n\
    ++this->smem_iterator_B0_;\n\
\n\
    __syncthreads();\n\
\n\
    // Pair of fragments used to overlap shared memory loads and math instructions\n\
    WarpFragmentA0 warp_frag_A0[2];\n\
    WarpFragmentB0 warp_frag_B0[2];\n\
\n\
    this->warp_tile_iterator_A0_.set_kgroup_index(0);\n\
    this->warp_tile_iterator_B0_.set_kgroup_index(0);\n\
\n\
    this->warp_tile_iterator_A0_.load(warp_frag_A0[0]);\n\
    this->warp_tile_iterator_B0_.load(warp_frag_B0[0]);\n\
\n\
    ++this->warp_tile_iterator_A0_;\n\
    ++this->warp_tile_iterator_B0_;\n\
\n\
    Operator0 warp_mma0;\n\
\n\
    int smem_write_stage_idx = 1;\n\
\n\
    // Avoid reading out of bounds\n\
    if (gemm_k_iterations_0 <= 1) {\n\
      iterator_A.clear_mask();\n\
      iterator_B0.clear_mask();\n\
    }\n\
\n\
    // Issue loads during the first warp-level matrix multiply-add *AFTER* issuing \n\
    // shared memory loads (which have the tightest latency requirement).\n\
    iterator_A.load(tb_frag_A);\n\
\n\
    //\n\
    // Mainloop\n\
    //\n\
\n\
    // Note: The main loop does not support Base::WarpGemmIterations == 2.\n\
    CUTLASS_GEMM_LOOP\n\
    for (; gemm_k_iterations_0 > 0; --gemm_k_iterations_0) {\n\
\n\
      //\n\
      // Loop over GEMM K dimension\n\
      //\n\
\n\
      CUTLASS_PRAGMA_UNROLL\n\
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations0; ++warp_mma_k) {\n\
\n\
        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group\n\
        // as the case may be.\n\
\n\
        if (warp_mma_k == Base::kWarpGemmIterations0 - 1) {\n\
\n\
          // Write fragments to shared memory\n\
          this->smem_iterator_A_.store(tb_frag_A);\n\
\n\
          this->smem_iterator_B0_.store(tb_frag_B0);\n\
\n\
          __syncthreads();\n\
\n\
          // Issue loads during the first warp-level matrix multiply-add *AFTER* issuing \n\
          // shared memory loads (which have the tightest latency requirement).\n\
          iterator_A.load(tb_frag_A);\n\
          \n\
          ++this->smem_iterator_B0_;\n\
          ++this->smem_iterator_A_;\n\
        \n\
\n\
          // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory\n\
          if (smem_write_stage_idx == 1) {\n\
            this->smem_iterator_A_.add_tile_offset({0, -Base::Stage0});\n\
            this->smem_iterator_B0_.add_tile_offset({-Base::Stage0, 0});\n\
          }\n\
          else {\n\
            this->warp_tile_iterator_A0_.add_tile_offset(\n\
                {0, -Base::Stage0 * Policy0::kPartitionsK * Base::kWarpGemmIterations0});\n\
            this->warp_tile_iterator_B0_.add_tile_offset(\n\
                {-Base::Stage0 * Policy0::kPartitionsK * Base::kWarpGemmIterations0,\n\
                 0});\n\
          }\n\
\n\
          smem_write_stage_idx ^= 1;\n\
        }\n\
\n\
        this->warp_tile_iterator_A0_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations0);\n\
        this->warp_tile_iterator_B0_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations0);\n\
        \n\
        this->warp_tile_iterator_A0_.load(warp_frag_A0[(warp_mma_k + 1) % 2]);\n\
        this->warp_tile_iterator_B0_.load(warp_frag_B0[(warp_mma_k + 1) % 2]);\n\
\n\
        ++this->warp_tile_iterator_A0_;\n\
        ++this->warp_tile_iterator_B0_;\n\
\n\
        if (warp_mma_k == 0) {\n\
\n\
          iterator_B0.load(tb_frag_B0);\n\
\n\
          ++iterator_A;\n\
          ++iterator_B0;\n\
\n\
          // Avoid reading out of bounds if this was the last loop iteration\n\
          if (gemm_k_iterations_0 <= 2) {\n\
            iterator_A.clear_mask();\n\
            iterator_B0.clear_mask();\n\
          }\n\
        }\n\
\n\
        warp_mma0(accum0, warp_frag_A0[warp_mma_k % 2], warp_frag_B0[warp_mma_k % 2], accum0);\n\
      }\n\
    }\n"
            return accu_code + code

        def gen_other_gemms_2stage(b2b_num):
            
            code = ""
            
            def gemm_teamplate(id):
                code = "// " + str(id + 1) + " Gemm" 
                code += "    /// Iterator to load a warp-scoped tile of A1 operand from intermediate accumulator tile\n"
                
                code += "    " + helper.var_idx("FragmentC", id - 1) + helper.var_idx(" after_epilogue_accu", id - 1) + ";\n"
                code += "    " + helper.var_idx("epilogue_", id - 1) + helper.var_idx("(output_op_", id - 1) + helper.var_idx(", accum", id - 1) \
                               + helper.var_idx(", after_epilogue_accu", id - 1) + helper.var_idx(", iterator_C", id - 1) +");\n"
                
                #    FragmentIteratorA1 warp_tile_iterator_A1_(accum0); 
                code += "    " + helper.var_idx("FragmentIteratorA", id) + helper.var_idx(" warp_tile_iterator_A", id) +"_(" + helper.var_idx("after_epilogue_accu", id - 1) + ");\n"
                #    FragmentB1 tb_frag_B1;
                code += "    " +  helper.var_idx("FragmentB", id) + " " + helper.var_idx("tb_frag_B", id) + ";\n"
                #    tb_frag_B1.clear();
                code += "    " +  helper.var_idx("tb_frag_B", id)  + ".clear();\n"
                #    iterator_B1.load(tb_frag_B1);
                code += "    " + helper.var_idx("iterator_B", id) + ".load(" + helper.var_idx("tb_frag_B", id) + ");\n"
                #    ++iterator_B1;
                code += "    " +  "++" +  helper.var_idx("iterator_B", id) + ";\n"
                #    this->smem_iterator_B1_.store(tb_frag_B1);
                code += "    " +  helper.var_idx("this->smem_iterator_B", id) + "_.store(" + helper.var_idx("tb_frag_B", id) + ");\n"
                #    ++this->smem_iterator_B1_;
                code += "    " +  helper.var_idx("++this->smem_iterator_B", id) + "_;\n"
                #    __syncthreads();
                code += "    " +  "__syncthreads();\n"
                #    WarpFragmentA1 warp_frag_A1[2];
                code += "    " + helper.var_idx("WarpFragmentA", id) + helper.var_idx(" warp_frag_A", id) + "[2];\n"
                #    WarpFragmentB1 warp_frag_B1[2];
                code += "    " + helper.var_idx("WarpFragmentB", id) + helper.var_idx(" warp_frag_B", id) + "[2];\n"
                #    this->warp_tile_iterator_B1_.set_kgroup_index(0);
                code += "    " + helper.var_idx("this->warp_tile_iterator_B", id) + "_.set_kgroup_index(0);\n"
                #    warp_tile_iterator_A1_.load(warp_frag_A1[0], output_op_0);
                code += "    " + helper.var_idx("warp_tile_iterator_A", id) + helper.var_idx("_.load(warp_frag_A", id) + "[0]);\n"
                #    this->warp_tile_iterator_B1_.load(warp_frag_B1[0]);
                code += "    " + helper.var_idx("this->warp_tile_iterator_B", id) + helper.var_idx("_.load(warp_frag_B", id) + "[0]);\n"
                #    ++warp_tile_iterator_A1_;
                code +=  "    " + helper.var_idx("++warp_tile_iterator_A", id) + "_;\n"
                #    ++this->warp_tile_iterator_B1_;
                code +=  "    " + helper.var_idx("++this->warp_tile_iterator_B", id) + "_;\n"
                #    Operator1 warp_mma1;
                code +=  "    " + helper.var_idx("Operator", id) + " " + helper.var_idx("warp_mma", id) + ";\n"
                #    smem_write_stage_idx = 1;
                code +=  "    " + "smem_write_stage_idx = 1;\n"
                #    int gemm_k_iterations_1 = FragmentIteratorA1::Policy::kIterations / Base::kWarpGemmIterations1;
                code += "    " + helper.var_idx("int gemm_k_iterations_", id) + " = " + helper.var_idx("FragmentIteratorA", id) + helper.var_idx("::Policy::kIterations / Base::kWarpGemmIterations", id) +";\n"
                #    if (gemm_k_iterations_1 <= 1) {
                #      iterator_B1.clear_mask();
                #    }
                code += "    "  + "if ("  + helper.var_idx("gemm_k_iterations_", id) + " <= 1 ){\n" \
                    + "    "  + "    " + helper.var_idx("iterator_B", id) + ".clear_mask();\n" \
                    + "    "  +"}\n"
                #    CUTLASS_PRAGMA_UNROLL
                code += "    " + "CUTLASS_PRAGMA_UNROLL\n"
                #    for (; gemm_k_iterations_1 > 0; --gemm_k_iterations_1) {
                code += "    " + helper.var_idx("for (; gemm_k_iterations_", id) + helper.var_idx(" > 0; --gemm_k_iterations_", id) + ") {\n"
                #      CUTLASS_PRAGMA_UNROLL
                code += "    " + "    " + "CUTLASS_PRAGMA_UNROLL\n"
                #      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations1; ++warp_mma_k) {
                code += "    " + "    " + helper.var_idx("for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations", id) + "; ++warp_mma_k) {\n"
                #        if (warp_mma_k == Base::kWarpGemmIterations1 - 1) {
                code += "    " + "    " + "    " + helper.var_idx("if (warp_mma_k == Base::kWarpGemmIterations", id) + " - 1) {\n"
                #          this->smem_iterator_B1_.store(tb_frag_B1);
                code += "    " + "    " + "    " + "    " + helper.var_idx(" this->smem_iterator_B", id) + helper.var_idx("_.store(tb_frag_B", id) + ");\n"
                #          __syncthreads();
                code += "    " + "    " + "    " + "    " + "__syncthreads();\n"
                #          ++smem_iterator_B1_;
                code += "    " + "    " + "    " + "    " + helper.var_idx(" ++smem_iterator_B", id)  + "_;\n"
                #          if (smem_write_stage_idx == 1) {
                #            smem_iterator_B1_.add_tile_offset({-Base::Stage, 0});
                #          }
                code += "    " + "    " + "    " + "    "  + "if ( smem_write_stage_idx == 1 ) {\n" \
                    + "    " + "    " + "    " + "    " + "    " + helper.var_idx("smem_iterator_B", id) + helper.var_idx("_.add_tile_offset({-Base::Stage", i) + ", 0});\n" \
                    + "    " + "    " + "    " + "    "  +"}\n"
                #          else {
                #            this->warp_tile_iterator_B1_.add_tile_offset(
                #                {-Base::Stage * Policy1::kPartitionsK *
                #                     Base::kWarpGemmIterations1,
                #                 0});
                #          }
                code += "    " + "    " + "    " + "    "  + "else {\n" \
                    + "    " + "    " + "    " + "    " + "    " + helper.var_idx("this->warp_tile_iterator_B", id) + "_.add_tile_offset(\n" \
                    + "    " + "    " + "    " + "    " + "    " + helper.var_idx("{-Base::Stage", id) + helper.var_idx(" * Policy", id) + "::kPartitionsK *\n" \
                    + "    " + "    " + "    " + "    " + "    " + helper.var_idx("Base::kWarpGemmIterations", id) + ",\n" \
                    + "    " + "    " + "    " + "    " + "    " + "0});\n" \
                    + "    " + "    " + "    " + "    "  + "}\n"

                #          smem_write_stage_idx ^= 1;
                #        }
                code += "    " + "    " + "    " + "    "  + "smem_write_stage_idx ^= 1;\n" \
                    + "    " + "    " + "    " + "}\n"

                #        this->warp_tile_iterator_B1_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations1);
                code += "    " + "    " + "    " + helper.var_idx("this->warp_tile_iterator_B", id) + helper.var_idx("_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations", id) + ");\n"
                #        warp_tile_iterator_A1_.load(warp_frag_A1[(warp_mma_k + 1) % 2], output_op_0);
                code += "    " + "    " + "    " + helper.var_idx("warp_tile_iterator_A", id) + helper.var_idx("_.load(warp_frag_A", id) + "[(warp_mma_k + 1) % 2]);\n"
                #        this->warp_tile_iterator_B1_.load(warp_frag_B1[(warp_mma_k + 1) % 2]);
                code += "    " + "    " + "    " + helper.var_idx("this->warp_tile_iterator_B", id) + helper.var_idx("_.load(warp_frag_B", id) + "[(warp_mma_k + 1) % 2]);\n"
                #        ++warp_tile_iterator_A1_;
                code += "    " + "    " + "    " + helper.var_idx("++warp_tile_iterator_A", id) + "_;\n"
                #        ++this->warp_tile_iterator_B1_;
                code += "    " + "    " + "    " + helper.var_idx("++this->warp_tile_iterator_B", id) + "_;\n"
                #        if (warp_mma_k == 0) {
                #          iterator_B1.load(tb_frag_B1);
                #          ++iterator_B1;
                #          if (gemm_k_iterations_1 <= 2) {
                #            iterator_B1.clear_mask();
                #          }
                #        }
                code += "    " + "    " + "    " + " if (warp_mma_k == 0) {\n" \
                    + "    " + "    " + "    " + "    " + helper.var_idx("iterator_B", id) + helper.var_idx(".load(tb_frag_B", id) + ");\n" \
                    + "    " + "    " + "    " + "    " + helper.var_idx("++iterator_B", id) +";\n" \
                    + "    " + "    " + "    " + "    " + helper.var_idx("if (gemm_k_iterations_", id) +" <= 2) {\n" \
                    + "    " + "    " + "    " + "    " + "    " + helper.var_idx("iterator_B", id) + ".clear_mask();\n" \
                    + "    " + "    " + "    " + "    " + "}\n" \
                    + "    " + "    " + "    " + "}\n"
                #        warp_mma1(accum, warp_frag_A1[warp_mma_k % 2], warp_frag_B1[warp_mma_k % 2], accum);
                #      }
                #    }
                code += "    " + "    " + "    " + helper.var_idx("warp_mma", id) + helper.var_idx("(accum", id) + helper.var_idx(", warp_frag_A", id) + helper.var_idx("[warp_mma_k % 2], warp_frag_B", id) + helper.var_idx("[warp_mma_k % 2], accum", id) + ");\n" \
                    + "    " + "    " + "}\n" \
                    + "    " + "}\n\n\n"

                return code

            for i in range (1, b2b_num):
                clear_accu = ""
                if i != b2b_num - 1:
                    clear_accu = "    " + helper.var_idx("FragmentC", i) +  helper.var_idx(" accum", i) +";\n"
                    clear_accu += "    " + helper.var_idx("accum", i) +".clear();\n"
                code += clear_accu + gemm_teamplate(i)
            
            return code

        operator_code = " CUTLASS_DEVICE\n\
  void operator()(\n " + gen_operator_param(self.b2b_num) + ") {\n"
        if first_use_1stage:
            operator_code += gen_first_gemm_1stage(self.b2b_num)
        else:
            operator_code += gen_first_gemm_2stage(self.b2b_num)
        operator_code += gen_other_gemms_2stage(self.b2b_num) + "}\n"
        return operator_code

    def gen_construct_func(self):
        name = self.gen_class_name
        func_code = "CUTLASS_DEVICE\n"
        func_code += name + "(\n" \
                    + "    " + "typename Base::B2bMmaSharedStorage &shared_storage,\n" \
                    + "    " + "int thread_idx,\n" \
                    + "    " + "int warp_idx,\n" \
                    + "    " + "int lane_idx\n" \
                    + "):\n"
        func_code +=  "    " + "Base(shared_storage, thread_idx, warp_idx, lane_idx),\n" \
                    + "    " + "smem_iterator_A_(shared_storage.sharedStorage0.operand_A_ref(), thread_idx),\n"
        
        for i in range(self.b2b_num):
            final = ",\n"
            if i == self.b2b_num - 1:
                final = " {\n"
            func_code += helper.var_idx("smem_iterator_B", i) + helper.var_idx("_(shared_storage.sharedStorage", i) +".operand_B_ref(), thread_idx)" + final

        func_code +=  "    " + "int warp_idx_mn = warp_idx % (Base::WarpCount0::kM * Base::WarpCount0::kN);\n"
        func_code +=  "    " + "int warp_idx_k = warp_idx / (Base::WarpCount0::kM * Base::WarpCount0::kN);\n"

        func_code +=  "    " + "int warp_idx_m = warp_idx_mn % Base::WarpCount0::kM;\n"
        func_code +=  "    " + "int warp_idx_n = warp_idx_mn / Base::WarpCount0::kM;\n"

        for i in range(self.b2b_num):
            func_code +=  "    " + helper.var_idx("int tile_offset_k", i) + helper.var_idx(" = Base::kWarpGemmIterations", i) + " * warp_idx_k;\n"

        func_code +=  "    " + "this->warp_tile_iterator_A0_.add_tile_offset({warp_idx_m, tile_offset_k0});\n"

        for i in range(self.b2b_num):
            func_code +=  "    " + helper.var_idx("this->warp_tile_iterator_B", i) + helper.var_idx("_.add_tile_offset({tile_offset_k", i) + ", warp_idx_n});\n"

        func_code += "}\n"
        
        return func_code

    def gen_member_func(self, first_use_1stage):
        code = "public:\n"
        code += self.gen_operator(first_use_1stage)
        code += self.gen_construct_func()

        return code

    def gen_code(self, first_use_1stage):

        def gen_template_args(b2b_num):
            template_param = []
            template_param.append(("typename", "Shape0"))
            template_param.append(("typename", "IteratorA0"))
            template_param.append(("typename", "SmemIteratorA0"))
            template_param.append(("typename", "IteratorB0"))
            template_param.append(("typename", "SmemIteratorB0"))

            for i in range(1, b2b_num):
                template_param.append(("typename", helper.var_idx("Shape", i)))
                template_param.append(("typename", helper.var_idx("FragmentIteratorA", i)))
                template_param.append(("typename", helper.var_idx("IteratorB", i)))
                template_param.append(("typename", helper.var_idx("SmemIteratorB", i)))

            template_param.append(("typename", "ElementC"))
            template_param.append(("typename", "LayoutC"))

            for i in range(0, b2b_num - 1):
                template_param.append(("typename", helper.var_idx("OutputOp", i)))

            for i in range(0, b2b_num - 1):
                template_param.append(("typename", helper.var_idx("FusedAddBiasEpilogue", i)))
            
            for i in range(0, b2b_num):
                template_param.append(("typename", helper.var_idx("Policy", i)))
            for i in range(0, b2b_num):
                template_param.append((int, helper.var_idx("Stage", i)))

            template_param.append(("typename","TransformA0", "NumericArrayConverter<typename SmemIteratorA0_::Element, typename IteratorA0_::Element, IteratorA0_::Fragment::kElements>"))

            for i in range(0, b2b_num):
                cvtr = helper.var_idx("NumericArrayConverter<typename SmemIteratorB", i) + helper.var_idx("_::Element, typename IteratorB", i) + helper.var_idx("_::Element, IteratorB", i) + "_::Fragment::kElements>"
                template_param.append(("typename", helper.var_idx("TransformB", i), cvtr))

            template_param.append(("typename", "Enable", "bool"))

            return template_param

        template_param = gen_template_args(self.b2b_num)
        inheritance_code = "public B2bMmaBase<"
        for i in range(self.b2b_num):
            inheritance_code += helper.var_idx("Shape", i) + "_, "
        for i in range(self.b2b_num):
            inheritance_code += helper.var_idx("Policy", i) + "_, "
        for i in range(self.b2b_num - 1):
            inheritance_code += helper.var_idx("Stage", i) + "_, "
        inheritance_code += helper.var_idx("Stage", self.b2b_num - 1) + "_"
        inheritance_code += ">"

        code_body = ""
        using_code= self.gen_using()
        func_code = self.gen_member_func(first_use_1stage)

        code_body = using_code + func_code

        class_code = gen_ir.gen_template_class(self.gen_class_name, template_param, code_body, inheritance_code = inheritance_code)

        code = self.gen_include_header()
        code += gen_ir.gen_namespace("cutlass", gen_ir.gen_namespace("gemm", gen_ir.gen_namespace("threadblock", class_code)))
        # print(code)
        return code


class gen_b2b_mma_base:
    def __init__(self, template_param, gen_class_name, b2b_num, cutlass_deps_root, project_root):
        self.gen_class_name = gen_class_name
        self.template_param = template_param
        self.b2b_num = b2b_num
        self.cutlass_deps_root = cutlass_deps_root
        self.project_root = project_root

    def gen_include_header(self):
        code = '''
#pragma once

#include \"{cutlass_dirs}cutlass/aligned_buffer.h\"
#include \"{cutlass_dirs}cutlass/arch/memory.h\"
#include \"{cutlass_dirs}cutlass/array.h\"
#include \"{cutlass_dirs}cutlass/cutlass.h\"
#include \"{cutlass_dirs}cutlass/gemm/gemm.h\"
#include \"{cutlass_dirs}cutlass/matrix_shape.h\"
#include \"{cutlass_dirs}cutlass/numeric_types.h\"\n'''.format(cutlass_dirs=self.cutlass_deps_root)
        return code

    def gen_shared_storage(self):
        code = \
" template< \n\
    typename Shape_,\n\
    typename Policy_,\n\
    int ThisStage_\n\
>\n\
class SharedStorage {\n\
public:\n\
    using Shape = Shape_;\n\
    using Policy = Policy_;\n\
    static int const ThisStage = ThisStage_;\n\
    using Operator = typename Policy::Operator;\n\
    \
    using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;\n\
    \
    /// Tensor reference to the B operand \n\
    using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;\n\
\n\
    /// Shape of the A matrix operand in shared memory \n\
    using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,\n\
                               Shape::kK * ThisStage +\n\
                                   Policy::SmemPaddingA::kColumn>;\n\
\n\
    /// Shape of the B matrix operand in shared memory\n\
    using ShapeB =\n\
        MatrixShape<Shape::kK * ThisStage + Policy::SmemPaddingB::kRow,\n\
                    Shape::kN + Policy::SmemPaddingB::kColumn>;\n\
\n\
   public:\n\
\n\
    /// Buffer for A operand\n\
    AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;\n\
\n\
    /// Buffer for B operand\n\
    AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;\n\
\n\
   public:\n\
\n\
    /// Returns a layout object for the A matrix\n\
    CUTLASS_DEVICE\n\
    static typename Operator::LayoutA LayoutA() {\n\
      return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});\n\
    }\n\
\n\
    /// Returns a layout object for the B matrix\n\
    CUTLASS_HOST_DEVICE\n\
    static typename Operator::LayoutB LayoutB() {\n\
      return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});\n\
    }\n\
\n\
    /// Returns a TensorRef to the A operand\n\
    CUTLASS_HOST_DEVICE\n\
    TensorRefA operand_A_ref() {\n\
      return TensorRefA{operand_A.data(), LayoutA()};\n\
    }\n\
\n\
    /// Returns a TensorRef to the B operand\n\
    CUTLASS_HOST_DEVICE\n\
    TensorRefB operand_B_ref() {\n\
      return TensorRefB{operand_B.data(), LayoutB()};\n\
    }\n\
    CUTLASS_HOST_DEVICE\n\
    void * get_B_Shared_ptr() {\n\
      return operand_B.data();\n\
    }\n\
  };\n"
        return code

    def gen_using_and_misc(self, b2b_num):
        code_using = ""
        for i in range(b2b_num):
            code_using += "using Operator" +str(i) + " = typename Policy" + str(i) +"::Operator;\n"

        for i in range(b2b_num):
            code_using += "using WarpGemm" +str(i) + " = typename Policy" + str(i) +"::Operator::Shape;\n"

        for i in range(b2b_num):
            code_using += "using WarpCount" +str(i) + " = GemmShape<"   + helper.var_idx("Shape", i) +"::kM / " + helper.var_idx("WarpGemm", i) +"::kM, "\
                                                                        + helper.var_idx("Shape", i) +"::kN / " + helper.var_idx("WarpGemm", i) +"::kN, "\
                                                                        + helper.var_idx("Shape", i) +"::kK / " + helper.var_idx("WarpGemm", i) +"::kK>;\n"

        code_misc = ""
        for i in range(b2b_num):
            code_misc += "static int const " + helper.var_idx("kWarpGemmIterations", i) + " = (" + helper.var_idx("WarpGemm", i) + "::kK / " + helper.var_idx("Operator", i) +"::Policy::MmaShape::kK);\n"
     
        code = code_using + code_misc + self.gen_shared_storage()

        for i in range(b2b_num):
            code += "using " + helper.var_idx("SharedStorage", i) + " = SharedStorage<" + helper.var_idx("Shape", i) + ", " + helper.var_idx("Policy", i) +", " +  helper.var_idx("Stage", i) + ">;\n"

        def gen_union_shared_storage(b2b_num):
            code = ""
            for i in range(b2b_num):
                code += "    " +helper.var_idx("SharedStorage", i) + " " + helper.var_idx("sharedStorage", i) +";\n"
            return code

        code += "union B2bMmaSharedStorage {\n" + gen_union_shared_storage(self.b2b_num) + "};\n"

        for i in range(b2b_num - 1):
            code += helper.var_idx("void * C", i) + "_smm_ptr;\n"

        return code

    def gen_protected(self):
        code = "\nprotected:\n"
        code += "typename Operator0::IteratorA warp_tile_iterator_A0_;\n"
        for i in range(self.b2b_num):
            code += "typename Operator" +str(i) + "::IteratorB" +" warp_tile_iterator_B" + str(i) + "_;\n"
        return code

    def gen_public_member(self):
        code = "\npublic:\n"

        code += "CUTLASS_DEVICE\n"
        code += \
        "B2bMmaBase(\n" + \
        "    B2bMmaSharedStorage & shared_storage,\n" + \
        "    int thread_idx,\n" + \
        "    int warp_idx,\n" + \
        "    int lane_idx\n" + \
        "):\n" + \
        " warp_tile_iterator_A0_(shared_storage.sharedStorage0.operand_A_ref(), lane_idx),\n"
        for i in range(self.b2b_num):
            final = ",\n"
            if i == self.b2b_num-1:
                final = "\n"
            
            iterator = " warp_tile_iterator_B" + str(i) + "_"
            shared_storage = "shared_storage.sharedStorage" + str(i) + ".operand_B_ref()"
            code += iterator + "(" + shared_storage + ", lane_idx)" + final
        
        
        code += "{\n"
        for i in range(self.b2b_num - 1):
            code += helper.var_idx("    C", i) +  helper.var_idx("_smm_ptr = shared_storage.sharedStorage", i) + ".get_B_Shared_ptr();\n"
        code += "}\n"

        return code
        
    def gen_code(self):

        template_arg = []
        for i in range(self.b2b_num):
            template_arg.append(("typename", helper.var_idx("Shape", i)))
        for i in range(self.b2b_num):
            template_arg.append(("typename", helper.var_idx("Policy", i)))
        for i in range(self.b2b_num):
            template_arg.append((int, helper.var_idx("Stage", i)))
        
     

        code_body = self.gen_using_and_misc(self.b2b_num)
        code_body += self.gen_protected()
        code_body += self.gen_public_member()

        class_code = gen_ir.gen_template_class("B2bMmaBase", template_arg, code_body)

        code = self.gen_include_header() + gen_ir.gen_namespace("cutlass", gen_ir.gen_namespace("gemm", gen_ir.gen_namespace("threadblock", class_code)))

        return code


class gen_threadblock:
    def __init__(self, template_param, gen_class_name, b2b_num, output_dir, cutlass_deps_root, project_root):
        self.gen_class_name = gen_class_name
        self.template_param = template_param
        self.b2b_num = b2b_num
        self.file_dir = output_dir + "/threadblock/"

        self.cutlass_deps_root = cutlass_deps_root
        self.project_root = project_root


        self.gen_b2b_mma_base = gen_b2b_mma_base(template_param, gen_class_name, b2b_num, cutlass_deps_root, project_root)
        self.gen_b2b_mma_pipelined = gen_b2b_mme_pipelined(template_param, gen_class_name, b2b_num, cutlass_deps_root, project_root)
        self.gen_default_b2b_mma = gen_default_b2b_mma(template_param, gen_class_name, b2b_num, cutlass_deps_root, project_root)


    def gen_code(self, first_use_1stage):

        base_code = self.gen_b2b_mma_base.gen_code()
        print("[INFO]: Gen kernel code [b2b_mma_base.h]output Dir: is ", self.file_dir)

        with open(self.file_dir + "b2b_mma_base.h", "w+") as f:
            f.write(base_code)        
        pipeline_code = self.gen_b2b_mma_pipelined.gen_code(first_use_1stage = first_use_1stage)
        print("[INFO]: Gen kernel code [b2b_mma_pipelined.h]output Dir: is ", self.file_dir)

        with open(self.file_dir + "b2b_mma_pipelined.h", "w+") as f:
            f.write(pipeline_code)
        default_code = self.gen_default_b2b_mma.gen_code()
        print("[INFO]: Gen kernel code [default_b2b_mma.h]output Dir: is ", self.file_dir)

        with open(self.file_dir + "default_b2b_mma.h", "w+") as f:
            f.write(default_code)
