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

from typing import *

import helper
import gen_ir

import gen_kernel as gen_ker


class gen_device:
    def __init__(self, fuse_gemm_info, gen_class_name, user_header_file, cutlass_deps_root, project_root, output_dir = "../"):
        self.fuse_gemm_info = fuse_gemm_info
        self.raw_gemm_info = fuse_gemm_info
        self.b2b_num = len(fuse_gemm_info)
        self.user_header_file = user_header_file
        self.args = {}
        # device arg struct memebr
        self.arg_member = []
        self.gen_class_name = gen_class_name
        self.gen_kernel_name = gen_class_name + "Kernel"
        self.template_args = []
        self.__tempalate_arg_list = {'Stages': int, 'SplitKSerial': bool, 'IsBetaZero': bool, 'AlignmentA': int, 'AlignmentB': int}

        self.file_name = output_dir + "/device/" +gen_class_name +".h"
        self.sample_dir = output_dir


        self.cutlass_deps_root = cutlass_deps_root
        self.project_root = project_root
        self.this_file_root = output_dir + "/device/"

        self.first_use_1stage = False

        ## gen kernel
        self.gen_kernel = gen_ker.gen_kernel(self.template_args, self.gen_class_name, self.b2b_num, output_dir, cutlass_deps_root, project_root)


    def __check_arg_type(self, temp_arg):
        if temp_arg in self.__tempalate_arg_list.keys():
            return self.__tempalate_arg_list[temp_arg] 
        
        find_sub = False
        for candidate_arg in self.__tempalate_arg_list.keys():
            if (temp_arg.find(candidate_arg) != -1):
                return self.__tempalate_arg_list[candidate_arg] 
        
        return 'typename'

    # def gen_B2b2bGemm_class():
    def set_arch(self, sm_cap, mma_tp):
        if sm_cap == 75 or sm_cap == 80 or sm_cap == 86:
            self.arch = "cutlass::arch::Sm" + str(sm_cap)

        if mma_tp is 'hmma1688':
            self.mma_shape = [16, 8, 8]
            self.mma_tp = 'hmma'
        elif mma_tp is 'imma8816':
            self.mma_tp = 'imma'
            self.mma_shape = [8, 8, 16]
        else:
            return 0

    def gen_include_header(self):
        code = '''\
/* Auto Generated code - Do not edit.*/

#pragma once

#include \"{cutlass_root}cutlass/cutlass.h\"
#include \"{cutlass_root}cutlass/numeric_types.h\"
#include \"{cutlass_root}cutlass/arch/arch.h\"
#include \"{cutlass_root}cutlass/device_kernel.h\"

#include \"{cutlass_root}cutlass/gemm/threadblock/threadblock_swizzle.h\"

#include \"{cutlass_root}cutlass/gemm/device/default_gemm_configuration.h\"
#include \"{cutlass_root}cutlass/epilogue/thread/linear_combination_relu.h\"
#include \"{cutlass_root}cutlass/epilogue/thread/linear_combination.h\"

#include \"{project_root}../kernel/b2b_gemm.h\"
#include \"{project_root}../kernel/default_b2b_gemm.h\"
'''.format(cutlass_root=self.cutlass_deps_root, project_root=self.project_root, this_file_root=self.this_file_root)
        include_user_header = ""
        for header in self.user_header_file:
            include_user_header += "#include \"" + header + "\"\n"
        return code + include_user_header

    def gen_code(self, sm_cap, mma_tp, ifprint = True):
        self.set_arch(sm_cap, mma_tp)

        self.update_b2b_args()
        print(self.fuse_gemm_info)
        self.update_b2b_class_template_args()

        func_code = self.gen_all_func()
        member_var_code = "private:\n typename B2bGemmKernel::Params params_;\n"

        gen_code = gen_ir.gen_template_class(self.gen_class_name, self.template_args, func_code + member_var_code)
        code = self.gen_include_header() + gen_ir.gen_namespace("cutlass", gen_ir.gen_namespace("gemm", gen_ir.gen_namespace("device", gen_code)))

        if ifprint:
            print(code)

        print("[INFO]: Gen device code output Dir: is ", self.file_name)
        with open(self.file_name, 'w+') as f:
            f.write(code)


        gen_kernel = self.gen_kernel.gen_code(self.first_use_1stage)
        print(gen_kernel)

    def update_b2b_class_template_args(self):
        for arg in self.args.keys():
            self.template_args.append([self.__check_arg_type(arg), arg, self.args[arg]])

    def update_b2b_args(self):

        self.args['ElementA'] = helper.type_2_cutlass_type(self.fuse_gemm_info[0]['A_tp'])
        self.args['LayoutA'] = helper.type_2_cutlass_type(self.fuse_gemm_info[0]['A_format'])

        cnt = 0

        warp_M_tile = 32

        # Determine maximum N_tile
        Max_Ntile = 0
        for layer in self.fuse_gemm_info:
            n_tile = layer['mnk'][1]
            if n_tile > Max_Ntile:
                Max_Ntile = n_tile
        if Max_Ntile >= 256:
            warp_M_tile = 16

        stages_temp = []

        for layer in self.fuse_gemm_info:
            cnt_str = str(cnt)
            B_tp_str= 'ElementB' + cnt_str
            B_format_str = 'LayoutB' + cnt_str
            C_tp_str= 'ElementC' + cnt_str
            C_format_str = 'LayoutC' + cnt_str
            Acc_str = 'ElementAccumulator' + cnt_str

            self.args[B_tp_str] = helper.type_2_cutlass_type(layer['B_tp'])
            self.args[B_format_str] = helper.type_2_cutlass_type(layer['B_format'])
            self.args[C_tp_str] = helper.type_2_cutlass_type(layer['C_tp'])
            self.args[C_format_str] = helper.type_2_cutlass_type(layer['C_format'])
            self.args[Acc_str] = helper.type_2_cutlass_type(layer['Acc_tp'])
            

            mnk = layer['mnk'][:]

            tile_mnk = mnk[:]            

            tile_mnk[2] = 32 # force the ktile is 32

            #N tile gen
            if mnk[1] > 1024:
                assert(0)
            elif mnk[1] > 512:
                tile_mnk[1] = 1024
            elif mnk[1] > 256:
                tile_mnk[1] = 512
            elif mnk[1] > 128:
                tile_mnk[1] = 256
            elif mnk[1] > 64:
                tile_mnk[1] = 128
            elif mnk[1] > 32:
                tile_mnk[1] = 64
            else : 
                tile_mnk[1] = 32

            if tile_mnk[1] == 512:
                stages_temp.append(1)
            else:
                stages_temp.append(2)

            tile_mnk[0] = 4 * warp_M_tile



            epilogue_setted_type = helper.get_epilogue_tp(layer)
            cutlass_epilogue_name = "LinearCombinationRelu"
            if epilogue_setted_type.lower() == 'leakyrelu':
                cutlass_epilogue_name = "LinearCombinationLeakyRelu"
            elif epilogue_setted_type.lower() == 'identity':
                cutlass_epilogue_name = "LinearCombination"

            epilogue_str = 'EpilogueOutputOp' + cnt_str
            if cnt != len(self.fuse_gemm_info) - 1:
                n = layer['mnk'][1]
                Fragments = tile_mnk[1] // 8 * 2
                self.args[epilogue_str] = "cutlass::epilogue::thread::" + cutlass_epilogue_name + "<ElementC0_, " + str(Fragments) +", ElementAccumulator0_, ElementAccumulator0_>"
            else:
                n = layer['mnk'][1]
                n_mod_8 = n % 4
                N_align_elements = 1
                if n_mod_8 == 0:
                    N_align_elements = 8
                elif n_mod_8 == 4:
                    N_align_elements = 4
                elif n_mod_8 == 2 or n_mod_8 == 6:
                    N_align_elements = 2

                self.args[epilogue_str] = "cutlass::epilogue::thread::" + cutlass_epilogue_name+ "<ElementC0_, " + str(N_align_elements) + ", ElementAccumulator0_, ElementAccumulator0_>"

            

            ThreadBlockShape_str = 'ThreadblockShape' + cnt_str

            self.args[ThreadBlockShape_str] = helper.cvt_2_cutlass_shape(tile_mnk)

            WarpShape_str = 'WarpShape' + cnt_str
            tile_mnk[0] = warp_M_tile
            self.args[WarpShape_str] = helper.cvt_2_cutlass_shape(tile_mnk)
            cnt += 1


        self.args['ElementD'] = helper.type_2_cutlass_type(self.fuse_gemm_info[self.b2b_num - 1]['C_tp'])
        self.args['LayoutD'] = helper.type_2_cutlass_type(self.fuse_gemm_info[self.b2b_num - 1]['C_format'])
        
        self.args['InstructionShape'] = helper.cvt_2_cutlass_shape(self.mma_shape)
        self.args['OperatorClass'] = 'arch::OpClassTensorOp'
        self.args['ArchTag'] = self.arch
        self.args['ThreadblockSwizzle'] = 'threadblock::GemmBatchedIdentityThreadblockSwizzle'
        

        for i in range(self.b2b_num):
            self.args[helper.var_idx('Stages', i)] = "2"

        self.args['AlignmentA'] = str(8)
        self.args['AlignmentB'] = str(8)
        self.args['SplitKSerial'] = 'false'
        self.args['Operator'] = 'typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB0_, ElementC0_, ElementAccumulator0_>::Operator'
        self.args['IsBetaZero'] = 'false'


    def gen_using_kernel(self):
        code = "using B2bGemmKernel = typename kernel::DefaultB2bGemm<\n"
        code += "    " + "ElementA,\n"
        code += "    " + "LayoutA,\n"

        for i in range(self.b2b_num):
            code += "    " + helper.var_idx("ElementB", i) + ",\n"
            code += "    " + helper.var_idx("LayoutB", i) + ",\n"
            code += "    " + helper.var_idx("ElementC", i) + ",\n"
            code += "    " + helper.var_idx("LayoutC", i) + ",\n"
            code += "    " + helper.var_idx("ElementAccumulator", i) + ",\n"
            code += "    " + helper.var_idx("EpilogueOutputOp", i) + ",\n"
            code += "    " + helper.var_idx("ThreadblockShape", i) + ",\n"
            code += "    " + helper.var_idx("WarpShape", i) + ",\n"

        code +=  "    " + "ElementD,\n"
        code +=  "    " + "LayoutD,\n"
        code +=  "    " + "InstructionShape,\n"
        code +=  "    " + "OperatorClass,\n"
        code +=  "    " + "ArchTag,\n"
        code +=  "    " + "ThreadblockSwizzle,\n"

        for i in range(self.b2b_num):
            code +=  "    " + helper.var_idx("Stages", i) + ",\n"


        code +=  "    " + "AlignmentA,\n"
        code +=  "    " + "AlignmentB,\n"
        code +=  "    " + "SplitKSerial,\n"
        code +=  "    " + "Operator,\n"
        code +=  "    " + "IsBetaZero_\n"

        code += ">::B2bGemmKernel;\n\n"

        return code
    
    def gen_args(self):

        def gen_arg_member(b2b_num):
            data_members = []

            for i in range(b2b_num):
                member_type = "GemmCoord"
                member_name = "problem_size_" + str(i)
                data_members.append((member_type, member_name))

            member_type = "TensorRef<ElementA const, LayoutA>"
            member_name = "ref_A0"
            data_members.append((member_type, member_name))
            
            for i in range(b2b_num):
                member_type = "TensorRef<ElementB" + str(i) + " const, LayoutB" + str(i) +">"
                member_name = "ref_B" + str(i)
                data_members.append((member_type, member_name))
                member_type = "TensorRef<ElementC" + str(i) + " const, LayoutC" + str(i) +">"
                member_name = "ref_C" + str(i)
                data_members.append((member_type, member_name))
            
            member_type = "TensorRef<ElementD, LayoutD>"
            member_name = helper.var_idx("ref_D", b2b_num - 1)
            data_members.append((member_type, member_name))

            for i in range(b2b_num):
                member_type = "typename EpilogueOutputOp" + str(i) + "::Params"
                member_name = "epilogue" + str(i)
                data_members.append((member_type, member_name))

            data_members.append(('int', 'batch_count'))

            return data_members
        
        def gen_arg_struct_default_ctor(struct_name, data_members, inital_param_num, inital_value):
            constructs_code = gen_ir.indentation + "CUTLASS_HOST_DEVICE\n" + \
                              gen_ir.indentation + struct_name + " (): "
            for i in range(inital_param_num):
                final_param = ','
                if i == inital_param_num - 1:
                    final_param = '{ }'
                constructs_code +=  data_members[i][1] + inital_value + final_param

            constructs_code += "\n"
            return constructs_code

        def gen_arg_struct_ctor(struct_name, data_members):
            constructs_code = gen_ir.indentation + "CUTLASS_HOST_DEVICE\n" + \
                              gen_ir.indentation + struct_name + " (\n"
            cnt = 0
            param_num = len(data_members)
            for param in data_members:
                final = ',\n'
                if cnt == param_num - 1:
                    final = '\n):\n'
                constructs_code +=  gen_ir.indentation + param[0] + " " + param[1] + "_" + final
                cnt += 1

            cnt = 0
            for param in data_members:
                final = '),\n'
                if cnt == param_num - 1:
                    final = ") { }\n"
                constructs_code +=  gen_ir.indentation + param[1] + "(" + param[1] + "_" + final
                cnt += 1

            constructs_code += "\n"
            return constructs_code    

        # (variable type, variable name)
        struct_member = gen_arg_member(self.b2b_num)
        self.arg_member = struct_member

        codeBody = ""
        for each_member in struct_member:
            codeBody += gen_ir.indentation + each_member[0] + " " + each_member[1] + ";\n"

        codeBody += gen_arg_struct_default_ctor("Arguments", struct_member, self.b2b_num, "(0,0,0)") + "\n"
        codeBody += gen_arg_struct_ctor("Arguments", struct_member) + "\n"
        struct_code = gen_ir.gen_struct("Arguments", codeBody)
        return struct_code

    def gen_func_constructs(self):
        code = self.gen_class_name +"() {}"
        return code

    def gen_func_initialize(self):
        code = "Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {\n" + \
                "// Determine grid shape\n" + \
                "ThreadblockSwizzle threadblock_swizzle;\n" + \
                "cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(\n" + \
                "  args.problem_size_0, \n" + \
                "  { ThreadblockShape0::kM, ThreadblockShape0::kN, ThreadblockShape0::kK },\n" + \
                "  args.batch_count);\n" + \
                "// Initialize the Params structure\n" + \
                "params_ = typename B2bGemmKernel::Params{\n"
        for i in range(self.b2b_num):
            code += helper.var_idx("  args.problem_size_", i) + ",\n"
        code += "  grid_shape,\n" + \
                "  args.ref_A0.non_const_ref(),\n"
        for i in range(self.b2b_num):
            code += helper.var_idx("  args.ref_B", i) + ".non_const_ref(),\n"
            code += helper.var_idx("  args.ref_C", i) + ".non_const_ref(),\n"

        code += helper.var_idx("  args.ref_D", self.b2b_num - 1) + ",\n"
        for i in range(self.b2b_num):
            code += helper.var_idx("  args.epilogue", i) + ",\n"

        code += "  args.batch_count\n"
        code += "};\n" + \
                "return Status::kSuccess;\n" + \
                "}\n"
        return code 

    def gen_func_run(self):
        code = "Status run(cudaStream_t stream = nullptr) {\n" + \
                "\n" + \
                "  ThreadblockSwizzle threadblock_swizzle;\n" + \
                "\n" + \
                "  dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);\n" + \
                "  dim3 block(B2bGemmKernel::kThreadCount, 1, 1);\n" + \
                "\n" + \
                "  cudaError_t result;\n" + \
                "\n" + \
                "  int smem_size = int(sizeof(typename B2bGemmKernel::SharedStorage));\n" + \
                "  if (smem_size >= (48 << 10)) {\n" + \
                "    result = cudaFuncSetAttribute(Kernel<B2bGemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);\n" + \
                "\n" + \
                "    if (result != cudaSuccess) {\n" + \
                "      return Status::kErrorInternal;\n" + \
                "    }\n" + \
                "  }\n" + \
                "  cutlass::Kernel<B2bGemmKernel><<<grid, block, smem_size, stream>>>(params_);\n" + \
                "  result = cudaGetLastError();\n" + \
                "  return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;\n" + \
                "  }\n"
        
        return code
    def gen_func_operator(self):
        opeartor_with_arg_code = "Status operator()(\n" + \
                                "  Arguments const &args,\n" + \
                                "  void *workspace = nullptr,\n" + \
                                "  cudaStream_t stream = nullptr) {\n" + \
                                "  Status status = initialize(args, workspace);\n" + \
                                "  \n" + \
                                "  if (status == Status::kSuccess) {\n" + \
                                "    status = run(stream);\n" + \
                                "  }\n" + \
                                "  return status;\n" + \
                                "}\n"
        operator_code = "Status operator()(\n" + \
                        "  cudaStream_t stream = nullptr) {\n" + \
                        "   Status status = run(stream);\n" + \
                        "   return status;\n" + \
                        "}\n"
        return opeartor_with_arg_code + "\n" + operator_code

    def gen_all_func(self):
        return  self.gen_using_kernel() + "\n" + \
                self.gen_args() + "\n" + \
                self.gen_func_constructs()  + "\n" + \
                self.gen_func_initialize() + "\n" + \
                self.gen_func_run() + "\n" + \
                self.gen_func_operator()
