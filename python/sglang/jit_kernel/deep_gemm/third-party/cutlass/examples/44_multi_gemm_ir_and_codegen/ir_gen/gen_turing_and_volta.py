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

import helper
import gen_ir as ir

class gen_turing_impl:
    def __init__(self,fuse_gemm_info, gen_class_name, user_header_file, output_dir = "../"):
        self.fuse_gemm_info = fuse_gemm_info
        self.class_name = gen_class_name
        self.gen_class_name = gen_class_name + "_turing_impl"
        self.user_header_file = ""
        for header in user_header_file: 
            self.user_header_file += "#include \"" + header + "\"\n"
        self.output_dir = output_dir
        self.b2b_num = len(fuse_gemm_info)

        self.gen_turing_unfused = gen_volta_turing_fuse_act_impl(fuse_gemm_info, gen_class_name, user_header_file, output_dir)

    def gen_using(self):
        code_using = "using b2b_gemm = typename cutlass::gemm::device::" + self.class_name + "<cutlass::half_t>;"
        
        return code_using + "\n"

    def gen_initialize(self):
        code = ""
        for i in range(self.b2b_num):
            code_this = ""

            code_this += helper.var_idx(helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp']) + " alpha", i) + " = " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp']) + "(1);\n"
            beta = "(1)"
            
            if helper.get_epilogue_add_bias_or_not(self.fuse_gemm_info[i]) is False:
                beta = "(0)"
            code_this += helper.var_idx(helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp']) + " beta", i) + " = " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp']) + beta + ";\n"
            k_str = str(self.fuse_gemm_info[i]['mnk'][2])
            if i == 0:
                k_str = "K0"
            code_this += helper.var_idx("cutlass::gemm::GemmCoord problem_size_", i) + "(M, " + str(self.fuse_gemm_info[i]['mnk'][1]) + ", " + k_str + ");\n" 
            code += code_this
        code += "typename b2b_gemm::Arguments arguments{\n"

        for i in range(self.b2b_num):
            code += "    " + helper.var_idx("problem_size_", i) + ",\n"


        code += "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['A_tp']) + "*>(" + helper.var_idx("A", 0) + "), " + helper.var_idx("problem_size_", 0) + ".k()},\n"

        for i in range(self.b2b_num):
            
            ldmB = str(self.fuse_gemm_info[i]['mnk'][2])
            if i == 0:
                ldmB = "K0"

            if self.fuse_gemm_info[i]['B_format'] is 'Row':
                ldmB = str(self.fuse_gemm_info[i]['mnk'][1])
            
            ldmC = str(helper.get_epilogue_bias_ldm(self.fuse_gemm_info[i]))

            code += "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['B_tp']) + "*>(" + helper.var_idx("B", i) + "), " + ldmB + "},\n"
            code += "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['C_tp']) + "*>(" + helper.var_idx("C", i) + "), " + ldmC + "},\n"
        code += "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['C_tp']) + "*>(" + helper.var_idx("D", self.b2b_num -1) + "), " + helper.var_idx("problem_size_", self.b2b_num - 1) + ".n()},\n"


        for i in range(self.b2b_num):
            code += "    " + "{ " + helper.var_idx("alpha", i) + ", " + helper.var_idx("beta", i) 
            for epilogue_arg in  helper.get_epilogue_args(self.fuse_gemm_info[i]):
                arg_name = helper.var_idx("Epilogue", i) + "_" +  epilogue_arg[1]
                code += ", " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp']) + "(" + str(arg_name) + ")"
            code += "},\n"
        code += "    " + "Batch};\n\n"

        code += "    " "b2b_gemm gemm_op;\n"
        code += "    " + "gemm_op.initialize(arguments);\n"
        return code + "\n"
        


    def gen_run(self):
        code = "    " + "gemm_op(stream);\n"

        return code

    def gen_wrapper(self):
        code_body = ""

        arg_lists = []
        arg_lists.append(["int", "M"])
        arg_lists.append(["int", "K0"])
        arg_lists.append(["int", "Batch"])
        arg_lists.append(["void*", helper.var_idx("A", 0)])
        for i in range(self.b2b_num):
            arg_lists.append(["void*", helper.var_idx("B", i)])
            arg_lists.append(["void*", helper.var_idx("C", i)])
            arg_lists.append(["void*", helper.var_idx("D", i)])
            epilogue_args = helper.get_epilogue_args(self.fuse_gemm_info[i])
            acc_tp = helper.get_epilogue_compute_tp(self.fuse_gemm_info[i])
            for arg in epilogue_args:
                arg_tp = arg[0]
                arg_name = helper.var_idx("Epilogue", i) + "_" +  arg[1]
                arg_lists.append([arg_tp, arg_name])
        
        if self.b2b_num == 1:
            code_body += self.gen_turing_unfused.gen_using(False)  #False -> Turing, True -> Volta
            code_body += self.gen_turing_unfused.gen_initialize()
            code_body += self.gen_turing_unfused.gen_run()
        else:
            code_body += self.gen_using()
            code_body += self.gen_initialize()
            code_body += self.gen_run()

        code = ir.gen_func(self.gen_class_name, arg_lists, code_body)

        return code 

    def gen_code(self):

        code = self.gen_wrapper()
        helper.write_2_headfile("turing_impl.h", self.output_dir, self.user_header_file + "\n" + code)

class gen_volta_turing_fuse_act_impl:
    def __init__(self, fuse_gemm_info, gen_class_name, user_header_file, output_dir = "../"):
        self.fuse_gemm_info = fuse_gemm_info
        self.gen_class_name = gen_class_name + "_volta_impl"
        self.user_header_file = ""
        for header in user_header_file: 
            self.user_header_file +=  "#include \"" + header + "\"\n"
        self.output_dir = output_dir
        self.b2b_num = len(fuse_gemm_info)

    def perf_tiling(self, layer_mnk):
        mnk = layer_mnk[:]
        block_tile = mnk[:]  
        block_tile[2] = 32 # force the K tile to be 32

        # M tile gen
        block_tile[0] = 32

        # N tile gen
        if mnk[1] > 128:
            block_tile[1] = 256
        elif mnk[1] > 64:
            block_tile[1] = 128
        elif mnk[1] > 32:
            block_tile[1] = 64
        else : 
            block_tile[1] = 32
        
        warp_tile = block_tile[:]          
        if block_tile[1] == 256:
            warp_tile[1] = 64
        elif block_tile[1] == 128:
            warp_tile[1] = 32
        elif block_tile[1] == 64:
            warp_tile[1] = 32
        else :
            warp_tile[1] = 32

        warp_tile[0] = 32

        return block_tile, warp_tile


    def process_epilogue(self, epilogue_tp, n, C_tp, Acc_tp):
        epilogue_setted_type = epilogue_tp
        cutlass_epilogue_name = "LinearCombinationRelu"
        if epilogue_setted_type.lower() == 'leakyrelu':
            cutlass_epilogue_name = "LinearCombinationLeakyRelu"
        elif epilogue_setted_type.lower() == 'identity':
            cutlass_epilogue_name = "LinearCombination"


        n_mod_8 = n % 4
        N_align_elements = 1
        if n_mod_8 == 0:
            N_align_elements = 8
        elif n_mod_8 == 4:
            N_align_elements = 4
        elif n_mod_8 == 2 or n_mod_8 == 6:
            N_align_elements = 2

        epilogue_str = "cutlass::epilogue::thread::" + cutlass_epilogue_name+ "<" + C_tp + ", " + str(N_align_elements) + ", " + Acc_tp + ", " + Acc_tp + ">"

        return epilogue_str

    def gen_using(self, volta = True):
        code_using = ""
        volta_arch = "cutlass::arch::Sm70"
        volta_tc = "cutlass::gemm::GemmShape<8, 8, 4>"

        turing_arch = "cutlass::arch::Sm75"
        turing_tc = "cutlass::gemm::GemmShape<16, 8, 8>"

        arch = ""
        tc = ""
        if volta:
            arch = volta_arch
            tc = volta_tc
        else:
            arch = turing_arch
            tc = turing_tc

        for i in range(self.b2b_num):
            
            k = self.fuse_gemm_info[i]['mnk'][2]

            k_mod_8 = k % 4
            ab_ldm = 1
            if k_mod_8 == 0:
                ab_ldm = 8
            elif k_mod_8 == 4:
                ab_ldm = 4
            elif k_mod_8 == 2 or k_mod_8 == 6:
                ab_ldm = 2

            block_tile, warp_tile = self.perf_tiling(self.fuse_gemm_info[i]['mnk'])

            this_gemm_config =  helper.var_idx("using Gemm", i) + " = cutlass::gemm::device::GemmBatched<\n"
            this_gemm_config += "    " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['A_tp']) + ",\n"
            this_gemm_config += "    " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['A_format']) + ",\n"
            this_gemm_config += "    " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['B_tp']) + ",\n"
            this_gemm_config += "    " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['B_format']) + ",\n"
            this_gemm_config += "    " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['C_tp']) + ",\n"
            this_gemm_config += "    " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['C_format']) + ",\n"
            this_gemm_config += "    " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp']) + ",\n"
            this_gemm_config += "    " + "cutlass::arch::OpClassTensorOp,\n"
            this_gemm_config += "    " + arch + ",\n"
            this_gemm_config += "    " + "cutlass::gemm::GemmShape<" + str(block_tile[0]) + ", " + str(block_tile[1]) + ", " + str(block_tile[2]) + ">,\n"
            this_gemm_config += "    " + "cutlass::gemm::GemmShape<" + str(warp_tile[0]) + ", " + str(warp_tile[1]) + ", " + str(warp_tile[2]) + ">,\n"
            this_gemm_config += "    " + tc + ",\n"
            this_gemm_config += "    " + self.process_epilogue(helper.get_epilogue_tp(self.fuse_gemm_info[i]), self.fuse_gemm_info[i]['mnk'][1], helper.type_2_cutlass_type(self.fuse_gemm_info[i]['C_tp']), helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp'])) + ",\n"
            this_gemm_config += "    " + "cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,\n"
            this_gemm_config += "    " + "2,\n"
            this_gemm_config += "    " + str(ab_ldm) + ",\n"
            this_gemm_config += "    " + str(ab_ldm) + ">;\n"

            code_using += this_gemm_config + "\n"

        return code_using + "\n"

    def gen_initialize(self):
        code = ""
        for i in range(self.b2b_num):
            code_this = ""

            N_str = str(self.fuse_gemm_info[i]['mnk'][1])

            code_this += helper.var_idx(helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp']) + " alpha", i) + " = " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp']) + "(1);\n"
            beta = "(1)"
            if helper.get_epilogue_add_bias_or_not( self.fuse_gemm_info[i]) is False:
                beta = "(0)"
            code_this += helper.var_idx(helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp']) + " beta", i) + " = " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp']) + beta + ";\n"

            k_str = str(self.fuse_gemm_info[i]['mnk'][2])
            if i == 0:
                k_str = "K0"
            code_this += helper.var_idx("cutlass::gemm::GemmCoord problem_size_", i) + "(M, " + str(self.fuse_gemm_info[i]['mnk'][1]) + ", " + k_str + ");\n" 
            code_this += helper.var_idx("typename Gemm", i) + helper.var_idx("::Arguments arguments_", i) + "{\n"
            code_this += "    " + helper.var_idx("problem_size_", i) + ",\n"
            ldmA = k_str
            ldmB = k_str
            ldmC = str(self.fuse_gemm_info[i]['mnk'][1])

            ldmBias = str(helper.get_epilogue_bias_ldm(self.fuse_gemm_info[i]))

            if self.fuse_gemm_info[i]['A_format'] is 'Col':
                ldmA = "M"
            if self.fuse_gemm_info[i]['B_format'] is 'Row':
                ldmB = str(self.fuse_gemm_info[i]['mnk'][1])
            if self.fuse_gemm_info[i]['C_format'] is 'Col':
                ldmC = "M"

            if i == 0:
                code_this += "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['A_tp']) + "*>(" + helper.var_idx("A", i) + "), " + ldmA + "}, " + "M * " + ldmA + ",\n"
            else:
                code_this += "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['A_tp']) + "*>(" + helper.var_idx("D", i - 1) + "), " + ldmA + "}, " + "M * " + ldmA + ",\n"

            code_this += "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['B_tp']) + "*>(" + helper.var_idx("B", i) + "), " + ldmB + "}, " + N_str + " * " + ldmB + ",\n"
            
            M_bias = str(helper.get_epilogue_bias_shape(self.fuse_gemm_info[i])[0])

            code_this += "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['C_tp']) + "*>(" + helper.var_idx("C", i) + "), " + ldmBias + "}, " + M_bias + " * " + N_str + ",\n"
            code_this += "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['C_tp']) + "*>(" + helper.var_idx("D", i) + "), " + ldmC + "}, " + "M * " + ldmC + ",\n"
            code_this += "    " + "{ " + helper.var_idx("alpha", i) + ", " + helper.var_idx("beta", i) 
            for epilogue_arg in  helper.get_epilogue_args(self.fuse_gemm_info[i]):
                arg_name = helper.var_idx("Epilogue", i) + "_" +  epilogue_arg[1]
                code_this += ", " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp']) + "(" + str(arg_name) + ")"
            code_this += " },\n"
            code_this += "    " + "Batch};\n"

            code_this += "    " + helper.var_idx("Gemm", i) + helper.var_idx(" gemm_op_", i) + ";\n"
            code_this += "    " + helper.var_idx("gemm_op_", i) + helper.var_idx(".initialize(arguments_", i) + ", nullptr);\n"

            code += code_this + "\n"
        return code + "\n"
        

    def gen_run(self):
        code = ""
        for i in range(self.b2b_num):
            code_this = ""
            code_this += "    " + helper.var_idx("gemm_op_", i) + "(stream);\n"

            code += code_this 
        return code

    def gen_wrapper(self):
        code_body = ""

        arg_lists = []
        arg_lists.append(["int", "M"])
        arg_lists.append(["int", "K0"])
        arg_lists.append(["int", "Batch"])
        arg_lists.append(["void*", helper.var_idx("A", 0)])
        for i in range(self.b2b_num):
            arg_lists.append(["void*", helper.var_idx("B", i)])
            arg_lists.append(["void*", helper.var_idx("C", i)])
            arg_lists.append(["void*", helper.var_idx("D", i)])
            epilogue_args = helper.get_epilogue_args(self.fuse_gemm_info[i])
            acc_tp = helper.get_epilogue_compute_tp(self.fuse_gemm_info[i])
            for arg in epilogue_args:
                arg_tp = arg[0]
                arg_name = helper.var_idx("Epilogue", i) + "_" +  arg[1]
                arg_lists.append([arg_tp, arg_name])
        code_body += self.gen_using()
        code_body += self.gen_initialize()
        code_body += self.gen_run()

        code = ir.gen_func(self.gen_class_name, arg_lists, code_body)

        return code 

    def gen_code(self):
        code = self.gen_wrapper()
        helper.write_2_headfile("volta_impl.h", self.output_dir, self.user_header_file + "\n" +  code)

class gen_one_API:
    def __init__(self, fuse_gemm_info, gen_class_name, user_header_file, output_dir = "../"):
        self.fuse_gemm_info = fuse_gemm_info
        self.gen_class_name = gen_class_name
        self.user_header_file = ""
        for header in user_header_file: 
            self.user_header_file += "#include \"" + header + "\"\n"
        self.output_dir = output_dir
        self.b2b_num = len(fuse_gemm_info)

        self.gen_volta = gen_volta_turing_fuse_act_impl(fuse_gemm_info, gen_class_name, user_header_file, output_dir)

        self.gen_turing = gen_turing_impl(fuse_gemm_info, gen_class_name, user_header_file, output_dir)

    def gen_CUTLASS_irrelevant_API(self):
        code = ""
        code += "#include <cuda_runtime.h>\n"
        code += "#include <cassert>\n"

        param_name = "Fused" + str(self.b2b_num) + "xGemm_"
        for i in range(self.b2b_num):
            param_name += str(self.fuse_gemm_info[i]['mnk'][1]) + "_"
        param_name += "Params"
        params = ""
        params += "    " + "int M;\n"
        params += "    " + "int K0;\n"
        params += "    " + "int Batch;\n"
        params += "    " + "const void* A0;\n"
        for i in range(self.b2b_num):
            params += "    " + "const void* " + helper.var_idx("B", i) + ";\n"
            params += "    " + "const void* " + helper.var_idx("C", i) + ";\n"
            epilogue_args = helper.get_epilogue_args(self.fuse_gemm_info[i])
            acc_tp = helper.get_epilogue_compute_tp(self.fuse_gemm_info[i])
            for arg in epilogue_args:
                arg_tp = arg[0]
                arg_name = helper.var_idx("Epilogue", i) + "_" +  arg[1]
                params += "    " + arg_tp + " " + arg_name + ";\n"
            params += "    " + "void* " + helper.var_idx("D", i) + ";\n"
        code += ir.gen_struct(param_name, params)
        code += "using Param = " + param_name + ";\n"
        code += "void one_api( const  Param & param, int sm, cudaStream_t stream);\n"

        
        return code

    def gen_one_api(self):
        code = ""
        code += "/* Auto Generated code - Do not edit.*/\n"
        code += "#include \"cutlass_irrelevant.h\"\n"
        code += "#include \"api.h\"\n"
        code += "void one_api( const  Param & param, int sm, cudaStream_t stream) {\n"
        
        code += "    " + "if (sm == 70) \n"
        code += "    " + "    " + self.gen_class_name + "_volta_impl(param.M, param.K0, param.Batch, const_cast<void*>(param.A0), "
        for i in range(self.b2b_num):
            code += helper.var_idx("const_cast<void*>(param.B", i) + "), "
            code += helper.var_idx("const_cast<void*>(param.C", i) + "), "
            code += helper.var_idx("param.D", i) + ", "
            epilogue_args = helper.get_epilogue_args(self.fuse_gemm_info[i])
            for arg in epilogue_args:
                arg_name = helper.var_idx("Epilogue", i) + "_" +  arg[1]
                code += "param." + arg_name + ", "
        code += "stream);\n"
        code += "    " + "else if(sm >= 75) \n"
        code += "    " + "    " + self.gen_class_name + "_turing_impl(param.M, param.K0, param.Batch, const_cast<void*>(param.A0), "
        for i in range(self.b2b_num):
            code += helper.var_idx("const_cast<void*>(param.B", i) + "), "
            code += helper.var_idx("const_cast<void*>(param.C", i) + "), "
            code += helper.var_idx("param.D", i) + ", "
            epilogue_args = helper.get_epilogue_args(self.fuse_gemm_info[i])
            for arg in epilogue_args:
                arg_name = helper.var_idx("Epilogue", i) + "_" +  arg[1]
                code += "param." + arg_name + ", "
        code += "stream);\n"
        code += "    " + "else assert(0);\n"
        code += "}\n"
        return code

    def gen_code(self):

        turing_code = self.gen_turing.gen_wrapper()
        volta_code = self.gen_volta.gen_wrapper()
        cutlass_irrelevant_code = self.gen_CUTLASS_irrelevant_API()

        one_api_code = self.gen_one_api()
        with open(self.output_dir + "one_api.cu", "w+") as f:
            f.write(one_api_code)
 
        helper.write_2_headfile("cutlass_irrelevant.h", self.output_dir, cutlass_irrelevant_code)

        helper.write_2_headfile("api.h", self.output_dir, self.user_header_file + "\n" +  turing_code + volta_code)
