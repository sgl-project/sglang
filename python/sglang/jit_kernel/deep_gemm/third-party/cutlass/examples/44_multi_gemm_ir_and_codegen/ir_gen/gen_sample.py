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

class gen_test:
    def __init__(self, fuse_gemm_info, gen_class_name, user_header_file, output_dir = "../"):
        self.fuse_gemm_info = fuse_gemm_info
        self.gen_class_name = gen_class_name
        self.user_header_file = user_header_file
        self.sample_dir = output_dir
        self.b2b_num = len(fuse_gemm_info)

    def gen_cpp_sample(self):
        code = "/* Auto Generated code - Do not edit.*/\n"
        code +=  "#include <cstdio> \n"

        code += "#include \"cutlass/gemm/device/gemm_batched.h\" \n"
        code += "#include \"cutlass/cutlass.h\" \n"

        code += "#include \"../cutlass_irrelevant.h\" \n"
        code += "#include \"../cutlass_verify.h\" \n"

        code += "#include \"leaky_bias.h\" \n"

        code +=  "#include \"utils.h\" \n"
        


        code += "int main(int args, char * argv[]) {\n"
        code += "    " + "int M = atoi(argv[1]);\n"
        code += "    " + "int K0 = " + str(self.fuse_gemm_info[0]['mnk'][0]) + ";\n"
        code += "    " + "if(args == 3);\n"
        code += "    " + "    " + "K0 = atoi(argv[2]);\n"
        code += "    " + "int B = 1;\n"
        code += "    " + "if(args == 4);\n"
        code += "    " + "    " + "B = atoi(argv[3]);\n"

        code += "    " + "srand(1234UL);\n"
        code += "    " + "int device_id = 0;\n"
        code += "    " + "cudaGetDevice(&device_id);\n"
        code += "    " + "cudaDeviceProp prop;\n"
        code += "    " + "cudaGetDeviceProperties(&prop, device_id);\n"
        code += "    " + "int sm = prop.major *10 + prop.minor;\n"
        code += "using ElementCompute = cutlass::half_t;\n"

        for i in range(self.b2b_num):
            code += "    " + helper.var_idx("ElementCompute alpha", i) + " = ElementCompute(1);\n"
            addbias = helper.get_epilogue_add_bias_or_not( self.fuse_gemm_info[i])
            if addbias:
                code += "    " + helper.var_idx("ElementCompute beta", i) + " = ElementCompute(1);\n"
            else:
                code += "    " + helper.var_idx("ElementCompute beta", i) + " = ElementCompute(0);\n"

        code += "    " + "size_t flops = 0;\n"

        for i in range(self.b2b_num):
            m = self.fuse_gemm_info[i]['mnk'][0]
            n = self.fuse_gemm_info[i]['mnk'][1]
            k = self.fuse_gemm_info[i]['mnk'][2]

            bias_shape = helper.get_epilogue_bias_shape(self.fuse_gemm_info[i])
            
            this_k = "K0"
            if (i > 0):
                this_k = str(k)

            code += "    " + "flops += size_t(2) * size_t(M) * size_t(B) * " + "size_t(" + str(n) + ") * size_t(" + this_k + ");\n"

            code += "    " + helper.var_idx("cutlass::gemm::GemmCoord problem_size_", i) + "(" + "M" + ", " + str(n) + ", " + this_k + ");\n"

            code += "    " + helper.var_idx("memory_unit<cutlass::half_t> Mat_A", i) + helper.var_idx("(B * problem_size_", i) + helper.var_idx(".m() * problem_size_", i) + ".k());\n"
            code += "    " + helper.var_idx("memory_unit<cutlass::half_t> Mat_B", i) + helper.var_idx("(B * problem_size_", i) + helper.var_idx(".n() * problem_size_", i) + ".k());\n"
            code += "    " + helper.var_idx("memory_unit<cutlass::half_t> Mat_C", i) + "(B * " + str(bias_shape[0]) + " * " + str(bias_shape[1]) + ");\n"
            code += "    " + helper.var_idx("memory_unit<cutlass::half_t> Mat_D_cutlass_ref", i) + helper.var_idx("(B * problem_size_", i) + helper.var_idx(".m() * problem_size_", i) + ".n());\n"

            code += "    " + helper.var_idx("Mat_A", i) + ".init();\n"
            code += "    " + helper.var_idx("Mat_B", i) + ".init();\n"
            code += "    " + helper.var_idx("Mat_C", i) + ".init();\n"



        code += "    " + helper.var_idx("memory_unit<cutlass::half_t> Mat_D", self.b2b_num - 1) +  helper.var_idx("(B * problem_size_", i) + helper.var_idx(".m() * problem_size_",self.b2b_num - 1) + ".n());\n"

        params = []
        params.append("M")
        params.append("B")

        params.append("Mat_A0.device_ptr")
        for i in range(self.b2b_num):
            params.append(helper.var_idx("Mat_B", i) + ".device_ptr")
            params.append(helper.var_idx("Mat_C", i) + ".device_ptr")
            if i != self.b2b_num-1:
                params.append(helper.var_idx("Mat_D_cutlass_ref", i) + ".device_ptr")
        params.append(helper.var_idx("Mat_D", self.b2b_num - 1) + ".device_ptr")
    
        code += "    " + "Param arguments = {\n"
        code += "    " + "    " + "M,\n"
        code += "    " + "    " + "K0,\n"
        code += "    " + "    " + "B,\n"

        code += "    " + "    " + "reinterpret_cast<const void*>(Mat_A0.device_ptr),\n"
        cnt = 1
        for i in range(self.b2b_num):
            bias_flag = helper.get_epilogue_add_bias_or_not( self.fuse_gemm_info[i])
            code += "    " + "    " + "reinterpret_cast<const void*>(" + helper.var_idx("Mat_B", i) + ".device_ptr" + "),\n"
            cnt += 1
            if bias_flag:
                code += "    " + "    " + "reinterpret_cast<const void*>(" + helper.var_idx("Mat_C", i) + ".device_ptr" + "),\n"
                cnt += 1
            else:
                code += "    " + "    " + "reinterpret_cast<const void*>(NULL),\n"

            epilogue_args = helper.get_epilogue_args(self.fuse_gemm_info[i])
            acc_tp = helper.get_epilogue_compute_tp(self.fuse_gemm_info[i])
            for arg in epilogue_args:
                arg_value = str(arg[2])

                code +=  "    " + "    " + helper.type_2_cutlass_type(acc_tp)  + "(" + arg_value + "),\n"

            if i != self.b2b_num - 1:
                code += "    " + "    " + "reinterpret_cast<void*>(" + helper.var_idx("Mat_D_cutlass_ref", i) + ".device_ptr" + "),\n"
            else:
                code += "    " + "    " + "reinterpret_cast<void*>(" + helper.var_idx("Mat_D", i) + ".device_ptr" + ")};\n"




        code += "    " + "TI(FUSED_CUTLASS);\n"
        code += "    " + "for(int i = 0; i < 100; i++){\n"
        code += "    " + "    " + "one_api(arguments, sm, NULL);\n"

        code += "    " + "}\n"
        code += "    " + "TO(FUSED_CUTLASS, \"FUSED_CUTLASS\", 100);\n"

        code += "\n"

        for i in range(self.b2b_num):
            code_this = ""

            N_str = str(self.fuse_gemm_info[i]['mnk'][1])

            code_this += "    " + helper.var_idx("typename Gemm", i) + helper.var_idx("::Arguments arguments_", i) + "{\n"
            code_this += "    " + "    " + helper.var_idx("problem_size_", i) + ",\n"
            ldmA = str(self.fuse_gemm_info[i]['mnk'][2])
            if i == 0:
                ldmA = "K0"
            ldmB = str(self.fuse_gemm_info[i]['mnk'][2])
            if i == 0:
                ldmB = "K0"
            ldmC = str(self.fuse_gemm_info[i]['mnk'][1])

            ldmBias = str(helper.get_epilogue_bias_ldm(self.fuse_gemm_info[i]))

            if self.fuse_gemm_info[i]['A_format'] is 'Col':
                ldmA = "M"
            if self.fuse_gemm_info[i]['B_format'] is 'Row':
                ldmB = str(self.fuse_gemm_info[i]['mnk'][1])
            if self.fuse_gemm_info[i]['C_format'] is 'Col':
                ldmC = "M"

            if i == 0:
                code_this += "    " + "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['A_tp']) + "*>(" + helper.var_idx("Mat_A", i) + ".device_ptr), " + ldmA + "}, " + "M * " + ldmA + ",\n"
            else:
                code_this += "    " + "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['A_tp']) + "*>(" + helper.var_idx("Mat_D_cutlass_ref", i - 1) + ".device_ptr), " + ldmA + "}, " + "M * " + ldmA + ",\n"

            code_this += "    " + "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['B_tp']) + "*>(" + helper.var_idx("Mat_B", i) + ".device_ptr), " + ldmB + "}, " + N_str + " * " + ldmB + ",\n"
            
            M_bias = str(helper.get_epilogue_bias_shape(self.fuse_gemm_info[i])[0])

            code_this += "    " + "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['C_tp']) + "*>(" + helper.var_idx("Mat_C", i) + ".device_ptr), " + ldmBias + "}, " + M_bias + " * " + N_str + ",\n"
            code_this += "    " + "    " + "{reinterpret_cast<" + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['C_tp']) + "*>(" + helper.var_idx("Mat_D_cutlass_ref", i) + ".device_ptr), " + ldmC + "}, " + "M * " + ldmC + ",\n"
            code_this += "    " + "    " + "{ " + helper.var_idx("alpha", i) + ", " + helper.var_idx("beta", i) 
            for epilogue_arg in  helper.get_epilogue_args(self.fuse_gemm_info[i]):
                arg_value = str(epilogue_arg[2])
                code_this += ", " + helper.type_2_cutlass_type(self.fuse_gemm_info[i]['Acc_tp']) + "(" + str(arg_value) + ")"
            code_this += "    " + " },\n"
            code_this += "    " + "    " + "B};\n"

            code += code_this



        code += "    " + "TI(UNFUSED_CUTLASS);\n"
        code += "    " + "for(int i = 0; i < 100; i++){\n"
        code += "    " + "    " + self.gen_class_name + "_verify(\n"
        for i in range(self.b2b_num):
            code += "    " + "    " + "    " + helper.var_idx("arguments_", i) + ",\n"
        code += "    " + "    " + "    " + "NULL);\n"

        code += "    " + "}\n"
        code += "    " + "TO(UNFUSED_CUTLASS, \"UNFUSED_CUTLASS\", 100);\n"

        code += "    " + helper.var_idx("Mat_D_cutlass_ref", self.b2b_num - 1) + ".d2h();\n"
        code += "    " + helper.var_idx("Mat_D", self.b2b_num - 1) + ".d2h();\n"
        code += "    " + helper.var_idx("check_result(Mat_D_cutlass_ref", self.b2b_num - 1) + helper.var_idx(".host_ptr, Mat_D", self.b2b_num - 1) \
                       + helper.var_idx(".host_ptr, Mat_D", self.b2b_num - 1) + ".elements);\n"

        code += "\n\n}\n"

        with open(self.sample_dir + "sample.cu", "w+") as f:
            f.write(code)
