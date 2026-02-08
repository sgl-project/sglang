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

def type_2_cutlass_type(input_type = "fp16"):
    # float point type
    if input_type == "fp32":
        return "float"
    if input_type == "bf16":
        return "cutlass::bfloat16_t"
    if input_type == "fp16":
        return "cutlass::half_t"

    # integer type
    if(input_type == "int32"):
        return "int32_t"
    if(input_type == "int8"):
        return "int8_t"

    if input_type == 'Row':
        return 'cutlass::layout::RowMajor'
    if input_type == 'Col':
        return 'cutlass::layout::ColumnMajor'

def cvt_2_cutlass_shape(gemm_shape):
    # gemm shape
    if len(gemm_shape) == 3:
        val = "cutlass::gemm::GemmShape<"  \
                                        + str(gemm_shape[0]) + ", " \
                                        + str(gemm_shape[1]) + ", " \
                                        + str(gemm_shape[2]) + ">" 
        return val


def write_2_headfile(filename, file_dir, string):
    with open(file_dir + filename, 'w') as f:
        f.write("/* Auto Generated code - Do not edit.*/\n\n\n#pragma once\n" + string)

def var_idx(variable, index):
    return variable + str(index)


def list_2_string(input_list, ):
    rtn_string = ""
    
    cnt = 0

    for element in input_list:
        final = ", \n"
        if cnt == len(input_list) - 1:
            final = "\n"
        cnt += 1
        rtn_string += str(element) + final

    return rtn_string


def get_epilogue_info(layer_info):
    return layer_info['epilogue']

def get_epilogue_tp(layer_info):
    epilogue_info = get_epilogue_info(layer_info)
    return epilogue_info['tp']

def get_epilogue_add_bias_or_not(layer_info):
    epilogue_info = get_epilogue_info(layer_info)
    return epilogue_info['bias']['addbias']

def get_epilogue_add_bias_tp(layer_info):
    epilogue_info = get_epilogue_info(layer_info)
    return epilogue_info['bias']['bias_tp']

def get_epilogue_args(layer_info):
    epilogue_info = get_epilogue_info(layer_info)
    return epilogue_info['args']

def get_epilogue_bias_shape(layer_info):
    bias_tp = get_epilogue_add_bias_tp(layer_info).lower()
    mn_shape = layer_info['mnk'][:-1]

    if bias_tp == 'mat':
        mn_shape[0] = 'M'
        return mn_shape
    elif bias_tp == 'vec':
        mn_shape[0] = 1
        return mn_shape
    else:
        assert(0)

def get_epilogue_bias_ldm(layer_info):
    bias_tp = get_epilogue_add_bias_tp(layer_info).lower()
    mn_shape = layer_info['mnk'][:-1]

    c_layout = layer_info['C_format'].lower()

    if c_layout != 'row':
        assert(0)

    if bias_tp == 'mat':
        return mn_shape[1]
    elif bias_tp == 'vec':
        return 0
    else:
        assert(0)

def get_epilogue_compute_tp(layer_info):
    return layer_info['Acc_tp']
