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


indentation = "    "


def append_word(word):
    code = ""
    code += word
    code += " "
    return code


def gen_namespace(namespace, codeBody):
    code_gen = "namespace " + namespace + " {\n"
    code_gen += codeBody
    code_gen += "} // namespace " + namespace + "\n"
    return code_gen


def gen_expression(type, lval, rval = None):
    code_gen = ""
    code_gen += append_word(type)
    code_gen += append_word(lval)
    if rval is not None:
        code_gen += append_word("=")
        code_gen += append_word(rval)
    return code_gen


def gen_class(name, codeBody, inheritance_code = None):
    code_gen = ""
    if inheritance_code is None:
        code_gen = "class " + name + "{\n"
    else:
        code_gen = "class " + name + " : "+ inheritance_code + "{\n"
    code_gen += codeBody
    code_gen += "}; // class " + name + "\n"
    return code_gen


def gen_struct(name, codeBody, specialized = None):
    specialized_code = ""
    if specialized is not None:
        specialized_code = "<" + specialized + ">"
    code_gen = "struct " + name + specialized_code + "{\n"
    code_gen += codeBody
    code_gen += "}; // struct " + name + "\n"
    return code_gen


def gen_template_arg(arg_type, arg_name, default_val = None):
    rval = None
    if default_val is not None:
        rval = str(default_val)

    arg_typename = ""
    if arg_type is int:
        arg_typename = "int"
    elif arg_type is bool:
        arg_typename = "bool"
    else:
        arg_typename = "typename"

    internal_arg_name = arg_name + "_"

    code_gen = indentation
    code_gen += gen_expression(arg_typename, internal_arg_name, rval)

    return code_gen


def gen_template_args(args, set_default = True):
    arg_len = len(args)
    cnt = 1
    code_gen = ""
    for arg_tuple in args:
        arg_type = arg_tuple[0]
        arg_name = arg_tuple[1]
        arg_default_val = None
        if len(arg_tuple) == 3 and set_default:
            arg_default_val = arg_tuple[2]

        code_gen += gen_template_arg(arg_type, arg_name, arg_default_val)
        if cnt != arg_len:
            code_gen += ",\n"
        cnt += 1

    return code_gen


def gen_template_head(args, set_default = True):
    code_gen = "template <\n"
    code_gen += gen_template_args(args, set_default)
    code_gen += ">\n"
    return code_gen


def export_template_args(args):
    code_gen = "public:\n"
    for arg_tuple in args:
        code_gen += indentation
        arg_type = arg_tuple[0]
        arg_name = arg_tuple[1]
        internal_arg_name = arg_name + "_"

        typename = ""
        if arg_type is int:
            typename = "static int const"
        elif arg_type is bool:
            typename = "static bool const"
        else:
            typename = "using"

        code_gen += gen_expression(typename, arg_name, internal_arg_name)
        code_gen += ";\n"
    return code_gen


def gen_template_class(class_name, args, codeBody, set_default = True, inheritance_code = None):
    code_gen = ""

    code_gen += gen_template_head(args, set_default)
    code_gen += gen_class(class_name, export_template_args(args) + codeBody, inheritance_code)

    return code_gen


def gen_template_struct(struct_name, args, codeBody, speicalized = None, set_default = True, export_args = True):
    code_gen = ""
    code_gen += gen_template_head(args, set_default)
    code = export_template_args(args) + codeBody
    if export_args is False:
        code = codeBody
    code_gen += gen_struct(struct_name, code , speicalized)

    return code_gen


def gen_declare_template_struct(name, *params):
    code = name + "<"
    cnt = 0
    param_num = len(params)
    for param in params:
        final = ", "
        if cnt == param_num - 1:
            final = ""
        code += param + final
        cnt += 1
    code += ">;\n"
    return code


def filtered_param(params, name_and_value_pair, keep_ = False):
    rtn_template_args = []
    speicalized_template_args = []

    for param in params:
        param_name = ""
        if len(param) >= 1:
            param_name = param[1]
        else:
            param_name = param[0]
        
        hit_flag = False
        set_value = ""
        for n_v_pair in name_and_value_pair:
            
            filter_name = n_v_pair[0]
            set_value = n_v_pair[1]

            if param_name == (filter_name + "_") or param_name == filter_name :
                hit_flag = True
                break

            
        if hit_flag is False:
            rtn_template_args.append(param)

        if hit_flag is True:
            speicalized_template_args.append(set_value)
        else:
            if keep_ is True:
                speicalized_template_args.append(param_name + "_")
            else:
                speicalized_template_args.append(param_name)

    
    specialized_template_arg_str = helper.list_2_string(speicalized_template_args)
    
    return rtn_template_args, specialized_template_arg_str

            
def gen_func(func_name, arg_lists, code_body, only_declare = False, with_cudaStream = True):
    code = "void " + func_name + "(\n"
    for arg in arg_lists:
        arg_tp = arg[0]
        arg_nm = arg[1]
        code += "    " + arg_tp + " " + arg_nm + ",\n"
    code += "cudaStream_t stream)"
    if only_declare :
        return code
    code += "{\n"

    code += code_body + "\n"
    code += "}\n"
    return code


def indent_level(code, level = 0):
    rtn_code = ""
    for i in range(level):
        rtn_code += "    "
    
    rtn_code += code

    return rtn_code
