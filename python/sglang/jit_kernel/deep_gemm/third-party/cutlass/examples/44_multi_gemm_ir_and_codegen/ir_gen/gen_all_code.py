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

import gen_turing_and_volta as api_generator
import gen_sample as sample_creater
import gen_cmake as cmake_creater
import gen_verify as verify_creater
import gen_device as b2b_fused_generator
import replace_fix_impl_header

import argparse
import os
import json


parser = argparse.ArgumentParser(description="Generates Fused Multi-GEMM CUTLASS Kernels")
parser.add_argument("--config-file", default="config.json", help="JSON file containing configuration to generate")
parser.add_argument("--gen-name", default="FusedMultiGemmForward", help="Specific the output name")
parser.add_argument("--output-dir", default="", help="Specifies the output dir")
parser.add_argument("--cutlass-dir", default="", help="Specifies the dependent CUTLASS repo dir")
parser.add_argument("--gen-include-cutlass-dir", default="", help="Specifies the generated CUTLASS code include dir, if needed.")
args = parser.parse_args()

gen_name = args.gen_name

cutlass_deps_dir = args.cutlass_dir

output_dir = args.output_dir
output_dir += "/"

cutlass_deps_root = args.gen_include_cutlass_dir
if cutlass_deps_root == '':
    cutlass_deps_root = cutlass_deps_dir + "/include/"
cutlass_deps_root +='/'


if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

if not os.path.exists(output_dir + "/" + "auto_gen"):
    os.mkdir(output_dir + "/" + "auto_gen") 

if not os.path.exists(output_dir + "/" + "fixed_impl"):
    os.mkdir(output_dir + "/" + "fixed_impl" )

if not os.path.exists(output_dir + "/" + "sample"):
    os.mkdir(output_dir + "/" + "sample" )

if not os.path.exists(output_dir + "/" + "auto_gen" + "/" + "device"):
    os.mkdir(output_dir + "/" + "auto_gen" + "/" + "device") 
if not os.path.exists(output_dir + "/" + "auto_gen" + "/" + "kernel"):
    os.mkdir(output_dir + "/" + "auto_gen" + "/" + "kernel")
if not os.path.exists(output_dir + "/" + "auto_gen" + "/" + "threadblock"):
    os.mkdir(output_dir + "/" + "auto_gen" + "/" + "threadblock")

with open(args.config_file, 'r') as infile:
    gemm_info_dict = json.load(infile)

keys = sorted(gemm_info_dict.keys())
fuse_gemm_info = [gemm_info_dict[k] for k in keys]


for_cutlass_gen_user_include_header_file = [
    cutlass_deps_root + "cutlass/epilogue/thread/linear_combination_leaky_relu.h",
    cutlass_deps_root + "cutlass/epilogue/thread/linear_combination.h",
]

for_fused_wrapper = [
    cutlass_deps_root + "cutlass/epilogue/thread/linear_combination_leaky_relu.h",
    cutlass_deps_root + "cutlass/epilogue/thread/linear_combination.h",
    "auto_gen/device/" + gen_name + ".h",
    cutlass_deps_root + "cutlass/gemm/device/gemm_batched.h",
    cutlass_deps_root + "cutlass/cutlass.h",
]

# Copy fixed implementation to the output directory
fix_impl = replace_fix_impl_header.replace_fix_impl("../fixed_impl/", output_dir +"/fixed_impl/", cutlass_deps_root)
fix_impl.gen_code()

auto_gen_output_dir = output_dir + "/auto_gen/"
project_root = ""
turing_plus = b2b_fused_generator.gen_device(fuse_gemm_info, gen_name, for_cutlass_gen_user_include_header_file, cutlass_deps_root, project_root, auto_gen_output_dir)
turing_plus.gen_code(75, 'hmma1688', False)

api = api_generator.gen_one_API(fuse_gemm_info, gen_name, for_fused_wrapper, output_dir)
api.gen_code()

# Generate C++ sample
os.system("cp ../leaky_bias.h " + output_dir + "/sample/")
os.system("cp ../utils.h " + output_dir + "/sample/")

sample_dir = output_dir + "/sample/"
sample = sample_creater.gen_test(fuse_gemm_info, gen_name, for_cutlass_gen_user_include_header_file, sample_dir)
sample.gen_cpp_sample()

cmake_gen = cmake_creater.gen_build_sys(cutlass_deps_dir, output_dir)
cmake_gen.gen_code()

verify = verify_creater.gen_verify(fuse_gemm_info, gen_name, for_fused_wrapper, output_dir)
verify.gen_code()
