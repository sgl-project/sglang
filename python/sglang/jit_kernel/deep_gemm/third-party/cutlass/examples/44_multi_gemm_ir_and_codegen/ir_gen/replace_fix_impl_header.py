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

import os

class replace_fix_impl:
    def __init__(self, src_dir, dst_dir, cutlass_deps_root):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.cutlass_deps_root = cutlass_deps_root



    def gen_code(self):
        for sub_dir in os.walk(self.src_dir):
            files_in_sub_dir = sub_dir[2]
 
            src_dirs = sub_dir[0]
            output_dirs = self.dst_dir + sub_dir[0][len(self.src_dir):]

            if not os.path.exists(output_dirs):
                os.mkdir(output_dirs) 

            for f in files_in_sub_dir:
                with open(src_dirs +"/" + f, 'r') as current_file:
                    output_lines = []
                    lines = current_file.readlines()

                    for line in lines:
                        if(len(line) >= len("#include \"cutlass") and line[:len("#include \"cutlass")] == "#include \"cutlass"):
                            new_line = "#include \"" + self.cutlass_deps_root + line[len("#include \""):]
                            # print(new_line)
                            output_lines.append(new_line)
                        else:
                            output_lines.append(line)

                    with open(output_dirs + "/"  + f, "w+") as dest_file:
                        dest_file.writelines(output_lines)
