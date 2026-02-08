This example provides utilities for generating back-to-back (B2B) GEMMs using CUTLASS.

## Quick start
A configuration file containing the GEMMs to be fused together is located in [config.json](config.json). Edit
this to change the configuration that you would like to run.
```shell
cd ir_gen

# Set up basic variables
out_dir=directory_to_emit_files
cutlass_dir=$(pwd)/../../..
config_file=$(pwd)/../config.json

# Generate code for GEMMs described in `config_file`
./generate.sh $config_file $out_dir $cutlass_dir

# Build the generated code
cd $out_dir
mkdir build && cd build
cmake .. -DGPU_ARCHS="75;80"
make -j

# Run the generated code with M=1024 K0=32 and Batch=1
./sample 1024 32 1
```

## Current restrictions
This experimental example has the following restrictions:
1. N tile should not exceed 256, or register spilling will occur.
2. Only FP16 is supported currently
3. Matrix A must be row major, matrix B must be column major, matrices C and D must be row major.

## Copyright

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
