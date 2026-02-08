# Copyright (c) 2014 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


set_property(
  SOURCE
      77_blackwell_fmha.cu
      77_blackwell_fmha_gen.cu
      77_blackwell_mla.cu
      77_blackwell_fmha_bwd.cu
      77_blackwell_mla_fwd.cu
  PROPERTY
      COMPILE_FLAGS "--use_fast_math -ftemplate-backtrace-limit=0"
)

set(TEST_BASIC --b=1 --h=4 --q=512 --k=512 --d=128 --verify --mask=no)
set(TEST_CAUSAL_00 --b=1 --h=4 --q=512 --k=512 --d=128 --verify --mask=causal)
set(TEST_CAUSAL_01 --verify --iterations=0 --b=1 --h=1 --h_k=1 --q=1013 --k=1024 --d=128 --mask=causal --causal-type=qend)
set(TEST_VARLEN --b=1 --h=4 --q=512 --k=512 --d=128 --verify --mask=residual --varlen)
set(TEST_HDIM64 --b=2 --h=4 --q=512 --k=512 --d=64 --verify)
set(TEST_GQA --b=2 --h=4 --h_k=2 --q=512 --k=512 --d=64 --verify)

set(TEST_VARLEN_00 --verify --varlen --mask=causal,residual --d=128 --h=8 --h_k=4 --varlen-q=128 --varlen-k=128)
set(TEST_VARLEN_01 --verify --varlen --mask=causal,residual --d=64 --h=4 --h_k=4 --varlen-q=128 --varlen-k=128)
set(TEST_VARLEN_02 --verify --varlen --mask=causal,residual --d=128 --h=4 --h_k=2 --varlen-q=128 --varlen-k=128)
set(TEST_VARLEN_03 --verify --varlen --mask=causal,residual --d=128 --h=8 --h_k=8 --varlen-q=256:256 --varlen-k=512:512)
set(TEST_VARLEN_04 --verify --varlen --mask=causal,residual --d=128 --h=8 --h_k=4 --varlen-q=256:256 --varlen-k=512:512)
set(TEST_VARLEN_05 --verify --varlen --mask=causal,residual --d=128 --h=8 --h_k=1 --varlen-q=256:256 --varlen-k=512:512)
set(TEST_VARLEN_06 --verify --varlen --mask=causal,residual --d=128 --h=8 --h_k=2 --varlen-q=256:256:256:256 --varlen-k=256:768:512:512)
set(TEST_VARLEN_07 --verify --varlen --mask=causal,residual --d=128 --h=8 --h_k=2 --varlen-q=256:256:256:256 --varlen-k=256:0:1280:512)
set(TEST_VARLEN_08 --verify --varlen --mask=causal,residual --d=128 --h=8 --h_k=2 --varlen-q=256:0:512:256 --varlen-k=256:256:1024:512)
set(TEST_VARLEN_09 --verify --varlen --mask=causal,residual --d=64 --h=16 --h_k=16 --varlen-q=100:300 --varlen-k=100:300)
set(TEST_VARLEN_10 --verify --varlen --mask=causal,residual --d=64 --h=4 --h_k=4 --varlen-q=3:2 --varlen-k=2:5)
set(TEST_VARLEN_11 --verify --varlen --mask=causal,residual --d=64 --h=4 --h_k=2 --varlen-q=17:10 --varlen-k=13:10)
set(TEST_VARLEN_12 --verify --varlen --mask=causal,residual --d=64 --h=4 --h_k=4 --varlen-q=177:845 --varlen-k=257:766)
set(TEST_VARLEN_13 --verify --varlen --mask=causal,residual --d=64 --h=4 --h_k=2 --varlen-q=177:366:479 --varlen-k=257:0:766)
set(TEST_VARLEN_14 --verify --varlen --mask=causal,residual --d=64 --h=4 --h_k=4 --varlen-q=1 --varlen-k=1)
set(TEST_VARLEN_15 --verify --varlen --mask=causal --causal-type=qbegin --d=128 --h=4 --h_k=4 --varlen-q=128 --varlen-k=128)
set(TEST_VARLEN_16 --verify --varlen --mask=causal --causal-type=qbegin --d=128 --h=4 --h_k=4 --varlen-q=17 --varlen-k=257)
set(TEST_VARLEN_17 --verify --varlen --mask=causal --causal-type=qbegin --d=128 --h=4 --h_k=4 --varlen-q=17 --varlen-k=25)
set(TEST_VARLEN_18 --verify --varlen --mask=causal --causal-type=qend --d=128 --h=4 --h_k=4 --varlen-q=128 --varlen-k=128)
set(TEST_VARLEN_19 --verify --varlen --mask=causal --causal-type=qend --d=128 --h=4 --h_k=4 --varlen-q=17 --varlen-k=257)
set(TEST_VARLEN_20 --verify --varlen --mask=causal --causal-type=qend --d=128 --h=4 --h_k=4 --varlen-q=17 --varlen-k=25)
set(TEST_VARLEN_21 --verify --varlen --mask=causal --causal-type=qend --d=128 --h=4 --h_k=4 --varlen-q=1013 --varlen-k=1024)
set(TEST_VARLEN_22 --verify --varlen --mask=causal --causal-type=qend --d=128 --h=4 --h_k=4 --varlen-q=1024 --varlen-k=1035)



set(TEST_MLA_FWD_VARLEN_00 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=8 --h_k=4 --varlen-q=128 --varlen-k=128)
set(TEST_MLA_FWD_VARLEN_01 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=4 --h_k=4 --varlen-q=128 --varlen-k=128)
set(TEST_MLA_FWD_VARLEN_02 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=4 --h_k=2 --varlen-q=128 --varlen-k=128)
set(TEST_MLA_FWD_VARLEN_03 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=8 --h_k=8 --varlen-q=256:256 --varlen-k=512:512)
set(TEST_MLA_FWD_VARLEN_04 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=8 --h_k=4 --varlen-q=256:256 --varlen-k=512:512)
set(TEST_MLA_FWD_VARLEN_05 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=8 --h_k=1 --varlen-q=256:256 --varlen-k=512:512)
set(TEST_MLA_FWD_VARLEN_06 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=8 --h_k=2 --varlen-q=256:256:256:256 --varlen-k=256:768:512:512)
set(TEST_MLA_FWD_VARLEN_07 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=8 --h_k=2 --varlen-q=256:256:256:256 --varlen-k=256:0:1280:512)
set(TEST_MLA_FWD_VARLEN_08 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=8 --h_k=2 --varlen-q=256:0:512:256 --varlen-k=256:256:1024:512)
set(TEST_MLA_FWD_VARLEN_09 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=16 --h_k=16 --varlen-q=100:300 --varlen-k=100:300)
set(TEST_MLA_FWD_VARLEN_10 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=4 --h_k=4 --varlen-q=2:3 --varlen-k=2:5)
set(TEST_MLA_FWD_VARLEN_11 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=4 --h_k=2 --varlen-q=11:10 --varlen-k=13:10)
set(TEST_MLA_FWD_VARLEN_12 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=4 --h_k=4 --varlen-q=177:766 --varlen-k=257:845)
set(TEST_MLA_FWD_VARLEN_13 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=4 --h_k=2 --varlen-q=177:0:479 --varlen-k=257:0:766)
set(TEST_MLA_FWD_VARLEN_14 --verify --varlen --mask=causal,residual --dl=128 --dr=64 --h=4 --h_k=4 --varlen-q=1 --varlen-k=1)
set(TEST_MLA_FWD_VARLEN_15 --verify --varlen --mask=causal --causal-type=qbegin --d=128 --h=4 --h_k=4 --varlen-q=128 --varlen-k=128)
set(TEST_MLA_FWD_VARLEN_16 --verify --varlen --mask=causal --causal-type=qbegin --d=128 --h=4 --h_k=4 --varlen-q=17 --varlen-k=257)
set(TEST_MLA_FWD_VARLEN_17 --verify --varlen --mask=causal --causal-type=qbegin --d=128 --h=4 --h_k=4 --varlen-q=17 --varlen-k=25)
set(TEST_MLA_FWD_VARLEN_18 --verify --varlen --mask=causal --causal-type=qend --d=128 --h=4 --h_k=4 --varlen-q=128 --varlen-k=128)
set(TEST_MLA_FWD_VARLEN_19 --verify --varlen --mask=causal --causal-type=qend --d=128 --h=4 --h_k=4 --varlen-q=17 --varlen-k=257)
set(TEST_MLA_FWD_VARLEN_20 --verify --varlen --mask=causal --causal-type=qend --d=128 --h=4 --h_k=4 --varlen-q=17 --varlen-k=25)
set(TEST_MLA_FWD_VARLEN_21 --verify --varlen --mask=causal --causal-type=qend --d=128 --h=4 --h_k=4 --varlen-q=1013 --varlen-k=1024)
set(TEST_MLA_FWD_VARLEN_22 --verify --varlen --mask=causal --causal-type=qend --d=128 --h=4 --h_k=4 --varlen-q=1024 --varlen-k=1035)


set(TEST_GEN_BASIC --b=1 --h=4 --k=512 --d=128 --verify)
set(TEST_GEN_VARLEN --b=1 --h=4 --k=512 --d=128 --verify  --varlen)
set(TEST_GEN_HDIM64 --b=2 --h=4 --k=512 --d=64 --verify)
set(TEST_GEN_GQA --b=2 --h=4 --h_k=2 --k=512 --d=128 --verify)
set(TEST_GEN_REMAP --b=2 --h=4 --h_k=2 --k=512 --d=128 --verify --remap)
set(TEST_GEN_CACHEONLY --b=2 --h=4 --h_k=2 --k=512 --d=128 --verify --cache-only)

set(TEST_MLA_BASIC --b=1 --k=512 --page=128 --verify)
set(TEST_BWD_MLA_BASIC --b=1 --h=4 --q=512 --k=512 --d=192 --d_vo=128 --verify --mask=no)
set(TEST_BWD_MLA_VARLEN --b=1 --h=4 --q=512 --k=512 --d=192 --d_vo=128 --verify --mask=residual --varlen)

set(TEST_MLA_SEP_REDUCTION  --b=1 --k=4096 --split_kv=8 --page=128 --verify)
set(TEST_MLA_FUSE_REDUCTION --b=1 --k=4096 --split_kv=8 --page=128 --fuse_reduction --verify)

if(NOT WIN32 AND (NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang")) AND (CUTLASS_NVCC_ARCHS MATCHES 100a))

  foreach(PREC fp8 fp16)
    string(TOUPPER "${PREC}" PREC_MACRO)

    cutlass_example_add_executable(
        77_blackwell_fmha_${PREC}
        77_blackwell_fmha.cu
        TEST_COMMAND_OPTIONS
        TEST_BASIC
        TEST_CAUSAL_00
        TEST_CAUSAL_01
        TEST_VARLEN
        TEST_HDIM64
        TEST_GQA
        TEST_VARLEN_00
        TEST_VARLEN_01
        TEST_VARLEN_02
        TEST_VARLEN_03
        TEST_VARLEN_04
        TEST_VARLEN_05
        TEST_VARLEN_06
        TEST_VARLEN_07
        TEST_VARLEN_08
        TEST_VARLEN_09
        TEST_VARLEN_10
        TEST_VARLEN_11
        TEST_VARLEN_12
        TEST_VARLEN_13
        TEST_VARLEN_14
        TEST_VARLEN_15
        TEST_VARLEN_16
        TEST_VARLEN_17
        TEST_VARLEN_18
        TEST_VARLEN_19
        TEST_VARLEN_20
        TEST_VARLEN_21
        TEST_VARLEN_22
        )
    target_include_directories(77_blackwell_fmha_${PREC} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(77_blackwell_fmha_${PREC} PRIVATE ${PREC_MACRO})
  
    cutlass_example_add_executable(
        77_blackwell_fmha_gen_${PREC}
        77_blackwell_fmha_gen.cu
        TEST_COMMAND_OPTIONS
        TEST_GEN_BASIC
        TEST_GEN_VARLEN
        # TEST_GEN_HDIM64
        TEST_GEN_GQA
        TEST_GEN_REMAP
        TEST_GEN_CACHEONLY
        )
    target_include_directories(77_blackwell_fmha_gen_${PREC} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(77_blackwell_fmha_gen_${PREC} PRIVATE ${PREC_MACRO})

    cutlass_example_add_executable(
        77_blackwell_mla_2sm_${PREC}
        77_blackwell_mla.cu
        TEST_COMMAND_OPTIONS
        TEST_MLA_BASIC
        TEST_MLA_SEP_REDUCTION
        TEST_MLA_FUSE_REDUCTION
        )
    target_include_directories(77_blackwell_mla_2sm_${PREC} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(77_blackwell_mla_2sm_${PREC} PRIVATE ${PREC_MACRO})
    target_compile_options(77_blackwell_mla_2sm_${PREC} PRIVATE -Xptxas -v)

    cutlass_example_add_executable(
        77_blackwell_mla_2sm_cpasync_${PREC}
        77_blackwell_mla.cu
        TEST_COMMAND_OPTIONS
        TEST_MLA_BASIC
        TEST_MLA_SEP_REDUCTION
        TEST_MLA_FUSE_REDUCTION
        )
    target_include_directories(77_blackwell_mla_2sm_cpasync_${PREC} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(77_blackwell_mla_2sm_cpasync_${PREC} PRIVATE ${PREC_MACRO} CPASYNC)
    target_compile_options(77_blackwell_mla_2sm_cpasync_${PREC} PRIVATE -Xptxas -v)

    cutlass_example_add_executable(
        77_blackwell_fmha_bwd_${PREC}
        77_blackwell_fmha_bwd.cu
        TEST_COMMAND_OPTIONS
        TEST_BASIC
        TEST_VARLEN
        # NOTE: bwd doesn't support GQA yet, --h_k will just get ignored in these tests
        TEST_VARLEN_00
        TEST_VARLEN_01
        TEST_VARLEN_02
        TEST_VARLEN_03
        TEST_VARLEN_04
        TEST_VARLEN_05
        TEST_VARLEN_06
        TEST_VARLEN_07
        TEST_VARLEN_08
        TEST_VARLEN_09
        TEST_VARLEN_10
        TEST_VARLEN_11
        TEST_VARLEN_12
        TEST_VARLEN_13
        TEST_VARLEN_14
        TEST_BWD_MLA_BASIC
        TEST_BWD_MLA_VARLEN
        )
    target_include_directories(77_blackwell_fmha_bwd_${PREC} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(77_blackwell_fmha_bwd_${PREC} PRIVATE ${PREC_MACRO})
    target_compile_options(77_blackwell_fmha_bwd_${PREC} PRIVATE -Xptxas -v)

    cutlass_example_add_executable(
        77_blackwell_mla_fwd_${PREC}
        77_blackwell_mla_fwd.cu
        TEST_COMMAND_OPTIONS
        TEST_BASIC
        TEST_CAUSAL_00
        TEST_VARLEN
        TEST_HDIM64
        TEST_GQA
        TEST_MLA_FWD_VARLEN_00
        TEST_MLA_FWD_VARLEN_01
        TEST_MLA_FWD_VARLEN_02
        TEST_MLA_FWD_VARLEN_03
        TEST_MLA_FWD_VARLEN_04
        TEST_MLA_FWD_VARLEN_05
        TEST_MLA_FWD_VARLEN_06
        TEST_MLA_FWD_VARLEN_07
        TEST_MLA_FWD_VARLEN_08
        TEST_MLA_FWD_VARLEN_09
        TEST_MLA_FWD_VARLEN_10
        TEST_MLA_FWD_VARLEN_11
        TEST_MLA_FWD_VARLEN_12
        TEST_MLA_FWD_VARLEN_13
        TEST_MLA_FWD_VARLEN_14
        TEST_MLA_FWD_VARLEN_15
        TEST_MLA_FWD_VARLEN_16
        TEST_MLA_FWD_VARLEN_17
        TEST_MLA_FWD_VARLEN_18
        TEST_MLA_FWD_VARLEN_19
        TEST_MLA_FWD_VARLEN_20
        TEST_MLA_FWD_VARLEN_21
        TEST_MLA_FWD_VARLEN_22
        )
    target_include_directories(77_blackwell_mla_fwd_${PREC} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(77_blackwell_mla_fwd_${PREC} PRIVATE ${PREC_MACRO})
    target_compile_options(77_blackwell_mla_fwd_${PREC} PRIVATE -Xptxas -v)
  endforeach()

  # Add a target that builds all examples
  add_custom_target(77_blackwell_fmha_all
    DEPENDS
      77_blackwell_fmha_fp8
      77_blackwell_fmha_fp16
      77_blackwell_fmha_gen_fp8
      77_blackwell_fmha_gen_fp16
      77_blackwell_mla_2sm_fp8
      77_blackwell_mla_2sm_fp16
      77_blackwell_mla_2sm_cpasync_fp8
      77_blackwell_mla_2sm_cpasync_fp16
      77_blackwell_fmha_bwd_fp8
      77_blackwell_fmha_bwd_fp16
      77_blackwell_mla_fwd_fp8
      77_blackwell_mla_fwd_fp16
  )
endif()
