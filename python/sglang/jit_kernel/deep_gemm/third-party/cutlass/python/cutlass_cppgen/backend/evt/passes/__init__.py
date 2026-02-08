#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cutlass_cppgen.backend.evt.passes.graph_drawer import EVTGraphDrawer
from cutlass_cppgen.backend.evt.passes.pass_argument_type import PassGetArgumentType
from cutlass_cppgen.backend.evt.passes.pass_dag_2_tree import PassDAG2Tree
from cutlass_cppgen.backend.evt.passes.pass_get_impl import PassGetImpl
from cutlass_cppgen.backend.evt.passes.pass_fix_element_d import PassFixElementD
from cutlass_cppgen.backend.evt.passes.pass_layout_elimination import PassLayoutManipulateElimination
from cutlass_cppgen.backend.evt.passes.pass_manager import EVTPassManager
from cutlass_cppgen.backend.evt.passes.pass_preprocess_red import PassPreprocessRed
from cutlass_cppgen.backend.evt.passes.pass_shape_type_propagation import PassShapeTypePropagation
from cutlass_cppgen.backend.evt.passes.smem_size_calculator import GetSmemSize
