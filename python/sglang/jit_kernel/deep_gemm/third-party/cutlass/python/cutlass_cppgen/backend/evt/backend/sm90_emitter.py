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

"""
Emitter for Sm90 Epilogue Visitor
"""

from cutlass_library import DataTypeTag, EpilogueScheduleTag
from cutlass_cppgen.backend import GemmOperationUniversal
from cutlass_cppgen.backend.evt.backend.emitter_base import FusionCallbacks


class CollectiveEpilogue:
    def __init__(self, tile_description,
                 schedule,
                 element_c,
                 element_d,
                 fusion_callbacks) -> None:

        self.cta_tile_mnk = tile_description.threadblock_shape
        self.element_c = element_c
        self.element_d = element_d
        self.schedule = schedule
        self.fusion_callbacks = fusion_callbacks

    @property
    def CtaTileMNK(self) -> str:
        """
        The threadblock shape
        """
        return f"cute::Shape<_{self.cta_tile_mnk[0]}, _{self.cta_tile_mnk[1]}, _{self.cta_tile_mnk[2]}>"

    @property
    def EpilogueTileType(self) -> str:
        """
        The epilogue tile type
        """
        return "cutlass::epilogue::collective::EpilogueTileAuto"

    @property
    def Schedule(self) -> str:
        return EpilogueScheduleTag[self.schedule]

    def emit(self):
        callback_decl, callback_name = self.fusion_callbacks.emit()
        return callback_name, f"""
using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
  {self.CtaTileMNK}, {self.EpilogueTileType},
  {DataTypeTag[self.element_c]}, {DataTypeTag[self.element_d]},
  {self.Schedule}
>;
{callback_decl}
"""


class Sm90Emitter:
    def __init__(self, operation: GemmOperationUniversal, graph) -> None:
        fusion_callbacks = FusionCallbacks(graph, cc=90, emit_CD=False)

        self.collective_epilogue = CollectiveEpilogue(
            tile_description=operation.tile_description,
            schedule=operation.tile_description.epilogue_schedule,
            element_c=operation.C.element,
            element_d=operation.C.element,
            fusion_callbacks=fusion_callbacks
        )

    def emit(self):
        return self.collective_epilogue.emit()
