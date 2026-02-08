#################################################################################################
#
# Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Valid WGMMA shapes, MMA multipliers, and cluster sizes for SM90, associated with levels.
These shape and level pairs are defined as dicts, where keys are shapes and values are their
associated levels. If the user input level for that category (MMA multiplier, WGMMA shape, cluster
size) is smaller than a shape's associated level, it will be excluded, and otherwise, included.
Higher levels are therefore less likely emitted, but lower levels are more emitted more frequently.
Level 0 is always emitted. The default behavior in `generator.py` is that level 1 is only emitted
when the `--kernel` argument is non-empty.
"""

# NOTE: more combinations are possible here.
# Levels [0, 3] exist in order to control exactly what configs are generated in different dtypes.
# The rest are only used in the exhaustive mode (when the corresponding level digit is 9).
# MMA multipliers are multiplied by MMA instruction shapes (WGMMA shapes) to get CTA shapes.
SM90_MMA_MULTIPLIERS = {
    (2, 1, 4): 0,
    (1, 1, 4): 1,
    (4, 1, 4): 2,
    (2, 2, 4): 3,
    (2, 1, 8): 4,
    (4, 1, 8): 4,
    (1, 1, 8): 4,
    (2, 2, 8): 4,
    (2, 1, 16): 5,
    (4, 1, 16): 5,
    (1, 1, 16): 5,
    (2, 2, 16): 5,
}

# Level 0: only (1, 2, 1) -- fp8 dense gemms in pruned case
# Level 1: clusters with 2 CTAs -- all but fp8 (s8, u8, f16, b16, f32, tf32) dense gemms in pruned case
# Level 2: clusters with 1 or 2 CTAs
# Level 3: clusters with 1, 2, or 4 CTAs
# Level 4: clusters with 1, 2, 4, or 8 CTAs
# Level 5: clusters with 1, 2, 4, 8, or 16 CTAs
SM90_CLUSTER_SIZES = {
    (1, 2, 1): 0,
    (2, 1, 1): 1,
    (1, 1, 1): 2,
    (2, 2, 1): 3,
    (1, 4, 1): 3,
    (4, 1, 1): 3,
    (2, 4, 1): 4,
    (4, 2, 1): 4,
    (1, 8, 1): 4,
    (8, 1, 1): 4,
    (4, 4, 1): 5,
}


# WGMMA shapes
# Level 0: "default" shape only,
# Level 1: additional shapes for the unpruned case (tf32 only)
# Level 2: shapes that are all powers of 2
# Level 3: all other shapes
SM90_WGMMA_SHAPES_FP16_BF16_DENSE = {
    (64, 8, 16): 2,
    (64, 16, 16): 2,
    (64, 24, 16): 3,
    (64, 32, 16): 2,
    (64, 40, 16): 3,
    (64, 48, 16): 3,
    (64, 56, 16): 3,
    (64, 64, 16): 2,
    (64, 72, 16): 3,
    (64, 80, 16): 3,
    (64, 88, 16): 3,
    (64, 96, 16): 3,
    (64, 104, 16): 3,
    (64, 112, 16): 3,
    (64, 120, 16): 3,
    (64, 128, 16): 0,
    (64, 136, 16): 3,
    (64, 144, 16): 3,
    (64, 152, 16): 3,
    (64, 160, 16): 3,
    (64, 168, 16): 3,
    (64, 176, 16): 3,
    (64, 184, 16): 3,
    (64, 192, 16): 3,
    (64, 200, 16): 3,
    (64, 208, 16): 3,
    (64, 216, 16): 3,
    (64, 224, 16): 3,
    (64, 232, 16): 3,
    (64, 240, 16): 3,
    (64, 248, 16): 3,
    (64, 256, 16): 1,
}

SM90_WGMMA_SHAPES_TF32_DENSE = {
    (64, 8, 8): 2,
    (64, 16, 8): 2,
    (64, 24, 8): 3,
    (64, 32, 8): 2,
    (64, 40, 8): 3,
    (64, 48, 8): 3,
    (64, 56, 8): 3,
    (64, 64, 8): 2,
    (64, 72, 8): 3,
    (64, 80, 8): 3,
    (64, 88, 8): 3,
    (64, 96, 8): 3,
    (64, 104, 8): 3,
    (64, 112, 8): 3,
    (64, 120, 8): 3,
    (64, 128, 8): 0,
    (64, 136, 8): 3,
    (64, 144, 8): 3,
    (64, 152, 8): 3,
    (64, 160, 8): 3,
    (64, 168, 8): 3,
    (64, 176, 8): 3,
    (64, 184, 8): 3,
    (64, 192, 8): 3,
    (64, 200, 8): 3,
    (64, 208, 8): 3,
    (64, 216, 8): 3,
    (64, 224, 8): 3,
    (64, 232, 8): 3,
    (64, 240, 8): 3,
    (64, 248, 8): 3,
    (64, 256, 8): 1,
}

SM90_WGMMA_SHAPES_FP8_DENSE = {
    (64, 8, 32): 2,
    (64, 16, 32): 2,
    (64, 24, 32): 3,
    (64, 32, 32): 2,
    (64, 40, 32): 3,
    (64, 48, 32): 3,
    (64, 56, 32): 3,
    (64, 64, 32): 2,
    (64, 72, 32): 3,
    (64, 80, 32): 3,
    (64, 88, 32): 3,
    (64, 96, 32): 3,
    (64, 104, 32): 3,
    (64, 112, 32): 3,
    (64, 120, 32): 3,
    (64, 128, 32): 0,
    (64, 136, 32): 3,
    (64, 144, 32): 3,
    (64, 152, 32): 3,
    (64, 160, 32): 3,
    (64, 168, 32): 3,
    (64, 176, 32): 3,
    (64, 184, 32): 3,
    (64, 192, 32): 3,
    (64, 200, 32): 3,
    (64, 208, 32): 3,
    (64, 216, 32): 3,
    (64, 224, 32): 3,
    (64, 232, 32): 3,
    (64, 240, 32): 3,
    (64, 248, 32): 3,
    (64, 256, 32): 1,
}

SM90_WGMMA_SHAPES_INT8_DENSE = {
    (64, 8, 32): 2,
    (64, 16, 32): 2,
    (64, 24, 32): 3,
    (64, 32, 32): 2,
    (64, 48, 32): 3,
    (64, 64, 32): 2,
    (64, 80, 32): 3,
    (64, 96, 32): 3,
    (64, 112, 32): 3,
    (64, 128, 32): 0,
    (64, 144, 32): 3,
    (64, 160, 32): 3,
    (64, 176, 32): 3,
    (64, 192, 32): 3,
    (64, 208, 32): 3,
    (64, 224, 32): 3,
    (64, 240, 32): 3,
    (64, 256, 32): 1,
}
