#################################################################################################
#
# Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Valid tcgen05 shapes and cluster sizes for SM100, associated with levels.
These shape and level pairs are defined as dicts, where keys are shapes and values are their
associated levels. If the user input level for that category (tcgen05 shape, cluster
size) is smaller than a shape's associated level, it will be excluded, and otherwise, included.
Higher levels are therefore less likely emitted, but lower levels are more emitted more frequently.
Level 0 is always emitted. 
"""

try:
    from .library import DynamicClusterShape
except:
    from library import DynamicClusterShape

SM100_CLUSTER_SHAPES_1SM = {
    tuple(DynamicClusterShape) : 0,
    # size 1 cluster
    (1, 1, 1): 1,
    # size 2 cluster
    (1, 2, 1): 2,
    (2, 1, 1): 5,
    # size 4 clusters
    (2, 2, 1): 6,
    (1, 4, 1): 3,
    (4, 1, 1): 6,
    # size 8 clusters
    (2, 4, 1): 7,
    (4, 2, 1): 7,
    (1, 8, 1): 8,
    (8, 1, 1): 8,
    # size 16 cluster
    (4, 4, 1): 4,
}

SM100_CLUSTER_SHAPES_2SM = {
    tuple(DynamicClusterShape) : 0,
    # size 2 cluster
    (2, 1, 1): 1,
    # size 4 clusters
    (2, 2, 1): 2,
    (4, 1, 1): 2,
    # size 8 clusters
    (2, 4, 1): 3,
    (4, 2, 1): 3,
    (8, 1, 1): 6,
    # size 16 cluster
    (4, 4, 1): 4,
}

# MMA shapes

# 16b Dense

SM100_MMA_SHAPES_16b_DENSE_1SM = {
    (64,   8, 16): 5,
    (64,  16, 16): 2,
    (64,  24, 16): 5,
    (64,  32, 16): 2,
    (64,  40, 16): 5,
    (64,  48, 16): 5,
    (64,  56, 16): 5,
    (64,  64, 16): 2,
    (64,  72, 16): 5,
    (64,  80, 16): 5,
    (64,  88, 16): 5,
    (64,  96, 16): 5,
    (64, 104, 16): 5,
    (64, 112, 16): 5,
    (64, 120, 16): 5,
    (64, 128, 16): 0,
    (64, 136, 16): 5,
    (64, 144, 16): 5,
    (64, 152, 16): 5,
    (64, 160, 16): 5,
    (64, 168, 16): 5,
    (64, 176, 16): 5,
    (64, 184, 16): 5,
    (64, 192, 16): 3,
    (64, 200, 16): 5,
    (64, 208, 16): 5,
    (64, 216, 16): 5,
    (64, 224, 16): 5,
    (64, 232, 16): 5,
    (64, 240, 16): 5,
    (64, 248, 16): 5,
    (64, 256, 16): 3,

    (128,  16, 16): 2,
    (128,  32, 16): 2,
    (128,  48, 16): 5,
    (128,  64, 16): 2,
    (128,  80, 16): 5,
    (128,  96, 16): 5,
    (128, 112, 16): 5,
    (128, 128, 16): 0,
    (128, 144, 16): 5,
    (128, 160, 16): 5,
    (128, 176, 16): 5,
    (128, 192, 16): 3,
    (128, 208, 16): 5,
    (128, 224, 16): 5,
    (128, 240, 16): 5,
    (128, 256, 16): 0,

}


SM100_MMA_SHAPES_16b_DENSE_2SM = {
    (128,  32, 16): 2,
    (128,  64, 16): 2,
    (128,  96, 16): 5,
    (128, 128, 16): 0,
    (128, 160, 16): 5,
    (128, 192, 16): 5,
    (128, 224, 16): 5,
    (128, 256, 16): 0,

    (256,  32, 16): 2,
    (256,  64, 16): 2,
    (256,  96, 16): 5,
    (256, 128, 16): 0,
    (256, 160, 16): 5,
    (256, 192, 16): 3,
    (256, 224, 16): 5,
    (256, 256, 16): 0,
}

# TF32 Dense

SM100_MMA_SHAPES_TF32_DENSE_1SM = {
    (64,   8, 8): 5,
    (64,  16, 8): 2,
    (64,  24, 8): 5,
    (64,  32, 8): 2,
    (64,  40, 8): 5,
    (64,  48, 8): 5,
    (64,  56, 8): 5,
    (64,  64, 8): 1,
    (64,  72, 8): 5,
    (64,  80, 8): 5,
    (64,  88, 8): 5,
    (64,  96, 8): 5,
    (64, 104, 8): 5,
    (64, 112, 8): 5,
    (64, 120, 8): 5,
    (64, 128, 8): 0,
    (64, 136, 8): 5,
    (64, 144, 8): 5,
    (64, 152, 8): 5,
    (64, 160, 8): 5,
    (64, 168, 8): 5,
    (64, 176, 8): 5,
    (64, 184, 8): 5,
    (64, 192, 8): 3,
    (64, 200, 8): 5,
    (64, 208, 8): 5,
    (64, 216, 8): 5,
    (64, 224, 8): 5,
    (64, 232, 8): 5,
    (64, 240, 8): 5,
    (64, 248, 8): 5,
    (64, 256, 8): 3,

    (128,  16, 8): 2,
    (128,  32, 8): 2,
    (128,  48, 8): 5,
    (128,  64, 8): 2,
    (128,  80, 8): 5,
    (128,  96, 8): 5,
    (128, 112, 8): 5,
    (128, 128, 8): 0,
    (128, 144, 8): 5,
    (128, 160, 8): 5,
    (128, 176, 8): 5,
    (128, 192, 8): 3,
    (128, 208, 8): 5,
    (128, 224, 8): 5,
    (128, 240, 8): 5,
    (128, 256, 8): 0,

}

SM100_MMA_SHAPES_TF32_DENSE_2SM = {
    (128,  32, 8): 2,
    (128,  64, 8): 1,
    (128,  96, 8): 5,
    (128, 128, 8): 0,
    (128, 160, 8): 5,
    (128, 192, 8): 5,
    (128, 224, 8): 5,
    (128, 256, 8): 0,

    (256,  32, 8): 2,
    (256,  64, 8): 1,
    (256,  96, 8): 5,
    (256, 128, 8): 0,
    (256, 160, 8): 5,
    (256, 192, 8): 5,
    (256, 224, 8): 5,
    (256, 256, 8): 0,
}

# F8F6F4
SM100_MMA_SHAPES_F8F6F4_DENSE_1SM = {
    (64,   8, 32): 4,
    (64,  16, 32): 4,
    (64,  24, 32): 5,
    (64,  32, 32): 3,
    (64,  40, 32): 5,
    (64,  48, 32): 5,
    (64,  56, 32): 5,
    (64,  64, 32): 2,
    (64,  72, 32): 5,
    (64,  80, 32): 5,
    (64,  88, 32): 5,
    (64,  96, 32): 5,
    (64, 104, 32): 5,
    (64, 112, 32): 5,
    (64, 120, 32): 5,
    (64, 128, 32): 0,
    (64, 136, 32): 5,
    (64, 144, 32): 5,
    (64, 152, 32): 5,
    (64, 160, 32): 5,
    (64, 168, 32): 5,
    (64, 176, 32): 5,
    (64, 184, 32): 5,
    (64, 192, 32): 5,
    (64, 200, 32): 5,
    (64, 208, 32): 5,
    (64, 216, 32): 5,
    (64, 224, 32): 5,
    (64, 232, 32): 5,
    (64, 240, 32): 5,
    (64, 248, 32): 5,
    (64, 256, 32): 0,

    (128,  16, 32): 4,
    (128,  32, 32): 3,
    (128,  48, 32): 5,
    (128,  64, 32): 2,
    (128,  80, 32): 5,
    (128,  96, 32): 5,
    (128, 112, 32): 5,
    (128, 128, 32): 0,
    (128, 144, 32): 5,
    (128, 160, 32): 5,
    (128, 176, 32): 5,
    (128, 192, 32): 5,
    (128, 208, 32): 5,
    (128, 224, 32): 5,
    (128, 240, 32): 5,
    (128, 256, 32): 0,

}

SM100_MMA_SHAPES_F8F6F4_DENSE_2SM = {
    (128,  32, 32): 3,
    (128,  64, 32): 2,
    (128,  96, 32): 5,
    (128, 128, 32): 1,
    (128, 160, 32): 5,
    (128, 192, 32): 5,
    (128, 224, 32): 5,
    (128, 256, 32): 1,

    (256,  32, 32): 2,
    (256,  64, 32): 2,
    (256,  96, 32): 5,
    (256, 128, 32): 0,
    (256, 160, 32): 5,
    (256, 192, 32): 5,
    (256, 224, 32): 5,
    (256, 256, 32): 0,
}

# MXF8F6F4
SM100_MMA_SHAPES_MXF8F6F4_DENSE_1SM = {
    (128,  64, 32): 1,
    (128, 128, 32): 0,
    (128, 192, 32): 1,
    (128, 256, 32): 0,
}


SM100_MMA_SHAPES_MXF8F6F4_DENSE_2SM = {
    (256,  64, 32): 1,
    (256, 128, 32): 0,
    (256, 192, 32): 1,
    (256, 256, 32): 0,


}

# MXF4NVF4
SM100_MMA_SHAPES_MXF4NVF4_DENSE_1SM = {
    (128,  64, 64): 1,
    (128, 128, 64): 0,
    (128, 192, 64): 1,
    (128, 256, 64): 0,
}

SM100_MMA_SHAPES_MXF4NVF4_DENSE_2SM = {
    # Multiples of 16 for N
    (256,  64, 64): 1,
    (256, 128, 64): 0,
    (256, 192, 64): 1,
    (256, 256, 64): 0,

}
