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

# this file creates the test/unit/gemm/device simt tests


outputDir = ""

################################################################################
# parameters
# Edge - for tiles, the edges represent the length of one side
# Ratio - the maximum ratio between 2 edges, limits the skinnyness of tiles
# MaxEdge - maximum length of each edge
# Min/Max - minimum/maximum of the product of edge lengths
################################################################################

warpsPerThreadblockEdge = [1, 2, 4, 8, 16]
warpsPerThreadblockRatio = 2
warpsPerThreadblockMax = 16
# NOTE 1x32 and 2x16 warp tile shapes fail validation for ~10% of cases

warpShapeEdges = [8, 16, 32, 64, 128, 256]
warpShapeRatio = 4
warpShapeMax = 64*64
warpShapeMin = 8*8

threadblockEdgeMax = 256

#      char,      type               bits/elem, max tile,   L0 threadblock tiles
precisions = [
       ["c", "cutlass::complex<float>",     64,  64*128, [ [ 64, 128], [ 64,  32]             ] ],
       ["q", "cutlass::Quaternion<float>",  64,  64*128, [ [ 64, 128], [ 64,  32]             ] ],
       ["d", "double",                      64,   64*64, [ [ 64,  64], [ 32,  32]             ] ],
       ["h", "cutlass::half_t",             16, 128*256, [ [256, 128], [ 64, 128], [ 64,  32] ] ],
       ["i", "int",                         32, 128*128, [ [128,  64], [ 16, 32]              ] ],
       ["s", "float",                       32, 128*128, [ [128, 256], [128, 128], [ 64,  64] ] ],
       ["z", "cutlass::complex<double>",   128,   64*64, [ [ 32,  64], [ 16,  32]             ] ],
       ]
# L1 will have a single kernel for every unique shape
# L2 will have everything else

transposes = [
       [False, False],
       [False, True],
       [True, False],
       [True, True]
       ]

################################################################################
# warps per threadblock
################################################################################
warpsPerThreadblocks = []
for warpsPerThreadblock0 in warpsPerThreadblockEdge:
    for warpsPerThreadblock1 in warpsPerThreadblockEdge:
        if warpsPerThreadblock0 / warpsPerThreadblock1 <= warpsPerThreadblockRatio and warpsPerThreadblock1 / warpsPerThreadblock0 <= warpsPerThreadblockRatio and warpsPerThreadblock0 * warpsPerThreadblock1 <= warpsPerThreadblockMax:
            warpsPerThreadblocks.append([warpsPerThreadblock0,
                warpsPerThreadblock1])
print("WarpsPerThreadblocks",warpsPerThreadblocks)

################################################################################
# warp shapes
################################################################################
warpNumThreads = 32
warpShapes = []
for warp0 in warpShapeEdges:
    for warp1 in warpShapeEdges:
        if warp0 / warp1 <= warpShapeRatio and warp1 / warp0 <= warpShapeRatio and warp0*warp1 <= warpShapeMax and warp0*warp1 > warpShapeMin:
            warpShapes.append([warp0, warp1])
print("WarpShapes", warpShapes)

numL0 = 0
numL1 = 0
numL2 = 0

################################################################################
# create kernels
# create a file for each precision/transpose
# each file contains many tile sizes
################################################################################

# precisions
for precision in precisions:

    # get precision char
    precisionChar = precision[0]
    precisionType = precision[1]
    precisionBits = precision[2]
    threadblockMaxElements = precision[3]
    threadblockTilesL0 = precision[4]

    # transposes
    for transpose in transposes:

        # get transpose char
        columnMajorA = transpose[0]
        columnMajorB = transpose[1]
        transCharA = "n" if columnMajorA else "t"
        transCharB = "n" if columnMajorB else "t"

        # open file
        fileName="simt_%sgemm_%s%s_sm50.cu" % (precisionChar, transCharA, transCharB)
        print("\n", fileName)
        filePath = "%s%s" % (outputDir, fileName)
        out = open(filePath, "w+")

        # write file header
        out.write("/***************************************************************************************************\n"
" * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.                 \n"
" * SPDX-License-Identifier: BSD-3-Clause                                                           \n"
" *                                                                                                 \n"
" * Redistribution and use in source and binary forms, with or without                              \n"
" * modification, are permitted provided that the following conditions are met:                     \n"
" *                                                                                                 \n"
" * 1. Redistributions of source code must retain the above copyright notice, this                  \n"
" * list of conditions and the following disclaimer.                                                \n"
" *                                                                                                 \n"
" * 2. Redistributions in binary form must reproduce the above copyright notice,                    \n"
" * this list of conditions and the following disclaimer in the documentation                       \n"
" * and/or other materials provided with the distribution.                                          \n"
" *                                                                                                 \n"
" * 3. Neither the name of the copyright holder nor the names of its                                \n"
" * contributors may be used to endorse or promote products derived from                            \n"
" * this software without specific prior written permission.                                        \n"
" *                                                                                                 \n"
" * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\"                   \n"
" * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE                       \n"
" * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                  \n"
" * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE                    \n"
" * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL                      \n"
" * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR                      \n"
" * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER                      \n"
" * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,                   \n"
" * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE                   \n"
" * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                            \n"
" *\n"
" **************************************************************************************************/\n"
"/*! \\file\n"
"    \\brief Tests for device-wide GEMM interface\n"
"*/\n"
"\n"
"#include <iostream>\n"
"\n"
"#include \"cutlass/cutlass.h\"\n"
"#include \"cutlass/gemm/device/gemm.h\"\n"
"#include \"cutlass/numeric_types.h\"\n"
"\n"
"#include \"../../common/cutlass_unit_test.h\"\n"
"\n"
"#include \"cutlass/util/host_tensor.h\"\n"
"#include \"cutlass/util/tensor_view_io.h\"\n"
"#include \"cutlass/util/reference/host/tensor_fill.h\"\n"
"#include \"cutlass/util/reference/host/tensor_copy.h\"\n"
"#include \"cutlass/util/reference/host/tensor_compare.h\"\n"
"#include \"cutlass/util/reference/host/gemm.h\"\n"
"\n"
"#include \"testbed.h\"\n"
"\n")
        foundThreadblockTilesL0 = {}
        foundThreadblockTilesL1 = {}

        ########################################################################
        # for each combination of tile sizes
        ########################################################################
        for warpsPerThreadblock in warpsPerThreadblocks:
            for warpShape in warpShapes:
                warpThreadsM = 0
                if warpShape[0] > warpShape[1]:
                    warpThreadsM = 8
                else:
                    warpThreadsM = 4
                warpThreadsN = warpNumThreads / warpThreadsM

                # skip shapes with conflicting rectangularity
                # they are unlikely to be fastest
                blockG = warpsPerThreadblock[0] > warpsPerThreadblock[1]
                blockL = warpsPerThreadblock[0] < warpsPerThreadblock[1]
                warpG = warpShape[0] > warpShape[1]
                warpL = warpShape[0] < warpShape[1]

                blockG2 = warpsPerThreadblock[0] > warpsPerThreadblock[1]*2
                blockL2 = warpsPerThreadblock[0]*2 < warpsPerThreadblock[1]
                warpG2 = warpShape[0] > warpShape[1]*2
                warpL2 = warpShape[0]*2 < warpShape[1]

                if blockG2 and warpL: continue
                if blockL2 and warpG: continue
                if warpG2 and blockL: continue
                if warpL2 and blockG: continue

                # check threadblock ratios and max
                threadblockTile = [warpShape[0]*warpsPerThreadblock[0],
                        warpShape[1]*warpsPerThreadblock[1]]
                if threadblockTile[0] * threadblockTile[1] > threadblockMaxElements: continue
                if threadblockTile[0] > threadblockEdgeMax: continue
                if threadblockTile[1] > threadblockEdgeMax: continue
                totalThreads = warpNumThreads*warpsPerThreadblock[0]*warpsPerThreadblock[1]

                # calculate unroll
                # ensure that every iteration at least a full load of A,B are done
                unrollMin = 8
                unrollMin0 = totalThreads / threadblockTile[0]
                unrollMin1 = totalThreads / threadblockTile[1]
                unroll = max(unrollMin, unrollMin0, unrollMin1)

                threadTileM = warpShape[0] / warpThreadsM
                threadTileN = warpShape[1] / warpThreadsN
                if threadTileM < 2 or threadTileN < 2: continue
                if threadTileM*threadTileN*precisionBits > 8*8*32: continue

                # epilogue currently only supports N < WarpNumThreads
                if threadblockTile[1] < warpNumThreads: continue

                # limit smem
                smemBitsA = threadblockTile[0]*unroll*2*precisionBits
                smemBitsB = threadblockTile[1]*unroll*2*precisionBits
                smemKBytes = (smemBitsA+smemBitsB)/8/1024
                if (smemKBytes > 48): continue

                # test level 0
                testLevel = -1
                for tileId in range(0, len(threadblockTilesL0)):
                    tbTile = threadblockTilesL0[tileId]
                    if tbTile[0] == threadblockTile[0] and tbTile[1] == threadblockTile[1]:
                        if tuple(tbTile) not in foundThreadblockTilesL0:
                            testLevel = 0
                            numL0 += 1
                            foundThreadblockTilesL0[tuple(tbTile)] = True

                # test level 1
                if testLevel < 0:
                    threadblockTileAlreadyUsed = False
                    if tuple(threadblockTile) not in foundThreadblockTilesL1:
                        testLevel = 1
                        numL1 += 1
                        foundThreadblockTilesL1[tuple(threadblockTile)] = True

                # test level 2
                if testLevel < 0:
                    testLevel = 2
                    numL2 += 1

                ################################################################
                # write this tile to file
                ################################################################

                print("%ix%ix%i__%ix%i_%ix%i_%ix%i L%i" % (
                        threadblockTile[0], threadblockTile[1], unroll,
                        threadTileM, threadTileN,
                        warpThreadsM, warpThreadsN,
                        warpsPerThreadblock[0], warpsPerThreadblock[1], testLevel))

                out.write("////////////////////////////////////////////////////////////////////////////////\n"
                        "// Elements / Thread: %3i x %3i\n"
                        "//    Threads / Warp: %3i x %3i\n"
                        "//     Warps / Block: %3i x %3i\n"
                        "//       Threadblock: %3i x %3i x %2i\n"
                        % ( threadTileM, threadTileN,
                            warpThreadsM, warpThreadsN,
                            warpsPerThreadblock[0], warpsPerThreadblock[1],
                            threadblockTile[0], threadblockTile[1], unroll
                            )
                        )

                out.write("CUTLASS_TEST_L%i(SM50_device_%sgemm_%s%s, %ix%ix%i_%ix%ix1_%ix%i_%ix%i_%ix%i, {\n" % (
                    testLevel,
                    precisionChar,
                    transCharA,
                    transCharB,
                    threadblockTile[0],
                    threadblockTile[1],
                    unroll,
                    warpShape[0],
                    warpShape[1],
                    threadTileM,
                    threadTileN,
                    warpThreadsM,
                    warpThreadsN,
                    warpsPerThreadblock[0],
                    warpsPerThreadblock[1]
                    ))
                out.write("    using precision = %s;\n" % precisionType)
                out.write("    using ThreadblockShape = cutlass::gemm::GemmShape<%i, %i, %i>;\n" % (
                    threadblockTile[0],
                    threadblockTile[1],
                    unroll))
                out.write("    using WarpShape = cutlass::gemm::GemmShape<%i, %i, %i>;\n\n" % (
                    warpShape[0],
                    warpShape[1],
                    unroll))
                out.write("    static int const kEpilogueElementsPerAccess = 1;\n"
                    "    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;\n"
                    "    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<\n"
                    "        precision, kEpilogueElementsPerAccess, precision, precision>;\n\n")

                out.write("    using Gemm = cutlass::gemm::device::Gemm<\n"
                    "        precision, cutlass::layout::%sMajor,\n"
                    "        precision, cutlass::layout::%sMajor,\n"
                    "        precision, cutlass::layout::RowMajor,\n"
                    "        precision,\n"
                    "        cutlass::arch::OpClassSimt,\n"
                    "        cutlass::arch::Sm50,\n"
                    "        ThreadblockShape, WarpShape, InstructionShape,\n"
                    "        EpilogueOutputOp,\n"
                    "        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,\n"
                    "        2 // Stages\n"
                    "    >;\n" % (
                        "Column" if columnMajorA else "Row",
                        "Column" if columnMajorB else "Row",
                        ))
                out.write("    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());\n"
                    "} )\n\n")


        out.close()
print("NumKernels:", numL0, numL1, numL2)

