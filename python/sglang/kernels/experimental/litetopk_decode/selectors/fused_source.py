# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the small variant with four local edits to the final large selector.

Only route ownership, two bins per thread, and the 16-warp prefix scan differ.
Kernel module names and all other device methods remain unchanged. Source hashes
identify these reviewed trimmed files; binary reference hashes are separate.
"""
import hashlib

SOURCE_SHA256 = {
    "small": "7737ac6711265de985117ac63d3fb0dfa0f66b87c666e82ba8289750cd64356d",
    "large": "7f38100428cc6fc821e9601fe4415dc8f17c0148bccfe5fe1d261298c8187f1d",
}
VENDOR_SHA256 = "115462632335e45b3ba2349ad996b7e0040b7adea75ed513031c08519626dd9f"


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"selector source anchor count != 1: {old[:100]!r}")
    return text.replace(old, new, 1)


def source(base, route):
    if route not in SOURCE_SHA256:
        raise ValueError("route must be small or large")
    if hashlib.sha256(base.encode()).hexdigest() != SOURCE_SHA256["large"]:
        raise ValueError("unqualified selector source")
    text = base
    if route == "small":
        text = replace(text, "        if route_small == cutlass.Int32(0):\n",
                       "        if route_small != cutlass.Int32(0):\n")
        start = text.index("                h0 = cutlass.Int32(0)\n")
        end = text.index("                hincl =", start)
        text = text[:start] + """                hh = cute.make_rmem_tensor((2,), cutlass.Int32)
                hs = cutlass.Int32(0)
                for hi in cutlass.range_constexpr(2):
                    hv = cutlass.Int32(0)
                    if short == cutlass.Int32(0):
                        hv = histogram[hbase + tidx * cutlass.Int32(2) + cutlass.Int32(hi)]
                    hh[hi] = hv
                    hs = hs + hv
""" + text[end:]
        text = replace(text, "                warp_total = s_cbuf[lane]\n",
                       "                warp_total = cutlass.Int32(0)\n"
                       "                if lane < cutlass.Int32(NW):\n"
                       "                    warp_total = s_cbuf[lane]\n")
        start = text.index("                run0 = hbefore + hincl - hs\n")
        end = text.index("                cute.arch.barrier()  # crossing", start)
        text = text[:start] + """                running = hbefore + hincl - hs
                if short == cutlass.Int32(0):
                    for hi in cutlass.range_constexpr(2):
                        after = running + hh[hi]
                        if running <= k and k < after:
                            s_cbuf[NW + 0] = tidx * cutlass.Int32(2) + cutlass.Int32(hi)
                            s_cbuf[NW + 1] = running
                            s_cbuf[NW + 2] = hh[hi]
                        running = after
                    if tidx == cutlass.Int32(BLK - 1):
                        s_cbuf[NW + 3] = running
""" + text[end:]
    compile(text, f"fused_{route}.py", "exec")
    if hashlib.sha256(text.encode()).hexdigest() != SOURCE_SHA256[route]:
        raise ValueError(f"{route} selector generation drift")
    return text
