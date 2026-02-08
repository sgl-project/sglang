# Examples of using the CUTLASS Python interface

* [00_basic_gemm](/examples/python/00_basic_gemm.ipynb)

    Shows how declare, configure, compile, and run a CUTLASS GEMM using the Python interface

* [01_epilogue](/examples/python/01_epilogue.ipynb)

    Shows how to fuse elementwise activation functions to GEMMs via the Python interface

* [02_pytorch_extension_grouped_gemm](/examples/python/02_pytorch_extension_grouped_gemm.ipynb)

    Shows how to declare, compile, and run a grouped GEMM operation via the Python interface,
    along with how the emitted kernel can be easily exported to a PyTorch CUDA extension.

* [03_basic_conv2d](/examples/python/03_basic_conv2d.ipynb)

    Shows how to declare, configure, compile, and run a CUTLASS Conv2d using the Python interface

* [04_epilogue_visitor](/examples/python/04_epilogue_visitor.ipynb)

    Shows how to fuse elementwise activation functions to GEMMs via the Python Epilogue Visitor interface

# Copyright

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
