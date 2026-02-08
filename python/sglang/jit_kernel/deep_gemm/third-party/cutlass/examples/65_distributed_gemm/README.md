# Distributed GEMM

This example implements Tensor Parallel GEMMs for the Hopper architecture with the experimental
[Distributed GEMM](../../include/cutlass/experimental/distributed) API in CUTLASS.

This example requires Hopper GPUs with an any-to-any NVLink network.
Please refer to [REQUIREMENTS.md](REQUIREMENTS.md) for more information.

By default, the example assumes 8 GPUs (TP=8) and runs an All Gather + GEMM operation, which rotates
operand A. To run with a different number of GPUs or schedule, please refer to
[65_distributed_gemm.cu](65_distributed_gemm.cu).


## Getting started

Command line arguments are mostly similar to other examples:

```
--m=<int>                   Sets the M extent of the GEMM
--n=<int>                   Sets the N extent of the GEMM
--k=<int>                   Sets the K extent of the GEMM
--l=<int>                   Sets the L extent (batch) of the GEMM (default: 1)
--alpha=<f32>               Epilogue scalar alpha (default: 1.0)
--beta=<f32>                Epilogue scalar beta (default: 0.0)
--iterations=<int>          Number of profiling iterations to perform (default: 100)
--warmup-iterations=<int>   Number of warmup iterations prior to profiling (default: 10)
--eps=<f32>                 Threshold for error compared to reference GEMM (default: 0.0)
```

Sample run command:

```bash
./65_distributed_gemm --m=16384 --n=106496 --k=16384 --warmup-iterations=10 --iterations=100
```

This executes a GEMM with shape `<16384, 106496, 16384>`, and reports average runtime
over 100 iterations, with 10 warmup iterations.
A reference check with respect to a single-device GEMM is also performed by default.

## Trying out other schedules

Schedules that are currently supported are:

* All Gather + GEMM:
  * `AllGather1D_TilingCD_RotatingA`
  * `AllGather1D_TilingCD_RotatingB`

* GEMM + Reduce Scatter:
  * `ReduceScatter1D_TilingA_RotatingC`
  * `ReduceScatter1D_TilingB_RotatingC`

To try out different schedules, simply change this line in the example, and set your desired
schedule:

```cpp
using DistSchedule = cutlass::distributed::schedules::AllGather1D_TilingCD_RotatingA<TP>;
```

If you're interesting it trying out other TP values (run on a different number of GPUs), the
procedure is the same, simply modify the following line in the example:

```cpp
using TP = _8;
```

## References
* [Distributed GEMM Blog](https://blog.shi-labs.com/distributed-gemm-88be6a481e2b)
* [Distributed GEMM Talk on CUDA Mode](https://www.youtube.com/watch?v=NHRTCQBZokg)

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

