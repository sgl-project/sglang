# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Local CuTe vocabulary gather plus the ordinary framework TP all-reduce."""

from functools import cache

import cutlass as c
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from sglang.kernels.jit.kimi_k3.host import fake
from sglang.kernels.jit.kimi_k3.primitives import launch


class EmbeddingGather:
    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (7, args[0].shape[0], 1), 128, pdl=False)

    @cute.kernel
    def kernel(self, args):
        ids, weight, out, start, vocab = args[0:5]
        lane, _, _ = cute.arch.thread_idx()
        tile, token, _ = cute.arch.block_idx()
        token_id = c.Int64(ids[token])
        row, column = token_id - start, tile * 1024 + lane * 8
        values = cute.make_rmem_tensor((8,), c.BFloat16)
        values.fill(c.BFloat16(0))
        if (token_id >= 0) & (token_id < vocab) & (row >= 0) & (row < weight.shape[0]):
            source = weight.iterator + row * 7168 + column
            cute.autovec_copy(cute.make_tensor(source, cute.make_layout((8,))), values)
        destination = out.iterator + token * 7168 + column
        cute.autovec_copy(values, cute.make_tensor(destination, cute.make_layout((8,))))


@cache
def compile_gather(dtype, shard, arch):
    batch = cute.sym_int32()
    args: tuple = (fake((batch,), None, dtype, dtype.itemsize), fake((shard, 7168)))
    args += (fake((batch, 7168)), c.Int64(0), c.Int64(0))
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--gpu-arch={arch} --enable-tvm-ffi"
    return cute.compile(EmbeddingGather(), *args, stream, options=options)
