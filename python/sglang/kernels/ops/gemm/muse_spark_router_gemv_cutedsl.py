"""CuTe DSL skinny FP32 router GEMV for MuseSpark decode: ``logits[T, E] = x[T, K] @ w[E, K]^T``.

cuBLAS runs this as a split-K GEMV plus a separate reduction kernel on the router side
stream, which then finishes after the main-stream expert input and stalls the MoE.
Here one CTA owns one (T = 1) or two weight rows over the whole K range: each thread
issues all of its 16-byte weight loads before the first FMA, streams the T token rows
of x against them, and the partial sums are reduced by a butterfly shuffle within each
warp and then across warps in a fixed order through shared memory. There is no split-K
workspace or second launch, and the summation order is fixed, so results are
deterministic (they differ from cuBLAS only by FP32 summation order).
"""

from functools import cache
from typing import Optional

import torch

from sglang.kernels.jit.utils import is_arch_support_pdl

MAX_TOKENS = 8
# GB300 sweep (cold weight, side stream): 512 threads; one expert per CTA at T = 1
# (more CTAs in flight), two above (x rows are re-read from L2 once per CTA).
_THREADS = 512
_VEC = 4  # FP32 elements per 16-byte load


def _experts_per_cta(tokens: int) -> int:
    return 1 if tokens == 1 else 2


@cache
def _is_sm100_family(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index)[0] == 10


def router_gemv_cute_eligible(x: torch.Tensor, w: torch.Tensor) -> bool:
    """FP32 decode-sized inputs with 16-byte aligned rows on SM100-family GPUs.

    With TF32 matmuls allowed (``float32_matmul_precision != "highest"``, as the
    server sets for MuseSpark) cuBLAS runs multi-row logits as a TF32 GEMM, so
    only the single-row call, which cuBLAS still runs as an FP32 GEMV, moves here.
    """
    return (
        x.is_cuda
        and x.dtype == torch.float32
        and w.dtype == torch.float32
        and x.dim() == 2
        and w.dim() == 2
        and 1 <= x.shape[0] <= MAX_TOKENS
        and (x.shape[0] == 1 or torch.get_float32_matmul_precision() == "highest")
        and x.shape[1] == w.shape[1]
        and x.device == w.device
        and x.stride(1) == 1
        and x.stride(0) % _VEC == 0
        and x.data_ptr() % 16 == 0
        and w.is_contiguous()
        and w.data_ptr() % 16 == 0
        and w.shape[1] % (_THREADS * _VEC) == 0
        and w.shape[0] % 2 == 0
        and _is_sm100_family(x.device.index)
    )


def router_gemv_cute(
    x: torch.Tensor, w: torch.Tensor, use_pdl: Optional[bool] = None
) -> torch.Tensor:
    """``x @ w.T`` as a contiguous FP32 ``[T, E]`` tensor."""
    if use_pdl is None:
        use_pdl = is_arch_support_pdl()
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32, device=x.device)
    _compile(x.shape[0], w.shape[1], bool(use_pdl))(x, w, out)
    return out


@cache
def _compile(tokens, hidden, use_pdl):
    import cutlass
    import cutlass.cute as cute

    kernel = _kernel_type()(tokens, hidden, use_pdl)
    x = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (tokens, hidden),
        (cute.sym_int64(divisibility=_VEC), 1),
        assumed_align=16,
    )
    w = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (cute.sym_int(), hidden),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (tokens, cute.sym_int()), stride_order=(1, 0), assumed_align=4
    )
    return cute.compile(
        kernel,
        x,
        w,
        out,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cache
def _kernel_type():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32

    class RouterGemv:
        def __init__(self, tokens, hidden, use_pdl):
            self.tokens = tokens
            self.experts = _experts_per_cta(tokens)
            self.warps = _THREADS // 32
            self.iters = hidden // (_THREADS * _VEC)
            self.outputs = tokens * self.experts
            self.use_pdl = use_pdl

        @cute.jit
        def __call__(self, mX: cute.Tensor, mW: cute.Tensor, mOut: cute.Tensor, stream):
            red = cute.make_layout((self.warps, self.outputs))
            self.kernel(mX, mW, mOut).launch(
                grid=[mW.shape[0] // self.experts, 1, 1],
                block=[_THREADS, 1, 1],
                smem=cute.size_in_bytes(Float32, red),
                stream=stream,
                use_pdl=self.use_pdl,
            )

        @cute.kernel
        def kernel(self, mX: cute.Tensor, mW: cute.Tensor, mOut: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            warp = tidx // 32
            lane = tidx % 32
            n0 = bidx * self.experts
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            )
            # Partition a complete CTA tile so the copy carries an explicit
            # contiguous four-value layout. A per-thread local_tile(1, 4)
            # loses vectorization and lowers to four strided scalar loads.
            copy_slice = cute.make_tiled_copy(
                atom,
                cute.make_layout(
                    ((_THREADS, 1), (_VEC, 1)),
                    stride=((_VEC, 1), (1, _THREADS * _VEC)),
                ),
                (1, _THREADS * _VEC),
            ).get_slice(tidx)
            # Every weight load of this thread goes out before the first FMA.
            ws = []
            for e in cutlass.range_constexpr(self.experts):
                for i in cutlass.range_constexpr(self.iters):
                    src = copy_slice.partition_S(
                        cute.local_tile(mW, (1, _THREADS * _VEC), (n0 + e, i))
                    )
                    frag = cute.make_fragment_like(src)
                    cute.copy(atom, src, frag)
                    ws.append(frag)
            acc = []  # index t * experts + e
            for t in cutlass.range_constexpr(self.tokens):
                xs = []
                for i in cutlass.range_constexpr(self.iters):
                    src = copy_slice.partition_S(
                        cute.local_tile(mX, (1, _THREADS * _VEC), (t, i))
                    )
                    frag = cute.make_fragment_like(src)
                    cute.copy(atom, src, frag)
                    xs.append(frag)
                for e in cutlass.range_constexpr(self.experts):
                    s = Float32(0.0)
                    for i in cutlass.range_constexpr(self.iters):
                        for j in cutlass.range_constexpr(_VEC):
                            s = s + xs[i][j] * ws[e * self.iters + i][j]
                    acc.append(s)
            for v in cutlass.range_constexpr(self.outputs):
                for step in cutlass.range_constexpr(5):
                    acc[v] = acc[v] + cute.arch.shuffle_sync_bfly(
                        acc[v], offset=1 << step
                    )
            smem = cutlass.utils.SmemAllocator()
            red = smem.allocate_tensor(
                Float32, cute.make_layout((self.warps, self.outputs)), 16
            )
            if lane == 0:
                for v in cutlass.range_constexpr(self.outputs):
                    red[warp, v] = acc[v]
            cute.arch.sync_threads()
            # Fixed cross-warp order keeps the result deterministic.
            if tidx < self.outputs:
                total = red[0, tidx]
                for wi in cutlass.range_constexpr(1, self.warps):
                    total = total + red[wi, tidx]
                t = tidx // self.experts
                mOut[t, n0 + tidx - t * self.experts] = total

    return RouterGemv
