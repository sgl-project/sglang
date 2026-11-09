# The GEMM and parts of the epilogue are adapted from the vendored FlashInfer kernel in
# kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py (Apache-2.0); this file has been modified.
"""Shared-expert act_and_mul in the SM107 gate_up split-K epilogue (default off).

SGLANG_ENABLE_SM107_SHARED_EXPERT_FUSED_ACT=1, with the SM107 small-M routes on (sm107_bf16_smallm): at 1-16 rows the
routes run the shared expert's gate_up (N 1280, K 2560) on FlashInfer 0.7.2's cluster split-K kernel at tactic
(64, 8, 4, 5), and the stock SiluAndMul then launches the JIT act_and_mul kernel (kernels/jit/csrc/elementwise/
activation.cuh, built with --use_fast_math) on the BF16 gate_up output. Here Qwen2MoeMLP.forward runs one kernel
instead: the same split-K GEMM, whose epilogue writes silu(gate) * up.

Output bits equal the stock pair's:
* The kernel is the vendored copy of that FlashInfer kernel (kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py;
  its TMA, MMA and rank-ordered FP32 split-K reduction are FlashInfer 0.7.2's), at the same tactic. Only the epilogue
  warp is overridden, after the reduction.
* One CTA owns the gate and up tiles of the same columns: the kernel reads a row-permuted copy of the weight in which
  kernel-M tile t holds gate rows 32t..32t+31, then up rows 640+32t..640+32t+31. A row's dot product does not depend
  on where its row sits in the tile (same MMA instruction, same K order).
* The epilogue rounds the FP32 sum to BF16 with the stock store's conversion, widens it back (exact), stages the tile
  in shared memory as the vendored "gate" epilogue does, and computes act_and_mul_kernel's
  float(bf16(gate / (1 + expf(-gate)) * up)) with the instructions nvcc emits for it under --use_fast_math:
  mul.ftz by -log2(e), ex2.approx.ftz, add.ftz, div.approx.ftz, mul.ftz, then cvt.rn.bf16. Every finite (gate, up)
  BF16 pair was checked equal to the stock kernel's output.

The saving is per call; whether a step gets shorter depends on whether the shared-expert branch is on the critical
path, which it is not when it already finishes before the routed experts.

Memory: each shared-expert gate_up weight gets a row-permuted copy (2 x 640 x 2560 BF16 = 6.25 MiB per layer), made
at its first eager call, outside the KV budget. At load, ``precompile`` checks the kernel against the stock pair,
including the gate < -87.3 edge where only the fast-math divide matches, and turns the fused act off on a mismatch.
"""

import logging
import types
import weakref
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

# The route this replaces: sm107_bf16_smallm.ROUTES[(m, 1280, 2560)] for m in 1..16.
_INTER, _HIDDEN = 640, 2560
_N, _K = 2 * _INTER, _HIDDEN
_ROUTE = ("cl", (64, 8, 4, 5, 128))
_MAX_ROWS = 16
_TILE_M = 64
_HALF = (
    _TILE_M // 2
)  # gate rows per kernel-M tile; the other half are the up rows of the same columns
_BARRIER_ID = 7  # named barrier among the 4 epilogue warps (the vendored gate epilogue uses the same id)

# act_and_mul_kernel's silu(g) * u under --use_fast_math, one PTX instruction per C operation:
# expf(-g) -> ex2.approx.ftz(g * -log2(e)); 1 + e; g / d -> div.approx.ftz; * u.
SILU_MUL_PTX = (
    "{\n\t.reg .f32 ar_t, ar_e, ar_d, ar_s;\n\t"
    "mul.ftz.f32 ar_t, $1, 0fBFB8AA3B;\n\t"
    "ex2.approx.ftz.f32 ar_e, ar_t;\n\t"
    "add.ftz.f32 ar_d, ar_e, 0f3F800000;\n\t"
    "div.approx.ftz.f32 ar_s, $1, ar_d;"
    "\n\tmul.ftz.f32 $0, ar_s, $2;\n\t}"
)

_enabled = False
_deps = None  # modules _ineligible reads, resolved at install
_prepared_weights = (
    weakref.WeakKeyDictionary()
)  # gate_up_proj -> (weight data_ptr, row-permuted weight)
_km = None  # the CuTe DSL side, built on first use
_compiled: dict = {}
_anchors: dict = {}  # device -> (16, 1280) BF16 shape anchor for the kernel's C operand (never written)
_logged: set = set()


def is_enabled() -> bool:
    return _enabled


def permute_gate_up(weight: torch.Tensor) -> torch.Tensor:
    """(2I, K) [gate; up] -> (2I, K) with 64-row tile t = gate rows 32t..32t+31, then up rows I+32t..I+32t+31."""
    n, k = weight.shape
    inter = n // 2
    if n % 2 or inter % _HALF:
        raise ValueError(f"gate_up rows {n} must be 2 x a multiple of {_HALF}")
    return (
        weight.view(2, inter // _HALF, _HALF, k).transpose(0, 1).contiguous().view(n, k)
    )


def install() -> None:
    """Called by sm107_bf16_smallm.install once its routes are on."""
    global _enabled, _deps
    if _deps is None:
        _deps = _resolve_deps()
    _enabled = True
    logger.info(
        "SM107 shared-expert act_and_mul fused into the gate_up split-K epilogue at 1-%d rows (N %d, K %d)",
        _MAX_ROWS,
        _N,
        _K,
    )


def _resolve_deps():
    from types import SimpleNamespace

    from sglang.kernels.fused_op import get_fused_op_backend
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.quantization import sm107_bf16_smallm, unquant
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph,
    )

    return SimpleNamespace(
        get_fused_op_backend=get_fused_op_backend,
        SiluAndMul=SiluAndMul,
        routes=sm107_bf16_smallm,
        unquant=unquant,
        is_in_tc_piecewise_cuda_graph=is_in_tc_piecewise_cuda_graph,
    )


def _fallback(m: int, reason: str) -> None:
    if reason not in _logged:
        _logged.add(reason)
        logger.debug(
            "shared-expert fused act_and_mul not taken at %d rows: %s", m, reason
        )


def _ineligible(mlp, x: torch.Tensor, m: int) -> Optional[str]:
    """Why the stock pair at this call is not (the gate_up cluster route + JIT act_and_mul) on contiguous operands."""
    d = _deps
    if torch.compiler.is_compiling():
        return "compiling"
    if d.is_in_tc_piecewise_cuda_graph():
        return "tc_piecewise"
    gate_up = mlp.gate_up_proj
    weight = gate_up.weight
    if mlp._enable_silu_fp4_quant_fusion:
        return "fp4_quant_fusion"
    if (
        type(gate_up.quant_method) is not d.unquant.UnquantizedLinearMethod
        or gate_up.bias is not None
    ):
        return "linear_method"
    if gate_up.gather_output:
        return "gather_output"
    if (
        not d.unquant._sm107_smallm_on
        or d.routes._routes is None
        or not d.routes._checked
        or d.routes._routes.get((m, _N, _K)) != _ROUTE
    ):
        return "route"
    backend = d.unquant.get_bf16_gemm_backend()
    if not (backend.is_cutedsl() or backend.is_gemv()):
        return "backend"
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and not weight.requires_grad
    ):
        return "dtype"
    if not (
        x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 32 == 0
        and weight.data_ptr() % 32 == 0
    ):
        return "layout"
    act_fn = mlp.act_fn
    if type(act_fn) is not d.SiluAndMul or d.get_fused_op_backend() is not None:
        return "act_fn"
    method = act_fn._forward_method
    if method is None:
        method = act_fn._resolve_forward_method()
    if not (
        isinstance(method, types.MethodType)
        and method.__func__ is d.SiluAndMul.forward_cuda
    ):
        return "act_fn"
    return None


def _prepared(gate_up, device: torch.device):
    """The row-permuted weight and the shape anchor, built outside graph capture (None while capturing)."""
    weight = gate_up.weight
    state = _prepared_weights.get(gate_up)
    anchor = _anchors.get(device.index)
    if state is not None and state[0] == weight.data_ptr() and anchor is not None:
        return state[1], anchor
    if torch.cuda.is_current_stream_capturing():
        return None
    if state is None or state[0] != weight.data_ptr():
        state = _prepared_weights[gate_up] = (
            weight.data_ptr(),
            permute_gate_up(weight.detach()),
        )
    if anchor is None:
        anchor = _anchors[device.index] = torch.empty(
            (_MAX_ROWS, _N), dtype=torch.bfloat16, device=device
        )
    return state[1], anchor


def fused_act(mlp, x: torch.Tensor) -> Optional[torch.Tensor]:
    """silu(gate) * up of mlp.gate_up_proj(x) from one kernel, or None (the stock pair runs)."""
    if not _enabled or x.dim() != 2:
        return None
    m = x.shape[0]
    if not 1 <= m <= _MAX_ROWS or tuple(mlp.gate_up_proj.weight.shape) != (_N, _K):
        return None
    reason = _ineligible(mlp, x, m)
    if reason is not None:
        _fallback(m, reason)
        return None
    prepared = _prepared(mlp.gate_up_proj, x.device)
    if prepared is None:
        _fallback(m, "capture_before_warmup")
        return None
    w_perm, anchor = prepared
    out = torch.empty((m, _INTER), dtype=x.dtype, device=x.device)
    run(x, w_perm, out, anchor[:m])
    if m not in _logged:
        _logged.add(m)
        logger.debug("shared-expert fused act_and_mul at %d rows", m)
    return out


def _kernel_module():
    """Define the CuTe DSL kernel on first use (cutlass loads only when the switch is on)."""
    global _km
    if _km is not None:
        return _km
    from types import SimpleNamespace

    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass import Int32
    from cutlass._mlir.dialects import llvm
    from cutlass.cute import experimental as cute_ext
    from cutlass.cutlass_dsl import T, dsl_user_op

    from sglang.kernels.ops.gemm import dense_bf16_gemm_sm100_splitk_epilogue as epi

    def make_silu_mul(asm: str):
        @dsl_user_op
        def silu_mul(g, u, *, loc=None, ip=None):
            return cutlass.Float32(
                llvm.inline_asm(
                    T.f32(),
                    [g.ir_value(loc=loc, ip=ip), u.ir_value(loc=loc, ip=ip)],
                    asm,
                    "=f,f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            )

        return silu_mul

    class ActMulSplitKKernel(epi.SplitKDenseGemmKernel):
        """The vendored cluster split-K GEMM; the owner rank's epilogue writes silu(gate) * up, not gate_up."""

        def __init__(self, tactic, silu_mul_ptx: str):
            super().__init__(tactic=tactic, use_pdl=True, has_bias=False)
            if tactic.mma_m != _TILE_M:
                raise ValueError(f"act_and_mul epilogue needs mma_m={_TILE_M}")
            if (_HALF * tactic.mma_n) % self.epilog_threads:
                raise ValueError(
                    f"{_HALF} x {tactic.mma_n} outputs do not split over {self.epilog_threads} threads"
                )
            # kernel() allocates the (cta_m, cta_n) FP32 staging tile for mode "gate" only; epilog_warp is ours.
            self.epilogue_mode = "gate"
            self.silu_mul = make_silu_mul(silu_mul_ptx)

        @cute.experimental.jit
        def epilog_warp(
            self,
            bar_tma_epilog,
            bar_mma_epilog,
            bar_tmem_alloc,
            tmem_base_ptr,
            acc_layout: cutlass.Constexpr,
            gD_tile: cute.Tensor,
            gBias_tile: cute.Tensor,
            epi_tid: cutlass.Int32,
            c_dtype: cutlass.Constexpr,
            d_layout: cutlass.Constexpr,
            mailbox,
            bar_reduce,
            split_rank: cutlass.Int32,
            gate_tile,
            mX: cute.Tensor,
            mOut: cute.Tensor,
            bidx: cutlass.Int32,
            n_idx: cutlass.Int32,
        ):
            # Verbatim from the vendored epilog_warp (has_bias=False): TMEM -> RMEM, then the rank-ordered reduce.
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
            epi._tcgen05_fence(before=False)

            acc_view = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
                acc_layout,
            )[((None, None), 0, 0, 0)]

            epi_tile = (self.cta_m, self.cta_n)
            tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (self.cta_m, self.cta_n, self.cta_k),
                    d_layout,
                    c_dtype,
                    self.acc_dtype,
                    epi_tile,
                    False,
                ),
                acc_view,
            )
            gD_epi = cute.flat_divide(gD_tile, epi_tile)
            rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
            rAcc = cute_ext.allocate(
                self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
            )
            rD = cute_ext.allocate(
                c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
            )
            thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

            cute.arch.mbarrier_wait(bar_mma_epilog, 0)
            epi._tcgen05_fence(before=False)
            cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
            cute.arch.fence_view_async_tmem_load()
            epi._tcgen05_fence(before=True)
            cute.arch.mbarrier_arrive(bar_tmem_alloc)

            if cutlass.const_expr(self.split_k > 1):
                values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
                values_per_peer = cutlass.const_expr(
                    self.epilog_threads * values_per_thread
                )
                if split_rank != epi.OWNER_RANK:
                    for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                        epi._store_shared_remote_v4(
                            rAcc[value_idx],
                            rAcc[value_idx + 1],
                            rAcc[value_idx + 2],
                            rAcc[value_idx + 3],
                            mailbox.iterator
                            + (split_rank - Int32(1)) * values_per_peer
                            + epi_tid * values_per_thread
                            + value_idx,
                            bar_reduce,
                            Int32(epi.OWNER_RANK),
                        )
                else:
                    if epi_tid == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bar_reduce, self.expected_transaction_bytes
                        )
                    epi._mbarrier_wait_acquire_cluster(bar_reduce, Int32(0))
                    for peer in cutlass.range_constexpr(self.split_k - 1):
                        for value_idx in cutlass.range_constexpr(values_per_thread):
                            rAcc[value_idx] = (
                                rAcc[value_idx]
                                + mailbox[
                                    peer * values_per_peer
                                    + epi_tid * values_per_thread
                                    + value_idx
                                ]
                            )

            # The act_and_mul epilogue (owner rank).
            if split_rank == epi.OWNER_RANK:
                # The stock store's BF16 rounding of gate_up, widened back exactly (act_and_mul reads that BF16).
                rD.store(rAcc.load().to(c_dtype))
                rAcc.store(rD.load().to(self.acc_dtype))
                sStage_epi = cute.flat_divide(gate_tile, epi_tile)
                cute_ext.partition_and_copy(thr_t2r, rAcc, sStage_epi[None, None, 0, 0])
                cute.arch.barrier(
                    barrier_id=_BARRIER_ID, number_of_threads=self.epilog_threads
                )

                rows = cute.size(mOut, mode=[0])
                for it in cutlass.range_constexpr(
                    (_HALF * self.cta_n) // self.epilog_threads
                ):
                    elem = it * self.epilog_threads + epi_tid
                    j_local = elem % _HALF
                    m_local = elem // _HALF
                    m_global = n_idx * self.cta_n + m_local
                    if m_global < rows:
                        gate = gate_tile[j_local, m_local]
                        up = gate_tile[_HALF + j_local, m_local]
                        mOut[m_global, bidx * _HALF + j_local] = self.silu_mul(
                            gate, up
                        ).to(c_dtype)

    @cute.experimental.jit
    def bmm_actmul(
        gemm_op: cutlass.Constexpr,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        out: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        # As the vendored _bmm_gate: C is only the grid's shape anchor here; out is the (rows, I) act output.
        c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
        gemm_op(
            cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
            cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
            c,
            cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
            c,
            out,
            stream,
        )

    _km = SimpleNamespace(
        epi=epi,
        cute_ext=cute_ext,
        stream=cuda_driver.CUstream,
        kernel=ActMulSplitKKernel,
        bmm=bmm_actmul,
    )
    return _km


def run(
    x: torch.Tensor,
    w_perm: torch.Tensor,
    out: torch.Tensor,
    anchor: torch.Tensor,
    *,
    silu_mul_ptx: str = SILU_MUL_PTX,
) -> torch.Tensor:
    """out[m, j] = bf16(silu(g) * u), g/u = bf16(x @ w_perm.T) at the gate/up rows of column j; anchor is (m, 2I).

    ``silu_mul_ptx`` exists for tests, which build a deliberately different epilogue as a control.
    """
    km = _kernel_module()
    m = x.shape[0]
    tensors = km.epi._to_cute_swap(x, w_perm.t(), anchor, None)
    key = (x.get_device(), tensors[4], silu_mul_ptx)
    compiled = _compiled.get(key)
    if compiled is None:
        with torch.cuda.device(x.device):
            kernel = km.kernel(km.epi.SplitKTactic(*_ROUTE[1][:4]), silu_mul_ptx)
            rep = km.epi._make_compile_repr_tensors(x.dtype, False, *tensors[4])
            out_rep = km.epi._from_dlpack_dynamic(
                torch.empty((8, _TILE_M), dtype=x.dtype, device=x.device), 1
            )
            compiled = km.cute_ext.compile(
                km.bmm,
                kernel,
                *rep[:3],
                out_rep,
                km.stream(torch.cuda.current_stream().cuda_stream),
            )
        _compiled[key] = compiled
    assert (
        out.shape == (m, w_perm.shape[0] // 2)
        and out.stride(1) == 1
        and out.data_ptr() % 32 == 0
    )
    compiled(
        *tensors[:3],
        km.epi._from_dlpack_dynamic(out, 1),
        km.stream(torch.cuda.current_stream(x.device).cuda_stream),
    )
    return out


def precompile(
    *,
    stock_gemm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device=None,
) -> bool:
    """JIT-compile the kernel at load and check it bit for bit against the stock pair (``stock_gemm``, then the JIT
    act_and_mul): at 1-16 rows on random operands, and on one-hot rows that reach the epilogue with the edge pair
    (gate -87.5, up -80), where 1 + exp(-gate) passes 2^126 and only the fast-math divide gives the stock result. A
    mismatch turns the fused act off. Returns whether it stays on."""
    global _enabled
    if not _enabled:
        return False
    from sglang.srt.layers.activation import silu_and_mul

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(0)
    w = (torch.randn(_N, _K, generator=gen, device=device) * 0.02).to(torch.bfloat16)
    cases = [
        (
            (torch.randn(m, _K, generator=gen, device=device) * 4).to(torch.bfloat16),
            w,
        )
        for m in range(1, _MAX_ROWS + 1)
    ]
    edge = w.clone()
    edge[600, 5], edge[_INTER + 600, 5] = -87.5, -80.0
    one_hot = torch.zeros(_MAX_ROWS, _K, dtype=torch.bfloat16, device=device)
    rows = torch.arange(_MAX_ROWS, device=device)
    # Row 5 then reaches the epilogue with exactly (gate, up) = edge[(600, 1240), 5].
    one_hot[rows, rows] = 1.0
    cases.append((one_hot, edge))
    anchor = torch.empty(_MAX_ROWS, _N, dtype=torch.bfloat16, device=device)
    for x, weight in cases:
        m = x.shape[0]
        out = torch.empty(m, _INTER, dtype=torch.bfloat16, device=device)
        got = run(x, permute_gate_up(weight), out, anchor[:m])
        want = silu_and_mul(stock_gemm(x, weight))
        if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
            logger.warning(
                "SGLANG_ENABLE_SM107_SHARED_EXPERT_FUSED_ACT: the fused act differs from the stock pair at %d rows on "
                "this build; the stock pair stays",
                m,
            )
            _enabled = False
            return False
    return True
