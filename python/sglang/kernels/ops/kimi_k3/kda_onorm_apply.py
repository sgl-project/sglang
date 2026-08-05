"""Fused gated-RMSNorm apply for the two-stage split verify path.

Stage A (kda_decode_mtp_split, onorm off) emits raw bf16 per-token outputs
with no V-slice cluster; this kernel performs the per-(token, head) RMS
reduction and the gate/weight apply in a second, cluster-free launch. The
verify kernel therefore never co-schedules clusters of V-slice CTAs, which
was the multi-request cliff for the split family (see
benchmark/bench_linear_attention/WORKLOG.md, [2stage]).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

WARP_SIZE = 32
HEAD_DIM = 128
BLOCK_WARPS = 8


@cute.kernel
def _kda_onorm_apply_kernel(
    raw: cute.Tensor,  # [1, T, H, K] bf16 stage-A output (cluster-free noonorm)
    gate: cute.Tensor,  # [1, T, H, K] bf16 silu gate
    weight: cute.Tensor,  # [K] fp32 norm weight
    out: cute.Tensor,  # [1, T, H, K] bf16 final
    H: cutlass.Constexpr[int],
    NUM_ROWS: cutlass.Constexpr[int],
    eps: cutlass.Constexpr[float],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    cute.arch.griddepcontrol_wait()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    in_warp_tid = tidx % WARP_SIZE
    row = bidx * BLOCK_WARPS + warp_idx
    if row < NUM_ROWS:
        tok = row // H
        head = row % H
        r_v = cute.make_rmem_tensor(
            cute.make_layout((HEAD_DIM // WARP_SIZE,), stride=(1,)), cutlass.Float32
        )
        sumsq = cutlass.Float32(0.0)
        for c in range(HEAD_DIM // WARP_SIZE):
            idx = in_warp_tid * (HEAD_DIM // WARP_SIZE) + c
            v = cutlass.Float32(raw[0, tok, head, idx])
            r_v[c] = v
            sumsq = cute.fma(v, v, sumsq)
        for offset in [16, 8, 4, 2, 1]:
            sumsq += cute.arch.shuffle_sync_bfly(
                sumsq, offset=offset, mask=-1, mask_and_clamp=31
            )
        rms = cute.math.rsqrt(sumsq / cutlass.Float32(HEAD_DIM) + eps, fastmath=True)
        for c in range(HEAD_DIM // WARP_SIZE):
            idx = in_warp_tid * (HEAD_DIM // WARP_SIZE) + c
            gate_raw = cutlass.Float32(gate[0, tok, head, idx])
            g = cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-gate_raw, fastmath=True)
            )
            out[0, tok, head, idx] = cutlass.BFloat16(
                r_v[c] * rms * cutlass.Float32(weight[idx]) * g
            )
    # Signal dependents (the next layer's PDL-launched verify kernel) as soon
    # as our rows are written; their wait overlaps with our drain.
    cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _run_kda_onorm_apply(
    raw: cute.Tensor,
    gate: cute.Tensor,
    weight: cute.Tensor,
    out: cute.Tensor,
    H: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    eps: cutlass.Constexpr[float],
    stream: cuda.CUstream,
):
    rows = T * H
    _kda_onorm_apply_kernel(raw, gate, weight, out, H, rows, eps).launch(
        grid=((rows + BLOCK_WARPS - 1) // BLOCK_WARPS, 1, 1),
        block=[BLOCK_WARPS * WARP_SIZE, 1, 1],
        stream=stream,
        use_pdl=True,
    )


_ONORM_APPLY_COMPILED = {}


def kda_onorm_apply(*, raw, gate, weight, out, eps):
    """Normalize ``raw`` (fp32) into ``out`` (bf16) with the sigmoid gate and
    norm weight, one warp per (token, head) row."""
    import torch

    from sglang.kernels.ops.kimi_k3.kda_decode_mtp_split import _cute_tensor

    T = raw.shape[1]
    H = raw.shape[2]
    if raw.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("expected raw bf16 -> out bf16 (rms from bf16 raw)")
    if gate.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise ValueError("expected gate bf16, weight fp32")
    key = (T, H, float(eps))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _ONORM_APPLY_COMPILED.get(key)
    args = (raw, gate, weight, out)
    if compiled is None:
        compiled = cute.compile(
            _run_kda_onorm_apply,
            *(_cute_tensor(t, dynamic=False) for t in args),
            H=H,
            T=T,
            eps=float(eps),
            stream=stream,
        )
        _ONORM_APPLY_COMPILED[key] = compiled
    compiled(
        *(_cute_tensor(t, dynamic=False) for t in args),
        stream,
    )
    return out
