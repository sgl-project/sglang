from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


def _causal_depthwise_conv1d(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Deterministic causal depthwise conv1d without cuDNN.

    Args:
        x: [B, D, L] input tensor (fp32)
        weight: [D, 1, K] depthwise conv kernel
        bias: optional [D] bias

    Returns: [B, D, L] convolution output
    """
    K = weight.shape[2]
    x_padded = F.pad(x, (K - 1, 0))
    L = x.shape[2]
    out = x_padded[:, :, 0:L] * weight[:, 0, 0].view(1, -1, 1)
    for k in range(1, K):
        out = out + x_padded[:, :, k : k + L] * weight[:, 0, k].view(1, -1, 1)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out


try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:
    _HAVE_TRITON = False

try:
    # fla's analytic backward for the chunked gated delta rule. fla's forward computes the same
    # function as our torch_chunk (verified: fwd matches to bf16 ULP, and fla's full autograd
    # matches autograd through our torch-loop reference to rel ~1e-3 on every grad), so fla's bwd
    # is a valid adjoint for OUR forward when driven by OUR saved tensors (our T-matrix as fla's
    # `A`, our fp32 cumsum as fla's `g`). _ChunkGDR uses it so training gets the fused scan too.
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd as _fla_chunk_gdr_bwd

    _HAVE_FLA_BWD = True
except ImportError:
    _HAVE_FLA_BWD = False


if _HAVE_TRITON:
    from triton.language.extra import libdevice

    @triton.jit
    def _l2norm_bf16_kernel(x_ptr, o_ptr, N, D: tl.constexpr, eps):
        # One program per row: y = (x * rsqrt(sum(x*x) + eps)).to(bf16). The tl.sum reduction tree
        # differs from torch's .sum(-1) by ~1 bf16 ULP, so this is shared by BOTH Megatron's and
        # SGLang's torch_chunk_gated_delta_rule AND the decode prep kernel (which inlines the same
        # reduction), keeping all three bit-identical. Row-independent, so a single decode token
        # produces the same value as that row inside a full teacher-forced sequence.
        row = tl.program_id(0)
        if row < N:
            d = tl.arange(0, D)
            x = tl.load(x_ptr + row * D + d).to(tl.float32)
            y = x * libdevice.rsqrt(tl.sum(x * x, 0) + eps)
            tl.store(o_ptr + row * D + d, y.to(tl.bfloat16))

    @triton.jit
    def _chunk_cumsum_kernel(g_ptr, o_ptr, N, C: tl.constexpr):
        # One program per (batch*head*chunk) row: serial left-to-right fp32 scan over the
        # chunk_size dim. A serial scan (not tl.cumsum's parallel tree, not torch.cumsum's
        # blocked reduction) is what makes the decode incremental append gcum[i]=gcum[i-1]+g[i]
        # bit-identical to prefill by construction (same fp32 additions, same order). Shared by
        # BOTH Megatron's and SGLang's torch_chunk_gated_delta_rule AND the decode prep kernel
        # (which inlines the same running add), so all three engines stay bit-for-bit identical.
        # Length-invariant: exactly C serial adds per row regardless of how many chunks exist.
        row = tl.program_id(0)
        if row < N:
            acc = 0.0
            for j in range(C):
                acc += tl.load(g_ptr + row * C + j)
                tl.store(o_ptr + row * C + j, acc)

    @triton.jit
    def _fwd_sub_kernel(A, C: tl.constexpr, QR: tl.constexpr = 16):
        # One program per (batch*head*chunk). A points at a [C, C] row-major fp32 matrix
        # holding the strictly-lower A_strict (zeros on/above the diagonal). The whole
        # matrix is loaded into a register tile M once; the substitution runs entirely
        # in registers (no global read-after-write across loop iterations, which would
        # be an unordered hazard), then M is stored once.
        #
        # Forward substitution:
        #   for i in 1..C-1: M[i, :i] += sum_{m<i} M[i, m] * M[m, :i]
        # The reduction over m is done as a tl.dot (matmul) so its fp32 accumulation order is
        # FIXED regardless of the launch's num_warps (tl.sum's tree is num_warps-sensitive). This
        # lets the decode append reproduce this row inside a fused kernel launched at num_warps=8
        # while prefill runs at the default; the tl.dot path is exactly num_warps-invariant (0 ULP
        # across nw=1/4/8) and ~9e-8 off the old tl.sum path (fp32 noise). Megatron prefill and
        # SGLang both call this same kernel so they stay identical to each other; gradients still
        # flow via _SolveFwdSub (backward depends only on the output T). Deterministic, len-invariant.
        #
        # The lhs is a small [QR, C] tile (only row 0 = a_im, QR>=16 for tl.dot) instead of the full
        # [C, C]. The K-dim (m) reduction is what fixes the fp32 order, and it is independent of the
        # lhs M-tile height, so the QR tile is bit-identical to the old [C,C] lhs (verified 0 ULP
        # across seeds/scales/num_warps) while doing 4x less dot work per step. This is the single
        # dominant cost of batch-invariant prefill: on 8x2048 it drops the solve 58.7ms -> 0.9ms
        # (68x), taking torch_chunk_gated_delta_rule from 66ms -> ~7ms. QR row 0 is used (not row i)
        # so the extract mask is num_warps-invariant (single nonzero lane), same as the decode kernel.
        pid = tl.program_id(0)
        base = A + pid * C * C
        r = tl.arange(0, C)
        qr = tl.arange(0, QR)
        M = tl.load(base + r[:, None] * C + r[None, :])  # [C, C] fp32 in registers
        for i in range(1, C):
            row_i = tl.sum(tl.where(r[:, None] == i, M, 0.0), 0)  # current M[i, :]
            a_im = tl.where(r < i, row_i, 0.0)  # [C] = M[i, m] for m<i, else 0
            # acc[j] = sum_m a_im[m] * M[m, j], as a [QR,C]@[C,C] matmul with only row 0 of the lhs
            # nonzero; extract row 0. tl.dot's K-reduction order is num_warps- and QR-invariant.
            lhs = tl.where(qr[:, None] == 0, a_im[None, :], 0.0)  # [QR,C], row 0 = a_im
            accfull = tl.dot(lhs, M, input_precision="ieee")  # [QR,C], row 0 = acc
            acc = tl.sum(tl.where(qr[:, None] == 0, accfull, 0.0), 0)  # [C(j)]
            new_row = tl.where(r < i, row_i + acc, row_i)  # update only cols < i
            M = tl.where(r[:, None] == i, new_row[None, :], M)
        tl.store(base + r[:, None] * C + r[None, :], M)

    # ----- Incremental (O(1)-row/step) chunk-replay decode kernels -----
    # Each decode step appends ONE token to the partial chunk. A's / T's / v_new's leading blocks
    # are constant across steps (old tokens' prep + stable gcum prefix), so only row i=W-1 is new.
    # Per-(slot,head) programs read per-slot width via i_ptr[b]. Buffer layouts (batch-major):
    #   qn/kn/kb [B,CS,HV,Dk], vb [B,CS,HV,Dv], gbuf/gcum [B,HV,CS] (head-major), A_row [B,HV,CS],
    #   T [B,HV,CS,CS], vnew [B,HV,CS,Dv], S/Snew [B,HV,Dk,Dv], out [B,HV,Dv].

    @triton.jit
    def _gdn_step_kernel(
        q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, Alog_ptr, dtb_ptr,  # new token: q/k [B,HK,Dk], v [B,HV,Dv], a/b [B,HV]
        qn_ptr, kn_ptr, kb_ptr, vb_ptr, g_ptr, gcum_ptr, A_ptr, T_ptr,
        vnew_ptr, S_ptr, out_ptr, i_ptr, inext_ptr, row_ptr,
        scale, HV: tl.constexpr, HK: tl.constexpr, Dk: tl.constexpr, Dv: tl.constexpr,
        CS: tl.constexpr, PAD_ROW: tl.constexpr, SPREC: tl.constexpr,
    ):
        """Whole GDN decode step for the new token in ONE launch (was 5 kernels): prep + gcum append
        + A row + T-row append + core/v_new + boundary fold. One program per (slot, value-head).

        Reproduces row i of torch_chunk_gated_delta_rule bit-for-bit. Row i's kn/kb/vb/gcum/qn and the
        appended T row are all computed HERE in registers; they are stored to the global buffers (for
        the NEXT step, which reads them as frozen rows r<i) and simultaneously OVERLAID from those same
        registers wherever this step needs row i. Rows r<i are read from global (written by prior steps,
        so behind a launch barrier). Overlaying row i from registers avoids an intra-program store->load
        hazard (unordered without a fence) while being byte-identical to reloading a just-stored fp32.

        Bit-identity notes:
        (1) gcum[i]=gcum[i-1]+g[i] reloads g_r from the buffer (rounded fp32) before the add: g is the
            register product -exp(A_log)*sp, so `prev + g` would contract into one FMA (one rounding) and
            diverge from the shared serial _chunk_cumsum_kernel (two stored fp32 operands, two roundings).
            The reload inserts the rounding barrier (FMA contraction otherwise drifts ~1.9e-6).
        (2) The A dot and the T-append are fp32; the append uses the SAME tl.dot formulation as
            _fwd_sub_kernel's per-i step, so it is bit-identical to prefill AND num_warps-invariant.
        (3) This kernel launches at num_warps=8 (for the incr matmuls). Every reduction that must match
            another engine is num_warps-invariant: the l2norm tl.sum (verified 0-ULP across nw), the
            tl.dot solve/A/attn matmuls (fixed accumulation order), and the row-selecting tl.sum
            collapses (a single nonzero lane). So nw=8 is safe.
        (4) The scan (SPREC) dots -- vprime, attn2, attn_inter, intra, and the boundary fold -- place
            the new token at row i of a full CS=64 tile, the SAME geometry _chunk_scan_kernel uses, so
            tf32 tensor-core accumulation matches the batched scan bit-for-bit (a smaller QR<CS tile
            would put the token at a different row of a different-height MMA and diverge ~6e-4)."""
        pid = tl.program_id(0)
        bt = pid // HV  # batch row (indexes the per-step IO tensors q/k/v/a/b/out)
        h = pid % HV
        ar = tl.load(row_ptr + bt)  # arena row for this slot (indexes the persistent state buffers)
        i = tl.load(i_ptr + ar)
        W = i + 1
        rep: tl.constexpr = HV // HK
        hk = h // rep
        r = tl.arange(0, CS)
        dk = tl.arange(0, Dk)
        dv = tl.arange(0, Dv)
        valid = r < W
        lower = r[:, None] >= r[None, :]  # lower-tri incl diag (matches _chunk_scan_kernel)
        base = (ar * HV + h) * CS
        row_i = r[:, None] == i  # [CS,1] mask selecting row i
        # --- prep row i (shared l2norm reduction, GQA repeat, scale, gating) ---
        q = tl.load(q_ptr + (bt * HK + hk) * Dk + dk).to(tl.float32)
        k = tl.load(k_ptr + (bt * HK + hk) * Dk + dk).to(tl.float32)
        qn_i = (q * libdevice.rsqrt(tl.sum(q * q, 0) + 1e-6)).to(tl.bfloat16).to(tl.float32) * scale  # [Dk]
        kn_i = (k * libdevice.rsqrt(tl.sum(k * k, 0) + 1e-6)).to(tl.bfloat16).to(tl.float32)  # [Dk]
        av = tl.load(a_ptr + bt * HV + h).to(tl.float32)
        bx = tl.load(b_ptr + bt * HV + h).to(tl.float32)
        x = av + tl.load(dtb_ptr + h)
        sp = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
        g = -libdevice.exp(tl.load(Alog_ptr + h)) * sp
        beta = (1.0 / (1.0 + libdevice.exp(-bx))).to(tl.bfloat16).to(tl.float32)
        v = tl.load(v_ptr + (bt * HV + h) * Dv + dv).to(tl.float32)
        kb_i = kn_i * beta  # [Dk]
        vb_i = v * beta  # [Dv]
        tl.store(qn_ptr + ((ar * CS + i) * HV + h) * Dk + dk, qn_i)
        tl.store(kn_ptr + ((ar * CS + i) * HV + h) * Dk + dk, kn_i)
        tl.store(kb_ptr + ((ar * CS + i) * HV + h) * Dk + dk, kb_i)
        tl.store(vb_ptr + ((ar * CS + i) * HV + h) * Dv + dv, vb_i)
        tl.store(g_ptr + base + i, g)
        # gcum[i]=gcum[i-1]+g[i]; reload g_r (rounded fp32) to insert the rounding barrier (see note 1).
        g_r = tl.load(g_ptr + base + i)
        prev = tl.where(i > 0, tl.load(gcum_ptr + base + i - 1, mask=(i > 0), other=0.0), 0.0)
        gi = prev + g_r
        tl.store(gcum_ptr + base + i, gi)
        # Barrier so the row-i stores above are visible to the reloads below (intra-program store->load
        # to the same addresses is otherwise unordered). Reloading the full [0..i] tiles from global --
        # rather than overlaying row i from registers -- makes every downstream tl.dot consume the exact
        # same fp32 tile the separate a_row/incr kernels did, keeping the fused step bit-identical to the
        # prior 2-launch path (register overlay drifted ~3e-3 through the attn/v_new matmuls).
        tl.debug_barrier()
        kn = tl.load(kn_ptr + ((ar * CS + r[:, None]) * HV + h) * Dk + dk[None, :], mask=valid[:, None], other=0.0)
        kb = tl.load(kb_ptr + ((ar * CS + r[:, None]) * HV + h) * Dk + dk[None, :], mask=valid[:, None], other=0.0)
        vb = tl.load(vb_ptr + ((ar * CS + r[:, None]) * HV + h) * Dv + dv[None, :], mask=valid[:, None], other=0.0)
        gcum = tl.load(gcum_ptr + base + r, mask=valid, other=0.0)
        # Everything below places the new token at ROW i of a full CS=64 tile (row_i mask), matching
        # _chunk_scan_kernel's tile geometry so the tf32 (SPREC) dots accumulate bit-identically to the
        # scan (tf32 tensor-core reductions are only tile-shape reproducible; the old QR=16 tiles put
        # the token at row 0 of a 16-high tile and diverged ~6e-4). The ieee WY-prep dots (A, T-append,
        # val, kcd) are M-tile invariant so they stay bit-identical to prefill's aten bmm regardless.
        # --- A row i = -(kb[i] @ kn^T) * exp(gcum[i]-gcum[j]) for j<i (strictly-lower) ---
        # Use the RELOADED kb[i] (kb_ri), NOT the kb_i register (= kn_i*beta): feeding the register
        # into the dot lets the compiler carry kn_i's pre-bf16-cast precision into the matmul, so the
        # A entries land ~4e-5 off prefill's aten bmm (which consumes the stored, bf16-rounded kb).
        # Build the decay as the full 2D exp(gcum[a]-gcum[b]) from the reloaded gcum tile, the exact
        # expression prefill's decay_mask (and the core `decay` below) uses. Both mismatches are
        # invisible under ieee (the solve/scan absorb them) but tf32 truncates each dot input, so they
        # amplify through T -> v_solved -> intra (~1e-3 decode-vs-prefill) unless matched here.
        decay_full = libdevice.exp(gcum[:, None] - gcum[None, :])  # [CS,CS]
        kb_ri = tl.sum(tl.where(row_i, kb, 0.0), 0)  # reloaded kb[i], single-lane collapse
        kb_tile = tl.where(row_i, kb_ri[None, :], 0.0)  # [CS,Dk], row i = kb[i]
        kept2 = (r[:, None] > r[None, :]) & valid[None, :]  # strictly-lower, cols valid
        decay_a = tl.where(kept2, decay_full, 0.0)  # [CS,CS]
        A = -tl.dot(kb_tile, tl.trans(kn), input_precision="ieee") * decay_a
        kept = (i > r) & valid
        A = tl.where(kept[None, :], A, 0.0)
        Arow = tl.sum(tl.where(row_i, A, 0.0), 0)  # [CS] tile row i
        tl.store(A_ptr + base + r, Arow)
        # --- append forward-substitution row: T[i,j<i] = A[i,j] + sum_{m<i} A[i,m]*T[m,j] ---
        # Same tl.dot formulation as _fwd_sub_kernel (bit-identical to prefill, num_warps-invariant).
        M = tl.load(T_ptr + (base + r[:, None]) * CS + r[None, :])  # cached final rows m<i
        a_im = tl.where(r < i, Arow, 0.0)  # [C], zero for m>=i
        lhs = tl.where(row_i, a_im[None, :], 0.0)  # [C,C], row i = a_im
        accfull = tl.dot(lhs, M, input_precision="ieee")  # [C,C], row i = sum_m a_im[m]*M[m,j]
        acc = tl.sum(tl.where(row_i, accfull, 0.0), 0)  # [C(j)]
        Tnew = tl.where(r < i, Arow + acc, Arow)  # T[i,:] (strictly-lower part filled)
        tl.store(T_ptr + (base + i) * CS + r, Tnew)
        # --- core row i = attn_inter[i] + attn2[i,:] @ v_new, caching v_new[i] and optional Snew ---
        # Reload the just-stored T row i and qn row i from global (after the barrier above) so the core
        # matmuls consume the same fp32 tiles the separate incr kernel did (bit-identical to prior path).
        S = tl.load(S_ptr + (ar * HV + h) * Dk * Dv + dk[:, None] * Dv + dv[None, :])
        Trow = tl.load(T_ptr + (base + i) * CS + r)
        Trow = Trow + tl.where(r == i, 1.0, 0.0)  # + eye row i
        Ttile = tl.where(row_i, Trow[None, :], 0.0)  # [CS,CS], row i = Trow
        val = tl.dot(Ttile, vb, input_precision="ieee")  # [CS,Dv], row i = T[i,:]@vb
        kg = kb * libdevice.exp(gcum)[:, None]  # [CS,Dk]
        kcd = tl.dot(Ttile, kg, input_precision="ieee")  # [CS,Dk], row i = T[i,:]@(kb*exp g)
        # Mirror _chunk_scan_kernel's core exactly, incl the `bar` rounding barrier (data-dependent so
        # the compiler cannot fold the following add/sub into an accumulator-form FMA). The scan and
        # this step must fold in the identical order for tf32 decode==prefill.
        bar = tl.sum(tl.where(r == 0, gcum, 0.0), 0) < 1e30  # always true, not provably so
        vprime = tl.where(bar, tl.dot(kcd, S, input_precision=SPREC), 0.0)  # [CS,Dv] k_cumdecay@S
        vnew_tile = val - vprime  # [CS,Dv]
        vnew_row = tl.sum(tl.where(row_i, vnew_tile, 0.0), 0)  # [Dv] v_new[i]
        vn_cached = tl.load(vnew_ptr + (base + r[:, None]) * Dv + dv[None, :], mask=(r < i)[:, None], other=0.0)
        vnew_full = tl.where(row_i, vnew_row[None, :], vn_cached)  # [CS,Dv] rows 0..i
        qn = tl.load(qn_ptr + ((ar * CS + i) * HV + h) * Dk + dk)  # [Dk] row-i query
        qn_tile = tl.where(row_i, qn[None, :], 0.0)  # [CS,Dk], row i = qn_i
        decay = tl.where(lower, libdevice.exp(gcum[:, None] - gcum[None, :]), 0.0)  # [CS,CS]
        attn2 = tl.dot(qn_tile, tl.trans(kn), input_precision=SPREC) * decay
        attn2 = tl.where(lower, attn2, 0.0)
        qg = qn_tile * libdevice.exp(gcum)[:, None]  # [CS,Dk], row i = qn_i*exp(gcum[i])
        attn_inter = tl.where(bar, tl.dot(qg, S, input_precision=SPREC), 0.0)  # (q*exp g)@S
        intra = tl.where(bar, tl.dot(attn2, vnew_full, input_precision=SPREC), 0.0)
        core = attn_inter + intra  # [CS,Dv]
        core_row = tl.sum(tl.where(row_i, core, 0.0), 0)  # [Dv]
        # Store the core row (PAD rows -> zeros) directly into out; tl.store casts fp32 core_row to
        # the out buffer's dtype. Prod out is bf16 (was fp32 + a python .to(bf16)/valid-multiply);
        # the bf16 store is bit-identical to the old python cast (same fp32 core_row). A test may
        # pass an fp32 buffer to keep the pre-cast fp32 regression guard. out is batch-indexed (bt).
        core_out = tl.where(ar == PAD_ROW, 0.0, core_row)
        tl.store(out_ptr + (bt * HV + h) * Dv + dv, core_out)
        tl.store(vnew_ptr + (base + i) * Dv + dv, vnew_row)
        # In-kernel width advance + boundary fold (was python-side, blocking cuda-graph capture).
        # On the 64th token fold Snew straight into the boundary S (S already read into registers
        # above, so overwriting S_ptr here is safe) and restart the partial chunk at width 0; else
        # advance to width W. inext_ptr is a separate buffer from i_ptr so this store never races the
        # in-flight reads of i_ptr this launch (python rolls i_buf<-inext_buf between launches). All
        # HV programs for this arena row write the SAME next width, so the concurrent stores tie.
        if W == CS:
            glast = tl.sum(tl.where(r == (CS - 1), gcum, 0.0), 0)
            kdec = kn * libdevice.exp(glast - gcum)[:, None]
            # Fixed token-order rank-1 accumulation (NOT tl.dot): must bit-match _chunk_scan_kernel's
            # boundary fold. A tl.dot tiles the [Dk,Dv] output differently between the two separately
            # compiled kernels and drifts ~1 fp32 ULP under any precision; the unrolled outer-product
            # sum has one accumulation order in both. See the scan kernel's fold comment.
            upd = tl.zeros([Dk, Dv], tl.float32)
            for c in range(CS):
                kc = tl.sum(tl.where(r[:, None] == c, kdec, 0.0), 0)  # [Dk] row c
                vc = tl.sum(tl.where(r[:, None] == c, vnew_full, 0.0), 0)  # [Dv] row c
                upd += kc[:, None] * vc[None, :]
            sc = tl.where(bar, S * libdevice.exp(glast), 0.0)  # round before add (matches scan)
            Snew = sc + tl.where(bar, upd, 0.0)
            tl.store(S_ptr + (ar * HV + h) * Dk * Dv + dk[:, None] * Dv + dv[None, :], Snew)
            tl.store(inext_ptr + ar, 0)
        else:
            tl.store(inext_ptr + ar, W)

    @triton.jit
    def _chunk_scan_kernel(
        q_ptr, k_ptr, gcum_ptr, vsolved_ptr, kcd_ptr,  # per-chunk streamed inputs (fp32)
        Sinit_ptr, core_ptr, last_ptr, bnd_ptr,        # recurrent state in / out
        nchunks_ptr, bchunk,                           # per-seq real chunk count [N], boundary chunk
        HV: tl.constexpr, NC: tl.constexpr, C: tl.constexpr, Dk: tl.constexpr, Dv: tl.constexpr,
        PREC: tl.constexpr,
    ):
        """Fused cross-chunk GDN recurrence: the SERIAL scan over `last_recurrent_state` (the torch
        loop at torch_chunk_gated_delta_rule lines 563-573 / Megatron 1348-1358) in ONE launch per
        (sequence, value-head). The full [Dk,Dv] state stays resident in registers across the whole
        chunk loop, so it never round-trips through HBM (the torch loop pushed the state 32x). The
        intra-chunk WY work (l2norm, cumsum, T-solve, v_solved=T@v_beta, k_cumdecay=T@(k_beta*exp g))
        stays on the existing parallel torch path and is streamed in per chunk.

        Under tf32 (see GDN_SCAN_PREC) the dots run on tensor cores, which do NOT spill the full
        [Dk,Dv] fp32 state, so there is no V-tiling: grid is (N,HV) and each program holds the whole
        [Dk,Dv] S. (fp32-IEEE FMA units DO spill the full slab ~16x, but tf32 is the shipping path.)

        Cross-engine identity is by construction: sglang prefill, Megatron's forward, and the decode
        chunk-replay all run THIS kernel at this same PREC, so the dots need not match aten's fp32 bmm.
        The one hard requirement is decode==prefill: the decode _gdn_step_kernel maps op-for-op to this
        fold and, crucially, uses the SAME C=64 tile geometry (tf32 tensor-core accumulation is only
        tile-shape reproducible; the old QR=16 decode tiles diverged ~6e-4). exp is libdevice.exp on
        rounded-fp32 gcum. Multi-seq: each program loops only its seq's real chunks; zero-pad chunks
        are exact recurrent no-ops (g=0 -> decay 1, k=v=0 -> zero update).
        """
        pid = tl.program_id(0)
        h = pid % HV
        n = pid // HV
        nchunks = tl.load(nchunks_ptr + n)
        r = tl.arange(0, C)
        dk = tl.arange(0, Dk)
        dv = tl.arange(0, Dv)
        lower = r[:, None] >= r[None, :]  # keep lower incl diag (matches tril + masked_fill(triu(1)))
        s_off = (n * HV + h) * Dk * Dv
        S = tl.load(Sinit_ptr + s_off + dk[:, None] * Dv + dv[None, :])  # [Dk,Dv] resident in regs
        for i in range(0, nchunks):
            cbase = ((n * HV + h) * NC + i) * C
            q = tl.load(q_ptr + cbase * Dk + r[:, None] * Dk + dk[None, :])  # [C,Dk] scaled+l2normed
            k = tl.load(k_ptr + cbase * Dk + r[:, None] * Dk + dk[None, :])  # [C,Dk] l2normed
            vs = tl.load(vsolved_ptr + cbase * Dv + r[:, None] * Dv + dv[None, :])  # [C,Dv] T@v_beta
            kcd = tl.load(kcd_ptr + cbase * Dk + r[:, None] * Dk + dk[None, :])  # [C,Dk] T@(kb*exp g)
            gc = tl.load(gcum_ptr + cbase + r)  # [C] rounded-fp32 cumulative decay
            # Each matmul result feeding an add/sub is rounded to a register FIRST (via the `bar`
            # tl.where) so the scan and the decode _gdn_step_kernel fold in exactly the same order --
            # decode must reproduce prefill, and a bare `x +/- tl.dot` lets the compiler fold into an
            # accumulator-form dot / FMA (one fused rounding) and drift ~2e-4 through the scan. `bar`
            # is a data-dependent predicate the compiler cannot prove (unlike a tautological `r>=0`
            # which const-folds away), so the round survives identically in both kernels.
            bar = tl.sum(tl.where(r == 0, gc, 0.0), 0) < 1e30  # always true, not provably so
            # decay[a,b] = exp(gcum[a]-gcum[b]) for a>=b else 0 (== torch decay_mask[:,:,i]).
            decay = tl.where(lower, libdevice.exp(gc[:, None] - gc[None, :]), 0.0)
            # attn2 = (q@k^T)*decay, strict-upper zeroed, diagonal kept.
            attn2 = tl.dot(q, tl.trans(k), input_precision=PREC) * decay
            attn2 = tl.where(lower, attn2, 0.0)
            qg = q * libdevice.exp(gc)[:, None]  # [C,Dk]
            attn_inter = tl.where(bar, tl.dot(qg, S, input_precision=PREC), 0.0)  # (q*exp g)@S
            vprime = tl.where(bar, tl.dot(kcd, S, input_precision=PREC), 0.0)  # k_cumdecay@S
            vnew = vs - vprime  # [C,Dv] == v_i - v_prime
            intra = tl.where(bar, tl.dot(attn2, vnew, input_precision=PREC), 0.0)
            core = attn_inter + intra  # [C,Dv]
            tl.store(core_ptr + cbase * Dv + r[:, None] * Dv + dv[None, :], core)
            # state update: S = S*exp(glast) + (k * exp(glast-gcum))^T @ v_new
            glast = tl.sum(tl.where(r == C - 1, gc, 0.0), 0)  # gcum[C-1], single-lane collapse
            kdec = k * libdevice.exp(glast - gc)[:, None]  # [C,Dk]
            # The boundary fold contracts all C rows into the recurrent state. A tl.dot here drifts
            # ~1 fp32 ULP between the two separately-compiled scan/step kernels (the [Dk,Dv] output
            # is MMA-tiled differently under their different register pressure -- true under ieee AND
            # tf32) which slowly desyncs the decode-replay state from prefill across multi-chunk
            # decode and compounds over layers. Fold instead as a fixed token-order rank-1
            # accumulation: a c=0..C-1 unrolled sum of outer products has one accumulation order
            # baked identically into both kernels (the row-select tl.sum is single-lane, so
            # num_warps- and tile-invariant), so decode==prefill EXACTLY. One fold per chunk, off the
            # per-token hot path.
            upd = tl.zeros([Dk, Dv], tl.float32)
            for c in range(C):
                kc = tl.sum(tl.where(r[:, None] == c, kdec, 0.0), 0)  # [Dk] row c
                vc = tl.sum(tl.where(r[:, None] == c, vnew, 0.0), 0)  # [Dv] row c
                upd += kc[:, None] * vc[None, :]
            sc = tl.where(bar, S * libdevice.exp(glast), 0.0)  # round before add (else FMA drift)
            S = sc + tl.where(bar, upd, 0.0)
            if i == bchunk:  # snapshot state entering the trailing partial chunk (n_seq==1 boundary)
                tl.store(bnd_ptr + s_off + dk[:, None] * Dv + dv[None, :], S)
        tl.store(last_ptr + s_off + dk[:, None] * Dv + dv[None, :], S)


# Matmul precision for the GDN cross-chunk scan / decode-replay dots. "tf32" runs the dots on TF32
# tensor cores (~5x faster than "ieee", which has no tensor-core path on B200 and additionally spills
# the full-Dv state), is deterministic, and is num_warps- / V-tile-invariant (the K-reductions are
# single-dot, K=Dk=128 or C=64; V is the N dim not K). Cross-engine identity is by construction: the
# sglang prefill scan, Megatron's forward, AND the decode chunk-replay all run these SAME kernels at
# this SAME precision, so tf32 need not match aten's fp32 bmm. The one hard requirement is that the
# decode _gdn_step_kernel reproduce the scan bit-for-bit: tf32 tensor-core accumulation is only
# tile-shape reproducible, so decode's fold dots use the SAME C=64 tile geometry as the scan (NOT the
# old QR=16 tiles, which diverged ~6e-4 under tf32). Verified 0.0 batched-vs-C64-incremental (all
# precisions) and decode==prefill replay tests. ieee is still bit-identical but ~5x slower here.
GDN_SCAN_PREC = "tf32"


def _fused_chunk_scan(query, key, gcum, v_solved, k_cumdecay, s_init, nchunks, bchunk, want_boundary):
    """Launch _chunk_scan_kernel: the cross-chunk recurrence over the precomputed intra-chunk WY
    tensors. All inputs are fp32 contiguous [N,HV,NC,C,*] (gcum [N,HV,NC,C]); s_init [N,HV,Dk,Dv].
    Returns (core_attn_out [N,HV,NC,C,Dv], last_state [N,HV,Dk,Dv], boundary_state or None)."""
    N, HV, NC, C, Dk = query.shape
    Dv = v_solved.shape[-1]
    core = torch.zeros(N, HV, NC, C, Dv, device=query.device, dtype=torch.float32)
    last = torch.empty(N, HV, Dk, Dv, device=query.device, dtype=torch.float32)
    bnd = torch.empty(1, HV, Dk, Dv, device=query.device, dtype=torch.float32) if want_boundary else last
    # tf32 tensor cores hold the full [Dk,Dv] state resident (no V-tiling): grid is (N,HV).
    _chunk_scan_kernel[(N * HV,)](
        query, key, gcum, v_solved, k_cumdecay,
        s_init, core, last, bnd,
        nchunks, bchunk if want_boundary else -1,
        HV=HV, NC=NC, C=C, Dk=Dk, Dv=Dv, PREC=GDN_SCAN_PREC,
        num_warps=8, num_stages=1,
    )
    return core, last, (bnd if want_boundary else None)


def _solve_fwd_sub(attn: torch.Tensor) -> torch.Tensor:
    """Forward-substitution triangular solve, replacing the 64-iteration Python loop.

    `attn` is `[..., C, C]` holding strictly-lower A_strict (zeros on/above diagonal).
    Returns the substituted matrix (strictly-lower part updated) BEFORE the `+ eye` at
    the call site, i.e. `(I - A_strict)^-1 - I`. Deterministic and length-invariant (one
    independent program per (b, h, chunk), so Megatron prefill and SGLang match each
    other), and differentiable via the analytic VJP grad_A = tril(T^T @ grad @ T^T, -1)
    with T = (I - A_strict)^-1. ~1 fp32 ULP off the old Python loop (different reduction
    order), which does not matter since both engines share this kernel.
    """
    return _SolveFwdSub.apply(attn)


class _SolveFwdSub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn):
        C = attn.shape[-1]
        A = attn.detach().reshape(-1, C, C).contiguous().float()
        _fwd_sub_kernel[(A.shape[0],)](A, C=C)
        sub = A.reshape(attn.shape).to(attn.dtype)
        if ctx.needs_input_grad[0]:
            eye = torch.eye(C, dtype=sub.dtype, device=sub.device)
            ctx.save_for_backward(sub + eye)  # T = (I - A_strict)^-1
        return sub

    @staticmethod
    def backward(ctx, grad_sub):
        (T,) = ctx.saved_tensors
        C = T.shape[-1]
        Tt = T.transpose(-1, -2)
        grad_A = Tt @ grad_sub @ Tt
        mask = torch.tril(torch.ones(C, C, dtype=torch.bool, device=T.device), diagonal=-1)
        return grad_A * mask


def l2norm_bf16(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize over the last dim and round to bf16, via the shared _l2norm_bf16_kernel.

    Replaces torch's `(x * rsqrt((x*x).sum(-1,keepdim)+eps)).to(bf16)` in both Megatron's and
    SGLang's torch_chunk_gated_delta_rule so the reduction order matches the decode prep kernel
    bit-for-bit. Differentiable via the analytic VJP grad_x = n*grad_y - n^3*x*(grad_y . x) with
    n = rsqrt(sum(x^2)+eps) (the bf16 cast is treated as identity in backward, as torch's .to does).
    Falls back to torch when Triton or CUDA is unavailable.
    """
    if not _HAVE_TRITON or not x.is_cuda:
        xf = x.float()
        return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + eps)).to(torch.bfloat16)
    return _L2NormBf16.apply(x, eps)


class _L2NormBf16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        D = x.shape[-1]
        xf = x.detach().reshape(-1, D).contiguous().float()
        o = torch.empty_like(xf, dtype=torch.bfloat16)
        N = xf.shape[0]
        _l2norm_bf16_kernel[(N,)](xf, o, N, D=D, eps=eps)
        if x.requires_grad:
            ctx.save_for_backward(xf)
            ctx.eps = eps
        return o.reshape(x.shape).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_y):
        (xf,) = ctx.saved_tensors  # normalized-over dtype (fp32 in prod, fp64 under gradcheck)
        D = xf.shape[-1]
        x2 = xf.reshape(-1, D)
        gy = grad_y.reshape(-1, D).to(x2.dtype)
        s = (x2 * x2).sum(-1, keepdim=True) + ctx.eps
        n = torch.rsqrt(s)
        grad_x = n * gy - (n * n * n) * x2 * (gy * x2).sum(-1, keepdim=True)
        return grad_x.reshape(grad_y.shape).to(grad_y.dtype), None


def chunk_cumsum(g: torch.Tensor) -> torch.Tensor:
    """Serial fp32 cumsum over the last dim, via the shared _chunk_cumsum_kernel.

    Replaces the per-chunk `torch.stack([g[:,:,i].cumsum(-1) ...])` in both Megatron's and
    SGLang's torch_chunk_gated_delta_rule. A serial left-to-right scan makes the decode
    incremental append gcum[i]=gcum[i-1]+g[i] bit-identical to prefill by construction, and
    is length-invariant (C adds/row regardless of chunk count), so all engines match bit-for-bit.
    Differentiable via the cumsum VJP grad_x[i]=sum_{j>=i} grad_y[j] (reverse cumsum). Falls back
    to per-chunk torch.cumsum when Triton or CUDA is unavailable.
    """
    if not _HAVE_TRITON or not g.is_cuda:
        return torch.stack([g[:, :, i].cumsum(dim=-1) for i in range(g.shape[2])], dim=2)
    return _ChunkCumsum.apply(g)


class _ChunkCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g):
        C = g.shape[-1]
        gf = g.detach().reshape(-1, C).contiguous().float()
        o = torch.empty_like(gf)
        N = gf.shape[0]
        _chunk_cumsum_kernel[(N,)](gf, o, N, C=C)
        return o.reshape(g.shape).to(g.dtype)

    @staticmethod
    def backward(ctx, grad_y):
        # cumsum VJP: grad_x[i] = sum_{j>=i} grad_y[j] = flip(cumsum(flip(grad_y))).
        gy = grad_y.flip(-1).cumsum(-1).flip(-1)
        return gy


class _ChunkGDR(torch.autograd.Function):
    """One autograd Function wrapping the WHOLE chunked gated delta rule (intra-chunk WY prep +
    cross-chunk recurrence) on a padded-batch [N, HV, T_pad, *] fp32 layout. Forward runs the same
    deterministic kernels the inference path uses (chunk_cumsum, _solve_fwd_sub, _fused_chunk_scan)
    so training and inference forwards agree numerically. Backward drives fla's analytic
    chunk_gated_delta_rule_bwd from OUR saved tensors: our T-matrix as fla's `A`, our fp32 cumsum as
    fla's `g`, unscaled l2normed q/k with `scale` passed separately. fla's bwd internally recomputes
    w,u (recompute_w_u_fwd) and runs prepare_wy_repr_bwd, so it is the adjoint of the FULL prep+scan,
    not just the scan; hence this Function must span both. Validated: grads match autograd through a
    pure-torch replica of OUR forward to rel ~1e-3 (bf16 level) on every input. l2norm and the GQA
    repeat_interleave stay OUTSIDE (their own autograd) so their VJPs are unchanged.

    inputs:  query,key [N,HV,T_pad,Dk] (l2normed, GQA-repeated, UNSCALED, fp32); value [N,HV,T_pad,Dv];
             g [N,HV,T_pad] (pre-cumsum log-decay); beta [N,HV,T_pad]; initial_state [N,HV,Dk,Dv].
    returns: core [N,HV,T_pad,Dv], last_state [N,HV,Dk,Dv].
    """

    @staticmethod
    def forward(ctx, query, key, value, g, beta, initial_state, chunk_size):
        N, HV, T_pad, Dk = query.shape
        Dv = value.shape[-1]
        C = chunk_size
        NC = T_pad // C
        scale = 1.0 / (Dk ** 0.5)
        # Intra-chunk WY prep. Mirrors torch_chunk_gated_delta_rule exactly (same shared kernels)
        # so the training forward equals the inference forward numerically. The T-matrix (attn) is
        # scale-independent (built from k_beta,key only); only the scan consumes the scaled query.
        q_s = (query * scale).reshape(N, HV, NC, C, Dk)
        k_c = key.reshape(N, HV, NC, C, Dk)
        v_c = value.reshape(N, HV, NC, C, Dv)
        beta_c = beta.reshape(N, HV, NC, C)
        v_beta = v_c * beta_c.unsqueeze(-1)
        k_beta = k_c * beta_c.unsqueeze(-1)
        gcum = chunk_cumsum(g.reshape(N, HV, NC, C))
        decay = ((gcum.unsqueeze(-1) - gcum.unsqueeze(-2)).tril().exp().float()).tril()
        m0 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=query.device), 0)
        attn = -((k_beta @ k_c.transpose(-1, -2)) * decay).masked_fill(m0, 0)
        attn = _solve_fwd_sub(attn) + torch.eye(C, dtype=attn.dtype, device=attn.device)
        v_solved = attn @ v_beta
        k_cumdecay = attn @ (k_beta * gcum.exp().unsqueeze(-1))
        nchunks = torch.full((N,), NC, dtype=torch.int32, device=query.device)
        core, last, _ = _fused_chunk_scan(
            q_s.contiguous(), k_c.contiguous(), gcum.contiguous(),
            v_solved.contiguous(), k_cumdecay.contiguous(),
            initial_state.contiguous(), nchunks, -1, False,
        )
        core = core.reshape(N, HV, T_pad, Dv)
        ctx.save_for_backward(query, key, value, gcum, beta, attn, initial_state)
        ctx.scale = scale
        ctx.C = C
        return core, last

    @staticmethod
    def backward(ctx, d_core, d_last):
        query, key, value, gcum, attn_T, beta, initial_state = (
            ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.saved_tensors[2],
            ctx.saved_tensors[3], ctx.saved_tensors[5], ctx.saved_tensors[4],
            ctx.saved_tensors[6],
        )
        N, HV, T_pad, Dk = query.shape
        Dv = value.shape[-1]
        C = ctx.C
        # To fla layout [B, T, H, *]. fla's A == our T-matrix [.,.,C]; fla's g == our fp32 cumsum.
        q_f = query.transpose(1, 2).contiguous()
        k_f = key.transpose(1, 2).contiguous()
        v_f = value.transpose(1, 2).contiguous()
        beta_f = beta.transpose(1, 2).contiguous()
        g_f = gcum.reshape(N, HV, T_pad).transpose(1, 2).contiguous()
        A_f = attn_T.reshape(N, HV, T_pad, C).transpose(1, 2).contiguous()
        do = (
            d_core.reshape(N, HV, T_pad, Dv).transpose(1, 2).contiguous()
            if d_core is not None
            else torch.zeros(N, T_pad, HV, Dv, dtype=query.dtype, device=query.device)
        )
        # d_last is None when the recurrent state is only cached (no grad flows back through it),
        # which is the common training case; fla's bwd accepts dht=None (state-only h0 path).
        dht = d_last.contiguous() if d_last is not None else None
        dq, dk, dv, db, dg, dh0 = _fla_chunk_gdr_bwd(
            q=q_f, k=k_f, v=v_f, g=g_f, beta=beta_f, A=A_f, scale=ctx.scale,
            initial_state=initial_state, do=do, dht=dht,
        )
        return (
            dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2),
            dg.transpose(1, 2), db.transpose(1, 2), dh0, None,
        )


def torch_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute exp in fp32 to match Megatron's fused (torch.compile) g_and_beta path,
    # which upcasts A_log.exp() to fp32. Rounding A_log.exp() to bf16 here diverged
    # gdn_g by ~6e-3 vs Megatron; fp32 is also the higher-precision choice.
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    beta = b.to(torch.bfloat16).sigmoid()
    return g.unsqueeze(0), beta.unsqueeze(0)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    chunk_size: int = 64,
    return_boundary_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """Pure PyTorch chunked gated delta rule matching Megatron's deterministic implementation.

    When return_boundary_state=True the 3rd tuple element is the recurrent state as it enters
    the final (possibly partial) chunk, i.e. the state after folding every COMPLETE 64-chunk.
    For an aligned sequence (sequence_length % chunk_size == 0) this equals last_recurrent_state.
    Decode chunk-replay seeds from this boundary state and replays the trailing partial-chunk
    tokens, reproducing Megatron's full-sequence per-token output bit-for-bit.
    """
    # Normalize memory layout: SGLang passes non-contiguous views (sliced from the fused
    # qkv projection) while Megatron passes contiguous tensors. Identical values in different
    # strides make the bf16 L2-norm reduction and downstream fp32 bmm tiling pick different
    # kernels, tipping ~1 bf16 ULP at the final cast (layer 0 core_attn_out 3.9e-3 diff).
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    initial_dtype = query.dtype
    # Shared triton l2norm (bit-identical reduction to the decode prep kernel and Megatron's
    # torch_chunk). Casts to bf16 then back to the input dtype, matching the old torch path.
    query = l2norm_bf16(query).to(initial_dtype)
    key = l2norm_bf16(key).to(initial_dtype)
    num_v_heads = value.shape[2]
    num_k_heads = key.shape[2]
    if num_v_heads // num_k_heads > 1:
        repeat = num_v_heads // num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    # Unpack the batch=1 packed row into a padded [N, num_heads, L, D] batch delimited by
    # query_start_loc. SGLang prefills N prompts as one concatenated row (query.view(1, T, ...)
    # in forward_extend), but each sequence must run the delta-rule recurrence independently
    # from its own ssm_states slot. Right-padding every sequence with zeros to a common multiple
    # of chunk_size is EXACT, not an approximation: a zero chunk is a recurrent-state no-op
    # (g=0 => decay exp(0)=1 so the state passes through unchanged; k=v=beta=0 => zero update)
    # and yields zero output rows that are sliced off. This is the same mechanism the trailing
    # partial-chunk pad already relies on, extended to whole trailing chunks of shorter sequences.
    # N==1 degenerates to the previous single-sequence padding, so it stays bit-identical.
    num_heads = key.shape[1]
    k_head_dim = key.shape[-1]
    v_head_dim = value.shape[-1]
    seq_starts = query_start_loc.tolist()
    n_seq = len(seq_starts) - 1
    seq_lens = [seq_starts[i + 1] - seq_starts[i] for i in range(n_seq)]
    max_chunks = max((sl + chunk_size - 1) // chunk_size for sl in seq_lens)
    sequence_length = max_chunks * chunk_size

    def _pack_to_batch(x: torch.Tensor) -> torch.Tensor:
        # [1, num_heads, T, ...] packed row -> [n_seq, num_heads, sequence_length, ...] padded batch
        out = x.new_zeros(n_seq, num_heads, sequence_length, *x.shape[3:])
        for i in range(n_seq):
            s, e = seq_starts[i], seq_starts[i + 1]
            out[i, :, : e - s] = x[0, :, s:e]
        return out

    query = _pack_to_batch(query)
    key = _pack_to_batch(key)
    value = _pack_to_batch(value)
    beta = _pack_to_batch(beta)
    g = _pack_to_batch(g)
    batch_size = n_seq
    total_sequence_length = sequence_length  # already a multiple of chunk_size
    scale = 1 / (query.shape[-1] ** 0.5)

    initial_state = ssm_states[cache_indices].to(query) if ssm_states is not None else None
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim,
            device=query.device, dtype=query.dtype,
        )
        if initial_state is None
        else initial_state
    )

    # State entering the final (possibly partial) chunk: the recurrent state after folding all
    # COMPLETE 64-chunks. Captured right after the (num_complete_chunks-1)-th fold so it is
    # correct both for aligned prompts (equals last_recurrent_state) and misaligned prompts
    # (the seed decode replays the trailing partial-chunk tokens from). boundary_state is only
    # consumed by _seed_gdn_replay_cache, which always passes a single sequence.
    assert not (return_boundary_state and n_seq > 1), "boundary_state only supported for n_seq==1"
    num_complete_chunks = seq_lens[0] // chunk_size
    boundary_state = last_recurrent_state  # correct when num_complete_chunks == 0
    num_chunks = total_sequence_length // chunk_size

    # Training (grad on): the whole WY prep + cross-chunk scan runs inside _ChunkGDR, one autograd
    # Function whose backward is fla's analytic chunk_gated_delta_rule_bwd driven from OUR saved
    # tensors (our T-matrix, our fp32 cumsum). Its forward uses the SAME deterministic kernels as
    # inference, so training and inference forwards agree, and it gets the fused-scan speedup too.
    # boundary_state is decode-seed-only (inference), so it is never requested under grad.
    if _HAVE_FLA_BWD and query.is_cuda and torch.is_grad_enabled():
        assert not return_boundary_state, "boundary_state (decode seed) is inference-only"
        core_attn_out, last_recurrent_state = _ChunkGDR.apply(
            query, key, value, g, beta, last_recurrent_state, chunk_size
        )
        core_attn_out = core_attn_out.reshape(n_seq, num_heads, -1, v_head_dim)
    else:
        query = query * scale
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
        )

        # Shared serial fp32 cumsum over the chunk dim (see chunk_cumsum). A serial left-to-right
        # scan is length-invariant (C adds/row regardless of chunk count) so sglang (short prefill)
        # and Megatron (full teacher-forced sequence) match bit-exactly, and it makes the decode
        # incremental append gcum[i]=gcum[i-1]+g[i] bit-identical to this prefix by construction.
        g = chunk_cumsum(g)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        # Forward-substitution triangular solve. One Triton launch (one program per
        # (b, h, chunk)) replaces the 64 sequential iterations of the Python loop:
        #   for i in range(1, chunk_size):
        #       row = attn[..., i, :i]; sub = attn[..., :i, :i]
        #       attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        # Deterministic, length-invariant, differentiable (~1 fp32 ULP off the loop).
        attn = _solve_fwd_sub(attn)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        # Cross-chunk serial recurrence over last_recurrent_state. The fused Triton scan keeps S
        # on-chip across the whole chunk loop (one launch per (seq, value-head)) and is bit-identical
        # to the torch loop (verified 0-ULP), so sglang prefill, Megatron, and the decode chunk-replay
        # all match. This raw-kernel scan is non-differentiable, so grad-on only reaches it when fla
        # is unavailable (else _ChunkGDR handles it above); there it must use the torch loop.
        use_fused_scan = _HAVE_TRITON and query.is_cuda and not torch.is_grad_enabled()
        if use_fused_scan:
            nchunks = torch.tensor(
                [(sl + chunk_size - 1) // chunk_size for sl in seq_lens],
                dtype=torch.int32, device=query.device,
            )
            core_attn_out, last_recurrent_state, bnd = _fused_chunk_scan(
                query.contiguous(), key.contiguous(), g.contiguous(),
                value.contiguous(), k_cumdecay.contiguous(),
                last_recurrent_state.contiguous(),
                nchunks, num_complete_chunks - 1, return_boundary_state,
            )
            if return_boundary_state and num_complete_chunks > 0:
                boundary_state = bnd
        else:
            core_attn_out = torch.zeros_like(value)
            mask = torch.triu(
                torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
            )
            for i in range(0, num_chunks):
                q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
                attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
                v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
                v_new = v_i - v_prime
                attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
                core_attn_out[:, :, i] = attn_inter + attn @ v_new
                last_recurrent_state = (
                    last_recurrent_state * g[:, :, i, -1, None, None].exp()
                    + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
                )
                if return_boundary_state and (i + 1) == num_complete_chunks:
                    boundary_state = last_recurrent_state

    # Repack the padded [n_seq, num_heads, padded_len, v_head_dim] batch back into the
    # [1, total_tokens, num_heads, v_head_dim] packed row the caller expects. Padded tail rows
    # (beyond each sequence's real length) are dropped. For n_seq==1 this equals the previous
    # single-sequence slice, keeping the output bit-identical.
    core_attn_out = core_attn_out.reshape(n_seq, num_heads, -1, v_head_dim)
    total_tokens = seq_starts[-1]
    packed_out = core_attn_out.new_zeros(1, num_heads, total_tokens, v_head_dim)
    for i in range(n_seq):
        s, e = seq_starts[i], seq_starts[i + 1]
        packed_out[0, :, s:e] = core_attn_out[i, :, : e - s]
    core_attn_out = packed_out.transpose(1, 2).contiguous().to(initial_dtype)
    if return_boundary_state:
        return core_attn_out, last_recurrent_state, boundary_state
    return core_attn_out, last_recurrent_state, None


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            self.decode_kernel = CuteDSLGDNKernel()
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            raise ValueError(
                "CuTe DSL backend only supports decode, not prefill. "
                "Use --linear-attn-prefill-backend triton instead."
            )
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel: use FlashInfer only when the selected FlashInfer kernel
        # supports MTP verify. On SM100+ FlashInfer GDN decode is supported, but
        # its MTP verify path is not, so keep Triton as the verify fallback.
        if (
            decode_backend.is_flashinfer() or prefill_backend.is_flashinfer()
        ) and flashinfer_kernel.supports_target_verify:
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)
        self.verify_intermediate_state_indices = torch.arange(
            self.req_to_token_pool.size, dtype=torch.int32, device=model_runner.device
        )
        # Batch-invariant decode chunk-replay slot arena. Replaces the old python dict keyed by
        # (layer_id, slot) with fixed-size GPU tensors indexed by an arena row, so the decode step
        # composes with cuda-graph capture (no python-driven per-slot dict lookup / tolist / loop).
        # Each arena holds, for R rows, the recurrent state entering the current partial 64-chunk
        # (boundary) plus the cached prep buffers / solved T rows / cached v_new rows for the tokens
        # since that boundary. Decode replays torch_chunk_gated_delta_rule over the growing partial
        # chunk to reproduce Megatron's full-sequence per-token output bit-for-bit.
        #
        # R = cuda_graph_max_bs + 1 (last row is the PAD dummy). The mamba pool has ~1400 slots, far
        # too many to preallocate per-slot state for (per-row state is ~HV*C*(4*Dk+2*Dv)+... fp32),
        # but at most cuda_graph_max_bs slots decode concurrently, so an arena sized by the max
        # decode batch matches the true worst case. A host slot->row map (rebuilt each step from the
        # batch's mamba_cache_indices) gathers each batch element to its persistent row. Rows are
        # reclaimed lazily: a slot absent from a decode batch cannot resume without re-prefilling
        # (retraction re-extends; disable_radix_cache => no state resume), and _seed overwrites the
        # row unconditionally, so a reused slot inheriting a stale row is harmless.
        # Two independent sizes. ARENA ROWS hold the expensive per-slot recurrent state
        # (boundary + cached prep/T/v_new, ~0.5 GB/row across all GDN layers): bounded by the number
        # of slots that decode CONCURRENTLY, i.e. cuda_graph_max_bs (decode batch is capped there
        # under graphs; eager decode never runs more distinct slots than fit the graph batch). A
        # 1425-row arena (full mamba pool) would be ~700 GB, infeasible. BATCH BUFFERS (row_map /
        # valid / out) are cheap [B]-shaped tensors indexed by batch row and must span the largest
        # decode batch the runner can dispatch (the whole mamba pool -- the kernel_warmup dummy run
        # decodes at pool size with req_pool_indices all-zero, so 1425 batch rows map to ONE slot ->
        # one arena row, but out[bt] is still written for every bt up to 1424).
        cg_max_bs = getattr(model_runner.server_args, "cuda_graph_max_bs", None) or 0
        self.gdn_arena_max_bs = max(cg_max_bs, 1)
        self.gdn_arena_rows = self.gdn_arena_max_bs + 1  # +1 pad row
        self.gdn_pad_row = self.gdn_arena_rows - 1
        self.gdn_arena = {}  # layer_id -> dict of [R, ...] state tensors (lazily allocated)
        self.gdn_slot_to_row = {}  # slot (int) -> arena row (int), across layers (shared batch)
        self.gdn_free_rows = list(range(self.gdn_arena_max_bs))  # rows [0, max_bs), pad row excluded
        # Largest decode batch the runner can dispatch (full req/mamba pool). Sizes the batch buffers.
        self.gdn_max_batch = int(self.req_to_token_pool.size)
        # Persistent GPU batch->row map (written each step from pinned host memory, so it is a
        # stable-pointer graph input). Filled by _gdn_write_row_map once per decode step.
        dev = model_runner.device
        self._gdn_row_map = torch.zeros(self.gdn_max_batch, dtype=torch.int32, device=dev)
        self._gdn_row_map_host = torch.zeros(
            self.gdn_max_batch, dtype=torch.int32, device="cpu"
        ).pin_memory()
        self._gdn_out = None  # [gdn_max_batch, HV, Dv] bf16 core buffer, shared across layers
        self._gdn_cur_row_map = None  # [B] view set per decode step

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            self.forward_metadata.mamba_track_mask_indices = (
                forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )
        # Build the decode arena batch->row map ONCE per step here (host-side, before the layers
        # run and before any cuda-graph replay), shared by all GDN layers. Only the batch-invariant
        # decode path uses the arena; other modes leave the stashed views untouched.
        if (
            is_batch_invariant_mode_enabled()
            and forward_batch.forward_mode.is_decode()
        ):
            self._gdn_write_row_map(self.forward_metadata.mamba_cache_indices.tolist())

    def init_forward_metadata_capture_cuda_graph(self, bs, num_tokens, req_pool_indices,
                                                 seq_lens, encoder_lens, forward_mode, spec_info):
        # The eager path builds the arena row map in init_forward_metadata, which cuda-graph
        # decode SKIPS (it dispatches here + replay instead). Build it from the same batch->slot
        # metadata so the fixed-pointer row-map buffer holds the right rows when the graph captures
        # (and each replay refreshes it below). Kernel reads the row map indirectly (persistent
        # buffer), so its data pointer is baked into the graph -- only the contents change per step.
        super().init_forward_metadata_capture_cuda_graph(
            bs, num_tokens, req_pool_indices, seq_lens, encoder_lens, forward_mode, spec_info)
        if is_batch_invariant_mode_enabled() and forward_mode.is_decode():
            self._gdn_write_row_map(self.forward_metadata.mamba_cache_indices.tolist())

    def init_forward_metadata_replay_cuda_graph(self, bs, req_pool_indices, seq_lens, seq_lens_sum,
                                                encoder_lens, forward_mode, spec_info, seq_lens_cpu):
        super().init_forward_metadata_replay_cuda_graph(
            bs, req_pool_indices, seq_lens, seq_lens_sum, encoder_lens, forward_mode, spec_info,
            seq_lens_cpu)
        if is_batch_invariant_mode_enabled() and forward_mode.is_decode():
            self._gdn_write_row_map(self.forward_metadata.mamba_cache_indices.tolist())

    def _alloc_gdn_arena(self, layer):
        """Allocate a layer's [R, ...] decode replay arena (R = cuda_graph_max_bs + 1, last row is
        the PAD dummy). Persistent fixed-pointer tensors so the decode step can be cuda-graph
        captured. Indexed by arena row (via the host-maintained slot->row map), not by slot."""
        C = FLA_CHUNK_SIZE
        R = self.gdn_arena_rows
        HV, Dk, Dv = layer.num_v_heads, layer.head_k_dim, layer.head_v_dim
        dev = self.device
        # Allocate as NORMAL tensors even though the first touch may be the warmup dummy decode
        # (which runs under torch.inference_mode). Inference tensors reject the in-place copy_/kernel
        # writes the persistent arena needs from later non-inference contexts (prefill seeding).
        with torch.inference_mode(False):
            z = lambda *s: torch.zeros(R, *s, device=dev, dtype=torch.float32)
            return {
                "boundary": z(HV, Dk, Dv),  # [R,HV,Dk,Dv] recurrent state entering the partial chunk
                "qn": z(C, HV, Dk), "kn": z(C, HV, Dk), "kb": z(C, HV, Dk), "vb": z(C, HV, Dv),
                "g": z(HV, C), "gcum": z(HV, C), "A_row": z(HV, C),
                "T": z(HV, C, C), "vnew": z(HV, C, Dv),
                # GPU-resident per-row width double-buffer: kernel reads i_buf[ar], writes next width
                # to inext_buf[ar]; python rolls i_buf<-inext_buf between launches (in-place, capturable).
                "i_buf": torch.zeros(R, device=dev, dtype=torch.int32),
                "inext_buf": torch.zeros(R, device=dev, dtype=torch.int32),
            }

    def _gdn_arena_for(self, layer):
        arena = self.gdn_arena.get(layer.layer_id)
        if arena is None:
            arena = self._alloc_gdn_arena(layer)
            self.gdn_arena[layer.layer_id] = arena
        return arena

    def _gdn_out_for(self, layer):
        """Lazily allocate the shared [gdn_max_batch, HV, Dv] bf16 core-output buffer (batch-indexed
        by the kernel, reused across all GDN layers within a step). The kernel writes bf16 directly
        (PAD rows zeroed in-kernel), so decode returns this buffer with no python cast/mask. Sized to
        the full pool so the pool-size warmup dummy decode writes out[bt] in bounds for every bt. A
        fixed-pointer persistent buffer so the decode step is cuda-graph capturable."""
        if self._gdn_out is None:
            with torch.inference_mode(False):  # normal tensor: written in-place from any mode
                self._gdn_out = torch.zeros(
                    self.gdn_max_batch, layer.num_v_heads, layer.head_v_dim,
                    device=self.device, dtype=torch.bfloat16,
                )
        return self._gdn_out

    def _gdn_get_row(self, slot, active_slots):
        """Return the arena row for `slot`, assigning one if new. Lazily reclaims a row from a slot
        absent from `active_slots` when the free list is empty (that slot cannot resume decoding
        without a re-prefill, which overwrites the row, so inheriting its stale state is harmless)."""
        r = self.gdn_slot_to_row.get(slot)
        if r is not None:
            return r
        if self.gdn_free_rows:
            r = self.gdn_free_rows.pop()
        else:
            for s, rr in list(self.gdn_slot_to_row.items()):
                if s not in active_slots:
                    del self.gdn_slot_to_row[s]
                    r = rr
                    break
            assert r is not None, "GDN arena overflow: no reclaimable row"
        self.gdn_slot_to_row[slot] = r
        return r

    def _gdn_write_row_map(self, cidx_cpu):
        """Build the batch->arena-row map for one decode step from the batch's cache_indices (a
        single host pass per step, shared by all layers -- not per layer). Writes the persistent GPU
        row-map tensor via a pinned copy and stashes the [B] view for the layer hot path. PAD slots
        map to the pad row; the kernel zeroes those output rows in-kernel (ar == PAD_ROW)."""
        B = len(cidx_cpu)
        assert B <= self.gdn_max_batch, (
            f"GDN decode batch {B} exceeds pool size {self.gdn_max_batch}"
        )
        # The arena has only gdn_arena_max_bs rows; the number of DISTINCT active slots in a step
        # (not the batch length -- duplicates, eg the all-zero warmup dummy batch, share one row)
        # must fit. Distinct slots are bounded by cuda_graph_max_bs in real decode.
        active = {s for s in cidx_cpu if s != PAD_SLOT_ID}
        assert len(active) <= self.gdn_arena_max_bs, (
            f"GDN decode distinct slots {len(active)} exceed arena rows {self.gdn_arena_max_bs}"
        )
        rm_host = self._gdn_row_map_host
        for j, slot in enumerate(cidx_cpu):
            rm_host[j] = self.gdn_pad_row if slot == PAD_SLOT_ID else self._gdn_get_row(slot, active)
        self._gdn_row_map[:B].copy_(rm_host[:B], non_blocking=True)
        self._gdn_cur_row_map = self._gdn_row_map[:B]

    def _gdn_launch_step(self, arena, layer, q_tok, k_tok, v_tok, a_tok, b_tok, row_map, B, out):
        """Advance `B` slots by one token in ONE launch over a (B*HV,) grid, indexing the arena via
        row_map (batch->arena row). Bit-for-bit reproduces row w-1 of torch_chunk_gated_delta_rule
        per slot (only indexing differs from the old per-slot launch). q/k_tok [B,HK,Dk], v_tok
        [B,HV,Dv], a/b_tok [B,HV]. Writes bf16 core rows to out[:B] (batch-indexed), PAD rows
        zeroed in-kernel."""
        C = FLA_CHUNK_SIZE
        HV, HK = layer.num_v_heads, layer.num_k_heads
        Dk, Dv = layer.head_k_dim, layer.head_v_dim
        scale = 1.0 / (Dk ** 0.5)
        _gdn_step_kernel[(B * HV,)](
            q_tok, k_tok, v_tok, a_tok, b_tok, layer.A_log, layer.dt_bias,
            arena["qn"], arena["kn"], arena["kb"], arena["vb"], arena["g"], arena["gcum"],
            arena["A_row"], arena["T"], arena["vnew"], arena["boundary"], out,
            arena["i_buf"], arena["inext_buf"], row_map,
            scale, HV=HV, HK=HK, Dk=Dk, Dv=Dv, CS=C, PAD_ROW=self.gdn_pad_row,
            SPREC=GDN_SCAN_PREC, num_warps=8,
        )
        # Roll width forward for the next step: i_buf <- inext_buf (in-place, capturable). Whole-arena
        # copy is safe: an inactive row already has i_buf == inext_buf (its last active step's roll
        # left them equal and no launch touched it since), so the copy is idempotent for those rows.
        arena["i_buf"].copy_(arena["inext_buf"])

    def _gdn_chunk_replay_decode(
        self,
        layer: "RadixLinearAttention",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Incremental (O(1)-row/step) chunk-replay decode (batch-invariant path), one launch/layer.

        Reproduces Megatron's full-sequence torch_chunk_gated_delta_rule output for each decode
        token bit-for-bit. Each active slot owns a persistent arena row (boundary state + cached prep
        buffers + solved T rows + cached v_new rows). A step appends one token: only row w-1 of
        A/T/v_new is new (leading blocks are stable across steps), so per-step work is O(w) rows for
        the A-row/append reductions and O(1) matmul rows, not the full O(w^2) chunk recompute. On the
        64th token the kernel folds a fresh boundary. All B slots advance in a single (B*HV,) launch
        indexed through the row map -- no python per-slot loop, dict lookup, or cache_indices.tolist
        in the layer hot path (the map is built once per step in _gdn_write_row_map).

        mixed_qkv: [B, q_dim+k_dim+v_dim] post-conv, one decode token per row.
        a, b: [B, HV] pre-gating inputs. Returns [1, B, HV, head_v_dim].
        """
        B = mixed_qkv.shape[0]
        q_flat, k_flat, v_flat = torch.split(
            mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        q_tok = q_flat.view(B, layer.num_k_heads, layer.head_k_dim).contiguous()
        k_tok = k_flat.view(B, layer.num_k_heads, layer.head_k_dim).contiguous()
        v_tok = v_flat.view(B, layer.num_v_heads, layer.head_v_dim).contiguous()
        a = a.contiguous()
        b = b.contiguous()
        arena = self._gdn_arena_for(layer)
        out = self._gdn_out_for(layer)
        row_map = self._gdn_cur_row_map
        self._gdn_launch_step(arena, layer, q_tok, k_tok, v_tok, a, b, row_map, B, out)
        # Kernel writes bf16 directly with PAD rows zeroed in-kernel (via PAD_ROW), so no python
        # cast or validity multiply. Returns [1, B, HV, Dv].
        return out[:B].unsqueeze(0)

    def _seed_gdn_replay_cache(
        self,
        layer: "RadixLinearAttention",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        ssm_states: torch.Tensor,
    ) -> None:
        """Seed the decode chunk-replay cache from a completed batch-invariant prefill.

        For each prefill sequence: the boundary state is the recurrent state after folding all
        complete 64-chunks (obtained by replaying torch_chunk over the complete-chunk prefix from
        the sequence's initial ssm state), and the partial-chunk buffer holds the raw post-conv
        q/k/v + pre-gating a/b of the trailing (seqlen % 64) tokens. Decode appends to this buffer
        and re-runs torch_chunk, so decode token 0 reproduces Megatron's row at position seqlen.

        mixed_qkv is [total_tokens, q_dim+k_dim+v_dim] post-conv; a, b are [total_tokens, HV].
        """
        q_flat, k_flat, v_flat = torch.split(
            mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        q_all = q_flat.view(-1, layer.num_k_heads, layer.head_k_dim)
        k_all = k_flat.view(-1, layer.num_k_heads, layer.head_k_dim)
        v_all = v_flat.view(-1, layer.num_v_heads, layer.head_v_dim)
        qsl_cpu = query_start_loc.tolist()
        cidx_cpu = cache_indices.tolist()
        zero_cidx = torch.zeros(1, dtype=torch.long, device=mixed_qkv.device)
        C = FLA_CHUNK_SIZE
        arena = self._gdn_arena_for(layer)
        out = self._gdn_out_for(layer)  # seeding core output is discarded; write row 0
        active = {s for s in cidx_cpu if s != PAD_SLOT_ID}
        for i in range(len(qsl_cpu) - 1):
            slot = cidx_cpu[i]
            if slot == PAD_SLOT_ID:
                continue
            start, end = qsl_cpu[i], qsl_cpu[i + 1]
            seqlen = end - start
            if seqlen <= 0:
                # Empty sequence: release this slot's arena row so it can be reclaimed.
                r = self.gdn_slot_to_row.pop(slot, None)
                if r is not None:
                    self.gdn_free_rows.append(r)
                continue
            comp = (seqlen // C) * C  # tokens in complete chunks
            # Boundary state = state entering the trailing partial chunk. Fold the complete-chunk
            # prefix from this sequence's initial ssm state (disable_radix_cache => zeros; a
            # non-zero prefix state is honored via ssm_states[slot]). Folding is causal so this
            # equals the boundary the full-sequence forward would produce.
            if comp > 0:
                init_state = (
                    ssm_states[slot].transpose(-1, -2).unsqueeze(0).float().contiguous()
                    if ssm_states is not None
                    else None
                )
                g_c, beta_c = torch_gdn_gating(
                    layer.A_log, a[start : start + comp], b[start : start + comp], layer.dt_bias
                )
                _, boundary, _ = torch_chunk_gated_delta_rule(
                    q_all[start : start + comp].unsqueeze(0),
                    k_all[start : start + comp].unsqueeze(0),
                    v_all[start : start + comp].unsqueeze(0),
                    g=g_c, beta=beta_c,
                    ssm_states=init_state,
                    cache_indices=zero_cidx,
                    query_start_loc=torch.tensor([0, comp], dtype=torch.int32, device=mixed_qkv.device),
                )
                boundary = boundary.detach()
            elif ssm_states is not None:
                # No complete chunk: boundary is the sequence's initial state (zeros when
                # disable_radix_cache, else the prefix-cached state), in [1, HV, K, V] layout.
                boundary = (
                    ssm_states[slot].transpose(-1, -2).unsqueeze(0).float().contiguous().detach()
                )
            else:
                boundary = torch.zeros(
                    1, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim,
                    device=mixed_qkv.device, dtype=torch.float32,
                )
            # Assign this slot an arena row, seed its boundary + reset its width, then advance the
            # trailing partial-chunk tokens through the SAME per-token step decode uses. This builds
            # the cached prep buffers + solved T rows + cached v_new rows so decode token 0 (the next
            # advance) reproduces Megatron's row at position seqlen. seqlen % 64 < 64 so no fold fires
            # during seeding. Runs eager during prefill so a per-slot single-row launch is fine.
            ar = self._gdn_get_row(slot, active)
            arena["boundary"][ar].copy_(boundary[0])
            arena["i_buf"][ar].zero_()
            arena["inext_buf"][ar].zero_()
            row_map = torch.tensor([ar], dtype=torch.int32, device=mixed_qkv.device)
            for t in range(start + comp, end):
                self._gdn_launch_step(
                    arena, layer,
                    q_all[t : t + 1].contiguous(), k_all[t : t + 1].contiguous(),
                    v_all[t : t + 1].contiguous(), a[t : t + 1].contiguous(),
                    b[t : t + 1].contiguous(), row_map, 1, out,
                )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)
        if is_batch_invariant_mode_enabled():
            # Deterministic fixed-order fp32 conv to bit-match Megatron's batch-invariant
            # path (gated_delta_net.py: _causal_depthwise_conv1d in fp32, fp32 silu, then
            # cast to bf16). The fused causal_conv1d_update CUDA kernel accumulates the
            # 4-tap FMA in a different order and diverges by ~1 bf16 ULP on the current
            # token even when all inputs (current token + cached columns) are bit-identical.
            # mixed_qkv is [B, dim] (one decode token); conv_states is [slots, dim, state_len]
            # holding the state_len (=width-1) preceding raw pre-conv columns, oldest first.
            #
            # Fixed-shape / capture-safe: operate on ALL B rows, indexing conv_states directly by
            # cache_indices. PAD slots (cache_indices == -1) index the mamba pool's reserved pad row
            # (the pool allocates size+1, so index -1 is the dummy last row); their conv output is
            # garbage but unused downstream (GDN zeroes PAD output rows in-kernel). This avoids the
            # boolean-mask gather/scatter (cache_indices[valid]) whose data-dependent shape forces a
            # GPU->CPU sync illegal under cuda-graph capture. Valid rows are bit-identical to the
            # masked path (same idx -> same conv_states row -> same window/conv).
            state_len = conv_states.shape[-1]
            prev = conv_states[cache_indices]  # [B, dim, state_len], oldest first
            window = torch.cat([prev, mixed_qkv.unsqueeze(-1)], dim=-1)  # [B, dim, width]
            conv_out = _causal_depthwise_conv1d(
                window.float(),
                layer.conv_weights.unsqueeze(1).float(),
                layer.bias.float() if layer.bias is not None else None,
            )  # [B, dim, width]; last column is the current token's conv output
            act = conv_out[..., -1]
            if layer.activation in ("silu", "swish"):
                act = F.silu(act)
            mixed_qkv = act.to(mixed_qkv.dtype)  # [B, dim]; PAD rows garbage, zeroed by GDN
            # Roll conv_states left: drop oldest, append the current raw pre-conv column. PAD rows
            # write the reserved pad row (index -1); duplicate pad writes are harmless scratch.
            conv_states[cache_indices] = window[:, :, -state_len:].to(conv_states.dtype)
        else:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices,
            )

        # Batch-invariant decode: differentiable incremental chunk-replay that reproduces
        # Megatron's chunked delta rule per token bit-for-bit (the fused recurrent kernel is a
        # different fp32 reduction order, ~7.8e-3 off). See _gdn_chunk_replay_decode.
        if is_batch_invariant_mode_enabled():
            core_attn_out = self._gdn_chunk_replay_decode(
                layer, mixed_qkv, a, b, cache_indices
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return core_attn_out

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_indices = self.verify_intermediate_state_indices
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            if is_batch_invariant_mode_enabled():
                dim, total_len = mixed_qkv.shape
                weight = layer.conv_weights.unsqueeze(1).float()
                bias = layer.bias.float() if layer.bias is not None else None
                # `mixed_qkv` is [dim, total_len] raw pre-conv input (pre-transpose above), a
                # packed row of N sequences delimited by query_start_loc. The causal conv must NOT
                # bleed across sequence boundaries: each sequence's first width-1 outputs see only
                # its own history (zeros for a fresh prefill, or its prefix conv_states window when
                # resuming), never the previous sequence's tail. Run the conv per sequence with the
                # right left-context. For a single fresh-prefill sequence (production: N==1 under
                # disable_radix_cache) this is bit-identical to convolving the whole row.
                state_len = conv_states.shape[-1]  # width - 1
                qsl_cpu = query_start_loc.tolist()
                cidx_cpu = cache_indices.tolist()
                has_init_cpu = has_initial_states.tolist()
                conv_out = mixed_qkv.new_zeros(1, dim, total_len, dtype=torch.float32)
                for i in range(len(qsl_cpu) - 1):
                    c = cidx_cpu[i]
                    if c == PAD_SLOT_ID:
                        continue
                    start, end = qsl_cpu[i], qsl_cpu[i + 1]
                    seqlen_i = end - start
                    if seqlen_i <= 0:
                        continue
                    seg = mixed_qkv[:, start:end]  # [dim, seqlen_i] raw pre-conv input (bf16)
                    seg_f = seg.float()
                    if has_init_cpu[i]:
                        # Prepend this sequence's prefix conv window; slice it back off after conv
                        # so output[t] uses seg's real history. _causal_depthwise_conv1d then
                        # left-pads state_len zeros, which fall entirely on the prepended context.
                        ctx = conv_states[c].float()  # [dim, state_len]
                        x_i = torch.cat([ctx, seg_f], dim=-1).unsqueeze(0).contiguous()
                        conv_out[:, :, start:end] = _causal_depthwise_conv1d(x_i, weight, bias)[
                            :, :, state_len:
                        ]
                    else:
                        x_i = seg_f.unsqueeze(0).contiguous()
                        conv_out[:, :, start:end] = _causal_depthwise_conv1d(x_i, weight, bias)
                    # Persist conv state. The fused causal_conv1d_fn kernel (non-BI path below)
                    # updates conv_states in place with the last state_len raw pre-conv input
                    # columns of each sequence; decode reads these as its initial conv window. The
                    # BI path skips the kernel, so without this the first width-1 decode tokens see
                    # stale conv_states (layer 0 post_conv_qkv diverged by ~23 at decode token 1,
                    # decaying to bit-exact by token 4).
                    if seqlen_i >= state_len:
                        conv_states[c] = seg[:, -state_len:]
                    elif has_init_cpu[i]:
                        # Shift the existing window left and append the new columns.
                        conv_states[c] = torch.cat(
                            [conv_states[c][:, seqlen_i:], seg], dim=-1
                        )
                    else:
                        keep = state_len - seqlen_i
                        conv_states[c, :, :keep] = 0
                        conv_states[c, :, keep:] = seg
                mixed_qkv = F.silu(conv_out[..., :total_len]).to(mixed_qkv.dtype).squeeze(0).transpose(0, 1)[:seq_len]
            else:
                mixed_qkv = causal_conv1d_fn(
                    mixed_qkv,
                    layer.conv_weights,
                    layer.bias,
                    activation=layer.activation,
                    conv_states=conv_states,
                    has_initial_state=has_initial_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                ).transpose(0, 1)[:seq_len]

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            core_attn_out = self.kernel_dispatcher.target_verify(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_state_cache,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
            )
        else:
            ran_torch_chunk = is_batch_invariant_mode_enabled()
            if ran_torch_chunk:
                g, beta = torch_gdn_gating(layer.A_log, a, b, layer.dt_bias)
                core_attn_out, last_recurrent_state, h = torch_chunk_gated_delta_rule(
                    query, key, value, g=g, beta=beta,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                )
                # Seed the decode chunk-replay cache: per slot, the recurrent state entering the
                # trailing partial chunk (boundary) and the raw post-conv q/k/v + pre-gating a/b
                # of the partial-chunk tokens. Decode continues the replay from here so its first
                # token reproduces Megatron's full-sequence row at position prompt_len bit-for-bit.
                self._seed_gdn_replay_cache(
                    layer, mixed_qkv, a, b, query_start_loc, cache_indices, ssm_states
                )
            else:
                g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
                core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                    q=query,
                    k=key,
                    v=value,
                    g=g,
                    beta=beta,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                )

            if (is_npu() or is_cpu()) and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state
            elif ran_torch_chunk and last_recurrent_state is not None:
                # Persist the SSM recurrent state so decode can continue the recurrence.
                # The non-BI CUDA extend kernel writes ssm_states in place (it receives
                # ssm_states as initial_state with cache_indices), but
                # torch_chunk_gated_delta_rule (batch-invariant path) only READS ssm_states
                # and returns the final state, so on CUDA it must be written back explicitly.
                # Without this the first decode tokens start from a stale SSM state (layer 0
                # decode core_attn_out diverged ~1.5, decaying over steps as the gated decay
                # washed out the error).
                #
                # torch_chunk_gated_delta_rule builds the state as [B, HV, K, V]
                # (k_i^T @ v_new), but the ssm_states pool is [slots, HV, V, K]. K==V==128 so
                # a shape check would pass but the last two dims are transposed; the read at
                # initial_state never exposed this because disable_radix_cache means prefill
                # always starts from zeros. Transpose to the pool layout on write.
                ssm_states[cache_indices] = (
                    last_recurrent_state.transpose(-1, -2)
                    .to(ssm_states.dtype, copy=False)
                    .contiguous()
                )

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out
