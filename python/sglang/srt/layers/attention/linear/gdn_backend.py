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


if _HAVE_TRITON:
    from triton.language.extra import libdevice

    @triton.jit
    def _fwd_sub_kernel(A, C: tl.constexpr):
        # One program per (batch*head*chunk). A points at a [C, C] row-major fp32 matrix
        # holding the strictly-lower A_strict (zeros on/above the diagonal). The whole
        # matrix is loaded into a register tile M once; the substitution runs entirely
        # in registers (no global read-after-write across loop iterations, which would
        # be an unordered hazard), then M is stored once.
        #
        # Forward substitution:
        #   for i in 1..C-1: M[i, :i] += sum_{m<i} M[i, m] * M[m, :i]
        # The reduction over m is a fixed tree order (tl.sum), deterministic and
        # independent of the number of chunks. It is NOT bit-identical to the Python
        # loop's `(row.unsqueeze(-1) * sub).sum(-2)` (that path is ~1 fp32 ULP different),
        # which is fine: Megatron prefill and SGLang both call this same kernel, so they
        # stay identical to each other, and gradients still flow (see _SolveFwdSub).
        pid = tl.program_id(0)
        base = A + pid * C * C
        r = tl.arange(0, C)
        M = tl.load(base + r[:, None] * C + r[None, :])  # [C, C] fp32 in registers
        for i in range(1, C):
            # a_im[m] = M[i, m] for m < i else 0 ; broadcast over columns j
            a_im = tl.where((r < i), tl.sum(tl.where(r[:, None] == i, M, 0.0), 0), 0.0)  # [C], row i
            # prod[m, j] = a_im[m] * M[m, j]
            prod = a_im[:, None] * M  # [C(m), C(j)]
            acc = tl.sum(prod, 0)  # [C(j)], reduce over m
            row_i = tl.sum(tl.where(r[:, None] == i, M, 0.0), 0)  # current M[i, :]
            new_row = tl.where(r < i, row_i + acc, row_i)  # update only cols < i
            M = tl.where(r[:, None] == i, new_row[None, :], M)
        tl.store(base + r[:, None] * C + r[None, :], M)

    @triton.jit
    def _gdn_replay_A_kernel(
        kn_ptr,
        kb_ptr,
        gcum_ptr,  # kn,kb [w,HV,Dk]; gcum [w,HV]
        A_ptr,  # out: [HV, CS, CS] strictly-lower A_strict
        W,
        HV: tl.constexpr,
        Dk: tl.constexpr,
        CS: tl.constexpr,
    ):
        """One program per value-head. Computes A_strict = -(kb @ kn^T)*decay (strictly lower)
        and STORES it to a global fp32 tensor. Splitting this out (vs inlining the solve) is what
        makes the replay bit-identical to torch_chunk: torch_chunk feeds the standalone
        _fwd_sub_kernel an A that was rounded to fp32 in global memory by its `@`, whereas an
        inline solve consumed A while still register-resident (tl.dot accumulator, unrounded),
        drifting ~1 fp32 ULP and tipping a bf16 boundary on ~1% of real decode tokens.
        """
        h = tl.program_id(0)
        r = tl.arange(0, CS)
        dk = tl.arange(0, Dk)
        valid = r < W
        kn = tl.load(
            kn_ptr + (r[:, None] * HV + h) * Dk + dk[None, :], mask=valid[:, None], other=0.0
        )
        kb = tl.load(
            kb_ptr + (r[:, None] * HV + h) * Dk + dk[None, :], mask=valid[:, None], other=0.0
        )
        gcum = tl.load(gcum_ptr + r * HV + h, mask=valid, other=0.0)
        # See main kernel for the exponent-masking rationale (guards inf*0=NaN on padded rows /
        # the discarded upper triangle).
        kept = (r[:, None] >= r[None, :]) & valid[:, None] & valid[None, :]
        gd = tl.where(kept, gcum[:, None] - gcum[None, :], 0.0)
        decay = tl.where(kept, libdevice.exp(gd), 0.0)
        A = -tl.dot(kb, tl.trans(kn), input_precision="ieee") * decay
        A = tl.where(r[:, None] > r[None, :], A, 0.0)
        tl.store(A_ptr + (h * CS + r[:, None]) * CS + r[None, :], A)

    @triton.jit
    def _gdn_fused_replay_kernel(
        qn_ptr,
        kn_ptr,
        kb_ptr,
        vb_ptr,
        gcum_ptr,  # qn,kn,kb [w,HV,Dk]; vb [w,HV,Dv]; gcum [w,HV]
        T_ptr,  # [HV, CS, CS] strictly-lower solve output (from _fwd_sub_kernel); eye added here
        S_ptr,
        out_ptr,
        Snew_ptr,
        W,
        B,
        HV: tl.constexpr,
        Dk: tl.constexpr,
        Dv: tl.constexpr,
        CS: tl.constexpr,
        COMMIT: tl.constexpr,
    ):
        """One program per (batch, value-head). Replays a single (<=64-token) partial chunk
        seeded by boundary state S, reproducing row w-1 of torch_chunk_gated_delta_rule
        bit-for-bit. When COMMIT (w==64) it also writes the new boundary state Snew.

        The elementwise reductions torch.cumsum / l2norm-sum are NOT done here (tl.cumsum and
        tl.sum diverge from torch's reduction order); they are precomputed in _gdn_replay_prep
        and passed in as qn/kn/kb/vb/gcum. The forward-substitution solve is ALSO not done here:
        T is precomputed via the shared _fwd_sub_kernel on a globally-rounded A (an inline solve
        drifts ~1 fp32 ULP, see _gdn_replay_A_kernel). This kernel does only tl.dot matmuls and
        libdevice.exp, all proven bit-exact vs torch_chunk under batch_invariant.
        """
        pid = tl.program_id(0)
        bt = pid // HV
        h = pid % HV
        r = tl.arange(0, CS)
        dk = tl.arange(0, Dk)
        dv = tl.arange(0, Dv)
        valid = r < W

        qn = tl.load(
            qn_ptr + (r[:, None] * HV + h) * Dk + dk[None, :], mask=valid[:, None], other=0.0
        )
        kn = tl.load(
            kn_ptr + (r[:, None] * HV + h) * Dk + dk[None, :], mask=valid[:, None], other=0.0
        )
        kb = tl.load(
            kb_ptr + (r[:, None] * HV + h) * Dk + dk[None, :], mask=valid[:, None], other=0.0
        )
        vb = tl.load(
            vb_ptr + (r[:, None] * HV + h) * Dv + dv[None, :], mask=valid[:, None], other=0.0
        )
        gcum = tl.load(gcum_ptr + r * HV + h, mask=valid, other=0.0)

        # Mask the EXPONENT before exp (matches torch_chunk's `(g[:,None]-g[None,:]).tril().exp()`).
        # Two overflow sources to guard, both -> exp(+large)=inf -> inf*0=NaN in a later tl.dot:
        #  (1) upper triangle i<j (discarded): raw diff can be large-positive.
        #  (2) PADDED rows i>=W: gcum there is loaded as 0, which is > any valid row's negative
        #      gcum, so gcum[pad_i]-gcum[valid_j] is large-positive and IS in the kept lower
        #      triangle. Restrict the kept mask to valid rows AND cols; zero the exponent elsewhere.
        kept = (r[:, None] >= r[None, :]) & valid[:, None] & valid[None, :]
        gd = tl.where(kept, gcum[:, None] - gcum[None, :], 0.0)
        decay = tl.where(kept, libdevice.exp(gd), 0.0)

        # Load the precomputed strictly-lower solve and add the unit diagonal (torch_chunk does
        # `attn = _solve_fwd_sub(attn) + eye`). input_precision="ieee" on every tl.dot below uses
        # true fp32 (no tf32), matching torch_chunk's bmm (batch_invariant persistent kernel, also
        # ieee) and Megatron bit-for-bit; default tl.dot would use tf32 and diverge ~1.5e-2.
        T = tl.load(T_ptr + (h * CS + r[:, None]) * CS + r[None, :])
        T = T + tl.where(r[:, None] == r[None, :], 1.0, 0.0)

        val = tl.dot(T, vb, input_precision="ieee")
        kg = kb * libdevice.exp(gcum)[:, None]
        kcd = tl.dot(T, kg, input_precision="ieee")

        S = tl.load(S_ptr + (bt * HV + h) * Dk * Dv + dk[:, None] * Dv + dv[None, :])
        attn2 = tl.dot(qn, tl.trans(kn), input_precision="ieee") * decay
        attn2 = tl.where(r[:, None] >= r[None, :], attn2, 0.0)
        # Materialize every tl.dot output that is then added/subtracted to another tensor via
        # tl.where. Triton otherwise keeps the dot accumulator in fused wide registers and rounds
        # `x +/- tl.dot(...)` differently from torch's materialize-then-add (2.9e-6, bf16-ULP
        # class). The tl.where forces an fp32 register round-trip, matching torch bit-for-bit.
        # (mul-by-decay above does NOT need this; only add/sub of a dot triggers the fusion.)
        keep = r[:, None] >= 0
        v_prime = tl.where(keep, tl.dot(kcd, S, input_precision="ieee"), 0.0)
        v_new = val - v_prime
        attn_inter = tl.where(
            keep, tl.dot(qn * libdevice.exp(gcum)[:, None], S, input_precision="ieee"), 0.0
        )
        intra = tl.where(keep, tl.dot(attn2, v_new, input_precision="ieee"), 0.0)
        core = attn_inter + intra

        last = W - 1
        core_last = tl.sum(tl.where(r[:, None] == last, core, 0.0), 0)
        tl.store(out_ptr + (bt * HV + h) * Dv + dv, core_last)

        if COMMIT:
            glast = tl.sum(tl.where(r == (CS - 1), gcum, 0.0), 0)
            kdec = kn * libdevice.exp(glast - gcum)[:, None]
            upd = tl.where(
                dk[:, None] >= 0, tl.dot(tl.trans(kdec), v_new, input_precision="ieee"), 0.0
            )
            Snew = S * libdevice.exp(glast) + upd
            tl.store(
                Snew_ptr + (bt * HV + h) * Dk * Dv + dk[:, None] * Dv + dv[None, :], Snew
            )


def _gdn_replay_prep(
    qh: torch.Tensor,
    kh: torch.Tensor,
    vh: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple:
    """Precompute the elementwise prep for the fused replay kernel, bit-identical to
    torch_chunk_gated_delta_rule's per-token math (l2norm, GQA repeat, scale, gating, per-chunk
    cumsum). tl.cumsum / tl.sum diverge from torch's reduction order, so these reductions run in
    torch and only tl.dot matmuls go into the kernel.

    qh,kh: [w,HK,Dk]; vh: [w,HV,Dv]; a,b: [w,HV]. Returns qn,kn,kb [w,HV,Dk], vb [w,HV,Dv],
    gcum [w,HV], all fp32 contiguous.
    """
    w, HK, Dk = qh.shape
    HV = vh.shape[1]
    rep = HV // HK
    scale = 1.0 / (Dk**0.5)
    # l2norm in fp32 -> bf16 round-trip (matches torch_chunk), on HK heads, then repeat to HV.
    qf = qh.float()
    kf = kh.float()
    qn = (qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).float()
    kn = (kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).float()
    qn = qn.repeat_interleave(rep, dim=1) * scale  # [w,HV,Dk]
    kn = kn.repeat_interleave(rep, dim=1)
    g, beta = torch_gdn_gating(A_log, a, b, dt_bias)  # g [1,w,HV] fp32, beta [1,w,HV] bf16
    g = g[0].contiguous()  # [w,HV]
    beta = beta[0].contiguous().float()  # [w,HV]
    kb = kn * beta[..., None]
    vb = vh.float() * beta[..., None]
    # Per-chunk cumsum in torch_chunk's exact layout: it transposes to [b,h,s] so the cumsum runs
    # over the CONTIGUOUS seq dim. Doing cumsum(0) on [w,HV] (HV contiguous) reduces in a different
    # memory order -> 3e-5 gcum diff. Transpose to [HV,w], cumsum(-1), transpose back.
    gcum = g.transpose(0, 1).contiguous().cumsum(-1).transpose(0, 1).contiguous()  # [w,HV]
    return (
        qn.contiguous(),
        kn.contiguous(),
        kb.contiguous(),
        vb.contiguous(),
        gcum.contiguous(),
    )


def _gdn_fused_replay(
    qh: torch.Tensor,
    kh: torch.Tensor,
    vh: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    S: torch.Tensor,
    commit: bool,
) -> tuple:
    """Fused single-launch chunk-replay for one decode token, over the partial-chunk buffer.

    qh,kh: [w,HK,Dk]; vh: [w,HV,Dv]; a,b: [w,HV]; S: [B,HV,Dk,Dv] boundary state. Returns
    (out [B,HV,Dv] fp32 = row w-1 of torch_chunk, Snew [B,HV,Dk,Dv] new boundary if commit else S).
    """
    qn, kn, kb, vb, gcum = _gdn_replay_prep(qh, kh, vh, a, b, A_log, dt_bias)
    w = qh.shape[0]
    HV = vh.shape[1]
    Dk = qh.shape[2]
    Dv = vh.shape[2]
    B = S.shape[0]
    C = FLA_CHUNK_SIZE
    # Compute A_strict, round it to fp32 global memory, then solve with the SAME _fwd_sub_kernel
    # torch_chunk uses. Splitting the solve out of the main kernel (rather than inlining it while
    # A is still a tl.dot accumulator) is what makes the replay bit-identical to torch_chunk; an
    # inline solve drifts ~1 fp32 ULP and tips a bf16 boundary on ~1% of real decode tokens.
    T = torch.empty(HV, C, C, device=S.device, dtype=torch.float32)
    _gdn_replay_A_kernel[(HV,)](
        kn, kb, gcum, T, w, HV=HV, Dk=Dk, CS=C, num_warps=4,
    )
    _fwd_sub_kernel[(HV,)](T, C=C)  # in-place strictly-lower solve, == torch_chunk's _solve_fwd_sub
    out = torch.empty(B, HV, Dv, device=S.device, dtype=torch.float32)
    Snew = torch.empty_like(S) if commit else S
    _gdn_fused_replay_kernel[(B * HV,)](
        qn, kn, kb, vb, gcum, T, S, out, Snew,
        w, B, HV=HV, Dk=Dk, Dv=Dv, CS=C, COMMIT=commit, num_warps=4,
    )
    return out, Snew


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
    query = query.float()
    key = key.float()
    query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    query = query.to(initial_dtype)
    key = key.to(initial_dtype)
    num_v_heads = value.shape[2]
    num_k_heads = key.shape[2]
    if num_v_heads // num_k_heads > 1:
        repeat = num_v_heads // num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
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

    # Per-chunk cumsum. torch.cumsum on the full [b, h, num_chunks, chunk_size] tensor
    # batches all num_chunks rows into one reduction whose fp32 accumulation order depends
    # on the row count, so a longer sequence (more chunks) shifts chunk 0 by ~1 bf16 ULP
    # (7.6e-6) and propagates through g.exp() to core_attn_out (1.5e-5). Running cumsum per
    # chunk keeps the reduction row count constant, making it independent of sequence length
    # so sglang (short prefill) and Megatron (full teacher-forced sequence) match bit-exactly.
    g = torch.stack([g[:, :, i].cumsum(dim=-1) for i in range(g.shape[2])], dim=2)
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

    initial_state = ssm_states[cache_indices].to(value) if ssm_states is not None else None
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    # State entering the final (possibly partial) chunk: the recurrent state after folding all
    # COMPLETE 64-chunks. Captured right after the (num_complete_chunks-1)-th fold so it is
    # correct both for aligned prompts (equals last_recurrent_state) and misaligned prompts
    # (the seed decode replays the trailing partial-chunk tokens from).
    num_complete_chunks = sequence_length // chunk_size
    boundary_state = last_recurrent_state  # correct when num_complete_chunks == 0

    for i in range(0, total_sequence_length // chunk_size):
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

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
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
        # Batch-invariant decode chunk-replay cache, keyed by (layer_id, slot). Each entry holds
        # the recurrent state entering the current partial 64-chunk (boundary) plus the raw
        # post-conv q/k/v and pre-gating a/b for every token since that boundary. Decode replays
        # torch_chunk_gated_delta_rule over the growing partial chunk to reproduce Megatron's
        # full-sequence per-token output bit-for-bit. Populated at prefill (trailing partial
        # chunk) and advanced each decode step; the 64th token folds a new boundary and clears
        # the token buffer. Only allocated for active slots (scales with batch, not pool size).
        self.gdn_replay_cache = {}

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

    def _gdn_chunk_replay_decode(
        self,
        layer: "RadixLinearAttention",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Incremental chunk-replay decode (batch-invariant path).

        Reproduces Megatron's full-sequence torch_chunk_gated_delta_rule output for each decode
        token bit-for-bit. For every active slot we keep, since the last completed 64-boundary:
        the recurrent boundary state and the raw post-conv q/k/v + pre-gating a/b of each token
        in the partial chunk. Each step appends the new token and replays the partial chunk
        (width = tokens-since-boundary) with the SAME torch_chunk_gated_delta_rule Megatron uses,
        seeded by the boundary state, taking row w-1. When the partial chunk reaches 64 tokens we
        fold a fresh boundary (torch_chunk's returned last_recurrent_state) and clear the buffer.
        The fused single-launch kernel (_gdn_fused_replay) was ~1e-6..1e-3 off torch_chunk in fp32
        on real activations and crossed a bf16 boundary on ~1% of tokens, so torch_chunk is used
        directly. Decode is inference-only, so no autograd is needed.

        mixed_qkv: [B, q_dim+k_dim+v_dim] post-conv, one decode token per row.
        a, b: [B, HV] pre-gating inputs. Returns [1, B, HV, head_v_dim].
        """
        B = mixed_qkv.shape[0]
        q_flat, k_flat, v_flat = torch.split(
            mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        q_tok = q_flat.view(B, layer.num_k_heads, layer.head_k_dim)
        k_tok = k_flat.view(B, layer.num_k_heads, layer.head_k_dim)
        v_tok = v_flat.view(B, layer.num_v_heads, layer.head_v_dim)
        cidx_cpu = cache_indices.tolist()
        outputs = mixed_qkv.new_empty(B, layer.num_v_heads, layer.head_v_dim)

        for j in range(B):
            slot = cidx_cpu[j]
            if slot == PAD_SLOT_ID:
                outputs[j].zero_()
                continue
            entry = self.gdn_replay_cache.get((layer.layer_id, slot))
            if entry is None:
                # No prefill-seeded partial chunk (aligned prompt or missing): start empty at a
                # zero-fold boundary. The prefill path seeds this for the common misaligned case.
                entry = {
                    "boundary": torch.zeros(
                        1, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim,
                        device=mixed_qkv.device, dtype=torch.float32,
                    ),
                    "q": [], "k": [], "v": [], "a": [], "b": [],
                }
                self.gdn_replay_cache[(layer.layer_id, slot)] = entry
            entry["q"].append(q_tok[j])
            entry["k"].append(k_tok[j])
            entry["v"].append(v_tok[j])
            entry["a"].append(a[j])
            entry["b"].append(b[j])
            w = len(entry["q"])
            q_seq = torch.stack(entry["q"], dim=0)  # [w, HK, Dk]
            k_seq = torch.stack(entry["k"], dim=0)
            v_seq = torch.stack(entry["v"], dim=0)  # [w, HV, Dv]
            a_seq = torch.stack(entry["a"], dim=0)  # [w, HV]
            b_seq = torch.stack(entry["b"], dim=0)
            commit = w == FLA_CHUNK_SIZE
            # Replay the partial chunk with the fused single-launch kernel, which is now
            # fp32-bit-identical to the torch_chunk_gated_delta_rule Megatron uses (row w-1 and,
            # on the 64th token, the folded boundary state). The earlier fused kernel drifted
            # ~1 fp32 ULP because it solved the forward-substitution inline off an unrounded
            # tl.dot accumulator; _gdn_fused_replay now rounds A to global memory and runs the
            # SAME _fwd_sub_kernel torch_chunk does, so decode reproduces the full-sequence
            # per-token output bit-for-bit. Decode is inference-only so no autograd is needed.
            out_j, last = _gdn_fused_replay(
                q_seq, k_seq, v_seq, a_seq, b_seq, layer.A_log, layer.dt_bias,
                entry["boundary"], commit,
            )
            outputs[j] = out_j[0].to(outputs.dtype)
            if commit:
                # Completed a 64-chunk: fold it into a fresh boundary and reset the token buffer.
                entry["boundary"] = last
                entry["q"].clear(); entry["k"].clear(); entry["v"].clear()
                entry["a"].clear(); entry["b"].clear()

        return outputs.unsqueeze(0)

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
        for i in range(len(qsl_cpu) - 1):
            slot = cidx_cpu[i]
            if slot == PAD_SLOT_ID:
                continue
            start, end = qsl_cpu[i], qsl_cpu[i + 1]
            seqlen = end - start
            if seqlen <= 0:
                self.gdn_replay_cache.pop((layer.layer_id, slot), None)
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
            entry = {
                "boundary": boundary,
                "q": [q_all[t] for t in range(start + comp, end)],
                "k": [k_all[t] for t in range(start + comp, end)],
                "v": [v_all[t] for t in range(start + comp, end)],
                "a": [a[t] for t in range(start + comp, end)],
                "b": [b[t] for t in range(start + comp, end)],
            }
            self.gdn_replay_cache[(layer.layer_id, slot)] = entry

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
            state_len = conv_states.shape[-1]
            valid = cache_indices != PAD_SLOT_ID
            idx = cache_indices[valid]
            x_new = mixed_qkv[valid]  # [Bv, dim]
            prev = conv_states[idx]  # [Bv, dim, state_len], oldest first
            window = torch.cat([prev, x_new.unsqueeze(-1)], dim=-1)  # [Bv, dim, width]
            conv_out = _causal_depthwise_conv1d(
                window.float(),
                layer.conv_weights.unsqueeze(1).float(),
                layer.bias.float() if layer.bias is not None else None,
            )  # [Bv, dim, width]; last column is the current token's conv output
            act = conv_out[..., -1]
            if layer.activation in ("silu", "swish"):
                act = F.silu(act)
            out = mixed_qkv.clone()
            out[valid] = act.to(mixed_qkv.dtype)
            mixed_qkv = out
            # Roll conv_states left: drop oldest, append the current raw pre-conv column.
            conv_states[idx] = window[:, :, -state_len:].to(conv_states.dtype)
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
                x = mixed_qkv.unsqueeze(0).float().contiguous()
                weight = layer.conv_weights.unsqueeze(1).float()
                bias = layer.bias.float() if layer.bias is not None else None
                conv_out = _causal_depthwise_conv1d(x, weight, bias)
                # Persist conv state. The fused causal_conv1d_fn kernel (non-BI path
                # below) updates conv_states in place with the last state_len=width-1
                # raw pre-conv input columns of each sequence; decode reads these as its
                # initial conv window. The BI path skips the kernel, so without this the
                # first width-1 decode tokens see stale conv_states (layer 0 post_conv_qkv
                # diverged by ~23 at decode token 1, decaying to bit-exact by token 4).
                # `mixed_qkv` is [dim, total_len] raw pre-conv input (pre-transpose above).
                state_len = conv_states.shape[-1]
                qsl_cpu = query_start_loc.tolist()
                cidx_cpu = cache_indices.tolist()
                has_init_cpu = has_initial_states.tolist()
                for i in range(len(qsl_cpu) - 1):
                    c = cidx_cpu[i]
                    if c == PAD_SLOT_ID:
                        continue
                    start, end = qsl_cpu[i], qsl_cpu[i + 1]
                    seqlen_i = end - start
                    if seqlen_i <= 0:
                        continue
                    seg = mixed_qkv[:, start:end]  # [dim, seqlen_i] raw pre-conv input
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
