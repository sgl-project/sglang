# SPDX-License-Identifier: Apache-2.0
"""Hand-written PTX/tcgen05 KDA chunked-prefill kernel (GB300 / sm_103a).

Vendored from the upstream ``kda_prefill`` artifact (commit 33583615): the CUDA
source in ``kernels/jit/csrc/attention/kda_prefill.cu`` plus the FLA-signature
``chunk_kda_fwd`` wrapper below, which is a drop-in for
``fla.ops.kda.chunk_fwd.chunk_kda_fwd`` on the inference forward path
(K = V = 128, chunk_size = 64).

The extension is JIT-compiled with ``torch.utils.cpp_extension`` on first use
(~1-2 min, cached under ``TORCH_EXTENSIONS_DIR``); concurrent TP ranks
serialize on torch's build lock and then load the cached .so.
"""

import os

import torch

K = 128  # head dim (qk == v), fixed by the kernel
CHUNK = 64  # kernel chunk size, fixed

_EXT_NAME = "kda_prefill_ptx"
_ext = None


def load_ext():
    """JIT-load the CUDA extension (cached after first call)."""
    global _ext
    if _ext is None:
        from torch.utils import cpp_extension

        src = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../../jit/csrc/attention/kda_prefill.cu",
        )
        # -lcuda: cuTensorMapEncodeTiled (driver API). The stubs dir covers
        # boxes whose real libcuda.so lives off the default linker path;
        # ld.so still binds the driver's libcuda.so.1 at import time.
        stubs = os.path.join(
            cpp_extension.CUDA_HOME or "/usr/local/cuda", "lib64", "stubs"
        )
        _ext = cpp_extension.load(
            name=_EXT_NAME,
            sources=[src],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "-use_fast_math",
                "-lineinfo",
                "-gencode",
                "arch=compute_103a,code=sm_103a",
            ],
            extra_cflags=["-O3"],
            extra_ldflags=[f"-L{stubs}", "-lcuda"],
        )
    return _ext


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context=None,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
):
    """Inference drop-in for fla's chunk_kda_fwd (forward only).

    Argument mapping onto the CUDA kernel:
      q/k/v [B,T,H,128] bf16 -> flat [T,H,128] (B>1 folds to varlen with a
        synthetic equal-length cu_seqlens; B==1 squeezes).
      g [B,T,H,128]: use_gate_in_kernel=False -> pre-transformed glog, fp32
        narrowed to bf16 here (the kernel's GM=0 contract); =True -> RAW gate
        input (bf16) with the transform fused in-kernel (GM=1 softplus /
        GM=2 safe-gate). Following fla, the safe-gate TRANSFORM is selected
        by `lower_bound is not None`; the `safe_gate` flag alone is fla's
        intra-path hint and does not change the math here.
      beta [B,T,H] bf16 (fp32 also accepted; widened to fp32 in the ext).
      use_qk_l2norm_in_kernel=True accepts raw q/k and applies FLA-compatible
        L2 normalization (eps=1e-6, bf16 rounding) in the CUDA tile loads.
      use_beta_sigmoid_in_kernel=True accepts beta logits and fuses sigmoid.
      cu_seqlens: host values are needed for the kernel's per-sequence piece
        table -- pass cu_seqlens_cpu to avoid the D2H sync; chunk_indices is
        accepted and ignored (the kernel derives its own piece table).
      initial_state [N,H,128,128] fp32 or None (zeros).
      return_intermediate_states=True returns dense fp32 chunk-boundary states
        [1, NT, H, 128, 128] at tuple index 10.

    Returns the fla-shaped 12-tuple: (o [B,T,H,128] bf16, final_state
    [N,H,128,128] fp32 or None, then Nones, ..., h, initial_state).
    """
    assert chunk_size == CHUNK, (
        f"kda_prefill supports chunk_size={CHUNK} only, got {chunk_size}"
    )
    if cp_context is not None or disable_recompute:
        raise NotImplementedError(
            "kda_prefill is the inference forward path: cp_context, "
            "and disable_recompute are training-side knobs it does not implement"
        )
    if allow_neg_eigval and use_beta_sigmoid_in_kernel:
        raise NotImplementedError(
            "allow_neg_eigval=True requires 2*sigmoid(beta), which is not "
            "implemented by the fused beta path; pass pre-activated beta with "
            "use_beta_sigmoid_in_kernel=False"
        )
    if state_v_first and initial_state is not None:
        # [V,K]-layout state: pure transpose (K==V==128), exact, ~us/call
        initial_state = initial_state.transpose(-1, -2).contiguous()
    assert q.dim() == 4 and q.shape[-1] == K and v.shape[-1] == K, (
        f"expected [B,T,H,{K}] q/k/v, got q={tuple(q.shape)} v={tuple(v.shape)}"
    )
    B, T, H, _ = q.shape

    cu_cpu = None
    if cu_seqlens is not None or cu_seqlens_cpu is not None:
        assert B == 1, "cu_seqlens requires B == 1 (flattened varlen batch)"
        src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
        cu_cpu = torch.as_tensor(src, dtype=torch.int32).cpu()
    elif B > 1:
        # eqlen batch == varlen with equal lengths (host-known, no sync)
        cu_cpu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32)

    Tt = B * T
    qf = q.reshape(Tt, H, K).contiguous()
    kf = k.reshape(Tt, H, K).contiguous()
    vf = v.reshape(Tt, H, K).contiguous()
    betaf = beta.reshape(Tt, H).contiguous()

    if use_gate_in_kernel:
        assert A_log is not None and dt_bias is not None, (
            "use_gate_in_kernel=True requires A_log and dt_bias"
        )
        assert g.dtype == torch.bfloat16, f"raw gate input must be bf16, got {g.dtype}"
        gf = g.reshape(Tt, H, K).contiguous()
        sg = lower_bound is not None  # fla: lb presence selects safe-gate
        a_log_flat = A_log.reshape(H).to(torch.float32).contiguous()
        dtb = dt_bias.reshape(H * K).to(torch.float32).contiguous()
        lb = float(lower_bound) if lower_bound is not None else 0.0
    else:
        gf = g.reshape(Tt, H, K)
        if gf.dtype != torch.bfloat16:  # pre-transformed glog: bf16 narrow
            gf = gf.to(torch.bfloat16)
        gf = gf.contiguous()
        sg, a_log_flat, dtb, lb = False, None, None, 0.0

    h = None
    if return_intermediate_states:
        lens = (cu_cpu[1:] - cu_cpu[:-1]).tolist() if cu_cpu is not None else [Tt]
        nt = sum((int(length) + CHUNK - 1) // CHUNK for length in lens)
        h = torch.empty(nt, H, K, K, dtype=torch.float32, device=q.device)

    o, Sf = load_ext().kda_prefill_fwd(
        qf,
        kf,
        vf,
        gf,
        betaf,
        float(scale),
        initial_state=initial_state,
        cu_seqlens=cu_cpu,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=a_log_flat,
        dt_bias=dtb,
        safe_gate=sg,
        lower_bound=lb,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        h_per_chunk=h,
        h_v_first=state_v_first,
    )

    o = o.view(B, T, H, K)
    if state_v_first:
        Sf = Sf.transpose(-1, -2).contiguous()
    final = Sf if output_final_state else None
    return (
        o,
        final,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None if h is None else h.unsqueeze(0),
        initial_state,
    )
