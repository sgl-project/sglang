"""K1-K3 fused tree-GDN target-verify — PyTorch reference implementation.

Executable SPEC for the Triton kernels in gdn_tree_triton.py: every
sequential ancestor walk is replaced by dense masked algebra.

Step-shared (build_tree_structure — ONCE per verify step, reused by every
GDN layer; depends only on the draft tree, not on layer weights):
    P     inclusive ancestor mask [T,T], built by log-depth repeated
          squaring of (I + parent-edge matrix) — no per-row walks.

Per layer (tree_gdn_fused_verify):
    K1    prefix = P @ g            (path sums as one matvec)
          A_ij  = beta_i * exp(prefix_i - prefix_j) * (k_i . k_j),
                  j a PROPER ancestor of i (strict mask)
    K2    (I + A) U = beta * (V - exp(prefix) * K H0^T)
          A is strictly lower triangular under the topological node order
          (parent[i] < i), so this is ONE batched unit-lower-triangular
          solve (cuBLAS trsm) — 16ish dense level steps inside the library,
          not 129 scalar ones.
    K3    O = exp(prefix) * (Q H0^T) + (QK ⊙ D_inclusive) @ U

Conventions (pinned by the float64 reference harness: fp64 reference at 1e-16,
production chain kernel at 1e-8):
  - state memory layout [*, HV, V, K]: element (k, v) at flat v*K + k
  - g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b)
  - q, k l2-normalized in-kernel (eps 1e-6 INSIDE the sqrt), then q *= K^-0.5
  - ratio form exp(prefix_i - prefix_j) only (every factor <= 1); off-mask
    entries are -inf BEFORE exp, never exp'd then masked
Optionally returns a LazyGDNVerifyState carrying the verify factors
(k raw, u, prefix) that the commit / fold pass consumes.
"""

from __future__ import annotations

import math
import weakref
from typing import NamedTuple, Optional

import msgspec

import torch
import torch.nn.functional as F


class TreeStructure(NamedTuple):
    parent: torch.Tensor          # [N, T] int64, parent[i] < i, root = -1
    anc_mask: torch.Tensor        # [N, T, T] bool, inclusive (j <= i)
    anc_f: torch.Tensor           # same, fp32 (the path-sum operator P)
    anc_u8: torch.Tensor          # same, uint8 contiguous (for Triton loads)
    logmask_incl: torch.Tensor    # [N, T, T] fp32: 0 on j <= i, -inf off
    logmask_strict: torch.Tensor  # [N, T, T] fp32: 0 on j <  i, -inf off
    max_depth: int                # max number of PROPER ancestors


def build_tree_structure(parent: torch.Tensor) -> TreeStructure:
    """Once per verify step; shared by all GDN layers. Log-depth, GEMM-only."""
    parent = parent.long()
    N, T = parent.shape
    device = parent.device
    eye = torch.eye(T, dtype=torch.float32, device=device).expand(N, T, T)
    edge = torch.zeros(N, T, T, dtype=torch.float32, device=device)
    edge.scatter_(2, parent.clamp_min(0).unsqueeze(-1), (parent >= 0).float().unsqueeze(-1))
    reach = eye + edge
    for _ in range(max(1, math.ceil(math.log2(max(T, 2))))):
        reach = (reach @ reach).clamp_(max=1.0)  # doubles reachable distance
    anc = reach > 0
    max_depth = int((anc.sum(-1) - 1).max().item())
    neg_inf = float("-inf")
    logmask_incl = torch.where(anc, 0.0, neg_inf).float()
    logmask_strict = logmask_incl.clone()
    logmask_strict.diagonal(dim1=-2, dim2=-1).fill_(neg_inf)
    return TreeStructure(
        parent=parent, anc_mask=anc, anc_f=anc.float(),
        anc_u8=anc.to(torch.uint8).contiguous(),
        logmask_incl=logmask_incl, logmask_strict=logmask_strict,
        max_depth=max_depth,
    )


import triton
import triton.language as tl


@triton.jit
def _tree_struct_walk_kernel(
    parent_ptr, anc_b_ptr, anc_f_ptr, anc_u8_ptr, lm_incl_ptr, lm_strict_ptr,
    stride_pn, stride_an, stride_ai,
    T: tl.constexpr, NP2: tl.constexpr,
):
    """One program per (batch row, node i): walk the parent chain and write
    the ancestor-closure row plus both logmask rows. Capture-safe."""
    i_n = tl.program_id(0)
    i = tl.program_id(1)
    offs = tl.arange(0, NP2)
    m = offs < T
    NEG_INF = float("-inf")
    anc_row = tl.zeros((NP2,), tl.int1)
    cur = i
    it = 0
    while cur >= 0 and it <= T:
        anc_row = anc_row | (offs == cur)
        cur = tl.load(parent_ptr + i_n * stride_pn + cur).to(tl.int32)
        it += 1
    base = i_n * stride_an + i * stride_ai
    tl.store(anc_b_ptr + base + offs, anc_row.to(tl.int8), mask=m)
    tl.store(anc_f_ptr + base + offs, anc_row.to(tl.float32), mask=m)
    tl.store(anc_u8_ptr + base + offs, anc_row.to(tl.int8), mask=m)
    incl = tl.where(anc_row, 0.0, NEG_INF)
    tl.store(lm_incl_ptr + base + offs, incl, mask=m)
    strict = tl.where(anc_row & (offs != i), 0.0, NEG_INF)
    tl.store(lm_strict_ptr + base + offs, strict, mask=m)


def build_tree_structure_into_fast(parent_src: torch.Tensor, buf: "TreeStructure") -> "TreeStructure":
    """Triton parent-chain walk variant of build_tree_structure_into:
    ~5-10us per call vs ~66us for the matmul-squaring build. Same outputs."""
    parent = parent_src.long()
    N, T = parent.shape
    buf.parent.copy_(parent)
    NP2 = triton.next_power_of_2(T)
    _tree_struct_walk_kernel[(N, T)](
        buf.parent, buf.anc_mask, buf.anc_f, buf.anc_u8,
        buf.logmask_incl, buf.logmask_strict,
        buf.parent.stride(0), buf.anc_mask.stride(0), buf.anc_mask.stride(1),
        T=T, NP2=NP2,
    )
    return buf


def build_tree_structure_into(parent_src: torch.Tensor, buf: "TreeStructure") -> "TreeStructure":
    """Recompute the tree structure from `parent_src` IN-PLACE into the
    persistent tensors of `buf` (capture-safe: no .item()/sync, stable
    pointers for CUDA-graph replay). Scratch allocs come from the caching
    allocator and are pointer-irrelevant to consumers."""
    parent = parent_src.long()
    N, T = parent.shape
    device = parent.device
    eye = torch.eye(T, dtype=torch.float32, device=device).expand(N, T, T)
    edge = torch.zeros(N, T, T, dtype=torch.float32, device=device)
    edge.scatter_(2, parent.clamp_min(0).unsqueeze(-1), (parent >= 0).float().unsqueeze(-1))
    reach = eye + edge
    for _ in range(max(1, math.ceil(math.log2(max(T, 2))))):
        reach = (reach @ reach).clamp_(max=1.0)
    anc = reach > 0
    neg_inf = float("-inf")
    buf.parent.copy_(parent)
    buf.anc_mask.copy_(anc)
    buf.anc_f.copy_(anc.float())
    buf.anc_u8.copy_(anc.to(torch.uint8))
    incl = torch.where(anc, 0.0, neg_inf).float()
    buf.logmask_incl.copy_(incl)
    incl_strict = incl.clone()
    incl_strict.diagonal(dim1=-2, dim2=-1).fill_(neg_inf)
    buf.logmask_strict.copy_(incl_strict)
    return buf


def alloc_tree_structure_buffers(
    N: int, T: int, max_depth: int, device
) -> "TreeStructure":
    """Persistent zeroed TreeStructure for in-place per-step rebuilds."""
    assert 0 <= max_depth < T
    return TreeStructure(
        parent=torch.zeros(N, T, dtype=torch.long, device=device),
        anc_mask=torch.zeros(N, T, T, dtype=torch.bool, device=device),
        anc_f=torch.zeros(N, T, T, dtype=torch.float32, device=device),
        anc_u8=torch.zeros(N, T, T, dtype=torch.uint8, device=device),
        logmask_incl=torch.zeros(N, T, T, dtype=torch.float32, device=device),
        logmask_strict=torch.zeros(N, T, T, dtype=torch.float32, device=device),
        max_depth=max_depth,
    )


def build_tree_structure_static(parent: torch.Tensor) -> TreeStructure:
    """CUDA-graph-capture-safe variant: no .item()/sync. max_depth is T-1 —
    it only bounds the commit kernel's ancestor walk."""
    parent = parent.long()
    N, T = parent.shape
    device = parent.device
    eye = torch.eye(T, dtype=torch.float32, device=device).expand(N, T, T)
    edge = torch.zeros(N, T, T, dtype=torch.float32, device=device)
    edge.scatter_(2, parent.clamp_min(0).unsqueeze(-1), (parent >= 0).float().unsqueeze(-1))
    reach = eye + edge
    for _ in range(max(1, math.ceil(math.log2(max(T, 2))))):
        reach = (reach @ reach).clamp_(max=1.0)
    anc = reach > 0
    max_depth = T - 1
    neg_inf = float("-inf")
    logmask_incl = torch.where(anc, 0.0, neg_inf).float()
    logmask_strict = logmask_incl.clone()
    logmask_strict.diagonal(dim1=-2, dim2=-1).fill_(neg_inf)
    return TreeStructure(
        parent=parent, anc_mask=anc, anc_f=anc.float(),
        anc_u8=anc.to(torch.uint8).contiguous(),
        logmask_incl=logmask_incl, logmask_strict=logmask_strict,
        max_depth=max_depth,
    )


_TREE_CACHE: dict[int, tuple] = {}


def get_tree_structure(parent: torch.Tensor) -> TreeStructure:
    """Per-step cached tree build for the sglang dispatch (one build shared by
    all GDN layers in a verify step).

    Cache key is the parent tensor's identity, guarded by a weakref so a
    recycled id can never alias a dead tensor. Semantics by mode:
    - CUDA graph: the backend reuses a persistent metadata buffer, so the
      build kernels are captured ONCE and replay each step, reading the
      buffer contents refreshed by that step's conv/tree kernels.
    - eager: the backend allocates a fresh metadata tensor per step, so the
      cache misses once per step and the tree is rebuilt exactly once.
    """
    key = id(parent)
    hit = _TREE_CACHE.get(key)
    if hit is not None:
        ref, tree = hit
        if ref() is parent:
            return tree
    tree = build_tree_structure_static(parent)
    _TREE_CACHE[key] = (weakref.ref(parent), tree)
    if len(_TREE_CACHE) > 32:
        for stale in [k for k, (r, _) in _TREE_CACHE.items() if r() is None]:
            _TREE_CACHE.pop(stale, None)
    return tree



class LazyGDNVerifyState(msgspec.Struct):
    """Verify factors the commit / fold pass consumes (no state written yet)."""

    k: torch.Tensor
    u: torch.Tensor
    prefix: torch.Tensor
    initial_state_source: torch.Tensor
    initial_state_indices: torch.Tensor
    retrieve_parent_token: torch.Tensor
    max_tree_depth: int
    use_qk_l2norm_in_kernel: bool
    cu_seqlens: Optional[torch.Tensor]


def tree_gdn_fused_verify(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    tree: TreeStructure,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    return_lazy_state: bool = False,
    solve_mode: str = "neumann",
):
    """Same contract as tree_sigmoid_gating_delta_rule_target_verify, with the
    tree passed as a prebuilt TreeStructure. Uniform-T batch (Y4). q,k:
    [B,T,H,K]; v: [B,T,HV,V]; a,b: [B,T,HV]; A_log,dt_bias: [HV];
    initial_state_source: [pool,HV,V,K] (kernel memory layout)."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    if scale is None:
        scale = K**-0.5

    g = -torch.exp(A_log.float()) * F.softplus(
        a.float() + dt_bias.float(), beta=softplus_beta, threshold=softplus_threshold
    )                                       # [B,T,HV]
    beta = torch.sigmoid(b.float())         # [B,T,HV]

    qf, kf = q.float(), k.float()
    if use_qk_l2norm_in_kernel:
        qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
        kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    qf = qf * scale

    # ---- K1: prefix (path-sum matvec), masks, Grams ------------------------
    prefix = torch.einsum("bij,bjh->bih", tree.anc_f, g)       # [B,T,HV] inclusive
    pre = prefix.permute(0, 2, 1)                              # [B,HV,T]

    kh = kf.permute(0, 2, 1, 3)                                # [B,H,T,K]
    qh = qf.permute(0, 2, 1, 3)
    KK = (kh @ kh.transpose(-1, -2)).repeat_interleave(G, dim=1)  # [B,HV,T,T]
    QK = (qh @ kh.transpose(-1, -2)).repeat_interleave(G, dim=1)

    delta = pre.unsqueeze(-1) - pre.unsqueeze(-2)              # prefix_i - prefix_j
    D = (delta + tree.logmask_incl.unsqueeze(1)).exp()         # [B,HV,T,T], 0 off-mask
    D_strict = (delta + tree.logmask_strict.unsqueeze(1)).exp()

    betah = beta.permute(0, 2, 1).unsqueeze(-1)                # [B,HV,T,1]
    A = betah * D_strict * KK                                  # strict ancestors only

    # ---- K2: RHS + one batched unit-lower-triangular solve -----------------
    idx = initial_state_indices.long()
    valid = (idx >= 0).view(B, 1, 1, 1).float()
    H0vk = initial_state_source.float()[idx.clamp_min(0)] * valid  # [B,HV,V,K]
    H0kv = H0vk.transpose(-1, -2)                                  # [B,HV,K,V]
    khv = kh.repeat_interleave(G, dim=1)                           # [B,HV,T,K]
    qhv = qh.repeat_interleave(G, dim=1)
    kH0 = khv @ H0kv                                               # [B,HV,T,V]
    qH0 = qhv @ H0kv
    epre = pre.exp().unsqueeze(-1)                                 # [B,HV,T,1]
    rhs = betah * (v.float().permute(0, 2, 1, 3) - epre * kH0)     # [B,HV,T,V]
    if solve_mode == "trsm":
        U = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
    else:
        # A is nilpotent (A^m = 0, m = depth+1): (I+A)^-1 = sum_j (-A)^j,
        # applied by repeated squaring — log2(T) tiny batched GEMMs, exact.
        # Using T (not depth) keeps the op count shape-static for CUDA graphs.
        neg_A = -A
        U = rhs + neg_A @ rhs                       # covers j < 2
        P2 = neg_A @ neg_A
        steps = max(1, math.ceil(math.log2(max(T, 2))))
        for i in range(1, steps):
            U = U + P2 @ U                          # covers j < 2^(i+1)
            if i < steps - 1:
                P2 = P2 @ P2

    # ---- K3: outputs --------------------------------------------------------
    O = epre * qH0 + (QK * D) @ U                                  # [B,HV,T,V]
    out = O.permute(0, 2, 1, 3).contiguous()                       # [B,T,HV,V]

    if return_lazy_state:
        lazy = LazyGDNVerifyState(
            k=k,                       # RAW k; commit kernel re-normalizes
            u=U,                       # [N,HV,T,V]
            prefix=prefix,             # [N,T,HV]
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            retrieve_parent_token=tree.parent,
            max_tree_depth=tree.max_depth,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=None,
        )
        return out, lazy
    return out


def fold_accepted(lazy, accepted_leaf: int) -> torch.Tensor:
    """The fold — the other half of the primitive (torch spec).

    Commits the accepted root→leaf path into the recurrent state using the
    factors the VERIFY pass already produced (k raw, U, prefix): no rollback
    (rejected siblings never enter the sum) and no re-solve (U_i depends only
    on i's ancestors, and the accepted set is a single path, so the verify
    pass's U rows are exactly the corrected pseudo-values the fold needs):

        S* = exp(prefix_leaf) · H0 + Σ_{j ⪯ leaf} exp(prefix_leaf − prefix_j) · k_j u_jᵀ

    Returns the new committed state in kernel memory layout [B, HV, V, K].
    The all-rejected edge case is accepted_leaf = root (length-1 path); the
    state always advances. The compiled equivalent is
    advance_ssm_states_along_accept_paths in chunk_tree_verify.py.
    """
    B, T, H, K = lazy.k.shape
    HV, V = lazy.u.shape[1], lazy.u.shape[-1]
    G = HV // H
    idx = lazy.initial_state_indices.long()
    valid = (idx >= 0).view(B, 1, 1, 1).float()
    H0 = lazy.initial_state_source.float()[idx.clamp_min(0)] * valid  # [B,HV,V,K]

    kf = lazy.k.float()
    if lazy.use_qk_l2norm_in_kernel:
        kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    k_hv = kf.repeat_interleave(G, dim=2)          # [B,T,HV,K]
    pre = lazy.prefix.float()                      # [B,T,HV]

    path = []
    p = int(accepted_leaf)
    parent = lazy.retrieve_parent_token.long()
    while p >= 0:
        path.append(p)
        p = int(parent[0, p])

    pre_leaf = pre[:, accepted_leaf]               # [B,HV]
    S = torch.exp(pre_leaf).view(B, HV, 1, 1) * H0
    for j in path:
        w = torch.exp(pre_leaf - pre[:, j]).view(B, HV, 1, 1)      # ratio ≤ 1
        S = S + w * torch.einsum("bhv,bhk->bhvk", lazy.u[:, :, j], k_hv[:, j])
    return S
