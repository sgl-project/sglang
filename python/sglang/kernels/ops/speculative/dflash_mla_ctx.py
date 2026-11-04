"""Every DFLASH MLA draft layer's context-KV rows from one GEMM, one norm launch and one RoPE."""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _latent_rmsnorm_kernel(
    lat_ptr,  # [n, L, R + P]
    w_ptr,  # [L, R]
    out_ptr,  # [L, n, R]
    n,
    eps,
    L: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
):
    row = tl.program_id(0)
    layer = tl.program_id(1)
    cols = tl.arange(0, R)
    x = tl.load(lat_ptr + (row * L + layer) * (R + P) + cols).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / R + eps)
    w = tl.load(w_ptr + layer * R + cols).to(tl.float32)
    tl.store(
        out_ptr + (layer * n + row) * R + cols,
        (x * rstd * w).to(out_ptr.dtype.element_ty),
    )


def _same_rope(rope: torch.nn.Module, other: torch.nn.Module) -> bool:
    """One RoPE module rotates exactly as the other: class, settings and caches."""
    if rope is other:
        return True
    if type(rope) is not type(other) or vars(rope).keys() != vars(other).keys():
        return False
    for name, value in vars(rope).items():
        theirs = vars(other)[name]
        if isinstance(value, torch.Tensor):
            if not (isinstance(theirs, torch.Tensor) and torch.equal(value, theirs)):
                return False
        elif isinstance(value, (bool, int, float, str, type(None))) and value != theirs:
            return False
    their_buffers = dict(other.named_buffers())
    return all(
        torch.equal(buf, their_buffers[name]) for name, buf in rope.named_buffers()
    )


def _latent(attn: torch.nn.Module) -> tuple:
    return (
        attn.kv_lora_rank,
        attn.qk_rope_head_dim,
        attn.kv_a_layernorm.variance_epsilon,
        attn.fused_qkv_a_proj_with_mqa.weight.dtype,
    )


class MlaContextKV:
    """The draft layers' stacked kv_a rows (a copy: built after the weights load) and norm weights."""

    def __init__(self, attns: Sequence[torch.nn.Module]):
        a0 = attns[0]
        self.attns = list(attns)
        self.R, self.P = int(a0.kv_lora_rank), int(a0.qk_rope_head_dim)
        self.eps = float(a0.kv_a_layernorm.variance_epsilon)
        self.w = torch.cat(
            [a.fused_qkv_a_proj_with_mqa.weight[a.q_lora_rank :] for a in attns], 0
        ).contiguous()
        self.norm_w = torch.stack([a.kv_a_layernorm.weight for a in attns]).contiguous()
        self.rotary_emb = a0.rotary_emb

    @staticmethod
    def blocker(attns: Sequence[torch.nn.Module]) -> Optional[str]:
        """Why one stacked GEMM, norm launch and RoPE cannot write these low-rank-q
        MLA layers' context-KV rows, or None when they can."""
        if not attns:
            return "no context attention layers"
        rank, _, _, dtype = _latent(attns[0])
        if dtype not in (torch.bfloat16, torch.float16):
            return f"kv_a weight dtype {dtype}"
        # the norm kernel reduces a whole latent row with one tl.arange
        if rank & (rank - 1):
            return f"kv_lora_rank {rank} is not a power of two"
        for i, attn in enumerate(attns[1:], 1):
            if _latent(attn) != _latent(attns[0]):
                return f"layer {i}'s latent shape, norm eps or kv_a dtype differs"
            if not _same_rope(attn.rotary_emb, attns[0].rotary_emb):
                return f"layer {i}'s RoPE differs from layer 0's"
        return None

    def project_all(
        self, positions: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(k_nope [L, n, 1, R] normalized, k_pe [L, n, 1, P] rotated); one RoPE head a layer."""
        n, L, R, P = hidden.shape[0], len(self.attns), self.R, self.P
        lat = F.linear(hidden, self.w).view(n, L, R + P)
        k_nope = torch.empty((L, n, R), dtype=lat.dtype, device=hidden.device)
        if n:
            _latent_rmsnorm_kernel[(n, L)](
                lat, self.norm_w, k_nope, n, self.eps, L=L, R=R, P=P
            )
        k_pe = lat[..., R:].contiguous()
        _, k_pe = self.rotary_emb(positions, torch.empty_like(k_pe), k_pe)
        k_pe = k_pe.view(n, L, P).transpose(0, 1).contiguous()
        return k_nope.unsqueeze(2), k_pe.unsqueeze(2)
