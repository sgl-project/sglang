"""Every DFLASH MLA draft layer's context-KV rows from one GEMM, one norm launch and one RoPE."""

import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.dflash_mla_ctx import stacked_latent_rmsnorm
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.runtime_context import get_exec, get_parallel

logger = logging.getLogger(__name__)


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


def _latent_config(attn: torch.nn.Module) -> tuple:
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
        self.kv_a_weight = torch.cat(
            [a.fused_qkv_a_proj_with_mqa.weight[a.q_lora_rank :] for a in attns], 0
        ).contiguous()
        self.norm_weight = torch.stack(
            [a.kv_a_layernorm.weight for a in attns]
        ).contiguous()
        self.rotary_emb = a0.rotary_emb

    @classmethod
    def for_worker(cls, worker) -> Optional["MlaContextKV"]:
        """The writer of a DFLASH worker's draft layers under --enable-amd-monokernel,
        or None. Build it on first use: it stacks the draft's kv_a rows, which load
        after the worker's init."""
        from sglang.srt.models.dflash import DFlashDraftModel, DFlashMLAAttention

        if not get_exec().kernel.enable_amd_monokernel:
            return None
        draft = worker.draft_model
        attns = [layer.self_attn for layer in draft.layers]
        # one GEMM serves every layer only when each reads the same projected row
        if (
            type(draft).prepare_context_hidden_for_kv
            is not DFlashDraftModel.prepare_context_hidden_for_kv
            or not all(
                isinstance(a, DFlashMLAAttention) and a.q_lora_rank is not None
                for a in attns
            )
        ):
            blocker = "not low-rank-q MLA layers reading one projected context row"
        else:
            blocker = cls.blocker(attns)
        ctx = None
        if blocker is None:
            ctx = cls(attns)
            # the stacked kv_a copy would keep serving the old weights
            draft._derived_weight_cache_error = (
                "Online weight updates are not supported with the stacked DFLASH MLA "
                "context KV of --enable-amd-monokernel."
            )
        if get_parallel().tp_rank == 0:
            status = (
                f"{len(attns)} layers" if blocker is None else f"disabled: {blocker}"
            )
            logger.info("DFLASH MLA fused context KV: %s", status)
        return ctx

    @staticmethod
    def blocker(attns: Sequence[torch.nn.Module]) -> Optional[str]:
        """Why one stacked GEMM, norm launch and RoPE cannot write these low-rank-q
        MLA layers' context-KV rows, or None when they can."""
        if not attns:
            return "no context attention layers"
        rank, _, _, dtype = _latent_config(attns[0])
        if dtype not in (torch.bfloat16, torch.float16):
            return f"kv_a weight dtype {dtype}"
        # the norm kernel reduces a whole latent row with one tl.arange
        if rank & (rank - 1):
            return f"kv_lora_rank {rank} is not a power of two"
        for i, attn in enumerate(attns[1:], 1):
            if _latent_config(attn) != _latent_config(attns[0]):
                return f"layer {i}'s latent shape, norm eps or kv_a dtype differs"
            if not _same_rope(attn.rotary_emb, attns[0].rotary_emb):
                return f"layer {i}'s RoPE differs from layer 0's"
        return None

    def project_all(
        self, positions: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(k_nope [L, n, 1, R] normalized, k_pe [L, n, 1, P] rotated); one RoPE head a layer."""
        n, L, R, P = hidden.shape[0], len(self.attns), self.R, self.P
        lat = F.linear(hidden, self.kv_a_weight).view(n, L, R + P)
        k_nope = stacked_latent_rmsnorm(lat, self.norm_weight, self.eps)
        k_pe = lat[..., R:].contiguous()
        _, k_pe = self.rotary_emb(positions, torch.empty_like(k_pe), k_pe)
        k_pe = k_pe.view(n, L, P).transpose(0, 1).contiguous()
        return k_nope.unsqueeze(2), k_pe.unsqueeze(2)

    def write(
        self,
        pool,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        cache_loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Every layer's context-KV rows into the committed prefix of each
        cache_loc_2d row; set_mla_kv_buffer_prefix_valid's masking, built once."""
        k_nope, k_pe = self.project_all(positions, hidden)
        offsets = torch.arange(cache_loc_2d.shape[1], device=cache_loc_2d.device)
        loc = KVWriteLoc(
            torch.where(offsets < commit_lens[:, None], cache_loc_2d, 0).reshape(-1),
            physical=True,
        )
        for i, attn in enumerate(self.attns):
            pool.set_mla_kv_buffer(attn.attn_mqa, loc, k_nope[i], k_pe[i])
