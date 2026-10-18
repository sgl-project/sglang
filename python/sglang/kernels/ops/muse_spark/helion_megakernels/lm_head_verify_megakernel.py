"""Muse Spark 1.2 spec-decode verify logits (one request, T = 8 or 16 draft
tokens) for TP4 as one persistent Helion megakernel: the vocab-parallel lm head
GEMV, the logit scale and softcap, and the all-gather of the vocab shards.

Replaces the stock FP32-output GEMM, logit scale, NCCL all-gather, logits copy
and softcap launches.

The shard is padded to whole G blocks, so every block lands unmasked and its
words travel tagged to every rank as they are produced; A polls them locally and
drops the padding.

Roots (rank r):

    G  lm head block -> capped FP32 logits of this rank's vocab shard (symm)  [V'/BV]
    A  every peer's shard block -> full logits, rank-major vocab              [V'/BA]

The softcap is elementwise, so capping each shard before the gather matches the
stock cap of the gathered logits.
"""

from __future__ import annotations

import helion
import helion.language as hl
import msgspec
import torch

G_BLOCK = 64
A_BLOCK = 128


def _muse_lm_head_verify_megakernel(
    x: torch.Tensor,  # [T, K] bf16, final-norm output
    w: torch.Tensor,  # [V / W, K] bf16, this rank's lm head shard
    shard: torch.Tensor,  # [T, V'] fp32 symm, V / W padded to G blocks
    logits: torch.Tensor,  # [T, W, V / W] fp32
    group_name: hl.ProcessGroupName,
    world: hl.constexpr,
    scale: hl.constexpr,
    cap: hl.constexpr,
) -> torch.Tensor:
    hl.specialize(x.size())
    hl.specialize(x.stride())
    hl.specialize(w.size())
    hl.specialize(w.stride())
    hl.specialize(shard.size())
    hl.specialize(shard.stride())
    hl.specialize(logits.size())
    hl.specialize(logits.stride())

    tokens = x.size(0)
    vocab = w.size(0)
    k_blk = hl.register_block_size(w.size(1))
    remote = torch.ops.symm_mem.get_remote_tensors(shard, group_name)

    hl.pdl_wait()
    # G. Stock: the FP32 lm head GEMM (_compute_lm_head), logit scale and softcap.
    for g_v in hl.tile(shard.size(1), block_size=G_BLOCK):
        acc = hl.zeros([g_v, tokens], dtype=torch.float32)
        rows = (g_v.index < vocab)[:, None]
        # Weights stream once: evict_first keeps the megakernel code in L2.
        for g_k in hl.tile(w.size(1), block_size=k_blk):
            acc = hl.dot(
                hl.load(w, [g_v, g_k], extra_mask=rows, eviction_policy="evict_first"),
                x[:, g_k].permute(1, 0),
                acc=acc,
            )
        shard[:, g_v] = torch.tanh(acc.permute(1, 0) * scale / cap) * cap

    # A. Stock: the vocab-shard all-gather (NCCL) and the logits copy.
    for a_t, a_v in hl.tile([tokens, shard.size(1)], block_size=[tokens, A_BLOCK]):
        keep = (a_v.index < vocab)[None, :]
        for peer in hl.static_range(world):
            hl.store(logits, [a_t, peer, a_v], remote[peer][a_t, a_v], extra_mask=keep)
    hl.pdl_launch_dependents()
    return logits


def make_config(
    *,
    k_blk: int = 128,
    num_warps: int = 4,
    num_stages: int = 5,
    pipeline: str = "dynamic",
    pid_type: str = "persistent_blocked",
    **extra: object,
) -> helion.Config:
    return helion.Config(
        block_sizes=[k_blk],
        cross_loop_pipeline=pipeline,
        indexing="pointer",
        num_stages=num_stages,
        num_warps=num_warps,
        pid_type=pid_type,
        **extra,
    )


def build_kernel(config: helion.Config) -> object:
    return helion.kernel(
        _muse_lm_head_verify_megakernel,
        config=config,
        static_shapes=True,
        backend="triton",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


class LogitScalars(msgspec.Struct, frozen=True):
    """Compile-time logit scale and softcap."""

    scale: float
    cap: float


class MuseLmHeadVerifyMegakernel:
    """Per-rank shard buffer and call for the T-token logits.

    The call takes the replicated final-norm output and this rank's lm head
    shard and writes the full capped FP32 logits, replicated on every rank.
    """

    def __init__(
        self,
        device: torch.device,
        group: object,
        scalars: LogitScalars,
        *,
        tokens: int,
        vocab_shard: int,
        config: helion.Config | None = None,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        self.kernel = build_kernel(config or make_config())
        self.group_name = group.group_name
        self.world = group.size()
        self.tokens = tokens
        padded = -(-vocab_shard // G_BLOCK) * G_BLOCK
        self.shard = symm_mem.empty(tokens, padded, dtype=torch.float32, device=device)
        symm_mem.rendezvous(self.shard, self.group_name)
        self.constants = (self.world, scalars.scale, scalars.cap)

    def __call__(
        self, *, hidden_states: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        self.kernel(
            hidden_states,
            weight,
            self.shard,
            out.view(self.tokens, self.world, -1),
            self.group_name,
            *self.constants,
        )
        return out


_INSTANCES: dict[
    tuple[int, object, LogitScalars, int, int], MuseLmHeadVerifyMegakernel
] = {}


def get_lm_head_verify_megakernel(
    *,
    device: torch.device,
    group: object,
    scalars: LogitScalars,
    tokens: int,
    vocab_shard: int,
) -> MuseLmHeadVerifyMegakernel:
    """Build once per (device, group, scalars, T, shard); the constructor
    rendezvouses a symmetric buffer, so every rank must reach a new key together,
    outside graph capture."""

    key = (device.index, group, scalars, tokens, vocab_shard)
    if key not in _INSTANCES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("build the Helion verify lm head before capture")
        _INSTANCES[key] = MuseLmHeadVerifyMegakernel(
            device, group, scalars, tokens=tokens, vocab_shard=vocab_shard
        )
    return _INSTANCES[key]


__all__ = [
    "LogitScalars",
    "MuseLmHeadVerifyMegakernel",
    "get_lm_head_verify_megakernel",
    "make_config",
]
