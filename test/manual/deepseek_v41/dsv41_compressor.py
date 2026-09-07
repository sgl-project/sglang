import torch
from dsv41_linear import Linear
from dsv41_norm import RMSNorm
from torch import nn


def pool_groups(kv: torch.Tensor, score: torch.Tensor, ratio: int) -> torch.Tensor:
    """Softmax-gated pooling of consecutive groups of `ratio` tokens along dim 1.
    kv, score: [b, n * ratio, d] fp32 -> [b, n, d]"""
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio))
    return (kv * score.softmax(dim=2)).sum(dim=2)


class Compressor(nn.Module):
    """Pools compress_ratio consecutive tokens into one pre-RoPE KV latent. Returns None
    while a group is still filling; the partial group is carried in kv_state / score_state."""

    def __init__(
        self,
        dim: int,
        head_dim: int,
        compress_ratio: int,
        norm_eps: float,
        max_batch_size: int,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.norm = RMSNorm(head_dim, norm_eps)
        # ratio 1 is a plain bf16 projection; the softmax pooling above 1 runs in fp32.
        self.wkv = Linear(
            dim, head_dim, dtype=torch.float32 if compress_ratio > 1 else torch.bfloat16
        )
        if compress_ratio > 1:
            self.wgate = Linear(dim, head_dim, dtype=torch.float32)
            state_shape = (max_batch_size, compress_ratio, head_dim)
            self.register_buffer(
                "kv_state",
                torch.zeros(state_shape, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "score_state",
                torch.full(state_shape, -torch.inf, dtype=torch.float32),
                persistent=False,
            )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:
            return self.norm(self.wkv(x))

        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:
                kv, self.kv_state[:bsz, :remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                score, self.score_state[:bsz, :remainder] = score.split(
                    [cutoff, remainder], dim=1
                )
            kv = pool_groups(kv, score, ratio)
        else:
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = pool_groups(self.kv_state[:bsz], self.score_state[:bsz], ratio)
        if not should_compress:
            return None
        return self.norm(kv.to(dtype))
