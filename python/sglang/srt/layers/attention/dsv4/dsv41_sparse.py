"""DeepSeek V4.1 low-ratio (compress ratios 1 and 2) compressor and indexer, torch path.

Only kv_source layers compress; the layers that follow with the same ratio read
the source's latent pool. index_source layers score the shared latents and
publish the top-k slots that the following layers attend to through FlashMLA.
The modules here are plain torch for bring-up; the attention itself already runs
on the FlashMLA sparse kernels.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.dsv41_compressor import (
    DeepseekV41Compressor as DeepseekV41Compressor,
)
from sglang.srt.layers.attention.dsv4.dsv41_compressor import RMSNorm as RMSNorm
from sglang.srt.layers.attention.dsv4.dsv41_compressor import (
    last_token_per_request as last_token_per_request,
)
from sglang.srt.layers.attention.dsv4.dsv41_compressor import (
    pair_partners_decode as pair_partners_decode,
)
from sglang.kernels.ops.attention.dsv4 import linear_bf16_fp32
from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix

_FUSED_ROPE_FQ4 = os.environ.get("SGLANG_SHALLOW_FUSED_ROPE_FQ4", "1") == "1"


def _rope_fq4(x, freqs, rope_dim):
    """fake_quant_fp4(rope_tail(x, freqs, rope_dim)), fused when enabled.

    The eager pair is 41 pointwise ops and runs at 16 call sites per decode step,
    measured as 656 elementwise launches (9.3% of a bs=1 step at 1.32 us each).
    The fused kernel is bit-exact against the pair on every production shape;
    SGLANG_SHALLOW_FUSED_ROPE_FQ4=0 restores the eager path.
    """
    if _FUSED_ROPE_FQ4:
        from sglang.kernels.ops.attention.dsv4.rope_fake_quant_fp4 import (
            rope_tail_fake_quant_fp4,
        )

        return rope_tail_fake_quant_fp4(x, freqs, rope_dim)
    return fake_quant_fp4(rope_tail(x, freqs, rope_dim))


class RMSNorm(nn.Module):
    """fp32 statistics and fp32 weight multiply, cast back at the very end."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and torch.version.cuda is not None
            and x.dtype in (torch.bfloat16, torch.float32)
            and self.weight.dtype in (torch.bfloat16, torch.float32)
            and x.shape[-1] in (128, 512)
            and x.numel() <= 64 * x.shape[-1]
            and x.is_contiguous()
            and self.weight.is_contiguous()
        ):
            from sglang.kernels.ops.attention.dsv4.rmsnorm_fp32 import rmsnorm_fp32

            return rmsnorm_fp32(x, self.weight, self.eps)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def token_req_indices(forward_batch, *, num_tokens=None) -> torch.Tensor:
    """req_pool_indices repeated once per token of the batch."""
    req = forward_batch.req_pool_indices.to(torch.int64)
    if forward_batch.forward_mode.is_decode():
        return req
    if forward_batch.forward_mode.is_target_verify():
        return torch.repeat_interleave(
            req, int(forward_batch.spec_info.draft_token_num), output_size=num_tokens
        )
    assert forward_batch.forward_mode.is_extend(), (
        "the V4.1 torch attention path serves extend and decode only"
    )
    return torch.repeat_interleave(
        req, forward_batch.extend_seq_lens.to(torch.int64), output_size=num_tokens
    )


def topk_from_scores(
    s: torch.Tensor,
    lens: torch.Tensor,
    topk: int,
    candidate_blocks: Optional[int] = None,
    candidate_block_size: Optional[int] = None,
    consume: Optional[torch.Tensor] = None,
    select_candidate_blocks=None,
    pre_masked: bool = False,
):
    """Masked two-level top-k over compressed-position scores.

    s [rows, n]; lens [rows] = visible compressed length per row; returns
    (idx [rows, k] in position order, reach [rows, k], candidate mask or None).
    Pure torch so the NPU eager path and the CANNON device leg share it.

    ``pre_masked`` says the caller already applied the same ``j >= lens`` mask
    -- what ``DeepseekV41Indexer.scores_masked`` guarantees. Re-masking would be
    a no-op on the values but another full-width [rows, n] read-modify-write,
    which is exactly the pass that fusion removed.
    """
    if not pre_masked:
        j = torch.arange(s.shape[-1], device=s.device)
        s = s.masked_fill(j[None, :] >= lens[:, None], -torch.inf)
    masks = None
    if candidate_blocks:
        masks = select_candidate_blocks(
            s,
            lens[:, None],
            topk_blocks=candidate_blocks,
            block_size=candidate_block_size,
        )
    elif consume is not None:
        s = s.masked_fill(~consume, -torch.inf)
    k = min(topk, s.shape[-1])
    idx = s.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
    reach = idx < lens[:, None]
    return idx, reach, masks


def rope_tail(
    x: torch.Tensor, freqs: torch.Tensor, rope_dim: int, inverse: bool = False
) -> torch.Tensor:
    """Rotate the last rope_dim features of x [T, ..., D] with complex freqs [T, rope_dim // 2]."""
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    tc = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs.conj() if inverse else freqs
    f = f.view(x.shape[0], *([1] * (x.ndim - 2)), rope_dim // 2)
    rotated = torch.view_as_real(tc * f).flatten(-2).to(x.dtype)
    return torch.cat([head, rotated], dim=-1)


def pair_partners_decode(
    kv: torch.Tensor,
    score: torch.Tensor,
    odd: torch.Tensor,
    req: torch.Tensor,
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ratio-2 pairing for decode, one token per request: an odd position reads
    its partner from the per-request state, an even one parks itself there.
    `req` must be unique per row (padded rows go to a spare row). A partner is
    returned for every row; only odd rows complete a group."""
    partner_kv = state_kv[req]
    partner_score = state_score[req]
    keep = odd.unsqueeze(-1)
    state_kv[req] = torch.where(keep, partner_kv, kv)
    state_score[req] = torch.where(keep, partner_score, score)
    return partner_kv, partner_score


class DeepseekV41Compressor(nn.Module):
    """Pools compress_ratio consecutive tokens into one pre-RoPE KV latent.
    Ratio 1 is a plain bf16 projection; ratio 2 gates two tokens with a softmax
    over their fp32 scores.

    The reference promotes the ratio-2 wkv/wgate to fp32 and upcasts the bf16
    activation so the whole pooling chain runs in fp32. Both casts are exact
    (the checkpoint stores bf16), so a bf16 x bf16 GEMM with fp32 accumulation
    and fp32 output sums the same products; only the reduction order differs,
    at ~1e-5 relative, two orders of magnitude under the bf16 rounding the
    latent goes through in finish(). The weights therefore stay bf16 here and
    the softmax pooling still sees fp32 inputs."""

    def __init__(
        self, hidden_size: int, head_dim: int, compress_ratio: int, eps: float
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.norm = RMSNorm(head_dim, eps)
        self.wkv = nn.Linear(hidden_size, head_dim, bias=False, dtype=torch.bfloat16)
        if compress_ratio > 1:
            self.wgate = nn.Linear(
                hidden_size, head_dim, bias=False, dtype=torch.bfloat16
            )

    def project(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.compress_ratio == 1:
            return self.wkv(x), None
        # Two GEMMs rather than one fused [2D, K] projection: the decode epilogue
        # kernel (pair_pool_decode) reads kv and score as contiguous [n, D] fp32
        # rows, which column slices of a fused output are not.
        kv = linear_bf16_fp32(x, self.wkv.weight)
        score = linear_bf16_fp32(x, self.wgate.weight)
        return kv, score

    def finish(self, kv: torch.Tensor) -> torch.Tensor:
        return self.norm(kv.to(torch.bfloat16))

    @staticmethod
    def pool_pairs(kv2: torch.Tensor, score2: torch.Tensor) -> torch.Tensor:
        """kv2, score2 [n, 2, D] fp32 -> [n, D]"""
        return (kv2 * score2.softmax(dim=1)).sum(dim=1)


class DeepseekV41Indexer(nn.Module):
    """Scores compressed positions with a small fp4 side attention. Only a
    kv_source layer owns index keys; the other index sources read the source's.

    The projections are replicated across TP, as in the c4 indexer: every rank
    scores with all heads, so the decode kernel path needs no cross-rank
    reduction and every rank selects the same top-k."""

    def __init__(
        self,
        config,
        layer_id: int,
        head_dim: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.n_local_heads = self.n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.owns_k = layer_id in config.kv_source_layers
        self.is_candidate_source = layer_id == config.candidate_source_layer
        self.uses_candidates = 0 <= config.candidate_source_layer < layer_id
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.n_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )
        if self.owns_k:
            self.wk = nn.Linear(
                head_dim, self.index_head_dim, bias=False, dtype=torch.bfloat16
            )
            self.k_norm = RMSNorm(self.index_head_dim, config.rms_norm_eps)

    def index_keys(self, latent: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Pre-RoPE latents [n, D] -> fp4-rounded index keys [n, index_head_dim]."""
        k = self.k_norm(self.wk(latent))
        return _rope_fq4(k, freqs, self.rope_head_dim)

    def queries(self, q_lora: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        q, _ = self.wq_b(q_lora)
        q = q.view(q.shape[0], self.n_local_heads, self.index_head_dim)
        return _rope_fq4(q, freqs, self.rope_head_dim)

    def head_weights(self, x: torch.Tensor) -> torch.Tensor:
        w, _ = self.weights_proj(x)
        return w * (self.softmax_scale * self.n_heads**-0.5)

    def scores(
        self, q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """q [t, H, d], k [n, d], weights [t, H] -> [t, n], summed over all heads;
        bf16 up to the reduction, as the reference does."""
        s = torch.einsum("bhd,nd->bhn", q, k)
        s = (s.relu() * weights.unsqueeze(-1)).sum(dim=1)
        return s.float()

    def scores_masked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        lens: torch.Tensor,
    ) -> torch.Tensor:
        """``scores`` with the visible-length mask the callers always apply next.

        q [t, H, d], k [n, d], weights [t, H], lens [t] -> [t, n] fp32 with
        every compressed position at or past a row's length set to -inf.

        Folding the mask in is what lets the NPU path replace five full-width
        passes with one kernel: relu, the weight multiply and the head
        reduction each walk the whole [t, H, n] logit tensor, and .float() plus
        masked_fill walk the [t, n] result. The scoring einsum deliberately
        stays here in torch -- it is the only Cube-eligible part, and a triton
        tl.dot would drop it onto the Vector unit (see
        kernels/ops/attention/dsv4/dsv41_index_score_reduce.py).
        """
        s = torch.einsum("bhd,nd->bhn", q, k)
        if envs.SGLANG_OPT_USE_DSV41_TRITON_INDEX_SCORE.get():
            from sglang.kernels.ops.attention.dsv4.dsv41_index_score_reduce import (
                index_score_reduce,
                index_score_reduce_available,
            )

            if index_score_reduce_available():
                return index_score_reduce(s, weights, lens)
        s = (s.relu() * weights.unsqueeze(-1)).sum(dim=1).float()
        j = torch.arange(s.shape[-1], device=s.device)
        return s.masked_fill(j[None, :] >= lens[:, None], -torch.inf)
