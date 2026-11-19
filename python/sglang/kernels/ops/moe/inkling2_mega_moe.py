"""Inkling2 routed experts on DeepGEMM MegaMoE (sgl-deep-gemm >= 0.2.9).

Coverage: SM100 / SM103, NVFP4 (packed `uint8`) or MXFP8 (`float8_e4m3fn`) expert weights, SiTU with
caps (4, 25), experts split over the EP `group`. Every rank passes the same tokens, at most
`max_num_tokens_per_rank * group.size()`, and gets the output of all experts for all of them.

Reads the same tensors as `Inkling2FlashInferMoE` (`weight_layout="up_gate_block16"`): w13 rows from
`up_gate_block16_rows_`, w2 rows plain, block scales from `swizzle_blockscale`.
"""

import deep_gemm
import torch
from deep_gemm.mega import SymmBuffer

_MMA_TYPES = ("nvfp4xnvfp4", "fp8xfp8")
# NVFP4 L2 activations take a static scale sized for the SiTU bound |act| <= 4 * 25.
_SITU_BOUND = 4.0 * 25.0
_BUFFERS = {}


def mega_moe_buffer(
    group, num_experts, max_num_tokens_per_rank, top_k, hidden, intermediate, mma_type
) -> SymmBuffer:
    """Symmetric buffer for one layer shape. NVFP4 and MXFP8 layers of a shape share one allocation,
    sized for the larger MMA type, since layers run one at a time."""
    shape = (num_experts, max_num_tokens_per_rank, top_k, hidden, intermediate)
    key = (id(group), *shape)
    if key not in _BUFFERS:
        tokens = -(
            -max_num_tokens_per_rank // deep_gemm._C.get_token_alignment_for_mega_moe()
        )
        tokens *= deep_gemm._C.get_token_alignment_for_mega_moe()
        largest = max(
            _MMA_TYPES,
            key=lambda m: deep_gemm._C.get_symm_buffer_size_for_mega_moe(
                group.size(),
                num_experts,
                tokens,
                top_k,
                hidden,
                intermediate,
                m,
                "situ",
                0,
            ),
        )
        _BUFFERS[key] = SymmBuffer(
            group, *shape, mma_type=largest, activation="situ", gather_output=True
        )
    if (key, mma_type) not in _BUFFERS:
        _BUFFERS[key, mma_type] = SymmBuffer(
            group,
            *shape,
            mma_type=mma_type,
            activation="situ",
            base=_BUFFERS[key],
            gather_output=True,
        )
    return _BUFFERS[key, mma_type]


class Inkling2MegaMoE:
    def __init__(
        self,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        *,
        group,
        num_experts: int,
        top_k: int,
        max_num_tokens_per_rank: int,
        max_topk_weight: float = 1.0,
        w13_alpha: torch.Tensor | None = None,
        w2_alpha: torch.Tensor | None = None,
    ):
        """`w13` `[E_local, 2 * I, K]`, `w2` `[E_local, H, I]` (packed NVFP4 halves K), scales as
        above. `w13_alpha` / `w2_alpha` `[E_local]`: NVFP4 per-expert global weight scales (required
        for NVFP4, ignored for MXFP8). `max_topk_weight` bounds `topk_weights`, which scale the
        L2 activations."""
        self.nvfp4 = w13.dtype == torch.uint8
        assert self.nvfp4 or w13.dtype == torch.float8_e4m3fn
        self.mma_type = _MMA_TYPES[0] if self.nvfp4 else _MMA_TYPES[1]
        self.group_size = 16 if self.nvfp4 else 32
        self.rank, self.num_ranks = group.rank(), group.size()
        self.buffer = mega_moe_buffer(
            group,
            num_experts,
            max_num_tokens_per_rank,
            top_k,
            w2.shape[1],
            w13.shape[1] // 2,
            self.mma_type,
        )
        weight = (lambda w: w.view(torch.int8)) if self.nvfp4 else (lambda w: w)
        self.l1 = (weight(w13), w13_scale.view(torch.int32))
        self.l2 = (weight(w2), w2_scale.view(torch.int32))
        self.kwargs = dict(
            recipe=(1, 1, self.group_size),
            activation="situ",
            fast_math=True,
            weight_layout="up_gate_block16",
            broadcast_y=True,
        )
        if self.nvfp4:
            assert w13_alpha is not None and w2_alpha is not None
            act_scale = 6.0 * 448.0 / (_SITU_BOUND * max_topk_weight)
            w13_alpha, w2_alpha = w13_alpha.float(), w2_alpha.float()
            self.kwargs.update(
                use_x_scales=True,
                l1_alphas=torch.stack([w13_alpha, w13_alpha], dim=1).contiguous(),
                l2_alphas=(w2_alpha / act_scale).contiguous(),
                l2_act_scales=torch.full_like(w2_alpha, act_scale),
            )

    def __call__(
        self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        """`x` `[T, H]` bf16, `topk_weights` / `topk_ids` `[T, top_k]` (global expert ids), the
        same on every rank. Returns `[T, H]` bf16, a view into the symmetric buffer that the next
        call on a layer of the same shape overwrites."""
        tokens = x.shape[0]
        per_rank = -(-tokens // self.num_ranks)
        start = min(self.rank * per_rank, tokens)
        local = min(tokens - start, per_rank)
        buf = self.buffer
        deep_gemm.mega_moe_pre_dispatch(
            x[start : start + local],
            topk_ids[start : start + local].int(),
            topk_weights[start : start + local].float(),
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=local,
            group_size=self.group_size,
            mma_type=self.mma_type,
            buf_x_scales=buf.x_scales if self.nvfp4 else None,
        )
        if local < per_rank:
            buf.topk_idx[local:per_rank].fill_(-1)
        y = buf.gathered_y[: per_rank * self.num_ranks]
        deep_gemm.fp8_fp4_mega_moe(y, self.l1, self.l2, buf, **self.kwargs)
        return y[:tokens]
