import torch
from dsv41_args import DeepseekV41Args
from dsv41_attention import Attention
from dsv41_engram import Engram, EngramLayout
from dsv41_hc import hc_mixes, hc_post, hc_pre
from dsv41_moe import MoE
from dsv41_norm import RMSNorm
from dsv41_shared import SharedAttentionRuntime
from torch import nn


def build_moe(args: DeepseekV41Args, layer_id: int) -> MoE:
    return MoE(
        dim=args.dim,
        moe_inter_dim=args.moe_inter_dim,
        n_routed_experts=args.n_routed_experts,
        n_activated_experts=args.n_activated_experts,
        score_func=args.score_func,
        gate_temp=args.gate_temp,
        norm_topk_prob=args.norm_topk_prob,
        route_scale=args.route_scale,
        swiglu_limit=args.swiglu_limit,
        expert_dtype=torch.float4_e2m1fn_x2 if args.expert_fp4 else torch.float8_e4m3fn,
        shared_expert_dtype=torch.float8_e4m3fn,
        has_vl_bias=args.vision_enabled,
    )


class Block(nn.Module):
    """One layer with an hc_mult-copy residual stream. The mixing coefficients a sublayer
    computes are used by the next one: attention consumes the previous FFN's pre-mix
    and the FFN consumes this attention's."""

    def __init__(
        self,
        args: DeepseekV41Args,
        layer_id: int,
        engram_layout: EngramLayout | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.attn = Attention(args, layer_id)
        self.ffn = build_moe(args, layer_id)
        self.engram = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(args, layer_id, engram_layout)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        mix_hc = (2 + args.hc_mult) * args.hc_mult
        hc_dim = args.hc_mult * args.dim
        fp32 = torch.float32
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=fp32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=fp32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=fp32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=fp32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=fp32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=fp32))

    def _mixes(self, x, hc_fn, hc_scale, hc_base):
        return hc_mixes(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps,
            norm_eps=self.norm_eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        pre_mix: torch.Tensor,
        image_mask: torch.Tensor | None,
        shared: SharedAttentionRuntime,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x [b, s, hc, d], pre_mix [b, s, hc] -> (x [b, s, hc, d], next pre_mix)"""
        residual = x
        attn_pre, attn_post, attn_comb = self._mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(hc_pre(x, pre_mix))
        x = self.attn(x, start_pos, shared)
        x = hc_post(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self._mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(hc_pre(x, attn_pre))
        x = self.ffn(x, image_mask)
        x = hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre
