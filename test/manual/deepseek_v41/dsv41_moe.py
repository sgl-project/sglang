import torch
import torch.nn.functional as F
from dsv41_linear import Linear
from torch import nn


class Gate(nn.Module):
    """The correction bias steers expert selection only; routing weights come from the
    unbiased scores. Tokens inside an image span select with bias_vl instead."""

    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        topk: int,
        score_func: str,
        gate_temp: float,
        norm_topk_prob: bool,
        route_scale: float,
        has_vl_bias: bool,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.topk = topk
        self.score_func = score_func
        self.gate_temp = gate_temp
        self.norm_topk_prob = norm_topk_prob
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, dim, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
        self.bias_vl = (
            nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
            if has_vl_bias
            else None
        )

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None):
        """x [n, dim] -> weights [n, topk] fp32, indices [n, topk]"""
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            # 1e-20 comes from training, independent of norm_eps.
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """SwiGLU FFN. swiglu_limit clamps the up branch on both sides and the gate branch
    from above, keeping fp8 / fp4 activations in range as in training."""

    def __init__(
        self, dim: int, inter_dim: int, dtype: torch.dtype, swiglu_limit: float
    ):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype)
        self.w2 = Linear(inter_dim, dim, dtype)
        self.w3 = Linear(dim, inter_dim, dtype)
        self.swiglu_limit = swiglu_limit

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through."""

    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        score_func: str,
        gate_temp: float,
        norm_topk_prob: bool,
        route_scale: float,
        swiglu_limit: float,
        expert_dtype: torch.dtype,
        shared_expert_dtype: torch.dtype,
        has_vl_bias: bool,
    ):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.gate = Gate(
            dim=dim,
            n_routed_experts=n_routed_experts,
            topk=n_activated_experts,
            score_func=score_func,
            gate_temp=gate_temp,
            norm_topk_prob=norm_topk_prob,
            route_scale=route_scale,
            has_vl_bias=has_vl_bias,
        )
        self.experts = nn.ModuleList(
            [
                Expert(dim, moe_inter_dim, expert_dtype, swiglu_limit)
                for _ in range(n_routed_experts)
            ]
        )
        self.shared_experts = Expert(
            dim, moe_inter_dim, shared_expert_dtype, swiglu_limit
        )

    def forward(
        self, x: torch.Tensor, image_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(
            x, None if image_mask is None else image_mask.flatten()
        )
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i, expert in enumerate(self.experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)
