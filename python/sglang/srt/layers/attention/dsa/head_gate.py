from __future__ import annotations

import torch

from sglang.srt.utils import is_cuda, is_hip
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()
_is_hip = is_hip()


if _is_cuda or _is_hip:

    def _scale_head_gate_graph_fake_impl(
        weights_raw: torch.Tensor,
        n_heads_inv_sqrt: float,
        softmax_scale: float,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            (weights_raw.shape[0], weights_raw.shape[1], q_scale.shape[-1]),
            dtype=torch.float32,
            device=weights_raw.device,
        )

    # In-graph head gate for the fused path: weights_proj is folded
    # into wk_weights_proj, so weights_raw is precomputed and there is no GEMM.
    @register_custom_op(fake_impl=_scale_head_gate_graph_fake_impl)
    def scale_head_gate_graph(
        weights_raw: torch.Tensor,
        n_heads_inv_sqrt: float,
        softmax_scale: float,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        weights = weights_raw * n_heads_inv_sqrt
        return weights.unsqueeze(-1) * q_scale * softmax_scale

    def _logits_head_gate_graph_fake_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        n_heads_inv_sqrt: float,
        softmax_scale: float,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            (x.shape[0], weight.shape[0], q_scale.shape[-1]),
            dtype=torch.float32,
            device=x.device,
        )

    # In-graph head gate for the NON-prefill path
    @register_custom_op(fake_impl=_logits_head_gate_graph_fake_impl)
    def logits_head_gate_graph(
        x: torch.Tensor,
        weight: torch.Tensor,
        n_heads_inv_sqrt: float,
        softmax_scale: float,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.mm(x, weight.t(), out_dtype=torch.float32)
        weights = out * n_heads_inv_sqrt
        weights = weights.unsqueeze(-1) * q_scale * softmax_scale
        return weights
