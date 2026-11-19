"""Plain-torch oracle for the Apple MPS Gated DeltaNet (GDN) kernels.

The math mirrors the model definition, not the kernels: fp32 gating
(``g = exp(-exp(A_log) * softplus(a + dt_bias))``, softplus threshold 20),
``beta = sigmoid(b)``, per-head L2 norm ``x / sqrt(sum(x^2) + 1e-6)`` with q
scaled by ``Dk^-0.5``, and the delta-rule recurrence over fp32 state. Cache
indices and query start locations are int32; initial-state flags are bool. A
negative cache index is a padding row whose output is zero and whose slot is
untouched.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _conv_window_silu(
    window: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """SiLU(depthwise causal conv) at the last position of each ``[N, C, K]`` window."""
    out = (window.float() * weight.float()).sum(-1)
    if bias is not None:
        out = out + bias.float()
    return F.silu(out)


def causal_conv1d_update_reference(
    x: torch.Tensor,
    conv_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """One decode step of the causal conv with SiLU; shifts slot windows in place."""
    y = torch.zeros_like(x)
    rows = (cache_indices >= 0).nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return y
    slots = cache_indices.index_select(0, rows).long()
    x_rows = x.index_select(0, rows)
    window = torch.cat(
        [
            conv_states.index_select(0, slots),
            x_rows.unsqueeze(-1),
        ],
        dim=-1,
    )
    y[rows] = _conv_window_silu(window, weight, bias).to(x.dtype)
    conv_states.index_copy_(0, slots, window[..., 1:].to(conv_states.dtype))
    return y


def causal_conv1d_varlen_reference(
    x: torch.Tensor,
    conv_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Varlen prefill conv with SiLU; saves each sequence's last K-1 inputs."""
    width = weight.shape[-1]
    state_len = width - 1
    y = torch.zeros_like(x)
    starts = query_start_loc.tolist()
    slots = cache_indices.tolist()
    initial = has_initial_state.tolist()
    for seq, slot in enumerate(slots):
        if slot < 0:
            continue
        start, end = starts[seq], starts[seq + 1]
        tokens = x[start:end].t()
        prev = (
            conv_states[slot]
            if initial[seq]
            else conv_states.new_zeros(tokens.shape[0], state_len)
        )
        padded = torch.cat([prev, tokens], dim=-1)
        if start != end:
            windows = padded.unfold(-1, width, 1)  # [C, L, K]
            y[start:end] = _conv_window_silu(windows.transpose(0, 1), weight, bias).to(
                x.dtype
            )
        conv_states[slot] = padded[:, -state_len:].to(conv_states.dtype)
    return y


def gated_delta_rule_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Gated delta rule over varlen sequences; writes the final states in place."""
    num_v_heads, head_v_dim = v.shape[1:]
    head_k_dim = k.shape[-1]
    group = num_v_heads // k.shape[1]

    def heads(t):
        return t.float().repeat_interleave(group, dim=1)

    qf = _l2_normalize(heads(q)) * head_k_dim**-0.5
    kf = _l2_normalize(heads(k))
    vf = v.float()
    g = torch.exp(-A_log.float().exp() * F.softplus(a.float() + dt_bias.float()))
    beta = torch.sigmoid(b.float())

    o = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
    starts = query_start_loc.tolist()
    slots = cache_indices.tolist()
    initial = has_initial_state.tolist()
    for seq, slot in enumerate(slots):
        if slot < 0:
            continue
        state = (
            ssm_states[slot].float()
            if initial[seq]
            else ssm_states.new_zeros(
                (num_v_heads, head_v_dim, head_k_dim), dtype=torch.float32
            )
        )
        for t in range(starts[seq], starts[seq + 1]):
            state = state * g[t, :, None, None]
            delta = (vf[t] - torch.einsum("hvk,hk->hv", state, kf[t])) * beta[
                t, :, None
            ]
            state = state + delta[:, :, None] * kf[t, :, None, :]
            o[t] = torch.einsum("hvk,hk->hv", state, qf[t])
        ssm_states[slot] = state.to(ssm_states.dtype)
    return o.to(q.dtype)
