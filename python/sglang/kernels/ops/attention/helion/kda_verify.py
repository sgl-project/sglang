"""Helion KDA verification with per-candidate state snapshots."""

from __future__ import annotations

import helion
import helion.language as hl
import torch


@helion.kernel(
    static_shapes=False,
    config=helion.Config(
        block_sizes=[8],
        num_warps=1,
        num_stages=1,
        indexing="pointer",
    ),
    ignore_warnings=[helion.exc.ProcessGroupNameNotFound],
)
def _kda_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    snapshots: torch.Tensor,
    snapshot_indices: torch.Tensor,
    parents: torch.Tensor | None,
    out: torch.Tensor,
    lower_bound: float,
    bounded_gate: hl.constexpr,
) -> torch.Tensor:
    N = cu_seqlens.size(0) - 1
    H = hl.specialize(q.size(1))
    HV = hl.specialize(v.size(1))
    K = hl.specialize(q.size(2))
    V = hl.specialize(v.size(2))
    block_v = hl.register_block_size(1, V)
    for tile_n, tile_h, tile_v in hl.tile([N, HV, V], block_size=[1, 1, block_v]):
        n, hv = tile_n.id, tile_h.id
        h = hv // (HV // H)
        keys = hl.arange(K)
        slot = state_indices[n].long()
        scratch = snapshot_indices[n].long()
        start = cu_seqlens[n].long()
        end = cu_seqlens[n + 1].long()
        state = hl.load(
            state_pool, [slot, hv, tile_v.index, keys], extra_mask=slot >= 0
        ).float()
        for token in range(start, end):
            step = token - start
            if parents is not None:
                use_parent = (step > 0) & (scratch >= 0)
                parent = parents[n, step].long()
                parent_state = hl.load(
                    snapshots,
                    [scratch, parent, hv, tile_v.index, keys],
                    extra_mask=use_parent,
                ).float()
                state = torch.where(use_parent, parent_state, state)
            raw_gate = a[token, hv, keys].float() + dt_bias[hv, keys].float()
            A = torch.exp(A_log[hv].float())
            if bounded_gate:
                log_decay = lower_bound * torch.sigmoid(A * raw_gate)
            else:
                softplus = torch.where(
                    raw_gate <= 20.0, torch.log(1.0 + torch.exp(raw_gate)), raw_gate
                )
                log_decay = -A * softplus
            beta = torch.sigmoid(b[token, hv].float())
            query = q[token, h, keys].float()
            key = k[token, h, keys].float()
            query = query / torch.sqrt((query * query).sum() + 1e-6)
            key = key / torch.sqrt((key * key).sum() + 1e-6)
            state = state * torch.exp(log_decay)[None, :]
            value = v[token, hv, tile_v].float()
            residual = (value - (state * key[None, :]).sum(-1)) * beta
            state = state + residual[:, None] * key[None, :]
            out[token, hv, tile_v] = (state * (query * K**-0.5)[None, :]).sum(-1)
            if scratch >= 0:
                snapshots[scratch, step, hv, tile_v.index, keys] = state
    return out


def helion_kda_verify(
    *,
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
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    cache_steps: int,
    retrieve_parent_token: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Verify packed sequences without mutating the committed state pool.

    Chain verification keeps an FP32 accumulator across candidates; tree
    verification reads each parent's snapshot, matching the reference contract.
    Negative state/snapshot indices are CUDA Graph padding slots.
    """
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("Helion KDA verify expects packed [1, T, H, K] inputs")
    tokens, heads, key_dim = q.shape[1:]
    value_heads, value_dim = v.shape[2:]
    if value_heads % heads or k.shape != q.shape:
        raise ValueError("Inconsistent Helion KDA verify query/key head layout")
    if intermediate_states_buffer is None or cache_steps <= 0:
        raise ValueError("Helion KDA verify requires per-candidate state snapshots")
    if intermediate_states_buffer.shape[1] < cache_steps:
        raise ValueError(
            "Helion KDA verify snapshot capacity is smaller than cache_steps"
        )
    out = torch.zeros_like(v)
    _kda_verify(
        q[0],
        k[0],
        v[0],
        a.reshape(tokens, value_heads, key_dim),
        b.reshape(tokens, value_heads),
        A_log.reshape(-1),
        dt_bias.reshape(value_heads, key_dim),
        ssm_states,
        cache_indices,
        query_start_loc,
        intermediate_states_buffer,
        intermediate_state_indices,
        retrieve_parent_token,
        out[0],
        0.0 if lower_bound is None else lower_bound,
        lower_bound is not None,
    )
    return out
