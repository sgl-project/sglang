"""KDA recurrence of DFlash draft layers on the target's Triton KDA kernels.

A context-scanning (``linear_attn_config.context_state="scan"``) draft layer
keeps one recurrent state per request slot, in the target KDA pool layout:
``conv_state`` [slots, K - 1, conv_dim] holds the last pre-conv q/k/v rows and
``ssm_state`` [slots, H, V, K] fp32 the delta-rule state. Commits advance both
over verified target rows; proposal blocks start from them without writing
them, the way the target's verify forward reads its committed state. A
block-local (``"reset"``) layer starts every block from zero and owns no state.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_KDA_KERNEL = TritonKDAKernel()


def roll_conv_tails(
    tails: torch.Tensor, rows: torch.Tensor, commit_lens: torch.Tensor
) -> torch.Tensor:
    """Conv tails [bs, K - 1, dim] after appending the first commit_lens of rows [bs, L, dim]."""
    width = tails.shape[1]
    joined = torch.cat((tails, rows), dim=1)
    index = commit_lens.to(torch.int64)[:, None] + torch.arange(
        width, device=rows.device
    )
    return joined.gather(1, index[:, :, None].expand(-1, -1, joined.shape[-1]))


def committed_row_segments(
    slots: torch.Tensor, commit_lens: torch.Tensor, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Varlen segments over dense [bs, block_size] rows, without a host sync.

    Each request's first commit_lens rows advance its slot; its remaining rows
    form a segment on slot -1, which the KDA kernels neither read nor write.
    """
    num_reqs = commit_lens.shape[0]
    starts = torch.arange(num_reqs, device=commit_lens.device) * block_size
    bounds = torch.stack((starts, starts + commit_lens.to(starts.dtype)), dim=1)
    query_start_loc = F.pad(bounds.flatten(), (0, 1), value=num_reqs * block_size)
    state_indices = torch.stack(
        (slots.to(torch.int64), torch.full_like(slots, -1, dtype=torch.int64)), dim=1
    )
    return query_start_loc.to(torch.int32), state_indices.flatten()


class DFlashKDALinearAttention(RadixLinearAttention):
    """``KimiDeltaAttention.attn`` of a DFlash draft layer."""

    def __init__(self, *, scans_context: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scans_context = scans_context
        self.conv_state: Optional[torch.Tensor] = None
        self.ssm_state: Optional[torch.Tensor] = None

    def allocate_context_state(
        self, *, num_slots: int, dtype: torch.dtype, device: torch.device
    ) -> int:
        """Allocate zeroed per-slot state; returns its size in bytes."""
        self.conv_state = torch.zeros(
            (
                num_slots,
                self.conv_weights.shape[-1] - 1,
                self.q_dim + self.k_dim + self.v_dim,
            ),
            dtype=dtype,
            device=device,
        )
        self.ssm_state = torch.zeros(
            (num_slots, self.num_v_heads, self.head_v_dim, self.head_k_dim),
            dtype=torch.float32,
            device=device,
        )
        return self.conv_state.nbytes + self.ssm_state.nbytes

    def forward(
        self,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """One proposal block per request; [1, T, H, V]."""
        slots = forward_batch.req_pool_indices
        num_tokens = mixed_qkv.shape[0]
        num_blocks = slots.shape[0]
        block_size = num_tokens // num_blocks
        if self.scans_context:
            conv_state, ssm_state = self._context_state()
            # Gathered copy: the conv kernel rolls the window it is given.
            tails = conv_state[slots]
        else:
            ssm_state = None
            tails = mixed_qkv.new_zeros(
                (num_blocks, self.conv_weights.shape[-1] - 1, mixed_qkv.shape[-1])
            )
        q, k, v = self._convolve_blocks(
            mixed_qkv.view(num_blocks, block_size, -1), tails
        )
        return _KDA_KERNEL.target_verify(
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            q=q,
            k=k,
            v=v,
            a=a.reshape(num_tokens, -1),
            b=b.reshape(1, num_tokens, -1),
            # None starts every block from zero.
            ssm_states=ssm_state,
            cache_indices=slots,
            query_start_loc=torch.arange(
                0,
                num_tokens + 1,
                block_size,
                dtype=torch.int32,
                device=mixed_qkv.device,
            ),
            intermediate_states_buffer=None,
            intermediate_state_indices=None,
            cache_steps=block_size,
            retrieve_parent_token=None,
            lower_bound=self.lower_bound,
        )

    def commit_extend(
        self,
        *,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        slots: torch.Tensor,
        prefix_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        extend_lens_cpu: List[int],
    ) -> None:
        """Advance each slot over its packed extend rows, as the target's extend does."""
        conv_state, ssm_state = self._context_state()
        num_tokens = mixed_qkv.shape[0]
        # A request without a committed prefix is a new occupant of its slot.
        has_prefix = prefix_lens > 0
        ssm_state[slots] = torch.where(
            has_prefix.view(-1, 1, 1, 1), ssm_state[slots], 0.0
        )
        query_start_loc = F.pad(extend_lens.cumsum(0), (1, 0)).to(torch.int32)
        cache_indices = slots.to(torch.int32)
        qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            self.conv_weights,
            self.bias,
            conv_states=conv_state.transpose(-1, -2),
            query_start_loc=query_start_loc,
            seq_lens_cpu=extend_lens_cpu,
            cache_indices=cache_indices,
            has_initial_state=has_prefix,
            activation=self.activation,
        ).transpose(0, 1)
        q, k, v = self._split_heads(qkv)
        _KDA_KERNEL.extend(
            q,
            k,
            v,
            a.view(1, num_tokens, -1, self.head_k_dim),
            b.view(1, num_tokens, -1),
            ssm_states=ssm_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.lower_bound,
            beta_is_raw=True,
        )

    def commit_verified(
        self,
        *,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        slots: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Advance each slot over the first commit_lens of its dense [bs, block_size] rows."""
        conv_state, ssm_state = self._context_state()
        num_tokens = mixed_qkv.shape[0]
        block_size = num_tokens // commit_lens.shape[0]
        rows = mixed_qkv.view(commit_lens.shape[0], block_size, -1)
        tails = conv_state[slots]
        # Before the conv overwrites rows with its output.
        conv_state[slots] = roll_conv_tails(tails, rows, commit_lens)
        q, k, v = self._convolve_blocks(rows, tails)
        query_start_loc, state_indices = committed_row_segments(
            slots, commit_lens, block_size
        )
        _KDA_KERNEL.decode(
            q,
            k,
            v,
            a.reshape(num_tokens, -1),
            b.reshape(1, num_tokens, -1),
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            ssm_states=ssm_state,
            cache_indices=state_indices,
            query_start_loc=query_start_loc,
            lower_bound=self.lower_bound,
        )

    def _context_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.conv_state is None or self.ssm_state is None:
            raise RuntimeError(
                "DFLASH kda context state is not allocated; the DFLASH worker "
                "allocates it with the memory pools."
            )
        return self.conv_state, self.ssm_state

    def _convolve_blocks(
        self, rows: torch.Tensor, tails: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Causal conv of rows [bs, L, dim] after tails [bs, K - 1, dim]; both are overwritten."""
        qkv = causal_conv1d_update(
            rows.transpose(1, 2),
            tails.transpose(1, 2),
            self.conv_weights,
            self.bias,
            activation=self.activation,
        )
        return self._split_heads(qkv.transpose(1, 2).reshape(-1, qkv.shape[1]))

    def _split_heads(
        self, qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[T, conv_dim] -> q, k, v of [1, T, heads, head_dim]."""
        q, k, v = qkv.split([self.q_dim, self.k_dim, self.v_dim], dim=-1)
        return (
            q.unflatten(-1, (-1, self.head_q_dim)).unsqueeze(0),
            k.unflatten(-1, (-1, self.head_k_dim)).unsqueeze(0),
            v.unflatten(-1, (-1, self.head_v_dim)).unsqueeze(0),
        )
