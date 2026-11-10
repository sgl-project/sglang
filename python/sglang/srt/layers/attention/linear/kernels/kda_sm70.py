# SPDX-License-Identifier: Apache-2.0
"""SM70 KDA backend. Fp16 recurrent kernel for decode, extend, and chain verify."""

from typing import Optional

import torch

from sglang.kernels.ops.attention.kda_sm70 import kda_sm70_recurrent
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


class Sm70KDAKernel(LinearAttnKernelBase):
    """Volta KDA. Safe-gate (``lower_bound``) and the softplus gate both live here."""

    supports_packed_decode: bool = False
    supports_safe_gate: bool = True
    supports_track_state_snapshot: bool = True
    # The bf16 fused chain-verify kernel is not this path.
    supports_fused_chain_verify: bool = False

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        return kda_sm70_recurrent(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            ssm_states,
            cache_indices,
            query_start_loc,
            lower_bound=lower_bound,
            raw_beta=True,
            commit_state=True,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ):
        if A_log is None or dt_bias is None:
            raise NotImplementedError("SM70 KDA extend requires A_log and dt_bias")
        out = kda_sm70_recurrent(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            ssm_states,
            cache_indices,
            query_start_loc,
            lower_bound=lower_bound,
            raw_beta=beta_is_raw,
            commit_state=True,
            track_state=kwargs.get("track_state"),
            track_chunk_idx=kwargs.get("track_chunk_idx"),
        )
        if return_intermediate_states:
            return out, None
        return out

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: Optional[torch.Tensor],
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        if retrieve_parent_token is not None:
            raise NotImplementedError("SM70 KDA verify does not support tree attention")
        if kwargs.get("cache_ring"):
            raise NotImplementedError("SM70 KDA verify does not write ReplaySSM rings")
        return kda_sm70_recurrent(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            ssm_states,
            cache_indices,
            query_start_loc,
            lower_bound=lower_bound,
            raw_beta=True,
            commit_state=False,
            intermediate_states=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
        )
