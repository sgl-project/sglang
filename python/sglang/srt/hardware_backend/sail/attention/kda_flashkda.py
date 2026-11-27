"""SAIL FlashKDA prefill: the PLA operator behind the FlashKDA interface."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.hardware_backend.sail.kernels import kda as sail_kda
from sglang.srt.layers.attention.linear.kernels.kda_flashkda import FlashKDAKernel


class SailFlashKDAKernel(FlashKDAKernel):
    """FlashKDA prefill backend served by PLA.

    The chunked prefill contract stays the same as the community kernel; only
    the launch is replaced and the sequence-length window (which exists to pick
    between two NV implementations) is dropped, since the operator covers the
    whole range.
    """

    @staticmethod
    def _should_fall_back(
        lower_bound: Optional[float],
        is_spec_decode: bool,
        query_start_loc: torch.Tensor,
        extend_seq_lens_cpu: Optional[list],
    ) -> bool:
        if lower_bound is None or is_spec_decode:
            return True
        return not sail_kda.pla_enabled()

    def _launch_fused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        out_buf: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: Optional[torch.Tensor],
        scale: float,
        lower_bound: Optional[float],
        initial_state: torch.Tensor,
        final_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> None:
        sail_kda.prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            out=out_buf,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            lower_bound=lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
        )
