"""forward_pre_o_proj on Kimi-K3's gated MLA attention (port of sgl-project/sglang#45202):
the TP-local core output o_proj takes, so the AMD decode monokernel can run o_proj and
its attention-TP reduction. KDA's counterpart splits KimiK3DeltaAttention.forward at
o_proj; MLA calls o_proj deep inside the DeepseekV2AttentionMLA cores, so its gated
o_proj wrapper hands the input back instead."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.utils.common import BumpAllocator


class KimiK3MLAPreOProjMixin:
    """forward_pre_o_proj for KimiK3MLAAttention, whose gated o_proj wrapper returns
    _take_pre_o_proj(x) while _return_pre_o_proj is set."""

    _return_pre_o_proj = False
    _pre_o_proj_out: Optional[torch.Tensor] = None

    def _take_pre_o_proj(self, x) -> tuple[torch.Tensor, None]:
        """o_proj's input in place of its GEMM and attention-TP reduction; the MLA
        cores only unpack a 2-tuple from it."""
        if isinstance(x, tuple):
            raise RuntimeError(
                f"forward_pre_o_proj: layer {self.layer_id} has a "
                "quantized o_proj input, so there is no gated bf16 "
                "output to hand back"
            )
        if self._pre_o_proj_out is not None:
            raise RuntimeError(
                f"forward_pre_o_proj: layer {self.layer_id} reached "
                "o_proj twice in one attention forward"
            )
        self._pre_o_proj_out = x
        return x, None

    def forward_pre_o_proj(
        self,
        *,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        **kwargs,
    ) -> torch.Tensor:
        """The gated TP-local core output o_proj takes; a monokernel runs o_proj."""
        assert self.use_output_gate, "forward_pre_o_proj needs the gated o_proj wrapper"
        assert not self._return_pre_o_proj, "forward_pre_o_proj is not re-entrant"
        self._return_pre_o_proj, self._pre_o_proj_out = True, None
        try:
            self.forward(
                positions, hidden_states, forward_batch, zero_allocator, **kwargs
            )
            out = self._pre_o_proj_out
        finally:
            self._return_pre_o_proj, self._pre_o_proj_out = False, None
        if out is None:
            raise RuntimeError("MLA forward returned without reaching o_proj")
        return out
