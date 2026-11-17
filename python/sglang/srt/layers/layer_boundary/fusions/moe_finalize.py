"""Deferred MoE finalize handoff shared by boundary providers."""

from __future__ import annotations

from typing import Callable

import torch

from sglang.srt.layers.layer_boundary.output import DeferredFinalize


class MoeDeferredFinalize(DeferredFinalize, frozen=True):
    """Unfinalized routed output plus the separately gated shared contribution.
    The next layer's fused finalize + AR + add + norm takes it; ``finish`` is
    the MoE's own unfused tail, for any other reader."""

    routed_output: torch.Tensor
    expert_weights: torch.Tensor
    permuted_indices: torch.Tensor
    gated_shared_output: torch.Tensor | None
    m: int
    finish: Callable[[], torch.Tensor]

    def complete(self) -> torch.Tensor:
        return self.finish()

    @classmethod
    def from_flashinfer(
        cls,
        deferred_output,
        *,
        gated_shared_output: torch.Tensor | None,
        m: int,
        reduce: Callable[[torch.Tensor], torch.Tensor],
    ) -> MoeDeferredFinalize:
        """``reduce`` is the MoE's own sum of its finalized output."""
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            finalize_flashinfer_trtllm_deferred_output,
        )

        top_k = int(deferred_output.top_k)
        return cls(
            routed_output=deferred_output.gemm2_out.view(
                -1, deferred_output.gemm2_out.shape[-1]
            ),
            expert_weights=deferred_output.expert_weights.view(-1, top_k)[:m],
            permuted_indices=deferred_output.expanded_idx_to_permuted_idx.view(
                -1, top_k
            )[:m],
            gated_shared_output=gated_shared_output,
            m=int(m),
            finish=lambda: reduce(
                finalize_flashinfer_trtllm_deferred_output(
                    deferred_output, gated_shared_output
                )
            ),
        )
