from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.kernel_plugins.nvidia.binding import (
    megakernel_agree,
    megakernel_draft_sampler,
)
from sglang.srt.layers.dp_attention import get_dp_tp_group

if TYPE_CHECKING:
    from sglang.srt.speculative.dspark_components.dspark_draft import DraftNoise


DraftResult = tuple[torch.Tensor, torch.Tensor | None]
StockDraft = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], DraftResult]
DraftStep = Callable[[torch.Tensor, torch.Tensor], DraftResult]
DraftSteps = Callable[[int], DraftStep | None]


class DsparkDraftStrategy:
    """Always-present DSpark proposal strategy with the stock implementation."""

    log_suffix = ""

    def run(
        self,
        *,
        model_hidden: torch.Tensor,
        sample_hidden: torch.Tensor,
        anchor_tokens: torch.Tensor,
        batch_size: int,
        stock: StockDraft,
    ) -> DraftResult:
        return stock(model_hidden, sample_hidden, anchor_tokens, batch_size)


class _SpecializedDsparkDraftStrategy(DsparkDraftStrategy):
    log_suffix = "; the NVIDIA megakernel may run it"

    def __init__(self, steps: DraftSteps) -> None:
        self._steps = steps

    def run(
        self,
        *,
        model_hidden: torch.Tensor,
        sample_hidden: torch.Tensor,
        anchor_tokens: torch.Tensor,
        batch_size: int,
        stock: StockDraft,
    ) -> DraftResult:
        step = self._steps(batch_size)
        if step is None:
            return super().run(
                model_hidden=model_hidden,
                sample_hidden=sample_hidden,
                anchor_tokens=anchor_tokens,
                batch_size=batch_size,
                stock=stock,
            )
        return step(model_hidden, anchor_tokens)


class BoundDsparkDraftStrategy(msgspec.Struct, frozen=True):
    strategy: DsparkDraftStrategy
    step_logits: torch.Tensor | None


_STOCK_DRAFT_STRATEGY = DsparkDraftStrategy()


def bind_dspark_draft_strategy(
    *,
    draft_model: torch.nn.Module,
    gamma: int,
    max_bs: int,
    greedy_draft: bool,
    noise: DraftNoise,
    draft_tokens: torch.Tensor,
    step_temperatures: torch.Tensor,
    allocate_step_logits: Callable[[], torch.Tensor],
) -> BoundDsparkDraftStrategy:
    """Resolve the DSpark proposal once and keep per-batch fallback in the facade."""

    factory = megakernel_draft_sampler(draft_model)
    if factory is None:
        return BoundDsparkDraftStrategy(_STOCK_DRAFT_STRATEGY, None)

    step_logits = None if greedy_draft else allocate_step_logits()
    steps = factory(
        gamma=gamma,
        max_bs=max_bs,
        greedy_draft=greedy_draft,
        noise=noise,
        draft_tokens=draft_tokens,
        step_logits=step_logits,
        step_temperatures=step_temperatures,
    )
    # A rank on the stock block waits in its per-step token sync (this group),
    # which a specialized step skips.
    served = tuple(bs for bs in range(1, max_bs + 1) if steps(bs) is not None)
    megakernel_agree(served, what="draft sampler batch sizes", group=get_dp_tp_group())
    return BoundDsparkDraftStrategy(_SpecializedDsparkDraftStrategy(steps), step_logits)
