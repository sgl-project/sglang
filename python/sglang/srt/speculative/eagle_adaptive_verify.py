from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildRaggedVerifyWindow,
    RaggedVerifyWindow,
)
from sglang.srt.speculative.acceptance_estimator import (
    AcceptancePrediction,
    OnlineAcceptanceEstimator,
)
from sglang.srt.speculative.adaptive_verify_scheduler import AdaptiveVerifyScheduler
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    gather_compact_to_strided,
)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.eagle_info import EagleVerifyInput


class RaggedVerifyForward(msgspec.Struct, frozen=True):
    window: RaggedVerifyWindow
    mrope_positions: Optional[torch.Tensor]
    layout: RaggedVerifyLayout
    num_draft_tokens: int

    def unpack_output(self, logits_output) -> None:
        logits_output.next_token_logits = self._to_strided(
            logits_output.next_token_logits
        )
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = self._to_strided(logits_output.hidden_states)

    def _to_strided(self, compact: torch.Tensor) -> torch.Tensor:
        return gather_compact_to_strided(
            compact,
            verify_lens=self.layout.verify_lens,
            verify_num_draft_tokens=self.num_draft_tokens,
        )


class AdaptiveVerifyStep(msgspec.Struct, frozen=True):
    prediction: AcceptancePrediction
    verify_lens: torch.Tensor
    layout: Optional[RaggedVerifyLayout]
    num_reqs: int


class EagleAdaptiveVerifier:
    """EAGLE adapter for adaptive verification: supplies the draft top-1 probs and
    the accept labels, and packs the verify window the scheduler picks."""

    def __init__(
        self,
        *,
        num_steps: int,
        num_draft_tokens: int,
        target_model_runner,
        device: torch.device,
        overlap: bool,
    ) -> None:
        assert num_draft_tokens == num_steps + 1
        self.num_draft_tokens = num_draft_tokens
        self.device = device
        self._model_runner = target_model_runner
        self.scheduler = AdaptiveVerifyScheduler(
            num_steps=num_steps,
            target_model_runner=target_model_runner,
            device=device,
            overlap=overlap,
        )
        self.estimator = OnlineAcceptanceEstimator(num_steps=num_steps, device=device)

    def applies_to(self, batch: ScheduleBatch) -> bool:
        if batch.forward_mode.is_idle() or batch.has_grammar:
            return False
        num_reqs = len(batch.seq_lens)
        if num_reqs == 0 or num_reqs > self.scheduler.max_num_reqs:
            return False
        if self._model_runner.model_config.model_is_mrope and any(
            mm is not None for mm in batch.multimodal_inputs
        ):
            return False
        return batch.req_pool_indices_cpu is not None

    def prepare_verify_budget(self, batch: ScheduleBatch, future_map: FutureMap):
        """Scheduler hook under overlap: plan this step's budget on the host."""
        draft_input = batch.spec_info
        if (
            draft_input is None
            or batch.forward_mode.is_extend()
            or batch.is_extend_in_batch
            or not self.applies_to(batch)
        ):
            self.scheduler.note_non_decode_step()
            return
        draft_input.verify_token_budget = self.scheduler.plan_budget(
            req_pool_indices_cpu=batch.req_pool_indices_cpu,
            resolved=future_map.resolve_confidence_cpu(batch, wait=True),
        )

    def begin_step(self):
        return self.scheduler.begin_step()

    def plan(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: EagleVerifyInput,
        draft_probs: torch.Tensor,
        verify_token_budget: Optional[int],
    ) -> AdaptiveVerifyStep:
        num_reqs = len(batch.seq_lens)
        features, predictions = self.estimator.predict(draft_probs)
        if not self.scheduler.overlap:
            verify_token_budget = self.scheduler.plan_budget_now(
                req_pool_indices_cpu=batch.req_pool_indices_cpu,
                confidence=predictions,
            )
        layout = self.scheduler.schedule_layout(
            confidence=predictions,
            verify_token_budget=verify_token_budget,
            num_reqs=num_reqs,
        )
        prediction = AcceptancePrediction(features=features, predictions=predictions)
        if layout is None:
            return AdaptiveVerifyStep(
                prediction=prediction,
                verify_lens=torch.full(
                    (num_reqs,),
                    self.num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                layout=None,
                num_reqs=num_reqs,
            )
        verify_input.ragged_verify_layout = layout
        verify_input.ragged_forward = self._build_forward(
            verify_input=verify_input, batch=batch, layout=layout
        )
        verify_input.retrieve_next_token = self._truncate_chain(
            retrieve_next_token=verify_input.retrieve_next_token,
            verify_lens=layout.verify_lens,
        )
        return AdaptiveVerifyStep(
            prediction=prediction,
            verify_lens=layout.verify_lens,
            layout=layout,
            num_reqs=num_reqs,
        )

    def finish_step(
        self,
        *,
        step: AdaptiveVerifyStep,
        accept_lens: torch.Tensor,
        timer_start,
    ) -> None:
        self.estimator.observe(
            features=step.prediction.features,
            predictions=step.prediction.predictions,
            num_correct_drafts=accept_lens - 1,
            verify_lens=step.verify_lens,
        )
        self.scheduler.end_step(
            start=timer_start, num_reqs=step.num_reqs, layout=step.layout
        )
        if self.scheduler.should_log():
            self.scheduler.log_state(estimator=self.estimator)

    def _build_forward(
        self,
        *,
        verify_input: EagleVerifyInput,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
    ) -> RaggedVerifyForward:
        bs = layout.bs
        strided = verify_input.draft_token.view(bs, self.num_draft_tokens)
        window = BuildRaggedVerifyWindow.execute(
            batch=batch,
            layout=layout,
            draft_block_ids=strided,
            draft_tokens=strided[:, 1:],
            bs=bs,
            device=self.device,
            verify_num_draft_tokens=self.num_draft_tokens,
            model_runner=self._model_runner,
        )
        valid = window.window_index >= 0
        mrope_positions = verify_input.prepared_mrope_positions
        if mrope_positions is not None:
            mrope_positions = torch.where(
                valid, mrope_positions[:, window.window_index.clamp(min=0)], 0
            )
        elif self._model_runner.model_config.model_is_mrope:
            mrope_positions = window.positions.unsqueeze(0).repeat(3, 1)
        return RaggedVerifyForward(
            window=window,
            mrope_positions=mrope_positions,
            layout=layout,
            num_draft_tokens=self.num_draft_tokens,
        )

    def _truncate_chain(
        self, *, retrieve_next_token: torch.Tensor, verify_lens: torch.Tensor
    ) -> torch.Tensor:
        steps = torch.arange(self.num_draft_tokens, device=self.device)
        last = (verify_lens.to(torch.int64) - 1).unsqueeze(1)
        return torch.where(
            steps.unsqueeze(0) >= last,
            torch.full_like(retrieve_next_token, -1),
            retrieve_next_token,
        )
