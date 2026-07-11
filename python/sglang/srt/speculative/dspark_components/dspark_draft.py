from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.dspark_components.kernels.sample_step_tokens import (
    SampleStepTokens,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context


class DraftBlockResult(msgspec.Struct, frozen=True):
    draft_tokens: torch.Tensor
    corrected_logits: Optional[torch.Tensor]
    greedy_mask: torch.Tensor
    temperatures: torch.Tensor


class DraftForwardResult(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    raw_hidden: torch.Tensor
    draft_hidden_3d: torch.Tensor
    can_run_graph: bool


class DraftProposal(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    draft_block: DraftBlockResult
    draft_hidden: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor] = None
    confidence_tap: Optional[torch.Tensor] = None
    folded: bool = False


def greedy_step_sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
    del step_idx
    return torch.argmax(step_logits, dim=-1)


class DsparkDraftSampler:

    def __init__(self, *, model, gamma, max_bs, device, confidence_fn=None, out=None):
        self.model = model
        self.markov_head = model.markov_head
        self.gamma = int(gamma)
        if out is not None:
            assert out.shape == (int(max_bs) * self.gamma,) and out.dtype == torch.int64
            self.out = out
        else:
            self.out = torch.empty(
                (int(max_bs) * self.gamma,), dtype=torch.int64, device=device
            )
        self.confidence_fn = confidence_fn
        self.confidence_out = (
            torch.empty((int(max_bs), self.gamma), dtype=torch.float32, device=device)
            if confidence_fn is not None
            else None
        )

    def __call__(self, hidden_states, input_ids):
        bs = hidden_states.shape[0] // self.gamma
        base_logits, confidence_tap = self.model.compute_base_logits(hidden_states)
        base_logits = base_logits.view(bs, self.gamma, -1)
        anchor = input_ids.view(bs, self.gamma)[:, 0]
        draft_tokens, _ = self.markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            hidden_states=hidden_states.view(bs, self.gamma, -1),
            sampler=greedy_step_sampler,
        )
        self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        if self.confidence_out is not None:
            confidence = self.confidence_fn(
                draft_hidden=hidden_states.view(bs, self.gamma, -1),
                anchor_tokens=anchor,
                draft_tokens=draft_tokens,
                confidence_tap=confidence_tap,
            )
            self.confidence_out[:bs].copy_(confidence)


def make_next_draft_input(
    *,
    bonus_tokens: torch.Tensor,
    new_seq_lens: torch.Tensor,
) -> DFlashDraftInputV2:
    return make_draft_input_v2(bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens)


def resolve_greedy_mask(
    *,
    bs: int,
    sampling_info,
    device: torch.device,
) -> torch.Tensor:
    if sampling_info is None:
        return torch.ones(bs, dtype=torch.bool, device=device)
    return (sampling_info.top_ks <= 1).view(-1)


def sample_draft_block(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_hidden: torch.Tensor,
    sampling_info,
    markov_head,
    device: torch.device,
) -> DraftBlockResult:
    bs = base_logits.shape[0]
    greedy_mask = resolve_greedy_mask(bs=bs, sampling_info=sampling_info, device=device)
    any_sampling = sampling_info is not None and not sampling_info.is_all_greedy
    fast_sampling = envs.SGLANG_DSPARK_FAST_SAMPLING.get()

    if sampling_info is None:
        temperatures = torch.ones(bs, dtype=torch.float32, device=device)
    else:
        temperatures = (
            sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
        )

    if not any_sampling:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            return torch.argmax(step_logits, dim=-1)

    else:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            if fast_sampling:
                exp_noise = torch.empty(
                    step_logits.shape, dtype=torch.float32, device=step_logits.device
                ).exponential_(1)
                return SampleStepTokens.execute(
                    step_logits=step_logits,
                    temperatures=temperatures,
                    greedy_mask=greedy_mask,
                    exp_noise=exp_noise,
                )
            else:
                probs = torch.softmax(
                    step_logits.float() / temperatures[:, None], dim=-1
                )
                argmax_tokens = torch.argmax(step_logits, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return torch.where(greedy_mask, argmax_tokens, sampled_tokens)

    draft_tokens, corrected_logits = markov_head.sample_block(
        base_logits,
        first_prev_tokens=anchor_tokens,
        hidden_states=draft_hidden,
        sampler=sampler,
    )
    return DraftBlockResult(
        draft_tokens=draft_tokens,
        corrected_logits=corrected_logits,
        greedy_mask=greedy_mask,
        temperatures=temperatures,
    )


class DraftBlockProposer:
    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        gamma: int,
        mask_token_id: int,
        draft_block_spec_info,
        dp_moe_sync: bool = False,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.gamma = gamma
        self._mask_token_id = mask_token_id
        self._draft_block_spec_info = draft_block_spec_info
        self._draft_sampler = None
        self._dp_moe_sync = dp_moe_sync

    def attach_draft_sampler(self, draft_sampler) -> None:
        self._draft_sampler = draft_sampler

    def _base_logits_context(self):
        if self._dp_moe_sync:
            return draft_tp_context(get_parallel().attn_tp_group)
        return nullcontext()

    def propose(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        target_model,
        sampling_info,
    ) -> DraftProposal:
        embed_module = target_model.get_input_embeddings()
        fwd = self._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device=device,
            embed_module=embed_module,
        )
        draft_block_ids = fwd.draft_block_ids

        draft_sampler = self._draft_sampler
        all_greedy = sampling_info is None or sampling_info.is_all_greedy
        folded_confidence = None
        confidence_tap = None
        folded = False
        if draft_sampler is not None and fwd.can_run_graph and all_greedy:
            folded = True
            if sampling_info is None:
                temperatures = torch.ones(bs, dtype=torch.float32, device=device)
            else:
                temperatures = (
                    sampling_info.temperatures.view(-1)
                    .to(torch.float32)
                    .clamp_min(1e-5)
                )
            draft_block = DraftBlockResult(
                draft_tokens=draft_sampler.out[: bs * self.gamma].view(bs, self.gamma),
                corrected_logits=None,
                greedy_mask=resolve_greedy_mask(
                    bs=bs, sampling_info=sampling_info, device=device
                ),
                temperatures=temperatures,
            )
            if draft_sampler.confidence_out is not None:
                folded_confidence = draft_sampler.confidence_out[:bs]
        else:
            with self._base_logits_context():
                base_logits, confidence_tap = self.draft_model.compute_base_logits(
                    fwd.raw_hidden
                )
                base_logits = base_logits.view(bs, self.gamma, -1)
            draft_block = sample_draft_block(
                base_logits=base_logits,
                anchor_tokens=draft_block_ids[:, 0],
                draft_hidden=fwd.draft_hidden_3d,
                sampling_info=sampling_info,
                markov_head=self.draft_model.markov_head,
                device=device,
            )
        return DraftProposal(
            draft_block_ids=draft_block_ids,
            draft_block=draft_block,
            draft_hidden=fwd.draft_hidden_3d,
            confidence=folded_confidence,
            confidence_tap=confidence_tap,
            folded=folded,
        )

    def run_idle_participation(self, batch: ScheduleBatch) -> None:
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
        device = self.draft_model_runner.device
        empty_long = torch.empty((0,), dtype=torch.int64, device=device)
        idle_batch = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=empty_long,
            req_pool_indices=empty_long,
            seq_lens=empty_long,
            out_cache_loc=empty_long,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
            positions=empty_long,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._fill_dp_moe_sync_metadata(idle_batch, batch)
        with torch.inference_mode():
            self.draft_model_runner.forward(idle_batch)

    def _run_forward(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        embed_module,
    ) -> DraftForwardResult:
        gamma = self.gamma
        prefix_lens = batch.seq_lens
        positions_2d = verify_window.positions_2d
        verify_cache_loc_2d = verify_window.verify_cache_loc_2d

        draft_block_ids = torch.full(
            (bs, gamma), int(self._mask_token_id), dtype=torch.long, device=device
        )
        draft_block_ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))
        draft_positions = positions_2d[:, :gamma].reshape(-1)
        draft_cache_loc = verify_cache_loc_2d[:, :gamma].reshape(-1)

        draft_owns_embed = hasattr(self.draft_model, "forward_embed")
        draft_input_embeds: Optional[torch.Tensor] = None
        if not draft_owns_embed:
            noise_embedding = embed_module(draft_block_ids)
            draft_input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        if batch.seq_lens_cpu is not None:
            draft_seq_lens_cpu = batch.seq_lens_cpu + gamma
            draft_seq_lens_sum = int(draft_seq_lens_cpu.sum())
        elif draft_input.reserved_seq_lens_cpu is not None:
            draft_seq_lens_cpu = draft_input.reserved_seq_lens_cpu
            draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
        else:
            raise RuntimeError("DSpark decode expected batch.seq_lens_cpu, got None")

        draft_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=draft_block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            out_cache_loc=draft_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=draft_seq_lens_cpu,
            positions=draft_positions,
            input_embeds=draft_input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._fill_dp_moe_sync_metadata(draft_forward_batch, batch)
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(draft_forward_batch)
        logits_output = draft_out.logits_output
        raw_hidden = logits_output.hidden_states
        if raw_hidden is None:
            raise RuntimeError("DSpark draft model returned no hidden states.")
        draft_hidden_3d = raw_hidden.view(bs, gamma, -1)
        return DraftForwardResult(
            draft_block_ids=draft_block_ids,
            raw_hidden=raw_hidden,
            draft_hidden_3d=draft_hidden_3d,
            can_run_graph=draft_out.can_run_graph,
        )

    def _fill_dp_moe_sync_metadata(
        self, forward_batch: ForwardBatch, batch: ScheduleBatch
    ) -> None:
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
        gnt, gnt_logprob = (
            self._draft_block_spec_info.get_spec_adjusted_global_num_tokens(batch)
        )
        device = self.draft_model_runner.device
        forward_batch.global_num_tokens_cpu = gnt
        forward_batch.global_num_tokens_for_logprob_cpu = gnt_logprob
        forward_batch.global_num_tokens_gpu = torch.tensor(gnt, dtype=torch.int64).to(
            device, non_blocking=True
        )
        forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
            gnt_logprob, dtype=torch.int64
        ).to(device, non_blocking=True)
        forward_batch.can_run_dp_cuda_graph = batch.can_run_dp_cuda_graph
