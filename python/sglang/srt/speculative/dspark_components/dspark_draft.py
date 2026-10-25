from __future__ import annotations

import bisect
import functools
import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

import msgspec
import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    MarkovGreedyStep,
)
from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
from sglang.srt.configs.draft_parallel import DraftParallelSpec, DraftTPExecution
from sglang.srt.environ import envs
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    enable_num_token_non_padded,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_parallel import resolve_draft_parallel_layout
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    spec_scale_global_num_tokens,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils.common import is_pin_memory_available
from sglang.srt.utils.invariants import Bucket, Invariant, NotNaN, expect

if TYPE_CHECKING:
    from sglang.srt.models.dspark import BaseDSparkMarkovHead

logger = logging.getLogger(__name__)


# Draft step logits: NaN is a bug, but -inf is legitimate (masking).
_DRAFT_STEP_LOGITS = Invariant("dspark.draft.step_logits", Bucket.GUARD, NotNaN())


def _make_num_token_non_padded(
    num_tokens: int, device: str | torch.device
) -> Optional[torch.Tensor]:
    if not enable_num_token_non_padded():
        return None
    return torch.tensor(
        num_tokens,
        dtype=torch.int32,
        pin_memory=is_pin_memory_available(device),
    ).to(device, non_blocking=True)


class DraftBlockResult(msgspec.Struct, frozen=True):
    draft_tokens: torch.Tensor
    corrected_logits: Optional[torch.Tensor]
    greedy_mask: torch.Tensor
    temperatures: torch.Tensor
    # Drafted by argmax: the sampling accept uses a one-hot draft distribution.
    greedy_draft: bool


class DraftForwardResult(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    raw_hidden: torch.Tensor
    draft_hidden_3d: torch.Tensor
    can_run_graph: bool


class DraftProposal(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    draft_block: DraftBlockResult
    # [bs, gamma]; None when the draft has no confidence head.
    confidence: Optional[torch.Tensor] = None
    confidence_raw: Optional[torch.Tensor] = None
    folded: bool = False
    # [bs, gamma + 1] ``[anchor | draft tokens]`` from the draft graph tail; None
    # when the caller builds them.
    verify_ids: Optional[torch.Tensor] = None


def select_draft_hidden_without_anchor(
    hidden_states: torch.Tensor,
    *,
    bs: int,
    gamma: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_token_num = gamma + 1
    expected_rows = bs * query_token_num
    if hidden_states.shape[0] != expected_rows:
        raise RuntimeError(
            f"DSpark draft returned {hidden_states.shape[0]} hidden rows, "
            f"expected {expected_rows}."
        )
    hidden_by_query = hidden_states.view(bs, query_token_num, *hidden_states.shape[1:])
    selected = hidden_by_query[:, 1:].contiguous()
    return (
        selected.view(bs * gamma, *hidden_states.shape[1:]),
        selected.view(bs, gamma, -1),
    )


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


class DraftParams(NamedTuple):
    temperatures: torch.Tensor
    greedy_mask: torch.Tensor
    # The temperature each step samples at: 0 on argmax rows.
    step_temperatures: torch.Tensor


@functools.cache
def _zero(device: torch.device) -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32, device=device)


def resolve_draft_params(
    *,
    sampling_info,
    bs: int,
    device: Optional[torch.device] = None,
    out: Optional[DraftParams] = None,
) -> DraftParams:
    """Written into the ``[:bs]`` rows of ``out`` when given (the graph's
    static buffers), else into fresh tensors on ``device``."""
    if out is not None:
        device = out.temperatures.device
    else:
        out = DraftParams(
            temperatures=torch.empty(bs, dtype=torch.float32, device=device),
            greedy_mask=torch.empty(bs, dtype=torch.bool, device=device),
            step_temperatures=torch.empty(bs, dtype=torch.float32, device=device),
        )
    temperatures, greedy_mask, step_temperatures = (buf[:bs] for buf in out)
    if sampling_info is None:
        temperatures.fill_(1.0)
        greedy_mask.fill_(True)
        step_temperatures.zero_()
    else:
        torch.clamp(
            sampling_info.temperatures.view(-1)[:bs].to(torch.float32),
            min=1e-5,
            out=temperatures,
        )
        torch.le(sampling_info.top_ks.view(-1)[:bs], 1, out=greedy_mask)
        torch.where(greedy_mask, _zero(device), temperatures, out=step_temperatures)
    return DraftParams(temperatures, greedy_mask, step_temperatures)


# Arbitrary; how often the proposer logs its restage-skip count, in graph steps.
STAGED_PARAMS_LOG_EVERY = 500

# XORed into the server seed so the draft noise differs from other samplers'.
_DRAFT_NOISE_DOMAIN = 0x2D2A9C317E44B1F3


def draft_noise_key(random_seed: int) -> int:
    return (int(random_seed) ^ _DRAFT_NOISE_DOMAIN) & 0x7FFFFFFFFFFFFFFF


class DraftNoise(msgspec.Struct, frozen=True):
    """Philox key of the draft noise, and a GPU counter each block advances."""

    key: int
    offset: torch.Tensor


def step_logits_width(markov_head: BaseDSparkMarkovHead, base_width: int) -> int:
    # Kept step logits are gathered whole over a sharded vocab.
    return base_width if markov_head.vocab_shard is None else markov_head.vocab_size


def _draft_step(
    markov_head: BaseDSparkMarkovHead,
    prev_tokens: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    hidden_states: Optional[torch.Tensor],
    state: Optional[torch.Tensor],
    temperatures: torch.Tensor,
    noise: DraftNoise,
    step: int,
    logits_out: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if logits_out is None and len(prev_tokens) == 1:
        # One greedy row, no logits kept: one kernel does the GEMV, add and argmax.
        w2_weight = markov_head.greedy_w2_weight()
        # The kernel reads one w2 row per base-logit column.
        if w2_weight is not None and w2_weight.shape[0] == base_logits.shape[-1]:
            tokens = MarkovGreedyStep.execute(
                base_logits=base_logits,
                prev_embeds=markov_head.get_prev_embeddings(prev_tokens),
                w2_weight=w2_weight,
            )
            return tokens, state
    bias, state = markov_head.step_bias(
        prev_tokens, hidden_states=hidden_states, state=state
    )
    shard = markov_head.vocab_shard
    sharded = (
        {}
        if shard is None
        else dict(
            vocab_start=shard.start,
            gather=shard.gather.gather_stacked,
            gather_logits=shard.gather,
        )
    )
    tokens = MarkovSampleStep.execute(
        base_logits=base_logits,
        bias=bias,
        temperatures=temperatures,
        key=noise.key,
        offset=noise.offset,
        step=step,
        step_dtype=markov_head.step_logits_dtype(base_logits.dtype),
        logits_out=logits_out,
        # Rows sample at their temperatures only when the step logits are kept.
        has_sampled_rows=None if logits_out is not None else False,
        **sharded,
    )
    return tokens, state


def draft_markov_block(
    *,
    markov_head,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_hidden: Optional[torch.Tensor],
    step_temperatures: torch.Tensor,
    noise: DraftNoise,
    tp_sync: SpecTpSync,
    logits_out: Optional[torch.Tensor] = None,
    greedy_draft: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Returns the drafted tokens and the [bs, gamma, vocab] step logits,
    written to ``logits_out`` or allocated here; a greedy draft keeps none."""
    bs, gamma = base_logits.shape[:2]
    if not greedy_draft and logits_out is None:
        logits_out = torch.empty(
            (bs, gamma, step_logits_width(markov_head, base_logits.shape[-1])),
            dtype=markov_head.step_logits_dtype(base_logits.dtype),
            device=base_logits.device,
        )
    state = markov_head.init_state(draft_hidden)
    prev_tokens = anchor_tokens.long()
    draft_tokens = []
    for step in range(gamma):
        prev_tokens, state = _draft_step(
            markov_head,
            prev_tokens,
            base_logits[:, step],
            hidden_states=None if draft_hidden is None else draft_hidden[:, step],
            state=state,
            temperatures=step_temperatures,
            noise=noise,
            step=step,
            logits_out=None if greedy_draft else logits_out[:, step],
        )
        prev_tokens = tp_sync.sync(SpecTpSyncSite.DSPARK_DRAFT, prev_tokens)
        draft_tokens.append(prev_tokens)
    if greedy_draft:
        return torch.stack(draft_tokens, dim=1), None
    noise.offset.add_(1)
    return torch.stack(draft_tokens, dim=1), logits_out


def sample_draft_block(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_hidden: Optional[torch.Tensor],
    sampling_info,
    markov_head,
    noise: DraftNoise,
    device: torch.device,
    tp_sync: SpecTpSync,
    greedy_draft: bool,
) -> DraftBlockResult:
    params = resolve_draft_params(
        sampling_info=sampling_info, bs=base_logits.shape[0], device=device
    )
    draft_tokens, step_logits = draft_markov_block(
        markov_head=markov_head,
        base_logits=base_logits,
        anchor_tokens=anchor_tokens,
        draft_hidden=draft_hidden,
        step_temperatures=params.step_temperatures,
        noise=noise,
        tp_sync=tp_sync,
        greedy_draft=greedy_draft,
    )
    return DraftBlockResult(
        draft_tokens=draft_tokens,
        corrected_logits=(
            None if greedy_draft else expect(_DRAFT_STEP_LOGITS, step_logits)
        ),
        greedy_mask=params.greedy_mask,
        temperatures=params.temperatures,
        greedy_draft=greedy_draft,
    )


class DraftBlockProposer:
    _staged_params_key = None
    num_staged_params_skips = 0
    num_staged_params_calls = 0
    deferred_commit = None

    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        gamma: int,
        mask_token_id: int,
        draft_block_spec_info,
        tp_sync: SpecTpSync,
        random_seed: int,
        greedy_draft: bool,
        dp_moe_sync: bool = False,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.gamma = gamma
        self.sample_from_anchor = bool(draft_model.sample_from_anchor)
        self.query_token_num = self.gamma if self.sample_from_anchor else self.gamma + 1
        self._mask_token_id = mask_token_id
        self._draft_block_spec_info = draft_block_spec_info
        self._tp_sync = tp_sync
        self._greedy_draft = greedy_draft
        self._noise = DraftNoise(
            key=draft_noise_key(random_seed),
            offset=torch.zeros(
                1, dtype=torch.int64, device=self.draft_model_runner.device
            ),
        )
        self._draft_sampler = None
        self._dp_moe_sync = dp_moe_sync
        # A local logits subphase, separate from the MoE model's global layout.
        self._base_logits_layout = resolve_draft_parallel_layout(
            DraftParallelSpec(
                tp_execution=(
                    DraftTPExecution.SINGLE_GROUP
                    if dp_moe_sync
                    else DraftTPExecution.TARGET_LAYOUT
                )
            )
        )
        # SGLANG_DSPARK_DEFER_CONTEXT_KV's staged commit, set by the worker.
        self.deferred_commit = None
        # Persistent (bs, gamma) mask-token buffer: only column 0 (the bonus
        # token) changes per step, so avoid a fresh torch.full every decode.
        self._draft_block_ids_buf: Optional[torch.Tensor] = None
        # (sampler, bs, temperatures, top_ks) last staged: SamplingBatchInfo rebinds
        # these tensors when the batch changes, never edits them in place
        self._staged_params_key: Optional[tuple] = None
        self.num_staged_params_skips = 0
        self.num_staged_params_calls = 0
        self._num_token_non_padded = (
            torch.empty((1,), dtype=torch.int32, device=self.draft_model_runner.device)
            if enable_num_token_non_padded()
            else None
        )

    @property
    def noise(self) -> DraftNoise:
        return self._noise

    def attach_draft_sampler(self, draft_sampler) -> None:
        self._draft_sampler = draft_sampler

    def _stage_sampling_params(self, draft_sampler, *, bs: int, sampling_info) -> None:
        """The draft graph's sampling params, restaged only when the sampler, bs or
        the batch's temperature / top-k tensors changed since the last stage."""
        key = (
            draft_sampler,
            bs,
            None if sampling_info is None else sampling_info.temperatures,
            None if sampling_info is None else sampling_info.top_ks,
        )
        staged = self._staged_params_key
        self.num_staged_params_calls += 1
        skip = (
            staged is not None
            and staged[1] == bs
            and all(staged[i] is key[i] for i in (0, 2, 3))
        )
        self.num_staged_params_skips += skip
        if self.num_staged_params_calls % STAGED_PARAMS_LOG_EVERY == 0:
            logger.debug(
                "DSpark draft: sampling params restage skipped on %d of %d steps",
                self.num_staged_params_skips,
                self.num_staged_params_calls,
            )
        if skip:
            return
        draft_sampler.stage_sampling_params(bs=bs, sampling_info=sampling_info)
        self._staged_params_key = key

    def _base_logits_context(self):
        return draft_tp_context(self._base_logits_layout)

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
        anchor_prepared: bool = False,
    ) -> DraftProposal:
        embed_module = unwrap_lora_layer(
            self.draft_model.embed_tokens
            if not self.sample_from_anchor
            else target_model.get_input_embeddings()
        )
        draft_sampler = self._draft_sampler
        fwd = self._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device=device,
            embed_module=embed_module,
            draft_sampler=draft_sampler,
            sampling_info=sampling_info,
            anchor_prepared=anchor_prepared,
        )
        draft_block_ids = fwd.draft_block_ids

        confidence = confidence_raw = verify_ids = None
        folded = False
        if (
            envs.SGLANG_DSPARK_FOLDED_PROPOSAL.get()
            and draft_sampler is not None
            and fwd.can_run_graph
        ):
            folded = True
            draft_block = DraftBlockResult(
                draft_tokens=draft_sampler.out[: bs * self.gamma].view(bs, self.gamma),
                corrected_logits=(
                    None if self._greedy_draft else draft_sampler.step_logits[:bs]
                ),
                greedy_mask=draft_sampler.params.greedy_mask[:bs],
                temperatures=draft_sampler.params.temperatures[:bs],
                greedy_draft=self._greedy_draft,
            )
            if draft_sampler.confidence_out is not None:
                confidence = draft_sampler.confidence_out[:bs]
                confidence_raw = draft_sampler.confidence_raw_out[:bs]
            if draft_sampler.verify_ids is not None:
                verify_ids = draft_sampler.verify_ids[:bs]
        else:
            with self._base_logits_context():
                base_logits, confidence_features = self.draft_model.compute_base_logits(
                    fwd.raw_hidden
                )
                base_logits = base_logits.view(bs, self.gamma, -1)
            draft_block = sample_draft_block(
                base_logits=base_logits,
                anchor_tokens=draft_block_ids[:, 0],
                draft_hidden=fwd.draft_hidden_3d,
                sampling_info=sampling_info,
                markov_head=self.draft_model.markov_head,
                noise=self._noise,
                device=device,
                tp_sync=self._tp_sync,
                greedy_draft=self._greedy_draft,
            )
            if self.draft_model.confidence_head is not None:
                confidence, confidence_raw = self.draft_model.compute_confidence(
                    anchor_tokens=draft_block_ids[:, 0],
                    draft_tokens=draft_block.draft_tokens,
                    confidence_features=confidence_features,
                )
        proposal_block_ids = (
            draft_block_ids
            if self.sample_from_anchor
            else draft_block_ids[:, : self.gamma].contiguous()
        )
        return DraftProposal(
            draft_block_ids=proposal_block_ids,
            draft_block=draft_block,
            confidence=confidence,
            confidence_raw=confidence_raw,
            folded=folded,
            verify_ids=verify_ids,
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
        # Hand-built batch bypasses ForwardBatch.init_new: rebind the write
        # loc to the draft's kernel-facing ids (no-op on plain pools).
        self.draft_model_runner.kv_index_translator.bind_own_plan(idle_batch)
        self._fill_dp_moe_sync_metadata(idle_batch, batch)
        with torch.inference_mode():
            self.draft_model_runner.forward(idle_batch)

    def draft_block_ids_view(self, *, bs: int, device) -> torch.Tensor:
        """The ``[bs, query_token_num]`` draft input ids: mask tokens past column 0."""
        buf = self._draft_block_ids_buf
        if buf is None or buf.shape[0] < bs or buf.device != torch.device(device):
            buf = torch.full(
                (bs, self.query_token_num),
                int(self._mask_token_id),
                dtype=torch.long,
                device=device,
            )
            self._draft_block_ids_buf = buf
        return buf[:bs]

    def _run_forward(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        embed_module,
        draft_sampler=None,
        sampling_info=None,
        anchor_prepared: bool = False,
    ) -> DraftForwardResult:
        gamma = self.gamma
        query_token_num = self.query_token_num
        prefix_lens = batch.seq_lens
        positions_2d = verify_window.positions_2d
        verify_cache_loc_2d = verify_window.verify_cache_loc_2d

        draft_block_ids = self.draft_block_ids_view(bs=bs, device=prefix_lens.device)
        if not anchor_prepared:
            draft_block_ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))
        draft_positions = positions_2d[:, :query_token_num].reshape(-1)
        draft_cache_loc = verify_cache_loc_2d[:, :query_token_num].reshape(-1)

        draft_owns_embed = envs.SGLANG_DSPARK_EMBED_IN_GRAPH.get() and hasattr(
            self.draft_model, "forward_embed"
        )
        draft_input_embeds: Optional[torch.Tensor] = None
        if not draft_owns_embed:
            noise_embedding = embed_module(draft_block_ids)
            draft_input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        if batch.seq_lens_cpu is not None:
            draft_seq_lens_cpu = batch.seq_lens_cpu + query_token_num
            draft_seq_lens_sum = int(draft_seq_lens_cpu.sum())
        elif draft_input.nxt_kv_lens_cpu is not None:
            draft_seq_lens_cpu = draft_input.nxt_kv_lens_cpu
            draft_seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
        else:
            raise RuntimeError("DSpark decode expected batch.seq_lens_cpu, got None")

        draft_num_tokens = bs * query_token_num
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
            global_num_token_non_padded=_make_num_token_non_padded(
                draft_num_tokens, device
            ),
            global_num_token_non_padded_cpu=draft_num_tokens,
        )
        verify_window.kv_loc_plan.bind(
            draft_forward_batch,
            self.draft_model_runner.kv_index_translator,
            cols=slice(0, query_token_num),
        )
        self._fill_dp_moe_sync_metadata(draft_forward_batch, batch)
        graph_runner = self.draft_model_runner.decode_cuda_graph_runner
        runs_graph = graph_runner is not None and graph_runner.can_run_graph(
            draft_forward_batch
        )
        if draft_sampler is not None and runs_graph:
            self._stage_sampling_params(
                draft_sampler, bs=bs, sampling_info=sampling_info
            )
        commit = self.deferred_commit
        if commit is not None:
            commit.before_draft_forward(
                graph_bs=(
                    graph_runner.capture_bs[
                        bisect.bisect_left(graph_runner.capture_bs, bs)
                    ]
                    if runs_graph
                    else None
                )
            )
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(draft_forward_batch)
        if commit is not None:
            assert draft_out.can_run_graph == runs_graph
            commit.after_draft_forward(graph_ran=draft_out.can_run_graph)
        logits_output = draft_out.logits_output
        raw_hidden = logits_output.hidden_states
        if raw_hidden is None:
            raise RuntimeError("DSpark draft model returned no hidden states.")
        if self.sample_from_anchor:
            expected_rows = bs * gamma
            if raw_hidden.shape[0] != expected_rows:
                raise RuntimeError(
                    f"DSpark draft returned {raw_hidden.shape[0]} hidden rows, "
                    f"expected {expected_rows}."
                )
            model_hidden = raw_hidden
            draft_hidden_3d = raw_hidden.view(bs, gamma, -1)
        else:
            model_hidden, draft_hidden_3d = select_draft_hidden_without_anchor(
                raw_hidden,
                bs=bs,
                gamma=gamma,
            )
        return DraftForwardResult(
            draft_block_ids=draft_block_ids,
            raw_hidden=model_hidden,
            draft_hidden_3d=draft_hidden_3d,
            can_run_graph=draft_out.can_run_graph,
        )

    def _fill_dp_moe_sync_metadata(
        self, forward_batch: ForwardBatch, batch: ScheduleBatch
    ) -> None:
        # The dense DSpark draft still reuses the target batch's graph tier.
        # Set graph eligibility before the DP-MoE-only metadata early return.
        forward_batch.can_run_decode_cuda_graph = batch.can_run_decode_cuda_graph
        forward_batch.is_extend_in_batch = batch.is_extend_in_batch
        forward_batch.dp_spec_prefill_coordination_applied = (
            batch.dp_spec_prefill_coordination_applied
        )
        device = self.draft_model_runner.device
        num_tokens = forward_batch.input_ids.numel()
        if self._num_token_non_padded is not None:
            self._num_token_non_padded.fill_(num_tokens)
            forward_batch.num_token_non_padded = self._num_token_non_padded
        forward_batch.num_token_non_padded_cpu = num_tokens
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
        # Graph bucket selection uses the raw per-rank request counts.  Keep
        # them separate from global_num_tokens_cpu below, which is scaled into
        # draft-token units for DP/MoE synchronization.
        forward_batch.original_global_num_tokens_cpu = batch.global_num_tokens
        if batch.dp_spec_prefill_coordination_applied:
            gnt = batch.global_num_tokens
            gnt_logprob = batch.global_num_tokens_for_logprob
        else:
            gnt, gnt_logprob = spec_scale_global_num_tokens(
                self._draft_block_spec_info,
                batch.global_num_tokens,
                batch.global_num_tokens_for_logprob,
            )
        forward_batch.original_global_num_tokens_cpu = batch.global_num_tokens
        num_tokens = forward_batch.input_ids.numel()
        num_token_non_padded = _make_num_token_non_padded(num_tokens, device)
        if num_token_non_padded is not None:
            forward_batch.global_num_token_non_padded = num_token_non_padded
        forward_batch.global_num_token_non_padded_cpu = num_tokens
        forward_batch.global_num_tokens_cpu = gnt
        forward_batch.global_num_tokens_for_logprob_cpu = gnt_logprob
        pin_memory = is_pin_memory_available(device)
        forward_batch.global_num_tokens_gpu = torch.tensor(
            gnt, dtype=torch.int64, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
            gnt_logprob, dtype=torch.int64, pin_memory=pin_memory
        ).to(device, non_blocking=True)
