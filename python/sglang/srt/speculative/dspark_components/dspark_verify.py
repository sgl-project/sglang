from __future__ import annotations

import logging
import random
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.embeddings.engram_hash import engram_commit_history
from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
    BlockwiseRejection,
    RejectionNoise,
)
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    AcceptGreedy,
    AcceptSampling,
    FinalizeAcceptLens,
    SelectMixedAccept,
    SoftmaxTemp,
    _row_argmax,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildCommitInjectLayout,
    BuildOutTokens,
    BuildRaggedVerifyWindow,
    RaggedVerifyWindow,
    ScatterCompactToStrided,
    build_ring_commit_inject_layout,
    scatter_compact_to_strided_into,
)
from sglang.kernels.ops.speculative.dspark.simulated_bonus import (
    simulated_bonus_partials_graph,
    simulated_bonus_sample,
    simulated_bonus_sample_graph,
)
from sglang.kernels.ops.speculative.dspark.static_accept_tail import static_accept_tail
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import (
    FutureRelayBuffers,
    offer_future_relay_buffers,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
from sglang.srt.speculative.dspark_components.dspark_deferred_commit import (
    DeferredContextCommit,
)
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    VerifyWindow,
    apply_logits_adjustments_strided,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_METHOD,
    sample_simulated_acc_len,
)
from sglang.srt.utils import is_npu
from sglang.srt.utils.invariants import Bucket, Invariant, NotNaN, expect

logger = logging.getLogger(__name__)

_is_npu = is_npu()

# Draft proposal probs feeding rejection sampling; the data layer is the
# in-kernel NaN-q guard in reject_sampling.py, so this is signal-only.
_VERIFY_DRAFT_PROBS = Invariant("dspark.verify.draft_probs", Bucket.GUARD, NotNaN())


def verify_logits_adjustments_are_noop(sampling_info) -> bool:
    if sampling_info is None:
        return True
    if sampling_info.has_custom_logit_processor:
        return False
    if getattr(sampling_info, "acc_linear_penalties", None) is not None:
        return False
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    if penalizer is not None and penalizer.is_required:
        return False
    if getattr(sampling_info, "grammar_mask", None) is not None:
        return False
    if getattr(sampling_info, "logit_bias", None) is not None:
        return False
    return True


class TargetVerifyResult(msgspec.Struct, frozen=True):
    logits_output: object
    can_run_cuda_graph: bool


def candidate_request_length_bound(
    reqs, pending_verify_tokens: int = 0
) -> Optional[int]:
    """Bound committed positions without reading asynchronous acceptance results.
    The overlap loop can hold one unprocessed result, so reserve its full width;
    the runner adds the current verify width. Aborted/embedding/multimodal requests
    return None: their visible token IDs may not track cache positions."""
    if not reqs:
        return None
    longest = 0
    for req in reqs:
        budget = req.sampling_params.max_new_tokens
        if (
            not isinstance(budget, int)
            or budget < 0
            or getattr(req, "to_finish", None) is not None
            or getattr(req, "input_embeds", None) is not None
            or getattr(req, "multimodal_inputs", None) is not None
        ):
            return None
        longest = max(longest, len(req.origin_input_ids) + budget)
    return longest + pending_verify_tokens


class TargetVerifyExecutor:
    _rejection_noise = None

    def __init__(
        self,
        *,
        target_worker,
        gamma: int,
        verify_num_draft_tokens: int,
        model_runner,
        kv_injector: TargetHiddenKvInjector,
        tp_sync: SpecTpSync,
        verify_epilogue=None,
        simulate_acc_len: float = 0.0,
        simulated_correct_len: Optional[torch.Tensor] = None,
        block_verification: bool = False,
        rejection_noise: Optional[RejectionNoise] = None,
    ) -> None:
        self.target_worker = target_worker
        # candidate_max_seq_len_upper_bound only feeds the V4.1 candidate graphs.
        self._target_is_dsv41 = (
            getattr(
                target_worker.model_runner.model_config.hf_text_config,
                "model_type",
                None,
            )
            == "deepseek_v41"
        )
        self.gamma = int(gamma)
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self.model_runner = model_runner
        self.kv_injector = kv_injector
        self._tp_sync = tp_sync
        self.verify_epilogue = verify_epilogue
        self._verify_backend_self_adds_seq_lens_cache: Optional[bool] = None
        self._simulate_acc_len = float(simulate_acc_len)
        self._block_verification = block_verification
        # Block-wise rejection draws (blockwise_rejection.py); None keeps the classic kernel.
        self._rejection_noise = rejection_noise
        self._simulated_correct_len = simulated_correct_len

    def accept_and_finalize(
        self,
        *,
        folded_accept: bool,
        bs: int,
        verify_ids_2d: torch.Tensor,
        target_logits: Optional[torch.Tensor],
        draft_block: DraftBlockResult,
        sampling_info,
        draft_input: DFlashDraftInputV2,
        layout: Optional[RaggedVerifyLayout],
        prefix_lens: torch.Tensor,
        draft_tokens: torch.Tensor,
        simulate_bonus_sampling_info=None,
    ) -> AcceptOuts:
        """Produce the per-request accept outcome after target verify.

        Folded path: the accept/finalize/out-token kernels already ran inside
        the target-verify cuda graph (DsparkVerifyEpilogue); read its buffers.
        Eager path: run them here. Both apply the SGLANG_SIMULATE_ACC_LEN
        override drawn by begin_simulated_step.
        """
        if folded_accept:
            self.verify_epilogue.note_static_verify(bs)
            return self.verify_epilogue.read_accept(bs)

        simulate = self._simulated_correct_len is not None
        # Simulated acceptance overwrites correct_len below, so a sampling accept
        # (draft/target softmax + rejection) would be computed only to be discarded.
        accept_sampling_info = None if simulate else sampling_info
        correct_len, bonus, cap_trim_lens = accept_draft_tokens(
            candidates=verify_ids_2d,
            target_logits=target_logits,
            draft_block=draft_block,
            sampling_info=accept_sampling_info,
            draft_input=draft_input,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            cutoff_layout=layout,
            fused_argmax=self._target_is_dsv41,
            block_verification=self._block_verification,
            rejection_noise=self._rejection_noise,
        )
        sync_site = (
            SpecTpSyncSite.DSPARK_ACCEPT_GREEDY
            if accept_sampling_info is None or accept_sampling_info.is_all_greedy
            else SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE
        )
        return build_accept_outs(
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_sync=self._tp_sync,
            sync_site=sync_site,
            simulated_correct_len=self._simulated_correct_len,
            target_logits=target_logits,
            simulate_bonus_sampling_info=simulate_bonus_sampling_info,
        )

    def begin_simulated_step(self) -> None:
        """Draw this step's SGLANG_SIMULATE_ACC_LEN correct length, before the
        target verify, so the folded accept in the graph applies it too."""
        if self._simulated_correct_len is None:
            return
        simulated_acc_len = sample_simulated_acc_len(
            self._simulate_acc_len, SIMULATE_ACC_METHOD, self.gamma + 1
        )
        self._simulated_correct_len.fill_(simulated_acc_len - 1)

    def run_idle_participation(
        self,
        *,
        batch: ScheduleBatch,
        idle_layout: Optional[RaggedVerifyLayout],
    ) -> None:
        """Run a dummy target-verify forward so an idle DP rank joins the
        token-keyed collective ops of the busy ranks' verify step."""
        device = self.model_runner.device
        if self.verify_epilogue is not None:
            self.verify_epilogue.begin_step(None, armed=False)
        num_dummy_tokens = (
            idle_layout.graph_num_tokens if idle_layout is not None else 0
        )
        verify_input = DFlashVerifyInput(
            draft_token=torch.zeros(
                (num_dummy_tokens,), dtype=torch.int64, device=device
            ),
            positions=torch.zeros(
                (num_dummy_tokens,), dtype=torch.int64, device=device
            ),
            draft_token_num=self.verify_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=idle_layout,
        )
        batch.out_cache_loc = torch.zeros(
            (num_dummy_tokens,), dtype=torch.int64, device=device
        )
        if idle_layout is not None:
            num_dummy_slots = int(idle_layout.verify_lens.numel())
            batch.seq_lens = torch.ones(
                (num_dummy_slots,), dtype=torch.int64, device=device
            )
            batch.req_pool_indices = torch.zeros(
                (num_dummy_slots,), dtype=torch.int64, device=device
            )
            batch.seq_lens_cpu = torch.ones((num_dummy_slots,), dtype=torch.int64)
            batch.seq_lens_sum = num_dummy_slots
            batch.forward_mode = ForwardMode.TARGET_VERIFY
        verify_input.live_seq_lens_cpu = batch.seq_lens_cpu
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True if not _is_npu else None,
        )

    def run_non_compact(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_ids_2d: torch.Tensor,
        verify_window: VerifyWindow,
        sampling_info,
    ) -> TargetVerifyResult:
        verify_w = self.verify_num_draft_tokens
        positions_2d = verify_window.positions_2d
        verify_cache_loc = verify_window.verify_cache_loc

        verify_input = DFlashVerifyInput(
            draft_token=verify_ids_2d.reshape(-1),
            positions=positions_2d.reshape(-1),
            draft_token_num=verify_w,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            live_seq_lens_cpu=batch.seq_lens_cpu,
            kv_loc_plan=verify_window.kv_loc_plan,
        )
        batch.out_cache_loc = verify_cache_loc
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if not self._verify_backend_self_adds_seq_lens():
            if seq_lens_cpu_backup is not None:
                batch.seq_lens_cpu = seq_lens_cpu_backup + verify_w
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            elif draft_input.nxt_kv_lens_cpu is not None:
                batch.seq_lens_cpu = draft_input.nxt_kv_lens_cpu
                batch.seq_lens_sum = int(draft_input.nxt_kv_lens_sum)

        result = self._forward_prepared_verify(
            batch=batch,
            verify_input=verify_input,
            seq_lens_cpu_backup=seq_lens_cpu_backup,
            seq_lens_sum_backup=seq_lens_sum_backup,
        )

        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=result.logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=verify_w,
            )

        return result

    def _forward_prepared_verify(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: DFlashVerifyInput,
        seq_lens_cpu_backup,
        seq_lens_sum_backup,
    ) -> TargetVerifyResult:
        if verify_input.live_seq_lens_cpu is None and self._target_is_dsv41:
            verify_input.candidate_max_seq_len_upper_bound = (
                candidate_request_length_bound(batch.reqs, self.verify_num_draft_tokens)
            )
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup

        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True if not _is_npu else None,
        )
        return TargetVerifyResult(
            logits_output=target_out.logits_output,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
        )

    def commit_hidden(
        self,
        *,
        batch: ScheduleBatch,
        layout: Optional[RaggedVerifyLayout],
        hidden_strided: Optional[torch.Tensor],
        verify_window: VerifyWindow,
        logits_output,
        commit_lens: torch.Tensor,
        bs: int,
        run_compact: bool,
    ) -> None:
        if run_compact:
            self.kv_injector.inject_ragged(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                commit_lens=commit_lens,
                bs=bs,
                kv_loc_plan=verify_window.kv_loc_plan,
            )
            return
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DSpark verify requires target hidden states, got None.")
        hidden = hidden.view(bs, self.verify_num_draft_tokens, -1)
        state_slot = None
        if self.kv_injector.swa_ring_size is not None:
            # A SWA ring needs the per-token draft req slot to address it
            # (state_slot * ring + pos % ring). Verify tokens are the latest in each
            # req so they always fall in the window; the commit gate (via commit_lens
            # + cache_loc_2d) drops rejected tokens, so no final_pos skip is needed.
            vlen = verify_window.verify_cache_loc_2d.shape[1]
            state_slot = (
                batch.req_pool_indices[:bs].view(-1, 1).expand(bs, vlen).reshape(-1)
            )
        cache_loc = self.kv_injector.ids_for(verify_window.kv_loc_plan)
        self.kv_injector.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc.view(verify_window.verify_cache_loc_2d.shape),
            positions=verify_window.positions_2d.reshape(-1),
            commit_lens=commit_lens,
            state_slot=state_slot,
        )

    def _run_ragged(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        ragged_window: RaggedVerifyWindow,
        sampling_info,
        kv_loc_plan,
    ) -> TargetVerifyResult:
        verify_input = DFlashVerifyInput(
            draft_token=ragged_window.verify_ids,
            positions=ragged_window.positions,
            draft_token_num=self.verify_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=layout,
            live_seq_lens_cpu=batch.seq_lens_cpu,
            # The packed rows are a selection of the planned verify window.
            kv_loc_plan=kv_loc_plan,
            kv_loc_cols=ragged_window.window_index,
        )
        batch.out_cache_loc = ragged_window.verify_cache_loc
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            verify_lens_cpu = (
                layout.verify_lens_cpu
                if layout.verify_lens_cpu is not None
                else layout.verify_lens.cpu().tolist()
            )
            batch.seq_lens_cpu = seq_lens_cpu_backup + torch.tensor(
                verify_lens_cpu, dtype=seq_lens_cpu_backup.dtype
            )
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

        return self._forward_prepared_verify(
            batch=batch,
            verify_input=verify_input,
            seq_lens_cpu_backup=seq_lens_cpu_backup,
            seq_lens_sum_backup=seq_lens_sum_backup,
        )

    def run_compact(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        sampling_info,
        verify_window: VerifyWindow,
        inject_gate: bool = False,
    ) -> tuple[TargetVerifyResult, torch.Tensor]:
        ragged_window = BuildRaggedVerifyWindow.execute(
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )
        if self.verify_epilogue is not None:
            self.verify_epilogue.begin_step(layout.verify_lens, armed=inject_gate)
        target_verify = self._run_ragged(
            batch=batch,
            layout=layout,
            ragged_window=ragged_window,
            sampling_info=sampling_info,
            kv_loc_plan=verify_window.kv_loc_plan,
        )
        logits_output = target_verify.logits_output

        stride = self.verify_num_draft_tokens
        if self.verify_epilogue is not None and target_verify.can_run_cuda_graph:
            strided_logits = self.verify_epilogue.strided_logits
            hidden_strided = self.verify_epilogue.strided_hidden
            assert strided_logits is not None and hidden_strided is not None, (
                "verify epilogue buffers unwritten after a graph replay -- the "
                "replayed graph was captured without the epilogue"
            )
            strided_logits = strided_logits[: bs * stride]
            hidden_strided = hidden_strided[: bs * stride]
        else:
            compact_logits = logits_output.next_token_logits
            strided_logits = ScatterCompactToStrided.execute(
                compact=compact_logits,
                layout=layout,
                fill_value=0.0,
                verify_num_draft_tokens=stride,
            )
            compact_hidden = logits_output.hidden_states
            if compact_hidden is None:
                raise RuntimeError(
                    "DSpark verify requires target hidden states, got None."
                )
            hidden_strided = ScatterCompactToStrided.execute(
                compact=compact_hidden,
                layout=layout,
                fill_value=0.0,
                verify_num_draft_tokens=stride,
            )
        apply_logits_adjustments_strided(
            next_token_logits=strided_logits,
            sampling_info=sampling_info,
            verify_num_draft_tokens=stride,
        )
        logits_output.next_token_logits = strided_logits
        logits_output.hidden_states = hidden_strided
        return target_verify, hidden_strided

    def _verify_backend_self_adds_seq_lens(self) -> bool:
        if self._verify_backend_self_adds_seq_lens_cache is None:
            backend = self.target_worker.model_runner.attn_backend
            self._verify_backend_self_adds_seq_lens_cache = hasattr(
                backend, "make_forward_metadata_from_raw_verify"
            )
        return self._verify_backend_self_adds_seq_lens_cache


class CommitInjectCtx(msgspec.Struct):
    draft_model: object
    block_pos_offsets: torch.Tensor
    resolve_pool: object
    resolve_req_to_token: object
    kv_injector: Optional[TargetHiddenKvInjector] = None
    # SGLANG_OPT_DSV41_POST_FORWARD_CUDA_GRAPH: returns the target's engram hasher
    # (or None), whose history commit the static epilogue then records in-graph.
    resolve_engram_hasher: Optional[object] = None


class AcceptOuts(msgspec.Struct):
    correct_len: torch.Tensor
    bonus: torch.Tensor
    cap_trim_lens: torch.Tensor
    commit_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    out_tokens: torch.Tensor


class StaticVerifyIO(msgspec.Struct, frozen=True):
    """A static verify graph's input and output tensors for one captured batch size."""

    candidates: torch.Tensor
    logits: torch.Tensor
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor


class DsparkVerifyEpilogue:
    def __init__(
        self,
        *,
        max_bs: int,
        verify_num_draft_tokens: int,
        device,
        tp_sync: SpecTpSync,
        commit_ctx: Optional[CommitInjectCtx] = None,
        fused_argmax: bool = False,
        simulated_correct_len: Optional[torch.Tensor] = None,
        simulate_fold: bool = False,
        fused_accept_tail: bool = False,
        accept_tail_min_bs: int = 1,
        accept_tail_max_bs: int = 0,
        deferred_commit: Optional[DeferredContextCommit] = None,
        post_fold: bool = False,
    ) -> None:
        self.max_bs = int(max_bs)
        # SGLANG_OPT_DSV41_DSPARK_POST_FOLD: static graphs also scatter new seq lens
        # and the bonus into the FutureMap's slots; built at the first warmup.
        self._post_fold = bool(post_fold)
        self.future_relay: Optional[FutureRelayBuffers] = None
        # SGLANG_DSPARK_DEFER_CONTEXT_KV: static graphs join its forked projection and
        # stage the commit's stores into it; the next draft forward runs them.
        self.deferred_commit = deferred_commit
        # SGLANG_SIMULATE_ACC_LEN with a sampled bonus inside the graph too: the bonus
        # is drawn at the simulated correct_len from these buffers, which
        # begin_sim_bonus writes before replay (<= 0 temperature: argmax).
        self.simulate_fold = bool(simulate_fold) and simulated_correct_len is not None
        self.sim_temperatures_buf = torch.zeros(
            (self.max_bs,), dtype=torch.float32, device=device
        )
        self.sim_seed_buf = torch.zeros((1,), dtype=torch.int32, device=device)
        self._sim_greedy_bs: Optional[int] = None
        self.stride = int(verify_num_draft_tokens)
        self._fused_argmax = bool(fused_argmax)
        self.gamma = self.stride - 1
        self.commit_ctx = commit_ctx
        self._tp_sync = tp_sync
        self.inject_gate_buf = torch.zeros((1,), dtype=torch.int32, device=device)
        self.verify_lens_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int64, device=device
        )
        self.draft_tokens_buf = torch.zeros(
            (self.max_bs * self.gamma,), dtype=torch.int64, device=device
        )
        self.correct_len_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int64, device=device
        )
        self.bonus_buf = torch.zeros((self.max_bs,), dtype=torch.int64, device=device)
        self.cap_trim_lens_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device=device
        )
        self.commit_lens_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device=device
        )
        self.new_seq_lens_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int64, device=device
        )
        self.out_tokens_buf = torch.zeros(
            (self.max_bs, self.stride), dtype=torch.int64, device=device
        )
        # Under the accept-site broadcast the fused tail serves only simulate_fold,
        # whose broadcast inputs (length, draw) the kernel can take.
        self._tail_broadcasts = bool(fused_accept_tail) and tp_sync.enabled(
            SpecTpSyncSite.DSPARK_ACCEPT_GRAPH
        )
        self._fused_accept_tail = bool(fused_accept_tail) and (
            self.simulate_fold or not self._tail_broadcasts
        )
        if fused_accept_tail and not self._fused_accept_tail:
            logger.info(
                "DSpark fused accept tail off: the accept-site TP broadcast is "
                "selected without SGLANG_DSPARK_SIMULATE_FOLD."
            )
        self._accept_tail_bs = (int(accept_tail_min_bs), int(accept_tail_max_bs))
        self._accept_tail_logged: set[int] = set()
        self.gated_commit_lens_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device=device
        )
        self.block_accept_lens_buf = (
            torch.zeros((self.max_bs,), dtype=torch.int32, device=device)
            if envs.SGLANG_OPT_DSV41_POST_FORWARD_CUDA_GRAPH.get()
            else None
        )
        # SGLANG_SIMULATE_ACC_LEN: the step's correct length, filled per step.
        self.simulated_correct_len = simulated_correct_len
        self.strided_logits: Optional[torch.Tensor] = None
        self.strided_hidden: Optional[torch.Tensor] = None
        self._static_step_state: Optional[tuple[int, bool]] = None
        # Set when a captured static epilogue recorded the engram history commit.
        self.folds_engram_commit = False
        # Per captured static batch size; the sampling-accept region's replays read them.
        self.static_io: dict[int, StaticVerifyIO] = {}

    def capture_hook(self, runner, out, forward_batch, num_tokens) -> None:
        if (
            runner.model_runner.is_draft_worker
            or not forward_batch.forward_mode.is_target_verify()
        ):
            return
        if (
            not isinstance(out, LogitsProcessorOutput)
            or out.next_token_logits is None
            or out.hidden_states is None
        ):
            return
        if not runner.ragged_verify_mode:
            self._static_epilogue(out, forward_batch)
            return
        self(
            compact_logits=out.next_token_logits,
            compact_hidden=out.hidden_states,
            input_ids=forward_batch.input_ids,
            seq_lens=forward_batch.seq_lens,
            req_pool_indices=forward_batch.req_pool_indices,
            bs=forward_batch.batch_size,
        )

    def begin_step(self, verify_lens, armed: bool) -> None:
        self._static_step_state = None
        if verify_lens is None:
            self.verify_lens_buf.zero_()
        else:
            bs = verify_lens.shape[0]
            self.verify_lens_buf[:bs].copy_(verify_lens)
            if bs < self.max_bs:
                self.verify_lens_buf[bs:].zero_()
        self.inject_gate_buf.fill_(1 if armed else 0)

    def begin_static_step(self, bs: int, armed: bool) -> None:
        state = (bs, armed)
        if self._static_step_state == state:
            return
        self.verify_lens_buf[:bs].fill_(self.stride)
        self.verify_lens_buf[bs:].zero_()
        self.inject_gate_buf.fill_(int(armed))
        self._static_step_state = state

    def begin_sim_bonus(self, bs: int, temperatures: Optional[torch.Tensor]) -> None:
        """The step's bonus temperatures (None: greedy) and seed, before replay."""
        if temperatures is None:
            if self._sim_greedy_bs != bs:
                self.sim_temperatures_buf.zero_()
                self._sim_greedy_bs = bs
            return
        self._sim_greedy_bs = None
        self.sim_temperatures_buf[:bs].copy_(temperatures.view(-1))
        self.sim_seed_buf.fill_(random.getrandbits(31))

    def _static_epilogue(self, out, forward_batch) -> None:
        bs = forward_batch.batch_size
        io = StaticVerifyIO(
            candidates=forward_batch.input_ids.view(bs, self.stride),
            logits=out.next_token_logits,
            hidden=out.hidden_states,
            seq_lens=forward_batch.seq_lens,
            req_pool_indices=forward_batch.req_pool_indices,
            out_cache_loc=forward_batch.out_cache_loc,
            positions=forward_batch.positions,
        )
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if capturing:
            self.static_io[bs] = io
        committed_engram = self.run_static(
            io, accept=self._accept_static_greedy, gate=self.inject_gate_buf
        )
        if committed_engram and capturing:
            self.folds_engram_commit = True

    def run_static(
        self, io: StaticVerifyIO, *, accept, gate: torch.Tensor, kv_projected=False
    ) -> bool:
        """Run ``accept`` into the accept buffers, then the commits gated by ``gate``;
        returns whether the engram history was committed."""
        bs = io.candidates.shape[0]
        verify_lens = self.verify_lens_buf[:bs]
        folds = self.folds_commit
        gated_commit_lens = accept(io, verify_lens=verify_lens, gate=gate, gated=folds)
        # Both accept branches leave the ungated commit lengths in commit_lens_buf.
        committed_engram = self._commit_engram_history(
            io.candidates,
            io.req_pool_indices,
            self.commit_lens_buf[:bs],
            verify_lens,
            gate,
        )
        if self._post_fold:
            self._scatter_future_relay(io.req_pool_indices, bs)
        if not folds:
            return committed_engram
        cache_loc = io.out_cache_loc
        state_slot = None
        if self.commit_ctx.kv_injector.swa_ring_size is not None:
            state_slot = (
                io.req_pool_indices.view(-1, 1).expand(bs, self.stride).reshape(-1)
            )
        commit = self.deferred_commit
        if commit is not None and (kv_projected or commit.join_projection()):
            commit.stage(
                cache_loc=cache_loc,
                positions=io.positions,
                commit_lens=gated_commit_lens,
                state_slot=state_slot,
            )
            return committed_engram
        self.commit_ctx.kv_injector.inject_target_hidden(
            target_hidden=io.hidden,
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc.view(bs, self.stride),
            positions=io.positions,
            commit_lens=gated_commit_lens,
            state_slot=state_slot,
        )
        return committed_engram

    def _accept_static_greedy(
        self, io: StaticVerifyIO, *, verify_lens, gate, gated: bool
    ) -> Optional[torch.Tensor]:
        bs = io.candidates.shape[0]
        if self._serves_accept_tail(bs):
            gated_commit_lens = self._static_accept_tail(
                candidates=io.candidates,
                logits=io.logits,
                seq_lens=io.seq_lens,
                verify_lens=verify_lens,
                gated=gated,
            )
            block = self.read_block_accept_lens(bs)
            if block is not None:
                # The static tail never cap-trims (it writes cap_trim_lens 0).
                block.copy_(self.commit_lens_buf[:bs])
            return gated_commit_lens
        commit_lens = self._accept(
            candidates=io.candidates,
            logits=io.logits,
            draft_tokens=io.candidates[:, 1:].contiguous(),
            seq_lens=io.seq_lens,
        )
        # Same staged locations as target verify; padded and fallback rows skip KV.
        return (
            torch.minimum(commit_lens, verify_lens.to(torch.int32)) * gate
            if gated
            else None
        )

    def engram_history(self) -> Optional[torch.Tensor]:
        """The engram history the static epilogue commits into, or None."""
        if self.commit_ctx is None or self.commit_ctx.resolve_engram_hasher is None:
            return None
        hasher = self.commit_ctx.resolve_engram_hasher()
        return None if hasher is None else hasher.history

    def _serves_accept_tail(self, bs: int) -> bool:
        low, high = self._accept_tail_bs
        return self._fused_accept_tail and low <= bs and (high == 0 or bs <= high)

    def _static_accept_tail(
        self, *, candidates, logits, seq_lens, verify_lens, gated: bool
    ) -> Optional[torch.Tensor]:
        """The greedy ``_accept`` and gated commit lengths in one launch; under
        simulate_fold the argmax is skipped and the bonus select runs in the tail."""
        bs = candidates.shape[0]
        target_predict = bonus_in = bonus_parts = None
        if not self.simulate_fold:
            target_predict = _row_argmax(logits, fused=self._fused_argmax).view(-1)
        elif self._tail_broadcasts:
            bonus_in = self._sim_bonus(logits, bs)
            # Rank 0's length and draw, as build_accept_outs broadcasts them.
            self._tp_sync.sync(
                SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, self.simulated_correct_len
            )
            self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, bonus_in)
        else:
            bonus_parts = (
                *simulated_bonus_partials_graph(**self._sim_bonus_args(logits, bs)),
                logits.shape[1],
            )
        self._log_accept_tail(bs, bonus_in=bonus_in)
        static_accept_tail(
            candidates=candidates,
            target_predict=target_predict,
            seq_lens=seq_lens,
            verify_lens=verify_lens,
            gate=self.inject_gate_buf,
            correct_len_out=self.correct_len_buf,
            bonus_out=self.bonus_buf,
            cap_trim_lens_out=self.cap_trim_lens_buf,
            commit_lens_out=self.commit_lens_buf,
            new_seq_lens_out=self.new_seq_lens_buf,
            out_tokens_out=self.out_tokens_buf,
            simulated_correct_len=self.simulated_correct_len,
            bonus_in=bonus_in,
            bonus_parts=bonus_parts,
            gated_commit_lens_out=self.gated_commit_lens_buf if gated else None,
        )
        return self.gated_commit_lens_buf[:bs] if gated else None

    def _sim_bonus_args(self, logits, bs: int) -> dict:
        return dict(
            target_logits=logits,
            # One length for every request: a stride-0 view, no copy node in the graph.
            correct_len=self.simulated_correct_len.expand(bs),
            temperatures=self.sim_temperatures_buf[:bs],
            seed=self.sim_seed_buf,
            bs=bs,
            rows_per_request=self.stride,
        )

    def _sim_bonus(self, logits, bs: int) -> torch.Tensor:
        return simulated_bonus_sample_graph(**self._sim_bonus_args(logits, bs))

    def _log_accept_tail(self, bs: int, *, bonus_in) -> None:
        """Once per captured batch size, from the branch that launches the kernel."""
        if (
            bs in self._accept_tail_logged
            or not torch.cuda.is_available()
            or not torch.cuda.is_current_stream_capturing()
        ):
            return
        self._accept_tail_logged.add(bs)
        logger.info(
            "DSpark verify graph bs=%d: fused accept tail (bonus=%s, tp-broadcast=%s)",
            bs,
            "sim-fold" if self.simulate_fold else "argmax",
            "inputs" if bonus_in is not None else "off",
        )

    def note_static_verify(self, bs: int) -> None:
        """Host, after a static verify graph replayed ``bs`` requests."""
        if self.deferred_commit is not None:
            self.deferred_commit.note_static_verify(bs)

    def _commit_engram_history(
        self, candidates, req_pool_indices, commit_lens, verify_lens, gate
    ) -> bool:
        """The post-verify engram history commit, gated per row: padded rows
        (verify_lens 0) and steps the graph does not commit (gate 0) are skipped."""
        history = self.engram_history()
        if history is None:
            return False
        engram_commit_history(
            history,
            candidates,
            req_pool_indices,
            commit_lens,
            live_rows=verify_lens,
            gate=gate,
        )
        return True

    def _scatter_future_relay(self, req_pool_indices: torch.Tensor, bs: int) -> None:
        """Unconditional: a step that takes another accept publishes and stashes on
        the host afterwards. Padded rows index slot 0, every pool's dummy row."""
        if self.commit_ctx is None:
            return
        if self.future_relay is None:
            assert not (
                torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
            ), "the future relay is built at a warmup, before capture"
            self.future_relay = FutureRelayBuffers(
                req_pool_size=self.commit_ctx.resolve_req_to_token().shape[0],
                device=req_pool_indices.device,
            )
            offer_future_relay_buffers(self.future_relay)
        rows = req_pool_indices[:bs].to(torch.int64)
        self.future_relay.new_seq_lens[rows] = self.new_seq_lens_buf[:bs]
        self.future_relay.output_tokens[rows] = self.bonus_buf[:bs]

    def read_accept(self, bs: int) -> AcceptOuts:
        return AcceptOuts(
            correct_len=self.correct_len_buf[:bs],
            bonus=self.bonus_buf[:bs],
            cap_trim_lens=self.cap_trim_lens_buf[:bs],
            commit_lens=self.commit_lens_buf[:bs],
            new_seq_lens=self.new_seq_lens_buf[:bs],
            out_tokens=self.out_tokens_buf[:bs],
        )

    def read_block_accept_lens(self, bs: int) -> Optional[torch.Tensor]:
        """commit_lens + cap_trim_lens from the in-graph accept, or None."""
        if self.block_accept_lens_buf is None:
            return None
        return self.block_accept_lens_buf[:bs]

    @property
    def folds_commit(self) -> bool:
        if self.commit_ctx is None:
            return False
        pool = self.commit_ctx.resolve_pool()
        return hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope")

    def _ensure_out(
        self, buf: Optional[torch.Tensor], compact: torch.Tensor
    ) -> torch.Tensor:
        if (
            buf is not None
            and buf.dtype == compact.dtype
            and buf.shape[1] == compact.shape[1]
        ):
            return buf
        assert not torch.cuda.is_current_stream_capturing(), (
            "DsparkVerifyEpilogue output buffers must be allocated during "
            "warmup, not inside graph capture (pool memory is unreadable "
            "post-replay)."
        )
        return torch.empty(
            (self.max_bs * self.stride, compact.shape[1]),
            dtype=compact.dtype,
            device=compact.device,
        )

    def __call__(
        self,
        *,
        compact_logits: torch.Tensor,
        compact_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        bs: int,
    ) -> None:
        self.strided_logits = self._ensure_out(self.strided_logits, compact_logits)
        self.strided_hidden = self._ensure_out(self.strided_hidden, compact_hidden)
        verify_lens = self.verify_lens_buf[:bs]
        self._scatter(compact_logits, compact_hidden, verify_lens, bs)
        candidates = torch.zeros(
            (bs * self.stride, 1), dtype=input_ids.dtype, device=input_ids.device
        )
        scatter_compact_to_strided_into(
            compact=input_ids.view(-1, 1),
            verify_lens=verify_lens,
            out=candidates,
            stride=self.stride,
            fill_value=0,
        )
        commit_lens = self._accept(
            candidates=candidates.view(bs, self.stride),
            logits=self.strided_logits[: bs * self.stride],
            draft_tokens=self.draft_tokens_buf[: bs * self.gamma].view(bs, self.gamma),
            seq_lens=seq_lens,
            cutoff_verify_lens=verify_lens,
        )
        if self.folds_commit:
            self._commit_inject(
                commit_lens, verify_lens, seq_lens, req_pool_indices, bs
            )

    def _scatter(self, compact_logits, compact_hidden, verify_lens, bs: int) -> None:
        scatter_compact_to_strided_into(
            compact=compact_logits,
            verify_lens=verify_lens,
            out=self.strided_logits[: bs * self.stride],
            stride=self.stride,
            fill_value=0.0,
        )
        scatter_compact_to_strided_into(
            compact=compact_hidden,
            verify_lens=verify_lens,
            out=self.strided_hidden[: bs * self.stride],
            stride=self.stride,
            fill_value=0.0,
        )

    def _accept(
        self, *, candidates, logits, draft_tokens, seq_lens, cutoff_verify_lens=None
    ) -> torch.Tensor:
        bs = candidates.shape[0]
        correct_len, bonus, cap_trim_lens = AcceptGreedy.execute(
            candidates=candidates,
            target_logits=logits,
            verify_num_draft_tokens=self.stride,
            cutoff_verify_lens=cutoff_verify_lens,
            fused_argmax=self._fused_argmax,
        )
        if self.simulate_fold:
            # Before the TP broadcast in build_accept_outs: every rank commits rank 0's draw.
            bonus = self._sim_bonus(logits, bs).to(bonus.dtype).view_as(bonus)
        accept = build_accept_outs(
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=seq_lens[:bs],
            draft_tokens=draft_tokens,
            gamma=self.gamma,
            verify_num_draft_tokens=self.stride,
            tp_sync=self._tp_sync,
            sync_site=SpecTpSyncSite.DSPARK_ACCEPT_GRAPH,
            simulated_correct_len=self.simulated_correct_len,
            out=self.read_accept(bs),
            block_accept_lens_out=self.read_block_accept_lens(bs),
        )
        return accept.commit_lens

    def _commit_inject(
        self, commit_lens, verify_lens, seq_lens, req_pool_indices, bs: int
    ) -> None:
        ctx = self.commit_ctx
        pool = ctx.resolve_pool()
        gated_commit_lens = (
            torch.minimum(commit_lens, verify_lens.to(torch.int32))
            * self.inject_gate_buf
        )
        if pool.swa_req_ring_size is not None:
            inject_layout = build_ring_commit_inject_layout(
                req_pool_indices=req_pool_indices,
                prefix_lens=seq_lens[:bs],
                block_pos_offsets=ctx.block_pos_offsets[: self.stride],
                commit_lens=gated_commit_lens,
                stride=self.stride,
                ring_stride=pool.swa_req_ring_size,
            )
        else:
            inject_layout = BuildCommitInjectLayout.execute(
                req_pool_indices=req_pool_indices,
                req_to_token=ctx.resolve_req_to_token(),
                prefix_lens=seq_lens[:bs],
                block_pos_offsets=ctx.block_pos_offsets[: self.stride],
                full_to_swa_mapping=pool.full_to_swa_index_mapping,
                commit_lens=gated_commit_lens,
                stride=self.stride,
            )
        with torch.inference_mode():
            ctx.draft_model.write_target_hidden_kv(
                main_hidden=self.strided_hidden[: bs * self.stride],
                swa_loc=inject_layout.swa_loc,
                positions=inject_layout.positions,
                pool=pool,
            )


def build_accept_outs(
    *,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    draft_tokens: torch.Tensor,
    gamma: int,
    verify_num_draft_tokens: int,
    tp_sync: SpecTpSync,
    sync_site: SpecTpSyncSite,
    simulated_correct_len: Optional[torch.Tensor] = None,
    target_logits: Optional[torch.Tensor] = None,
    simulate_bonus_sampling_info=None,
    out: Optional[AcceptOuts] = None,
    block_accept_lens_out: Optional[torch.Tensor] = None,
) -> AcceptOuts:
    """``out``: static buffers (the in-graph accept's) to copy the results into;
    ``block_accept_lens_out`` (with ``out``) gets commit_lens + cap_trim_lens."""
    bs = correct_len.shape[0]
    if simulated_correct_len is not None:
        # The accept runs as served; only its length is replaced.
        correct_len = simulated_correct_len.repeat(bs).to(correct_len.dtype)
        if simulate_bonus_sampling_info is not None:
            assert target_logits is not None
            bonus = sample_simulated_bonus(
                target_logits=target_logits,
                correct_len=correct_len,
                greedy_bonus=bonus,
                sampling_info=simulate_bonus_sampling_info,
                bs=bs,
                verify_num_draft_tokens=verify_num_draft_tokens,
            )
    tp_sync.sync(sync_site, correct_len)
    tp_sync.sync(sync_site, bonus)
    tp_sync.sync(sync_site, cap_trim_lens)

    finalized = FinalizeAcceptLens.execute(
        correct_len=correct_len,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=prefix_lens,
    )
    out_tokens = BuildOutTokens.execute(
        draft_tokens=draft_tokens,
        correct_len=correct_len,
        bonus=bonus,
        verify_num_draft_tokens=verify_num_draft_tokens,
        gamma=gamma,
    )
    accept = AcceptOuts(
        correct_len=correct_len,
        bonus=bonus,
        cap_trim_lens=finalized.cap_trim_lens,
        commit_lens=finalized.commit_lens,
        new_seq_lens=finalized.new_seq_lens,
        out_tokens=out_tokens,
    )
    if out is not None:
        out.correct_len.copy_(accept.correct_len)
        out.bonus.copy_(accept.bonus)
        out.cap_trim_lens.copy_(accept.cap_trim_lens)
        out.commit_lens.copy_(accept.commit_lens)
        out.new_seq_lens.copy_(accept.new_seq_lens)
        out.out_tokens.copy_(accept.out_tokens)
        if block_accept_lens_out is not None:
            torch.add(
                accept.commit_lens, accept.cap_trim_lens, out=block_accept_lens_out
            )
        return out
    return accept


def sample_simulated_bonus(
    *,
    target_logits: torch.Tensor,
    correct_len: torch.Tensor,
    greedy_bonus: torch.Tensor,
    sampling_info,
    bs: int,
    verify_num_draft_tokens: int,
) -> torch.Tensor:
    """Temperature-sample every verify row and return the row at the (simulated)
    correct_len as the bonus token. Temperature only: callers exclude top-p/top-k/min-p."""
    bonus = simulated_bonus_sample(
        target_logits=target_logits,
        correct_len=correct_len,
        temperatures=sampling_info.temperatures,
        bs=bs,
        rows_per_request=verify_num_draft_tokens,
    )
    return bonus.to(greedy_bonus.dtype).view_as(greedy_bonus)


_blockwise_rejection_noted = False


def _note_blockwise_rejection() -> None:
    global _blockwise_rejection_noted
    if not _blockwise_rejection_noted:
        _blockwise_rejection_noted = True
        logger.info("DSpark: sampled verify accepts use block-wise rejection")


def _blockwise_rejection_applies(
    *,
    sampling_info,
    rejection_noise: Optional[RejectionNoise],
    block_verification: bool,
) -> bool:
    return (
        rejection_noise is not None
        and not block_verification
        and not sampling_info.need_top_k_sampling
        and not sampling_info.need_top_p_sampling
    )


def _draft_probs(draft_block: DraftBlockResult, vocab: int, device) -> torch.Tensor:
    if draft_block.greedy_draft:
        # An argmax draft is one-hot: rejection accepts at the target probability.
        draft_tokens = draft_block.draft_tokens
        draft_probs = torch.zeros(
            (*draft_tokens.shape, vocab), dtype=torch.float32, device=device
        )
        draft_probs.scatter_(-1, draft_tokens.unsqueeze(-1), 1.0)
        return draft_probs
    bs, gamma_rows, vocab = draft_block.corrected_logits.shape
    draft_probs = SoftmaxTemp.execute(
        logits=draft_block.corrected_logits.reshape(bs * gamma_rows, vocab),
        temperatures=draft_block.temperatures,
        rows_per_request=gamma_rows,
    ).view(bs, gamma_rows, vocab)
    expect(_VERIFY_DRAFT_PROBS, draft_probs)
    return draft_probs


def accept_draft_tokens(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_block: DraftBlockResult,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_layout: Optional[RaggedVerifyLayout] = None,
    fused_argmax: bool = False,
    block_verification: bool = False,
    rejection_noise: Optional[RejectionNoise] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    greedy_mask = draft_block.greedy_mask
    cutoff_verify_lens = None if cutoff_layout is None else cutoff_layout.verify_lens
    all_greedy = sampling_info is None or sampling_info.is_all_greedy
    if all_greedy:
        return AcceptGreedy.execute(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
            fused_argmax=fused_argmax,
        )
    if _blockwise_rejection_applies(
        sampling_info=sampling_info,
        rejection_noise=rejection_noise,
        block_verification=block_verification,
    ):
        _note_blockwise_rejection()
        sampled = BlockwiseRejection.execute(
            candidates=candidates,
            target_logits=target_logits,
            draft_logits=(
                None if draft_block.greedy_draft else draft_block.corrected_logits
            ),
            target_temperatures=sampling_info.temperatures,
            draft_temperatures=draft_block.temperatures,
            noise=rejection_noise,
            cutoff_verify_lens=cutoff_verify_lens,
        )
    else:
        sampled = AcceptSampling.execute(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=_draft_probs(
                draft_block, target_logits.shape[-1], target_logits.device
            ),
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
            block_verification=block_verification,
        )
    if not sampling_info.is_any_greedy:
        return sampled
    greedy_len, greedy_bonus, greedy_trim = AcceptGreedy.execute(
        candidates=candidates,
        target_logits=target_logits,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
        fused_argmax=fused_argmax,
    )
    sampling_len, sampling_bonus, sampling_trim = sampled
    selected = SelectMixedAccept.execute(
        greedy_mask=greedy_mask,
        greedy_len=greedy_len,
        greedy_bonus=greedy_bonus,
        greedy_trim=greedy_trim,
        sampling_len=sampling_len,
        sampling_bonus=sampling_bonus,
        sampling_trim=sampling_trim,
    )
    return selected.correct_len, selected.bonus, selected.cap_trim_lens
