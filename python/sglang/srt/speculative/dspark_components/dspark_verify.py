from __future__ import annotations

import dataclasses
import os
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
    is_unified_kv_triton,
)
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    AcceptGreedy,
    AcceptSampling,
    FinalizeAcceptLens,
    SelectMixedAccept,
    SoftmaxTemp,
    accept_greedy_triton,
    finalize_accept_lens_triton,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildCommitInjectLayout,
    BuildOutTokens,
    BuildRaggedVerifyWindow,
    RaggedVerifyWindow,
    ScatterCompactToStrided,
    build_unified_commit_inject_layout,
    scatter_compact_to_strided_into,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
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


class TargetVerifyExecutor:
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
    ) -> None:
        self.target_worker = target_worker
        self.gamma = int(gamma)
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self.model_runner = model_runner
        self.kv_injector = kv_injector
        self._tp_sync = tp_sync
        self.verify_epilogue = verify_epilogue
        self._verify_backend_self_adds_seq_lens_cache: Optional[bool] = None
        self._simulate_acc_len = float(simulate_acc_len)
        self._simulated_correct_drafts_buf: Optional[torch.Tensor] = None

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
    ) -> AcceptOuts:
        """Produce the per-request accept outcome after target verify.

        Folded path: the accept/finalize/out-token kernels already ran inside
        the target-verify cuda graph (DsparkVerifyEpilogue); read its buffers.
        Eager path: run them here, including the SGLANG_SIMULATE_ACC_LEN
        override.
        """
        if folded_accept:
            return self.verify_epilogue.read_accept(bs)

        correct_len, bonus, cap_trim_lens = accept_draft_tokens(
            candidates=verify_ids_2d,
            target_logits=target_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            cutoff_layout=layout,
        )
        if self._simulate_acc_len > 0:
            correct_len = self._simulated_correct_len(
                bs=bs, dtype=correct_len.dtype, device=correct_len.device
            )

        site = (
            SpecTpSyncSite.DSPARK_ACCEPT_GREEDY
            if sampling_info is None or sampling_info.is_all_greedy
            else SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE
        )
        self._tp_sync.sync(site, correct_len)
        self._tp_sync.sync(site, bonus)
        self._tp_sync.sync(site, cap_trim_lens)

        finalized = FinalizeAcceptLens.execute(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=prefix_lens,
        )
        out_tokens = BuildOutTokens.execute(
            draft_tokens=draft_tokens,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            gamma=self.gamma,
        )
        return AcceptOuts(
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=finalized.cap_trim_lens,
            commit_lens=finalized.commit_lens,
            new_seq_lens=finalized.new_seq_lens,
            out_tokens=out_tokens,
        )

    def _simulated_correct_len(
        self, *, bs: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        buf = self._simulated_correct_drafts_buf
        if (
            buf is None
            or buf.numel() < bs
            or buf.dtype != dtype
            or buf.device != device
        ):
            buf = torch.empty((max(bs, 512),), dtype=dtype, device=device)
            self._simulated_correct_drafts_buf = buf

        simulated_acc_len = sample_simulated_acc_len(
            self._simulate_acc_len, SIMULATE_ACC_METHOD, self.gamma + 1
        )
        return buf[:bs].fill_(simulated_acc_len - 1)

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
            )
            return
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DSpark verify requires target hidden states, got None.")
        hidden = hidden.view(bs, self.verify_num_draft_tokens, -1)
        state_slot = None
        if is_unified_kv_triton():
            # unified_kv needs the per-token draft req slot to address the SWA ring
            # (state_slot * ring + pos % ring). Verify tokens are the latest in each
            # req so they always fall in the window; the commit gate (via commit_lens
            # + cache_loc_2d) drops rejected tokens, so no final_pos skip is needed.
            vlen = verify_window.verify_cache_loc_2d.shape[1]
            state_slot = (
                batch.req_pool_indices[:bs].view(-1, 1).expand(bs, vlen).reshape(-1)
            )
        self.kv_injector.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_window.verify_cache_loc,
            cache_loc_2d=verify_window.verify_cache_loc_2d,
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
    ) -> TargetVerifyResult:
        verify_input = DFlashVerifyInput(
            draft_token=ragged_window.verify_ids,
            positions=ragged_window.positions,
            draft_token_num=self.verify_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=layout,
            live_seq_lens_cpu=batch.seq_lens_cpu,
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


class AcceptOuts(msgspec.Struct):
    correct_len: torch.Tensor
    bonus: torch.Tensor
    cap_trim_lens: torch.Tensor
    commit_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    out_tokens: torch.Tensor


class DsparkVerifyEpilogue:
    def __init__(
        self,
        *,
        max_bs: int,
        verify_num_draft_tokens: int,
        device,
        tp_sync: SpecTpSync,
        commit_ctx: Optional[CommitInjectCtx] = None,
    ) -> None:
        self.max_bs = int(max_bs)
        self.stride = int(verify_num_draft_tokens)
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
        self.strided_logits: Optional[torch.Tensor] = None
        self.strided_hidden: Optional[torch.Tensor] = None

    def capture_hook(self, runner, out, forward_batch, num_tokens) -> None:
        if runner.model_runner.is_draft_worker or not runner.ragged_verify_mode:
            return
        if (
            not isinstance(out, LogitsProcessorOutput)
            or out.next_token_logits is None
            or out.hidden_states is None
        ):
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
        if verify_lens is None:
            self.verify_lens_buf.zero_()
        else:
            bs = verify_lens.shape[0]
            self.verify_lens_buf[:bs].copy_(verify_lens)
            if bs < self.max_bs:
                self.verify_lens_buf[bs:].zero_()
        self.inject_gate_buf.fill_(1 if armed else 0)

    def read_accept(self, bs: int) -> AcceptOuts:
        return AcceptOuts(
            correct_len=self.correct_len_buf[:bs],
            bonus=self.bonus_buf[:bs],
            cap_trim_lens=self.cap_trim_lens_buf[:bs],
            commit_lens=self.commit_lens_buf[:bs],
            new_seq_lens=self.new_seq_lens_buf[:bs],
            out_tokens=self.out_tokens_buf[:bs],
        )

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
        commit_lens = self._accept(input_ids, seq_lens, verify_lens, bs)
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

    def _accept(self, input_ids, seq_lens, verify_lens, bs: int) -> torch.Tensor:
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
        correct_len, bonus, cap_trim_lens = accept_greedy_triton(
            candidates=candidates.view(bs, self.stride),
            target_logits=self.strided_logits[: bs * self.stride],
            verify_num_draft_tokens=self.stride,
            cutoff_verify_lens=verify_lens,
        )
        self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, correct_len)
        self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, bonus)
        self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, cap_trim_lens)
        finalized = finalize_accept_lens_triton(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=seq_lens[:bs],
        )
        out_tokens = BuildOutTokens.execute(
            draft_tokens=self.draft_tokens_buf[: bs * self.gamma].view(bs, self.gamma),
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=self.stride,
            gamma=self.gamma,
        )
        self.correct_len_buf[:bs].copy_(correct_len)
        self.bonus_buf[:bs].copy_(bonus)
        self.cap_trim_lens_buf[:bs].copy_(cap_trim_lens.to(torch.int32))
        self.commit_lens_buf[:bs].copy_(finalized.commit_lens)
        self.new_seq_lens_buf[:bs].copy_(finalized.new_seq_lens)
        self.out_tokens_buf[:bs].copy_(out_tokens.view(bs, self.stride))
        return finalized.commit_lens

    def _commit_inject(
        self, commit_lens, verify_lens, seq_lens, req_pool_indices, bs: int
    ) -> None:
        ctx = self.commit_ctx
        pool = ctx.resolve_pool()
        gated_commit_lens = (
            torch.minimum(commit_lens, verify_lens.to(torch.int32))
            * self.inject_gate_buf
        )
        if is_unified_kv_triton():
            inject_layout = build_unified_commit_inject_layout(
                req_pool_indices=req_pool_indices,
                prefix_lens=seq_lens[:bs],
                block_pos_offsets=ctx.block_pos_offsets[: self.stride],
                commit_lens=gated_commit_lens,
                stride=self.stride,
                ring_stride=pool.unified_swa_ring_size,
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


def _dspark_rs_chunk_size() -> int:
    value = os.environ.get("SGLANG_DSPARK_RS_CHUNK_SIZE", "0")
    try:
        return max(int(value), 0)
    except ValueError:
        return 0


def _dspark_rs_full_batch_max() -> int:
    """Keep small batches on the original full-batch path.

    Chunking is intended to cap the probability workspace for large batches.
    Repeating the sampling kernel for small batches only adds launch and
    Python scheduling overhead, so callers can set this threshold explicitly.
    """
    value = os.environ.get("SGLANG_DSPARK_RS_FULL_BATCH_MAX", "0")
    try:
        return max(int(value), 0)
    except ValueError:
        return 0


def _dspark_rs_max_workspace_bytes() -> int:
    """Return the probability-workspace budget, or zero when disabled."""
    value = os.environ.get("SGLANG_DSPARK_RS_MAX_WORKSPACE_BYTES", "0")
    try:
        return max(int(value), 0)
    except ValueError:
        return 0


def _dspark_rs_dynamic_memory_enabled() -> bool:
    """Use current CUDA free memory when choosing the verification chunk."""
    return os.environ.get("SGLANG_DSPARK_RS_DYNAMIC_MEMORY", "1") == "1"


def _dspark_rs_memory_headroom_bytes() -> int:
    value = os.environ.get("SGLANG_DSPARK_RS_MEMORY_HEADROOM_BYTES", str(512 * 1024**2))
    try:
        return max(int(value), 0)
    except ValueError:
        return 512 * 1024**2


def _dspark_rs_available_workspace_bytes(device: Optional[torch.device]) -> int:
    """Return free CUDA memory available for probability workspaces."""
    if (
        not _dspark_rs_dynamic_memory_enabled()
        or device is None
        or device.type != "cuda"
        or not torch.cuda.is_available()
    ):
        return 0
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        reserved_bytes = torch.cuda.memory_reserved(device)
        allocated_bytes = torch.cuda.memory_allocated(device)
    except RuntimeError:
        return 0
    allocator_reusable = max(int(reserved_bytes) - int(allocated_bytes), 0)
    return max(
        int(free_bytes) + allocator_reusable - _dspark_rs_memory_headroom_bytes(),
        0,
    )


def _dspark_rs_trace_enabled() -> bool:
    return os.environ.get("SGLANG_DSPARK_RS_TRACE", "0") == "1"


def _dspark_rs_memory_trace_enabled() -> bool:
    return os.environ.get("SGLANG_DSPARK_RS_MEMORY_TRACE", "0") == "1"


def _dspark_rs_memory_snapshot(stage: str) -> None:
    if not _dspark_rs_memory_trace_enabled() or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    print(
        "[DSPARK_RS_MEMORY] "
        f"stage={stage} "
        f"allocated={torch.cuda.memory_allocated()} "
        f"reserved={torch.cuda.memory_reserved()} "
        f"peak_allocated={torch.cuda.max_memory_allocated()} "
        f"peak_reserved={torch.cuda.max_memory_reserved()}",
        flush=True,
    )


def _dspark_rs_estimate_workspace_bytes(
    *, bs: int, gamma_rows: int, verify_num_draft_tokens: int, vocab: int
) -> int:
    """Estimate simultaneous FP32 draft/target probability workspace."""
    return (
        bs
        * (gamma_rows + verify_num_draft_tokens)
        * vocab
        * 4  # FP32 probability workspace
    )


def _dspark_rs_plan_chunk_size(
    *,
    bs: int,
    gamma_rows: int,
    verify_num_draft_tokens: int,
    vocab: int,
    device: Optional[torch.device] = None,
    available_workspace_bytes: Optional[int] = None,
) -> tuple[int, int, int]:
    """Return (chunk_size, full_workspace_bytes, max_workspace_bytes).

    A positive ``SGLANG_DSPARK_RS_MAX_WORKSPACE_BYTES`` sets the target budget for the estimated
    simultaneous draft/target probability tensors.  With dynamic memory
    planning enabled, current free CUDA memory is also considered; the
    full-batch fast path is retained whenever it fits.  A positive fixed
    chunk size remains an upper bound for controlled tests.
    """
    full_workspace_bytes = _dspark_rs_estimate_workspace_bytes(
        bs=bs,
        gamma_rows=gamma_rows,
        verify_num_draft_tokens=verify_num_draft_tokens,
        vocab=vocab,
    )
    max_workspace_bytes = _dspark_rs_max_workspace_bytes()
    fixed_chunk_size = _dspark_rs_chunk_size()
    full_batch_max = _dspark_rs_full_batch_max()
    if available_workspace_bytes is None:
        available_workspace_bytes = (
            _dspark_rs_available_workspace_bytes(device)
            if max_workspace_bytes > 0
            else 0
        )

    planned_chunk_size = bs
    effective_workspace_bytes = max_workspace_bytes
    if available_workspace_bytes > 0:
        if full_workspace_bytes <= available_workspace_bytes:
            effective_workspace_bytes = 0
        elif effective_workspace_bytes <= 0:
            effective_workspace_bytes = available_workspace_bytes
        else:
            effective_workspace_bytes = min(
                effective_workspace_bytes, available_workspace_bytes
            )
    if effective_workspace_bytes > 0:
        per_request_bytes = _dspark_rs_estimate_workspace_bytes(
            bs=1,
            gamma_rows=gamma_rows,
            verify_num_draft_tokens=verify_num_draft_tokens,
            vocab=vocab,
        )
        budget_chunk_size = max(1, effective_workspace_bytes // per_request_bytes)
        planned_chunk_size = min(planned_chunk_size, budget_chunk_size)
    if fixed_chunk_size > 0:
        planned_chunk_size = min(planned_chunk_size, fixed_chunk_size)

    if planned_chunk_size >= bs:
        if full_batch_max > 0 and bs > full_batch_max:
            planned_chunk_size = full_batch_max
        else:
            return 0, full_workspace_bytes, max_workspace_bytes

    if planned_chunk_size <= 0:
        return 0, full_workspace_bytes, max_workspace_bytes
    return planned_chunk_size, full_workspace_bytes, max_workspace_bytes


def _dspark_rs_trace(
    *,
    bs: int,
    gamma_rows: int,
    verify_num_draft_tokens: int,
    vocab: int,
    chunk_size: int,
    full_workspace_bytes: int,
    max_workspace_bytes: int,
) -> None:
    if not _dspark_rs_trace_enabled():
        return
    num_chunks = 1 if chunk_size == 0 else (bs + chunk_size - 1) // chunk_size
    print(
        "[DSPARK_RS_TRACE] "
        f"bs={bs} gamma_rows={gamma_rows} verify_rows={verify_num_draft_tokens} "
        f"vocab={vocab} chunk_size={chunk_size or bs} chunks={num_chunks} "
        f"full_workspace_bytes={full_workspace_bytes} "
        f"max_workspace_bytes={max_workspace_bytes}",
        flush=True,
    )


def _dspark_can_slice_sampling_info(sampling_info) -> bool:
    if sampling_info is None:
        return False
    if getattr(sampling_info, "need_min_p_sampling", False):
        return False
    if getattr(sampling_info, "has_custom_logit_processor", False):
        return False
    if getattr(sampling_info, "grammar_mask", None) is not None:
        return False
    if getattr(sampling_info, "logit_bias", None) is not None:
        return False
    if getattr(sampling_info, "grammars", None):
        return False
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    if penalizer is not None and getattr(penalizer, "is_required", False):
        return False
    return True


def _dspark_slice_sampling_info(sampling_info, start: int, end: int):
    sliced = dataclasses.replace(sampling_info)
    for name in ("temperatures", "top_ps", "top_ks", "min_ps", "sampling_seed"):
        value = getattr(sampling_info, name, None)
        if value is not None:
            setattr(sliced, name, value[start:end])
    if getattr(sampling_info, "return_sampling_masks", None) is not None:
        sliced.return_sampling_masks = sampling_info.return_sampling_masks[start:end]
    sliced.is_all_greedy = bool(torch.all(sliced.top_ks <= 1).item())
    sliced.is_any_greedy = bool(torch.any(sliced.top_ks <= 1).item())
    sliced.need_top_p_sampling = bool(torch.any(sliced.top_ps != 1.0).item())
    sliced.need_top_k_sampling = bool(torch.any(sliced.top_ks != TOP_K_ALL).item())
    return sliced


def _dspark_slice_draft_input(draft_input, sampling_info, start: int, end: int):
    """Keep DSpark top-k metadata local to the current request chunk."""
    if draft_input is None:
        return None
    chunk_top_ks = sampling_info.top_ks[start:end]
    chunk_max_top_k = max(int(chunk_top_ks.max().item()), 1)
    uniform_top_k_value = None
    if bool(torch.all(chunk_top_ks == chunk_top_ks[0]).item()):
        uniform_top_k_value = int(chunk_top_ks[0].item())
    return dataclasses.replace(
        draft_input,
        max_top_k=chunk_max_top_k,
        uniform_top_k_value=uniform_top_k_value,
    )


def _accept_sampling_chunked(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_block: DraftBlockResult,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor],
    chunk_size: int,
    greedy_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs, gamma_rows, vocab = draft_block.corrected_logits.shape
    uniform_samples = torch.rand(
        (bs, gamma), dtype=torch.float32, device=target_logits.device
    )
    uniform_samples_final = torch.rand(
        (bs,), dtype=torch.float32, device=target_logits.device
    )
    correct_out = None
    bonus_out = None
    trim_out = None
    for start in range(0, bs, chunk_size):
        end = min(start + chunk_size, bs)
        chunk_bs = end - start
        chunk_sampling_info = _dspark_slice_sampling_info(sampling_info, start, end)
        chunk_draft_input = _dspark_slice_draft_input(
            draft_input, sampling_info, start, end
        )
        draft_probs = SoftmaxTemp.execute(
            logits=draft_block.corrected_logits[start:end].reshape(
                chunk_bs * gamma_rows, vocab
            ),
            temperatures=draft_block.temperatures[start:end],
            rows_per_request=gamma_rows,
        ).view(chunk_bs, gamma_rows, vocab)
        expect(_VERIFY_DRAFT_PROBS, draft_probs)
        if start == 0:
            _dspark_rs_memory_snapshot("after_first_draft_probs")
        chunk_target_logits = target_logits[
            start * verify_num_draft_tokens : end * verify_num_draft_tokens
        ]
        chunk_cutoff = (
            None if cutoff_verify_lens is None else cutoff_verify_lens[start:end]
        )
        sampling_len, sampling_bonus, sampling_trim = AcceptSampling.execute(
            candidates=candidates[start:end],
            target_logits=chunk_target_logits,
            draft_probs=draft_probs,
            sampling_info=chunk_sampling_info,
            draft_input=chunk_draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=chunk_cutoff,
            uniform_samples=uniform_samples[start:end],
            uniform_samples_final=uniform_samples_final[start:end],
        )
        if chunk_sampling_info.is_any_greedy:
            # Keep the mixed-batch semantics of the original full path: run
            # both acceptors for the chunk, then select per request.
            greedy_len, greedy_bonus, greedy_trim = AcceptGreedy.execute(
                candidates=candidates[start:end],
                target_logits=chunk_target_logits,
                verify_num_draft_tokens=verify_num_draft_tokens,
                cutoff_verify_lens=chunk_cutoff,
            )
            selected = SelectMixedAccept.execute(
                greedy_mask=greedy_mask[start:end],
                greedy_len=greedy_len,
                greedy_bonus=greedy_bonus,
                greedy_trim=greedy_trim,
                sampling_len=sampling_len,
                sampling_bonus=sampling_bonus,
                sampling_trim=sampling_trim,
            )
            correct_len, bonus, cap_trim_lens = (
                selected.correct_len,
                selected.bonus,
                selected.cap_trim_lens,
            )
        else:
            correct_len, bonus, cap_trim_lens = (
                sampling_len,
                sampling_bonus,
                sampling_trim,
            )
        if start == 0:
            _dspark_rs_memory_snapshot("after_first_accept_sampling")
        if correct_out is None:
            correct_out = torch.empty(
                (bs,), dtype=correct_len.dtype, device=correct_len.device
            )
            bonus_out = torch.empty((bs,), dtype=bonus.dtype, device=bonus.device)
            trim_out = torch.empty(
                (bs,), dtype=cap_trim_lens.dtype, device=cap_trim_lens.device
            )
        correct_out[start:end].copy_(correct_len)
        bonus_out[start:end].copy_(bonus)
        trim_out[start:end].copy_(cap_trim_lens)
    _dspark_rs_memory_snapshot("after_chunked_accept_sampling")
    return correct_out, bonus_out, trim_out


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
        )
    bs, gamma_rows, vocab = draft_block.corrected_logits.shape
    chunk_size, full_workspace_bytes, max_workspace_bytes = _dspark_rs_plan_chunk_size(
        bs=bs,
        gamma_rows=gamma_rows,
        verify_num_draft_tokens=verify_num_draft_tokens,
        vocab=vocab,
        device=target_logits.device,
    )
    _dspark_rs_trace(
        bs=bs,
        gamma_rows=gamma_rows,
        verify_num_draft_tokens=verify_num_draft_tokens,
        vocab=vocab,
        chunk_size=chunk_size,
        full_workspace_bytes=full_workspace_bytes,
        max_workspace_bytes=max_workspace_bytes,
    )
    if _dspark_rs_memory_trace_enabled():
        torch.cuda.reset_peak_memory_stats()
        _dspark_rs_memory_snapshot("before_draft_probs")
    if (
        chunk_size > 0
        and bs > chunk_size
        and _dspark_can_slice_sampling_info(sampling_info)
    ):
        return _accept_sampling_chunked(
            candidates=candidates,
            target_logits=target_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
            chunk_size=chunk_size,
            greedy_mask=greedy_mask,
        )
    draft_probs = SoftmaxTemp.execute(
        logits=draft_block.corrected_logits.reshape(bs * gamma_rows, vocab),
        temperatures=draft_block.temperatures,
        rows_per_request=gamma_rows,
    ).view(bs, gamma_rows, vocab)
    expect(_VERIFY_DRAFT_PROBS, draft_probs)
    _dspark_rs_memory_snapshot("after_full_draft_probs")
    if not sampling_info.is_any_greedy:
        result = AcceptSampling.execute(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )
        _dspark_rs_memory_snapshot("after_full_accept_sampling")
        return result
    greedy_len, greedy_bonus, greedy_trim = AcceptGreedy.execute(
        candidates=candidates,
        target_logits=target_logits,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    sampling_len, sampling_bonus, sampling_trim = AcceptSampling.execute(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    selected = SelectMixedAccept.execute(
        greedy_mask=greedy_mask,
        greedy_len=greedy_len,
        greedy_bonus=greedy_bonus,
        greedy_trim=greedy_trim,
        sampling_len=sampling_len,
        sampling_bonus=sampling_bonus,
        sampling_trim=sampling_trim,
    )
    result = selected.correct_len, selected.bonus, selected.cap_trim_lens
    _dspark_rs_memory_snapshot("after_mixed_accept_sampling")
    return result
