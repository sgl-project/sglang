from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
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
from sglang.srt.speculative.dspark_components.kernels.dspark_accept import (
    AcceptGreedy,
    AcceptSampling,
    FinalizeAcceptLens,
    SelectMixedAccept,
    SoftmaxTemp,
    accept_greedy_triton,
    finalize_accept_lens_triton,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_verify_window import (
    BuildCommitInjectLayout,
    BuildOutTokens,
    BuildRaggedVerifyWindow,
    RaggedVerifyWindow,
    ScatterCompactToStrided,
    scatter_compact_to_strided_into,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


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
        verify_epilogue=None,
        simulate_acc_len: float = 0.0,
    ) -> None:
        self.target_worker = target_worker
        self.gamma = int(gamma)
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self.model_runner = model_runner
        self.kv_injector = kv_injector
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
        if buf is None or buf.numel() < bs or buf.dtype != dtype:
            correct_target = int(
                round(min(max(self._simulate_acc_len - 1.0, 0.0), float(self.gamma)))
            )
            buf = torch.full(
                (max(bs, 512),), correct_target, dtype=dtype, device=device
            )
            self._simulated_correct_drafts_buf = buf
        return buf[:bs]

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
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
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
        )
        batch.out_cache_loc = verify_cache_loc
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if not self._verify_backend_self_adds_seq_lens():
            if seq_lens_cpu_backup is not None:
                batch.seq_lens_cpu = seq_lens_cpu_backup + verify_w
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            elif draft_input.reserved_seq_lens_cpu is not None:
                batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
                batch.seq_lens_sum = int(draft_input.reserved_seq_lens_sum)

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
            skip_attn_backend_init=True,
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
        self.kv_injector.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_window.verify_cache_loc,
            cache_loc_2d=verify_window.verify_cache_loc_2d,
            positions=verify_window.positions_2d.reshape(-1),
            commit_lens=commit_lens,
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
        commit_ctx: Optional[CommitInjectCtx] = None,
    ) -> None:
        self.max_bs = int(max_bs)
        self.stride = int(verify_num_draft_tokens)
        self.gamma = self.stride - 1
        self.commit_ctx = commit_ctx
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
    draft_probs = SoftmaxTemp.execute(
        logits=draft_block.corrected_logits.reshape(bs * gamma_rows, vocab),
        temperatures=draft_block.temperatures,
        rows_per_request=gamma_rows,
    ).view(bs, gamma_rows, vocab)
    if not sampling_info.is_any_greedy:
        return AcceptSampling.execute(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )
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
    return selected.correct_len, selected.bonus, selected.cap_trim_lens
