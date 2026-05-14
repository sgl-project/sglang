from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleVerifyInput, EagleVerifyOutput
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_utils import generate_token_bitmask, maybe_detect_nan

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

logger = logging.getLogger(__name__)


def _get_req_tail_token_id(req) -> int:
    if req.output_ids:
        return int(req.output_ids[-1])
    if req.origin_input_ids:
        return int(req.origin_input_ids[-1])
    raise RuntimeError(
        f"Request {req.rid} has no committed token to anchor external draft verification."
    )


def _normalize_token_id(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = _normalize_token_id(item)
            if normalized is not None:
                return normalized
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_linear_topk1_tree_metadata(
    batch_size: int,
    spec_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_index = (
        torch.arange(
            spec_steps,
            dtype=torch.long,
            device=device,
        )
        .expand(batch_size, -1)
        .contiguous()
    )

    if spec_steps <= 1:
        parent_list = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    else:
        parent_list = (
            torch.arange(
                -1,
                spec_steps - 1,
                dtype=torch.long,
                device=device,
            )
            .expand(batch_size, -1)
            .contiguous()
        )

    return selected_index, parent_list


class VerifyWorker:
    _mamba_verify_update = EAGLEWorker._mamba_verify_update

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int | None,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        del gpu_id, moe_ep_rank, moe_dp_rank, nccl_port
        self.server_args = server_args
        self.target_worker = target_worker
        self.tp_rank = int(tp_rank)
        self.attn_cp_rank = int(attn_cp_rank)
        self.dp_rank = 0 if dp_rank is None else int(dp_rank)
        self.pp_rank = int(getattr(target_worker, "pp_rank", 0))
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config
        self.page_size = server_args.page_size
        self.topk = 1
        self.speculative_num_steps = int(server_args.speculative_num_steps)
        self.speculative_num_draft_tokens = int(
            server_args.speculative_num_draft_tokens
        )
        self.enable_nan_detection = bool(server_args.enable_nan_detection)
        self.device = self.model_runner.device
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self.trace_timing_enabled = bool(
            getattr(server_args, "decoupled_spec_trace_dir", None)
        )
        self.total_accept_length = 0
        self.total_num_verified_reqs = 0

    def clear_cache_pool(self):
        return

    def _trace_timestamp_ns(self) -> int | None:
        if not self.trace_timing_enabled:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter_ns()

    def _trace_elapsed_ms(self, start_ns: int | None) -> float:
        if start_ns is None:
            return 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter_ns() - start_ns) / 1_000_000

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        return self.target_worker.update_weights_from_tensor(recv_req)

    def _get_verify_buffers(self, draft_token_num: int):
        if draft_token_num != self.speculative_num_draft_tokens:
            return None, None

        attn_backend = getattr(self.target_worker.model_runner, "attn_backend", None)
        if attn_backend is None:
            return None, None

        get_buffers = getattr(
            attn_backend, "get_verify_buffers_to_fill_after_draft", None
        )
        if get_buffers is None:
            return None, None

        try:
            return get_buffers()
        except Exception as exc:
            logger.debug("Falling back to eager verify buffers: %s", exc)
            return None, None

    def _get_pad_token_id(self) -> int:
        """Return an EOS token id used to pad short external draft tails."""
        hf_generation_config = getattr(self.model_config, "hf_generation_config", None)
        eos_token_id = _normalize_token_id(
            getattr(hf_generation_config, "eos_token_id", None)
        )
        if eos_token_id is not None:
            return eos_token_id

        hf_config = getattr(self.model_config, "hf_config", None)
        eos_token_id = _normalize_token_id(getattr(hf_config, "eos_token_id", None))
        if eos_token_id is not None:
            return eos_token_id

        get_text_config = getattr(hf_config, "get_text_config", None)
        text_config = (
            get_text_config()
            if callable(get_text_config)
            else getattr(hf_config, "text_config", None)
        )
        eos_token_id = _normalize_token_id(getattr(text_config, "eos_token_id", None))
        if eos_token_id is not None:
            return eos_token_id

        eos_token_ids = getattr(self.model_config, "hf_eos_token_id", None)
        if eos_token_ids:
            return min(int(token_id) for token_id in eos_token_ids)

        raise RuntimeError("External draft verification requires an EOS token id.")

    def _build_req_verify_tokens(self, req, pad_token_id: int) -> list[int]:
        tail_token = _get_req_tail_token_id(req)
        draft_buffer = list(getattr(req, "draft_buffer", []) or [])
        spec_depth = self.speculative_num_draft_tokens - 1
        draft_tokens = list(draft_buffer[:spec_depth])
        if len(draft_tokens) < spec_depth:
            draft_tokens.extend([int(pad_token_id)] * (spec_depth - len(draft_tokens)))
        return [tail_token, *draft_tokens]

    def _get_snapshot_tail_lens(self, batch: ScheduleBatch) -> list[int]:
        return [
            min(
                len(list(getattr(req, "draft_buffer", []) or [])),
                self.speculative_num_draft_tokens - 1,
            )
            for req in batch.reqs
        ]

    def _assert_verify_output_within_snapshot_tail(
        self, batch: ScheduleBatch, verify_output: EagleVerifyOutput
    ) -> list[int]:
        # req.draft_buffer is a per-forward snapshot bound before verify. Any
        # concurrent drafter appends belong to later verify rounds.
        real_tail_lens = self._get_snapshot_tail_lens(batch)
        raw_accept_lens = [int(x) for x in verify_output.num_correct_drafts_per_req_cpu]
        for req, raw_accept_len, real_tail_len in zip(
            batch.reqs, raw_accept_lens, real_tail_lens
        ):
            assert raw_accept_len <= real_tail_len, (
                "Decoupled verify has accepted padded draft tokens: "
                f"request_id={req.rid} "
                f"raw_accept_len={raw_accept_len} "
                f"snapshot_tail_len={real_tail_len}"
            )

        if verify_output.accept_tokens is None:
            raise RuntimeError("Decoupled verify did not produce accepted tokens.")

        return raw_accept_lens

    def draft(
        self,
        batch: ScheduleBatch,
        timings: dict | None = None,
    ) -> EagleVerifyInput:
        draft_token_num = self.speculative_num_draft_tokens
        if draft_token_num < 2:
            raise RuntimeError(
                "External draft verification requires at least one draft token per request."
            )

        start_ns = self._trace_timestamp_ns()
        batch.maybe_evict_swa()
        for req in batch.reqs:
            req.decode_batch_idx += 1
        seq_lens_sum = int(torch.sum(batch.seq_lens).item())
        batch.seq_lens_sum = seq_lens_sum
        if timings is not None:
            timings["seq_lens_sum"] = seq_lens_sum

        # Accumulate penalty
        sampling_info = getattr(batch, "sampling_info", None)
        penalizer_orchestrator = getattr(sampling_info, "penalizer_orchestrator", None)
        if (
            penalizer_orchestrator is not None
            and penalizer_orchestrator.is_required
            and batch.reqs
        ):
            penalizer_orchestrator.cumulate_output_tokens(
                torch.tensor(
                    [_get_req_tail_token_id(req) for req in batch.reqs],
                    dtype=torch.int64,
                    device=batch.device,
                )
            )

        pad_token_id = self._get_pad_token_id()
        if timings is not None:
            timings["draft_preamble_ms"] = self._trace_elapsed_ms(start_ns)

        start_ns = self._trace_timestamp_ns()
        full_draft_tokens_by_req = [
            self._build_req_verify_tokens(req, pad_token_id) for req in batch.reqs
        ]
        spec_steps = draft_token_num - 1
        bonus_tokens = torch.tensor(
            [tokens[0] for tokens in full_draft_tokens_by_req],
            dtype=torch.long,
            device=batch.device,
        )
        draft_tokens = torch.tensor(
            [tokens[1:] for tokens in full_draft_tokens_by_req],
            dtype=torch.long,
            device=batch.device,
        )

        batch_size = batch.batch_size()
        selected_index, parent_list = _build_linear_topk1_tree_metadata(
            batch_size,
            spec_steps,
            batch.device,
        )
        if timings is not None:
            timings["draft_build_tokens_ms"] = self._trace_elapsed_ms(start_ns)

        start_ns = self._trace_timestamp_ns()
        tree_mask_buf, position_buf = self._get_verify_buffers(draft_token_num)
        if timings is not None:
            timings["draft_get_verify_buffers_ms"] = self._trace_elapsed_ms(start_ns)

        start_ns = self._trace_timestamp_ns()
        (
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            flat_draft_tokens,
        ) = build_tree_kernel_efficient(
            bonus_tokens=bonus_tokens,
            parent_list=parent_list,
            top_scores_index=selected_index,
            draft_tokens=draft_tokens,
            seq_lens=batch.seq_lens,
            seq_lens_sum=seq_lens_sum,
            topk=1,
            spec_steps=spec_steps,
            num_verify_tokens=draft_token_num,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            tree_mask_buf=tree_mask_buf,
            position_buf=position_buf,
        )
        if timings is not None:
            timings["draft_tree_mask_numel"] = int(tree_mask.numel())
            timings["draft_build_tree_ms"] = self._trace_elapsed_ms(start_ns)

        start_ns = self._trace_timestamp_ns()
        terminal_indices = torch.tensor(
            self._get_snapshot_tail_lens(batch),
            dtype=torch.long,
            device=batch.device,
        )
        row_indices = torch.arange(batch_size, dtype=torch.long, device=batch.device)
        retrieve_next_token[row_indices, terminal_indices] = -1
        if timings is not None:
            timings["draft_terminal_mask_ms"] = self._trace_elapsed_ms(start_ns)

        return EagleVerifyInput(
            draft_token=flat_draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=spec_steps,
            topk=1,
            draft_token_num=draft_token_num,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def verify(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        timings: dict | None = None,
    ):
        if timings is None:
            timings = {}

        was_idle = batch.forward_mode.is_idle()
        seq_lens_pre_verify = batch.seq_lens.clone()

        start_ns = self._trace_timestamp_ns()
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.return_hidden_states = False
        batch.forward_mode = ForwardMode.IDLE if was_idle else ForwardMode.TARGET_VERIFY
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrieve_next_token.shape
            ).cpu()
        timings["prepare_verify_ms"] = self._trace_elapsed_ms(start_ns)

        start_ns = self._trace_timestamp_ns()
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        timings["target_forward_ms"] = self._trace_elapsed_ms(start_ns)
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        vocab_mask = None
        if batch.has_grammar:
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrieve_next_token.device)
                batch.sampling_info.vocab_mask = None

        maybe_detect_nan(
            logits_output.next_token_logits, "decoupled_verify: target model logits"
        )

        start_ns = self._trace_timestamp_ns()
        # Decoupled verify has no local draft-extend consumer for target hidden
        # states, but EagleVerifyInput.verify expects this attribute to exist.
        spec_info.hidden_states = None
        verify_output: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        logits_output.next_token_logits = logits_output.next_token_logits[
            verify_output.accept_indices
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                verify_output.accept_indices
            ]

        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            self._mamba_verify_update(
                batch, verify_output, logits_output, spec_info, seq_lens_pre_verify
            )

        timings["eagle_verify_ms"] = self._trace_elapsed_ms(start_ns)

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, verify_output, logits_output)

        batch.forward_mode = ForwardMode.IDLE if was_idle else ForwardMode.DECODE
        # Decoupled verify rebuilds verify inputs from fresh external draft
        # snapshots each round, so there is no in-process draft state to carry.
        batch.spec_info = None
        return (
            logits_output,
            verify_output,
            can_run_cuda_graph,
            timings,
        )

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if (
            batch.forward_mode.is_extend()
            or batch.is_extend_in_batch
            or batch.forward_mode.is_idle()
        ):
            model_worker_batch = batch.get_model_worker_batch()
            result = self.target_worker.forward_batch_generation(model_worker_batch)
            return result

        total_start_ns = self._trace_timestamp_ns()
        timings = {}
        start_ns = self._trace_timestamp_ns()
        spec_info = self.draft(batch, timings if self.trace_timing_enabled else None)
        draft_ms = self._trace_elapsed_ms(start_ns)
        valid_tail_lens = self._get_snapshot_tail_lens(batch)
        can_use_full_graph_path = (
            spec_info.draft_token_num == self.speculative_num_draft_tokens
        )
        (
            logits_output,
            verify_output,
            can_run_cuda_graph,
            timings,
        ) = self.verify(batch, spec_info, timings)

        start_ns = self._trace_timestamp_ns()
        num_correct_drafts_per_req_cpu = (
            self._assert_verify_output_within_snapshot_tail(batch, verify_output)
        )
        assert_ms = self._trace_elapsed_ms(start_ns)
        accepted_tokens = verify_output.accept_tokens
        num_correct_drafts = sum(num_correct_drafts_per_req_cpu)
        reported_can_run_cuda_graph = can_run_cuda_graph and can_use_full_graph_path

        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accepted_tokens,
            num_correct_drafts=num_correct_drafts,
            num_correct_drafts_per_req_cpu=num_correct_drafts_per_req_cpu,
            can_run_cuda_graph=reported_can_run_cuda_graph,
        )
        if self.trace_timing_enabled:
            if torch.is_tensor(accepted_tokens):
                accepted_tokens_num = int(accepted_tokens.numel())
            else:
                accepted_tokens_num = len(accepted_tokens or [])
            timings.update(
                batch_size=batch.batch_size(),
                draft_token_num=int(spec_info.draft_token_num),
                num_input_tokens=int(spec_info.draft_token.numel()),
                target_can_run_cuda_graph=bool(can_run_cuda_graph),
                reported_can_run_cuda_graph=bool(reported_can_run_cuda_graph),
                valid_tail_sum=int(sum(valid_tail_lens)),
                valid_tail_min=int(min(valid_tail_lens)) if valid_tail_lens else 0,
                valid_tail_max=int(max(valid_tail_lens)) if valid_tail_lens else 0,
                num_accepted_drafts=int(num_correct_drafts),
                accepted_tokens_num=accepted_tokens_num,
                draft_ms=draft_ms,
                verify_impl="eagle",
                assert_ms=assert_ms,
                total_worker_ms=self._trace_elapsed_ms(total_start_ns),
            )
            timings.setdefault("cuda_graph_replay_prepare_ms", 0.0)
            timings.setdefault("cuda_graph_replay_ms", 0.0)
            timings.setdefault("seq_lens_sum", 0)
            timings.setdefault("draft_tree_mask_numel", 0)
            timings.setdefault("eagle_verify_ms", 0.0)
            for timing_name in (
                "draft_preamble_ms",
                "draft_build_tokens_ms",
                "draft_get_verify_buffers_ms",
                "draft_build_tree_ms",
                "draft_terminal_mask_ms",
            ):
                timings.setdefault(timing_name, 0.0)
            result.decoupled_verify_timings = timings
        num_verified_reqs = len(num_correct_drafts_per_req_cpu)
        self.total_accept_length += int(result.num_correct_drafts)
        self.total_num_verified_reqs += num_verified_reqs
        return result
