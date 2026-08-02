from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.eagle import fill_bonus_tokens
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import compute_spec_v2_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dvr.cuda_graph_runner import (
    validate_dvr_attention_backend,
)
from sglang.srt.speculative.dvr.draft import SelfDraftBackend
from sglang.srt.speculative.dvr.sampling import (
    dvr_chain_rejection_sample,
    dvr_sample_from_probs,
    dvr_sampling_probs,
)
from sglang.srt.speculative.dvr.state import (
    DVRRollbackPlan,
    DVRStateLifecycle,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import (
    eagle_prepare_for_verify,
    verify_tree_greedy_func,
)
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    spec_stage_span,
)
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    sanitize_nan_logits,
)

logger = logging.getLogger(__name__)


class DecodeVerifyRollbackWorker(BaseSpecWorker):
    """DVR worker with pluggable self-decode, EAGLE/MTP, or DFlash proposals.

    User-visible "spec v1" is a synchronous compatibility mode over this same
    worker. Every draft backend enters the same target verify, state rollback,
    output, and v1/v2 flow.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.target_model_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.device = server_args.device
        self.num_draft_steps = server_args.speculative_num_steps
        self.num_draft_tokens = server_args.speculative_num_draft_tokens
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self.state_lifecycle = DVRStateLifecycle(
            server_args=server_args,
            model_runner=self.model_runner,
        )
        self.rollback_done_event = None
        # A one-token prefill cannot seed every draft backend. Consume one
        # target-only verify before normal draft; pool slots are overwritten on
        # every prefill, so stale request identities cannot survive slot reuse.
        self.seed_verify_slots = set()
        self.draft_graph_buffers = {}
        max_bs = max(
            server_args.cuda_graph_config.decode.max_bs or 0,
            server_args.max_running_requests or 0,
            1,
        )
        num_tokens = self.num_draft_tokens
        self.chain_retrieve_index = torch.arange(
            max_bs * num_tokens, dtype=torch.long, device=self.device
        ).view(max_bs, num_tokens)
        next_token = torch.arange(
            1, num_tokens + 1, dtype=torch.long, device=self.device
        )
        next_token[-1] = -1
        self.chain_retrieve_next = next_token.repeat(max_bs, 1)
        self.chain_retrieve_sibling = torch.full(
            (max_bs, num_tokens), -1, dtype=torch.long, device=self.device
        )
        self.chain_position_offsets = torch.arange(
            num_tokens, dtype=torch.long, device=self.device
        )
        if self.model_runner.spec_algorithm.is_dvr_eagle():
            from sglang.srt.speculative.dvr.eagle import EagleDraftBackend

            self.draft_backend = EagleDraftBackend.create(
                self,
                server_args,
                gpu_id,
                ps,
                nccl_port,
                target_worker,
            )
            log_prefix = "DVR EAGLE"
        elif self.model_runner.spec_algorithm.is_dvr_dflash():
            from sglang.srt.speculative.dvr.dflash import DFlashDraftBackend

            self.draft_backend = DFlashDraftBackend.create(
                self,
                server_args,
                gpu_id,
                ps,
                nccl_port,
                target_worker,
            )
            log_prefix = "DVR DFlash"
        else:
            self.draft_backend = SelfDraftBackend(self)
            log_prefix = "DVR self-decode"
        self.verify_plan_stream = self.draft_backend.create_verify_plan_stream()

        logger.info(
            "Initialized %s worker: num_steps=%s, num_draft_tokens=%s",
            log_prefix,
            self.num_draft_steps,
            self.num_draft_tokens,
        )

    @property
    def target_worker(self):
        return self.target_model_worker

    @property
    def draft_worker(self):
        return self.draft_backend.draft_worker

    def init_attention_backends(self):
        self.draft_backend.init_attention_backends()
        # Self-DVR target worker owns the model and attention backend. Scheduler
        # already initializes it before calling this self-draft worker hook.
        target_verify_backends, state_adapter = validate_dvr_attention_backend(
            self.model_runner.attn_backend,
            ForwardMode.TARGET_VERIFY,
            phase="target verify",
        )
        self.target_verify_attn_backends = tuple(target_verify_backends)
        self.state_lifecycle.bind_state_adapter(state_adapter)

    def init_cuda_graphs(self):
        self.draft_backend.init_cuda_graphs()

    def alloc_memory_pool(
        self,
        *,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self.draft_backend.alloc_memory_pool(
            memory_pool_config,
            req_to_token_pool,
            token_to_kv_pool_allocator,
        )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    # Target verify. DVR keeps the forward call in TARGET_VERIFY mode like EAGLE,
    # then locally adapts GDN's physical window and state restore/commit.

    @property
    def war_fastpath_runner(self):
        return self.draft_backend.war_fastpath_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return self.draft_backend.spec_v2_attn_backends

    def iter_runners(self):
        return self.draft_backend.iter_runners()

    def update_weights_from_disk(self, recv_req):
        success, message = self.draft_backend.update_weights_from_disk(recv_req)
        if not success:
            return success, message

        if recv_req.recapture_cuda_graph:
            # DVR owns additional graph runners beyond ModelRunner's decode
            # graph. Rebuild the complete draft graph set after both target and
            # draft weights have been updated.
            self.draft_graph_buffers.clear()
            self.draft_backend.reset_cuda_graphs()
            self.init_cuda_graphs()

        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req):
        return self.draft_backend.update_weights_from_ipc(recv_req)

    def clear_cache_pool(self):
        self.state_lifecycle.clear_cache_state()
        self.seed_verify_slots.clear()
        self.draft_backend.clear_cache_pool()

    def prepare_for_kv_cache_release(self, req) -> None:
        if req.req_pool_idx is not None:
            self.seed_verify_slots.discard(int(req.req_pool_idx))
        if getattr(self, "device", "cpu") != "cpu":
            current_stream = torch.get_device_module(self.device).current_stream()
            read_done = self.war_fastpath_runner.war_fastpath_read_done_event
            if read_done is not None:
                current_stream.wait_event(read_done)
            if (
                self.rollback_done_event is not None
                and self.rollback_done_event is not read_done
            ):
                # Overlap may have launched one extra DVR round before the prior
                # result finishes. Its request-owned state must be committed
                # before Radix donation reuses the physical slots.
                current_stream.wait_event(self.rollback_done_event)
        self.state_lifecycle.prepare_for_cache_release(req)

    def forward_batch_generation(
        self, model_worker_batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        batch = model_worker_batch
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self.state_lifecycle.prepare_target_extend(batch)
            batch_result = self.target_worker.forward_batch_generation(
                batch,
                capture_hidden_mode=self.draft_backend.target_capture_hidden_mode,
            )
            batch_result.new_seq_lens = batch.seq_lens
            self.state_lifecycle.finish_target_extend(batch)
            decoding_rids = {req.rid for req in batch.decoding_reqs or ()}
            for req in batch.reqs:
                request_slot = int(req.req_pool_idx)
                self.seed_verify_slots.discard(request_slot)
                if (
                    self.draft_backend.requires_short_prompt_verify
                    and req.rid not in decoding_rids
                    and len(req.origin_input_ids) <= 1
                ):
                    self.seed_verify_slots.add(request_slot)
            if on_publish is not None:
                on_publish(batch_result.new_seq_lens)
            batch_result.next_draft_input = self.draft_backend.finish_prefill(
                batch, batch_result
            )
            return batch_result

        sampling_info = batch.sampling_info
        penalizer = sampling_info.penalizer_orchestrator
        if (
            sampling_info.acc_additive_penalties is not None
            or sampling_info.acc_scaling_penalties is not None
            or (penalizer is not None and penalizer.is_required)
        ):
            raise ValueError(
                "DVR request-local sampling does not support dynamic token "
                "penalties (frequency_penalty, presence_penalty, "
                "repetition_penalty, or min_new_tokens)."
            )

        # DVR decode has one shared core: draft -> target verify -> rollback.
        if batch.spec_info is None:
            batch.spec_info = self.draft_backend.idle_input()
        if batch.batch_size() > self.chain_retrieve_index.shape[0]:
            raise RuntimeError(
                "DVR decode batch exceeds its fixed chain buffers: "
                f"batch_size={batch.batch_size()}, "
                f"capacity={self.chain_retrieve_index.shape[0]}."
            )
        final_reader = self.war_fastpath_runner
        # Ignore a synchronous or previous-iteration event. The final shared
        # reader publishes a fresh event for this transaction.
        final_reader.war_fastpath_read_done_event = None
        self.rollback_done_event = None
        seed_verify_rows = []
        for row, req in enumerate(batch.reqs):
            request_slot = int(req.req_pool_idx)
            if request_slot in self.seed_verify_slots:
                seed_verify_rows.append(row)
                self.seed_verify_slots.discard(request_slot)
        with spec_stage_span("dvr_prepare"):
            rollback_plan = self.state_lifecycle.prepare_rollback(batch)
        root_only_mask = None
        if len(seed_verify_rows) == batch.batch_size():
            verify_input = self.build_root_only_verify_input(batch)
        else:
            with self.draft_backend.context(), spec_stage_span("draft"):
                verify_input = self.draft_backend.propose(batch)
            if seed_verify_rows:
                root_only_mask = torch.zeros(
                    batch.batch_size(), dtype=torch.bool, device=batch.device
                )
                root_only_mask[seed_verify_rows] = True
        assert verify_input.is_verify_input()
        batch.spec_info = verify_input
        batch_result = self.verify(
            batch,
            verify_input,
            rollback_plan=rollback_plan,
            on_publish=on_publish,
            root_only_mask=root_only_mask,
        )
        # Prepare the next backend input and commit any private draft cache to
        # the accepted target endpoint.
        self.draft_backend.commit_draft_state(batch, batch_result)

        # Graph paths publish at their last shared-pool snapshot. Eager misses
        # use the conservative end-of-transaction fence.
        read_done = final_reader.war_fastpath_read_done_event
        if read_done is None:
            read_done = torch.get_device_module(self.device).Event()
            read_done.record()
            final_reader.war_fastpath_read_done_event = read_done
        return batch_result

    def build_root_only_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """Build a fixed-width verify input whose logical tree is only the root."""

        draft_input = batch.spec_info
        assert isinstance(draft_input, EagleDraftInput)
        batch_size = batch.seq_lens.shape[0]
        width = self.num_draft_tokens
        retrieve_index = self.chain_retrieve_index[:batch_size]
        terminal = self.chain_retrieve_sibling[:batch_size]
        # Keep the physical verify shape identical to the captured DVR graph.
        # spec_steps=0 makes every padded node unreachable, so sampling accepts
        # only the root while attention/GDN retain their fixed-shape contract.
        return EagleVerifyInput(
            draft_token=(
                draft_input.bonus_tokens.to(torch.long).repeat_interleave(width)
            ),
            custom_mask=None,
            positions=(
                batch.seq_lens[:, None] + self.chain_position_offsets[None, :]
            ).reshape(-1),
            retrieve_index=retrieve_index,
            retrieve_next_token=terminal,
            retrieve_next_sibling=terminal,
            retrieve_cum_len=None,
            spec_steps=0,
            topk=1,
            draft_token_num=width,
            capture_hidden_mode=self.draft_backend.target_capture_hidden_mode,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def sample_verified_tokens(
        self,
        verify_input: EagleVerifyInput,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        root_only_mask: Optional[torch.Tensor] = None,
    ):
        """Sample the target distribution and verify one top-k=1 DVR chain."""

        device = batch.device
        if batch.forward_mode.is_idle():
            empty = torch.empty(0, dtype=torch.int32, device=device)
            return empty, empty, empty

        sampling_info = batch.sampling_info
        batch_size = len(batch.seq_lens)
        num_tokens = verify_input.draft_token_num
        logits = logits_output.next_token_logits.view(batch_size, num_tokens, -1)
        sanitize_nan_logits(logits, "verify: target model logits")

        if sampling_info.logit_bias is not None:
            logits.add_(sampling_info.logit_bias[:, None, :])

        candidates = verify_input.draft_token.view(batch_size, num_tokens)
        predict = torch.zeros(batch_size * num_tokens, dtype=torch.int32, device=device)
        accept_index = torch.full(
            (batch_size, verify_input.max_tree_depth),
            -1,
            dtype=torch.int32,
            device=device,
        )
        num_correct_drafts = torch.empty(batch_size, dtype=torch.int32, device=device)

        if sampling_info.is_all_greedy:
            target_predict = torch.argmax(logits, dim=-1)
            predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=num_correct_drafts,
                candidates=candidates,
                retrieve_index=verify_input.retrieve_index,
                retrieve_next_token=verify_input.retrieve_next_token,
                retrieve_next_sibling=verify_input.retrieve_next_sibling,
                target_predict=target_predict,
                topk=verify_input.tree_topk,
            )
            if root_only_mask is not None:
                root_indices = verify_input.retrieve_index[:, 0].to(torch.long)
                predict[root_indices] = torch.where(
                    root_only_mask,
                    target_predict[:, 0].to(torch.int32),
                    predict[root_indices],
                )
                accept_index = torch.where(
                    root_only_mask[:, None],
                    torch.full_like(accept_index, -1),
                    accept_index,
                )
                accept_index[:, 0] = torch.where(
                    root_only_mask,
                    root_indices.to(torch.int32),
                    accept_index[:, 0],
                )
                num_correct_drafts = torch.where(
                    root_only_mask,
                    torch.zeros_like(num_correct_drafts),
                    num_correct_drafts,
                )
        else:
            target_probs = F.softmax(
                logits / sampling_info.temperatures[:, None, :], dim=-1
            )
            target_probs = dvr_sampling_probs(
                target_probs.flatten(0, 1), sampling_info, num_tokens
            ).view_as(logits)
            maybe_detect_nan(target_probs, "dvr verify: filtered target probabilities")

            positions = verify_input.positions.view(batch_size, num_tokens)
            if verify_input.spec_steps == 0:
                root_indices = verify_input.retrieve_index[:, 0]
                root_tokens = dvr_sample_from_probs(
                    target_probs[:, 0],
                    sampling_info.sampling_seed,
                    positions[:, 0],
                )
                predict[root_indices.to(torch.long)] = root_tokens.to(torch.int32)
                accept_index[:, 0] = root_indices
                num_correct_drafts.zero_()
            else:
                if not self.draft_backend.uses_point_proposals:
                    expected_shape = (
                        batch_size,
                        num_tokens - 1,
                        target_probs.shape[-1],
                    )
                    if (
                        verify_input.draft_probs is None
                        or tuple(verify_input.draft_probs.shape) != expected_shape
                    ):
                        actual_shape = (
                            None
                            if verify_input.draft_probs is None
                            else tuple(verify_input.draft_probs.shape)
                        )
                        raise ValueError(
                            "DVR rejection sampling requires one "
                            "target-vocabulary proposal row per draft edge; "
                            f"got {actual_shape}, expected {expected_shape}."
                        )
                dvr_chain_rejection_sample(
                    predicts=predict,
                    accept_index=accept_index,
                    accept_token_num=num_correct_drafts,
                    candidates=candidates,
                    retrieve_index=verify_input.retrieve_index,
                    target_probs=target_probs,
                    draft_probs=verify_input.draft_probs,
                    sampling_seed=sampling_info.sampling_seed,
                    positions=positions,
                    root_only_mask=root_only_mask,
                )

        tp_group = (
            get_parallel().attn_tp_group
            if is_dp_attention_enabled()
            else get_tp_group()
        )
        if tp_group.world_size > 1:
            tp_group.broadcast(predict, src=0)
            tp_group.broadcast(accept_index, src=0)
            tp_group.broadcast(num_correct_drafts, src=0)

        return predict, num_correct_drafts + 1, accept_index

    def verify(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        rollback_plan: Optional[DVRRollbackPlan] = None,
        on_publish=None,
        root_only_mask: Optional[torch.Tensor] = None,
    ) -> GenerationBatchResult:
        scheduler_seq_lens = batch.seq_lens
        assert spec_info.is_verify_input()
        # DVR only supports topk=1 chains, whose tree mask is exactly the
        # backend's native causal mask. Both draft backends therefore enter the
        # same target-verify preparation and forward path.
        spec_info.custom_mask = None
        verify_tokens = spec_info.draft_token_num
        spec_info.num_tokens_per_req = verify_tokens
        self.draft_backend.prepare_target_verify(batch, spec_info)

        device_module = torch.get_device_module(self.device)
        verify_plan_context = (
            device_module.stream(self.verify_plan_stream)
            if self.verify_plan_stream is not None
            else nullcontext()
        )
        with verify_plan_context, spec_stage_span("verify_prepare"):
            verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
                spec_info,
                self.req_to_token_pool,
                batch,
                self.target_worker,
            )

        current_stream = device_module.current_stream()
        self.draft_backend.finish_target_verify_prepare(batch, current_stream)
        if self.verify_plan_stream is not None:
            current_stream.wait_stream(self.verify_plan_stream)
            runner = self.model_runner.decode_cuda_graph_runner
            cuda_graph_bs = (
                None if not can_run_cuda_graph or runner is None else runner.bs
            )
            for backend in self.target_verify_attn_backends:
                backend.update_verify_buffers_to_fill_after_draft(
                    spec_info, cuda_graph_bs
                )

        with spec_stage_span("verify"):
            forward_output = self.target_worker.forward_batch_generation(
                batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
            )

        logits_output = forward_output.logits_output
        self.draft_backend.validate_target_output(logits_output)

        with spec_stage_span("verify_sample"):
            maybe_detect_nan(
                logits_output.next_token_logits, "verify: target model logits"
            )
            maybe_detect_inf(
                logits_output.next_token_logits, "verify: target model logits"
            )
            predict, accept_lens, accept_index = self.sample_verified_tokens(
                spec_info, batch, logits_output, root_only_mask
            )
            if not batch.forward_mode.is_idle() and accept_lens.numel() > 0:
                accept_tokens = predict[accept_index]
                bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
                fill_bonus_tokens[(accept_lens.shape[0],)](
                    accept_tokens,
                    accept_lens,
                    bonus_tokens,
                    accept_index.shape[1],
                )
            else:
                bonus_tokens = torch.empty(
                    (0,), device=predict.device, dtype=torch.int32
                )
        new_seq_lens = scheduler_seq_lens + accept_lens
        has_verify_tokens = not batch.forward_mode.is_idle() and accept_lens.numel() > 0

        next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)

        if on_publish is not None:
            # Acceptance fully determines the next logical lengths. Publish them
            # before target-state maintenance so overlap preparation can proceed;
            # request release remains fenced by rollback_done_event below.
            on_publish(new_seq_lens)

        with spec_stage_span("dvr_rollback"):
            self.state_lifecycle.rollback(
                batch=batch,
                plan=rollback_plan,
                accept_lens=accept_lens,
            )
        if rollback_plan is not None:
            self.rollback_done_event = device_module.Event()
            self.rollback_done_event.record()
        if rollback_plan is None:
            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                verify_tokens,
            )
        if has_verify_tokens and batch.return_logprob:
            with spec_stage_span("verify_logprob"):
                compute_spec_v2_logprobs(
                    batch,
                    logits_output,
                    predict,
                    accept_index,
                    spec_info.spec_steps,
                )

        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            speculative_num_draft_tokens=self.num_draft_tokens,
            routed_experts_output=forward_output.routed_experts_output,
            indexer_topk_output=forward_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )
        return batch_result
