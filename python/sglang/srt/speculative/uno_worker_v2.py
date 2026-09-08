from __future__ import annotations

import contextlib
import copy
import logging
import time
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)
from sglang.srt.runtime_context import get_schedule, get_spec
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_utils import default_tree_mask_mode
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    run_eagle_verify,
)
from sglang.srt.speculative.spec_info import SpecInputType, SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import get_plan_stream
from sglang.srt.speculative.uno_cuda_graph_runner import (
    UnoDecodeCudaGraphRunner,
)
from sglang.srt.speculative.uno_info import UnoDraftInput, UnoForwardInput
from sglang.srt.speculative.uno_tree import build_uno_tree_proposal
from sglang.srt.speculative.uno_utils import (
    build_uno_draft_input,
    pack_uno_tree_result,
    run_uno_sampling,
    sample_uno_candidates,
    sample_uno_clean_root,
)
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    log_info_on_rank0,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)


class UnoWorkerV2(BaseSpecWorker):
    """Single-model UNO worker with linear and native-EAGLE tree decode."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port

        self._target_worker = target_worker
        self._draft_worker = None

        self.model_runner = target_worker.model_runner
        self.lora_manager = self.model_runner.lora_manager
        self.uno_lora_id = self.model_runner.uno_lora_id
        self.device = target_worker.device

        self.enable_overlap = not get_schedule().disable_overlap_schedule
        configured_topk = int(get_spec().speculative_eagle_topk or 1)
        self.tree_mode = configured_topk > 1
        # Linear UNO stores F in speculative_num_draft_tokens.  Tree UNO reuses
        # EAGLE's native dimensions: F=steps+1, K=eagle_topk, Q=draft_tokens.
        default_forward_width = (
            int(get_spec().speculative_num_steps) + 1
            if self.tree_mode
            else int(get_spec().speculative_num_draft_tokens)
        )
        self.forward_width = default_forward_width
        self.verify_width = int(get_spec().speculative_num_draft_tokens)
        self.candidate_top_k = (
            int(get_spec().speculative_eagle_topk) if self.tree_mode else 1
        )
        self.tree_depth = int(get_spec().speculative_num_steps) if self.tree_mode else 1
        self.num_speculative_proposals = self.forward_width - 1
        self.tail_width = self.forward_width + 1

        # Compatibility fields read by speculative infrastructure.
        self.speculative_num_draft_tokens = self.verify_width
        self.speculative_num_steps = self.tree_depth
        self.topk = self.candidate_top_k

        # Ordinary scheduler overlap still serializes model work on the
        # forward stream, so one persistent proposal workspace is sufficient:
        # the next reuse is ordered after this step's tree build and verify.
        self._uno_tree_workspace = {} if self.tree_mode else None
        # The scheduler constructs speculative workers before allocating the
        # target KV pools. Build the private tree-draft backend later, from
        # init_attention_backends(), after those pools exist.
        self._uno_draft_attn_backend = None
        self._uno_draft_cuda_graph_runner = None
        self.plan_stream, self.plan_stream_ctx = (
            get_plan_stream(self.device)
            if self.tree_mode
            else (None, contextlib.nullcontext())
        )

        self._tail_offsets = torch.arange(
            self.tail_width,
            dtype=torch.int64,
            device=self.device,
        )

    def _build_uno_draft_attn_backend(self):
        """Build only the F/1 backend absent from the native Q/K target role."""

        model_runner = self.model_runner
        original_workspace_flag = model_runner.init_new_workspace
        try:
            with get_spec().override(
                speculative_num_steps=1,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=self.forward_width,
            ):
                return model_runner._get_attention_backend(init_new_workspace=True)
        finally:
            model_runner.init_new_workspace = original_workspace_flag

    def init_attention_backends(self):
        """Initialize only UNO's private backend after target pool allocation."""

        if self.tree_mode:
            self._uno_draft_attn_backend = self._build_uno_draft_attn_backend()

    def init_cuda_graphs(self):
        """Capture only the private F-wide tree-draft graph."""

        self._uno_draft_cuda_graph_runner = None
        if not self.tree_mode or self.model_runner.decode_cuda_graph_runner is None:
            return None
        if self._uno_draft_attn_backend is None:
            raise RuntimeError(
                "UNO tree draft graph capture requires its attention backend."
            )

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            empty_cache=False,
        )
        log_info_on_rank0(
            logger,
            "Capture UNO tree draft CUDA graph begin. "
            f"num_tokens_per_req={self.forward_width}, "
            f"avail mem={before_mem:.2f} GB",
        )
        with self._bind_uno_draft_runtime():
            self._uno_draft_cuda_graph_runner = UnoDecodeCudaGraphRunner(
                self.model_runner,
                tree_draft_attn_backend=self._uno_draft_attn_backend,
                tree_draft_width=self.forward_width,
            )

        after_mem = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            empty_cache=False,
        )
        capture_time = time.perf_counter() - tic
        self._additional_graph_memory_usage["draft_decode"] = before_mem - after_mem
        self._additional_graph_time_usage["draft_decode"] = capture_time
        log_info_on_rank0(
            logger,
            "Capture UNO tree draft CUDA graph end. "
            f"elapsed={capture_time:.2f} s, "
            f"mem usage={(before_mem - after_mem):.2f} GB, "
            f"avail mem={after_mem:.2f} GB.",
        )

        return None

    @property
    def draft_worker(self):
        # Both passes use the target runner and its KV pool.
        return None

    @property
    def last_shared_read_runner(self):
        # The target verify is the final phase that reads shared scheduler
        # buffers, so its runner owns the WAR-barrier completion event.
        return self._target_worker.model_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        """Return every attention backend touched by one UNO step.

        Linear UNO uses only the target runner's native backend. Tree UNO adds
        one private F-wide draft backend before finishing on the native Q-wide
        target backend. The scheduler ORs these capabilities when deciding
        whether FutureMap must carry a CPU sequence-length mirror.
        """

        target_backend = self._target_worker.model_runner.attn_backend
        if not self.tree_mode:
            return (target_backend,)
        return (target_backend, self._uno_draft_attn_backend)

    def __getattr__(self, name):
        # Scheduler-facing methods not implemented by this wrapper belong to
        # the target worker. Guard initialization to avoid recursive lookup.
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def _validate_batch(self, batch: ScheduleBatch) -> None:
        if batch.forward_mode.is_idle():
            raise NotImplementedError("UNO does not support idle batches.")

        if batch.forward_mode.is_mixed() or (
            batch.forward_mode.is_decode() and batch.is_extend_in_batch
        ):
            raise NotImplementedError(
                "UNO does not support mixed extend/decode batches."
            )

        if not batch.spec_algorithm.is_uno():
            raise RuntimeError(
                "UnoWorkerV2 received a batch whose speculative algorithm is not UNO."
            )

        sampling_info = batch.sampling_info
        if sampling_info is None:
            raise RuntimeError("UNO requires sampling metadata.")

        if sampling_info.need_min_p_sampling:
            raise NotImplementedError("UNO does not support min-p sampling.")

        if batch.has_grammar:
            raise NotImplementedError("UNO does not support grammar decoding.")

        if batch.return_logprob:
            raise NotImplementedError("UNO does not support returned logprobs.")

        if batch.return_hidden_states:
            raise NotImplementedError("UNO does not support returned hidden states.")

        penalizer = sampling_info.penalizer_orchestrator
        penalties_active = (
            (penalizer is not None and penalizer.is_required)
            or sampling_info.acc_additive_penalties is not None
            or sampling_info.acc_scaling_penalties is not None
        )
        if penalties_active:
            raise NotImplementedError("UNO does not support sampling penalties.")

        if sampling_info.logit_bias is not None:
            raise NotImplementedError("UNO does not support logit bias.")

        if sampling_info.has_custom_logit_processor:
            raise NotImplementedError("UNO does not support custom logit processors.")

        if any(req.lora_id is not None for req in batch.reqs):
            raise NotImplementedError("UNO does not support multi-LoRA.")

    def _make_forward_batch(
        self,
        *,
        spec_input_type: SpecInputType,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        prefix_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: Optional[int],
        req_pool_indices: torch.Tensor,
    ) -> ForwardBatch:
        input_ids = input_ids.reshape(-1)
        positions = positions.reshape(-1)
        out_cache_loc = out_cache_loc.reshape(-1)

        spec_info = UnoForwardInput(
            spec_input_type=spec_input_type,
            positions=positions,
            draft_token_num=self.forward_width,
        )

        return ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=len(prefix_lens),
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=prefix_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            spec_algorithm=SpeculativeAlgorithm.UNO,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )

    def _run_target_block(
        self,
        forward_batch: ForwardBatch,
        *,
        need_top1: bool = True,
    ) -> tuple:
        result = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=forward_batch,
            is_verify=True,
        )

        if result.logits_output is None:
            raise RuntimeError("UNO target block returned no logits output.")

        logits = result.logits_output.next_token_logits
        if logits is None:
            raise RuntimeError("UNO target block returned no next-token logits.")

        expected_rows = forward_batch.batch_size * self.forward_width
        if logits.ndim != 2 or logits.shape[0] != expected_rows:
            raise RuntimeError(
                "UNO target block returned an invalid logits shape: "
                f"expected ({expected_rows}, vocab_size), got "
                f"{tuple(logits.shape)}."
            )

        # DFlash consumes logits directly; only greedy acceptance needs top-1.
        if not need_top1:
            return result, None

        predictions = torch.argmax(logits, dim=-1).view(
            forward_batch.batch_size,
            self.forward_width,
        )
        return result, predictions

    @staticmethod
    def _accept_and_pack(
        *,
        candidates: torch.Tensor,
        target_top1: torch.Tensor,
        committed_seq_lens: torch.Tensor,
    ) -> tuple:
        if candidates.ndim != 2:
            raise RuntimeError(
                f"UNO candidates must be rank 2, got shape={tuple(candidates.shape)}."
            )
        if target_top1.shape != candidates.shape:
            raise RuntimeError(
                "UNO candidate and target shapes differ: "
                f"{tuple(candidates.shape)} versus {tuple(target_top1.shape)}."
            )

        batch_size, forward_width = candidates.shape
        device = candidates.device

        # forward_width is static configuration, so this branch does not inspect
        # or synchronize a device tensor.
        if forward_width == 1:
            accepted_specs = torch.zeros(
                batch_size,
                dtype=torch.int32,
                device=device,
            )
        else:
            matches = candidates[:, 1:] == target_top1[:, :-1]
            accepted_specs = (
                matches.to(torch.int32).cumprod(dim=1).sum(dim=1).to(torch.int32)
            )

        accepted_specs_long = accepted_specs.to(torch.int64)
        correction = target_top1.gather(
            1,
            accepted_specs_long[:, None],
        ).squeeze(1)

        output_ids = torch.zeros(
            (batch_size, forward_width + 1),
            dtype=torch.int64,
            device=device,
        )
        output_ids[:, :forward_width].copy_(candidates)
        output_ids.scatter_(
            1,
            (accepted_specs_long + 1)[:, None],
            correction[:, None],
        )

        accept_lens = accepted_specs + 2
        new_seq_lens = committed_seq_lens + accept_lens.to(committed_seq_lens.dtype)

        return output_ids, accept_lens, new_seq_lens, correction

    def _forward_prefill(
        self,
        batch: ScheduleBatch,
        on_publish,
    ) -> GenerationBatchResult:
        result = self.target_worker.forward_batch_generation(batch)
        if not isinstance(result.next_token_ids, torch.Tensor):
            raise RuntimeError("UNO target prefill returned no sampled seed tensor.")

        seed_tokens = result.next_token_ids.reshape(-1)
        if seed_tokens.shape[0] != len(batch.reqs):
            raise RuntimeError(
                "UNO prefill seed count does not match batch size: "
                f"{seed_tokens.shape[0]} versus {len(batch.reqs)}."
            )

        result.new_seq_lens = batch.seq_lens
        result.next_draft_input = UnoDraftInput(
            bonus_tokens=seed_tokens,
            new_seq_lens=batch.seq_lens,
            forward_width=self.forward_width,
        )

        if on_publish is not None:
            on_publish(result.new_seq_lens)
        return result

    def _forward_decode_tree(
        self,
        batch: ScheduleBatch,
        on_publish,
    ) -> GenerationBatchResult:
        """Run UNO's F-wide proposal pass, then native EAGLE Q-node verify."""

        draft_state = batch.spec_info

        if batch.seq_lens.is_cuda:
            batch.seq_lens.record_stream(
                torch.get_device_module(self.device).current_stream()
            )

        batch_size = len(batch.seq_lens)
        committed_seq_lens = batch.seq_lens.clone()
        seed_tokens = draft_state.bonus_tokens.reshape(-1).to(
            device=self.device,
            dtype=torch.int64,
        )

        committed_seq_lens_cpu = None
        if batch.seq_lens_cpu is not None:
            committed_seq_lens_cpu = batch.seq_lens_cpu.to(
                device="cpu",
                dtype=torch.int64,
            )
            draft_seq_lens_cpu = committed_seq_lens_cpu + self.forward_width
            draft_seq_lens_sum = int(draft_seq_lens_cpu.sum())
        elif draft_state.reserved_seq_lens_cpu is not None:
            # This host tensor is a planning upper bound only.  The device
            # frontier below remains the exact committed length.
            draft_seq_lens_cpu = draft_state.reserved_seq_lens_cpu
            draft_seq_lens_sum = draft_state.reserved_seq_lens_sum
        else:
            draft_seq_lens_cpu = None
            draft_seq_lens_sum = None

        draft_positions = (
            committed_seq_lens.to(torch.int64)[:, None]
            + (self._tail_offsets[None, : self.forward_width])
        )
        req_pool_indices_long = batch.req_pool_indices.to(torch.int64)
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        draft_locs = req_to_token[
            req_pool_indices_long[:, None],
            draft_positions,
        ].to(torch.int64)

        draft_input_ids = build_uno_draft_input(
            seed_tokens=seed_tokens,
            forward_width=self.forward_width,
            vocab_size=self.model_runner.model_config.vocab_size,
        )
        draft_forward_batch = self._make_forward_batch(
            spec_input_type=SpecInputType.UNO_DRAFT,
            input_ids=draft_input_ids,
            positions=draft_positions,
            out_cache_loc=draft_locs,
            prefix_lens=committed_seq_lens,
            seq_lens_cpu=draft_seq_lens_cpu,
            seq_lens_sum=draft_seq_lens_sum,
            req_pool_indices=batch.req_pool_indices,
        )
        draft_result, _ = self._run_draft_block(
            draft_forward_batch,
            need_top1=False,
        )
        draft_logits = draft_result.logits_output.next_token_logits.reshape(
            batch_size,
            self.forward_width,
            -1,
        )

        sampling_info = batch.sampling_info
        if sampling_info.is_all_greedy:
            clean_root_tokens = torch.argmax(
                draft_logits[:, 0, :],
                dim=-1,
            )
        else:
            clean_root_tokens = sample_uno_clean_root(
                seed_tokens=seed_tokens,
                draft_logits=draft_logits,
                sampling_info=sampling_info,
                max_top_k=draft_state.max_top_k,
                uniform_top_k_value=draft_state.uniform_top_k_value,
            )

        proposal = build_uno_tree_proposal(
            clean_root_tokens,
            draft_logits[:, 1:, :],
            max_nodes=self.verify_width,
            candidate_top_k=self.candidate_top_k,
            temperature=sampling_info.temperatures,
            workspace=self._uno_tree_workspace,
        )

        # The first pass wrote the carried seed at C.  Give EAGLE a shallow
        # batch whose KV-ready prefix is therefore C+1; its existing allocator
        # assigns all Q tree slots at C+1 and its compactor can stay unchanged.
        verify_batch = copy.copy(batch)
        verify_batch.seq_lens = committed_seq_lens + 1
        if committed_seq_lens_cpu is None:
            verify_batch.seq_lens_cpu = None
            verify_batch.seq_lens_sum = None
        else:
            verify_batch.seq_lens_cpu = committed_seq_lens_cpu + 1
            verify_batch.seq_lens_sum = int(verify_batch.seq_lens_cpu.sum())

        verify_input = build_eagle_verify_input(
            verify_batch,
            EagleDraftInput(bonus_tokens=proposal.root_tokens),
            proposal.parent_list,
            proposal.top_scores_index,
            proposal.draft_tokens,
            None,
            target_worker=self.target_worker,
            topk=self.candidate_top_k,
            num_steps=self.tree_depth,
            num_draft_tokens=self.verify_width,
            tree_mask_mode=default_tree_mask_mode(),
            device=self.device,
        )
        verify_batch.spec_info = verify_input
        if self.plan_stream is not None:
            # C+1 was produced on the forward stream immediately above.  The
            # generic EAGLE path receives an older, already-visible frontier;
            # UNO must explicitly order its freshly derived tensor before the
            # plan stream assigns Q verify cache locations from it.
            self.plan_stream.wait_stream(
                torch.get_device_module(self.device).current_stream()
            )
        eagle_result = run_eagle_verify(
            verify_batch,
            target_worker=self.target_worker,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=(self.model_runner.token_to_kv_pool_allocator),
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.candidate_top_k,
            num_draft_tokens=self.verify_width,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            uno_target_max_top_k=draft_state.max_top_k,
        )

        packed = pack_uno_tree_result(
            clean_root_tokens=clean_root_tokens,
            eagle_predict=eagle_result.next_token_ids,
            eagle_accept_lens=eagle_result.accept_lens,
            draft_width=self.forward_width,
        )
        new_seq_lens = eagle_result.new_seq_lens
        next_draft_input = UnoDraftInput(
            bonus_tokens=eagle_result.next_draft_input.bonus_tokens,
            new_seq_lens=new_seq_lens,
            forward_width=self.forward_width,
        )

        if on_publish is not None:
            on_publish(new_seq_lens)

        # Preserve EAGLE's verify ForwardBatch keep-alive refs verbatim. FutureMap
        # relays the wrapped bonus; on_publish above relays the new frontier.
        return GenerationBatchResult(
            logits_output=eagle_result.logits_output,
            next_token_ids=packed.output_ids.reshape(-1),
            accept_lens=packed.accept_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=self.forward_width,
            speculative_output_stride=self.forward_width + 1,
            num_non_draft_tokens_per_req=2,
            new_seq_lens=new_seq_lens,
            can_run_cuda_graph=eagle_result.can_run_cuda_graph,
            routed_experts_output=eagle_result.routed_experts_output,
            indexer_topk_output=eagle_result.indexer_topk_output,
            extra_keep_alive_refs=eagle_result.extra_keep_alive_refs,
        )

    def _forward_decode(
        self,
        batch: ScheduleBatch,
        on_publish,
    ) -> GenerationBatchResult:
        if self.tree_mode:
            return self._forward_decode_tree(batch, on_publish)

        draft_state = batch.spec_info

        if batch.seq_lens.is_cuda:
            batch.seq_lens.record_stream(
                torch.get_device_module(self.device).current_stream()
            )

        batch_size = len(batch.seq_lens)
        committed_seq_lens = batch.seq_lens.clone()

        seed_tokens = draft_state.bonus_tokens.reshape(-1).to(
            device=self.device,
            dtype=torch.int64,
        )

        if batch.seq_lens_cpu is not None:
            committed_seq_lens_cpu = batch.seq_lens_cpu.to(
                device="cpu",
                dtype=torch.int32,
            )
            draft_seq_lens_cpu = committed_seq_lens_cpu + self.forward_width
            verify_seq_lens_cpu = committed_seq_lens_cpu + self.tail_width
            draft_seq_lens_sum = int(draft_seq_lens_cpu.sum())
            verify_seq_lens_sum = int(verify_seq_lens_cpu.sum())
        elif draft_state.reserved_seq_lens_cpu is not None:
            # Triton only needs a safe host planning bound. The allocator's
            # retained reservation avoids a D2H copy when FutureMap keeps the
            # exact committed frontier on GPU.
            draft_seq_lens_cpu = draft_state.reserved_seq_lens_cpu
            verify_seq_lens_cpu = draft_state.reserved_seq_lens_cpu
            draft_seq_lens_sum = draft_state.reserved_seq_lens_sum
            verify_seq_lens_sum = draft_state.reserved_seq_lens_sum
        else:
            draft_seq_lens_cpu = None
            verify_seq_lens_cpu = None
            draft_seq_lens_sum = None
            verify_seq_lens_sum = None

        logical_positions = (
            committed_seq_lens.to(torch.int64)[:, None] + (self._tail_offsets[None, :])
        )
        req_pool_indices_long = batch.req_pool_indices.to(torch.int64)
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        tail_locs = req_to_token[
            req_pool_indices_long[:, None],
            logical_positions,
        ].to(torch.int64)

        draft_input_ids = build_uno_draft_input(
            seed_tokens=seed_tokens,
            forward_width=self.forward_width,
            vocab_size=self.model_runner.model_config.vocab_size,
        )
        draft_forward_batch = self._make_forward_batch(
            spec_input_type=SpecInputType.UNO_DRAFT,
            input_ids=draft_input_ids,
            positions=logical_positions[:, : self.forward_width],
            out_cache_loc=tail_locs[:, : self.forward_width],
            prefix_lens=committed_seq_lens,
            seq_lens_cpu=draft_seq_lens_cpu,
            seq_lens_sum=draft_seq_lens_sum,
            req_pool_indices=batch.req_pool_indices,
        )
        sampling_info = batch.sampling_info
        all_greedy = sampling_info.is_all_greedy
        max_top_k = draft_state.max_top_k
        uniform_top_k_value = draft_state.uniform_top_k_value
        draft_result, candidates = self._run_draft_block(
            draft_forward_batch,
            need_top1=all_greedy,
        )
        if not all_greedy:
            draft_logits = draft_result.logits_output.next_token_logits.reshape(
                batch_size,
                self.forward_width,
                -1,
            )
            candidates, draft_distribution = sample_uno_candidates(
                draft_logits=draft_logits,
                sampling_info=sampling_info,
                max_top_k=max_top_k,
                uniform_top_k_value=uniform_top_k_value,
            )

        verify_prefix_lens = committed_seq_lens + 1
        verify_forward_batch = self._make_forward_batch(
            spec_input_type=SpecInputType.UNO_VERIFY,
            input_ids=candidates,
            positions=logical_positions[:, 1:],
            out_cache_loc=tail_locs[:, 1:],
            prefix_lens=verify_prefix_lens,
            seq_lens_cpu=verify_seq_lens_cpu,
            seq_lens_sum=verify_seq_lens_sum,
            req_pool_indices=batch.req_pool_indices,
        )
        verify_result, target_top1 = self._run_target_block(
            verify_forward_batch,
            need_top1=all_greedy,
        )

        if all_greedy:
            output_ids, accept_lens, new_seq_lens, correction = self._accept_and_pack(
                candidates=candidates,
                target_top1=target_top1,
                committed_seq_lens=committed_seq_lens,
            )
        else:
            sampling_result = run_uno_sampling(
                candidates=candidates,
                next_token_logits=verify_result.logits_output.next_token_logits,
                sampling_info=sampling_info,
                committed_frontiers=committed_seq_lens,
                draft_distribution=draft_distribution,
                max_top_k=max_top_k,
                uniform_top_k_value=uniform_top_k_value,
            )
            output_ids = sampling_result.output_ids
            accept_lens = sampling_result.accept_lens
            new_seq_lens = sampling_result.new_seq_lens
            correction = sampling_result.next_seed_tokens

        next_draft_input = UnoDraftInput(
            bonus_tokens=correction,
            new_seq_lens=new_seq_lens,
            forward_width=self.forward_width,
        )

        if on_publish is not None:
            on_publish(new_seq_lens)

        return GenerationBatchResult(
            logits_output=verify_result.logits_output,
            next_token_ids=output_ids.reshape(-1),
            accept_lens=accept_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=self.forward_width,
            speculative_output_stride=self.tail_width,
            num_non_draft_tokens_per_req=2,
            new_seq_lens=new_seq_lens,
            can_run_cuda_graph=verify_result.can_run_cuda_graph,
            routed_experts_output=verify_result.routed_experts_output,
            indexer_topk_output=verify_result.indexer_topk_output,
        )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
    ) -> GenerationBatchResult:
        del grammar_barrier
        self._validate_batch(batch)

        if batch.forward_mode == ForwardMode.EXTEND:
            return self._forward_prefill(batch, on_publish)

        if batch.forward_mode == ForwardMode.DECODE:
            return self._forward_decode(batch, on_publish)

        raise RuntimeError(
            f"UNO expected an EXTEND or DECODE batch, got {batch.forward_mode}."
        )

    def update_weights_from_disk(self, recv_req):
        # The scheduler updates the target worker before calling the spec worker.
        return True, "UNO has no separate draft weights."

    def update_weights_from_ipc(self, recv_req):
        # The scheduler updates the target worker before calling the spec worker.
        return True, "UNO has no separate draft weights."

    def update_weights_from_tensor(self, recv_req):
        # This update route selects the spec worker instead of updating both.
        return self.target_worker.update_weights_from_tensor(recv_req)

    @contextlib.contextmanager
    def _bind_uno_draft_runtime(self):
        target_attn_backend = self.model_runner.attn_backend
        target_graph_runner = self.model_runner.decode_cuda_graph_runner
        self.model_runner.attn_backend = self._uno_draft_attn_backend
        self.model_runner.decode_cuda_graph_runner = self._uno_draft_cuda_graph_runner
        try:
            yield
        finally:
            self.model_runner.attn_backend = target_attn_backend
            self.model_runner.decode_cuda_graph_runner = target_graph_runner

    def _run_draft_block(
        self,
        forward_batch: ForwardBatch,
        *,
        need_top1: bool = True,
    ):
        batch_size = forward_batch.batch_size
        self.lora_manager.reset_lora_batch()
        backend_context = (
            self._bind_uno_draft_runtime()
            if self.tree_mode
            else contextlib.nullcontext()
        )
        attn_context = (
            forward_context(ForwardContext(attn_backend=self._uno_draft_attn_backend))
            if self.tree_mode
            else contextlib.nullcontext()
        )
        # The eager runner plans through model_runner.attn_backend, while
        # attention layers execute through ForwardContext. Tree UNO binds both
        # to the same private F/1 backend for this draft pass.
        with backend_context, attn_context:
            if self.num_speculative_proposals == 0:
                return self._run_target_block(
                    forward_batch,
                    need_top1=need_top1,
                )

            graph_runner = getattr(
                self.model_runner,
                "decode_cuda_graph_runner",
                None,
            )
            # If cuda-graph is on, reuse its captured LoRA routing
            # and replay it in _run_target_block.
            # Else, prepare routing for eager.
            if not (
                forward_batch.forward_mode.is_cuda_graph()
                and graph_runner is not None
                and graph_runner.can_run_graph(forward_batch)
            ):
                self.lora_manager.prepare_lora_token_segments(
                    lora_ids=[None, self.uno_lora_id] * batch_size,
                    segment_lens=[1, self.num_speculative_proposals] * batch_size,
                )
            result = self._run_target_block(
                forward_batch,
                need_top1=need_top1,
            )
            # Clear LoRA routing before verification
            self.lora_manager.reset_lora_batch()
            return result
