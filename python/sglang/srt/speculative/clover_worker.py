# SPDX-License-Identifier: Apache-2.0
"""Single-model Clover draft / causal verify on the synchronous V2 schema."""

import logging
import time

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_exec, get_parallel, get_spec
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.clover_draft import (
    CloverDraftCudaGraphRunner,
    CloverPreviewBackend,
)
from sglang.srt.speculative.clover_info import (
    CloverDraftInput,
    CloverPreviewInput,
    CloverVerifyInput,
)
from sglang.srt.speculative.clover_penalty import (
    CloverPenaltyState,
    apply_clover_penalties,
)
from sglang.srt.speculative.clover_utils import (
    clover_accept_tokens,
    clover_correct_drafts,
)
from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify
from sglang.srt.utils import get_available_gpu_memory
from sglang.srt.utils.async_probe import sanitize_nan_logits

logger = logging.getLogger(__name__)


class CloverWorker(BaseSpecWorker):
    """Preview B slots, verify causally, then commit only the accepted prefix.

    The first input is the pending token sampled by the previous round. Preview
    logits are same-position predictions; causal logits at j verify input j+1.
    With r accepted drafts, persistent KV/conv/SSM advances by r+1 inputs and
    output contains r drafts plus a new pending bonus token. Rejected slots are
    scratch space in the scheduler's existing V2 over-allocation protocol.

    Only boundary sampling uses temperature/top-k/top-p. Draft acceptance uses
    either probability thresholds or exact matches to the verify argmax.
    """

    def __init__(self, server_args, gpu_id, nccl_port, target_worker):
        super().__init__()
        self._target_worker = target_worker
        self._draft_worker = None
        self.model_runner = target_worker.model_runner
        self.device = target_worker.device
        self.gpu_id = gpu_id
        self.block_size = get_spec().speculative_num_draft_tokens
        self.speculative_num_draft_tokens = self.block_size
        self.mask_id = self.model_runner.model_config.hf_config.mask_token_id
        self.accept_rule = get_spec().speculative_clover_accept_rule
        self.accept_ratio = get_spec().speculative_clover_accept_ratio
        self.accept_delta = get_spec().speculative_clover_accept_delta
        self.draft_graph = None
        self.draft_backend = None
        self.adaptive_controller = None
        self.draft_graph_replay_ct = 0
        self.verify_graph_replay_ct = 0
        self._accept_graph = None
        self._accept_graph_key = None
        self.penalty_state = CloverPenaltyState()

    def alloc_memory_pool(self, **kwargs):
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.target_worker.get_memory_pool()
        )
        self.offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )

    def init_attention_backends(self):
        self.draft_backend = CloverPreviewBackend(self.model_runner, self.block_size)

    def init_cuda_graphs(self):
        # The target runner already resolved whether decode graphs are enabled.
        if self.model_runner.decode_cuda_graph_runner is not None:
            from sglang.srt.model_executor.cuda_graph_config import Backend

            if get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED:
                started = time.perf_counter()
                before_mem = get_available_gpu_memory(
                    self.device, self.gpu_id, empty_cache=False
                )
                self.draft_graph = CloverDraftCudaGraphRunner(
                    self.model_runner, attn_backend=self.draft_backend
                )
                after_mem = get_available_gpu_memory(
                    self.device, self.gpu_id, empty_cache=False
                )
                self._additional_graph_memory_usage["draft_decode"] = (
                    before_mem - after_mem
                )
                self._additional_graph_time_usage["draft_decode"] = (
                    time.perf_counter() - started
                )
                logger.info(
                    "CLOVER draft CUDA graphs ready: %s", self.draft_graph.capture_bs
                )

    def _sync(self, tensor):
        if get_parallel().tp_size > 1:
            get_tp_group().broadcast(tensor, src=0)
        return tensor

    def _preview_batch(self, batch, block, positions, cache_loc, plan):
        bs = batch.batch_size()
        width = self.block_size
        prefix = batch.seq_lens
        prefix_cpu = batch.seq_lens_cpu
        lengths = prefix + width
        fb = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=bs,
            input_ids=block.reshape(-1),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=lengths,
            seq_lens_cpu=prefix_cpu + width,
            orig_seq_lens=lengths,
            seq_lens_sum=int(prefix_cpu.sum()) + bs * width,
            out_cache_loc=cache_loc,
            positions=positions,
            extend_num_tokens=bs * width,
            extend_seq_lens=torch.full(
                (bs,), width, device=self.device, dtype=torch.int32
            ),
            extend_prefix_lens=prefix,
            extend_start_loc=torch.arange(bs, device=self.device, dtype=torch.int32)
            * width,
            extend_seq_lens_cpu=[width] * bs,
            extend_prefix_lens_cpu=prefix_cpu.tolist(),
            sampling_info=batch.sampling_info,
            spec_algorithm=batch.spec_algorithm,
            spec_info=CloverPreviewInput(width),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            mrope_positions=positions.unsqueeze(0).expand(3, -1),
        )
        plan.bind(fb, self.model_runner.kv_index_translator)
        return fb

    def _draft(self, fb, reqs):
        with forward_context(ForwardContext(attn_backend=self.draft_backend)):
            if self.draft_graph is not None and self.draft_graph.can_run_graph(fb):
                output = self.draft_graph.execute(fb)
                self.draft_graph_replay_ct += 1
                if self.draft_graph_replay_ct == 1:
                    logger.info("CLOVER draft CUDA graph replay active")
            else:
                self.draft_backend.init_forward_metadata(fb)
                output = self.model_runner.model(
                    fb.input_ids,
                    fb.positions,
                    fb,
                )
        # Own the candidate storage: verify reuses shared graph buffers.
        logits = output.full_logits.view(fb.batch_size, self.block_size, -1)
        apply_clover_penalties(
            logits,
            reqs,
            fb.input_ids.view(fb.batch_size, self.block_size),
            preview=True,
            state=self.penalty_state,
        )
        candidates = logits.argmax(-1)
        candidates[:, 0].copy_(fb.input_ids.view(fb.batch_size, self.block_size)[:, 0])
        return self._sync(candidates)

    def _accept_boundary(self, logits, candidates, can_run_graph, greedy=False):
        def compute():
            counts = clover_correct_drafts(
                logits,
                candidates,
                self.accept_ratio,
                self.accept_delta,
                accept_rule=self.accept_rule,
            )
            self._sync(counts)
            rows = torch.arange(logits.shape[0], device=logits.device)
            boundary = logits[rows, counts.long()].clone()
            if greedy:
                sanitize_nan_logits(boundary, "Clover greedy boundary")
                bonus = boundary.argmax(-1)
                return (
                    counts,
                    bonus,
                    counts + 1,
                    clover_accept_tokens(candidates, counts, bonus),
                )
            return counts, boundary

        if not (can_run_graph and logits.shape[0] == 1 and get_parallel().tp_size == 1):
            return compute()
        # Candidates are newly allocated each round: own their stable storage.
        # Keep just one graph, recapturing if the model's logits buffer changes.
        key = (
            logits.data_ptr(),
            tuple(logits.shape),
            logits.stride(),
            logits.dtype,
            greedy,
        )
        if self._accept_graph_key != key:
            self._accept_graph = None
            self._accept_candidates = torch.empty_like(candidates)
            self._accept_candidates.copy_(candidates)
            candidates = self._accept_candidates
            stream = torch.cuda.Stream(device=logits.device)
            stream.wait_stream(torch.cuda.current_stream(logits.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    compute()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = compute()
            torch.cuda.current_stream(logits.device).wait_stream(stream)
            self._accept_graph = graph
            self._accept_graph_outputs = outputs
            self._accept_graph_key = key
        else:
            self._accept_candidates.copy_(candidates)
        self._accept_graph.replay()
        return self._accept_graph_outputs

    @torch.no_grad()
    def forward_batch_generation(self, batch, on_publish=None, pp_proxy_tensors=None):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            result = self.target_worker.forward_batch_generation(
                batch, pp_proxy_tensors=pp_proxy_tensors
            )
            self._sync(result.next_token_ids)
            result.new_seq_lens = batch.seq_lens.clone()
            result.next_draft_input = CloverDraftInput.from_tokens(
                result.next_token_ids, result.new_seq_lens
            )
            if on_publish is not None:
                on_publish(result.new_seq_lens)
            return result

        bs = batch.batch_size()
        width = self.block_size
        state = batch.spec_info
        if not isinstance(state, CloverDraftInput):
            raise RuntimeError("CLOVER decode requires pending tokens from prefill")
        if bs == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.int64)
            return GenerationBatchResult(
                next_token_ids=empty,
                accept_lens=empty.to(torch.int32),
                new_seq_lens=empty,
                next_draft_input=CloverDraftInput.from_tokens(empty, empty),
                speculative_num_draft_tokens=width,
            )
        prefix = batch.seq_lens
        cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.req_to_token_pool.req_to_token,
            start_offset=prefix,
            end_offset=prefix + width,
            batch_size=bs,
            draft_token_num=width,
            device=self.device,
        )
        plan = self.model_runner.kv_index_translator.plan(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix,
            seq_lens_cpu=batch.seq_lens_cpu,
            write_virtual=cache_loc,
            read_extent=width,
        )
        positions = (prefix[:, None] + self.offsets).reshape(-1)
        block = torch.full(
            (bs, width), self.mask_id, device=self.device, dtype=torch.int64
        )
        block[:, 0].copy_(state.bonus_tokens)
        draft_fb = self._preview_batch(batch, block, positions, cache_loc, plan)
        candidates = self._draft(draft_fb, batch.reqs)

        verify_input = CloverVerifyInput(
            draft_token=candidates.reshape(-1),
            positions=positions,
            draft_token_num=width,
            kv_loc_plan=plan,
        )
        batch.out_cache_loc = cache_loc
        cpu_lengths, lengths_sum = batch.seq_lens_cpu, batch.seq_lens_sum
        try:
            batch.seq_lens_cpu = cpu_lengths + width
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            verify_fb, _ = verify_input.prepare_for_verify(batch, self.target_worker)
        finally:
            batch.seq_lens_cpu, batch.seq_lens_sum = cpu_lengths, lengths_sum
        result = self.target_worker.forward_batch_generation(
            None,
            forward_batch=verify_fb,
            is_verify=True,
        )
        self.verify_graph_replay_ct += int(result.can_run_cuda_graph)
        if result.can_run_cuda_graph and self.verify_graph_replay_ct == 1:
            logger.info("CLOVER verify CUDA graph replay active")
        logits = result.logits_output.next_token_logits.view(bs, width, -1)
        apply_clover_penalties(logits, batch.reqs, candidates, state=self.penalty_state)
        greedy = (
            result.can_run_cuda_graph
            and bs == 1
            and get_parallel().tp_size == 1
            and batch.sampling_info.is_all_greedy
            and not batch.sampling_info.has_custom_logit_processor
            and batch.sampling_info.sampling_mask_batch_indices is None
            and batch.sampling_info.sampling_support_logprobs_capture_indices is None
            and not envs.SGLANG_SANITIZE_NAN_LOGITS.get()
        )
        if greedy:
            num_correct_drafts, bonus_tokens, accept_lens, accept_tokens = (
                self._accept_boundary(logits, candidates, True, greedy=True)
            )
        else:
            num_correct_drafts, boundary_logits = self._accept_boundary(
                logits, candidates, result.can_run_cuda_graph
            )
            bonus_tokens = self.model_runner.sampler(
                LogitsProcessorOutput(next_token_logits=boundary_logits),
                batch.sampling_info,
                False,
                [0] * bs,
                [None] * bs,
                prefix + num_correct_drafts,
            ).to(torch.int64)
            self._sync(bonus_tokens)
            accept_lens = num_correct_drafts + 1
            accept_tokens = clover_accept_tokens(
                candidates, num_correct_drafts, bonus_tokens
            )
        accept_indices = torch.arange(
            bs * width, device=self.device, dtype=torch.int32
        ).view(bs, width)
        commit_mamba_states_after_verify(
            self.target_worker,
            batch,
            accept_lens,
            accept_indices,
            width,
        )
        new_seq_lens = prefix + accept_lens
        if on_publish is not None:
            on_publish(new_seq_lens)
        batch.forward_mode = ForwardMode.DECODE
        return GenerationBatchResult(
            logits_output=result.logits_output,
            next_token_ids=accept_tokens.reshape(-1),
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            next_draft_input=CloverDraftInput.from_tokens(bonus_tokens, new_seq_lens),
            speculative_num_draft_tokens=width,
            can_run_cuda_graph=result.can_run_cuda_graph,
        )
