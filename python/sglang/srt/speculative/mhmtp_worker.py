import logging
import os
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
from huggingface_hub import snapshot_download

from sglang.srt.distributed import GroupCoordinator, patch_tensor_parallel_group
from sglang.srt.layers.dp_attention import disable_dp_size
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
from sglang.srt.speculative.mhmtp_draft_cuda_graph_runner import (
    MHMTPDraftCudaGraphRunner,
)
from sglang.srt.speculative.mhmtp_utils import (
    MhmtpDraftInput,
    MhmtpVerifyInput,
    MhmtpVerifyOutput,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    generate_token_bitmask,
    select_top_k_tokens,
)
from sglang.srt.utils import empty_context, fast_topk, get_available_gpu_memory, is_cuda

if is_cuda():
    pass

logger = logging.getLogger(__name__)


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    """Context manager for draft model tensor parallelism.

    The draft model doesn't use data parallelism and has its own tensor parallel group.
    Note: We disable mscclpp because it doesn't support multiple communication groups.
    """
    with disable_dp_size(), patch_tensor_parallel_group(tp_group):
        yield


class MhmtpWorker(TpModelWorker):
    """Mhmtp speculative decoding worker implementation."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
        is_mtp: bool = False,
        moe_ep_rank=None,
    ):
        """Initialize MhmtpWorker.

        Args:
            server_args: Server configuration arguments.
            gpu_id: GPU device ID.
            tp_rank: Tensor parallel rank.
            dp_rank: Data parallel rank.
            nccl_port: NCCL communication port.
            target_worker: Target model worker for token verification.
            is_mtp: Whether to use multi-token prediction model.
            moe_ep_rank: Expert parallel rank for Mixture of Experts models.
        """
        self.server_args = server_args
        self.topk = 1
        self.speculative_num_steps = server_args.speculative_num_steps
        self.padded_static_len = self.speculative_num_steps + 1
        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.is_custom = is_mtp
        self.hot_token_id = None

        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        server_args.context_length = target_worker.model_runner.model_config.context_len

        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        with empty_context():
            super().__init__(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,
                dp_rank=dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                moe_ep_rank=None,
            )

        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(self.draft_model_runner.tp_group):
            self.init_attention_backend()
            self.init_cuda_graphs()

    def init_attention_backend(self) -> None:
        """Initialize multi-step attention backends for draft model."""
        if self.server_args.attention_backend != "flashinfer":
            raise ValueError(
                f"MHMTP speculative decoding not supported with "
                f"attention backend: {self.server_args.attention_backend}"
            )

        self._init_flashinfer_backend()
        self.draft_extend_attn_backend = None
        self.padded_static_len = self.speculative_num_steps + 1
        self.has_prefill_wrapper_verify = True
        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

    def _init_flashinfer_backend(self) -> None:
        """Initialize FlashInfer attention backend for draft model."""
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferMultiStepDraftBackend,
        )

        self.draft_attn_backend = FlashInferMultiStepDraftBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
        )

    def init_cuda_graphs(self) -> None:
        """Capture CUDA graphs for draft model inference."""
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Starting CUDA graph capture for draft model. "
            f"Available GPU memory: {before_mem:.2f} GB"
        )

        self.cuda_graph_runner_for_draft_extend = MHMTPDraftCudaGraphRunner(
            self, self.speculative_num_steps
        )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        elapsed_time = time.perf_counter() - tic
        mem_used = before_mem - after_mem
        logger.info(
            f"Completed CUDA graph capture. "
            f"Time: {elapsed_time:.2f}s, "
            f"Available GPU memory: {after_mem:.2f} GB, "
            f"Memory used: {mem_used:.2f} GB"
        )

    @property
    def draft_model_runner(self):
        """Get the draft model runner instance."""
        return self.model_runner

    def forward_batch_speculative_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        """Main forward pass for speculative generation."""
        if batch.forward_mode.is_decode():
            return self._forward_decode_mode(batch)
        elif batch.forward_mode.is_idle():
            return self._forward_idle_mode(batch)
        else:
            return self._forward_extend_mode(batch)

    def _forward_decode_mode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Forward pass in decode mode (verification phase)."""
        spec_info = batch.spec_info
        logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
            self.verify(batch, spec_info)
        )

        accept_length = verify_output.draft_input.accept_length_cpu
        if not accept_length or len(accept_length) == 0:
            batch.forward_mode = ForwardMode.DECODE
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )

        if batch.spec_info.verified_id is not None:
            with self.draft_tp_context(self.draft_model_runner.tp_group):
                batch.input_ids = batch.spec_info.verified_id
                batch.spec_info.hidden_states = verify_output.draft_input.pre_hiddens
                verify_input = self.draft_extend(batch)
                batch.spec_info = verify_input
                batch.forward_mode = ForwardMode.DECODE

        num_accepted = (
            sum(accept_length)
            if isinstance(accept_length, list)
            else accept_length.sum().item()
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=num_accepted,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _forward_idle_mode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Forward pass in idle mode (prefill phase)."""
        model_worker_batch = batch.get_model_worker_batch()
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)

        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=batch_result.next_token_ids,
            num_accepted_tokens=0,
            can_run_cuda_graph=False,
        )

    def _forward_extend_mode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Forward pass in extend mode (target extend + draft extend)."""
        logits_output, next_token_ids, bid = self.forward_target_extend(batch)

        with self.draft_tp_context(self.draft_model_runner.tp_group):
            self.forward_draft_extend(batch, logits_output, next_token_ids)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=0,
            can_run_cuda_graph=False,
        )

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, List[int], int]:
        """Run target model for one step to get full hidden states."""
        batch.seq_lens_cpu = None
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        bid = getattr(model_worker_batch, "bid", None) or id(batch)
        logits_output = batch_result.logits_output
        next_token_ids = batch_result.next_token_ids
        return logits_output, next_token_ids, bid

    def verify(
        self, batch: ScheduleBatch, spec_info: MhmtpVerifyInput
    ) -> Tuple[LogitsProcessorOutput, MhmtpVerifyOutput, ForwardBatch, bool]:
        """Verify draft tokens using the target model."""
        spec_info.prepare_for_verify(batch, self.page_size)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec_info
        model_worker_batch = batch.get_model_worker_batch()

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrive_next_token.shape
            ).cpu()

        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)

        logits_output = batch_result.logits_output
        can_run_cuda_graph = (
            batch_result.can_run_cuda_graph
            if hasattr(batch_result, "can_run_cuda_graph")
            else False
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
                vocab_mask = vocab_mask.to(spec_info.retrive_next_token.device)
                batch.sampling_info.vocab_mask = None

        self._detect_nan_if_needed(logits_output)
        spec_info.hidden_states = logits_output.hidden_states

        res: MhmtpVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]

        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = res.draft_input

        if (
            hasattr(res.draft_input, "req_pool_indices_for_draft_extend")
            and res.draft_input.req_pool_indices_for_draft_extend is not None
        ):
            batch.req_pool_indices = res.draft_input.req_pool_indices_for_draft_extend

        if batch.spec_info.verified_id is None:
            return logits_output, res, model_worker_batch, can_run_cuda_graph

        accept_length_cpu = res.draft_input.accept_length_cpu

        if not accept_length_cpu or len(accept_length_cpu) == 0:
            batch.forward_mode = ForwardMode.DECODE
            batch.spec_info = res.draft_input

            if (
                hasattr(res.draft_input, "req_pool_indices_for_draft_extend")
                and res.draft_input.req_pool_indices_for_draft_extend is not None
            ):
                batch.req_pool_indices = (
                    res.draft_input.req_pool_indices_for_draft_extend
                )

            return logits_output, res, model_worker_batch, can_run_cuda_graph

        # Similar to the truncation in “forward_draft_extend”, after verification,
        # we also need to replenish the states to facilitate subsequent
        # “draft_extend” processing with CUDA graph.
        split_lengths = [length + 1 for length in accept_length_cpu]

        split_hiddens = torch.split(res.draft_input.hidden_states, split_lengths, dim=0)
        split_verify_ids = torch.split(
            batch.spec_info.verified_id, split_lengths, dim=0
        )
        split_out_locs = torch.split(batch.out_cache_loc, split_lengths, dim=0)

        spec_pre_hiddens = spec_info.pre_hiddens
        spec_pre_verify_ids = spec_info.pre_verify_ids

        size = self.server_args.speculative_num_draft_tokens
        spec_per_batch = torch.split(
            spec_pre_hiddens, [size] * len(split_hiddens), dim=0
        )
        spec_verify_per_batch = torch.split(
            spec_pre_verify_ids, [size] * len(split_hiddens), dim=0
        )

        pre_hiddens_parts = []
        pre_verify_parts = []
        outloc_parts = []

        for i, (
            verify_h,
            verify_v,
            verify_loc,
            pre_h,
            pre_v,
            length,
        ) in enumerate(
            zip(
                split_hiddens,
                split_verify_ids,
                split_out_locs,
                spec_per_batch,
                spec_verify_per_batch,
                accept_length_cpu,
            )
        ):
            combined_hidden = torch.cat([pre_h, verify_h], dim=0)[-size:]
            combined_verify = torch.cat([pre_v, verify_v], dim=0)[-size:]

            pre_hiddens_parts.append(combined_hidden)
            pre_verify_parts.append(combined_verify)

            current_loc = verify_loc
            if len(verify_v) < size:
                if res.draft_input.unfinished_index is not None:
                    unfinished_req_index = res.draft_input.unfinished_index[i]
                else:
                    unfinished_req_index = i

                pre_loc = batch.spec_info.pre_out_locs[0][i]
                current_loc = torch.cat([pre_loc, current_loc], dim=0)[-size:]
                self.set_last_out_loc(batch.spec_info, current_loc, 0, i)

            outloc_parts.append(current_loc)

        res.draft_input.pre_hiddens = torch.cat(pre_hiddens_parts, dim=0)
        res.draft_input.pre_verify_ids = torch.cat(pre_verify_parts, dim=0)

        if batch.return_logprob:
            self.add_logprob_values(batch, res, logits_output)

        batch.spec_info.verified_id = res.draft_input.pre_verify_ids
        batch.out_cache_loc = torch.cat(outloc_parts, dim=0)

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def _detect_nan_if_needed(self, logits_output: LogitsProcessorOutput) -> None:
        """Detect NaN values in logits if enabled."""
        if self.enable_nan_detection:
            logits = logits_output.next_token_logits
            if torch.any(torch.isnan(logits)):
                logger.error("NaN detected in logits during sampling!")
                raise ValueError("Detected errors during sampling! NaN in the logits.")

    def _append_tensor(
        self,
        cur_tensor: Optional[torch.Tensor],
        tensor: Optional[torch.Tensor],
        dim: int = 0,
    ) -> Optional[torch.Tensor]:
        """Helper method to append tensors along a dimension."""
        if cur_tensor is None:
            return tensor
        if tensor is not None:
            return torch.cat([cur_tensor, tensor], dim=dim)
        return cur_tensor

    def append_out_locs(
        self,
        cur_out_locs: Optional[torch.Tensor],
        out_locs: torch.Tensor,
    ) -> torch.Tensor:
        """Append output cache locations."""
        return self._append_tensor(cur_out_locs, out_locs)

    def append_hidden_states(
        self,
        cur_hidden_states: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Append hidden state tensors."""
        return self._append_tensor(cur_hidden_states, hidden_states)

    def append_input_ids(
        self,
        cur_input_ids: Optional[torch.Tensor],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Append input ID tensors."""
        return self._append_tensor(cur_input_ids, input_ids)

    def append_positions(
        self,
        cur_positions: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Append position tensors."""
        return self._append_tensor(cur_positions, positions)

    def build_sliding_positions(
        self, cur_positions: torch.Tensor, size: torch.Tensor
    ) -> torch.Tensor:
        """Build sliding window positions."""
        return torch.arange(
            cur_positions[-1] - size,
            cur_positions[-1],
            device=cur_positions.device,
        )

    def set_last_hidden_states(
        self,
        spec_info: MhmtpDraftInput,
        hidden_states: torch.Tensor,
        mtp_index: int = 0,
        batch_index: int = 0,
    ) -> None:
        """Set last hidden states for speculative info."""
        if len(spec_info.last_hidden_states) == 0:
            for _ in range(self.speculative_num_steps):
                spec_info.last_hidden_states.append([])

        if len(spec_info.last_hidden_states[mtp_index]) > batch_index:
            spec_info.last_hidden_states[mtp_index][batch_index] = hidden_states
        else:
            spec_info.last_hidden_states[mtp_index].append(hidden_states)

    def set_last_out_loc(
        self,
        spec_info: MhmtpDraftInput,
        out_loc: torch.Tensor,
        mtp_index: int = 0,
        batch_index: int = 0,
    ) -> None:
        """Set last output location for speculative info."""
        if len(spec_info.pre_out_locs) == 0:
            for _ in range(self.speculative_num_steps):
                spec_info.pre_out_locs.append([])

        if len(spec_info.pre_out_locs[mtp_index]) > batch_index:
            spec_info.pre_out_locs[mtp_index][batch_index] = out_loc
        else:
            spec_info.pre_out_locs[mtp_index].append(out_loc)

    def add_logprob_values(
        self,
        batch: ScheduleBatch,
        res: MhmtpVerifyOutput,
        logits_output: LogitsProcessorOutput,
    ) -> None:
        """Add log probability values to batch results."""
        logits_output = res.logits_output
        top_logprobs_nums = batch.top_logprobs_nums
        token_ids_logprobs = batch.token_ids_logprobs
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits, dim=-1
        )
        batch_next_token_ids = res.verified_id
        num_tokens_per_req = [accept + 1 for accept in res.accept_length_per_req_cpu]

        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)

        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums_repeat_interleaved)

        if any(x is not None for x in token_ids_logprobs):
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(logprobs, token_ids_logprobs_repeat_interleaved)

        logits_output.next_token_logprobs = logprobs[
            torch.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
            batch_next_token_ids,
        ]

        pt = 0
        next_token_logprobs = logits_output.next_token_logprobs.tolist()
        verified_ids = batch_next_token_ids.tolist()
        for req, num_tokens in zip(batch.reqs, num_tokens_per_req):
            for _ in range(num_tokens):
                if req.return_logprob:
                    req.output_token_logprobs_val.append(next_token_logprobs[pt])
                    req.output_token_logprobs_idx.append(verified_ids[pt])
                    if req.top_logprobs_num > 0:
                        req.output_top_logprobs_val.append(
                            res.logits_output.next_token_top_logprobs_val[pt]
                        )
                        req.output_top_logprobs_idx.append(
                            res.logits_output.next_token_top_logprobs_idx[pt]
                        )
                pt += 1

    def _preprocess_tree_inputs(
        self,
        verified_id: torch.Tensor,
        score_list: List[torch.Tensor],
        token_list: List[torch.Tensor],
        parents_list: List[torch.Tensor],
        num_verify_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess tree structure inputs for speculative tree generation."""
        score_list = torch.cat(score_list, dim=1).flatten(1)
        ss_token_list = torch.cat(token_list, dim=1)

        top_scores = torch.topk(score_list, num_verify_tokens - 1, dim=-1)
        top_scores_index = torch.sort(top_scores.indices).values

        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: MhmtpDraftInput
    ) -> None:
        """Capture information needed for subsequent decode steps."""
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        draft_input.hidden_states = logits_output.hidden_states

    def draft_extend(self, batch: ScheduleBatch) -> MhmtpVerifyInput:
        """Extend draft tokens using draft model."""
        seq_len_bak = batch.seq_lens
        req_pool_bak = batch.req_pool_indices

        batch_size = min(batch.batch_size(), len(batch.spec_info.accept_length))
        batch.spec_info.prepare_for_draft_extend(batch, self.speculative_num_steps)
        batch.spec_info.accept_length = batch.spec_info.accept_length.add_(1)
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        batch.return_logprob = False

        cur_full_hidden_states = [None] * batch_size
        cur_full_input_ids = [None] * batch_size
        input_ids = batch.spec_info.verified_id
        hidden_states = batch.spec_info.hidden_states

        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        scores = None
        raw_out_loc = batch.out_cache_loc
        out_loc = raw_out_loc
        real_valid_id = None

        model_worker_batch = batch.get_model_worker_batch()
        if real_valid_id is None:
            a = batch.spec_info.accept_length.clone()
            a.fill_(self.server_args.speculative_num_draft_tokens)
            verified_indices = a.cumsum(dim=0).add_(-1)
            real_valid_id = input_ids[verified_indices]

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.input_ids = input_ids

        batch_size = len(batch.spec_info.accept_length)
        forward_batch.batch_size = batch_size
        seq_lens_sum = forward_batch.seq_lens.sum().item()
        batch.seq_lens_sum = seq_lens_sum
        forward_batch.seq_lens_sum = seq_lens_sum

        verified_length = self.server_args.speculative_num_draft_tokens
        verified_lengths = [verified_length] * batch_size
        batch_out_locs = out_loc.split(verified_lengths)

        device = input_ids.device
        start_positions = forward_batch.seq_lens - verified_length
        offsets = torch.arange(verified_length, device=device).repeat(batch_size)
        repeated_starts = torch.repeat_interleave(start_positions, verified_length)
        positions = repeated_starts + offsets

        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            self._draft_extend_with_cuda_graph(
                forward_batch,
                batch,
                hidden_states,
                positions,
                out_loc,
                input_ids,
                score_list,
                token_list,
                parents_list,
            )
        else:  # TODO: Fix and validate this implementation
            self._draft_extend_without_cuda_graph(
                forward_batch,
                batch,
                forward_batch.seq_lens,
                hidden_states,
                input_ids,
                batch_out_locs,
                verified_lengths,
                score_list,
                token_list,
                parents_list,
                cur_full_hidden_states,
                cur_full_input_ids,
            )

        parent_list, top_scores_index, draft_tokens = self._preprocess_tree_inputs(
            real_valid_id,
            score_list,
            token_list,
            parents_list,
            self.server_args.speculative_num_draft_tokens,
        )
        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            real_valid_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.server_args.speculative_num_draft_tokens,
        )

        ret = MhmtpVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.server_args.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

        ret.pre_out_locs = batch.spec_info.pre_out_locs
        ret.last_hidden_states = batch.spec_info.last_hidden_states
        ret.pre_hiddens = batch.spec_info.pre_hiddens
        ret.pre_verify_ids = batch.spec_info.pre_verify_ids
        batch.seq_lens = seq_len_bak
        batch.req_pool_indices = req_pool_bak
        return ret

    def _draft_extend_with_cuda_graph(
        self,
        forward_batch: ForwardBatch,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out_loc: torch.Tensor,
        input_ids: torch.Tensor,
        score_list: List[torch.Tensor],
        token_list: List[torch.Tensor],
        parents_list: List[torch.Tensor],
    ) -> None:
        """Draft extend using CUDA graph for performance."""
        batch_size = len(batch.spec_info.accept_length)
        forward_batch.spec_info.hidden_states = hidden_states
        forward_batch.spec_info.accept_length.fill_(
            self.server_args.speculative_num_draft_tokens
        )
        forward_batch.positions = positions
        forward_batch.out_cache_loc = out_loc
        forward_batch.input_ids = input_ids
        forward_batch.extend_seq_lens = batch.spec_info.accept_length
        forward_batch.extend_num_tokens = batch.spec_info.accept_length

        forward_batch.attn_backend = self.draft_attn_backend.attn_backends[0]
        forward_batch.mtp_index = 0

        output = self.cuda_graph_runner_for_draft_extend.replay(forward_batch)
        for i in range(self.server_args.speculative_num_steps):
            scores, tokens, parents = output[i]
            score_list.append(scores)
            token_list.append(tokens)
            parents_list.append(parents)

    def _draft_extend_without_cuda_graph(
        self,
        forward_batch: ForwardBatch,
        batch: ScheduleBatch,
        seq_lens: torch.Tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        batch_out_locs: List[torch.Tensor],
        verified_lengths: List[int],
        score_list: List[torch.Tensor],
        token_list: List[torch.Tensor],
        parents_list: List[torch.Tensor],
        cur_full_hidden_states: List,
        cur_full_input_ids: List,
    ) -> None:
        """
        Draft extend without CUDA graph (slower but more flexible).

        NOTE: This is a legacy implementation used as fallback when CUDA graph
        optimization is unavailable. Due to performance overhead without CUDA graphs,
        this code path has not been actively maintained and may have compatibility
        issues with recent model changes.

        TODO: Fix and validate this implementation:
        Ensure compatibility with multi-layer MTP models
        """
        scores = None
        batch_size = len(verified_lengths)
        verified_lengths_bak = verified_lengths.copy()

        for i in range(self.speculative_num_steps):
            updated_hidden_states = None
            updated_out_locs = None
            updated_input_ids = None
            updated_positions = None
            batch_hidden_states = hidden_states.split(verified_lengths)
            token_verified_lengths = [1] * batch_size

            batch.spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
            for j, vlen in enumerate(verified_lengths_bak):
                size = torch.tensor(vlen, device=input_ids.device)
                one_hidden_states = batch_hidden_states[j]
                batch_input_ids_split = input_ids.split(verified_lengths)
                one_input_ids = batch_input_ids_split[j]
                one_out_locs = batch_out_locs[j]
                offset = max(0, -vlen + i + 1)
                if offset:
                    size += offset
                    one_hidden_states = torch.cat(
                        [
                            batch.spec_info.last_hidden_states[i][j],
                            one_hidden_states,
                        ],
                        dim=0,
                    )
                    one_out_locs = torch.cat(
                        [
                            batch.spec_info.pre_out_locs[i][j][-offset:],
                            one_out_locs,
                        ]
                    )
                self.set_last_hidden_states(
                    batch.spec_info, one_hidden_states[-i - 1 : -i], i, j
                )
                self.set_last_out_loc(batch.spec_info, one_out_locs, i, j)
                updated_hidden_states = self.append_hidden_states(
                    updated_hidden_states, one_hidden_states
                )
                updated_out_locs = self.append_out_locs(updated_out_locs, one_out_locs)
                batch.spec_info.accept_length[j] = size
                verified_lengths[j] = size
                cur_full_input_ids[j] = self.append_input_ids(
                    cur_full_input_ids[j], one_input_ids
                )
                one_input_ids = cur_full_input_ids[j][-size:]
                updated_input_ids = self.append_input_ids(
                    updated_input_ids, one_input_ids
                )
                end_positions = forward_batch.seq_lens[j].unsqueeze(0)
                positions = self.build_sliding_positions(end_positions, size)
                updated_positions = self.append_positions(updated_positions, positions)
            hidden_states = updated_hidden_states
            input_ids = updated_input_ids
            out_loc = updated_out_locs
            positions = updated_positions

            forward_batch.spec_info.hidden_states = hidden_states
            forward_batch.positions = positions
            forward_batch.out_cache_loc = out_loc
            forward_batch.input_ids = input_ids
            forward_batch.extend_seq_lens = batch.spec_info.accept_length
            forward_batch.extend_num_tokens = batch.spec_info.accept_length
            forward_batch.mtp_index = i
            self.draft_attn_backend.attn_backends[i].init_forward_metadata(
                forward_batch
            )
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            logits_output = self.draft_model_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            hidden_states = logits_output.hidden_states
            self._detect_nan_if_needed(logits_output)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            hidden_states_bak = hidden_states
            hidden_states = hidden_states[
                torch.tensor(verified_lengths).cumsum(dim=0) - 1
            ]
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            num_groups = len(input_ids)
            group_size = len(forward_batch.input_ids) // num_groups
            input_id_parts = []
            for idx in range(num_groups):
                current_input = input_ids[idx]
                start_idx = idx * group_size
                end_idx = start_idx + group_size
                current_group = forward_batch.input_ids[start_idx:end_idx]
                current_group = torch.cat(
                    [current_group, current_input.unsqueeze(0)], dim=0
                )[-self.server_args.speculative_num_draft_tokens :]
                input_id_parts.append(current_group)
            final_input_ids = torch.cat(input_id_parts, dim=0)
            forward_batch.input_ids = final_input_ids
            hidden_states = hidden_states_bak
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        next_token_ids: torch.Tensor,
    ) -> LogitsProcessorOutput:
        """Forward draft extend from target model output."""
        batch_size = batch.batch_size()
        seq_lens = batch.seq_lens.tolist()
        hidden_states = logits_output.hidden_states
        input_hidden_states = hidden_states
        cur_full_hidden_states = [None] * batch_size
        cur_full_input_ids = list(batch.input_ids.split(seq_lens))

        batch.spec_info = MhmtpDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids,
        )
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        scores = None
        score_list = []
        token_list = []
        parents_list = []
        input_ids = next_token_ids
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )

        for i in range(self.speculative_num_steps):
            updated_input_ids = None

            if self.is_custom:
                batch.spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
                forward_batch.spec_info.hidden_states = hidden_states
            else:
                batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
                cur_full_hidden_states = self.append_hidden_states(
                    cur_full_hidden_states,
                    hidden_states,
                )
                forward_batch.spec_info.hidden_states = cur_full_hidden_states[-0:]

            for j in range(batch_size):
                cur_full_input_ids[j] = self.append_input_ids(
                    cur_full_input_ids[j],
                    input_ids[j].unsqueeze(0),
                )
                self.set_last_hidden_states(
                    batch.spec_info, hidden_states[-i - 1 : -i], i, j
                )
                self.set_last_out_loc(batch.spec_info, batch.out_cache_loc, i, j)

                one_input_ids = cur_full_input_ids[j][-seq_lens[j] :]
                updated_input_ids = self.append_input_ids(
                    updated_input_ids, one_input_ids
                )

            forward_batch.return_logprob = False
            forward_batch.mtp_index = i
            forward_batch.input_ids = updated_input_ids

            logits_output, _ = self.draft_model_runner.forward(forward_batch)

            self._detect_nan_if_needed(logits_output)
            assert isinstance(forward_batch.spec_info, MhmtpDraftInput)
            assert forward_batch.spec_info is batch.spec_info
            self.capture_for_decode(logits_output, forward_batch.spec_info)

            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            hidden_states_bak = logits_output.hidden_states
            topk_p = topk_p
            topk_index = topk_index
            hidden_states = hidden_states_bak[torch.tensor(seq_lens).cumsum(dim=0) - 1]
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            hidden_states = hidden_states_bak
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

        parent_list, top_scores_index, draft_tokens = self._preprocess_tree_inputs(
            next_token_ids,
            score_list,
            token_list,
            parents_list,
            self.server_args.speculative_num_draft_tokens,
        )
        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            next_token_ids,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.server_args.speculative_num_draft_tokens,
        )

        ret = MhmtpVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.server_args.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

        # Before verification, we need to concatenate some states to ensure the length
        # of the context window passed equals the number of MTP heads + 1. (for cuda graph)
        # If the number of generated tokens is insufficient, pad the beginning with
        # ids, hidden states, and cache locations from the previous stage.
        split_hiddens = torch.split(input_hidden_states, seq_lens, dim=0)
        split_verify_ids = torch.split(forward_batch.input_ids, seq_lens, dim=0)
        batch_size = len(split_hiddens)
        hidden_dim = split_hiddens[0].shape[1]
        size = self.server_args.speculative_num_draft_tokens
        ret.pre_hiddens = torch.empty(
            batch_size * size,
            hidden_dim,
            dtype=split_hiddens[0].dtype,
            device=split_hiddens[0].device,
        )
        ret.pre_verify_ids = torch.empty(
            batch_size * size,
            dtype=split_verify_ids[0].dtype,
            device=split_verify_ids[0].device,
        )
        for i, (hidden, verify_ids) in enumerate(zip(split_hiddens, split_verify_ids)):
            start_idx = i * size
            end_idx = start_idx + size

            for tensor_data, output_tensor in [
                (hidden, ret.pre_hiddens),
                (verify_ids, ret.pre_verify_ids),
            ]:
                last_four = (
                    tensor_data[-size:] if tensor_data.shape[0] >= size else tensor_data
                )
                if last_four.shape[0] == size:
                    output_tensor[start_idx:end_idx] = last_four
                else:
                    actual_len = last_four.shape[0]
                    output_tensor[start_idx : start_idx + actual_len] = last_four
                    output_tensor[start_idx + actual_len : end_idx].zero_()
        ret.pre_out_locs = batch.spec_info.pre_out_locs
        ret.last_hidden_states = batch.spec_info.last_hidden_states
        batch.spec_info = ret
        return logits_output


def load_token_map(token_map_path: str) -> torch.Tensor:
    """Load token map from file or Hugging Face hub.

    Args:
        token_map_path: Path to the token map file.

    Returns:
        Token map as a tensor.
    """
    if not os.path.exists(token_map_path):
        cache_dir = snapshot_download(
            os.path.dirname(token_map_path),
            ignore_patterns=["*.bin", "*.safetensors"],
        )
        token_map_path = os.path.join(cache_dir, os.path.basename(token_map_path))
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int32)
