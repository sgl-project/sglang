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
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
    assign_draft_cache_locs,
    fast_topk,
    select_top_k_tokens,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import empty_context, get_available_gpu_memory, is_cuda_available

if is_cuda_available():
    from sgl_kernel import segment_packbits

logger = logging.getLogger(__name__)


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    with disable_dp_size(), patch_tensor_parallel_group(tp_group):
        yield


class EAGLEWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.padded_static_len = self.speculative_num_steps + 1
        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override context length with target model's context length
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        with empty_context():
            super().__init__(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                server_args=server_args,
                nccl_port=nccl_port,
                dp_rank=dp_rank,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # EAGLE3 models don't share lm_head
            self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            self.hot_token_id = self.draft_model_runner.model.get_hot_token_id().to(
                embed.device
            )
        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(self.draft_model_runner.tp_group):
            self.init_attention_backend()
            self.init_cuda_graphs()

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners
        if self.server_args.attention_backend == "flashinfer":
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            self.draft_attn_backend = FlashInferMultiStepDraftBackend(
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend = None
            self.padded_static_len = self.speculative_num_steps + 1
            self.has_prefill_wrapper_verify = True
        elif self.server_args.attention_backend == "triton":
            from sglang.srt.layers.attention.triton_backend import (
                TritonMultiStepDraftBackend,
            )

            self.draft_attn_backend = TritonMultiStepDraftBackend(
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend = None
            self.padded_static_len = self.speculative_num_steps + 1
            self.has_prefill_wrapper_verify = False
        elif self.server_args.attention_backend == "flashinfer_mla":
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )

            self.draft_attn_backend = FlashInferMLAMultiStepDraftBackend(
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend = None
            self.padded_static_len = self.speculative_num_steps + 1
            self.has_prefill_wrapper_verify = True
        else:
            raise ValueError(
                f"EAGLE is not supportted in attention backend {self.server_args.attention_backend}"
            )

        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        # Capture draft
        tic = time.time()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.cuda_graph_runner = EAGLEDraftCudaGraphRunner(self)
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture draft cuda graph end. Time elapsed: {time.time() - tic:.2f} s. avail mem={after_mem:.2f} GB. mem usage={(before_mem - after_mem):.2f} GB."
        )

        # Capture extend
        if self.draft_extend_attn_backend:
            raise NotImplementedError()

    @property
    def draft_model_runner(self):
        return self.model_runner

    def forward_batch_speculative_generation(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, List[int], int, int]:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepeted,
            the batch id (used for overlap schedule), and number of accepeted tokens.
        """
        if batch.forward_mode.is_decode():
            with self.draft_tp_context(self.draft_model_runner.tp_group):
                spec_info, to_free_cache_loc = self.draft(batch)
            logits_output, verify_output, model_worker_batch = self.verify(
                batch, spec_info
            )

            # Free cache loc (we put it here to avoid synchronization and hide kernel launch overhead.)
            self.token_to_kv_pool_allocator.free(to_free_cache_loc)

            # If it is None, it means all requests are finished
            if batch.spec_info.verified_id is not None:
                with self.draft_tp_context(self.draft_model_runner.tp_group):
                    self.forward_draft_extend_after_decode(batch)
            return (
                logits_output,
                verify_output.verified_id,
                model_worker_batch.bid,
                sum(verify_output.accept_length_per_req_cpu),
            )
        elif batch.forward_mode.is_idle():
            model_worker_batch = batch.get_model_worker_batch()
            logits_output, next_token_ids, _ = (
                self.target_worker.forward_batch_generation(
                    ForwardBatch.init_new(
                        model_worker_batch, self.target_worker.model_runner
                    )
                )
            )
            return logits_output, next_token_ids, model_worker_batch.bid, 0, False
        else:
            logits_output, next_token_ids, bid = self.forward_target_extend(batch)
            with self.draft_tp_context(self.draft_model_runner.tp_group):
                self.forward_draft_extend(
                    batch, logits_output.hidden_states, next_token_ids
                )
            return logits_output, next_token_ids, bid, 0

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, List[int], int]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
            bid: The model batch ID. Used for overlap schedule.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, next_token_ids = self.target_worker.forward_batch_generation(
            model_worker_batch
        )
        return logits_output, next_token_ids, model_worker_batch.bid

    def draft(self, batch: ScheduleBatch):
        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        # Allocate cache locations
        out_cache_loc = batch.alloc_token_slots(
            num_seqs * self.topk * self.speculative_num_steps
        )
        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
        )
        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

        # Get forward batch
        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(
            forward_batch
        )
        if can_cuda_graph:
            score_list, token_list, parents_list = self.cuda_graph_runner.replay(
                forward_batch
            )
        else:
            # Initialize attention backend
            self.draft_attn_backend.init_forward_metadata(forward_batch)
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.draft_model_runner
            )
            # Run forward steps
            score_list, token_list, parents_list = self.draft_forward(forward_batch)

        ret = EagleVerifyInput.create(
            spec_info.verified_id,
            score_list,
            token_list,
            parents_list,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.server_args.speculative_num_draft_tokens,
        )
        return ret, out_cache_loc

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            out_cache_loc = out_cache_loc.view(forward_batch.batch_size, -1)
            forward_batch.out_cache_loc = out_cache_loc[
                :, self.topk * i : self.topk * (i + 1)
            ].flatten()
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_model_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            self._detect_nan_if_needed(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        return score_list, token_list, parents_list

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        spec_info.prepare_for_verify(batch)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec_info
        model_worker_batch = batch.get_model_worker_batch()
        logits_output, _ = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True
        )
        self._detect_nan_if_needed(logits_output)
        spec_info.hidden_states = logits_output.hidden_states
        res: EagleVerifyOutput = spec_info.verify(
            batch, logits_output, self.token_to_kv_pool_allocator
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepeted)
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepeted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepeted_indices]

        # Prepare the batch for the next draft forwards.
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = res.draft_input

        if batch.return_logprob:
            self.add_logprob_values(batch, res, logits_output)

        return logits_output, res, model_worker_batch

    def add_logprob_values(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
    ):
        # Extract args
        logits_output = res.logits_output
        top_logprobs_nums = batch.top_logprobs_nums
        token_ids_logprobs = batch.token_ids_logprobs
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits, dim=-1
        )
        batch_next_token_ids = res.verified_id
        num_tokens_per_req = [accept + 1 for accept in res.accept_length_per_req_cpu]

        # We should repeat top_logprobs_nums to match num_tokens_per_req.
        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)

        # Extract logprobs
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

        # Add output logprobs to the request
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

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: List[int],
    ):
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids,
        )
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.return_logprob = False
        logits_output = self.draft_model_runner.forward(forward_batch)
        self._detect_nan_if_needed(logits_output)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        self.capture_for_decode(logits_output, forward_batch.spec_info)

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        # Backup fileds that will be modified in-place
        seq_lens_backup = batch.seq_lens.clone()
        req_pool_indices_backup = batch.req_pool_indices
        accept_length_backup = batch.spec_info.accept_length
        return_logprob_backup = batch.return_logprob

        # Prepare metadata
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        batch.spec_info.prepare_extend_after_decode(
            batch,
            self.speculative_num_steps,
        )
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        batch.return_logprob = False
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )

        # Run
        logits_output = self.draft_model_runner.forward(forward_batch)

        self._detect_nan_if_needed(logits_output)
        self.capture_for_decode(logits_output, forward_batch.spec_info)

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = ForwardMode.DECODE
        batch.seq_lens = seq_lens_backup
        batch.req_pool_indices = req_pool_indices_backup
        batch.spec_info.accept_length = accept_length_backup
        batch.return_logprob = return_logprob_backup

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        draft_input.hidden_states = logits_output.hidden_states

    def _detect_nan_if_needed(self, logits_output: LogitsProcessorOutput):
        if self.enable_nan_detection:
            logits = logits_output.next_token_logits
            if torch.any(torch.isnan(logits)):
                logger.error("Detected errors during sampling! NaN in the logits.")
                raise ValueError("Detected errors during sampling! NaN in the logits.")


def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        cache_dir = snapshot_download(
            os.path.dirname(token_map_path),
            ignore_patterns=["*.bin", "*.safetensors"],
        )
        token_map_path = os.path.join(cache_dir, os.path.basename(token_map_path))
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int32)
