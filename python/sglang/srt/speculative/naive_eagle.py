import logging
import os
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
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
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
    assign_req_to_token_pool,
    create_draft_kv_indices,
)
from sglang.srt.speculative.navie_eagle_cuda_graph_runner import (
    NaiveEAGLECudaGraphRunner,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_cuda,
    next_power_of_2,
)

if is_cuda():
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

logger = logging.getLogger(__name__)


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    with disable_dp_size(), patch_tensor_parallel_group(tp_group):
        yield


import triton
import triton.language as tl


@triton.jit
def create_extend_spec_info(
    seq_len,
    accept_len,
    accept_len_cum,
    positions,
    accept_len_upper: tl.constexpr,  # 1
):
    pid = tl.program_id(axis=0)
    offset = 0 if pid == 0 else tl.load(accept_len_cum + pid - 1)
    seq_length = tl.load(seq_len + pid)
    accept_length = tl.load(accept_len + pid)  # 1
    positions_ptr = positions + offset
    data = tl.arange(0, accept_len_upper)
    mask = data < accept_length
    tl.store(positions_ptr + data, seq_length - accept_length + data, mask)


class NaiveEagleWorker(TpModelWorker):

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
        self.page_size = server_args.page_size
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
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,
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
            self.requests_all_greedy = server_args.requests_all_greedy
            self.init_cuda_graphs()

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        # self.server_args.disable_cuda_graph = True
        if self.server_args.disable_cuda_graph:
            logger.info("Eagle Cuda Graph is disabled")
            return

        # Capture draft
        tic = time.time()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.cuda_graph_runner = NaiveEAGLECudaGraphRunner(self)
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture draft cuda graph end. Time elapsed: {time.time() - tic:.2f} s. avail mem={after_mem:.2f} GB. mem usage={(before_mem - after_mem):.2f} GB."
        )

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
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        if batch.forward_mode.is_decode():
            return self.draft(batch)
        elif batch.forward_mode.is_idle():
            model_worker_batch = batch.get_model_worker_batch()
            logits_output, next_token_ids, _ = (
                self.target_worker.forward_batch_generation(model_worker_batch)
            )

            return logits_output, next_token_ids, model_worker_batch.bid, 0, False
        else:
            logits_output, next_token_ids, bid = self.forward_target_extend(batch)
            with self.draft_tp_context(self.draft_model_runner.tp_group):
                self.forward_draft_extend(
                    batch, logits_output.hidden_states, next_token_ids
                )
            return logits_output, next_token_ids, bid, 0, False

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
        logits_output, next_token_ids, _ = self.target_worker.forward_batch_generation(
            model_worker_batch
        )
        return logits_output, next_token_ids, model_worker_batch.bid

    def forward_draft_extend_after_decode(
        self, forward_batch: ForwardBatch, accept_index
    ):
        # Prepare metadata
        forward_batch.forward_mode = ForwardMode.NAIVE_DRAFT_EXTEND
        forward_batch.spec_info.prepare_extend_after_decode(
            forward_batch,
            1,
        )
        forward_batch.spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
        forward_batch.return_logprob = False
        forward_batch.req_to_token_pool = self.draft_model_runner.req_to_token_pool
        forward_batch.token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        forward_batch.attn_backend = self.draft_model_runner.attn_backend
        forward_batch.positions = forward_batch.spec_info.positions
        # Run
        logits_output, _ = self.draft_model_runner.forward(forward_batch)

        last = accept_index[:, 1]
        first = accept_index[:, 0]
        save_index = torch.where(last != -1, last, first)
        logits_output.hidden_states = logits_output.hidden_states[save_index]
        logits_output.next_token_logits = logits_output.next_token_logits[save_index]

        return logits_output

    def draft(self, batch: ScheduleBatch):
        num_seqs = batch.batch_size()
        draft_input_spec_info = batch.spec_info
        if self.page_size == 1:
            batch.out_cache_loc = batch.alloc_token_slots(
                num_seqs * 2,
                backup_state=False,  # hard code, 1 for target preill, 1 for draft
            )
            end_offset = batch.seq_lens + 2  # assign 2 tokens
            assign_req_to_token_pool[(num_seqs,)](
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                batch.seq_lens,
                end_offset,
                batch.out_cache_loc,
                batch.req_to_token_pool.req_to_token.shape[1],
                next_power_of_2(num_seqs),
            )
        else:
            raise NotImplementedError("TODO: Page size > 1 not supported yet")

        batch.input_ids = torch.column_stack(
            (batch.output_ids, draft_input_spec_info.topk_index.squeeze(1))
        ).flatten()
        positions = torch.column_stack((batch.seq_lens, batch.seq_lens + 1)).flatten()

        batch.spec_info = EagleVerifyInput(
            draft_token=batch.input_ids,
            custom_mask=None,
            positions=positions,
            retrive_index=None,
            retrive_next_token=None,
            retrive_next_sibling=None,
            retrive_cum_len=None,
            draft_token_num=2,
            spec_steps=1,
            topk=1,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_cpu=None,
            seq_lens_sum=None,
        )

        model_worker_batch = batch.get_model_worker_batch()

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.target_worker.model_runner
        )
        forward_batch.forward_mode = ForwardMode.NAIVE_TARGET_VERIFY

        can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(
            forward_batch
        )
        if can_cuda_graph:
            forward_batch.spec_info_topk_index = draft_input_spec_info.topk_index
            forward_batch.spec_info_topk_p = draft_input_spec_info.topk_p
            (
                logits_output,
                next_token_ids,
                accept_index,
                draft_logits_output,
                draft_input,
            ) = self.cuda_graph_runner.replay(forward_batch)
            forward_batch.input_ids = next_token_ids
        else:
            draft_token_num = 2
            kv_indptr = torch.empty(
                size=[1 + draft_token_num * num_seqs], dtype=torch.int32, device="cuda"
            )
            kv_indices = torch.empty(
                size=[
                    forward_batch.seq_lens_sum * draft_token_num
                    + (draft_token_num + 1) * num_seqs
                ],
                dtype=torch.int32,
                device="cuda",
            )
            req_to_token = forward_batch.req_to_token_pool.req_to_token
            create_draft_kv_indices[(num_seqs,)](
                kv_indptr,
                kv_indices,
                forward_batch.req_pool_indices,
                req_to_token,
                forward_batch.seq_lens + 2,
                2,
                req_to_token.shape[-1],
                next_power_of_2(num_seqs),
            )
            forward_batch.spec_info.kv_indptr = kv_indptr
            forward_batch.spec_info.kv_indices = kv_indices

            logits_output, next_token_ids, accept_index = self.draft_forward_and_verify(
                forward_batch,
                num_seqs,
                draft_input_spec_info.topk_p,
                draft_input_spec_info.topk_index,
            )

            # NOTE: We do not assign draft model here, because the res here is same as target model's assign

            # we pass accept length to 1 of all reqs cause we extend all tokens anyway.
            accept_length_for_draft_extend = torch.ones(
                (num_seqs,), dtype=torch.int32, device="cuda"
            )
            accept_length_cpu_for_draft_extend = accept_length_for_draft_extend.tolist()

            # here, we extend draft tokens anyway cause we want to adopt to cuda graph.
            draft_input = EagleDraftInput()
            draft_input.hidden_states = logits_output.hidden_states
            draft_input.accept_length = accept_length_for_draft_extend
            draft_input.accept_length_cpu = accept_length_cpu_for_draft_extend
            draft_input.verified_id = next_token_ids
            draft_input.seq_lens_for_draft_extend = forward_batch.seq_lens + (
                accept_length_for_draft_extend + 1
            )
            draft_input.req_pool_indices_for_draft_extend = (
                forward_batch.req_pool_indices
            )
            forward_batch.spec_info = draft_input
            draft_logits_output = self.forward_draft_extend_after_decode(
                forward_batch, accept_index
            )

        accept_length = torch.zeros((num_seqs,), dtype=torch.int32, device="cuda")
        torch.where(
            accept_index[:, 1] != -1,
            torch.tensor(1, dtype=accept_index.dtype, device=accept_index.device),
            torch.tensor(0, dtype=accept_index.dtype, device=accept_index.device),
            out=accept_length,
        )

        self._detect_nan_if_needed(draft_logits_output)
        self.capture_for_decode(draft_logits_output, draft_input)
        batch.spec_info = draft_input

        batch.seq_lens_sum = forward_batch.seq_lens_sum
        batch.input_ids = forward_batch.input_ids
        batch.forward_mode = ForwardMode.DECODE

        accept_length_cpu = accept_length.tolist()
        batch.spec_info.accept_length = accept_length
        batch.spec_info.accept_length_cpu = accept_length_cpu
        batch.seq_lens.add_(accept_length + 1)
        batch.seq_lens_sum = batch.seq_lens.sum().item()

        batch.extend_lens = [x + 1 for x in accept_length_cpu]
        batch.extend_num_tokens = forward_batch.extend_num_tokens
        batch.extend_num_tokens = sum(batch.extend_lens)

        accept_index_viewd = accept_index[accept_index != -1]
        verified_id = next_token_ids[accept_index_viewd]
        logits_output.next_token_logits = logits_output.next_token_logits[
            accept_index_viewd
        ]
        logits_output.hidden_states = logits_output.hidden_states[accept_index_viewd]

        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        new_accept_index = []
        unfinished_index = []
        has_finished = False
        accept_index_cpu = accept_index.tolist()
        next_token_ids_cpu = next_token_ids.tolist()
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            new_accept_index_ = []
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                id = next_token_ids_cpu[idx]
                # if not found_finished:
                req.output_ids.append(id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    accept_index[i, j + 1 :] = -1
                    break
                else:
                    new_accept_index_.append(idx)
            if not req.finished():
                new_accept_index.extend(new_accept_index_)
                unfinished_index.append(i)
            req.spec_verify_ct += 1

        if has_finished:
            accept_length = (accept_index != -1).sum(dim=1) - 1

        evict_mask = torch.full((num_seqs * 2,), True, dtype=torch.bool)
        evict_mask[accept_index[accept_index != -1]] = False
        if self.page_size != 1:
            # TODO: align_evict_mask_to_page_size, see eagle_utils.py/align_evict_mask_to_page_size
            pass

        self.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])

        cumsum = torch.cumsum(batch.spec_info.accept_length + 1, dim=0)
        output_idx = cumsum - 1
        output_ids = verified_id[output_idx]

        if not has_finished:
            batch.input_ids = batch.input_ids[accept_index_viewd]
            batch.out_cache_loc = batch.out_cache_loc[accept_index_viewd]
        else:
            if len(new_accept_index) > 0:
                new_accept_index = torch.tensor(new_accept_index, device="cuda")
                unfinished_index_device = torch.tensor(unfinished_index, device="cuda")
                batch.spec_info.accept_length_cpu = [
                    accept_length_cpu[i] for i in unfinished_index
                ]
                batch.spec_info.accept_length = accept_length[unfinished_index_device]
                batch.spec_info.topk_index = batch.spec_info.topk_index[
                    unfinished_index_device
                ]
                batch.spec_info.topk_p = batch.spec_info.topk_p[unfinished_index_device]
                batch.input_ids = batch.input_ids[new_accept_index]
            batch.out_cache_loc = batch.out_cache_loc[new_accept_index]

        return (
            logits_output,
            output_ids,
            model_worker_batch.bid,
            sum(accept_length_cpu),
            can_cuda_graph,
        )

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
        logits_output, _ = self.draft_model_runner.forward(forward_batch)
        self._detect_nan_if_needed(logits_output)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        self.capture_for_decode(logits_output, forward_batch.spec_info)

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, 1, dim=-1)

        draft_input.hidden_states = logits_output.hidden_states

    def _detect_nan_if_needed(self, logits_output: LogitsProcessorOutput):
        if self.enable_nan_detection:
            logits = logits_output.next_token_logits
            if torch.any(torch.isnan(logits)):
                logger.error("Detected errors during sampling! NaN in the logits.")
                raise ValueError("Detected errors during sampling! NaN in the logits.")

    def draft_forward_and_verify(
        self, forward_batch, bs, draft_top_k_p, draft_topk_index
    ):
        logits_output, _ = self.target_worker.model_runner.forward(forward_batch)

        accept_index = torch.full((bs, 2), -1, dtype=torch.int32, device="cuda")

        indices = torch.arange(bs, device="cuda", dtype=torch.int32)
        accept_index[:, 0] = indices * 2
        if forward_batch.sampling_info.is_all_greedy:
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            _, token_indices = fast_topk(probs, topk=1, dim=-1)
            next_token_ids = token_indices.squeeze(-1)
            draft_token = forward_batch.input_ids[2 * indices + 1]
            target_token = next_token_ids[2 * indices]

            mask = draft_token == target_token
            accept_index[:, 1] = torch.where(mask, 2 * indices + 1, accept_index[:, 1])
        else:
            # apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                forward_batch.sampling_info.temperatures, 2, dim=0
            )  # (bs * draft_token_num, 1)

            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )  # (bs * draft_token_num, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(forward_batch.sampling_info.top_ks, 2, dim=0),
            )  # (bs * draft_token_num, vocab_size)
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(forward_batch.sampling_info.top_ps, 2, dim=0),
            )

            target_verify_probs = target_probs[indices * 2]
            coins = torch.rand((bs), dtype=torch.float32, device="cuda")
            draft_p = draft_top_k_p.squeeze()

            target_p = torch.gather(
                target_verify_probs, dim=1, index=draft_topk_index
            ).squeeze(1)
            mask = coins < torch.min(
                torch.tensor([1], device=target_verify_probs.device),
                target_p / draft_p,
            )
            accept_index[:, 1] = torch.where(mask, 2 * indices + 1, -1)
            # prepare next_token_ids
            next_token_ids = torch.multinomial(target_probs, num_samples=1).squeeze(-1)
        return logits_output, next_token_ids, accept_index


def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        cache_dir = snapshot_download(
            os.path.dirname(token_map_path),
            ignore_patterns=["*.bin", "*.safetensors"],
        )
        token_map_path = os.path.join(cache_dir, os.path.basename(token_map_path))
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int32)
