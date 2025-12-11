import contextlib
import logging
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs,
    fill_accepted_out_cache_loc,
    fill_new_verified_id,
    select_top_k_tokens_tmp,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    detect_nan,
    draft_tp_context,
    generate_token_bitmask,
    load_token_map,
)
from sglang.srt.utils.common import (
    MultiprocessingSerializer,
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()

logger = logging.getLogger(__name__)


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class EagleDraftWorker(BaseDraftWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Set constant
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        with empty_context(), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            # Init draft worker
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner

        self.init_token_map()
        self.init_lm_head()

        # Init attention backend and cuda graphs
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.draft_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def init_token_map(self):
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if self.server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(self.server_args.speculative_token_map)
            self.server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners

        self.has_prefill_wrapper_verify = False
        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        Device2ExtendCudaGraphRunner = {
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
        }
        # Capture extend
        # FIXME cuda not support draft_extend capture
        if self.draft_extend_attn_backend and _is_npu:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    def draft(self, model_worker_batch: ModelWorkerBatch):
        draft_input: EagleDraftInput = model_worker_batch.spec_info
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            model_worker_batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Run draft
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch,
            )
        else:
            if (
                not forward_batch.forward_mode.is_idle()
                and self.speculative_num_steps > 1
            ):
                # Skip attention backend init for 1-step draft,
                # `draft_forward` only does sample in this case.
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        if model_worker_batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            model_worker_batch.seq_lens,
            model_worker_batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens_tmp(
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
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output, _ = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            )
            if self.server_args.enable_nan_detection:
                detect_nan(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        # Organize the results
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # Construct input_ids
        if not batch.forward_mode.is_idle():
            pt = 0
            for i, extend_len in enumerate(batch.extend_seq_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (input_ids[1:], next_token_ids[i].reshape(1))
                )
                pt += extend_len

        # Construct spec_info
        next_draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids,
            new_seq_lens=batch.seq_lens,
            # draft mode is same with decode mode, only 1 num token per batch
            num_tokens_per_batch=1,
            num_tokens_for_logprob_per_batch=1,
        )

        batch.spec_info = next_draft_input

        # Run forward
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
        logits_output, _ = self.draft_runner.forward(forward_batch)

        # Update spec_info for the next draft step
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        next_draft_input.topk_p, next_draft_input.topk_index = fast_topk(
            probs, self.topk, dim=-1
        )
        next_draft_input.hidden_states = logits_output.hidden_states
        return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ModelWorkerBatch, batch_result: GenerationBatchResult
    ):
        accept_index = batch_result.accept_index
        accept_lens = batch_result.accept_lens

        if accept_index is None:
            # Fallback to original dense behavior if accept_index is missing
            draft_input = EagleDraftInput(
                hidden_states=batch_result.logits_output.hidden_states,
                num_tokens_per_batch=self.speculative_num_steps + 1,
                num_tokens_for_logprob_per_batch=1,
            )
            select_index = (
                torch.arange(len(batch.seq_lens), device=self.device)
                * self.speculative_num_draft_tokens
                + batch_result.accept_lens
                - 1
            )
            with self.plan_stream_ctx:
                forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                    batch,
                    batch_result.next_token_ids,
                    self.speculative_num_draft_tokens,
                    self.draft_runner,
                    self.cuda_graph_runner_for_draft_extend,
                )
            if self.plan_stream:
                torch.get_device_module(self.device).current_stream().wait_stream(
                    self.plan_stream
                )
            can_cuda_graph = (
                self.cuda_graph_runner_for_draft_extend
                and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
            )
            if can_cuda_graph:
                draft_logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                    forward_batch
                )
            else:
                draft_logits_output, _ = self.draft_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                )
            draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
                select_index
            ]
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]
            probs = torch.softmax(draft_logits_output.next_token_logits, dim=-1)
            ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
            ret_hidden_states = draft_logits_output.hidden_states
            next_draft_input = batch_result.next_draft_input
            (
                next_draft_input.topk_p,
                next_draft_input.topk_index,
                next_draft_input.hidden_states,
            ) = (
                ret_topk_p,
                ret_topk_index,
                ret_hidden_states,
            )
            return

        # FULL FIX: Repack ALL tensors for draft extend to match accept_lens
        # This ensures seq_lens grows by accept_lens, not num_draft_tokens

        # Get accept_index for repacking all tensors
        flat_accept_index = accept_index.flatten()
        valid_mask = flat_accept_index != -1
        valid_indices = flat_accept_index[valid_mask]

        # Repack accepted tokens for draft extend

        # Repack hidden_states to only accepted positions
        full_hidden_states = batch_result.logits_output.hidden_states  # [bs * num_draft_tokens, H]
        repacked_hidden_states = full_hidden_states[valid_indices]  # [sum(accept_lens), H]

        # Repack input_ids (predict) - use sparse_predict + accept_index
        repacked_input_ids = batch_result.sparse_predict[valid_indices]  # [sum(accept_lens)]

        # Repack out_cache_loc to match repacked input_ids
        repacked_out_cache_loc = batch.out_cache_loc[valid_indices]
        batch.out_cache_loc = repacked_out_cache_loc

        draft_input = EagleDraftInput(
            hidden_states=repacked_hidden_states,  # Repacked [sum(accept_lens), H]
            num_tokens_per_batch=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_batch=1,
        )

        # select_index: For repacked tensor, pick the LAST token per request
        # With repacked layout: [req0_tok0, req0_tok1, req1_tok0, req1_tok1, req1_tok2, ...]
        # Last token for req i is at cumsum(accept_lens[:i+1]) - 1
        accept_lens_cumsum = torch.cumsum(accept_lens, dim=0)
        select_index = accept_lens_cumsum - 1  # Last accepted position per request

        # Prepare for draft extend using REPACKED tensors
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache_repacked(
                batch,
                repacked_input_ids,  # Repacked [sum(accept_lens)]
                accept_lens,         # Per-request accept lengths
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
            )

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        # Run draft extend batch (CUDA graphs disabled for variable sizes)
        draft_logits_output, _ = self.draft_runner.forward(
            forward_batch, skip_attn_backend_init=True
        )

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        draft_logits_output.hidden_states = draft_logits_output.hidden_states[
            select_index
        ]
        probs = torch.softmax(draft_logits_output.next_token_logits, dim=-1)
        ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
        ret_hidden_states = draft_logits_output.hidden_states

        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )


class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = EagleDraftWorker(
            server_args, gpu_id, tp_rank, dp_rank, moe_ep_rank, nccl_port, target_worker
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Target prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Draft prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        model_worker_batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                    )
                )
                return batch_output
        else:
            if model_worker_batch.spec_info is None:
                model_worker_batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=self.target_worker.model_config.hidden_size,
                    dtype=self.target_worker.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )
            with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                verify_input: EagleVerifyInput = self.draft_worker.draft(
                    model_worker_batch
                )
            assert verify_input.is_verify_input()
            model_worker_batch.spec_info = verify_input
            batch_output = self.verify(model_worker_batch)

            # FIX: Update req_to_token mapping to only include accepted slots
            # This is the V1 approach - update mapping instead of moving KV data
            if not model_worker_batch.forward_mode.is_idle():
                self._update_req_to_token_for_accepted(
                    model_worker_batch,
                    batch_output.accept_index,
                    batch_output.accept_lens,
                )

            with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.draft_worker._draft_extend_for_decode(
                    model_worker_batch, batch_output
                )
            return batch_output

    def verify(self, batch: ModelWorkerBatch):
        # Since batch.seq_lens is allocated in another stream, we need
        # record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        # Parse args
        verify_input: EagleVerifyInput = batch.spec_info
        verify_input.num_tokens_per_batch = self.speculative_num_steps + 1
        bs = len(batch.seq_lens)

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # Prepare grammar data on CPU if needed
        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrive_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrive_next_token.shape
            ).cpu()

        # Run target verify batch in the main compute stream (GPU compute)
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output

        # Generate vocab mask for constrained decoding
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrive_next_token.device)
                # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        # Sample
        if self.enable_nan_detection:
            detect_nan(logits_output)
        (
            predict,
            accept_length,
            accept_index,
        ) = verify_input.sample(batch, logits_output, vocab_mask)
        new_seq_lens = batch.seq_lens + accept_length
        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        if not batch.forward_mode.is_idle():
            all_verified_id = predict[accept_index]
            verified_id = torch.empty_like(accept_length, dtype=torch.int32)
            fill_new_verified_id[(bs,)](
                all_verified_id,
                accept_length,
                verified_id,
                self.speculative_num_steps + 1,
            )

            # Repack next_token_ids to only include accepted tokens
            # This ensures the scheduler receives a dense tensor of valid tokens
            flat_accept_index = accept_index.flatten()
            valid_mask = flat_accept_index != -1
            valid_indices = flat_accept_index[valid_mask]
            repacked_next_token_ids = predict[valid_indices]

            # Debug: Check if tree mode (scattered indices) or chain mode (contiguous indices)
            expected_chain_indices = []
            for i, acc_len in enumerate(accept_length.tolist()):
                expected_chain_indices.extend(range(i * self.speculative_num_draft_tokens,
                                                     i * self.speculative_num_draft_tokens + acc_len))
            is_tree_mode = set(valid_indices.tolist()) != set(expected_chain_indices)
            if is_tree_mode:
                print(f"[EAGLE3_DEBUG] Tree mode: accept_index={accept_index[0].tolist()[:5]}..., accept_len={accept_length.tolist()}")
        else:
            verified_id = torch.empty((0,), device=self.device, dtype=torch.int32)
            repacked_next_token_ids = predict

        # Construct the next draft input
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=repacked_next_token_ids,  # Repacked for scheduler
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
            accept_index=accept_index,
            sparse_predict=predict,  # Original sparse predict for draft extend
        )

    def _update_req_to_token_for_accepted(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
    ):
        """
        Update req_to_token mapping for tree mode with slot shifting.

        For tree mode, we need to:
        1. Update req_to_token[kv_committed:kv_committed+accept_len] = accepted slots
        2. Shift pre-allocated slots to fill the gap created by rejected tree slots
        3. NO freeing needed - rejected slots become "dead space" until request ends

        The key issue is slot DUPLICATION: after accepting scattered tree positions,
        the pre-alloc region still contains copies of some accepted slots. We must
        shift the pre-alloc slots to start right after accepted positions.

        Example:
          Before: req_to_token[100:126] = [A,B,C,D,...,M, N,O,P,...,Z]
                                           ↑-tree (13)-↑  ↑-prealloc-↑
          Accept positions [0, 2] → slots [A, C]
          After step 1: req_to_token[100:102] = [A, C]  ← correct
          BUT: req_to_token[102] still = C (unchanged!) ← DUPLICATE

          Step 2 shifts pre-alloc: req_to_token[102:115] = [N,O,P,...,Z]
          Now: no duplicates, next tree reads fresh slots
        """
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = len(batch.seq_lens)
        is_tree_mode = self.topk > 1
        tree_size = self.speculative_num_draft_tokens  # topk * num_steps + 1

        # accept_index already contains FLAT indices into batch tensors
        # e.g., [0, 2, 13, 14] for bs=2 where req 0 accepts tree positions 0,2
        #       and req 1 accepts tree positions 0,1 (flat indices 13, 14)
        flat_accept_index = accept_index.flatten()
        valid_mask = flat_accept_index != -1
        valid_indices = flat_accept_index[valid_mask]

        # Get the KV slots for accepted positions only
        accepted_out_cache_loc = batch.out_cache_loc[valid_indices]

        # For tree mode: FREE rejected slots (required to avoid memory leak)
        # In chain mode, rejected slots are naturally reused as pre-alloc
        # In tree mode, scattered acceptance leaves orphaned slots that must be freed
        if is_tree_mode:
            total_tree_slots = bs * tree_size
            all_indices = set(range(total_tree_slots))
            accepted_indices_set = set(valid_indices.tolist())
            rejected_indices = list(all_indices - accepted_indices_set)
            if rejected_indices:
                rejected_slots = batch.out_cache_loc[rejected_indices]
                self.token_to_kv_pool_allocator.free(rejected_slots)
                print(f"[EAGLE3_DEBUG tree_mode] bs={bs}, accept_lens={accept_length.tolist()}, freed={len(rejected_indices)} rejected slots")

        # Calculate new seq_lens after accepting tokens
        new_seq_lens = batch.seq_lens + accept_length

        # Step 1: Update req_to_token for accepted positions
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            new_seq_lens,
            accepted_out_cache_loc,
            bs,
        )

        # Step 2: For tree mode, shift pre-allocated slots to avoid duplication
        if is_tree_mode and batch.reqs is not None:
            self._shift_prealloc_slots(batch, accept_length, tree_size, new_seq_lens)

    def _shift_prealloc_slots(
        self,
        batch: ModelWorkerBatch,
        accept_length: torch.Tensor,
        tree_size: int,
        new_seq_lens: torch.Tensor,
    ):
        """
        Shift pre-allocated slots to fill the gap left by rejected tree slots.

        This prevents slot duplication where the same physical slot appears at
        multiple logical positions in req_to_token.

        Note: This currently uses CPU indexing which may cause a sync.
        TODO: Implement GPU-only version using Triton kernel for perf.
        """
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = len(batch.seq_lens)
        req_to_token = self.req_to_token_pool.req_to_token
        req_pool_indices = batch.req_pool_indices

        # Collect all pre-alloc slots across requests
        all_prealloc_slots = []
        prealloc_lengths = []

        for i in range(bs):
            req = batch.reqs[i]
            req_idx = req_pool_indices[i].item()
            seq_len = batch.seq_lens[i].item()
            kv_allocated = req.kv_allocated_len

            # Pre-alloc region: [seq_lens + tree_size : kv_allocated]
            prealloc_start = seq_len + tree_size
            prealloc_end = kv_allocated

            if prealloc_end > prealloc_start:
                # Read pre-alloc slots from req_to_token
                prealloc_slots = req_to_token[req_idx, prealloc_start:prealloc_end].clone()
                all_prealloc_slots.append(prealloc_slots)
                prealloc_lengths.append(len(prealloc_slots))
            else:
                prealloc_lengths.append(0)

        # Now write shifted pre-alloc slots
        if any(plen > 0 for plen in prealloc_lengths):
            # Flatten all pre-alloc slots
            if all_prealloc_slots:
                flat_prealloc = torch.cat(all_prealloc_slots)

                # Calculate destination positions: new_seq_lens[i] : new_seq_lens[i] + prealloc_len[i]
                dest_starts = new_seq_lens.clone()
                dest_ends = new_seq_lens + torch.tensor(
                    prealloc_lengths, device=new_seq_lens.device, dtype=new_seq_lens.dtype
                )

                # Write using the Triton kernel
                assign_req_to_token_pool_func(
                    req_pool_indices,
                    req_to_token,
                    dest_starts,
                    dest_ends,
                    flat_prealloc,
                    bs,
                )

                # Update kv_allocated_len on requests
                for i in range(bs):
                    accept_len = accept_length[i].item()
                    # New kv_allocated = old - (tree_size - accept_len)
                    # = old - tree_size + accept_len
                    # This shrinks kv_allocated by the number of rejected slots
                    rejected_count = tree_size - accept_len
                    batch.reqs[i].kv_allocated_len -= rejected_count

                print(f"[EAGLE3_DEBUG tree_mode] Shifted {sum(prealloc_lengths)} pre-alloc slots")

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
    ):
        """
        [DEPRECATED - BUGGY] Move accepted tokens to the target KV cache.

        Args:
            batch: The batch to run.
            accept_index: The index of the accepted tokens.
            accept_length: The length of the accepted tokens.
        """
        bs = len(batch.seq_lens)
        size = bs * self.speculative_num_draft_tokens

        tgt_cache_loc = torch.zeros(
            size,
            dtype=torch.int64,
            device=self.device,
        )
        accepted_out_cache_loc = torch.zeros(
            size, dtype=torch.int64, device=self.device
        )
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + accept_length,
            tgt_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        fill_accepted_out_cache_loc[(size,)](
            accept_index,
            batch.out_cache_loc,
            accepted_out_cache_loc,
            next_power_of_2(size),
        )
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            tgt_cache_loc, accepted_out_cache_loc
        )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.draft_worker.draft_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message
