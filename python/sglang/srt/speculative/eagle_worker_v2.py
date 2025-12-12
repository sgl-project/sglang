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

        # OPTION 5: Don't transfer accept_lens to CPU - keep everything on GPU!
        # The prepare_for_extend_to_fill_draft_kvcache_repacked function will use
        # repacked_input_ids.shape[0] for extend_num_tokens (no sync needed)

        # Prepare for draft extend using REPACKED tensors
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache_repacked(
                batch,
                repacked_input_ids,  # Repacked [sum(accept_lens)]
                accept_lens,         # Per-request accept lengths (GPU)
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
                accept_lens_cpu=None,  # Option 5: No CPU transfer!
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

            # ============================================================================
            # FIX FOR TREE MODE (EAGLE3 with topk > 1):
            # ============================================================================
            #
            # PROBLEM: V2's overlap mode pre-allocates KV cache slots and writes them
            # to req_to_token BEFORE verification. When tree mode accepts scattered
            # positions (e.g., [0, 2, 5] out of [0..21]), the req_to_token mapping
            # has incorrect entries that cause:
            #   1. Slot duplication - same physical slot at multiple logical positions
            #   2. Memory leak - rejected slots never returned to free pool
            #
            # SOLUTION: After verification, update req_to_token to only map accepted
            # positions, free rejected slots, and shift pre-allocated slots.
            #
            # This function is called AFTER verify() returns, and BEFORE
            # _draft_extend_for_decode() runs. The scheduler will later read the
            # repacked next_token_ids from GenerationBatchResult.
            #
            # CONNECTION TO SCHEDULER (scheduler_output_processor_mixin.py):
            #   - Worker returns repacked dense next_token_ids: shape [sum(accept_lens)]
            #   - Scheduler's _resolve_spec_overlap_token_ids() extracts per-request
            #     tokens using cumulative offsets, NOT stride-based indexing
            #   - Worker modifies batch.reqs[i].kv_allocated_len which scheduler sees
            #     (same object reference via get_model_worker_batch())
            # ============================================================================
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

            # ====================================================================
            # REPACK next_token_ids: Convert sparse to dense tensor
            # ====================================================================
            #
            # PROBLEM: The predict tensor is SPARSE - it has shape [bs * tree_size]
            # but only positions in accept_index contain valid tokens. The rest
            # contain garbage or uninitialized values.
            #
            # Example (bs=2, num_steps=5, topk=10, tree_size=32):
            #   predict shape: [64]  (2 * 32)
            #   predict = [tok0, ?, tok2, ?, ?, ?, ?, tok7, ..., tok15, ?, ..., tok32, ?, tok34, ?, ..., tok39, ...]
            #              ↑       ↑                    ↑          ↑             ↑         ↑              ↑
            #              valid   valid              valid      valid        valid      valid         valid
            #
            #   accept_index = [[0, 2, 7, 15, -1, -1],     # req 0: accept 4 tokens
            #                   [32, 34, 39, -1, -1, -1]]  # req 1: accept 3 tokens
            #   valid_indices = [0, 2, 7, 15, 32, 34, 39]
            #   repacked_next_token_ids = [tok0, tok2, tok7, tok15, tok32, tok34, tok39]
            #                             shape: [sum(accept_lens)] = [7]
            #
            # SOLUTION: Repack into dense tensor containing only valid tokens.
            # This is what the scheduler expects in _resolve_spec_overlap_token_ids().
            #
            # CONNECTION TO SCHEDULER:
            #   The scheduler uses CUMULATIVE OFFSET extraction:
            #     offset = 0
            #     for i, req in enumerate(batch.reqs):
            #         tokens = next_token_ids[offset : offset + accept_lens[i]]
            #         offset += accept_lens[i]
            #
            #   So repacked [tok0, tok2, tok7, tok22, tok24] with accept_lens=[3,2]:
            #     req 0: tokens[0:3] = [tok0, tok2, tok7]
            #     req 1: tokens[3:5] = [tok22, tok24]
            # ====================================================================
            flat_accept_index = accept_index.flatten()  # [bs * (num_steps + 1)]
            valid_mask = flat_accept_index != -1
            valid_indices = flat_accept_index[valid_mask]  # [sum(accept_lens)]
            repacked_next_token_ids = predict[valid_indices]  # [sum(accept_lens)]

            # ================================================================
            # DEBUG BLOCK: Detect and log tree mode (can be removed for production)
            # ================================================================
            # This entire block is ONLY for debug logging. The is_tree_mode
            # variable is not used anywhere else - only for the print statement.
            # Chain mode: accept_index values are contiguous per request [0,1,2], [32,33], etc.
            # Tree mode: accept_index values are scattered [0,2,7], [32,34], etc.
            #
            # ⚠️ DEBUG SYNCS: These .tolist() calls add ~0.5ms latency total
            # TODO: Remove this entire block for production (no functional impact)
            # ================================================================
            expected_chain_indices = []
            for i, acc_len in enumerate(accept_length.tolist()):  # GPU→CPU sync (debug only)
                expected_chain_indices.extend(range(i * self.speculative_num_draft_tokens,
                                                     i * self.speculative_num_draft_tokens + acc_len))
            is_tree_mode = set(valid_indices.tolist()) != set(expected_chain_indices)  # GPU→CPU sync (debug only)
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
        Update req_to_token mapping after tree verification to handle scattered acceptance.

        ============================================================================
        ⚠️ CPU-GPU SYNC WARNING - This function introduces multiple syncs!
        ============================================================================

        SYNCS IN THIS FUNCTION:
          1. valid_indices.tolist() - GPU tensor → CPU list for set operations
          2. token_to_kv_pool_allocator.free() - CPU-based allocator requires CPU tensor

        SYNCS IN _shift_prealloc_slots() (called from here):
          3. req_pool_indices[i].item() - O(bs) syncs in loop
          4. batch.seq_lens[i].item() - O(bs) syncs in loop
          5. accept_length[i].item() - O(bs) syncs in loop

        ESTIMATED LATENCY: ~1-5ms per decode iteration for tree mode
        (Chain mode avoids all of this!)

        See "CPU-GPU SYNC ANALYSIS" section in eagle3_v2_tree_flow_diagram.md
        for detailed analysis and removal strategies.

        ============================================================================
        CRITICAL INSIGHT (from debugging - see eagle3_debug_summary.md)
        ============================================================================

        `accept_index` from `verify_tree_greedy` kernel already contains FLAT indices!

        Example for bs=2, tree_size=32:
          - Values are like [0, 2, 7, 15, 32, 34, 39] (FLAT)
          - NOT per-request positions [0, 2, 7, 15, 0, 2, 7] (WRONG!)

        This is because verify_tree_greedy computes:
          for i in range(bs):
              indices.extend(range(i * tree_size, i * tree_size + acc_len))

        Do NOT add batch offsets - the indices are already correctly offset!
        (A previous bug attempt added `i * tree_size + pos` which caused CUDA assert)

        ============================================================================
        BACKGROUND: V2 Overlap Mode and Tree Speculative Decoding
        ============================================================================

        V2's overlap mode pre-allocates KV cache slots in prepare_for_decode() and
        writes them to req_to_token BEFORE the actual verification happens. This
        works perfectly for CHAIN mode (topk=1) where acceptance is always
        contiguous from position 0 (e.g., accept [0,1,2] out of [0,1,2,3,4,5,6]).

        However, TREE mode (topk>1) has SCATTERED acceptance patterns:
          - Tree structure: position 0 → topk children → topk*topk grandchildren...
          - Acceptance can be any valid path: [0, 2, 5, 11] (scattered indices)
          - Rejected positions [1, 3, 4, 6, 7, 8, 9, 10, 12...] leave "holes"

        ============================================================================
        TENSOR SHAPES AND DATA FLOW (Example: bs=2, num_steps=5, topk=10, tree_size=32)
        ============================================================================

        CRITICAL: tree_size = speculative_num_draft_tokens (configured separately!)
        It is NOT computed from topk * num_steps. In this example:
          - topk * num_steps = 10 * 5 = 50, but tree_size = 32 (configured)

        Input tensors:
          - batch.out_cache_loc: [bs * tree_size] = [2 * 32] = [64]
            Contains KV cache slot IDs allocated for this verify iteration
            Layout: [req0_slot0, req0_slot1, ..., req0_slot31, req1_slot0, ..., req1_slot31]

          - accept_index: [bs, max_accept_len] = [bs, num_steps+1] = [2, 6]
            Contains FLAT indices into out_cache_loc for accepted positions
            Example row: [0, 2, 7, 15, -1, -1] means accept flat positions 0, 2, 7, 15
            Value -1 indicates unused slot (fewer than max accepted)

            CRITICAL: These are already FLAT indices (0..31 for req 0,
            32..63 for req 1, etc.), NOT per-request positions!

          - accept_length: [bs] = [2]
            Number of accepted tokens per request (including bonus token)
            Example: [4, 3] means req 0 accepted 4 tokens, req 1 accepted 3 tokens

        ============================================================================
        WHY CHAIN MODE DOESN'T NEED THIS FIX
        ============================================================================

        Chain mode (topk=1):
          - Acceptance is prefix-contiguous: [0, 1, 2] then [3, 4, 5, 6] rejected
          - Rejected slots naturally form the "pre-alloc" region for next iteration
          - req_to_token layout after accept 3: [committed..., A, B, C, D, E, F, G, ...]
                                                              ↑accepted↑ ↑─pre-alloc─↑
          - Next iteration reads pre-alloc starting at position 3 → gets D, E, F, G
          - No slots are orphaned; all are tracked in contiguous req_to_token range

        Tree mode (topk>1):
          - Acceptance is scattered: [0, 2, 7, 15] accepted, [1, 3, 4, 5, 6, 8...31] rejected
          - We MUST update req_to_token to only contain accepted slots
          - Rejected slots become ORPHANED (not in req_to_token, not in free pool)
          - Memory leak if we don't explicitly free rejected slots!

        ============================================================================
        THE THREE-STEP FIX
        ============================================================================

        Step 1: Extract accepted slots and update req_to_token mapping
          - Use accept_index to gather accepted slots from out_cache_loc
          - Write accepted slots to req_to_token[seq_lens : seq_lens + accept_len]

        Step 2: Free rejected tree slots (tree mode only)
          - Compute rejected indices: all_indices - accepted_indices
          - Return rejected slots to allocator's free pool
          - This prevents memory leak

        Step 3: Shift pre-allocated slots (tree mode only)
          - V2 pre-allocates 2*tree_size slots per iteration
          - Pre-alloc region: req_to_token[seq_lens + tree_size : kv_allocated]
          - After accepting only N tokens, we need to shift pre-alloc to start
            at position seq_lens + N, not seq_lens + tree_size
          - Also update kv_allocated_len -= (tree_size - accept_len)

        ============================================================================
        CONNECTION TO SCHEDULER
        ============================================================================

        File: scheduler_output_processor_mixin.py
        Function: _resolve_spec_overlap_token_ids()

        The scheduler receives GenerationBatchResult with:
          - next_token_ids: REPACKED dense tensor [sum(accept_lens)]
            NOT the original sparse tensor [bs * tree_size]!
          - accept_lens: [bs] number of accepted tokens per request

        The scheduler extracts per-request tokens using CUMULATIVE OFFSETS:
          offset = 0
          for i, req in enumerate(batch.reqs):
              tokens = next_token_ids[offset : offset + accept_lens[i]]
              offset += accept_lens[i]

        This matches our repacking in verify():
          repacked_next_token_ids = predict[valid_indices]  # shape [sum(accept_lens)]

        The scheduler also sees our modification to kv_allocated_len because
        batch.reqs is the SAME object reference (from get_model_worker_batch).

        ============================================================================
        """
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = len(batch.seq_lens)
        is_tree_mode = self.topk > 1
        # tree_size = speculative_num_draft_tokens (configured separately, not derived!)
        # Example: num_steps=5, topk=10, but tree_size=32 (config [5, 10, 32])
        tree_size = self.speculative_num_draft_tokens

        # ========================================================================
        # STEP 0: Extract accepted positions from accept_index
        # ========================================================================
        # accept_index shape: [bs, num_steps + 1] e.g., [2, 6] for num_steps=5
        # Values are FLAT indices into out_cache_loc, or -1 for unused
        #
        # Example for bs=2, tree_size=32:
        #   accept_index = [[0, 2, 7, 15, -1, -1],  # req 0: accept flat positions 0, 2, 7, 15
        #                   [32, 34, 39, -1, -1, -1]] # req 1: accept flat positions 32, 34, 39
        #   flat_accept_index = [0, 2, 7, 15, -1, -1, 32, 34, 39, -1, -1, -1]
        #   valid_mask = [T, T, T, T, F, F, T, T, T, F, F, F]
        #   valid_indices = [0, 2, 7, 15, 32, 34, 39]  # shape [sum(accept_lens)] = [7]
        flat_accept_index = accept_index.flatten()  # [bs * (num_steps + 1)] = [2 * 6] = [12]
        valid_mask = flat_accept_index != -1
        valid_indices = flat_accept_index[valid_mask]  # [sum(accept_lens)] = [7]

        # ========================================================================
        # STEP 1: Gather accepted KV cache slots
        # ========================================================================
        # out_cache_loc shape: [bs * tree_size] e.g., [2 * 32] = [64]
        # Contains actual KV cache slot IDs (not positions)
        #
        # Example for bs=2, tree_size=32:
        #   out_cache_loc = [1001, 1002, 1003, ..., 1032,  # req 0's 32 slots (indices 0-31)
        #                   2001, 2002, 2003, ..., 2032]  # req 1's 32 slots (indices 32-63)
        #   valid_indices = [0, 2, 7, 15, 32, 34, 39]
        #   accepted_out_cache_loc = [1001, 1003, 1008, 1016, 2001, 2003, 2008]  # shape [7]
        accepted_out_cache_loc = batch.out_cache_loc[valid_indices]

        # ========================================================================
        # STEP 2 (TREE MODE ONLY): Free rejected slots to avoid memory leak
        # ========================================================================
        # Chain mode doesn't need this because rejected slots form the pre-alloc
        # region and are naturally reused. Tree mode has scattered acceptance,
        # leaving orphaned slots that would leak memory if not freed.
        #
        # Example for bs=2, tree_size=32, accept_lens=[4, 3]:
        #   total_tree_slots = 2 * 32 = 64
        #   all_indices = {0, 1, 2, ..., 63}
        #   accepted_indices_set = {0, 2, 7, 15, 32, 34, 39}
        #   rejected_indices = [1, 3, 4, 5, 6, 8, ..., 31, 33, 35, ..., 63]
        #   len(rejected_indices) = 64 - 7 = 57 slots to free
        if is_tree_mode:
            total_tree_slots = bs * tree_size
            all_indices = set(range(total_tree_slots))
            # ================================================================
            # PERFORMANCE WARNING: GPU-CPU syncs here add ~1-5ms latency!
            # ================================================================
            # Operations that cause sync:
            #   1. valid_indices.tolist() - GPU tensor to CPU list
            #   2. self.token_to_kv_pool_allocator.free() - CPU-based allocator
            #
            # Chain mode (topk=1) avoids this by having contiguous acceptance,
            # so rejected slots naturally become pre-alloc (no explicit free).
            #
            # Tree mode REQUIRES explicit free due to scattered acceptance.
            # This is an unavoidable cost for tree mode correctness.
            #
            # TODO: Consider GPU-only allocator or batched deferred free
            # ================================================================
            # ⚠️ SYNC #1: GPU→CPU transfer for set operations
            accepted_indices_set = set(valid_indices.tolist())  # GPU→CPU sync!
            rejected_indices = list(all_indices - accepted_indices_set)
            if rejected_indices:
                rejected_slots = batch.out_cache_loc[rejected_indices]
                # ⚠️ SYNC #2: CPU-based allocator requires sync
                # The allocator.free() internally transfers tensor to CPU
                self.token_to_kv_pool_allocator.free(rejected_slots)  # CPU allocator sync!
                print(f"[EAGLE3_DEBUG tree_mode] bs={bs}, accept_lens={accept_length.tolist()}, freed={len(rejected_indices)} rejected slots")

        # ========================================================================
        # STEP 3: Update req_to_token mapping with accepted slots
        # ========================================================================
        # Before: req_to_token[req_idx, seq_lens:seq_lens+tree_size] = all tree slots
        # After:  req_to_token[req_idx, seq_lens:seq_lens+accept_len] = accepted slots only
        #
        # The assign_req_to_token_pool_func Triton kernel writes:
        #   for each request i:
        #     req_to_token[req_pool_indices[i], seq_lens[i]:new_seq_lens[i]] = slots
        new_seq_lens = batch.seq_lens + accept_length

        assign_req_to_token_pool_func(
            batch.req_pool_indices,       # [bs]: request pool indices
            self.req_to_token_pool.req_to_token,  # [max_reqs, max_seq_len]: mapping table
            batch.seq_lens,               # [bs]: current sequence lengths (start positions)
            new_seq_lens,                 # [bs]: new sequence lengths (end positions)
            accepted_out_cache_loc,       # [sum(accept_lens)]: accepted slot IDs
            bs,
        )

        # ========================================================================
        # STEP 4 (TREE MODE ONLY): Shift pre-allocated slots
        # ========================================================================
        # V2 pre-allocates slots for NEXT iteration at positions:
        #   req_to_token[seq_lens + tree_size : kv_allocated]
        #
        # After accepting only N tokens, we need to:
        #   1. Move pre-alloc to start at position seq_lens + N (not seq_lens + tree_size)
        #   2. Update kv_allocated_len to reflect freed slots
        #
        # This prevents slot DUPLICATION where the same physical slot appears
        # at multiple logical positions in req_to_token.
        #
        # IMPORTANT: batch.reqs is the SAME object reference as the scheduler's reqs!
        # This is because get_model_worker_batch() passes reqs=self.reqs (not a copy).
        # Any changes we make to batch.reqs[i].kv_allocated_len here will be visible
        # to the scheduler in the next iteration, without explicit communication.
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

        ============================================================================
        ⚠️ CPU-GPU SYNC WARNING - This function has O(bs) sync points!
        ============================================================================

        SYNCS IN PHASE 1 (per-request loop):
          - req_pool_indices[i].item()  → O(bs) GPU→CPU syncs
          - batch.seq_lens[i].item()    → O(bs) GPU→CPU syncs
          - req_to_token[...].clone()   → O(bs) GPU reads

        SYNCS IN PHASE 3 (per-request loop):
          - accept_length[i].item()     → O(bs) GPU→CPU syncs

        TOTAL: ~4*bs GPU-CPU sync operations per decode iteration!

        For bs=8, this means ~32 sync points, adding significant latency.

        REMOVAL STRATEGIES (see eagle3_v2_tree_flow_diagram.md):
          - Option A: Vectorize with GPU tensors (avoid .item() calls)
          - Option B: Custom Triton kernel for slot shifting
          - Option C: Pre-compute on CPU batch tensors

        ============================================================================
        BACKGROUND: V2's Over-Allocation Strategy
        ============================================================================

        V2 pre-allocates MORE slots than needed to avoid per-iteration alloc() calls.
        In prepare_for_decode(), it allocates 2 * ALLOC_LEN_PER_DECODE slots:
          - First tree_size slots: for current verification tree
          - Remaining slots: pre-allocated for NEXT iteration

        The req_to_token layout BEFORE this function:

        req_to_token[req_idx, 0..kv_allocated]:
          [committed tokens...][tree slots (tree_size)][pre-alloc slots]
          ↑                    ↑                       ↑
          0                    seq_lens               seq_lens + tree_size

        After accepting only accept_len tokens (scattered), we need:
          [committed][accepted (accept_len)][pre-alloc slots shifted left]
          ↑          ↑                       ↑
          0          seq_lens               seq_lens + accept_len

        ============================================================================
        WHY SHIFTING IS NECESSARY
        ============================================================================

        WITHOUT shifting, after _update_req_to_token_for_accepted writes accepted slots:

          req_to_token[req_idx, seq_lens:seq_lens+accept_len] = [accepted A, C, H]
          req_to_token[req_idx, seq_lens+accept_len:seq_lens+tree_size] = [OLD B, D, E, ...]
                                                                         ↑ STALE DATA!
          req_to_token[req_idx, seq_lens+tree_size:kv_allocated] = [pre-alloc N, O, P, ...]

        The "OLD B, D, E..." region contains stale slot IDs that might duplicate
        the accepted slots or reference freed memory. Next iteration's tree would
        read garbage from positions seq_lens+accept_len to seq_lens+tree_size.

        WITH shifting:
          req_to_token[req_idx, seq_lens:seq_lens+accept_len] = [accepted A, C, H]
          req_to_token[req_idx, seq_lens+accept_len:...] = [pre-alloc N, O, P, ...]
                                                           ↑ CLEAN pre-alloc!

        ============================================================================
        EXAMPLE (bs=2, tree_size=32, num_steps=5, topk=10 from config [5, 10, 32])
        ============================================================================

        Before verification:
          req 0: seq_lens=100, kv_allocated=164 (100 + 2*32 pre-allocated)
          req_to_token[0, 100:132] = tree slots T0-T31
          req_to_token[0, 132:164] = pre-alloc slots P0-P31

        After accepting [0, 2, 7, 15] (accept_len=4):
          new_seq_lens = 104

          Step 3 already wrote: req_to_token[0, 100:104] = [T0, T2, T7, T15]

          NOW we shift pre-alloc:
            Source: req_to_token[0, 132:164] = [P0, P1, ..., P31] (32 slots)
            Dest:   req_to_token[0, 104:136] = [P0, P1, ..., P31]

          Update: kv_allocated = 164 - (32 - 4) = 136

        Final layout:
          req_to_token[0, 100:104] = [T0, T2, T7, T15]  (accepted)
          req_to_token[0, 104:136] = [P0...P31]         (shifted pre-alloc)
          kv_allocated = 136

        ============================================================================
        PERFORMANCE NOTE
        ============================================================================

        This function uses CPU indexing (.item()) and GPU reads (.clone()) which
        cause GPU-CPU synchronization. This adds ~1-5ms latency per iteration.

        TODO: Implement GPU-only version using custom Triton kernel to avoid syncs.

        ============================================================================
        CONNECTION TO SCHEDULER
        ============================================================================

        We modify batch.reqs[i].kv_allocated_len here. This is the SAME object
        as the scheduler's req (passed via get_model_worker_batch), so the
        scheduler sees the updated value in the next iteration when it calls
        prepare_for_decode() which computes:

          needed = kv_committed_len + 2 * ALLOC_LEN_PER_DECODE - kv_allocated_len

        By reducing kv_allocated_len by (tree_size - accept_len), we ensure
        the scheduler allocates the right number of slots next iteration.

        ============================================================================
        """
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = len(batch.seq_lens)
        req_to_token = self.req_to_token_pool.req_to_token  # [max_reqs, max_seq_len]
        req_pool_indices = batch.req_pool_indices  # [bs]

        # ========================================================================
        # PHASE 1: Read pre-alloc slots from current positions
        # ========================================================================
        # Pre-alloc region: [seq_lens + tree_size : kv_allocated]
        # We need to read these BEFORE writing, hence the loop with .clone()
        all_prealloc_slots = []
        prealloc_lengths = []

        for i in range(bs):
            req = batch.reqs[i]
            # ⚠️ SYNC #3: .item() causes GPU→CPU sync (O(bs) total)
            req_idx = req_pool_indices[i].item()  # GPU→CPU sync!
            # ⚠️ SYNC #4: .item() causes GPU→CPU sync (O(bs) total)
            seq_len = batch.seq_lens[i].item()  # GPU→CPU sync!
            kv_allocated = req.kv_allocated_len  # Already on CPU (Req object)

            # Pre-alloc starts after tree region
            prealloc_start = seq_len + tree_size
            prealloc_end = kv_allocated

            if prealloc_end > prealloc_start:
                # Read and clone to avoid issues when we overwrite
                prealloc_slots = req_to_token[req_idx, prealloc_start:prealloc_end].clone()
                all_prealloc_slots.append(prealloc_slots)
                prealloc_lengths.append(len(prealloc_slots))
            else:
                # No pre-alloc slots for this request
                prealloc_lengths.append(0)

        # ========================================================================
        # PHASE 2: Write pre-alloc slots to new (shifted) positions
        # ========================================================================
        # NOTE for bs > 1: Different requests may have different prealloc lengths,
        # including 0 (no prealloc). The Triton kernel handles this correctly:
        #   - prealloc_lengths = [10, 0, 12] for bs=3
        #   - flat_prealloc = torch.cat([req0_slots, req2_slots]) has length 22
        #   - Kernel computes cumulative offset: req0 gets [0:10], req1 gets nothing,
        #     req2 offset = 10 (not 0+10=10, not 10+0=10), reads [10:22]
        # The key is that prealloc_lengths has bs entries (with 0s), while
        # all_prealloc_slots only has non-empty tensors.
        if any(plen > 0 for plen in prealloc_lengths):
            if all_prealloc_slots:
                # Flatten all pre-alloc slots for batch write
                # Shape: [sum(prealloc_lengths)] - skips requests with 0 prealloc
                flat_prealloc = torch.cat(all_prealloc_slots)

                # Destination: write starting at new_seq_lens (after accepted tokens)
                # new_seq_lens = old_seq_lens + accept_length
                dest_starts = new_seq_lens.clone()  # [bs]
                dest_ends = new_seq_lens + torch.tensor(
                    prealloc_lengths, device=new_seq_lens.device, dtype=new_seq_lens.dtype
                )  # [bs]

                # Use Triton kernel to write slots to req_to_token
                # This writes flat_prealloc to req_to_token[req_idx, dest_start:dest_end]
                assign_req_to_token_pool_func(
                    req_pool_indices,
                    req_to_token,
                    dest_starts,
                    dest_ends,
                    flat_prealloc,
                    bs,
                )
                print(f"[EAGLE3_DEBUG tree_mode] Shifted {sum(prealloc_lengths)} pre-alloc slots")

        # ========================================================================
        # PHASE 3: Update kv_allocated_len for each request (ALWAYS for tree mode!)
        # ========================================================================
        # BUG FIX: This was previously inside the prealloc shifting conditional,
        # but we ALWAYS free rejected slots in Step 2, so we MUST always update
        # kv_allocated_len to match. Otherwise, kv_allocated_len > actual allocated
        # slots, causing double-free when request ends.
        #
        # We freed (tree_size - accept_len) rejected slots per request.
        # The kv_allocated_len must shrink by this amount so that:
        #   1. Next iteration's prepare_for_decode() allocates correct count
        #   2. The "dead" region between shifted pre-alloc and old kv_allocated
        #      is no longer tracked
        #
        # Example (tree_size=32):
        #   Before: kv_allocated=164, accept_len=4
        #   rejected = 32 - 4 = 28
        #   After: kv_allocated = 164 - 28 = 136
        for i in range(bs):
            # ⚠️ SYNC #5: .item() causes GPU→CPU sync (O(bs) total)
            accept_len = accept_length[i].item()  # GPU→CPU sync!
            rejected_count = tree_size - accept_len
            # This modifies the SAME Req object the scheduler has
            batch.reqs[i].kv_allocated_len -= rejected_count

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
