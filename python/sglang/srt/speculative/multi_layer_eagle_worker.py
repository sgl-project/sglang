# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import speculative_moe_backend_context
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    detect_nan,
    draft_tp_context,
    fast_topk,
    generate_token_bitmask,
    load_token_map,
    select_top_k_tokens,
)
from sglang.srt.utils import empty_context, get_available_gpu_memory, is_cuda, is_npu

_is_npu = is_npu()

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)


class MultiLayerEagleWorker(TpModelWorker):

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
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.draft_extend_attn_backend_list = []

        # Override the context length of the draft model to be the same as the target model.
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
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with ctx, speculative_moe_backend_context():
            super().__init__(
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
                is_multi_layer_eagle=True,
            )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_model_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_model_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            for i in range(self.speculative_num_steps):
                self.mtp_model_runner(i).model.set_embed_and_head(embed, head)

        # Init attention backend and cuda graphs
        for i in range(self.speculative_num_steps):
            self.mtp_model_runner(i).server_args.disable_cuda_graph = (
                backup_disable_cuda_graph
            )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.mtp_model_runner(0).tp_group
        ), speculative_moe_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners
        for step in range(self.speculative_num_steps):
            draft_backend_factory = DraftBackendFactory(
                self.server_args,
                self.mtp_model_runner(step),
                self.topk,
                self.speculative_num_steps,
            )

            # Initialize draft extend attention backend (respects speculative_attention_mode setting)
            self.draft_extend_attn_backend_list.append(
                draft_backend_factory.create_draft_extend_backend()
            )

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner_for_draft_extend_list = []

        if self.server_args.disable_cuda_graph:
            return

        # Capture extend
        for step in range(self.speculative_num_steps):
            if self.draft_extend_attn_backend_list[step] and not _is_npu:
                tic = time.perf_counter()
                before_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
                )
                self.cuda_graph_runner_for_draft_extend_list.append(
                    MultiLayerEagleDraftExtendCudaGraphRunner(self, step)
                )
                after_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
                )

    def mtp_model_runner(self, layer_id: int):
        return self.model_runner_list[layer_id]

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, seq_lens_cpu = self.forward_target_extend(
                batch
            )
            with self.draft_tp_context(
                self.mtp_model_runner(0).tp_group
            ), speculative_moe_backend_context():
                self.forward_draft_extend(
                    batch, logits_output.hidden_states, next_token_ids, seq_lens_cpu
                )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )
        else:
            with self.draft_tp_context(
                self.mtp_model_runner(0).tp_group
            ), speculative_moe_backend_context():
                spec_info = self.draft(batch)
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )

            with self.draft_tp_context(
                self.mtp_model_runner(0).tp_group
            ), speculative_moe_backend_context():
                # NOTE: We should use `check_forward_draft_extend_after_decode`
                # when DP attention is enabled, but it is slow. Skip it for now.
                if (
                    self.server_args.enable_dp_attention
                    or batch.spec_info.verified_id.shape[0] > 0
                ):
                    # decode is not finished
                    self.forward_draft_extend_after_decode(batch)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def check_forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        local_need_forward = batch.spec_info.verified_id.shape[0] > 0
        if not self.server_args.enable_dp_attention:
            return local_need_forward

        global_need_forward = torch.tensor(
            [
                (local_need_forward),
            ],
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(
            global_need_forward, group=get_tp_group().cpu_group
        )
        global_need_forward_cnt = global_need_forward[0].item()
        need_forward = global_need_forward_cnt > 0
        return need_forward

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int, Optional[torch.Tensor]]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        model_worker_batch.return_hidden_states_before_norm = True
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        return (
            logits_output,
            next_token_ids,
            model_worker_batch.seq_lens_cpu,
        )

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        from sglang.srt.speculative.eagle_worker import EAGLEWorker

        # FIXME: migrate multi-layer eagle worker to eagle worker
        return EAGLEWorker._draft_preprocess_decode(self, batch)

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        from sglang.srt.speculative.eagle_worker import EAGLEWorker

        # FIXME: migrate multi-layer eagle worker to eagle worker
        return EAGLEWorker._draft_preprocess_idle(self, batch)

    def draft(self, batch: ScheduleBatch):
        # Parse args
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
        else:
            self._draft_preprocess_decode(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info.num_tokens_per_batch = self.topk
        spec_info.num_tokens_for_logprob_per_batch = self.topk
        batch.return_hidden_states = False

        # Get forward batch
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.mtp_model_runner(0)
        )
        forward_batch.can_run_dp_cuda_graph = False
        forward_batch.return_hidden_states_before_norm = True

        # Parse args
        assert isinstance(spec_info, EagleDraftInput)
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
            0, topk_p, topk_index, hidden_states, scores, self.topk
        )
        if self.speculative_num_steps == 1:
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])
        else:
            for i in range(self.speculative_num_steps):
                score_list.append(tree_info[0][:, :, i].unsqueeze(-1))
                token_index = tree_info[1][:, i].unsqueeze(-1)
                token_list.append(token_index)
                if i == 0:
                    parents_list.append(tree_info[2])
                else:
                    parents_list.append(
                        torch.full(
                            (tree_info[2].size(0), 1),
                            i,
                            dtype=torch.long,
                            device=self.device,
                        )
                    )

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
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
            draft_token_num=self.server_args.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker
        pass

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        spec_info.prepare_for_verify(batch, self.page_size)
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode
        model_worker_batch.return_hidden_states_before_norm = True

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrive_next_token.shape
            ).cpu()

        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # Overlap the CPU operations for bitmask generation with the forward pass.
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
                # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        if self.enable_nan_detection:
            detect_nan(logits_output)

        spec_info.hidden_states = logits_output.hidden_states
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]

        if self.target_worker.model_runner.hybrid_gdn_config is not None:
            accepted_length = (
                torch.tensor(
                    res.accept_length_per_req_cpu,
                    device=logits_output.hidden_states.device,
                    dtype=torch.int64,
                )
                + 1
            )

            # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
            # res.accepted_indices.shape[0] > 0 skips DP attn idle batch
            if spec_info.topk > 1 and res.accepted_indices.shape[0] > 0:
                # accepted_indices=[0,2,3,4,5,7,9,10,11], accepted_length=[4, 3, 2], cumulative_accepted_lengths=[4, 7, 9]
                # first_token_indices_per_req=prepend(0, accepted_indices[cumulative_accepted_lengths[:-1]]) = [0, 5, 10]
                # last_token_indices_per_req=accepted_indices[cumulative_accepted_lengths - 1] = [4, 9, 11] (last token ID of each req)
                # max_relative_indices_per_req = [4,4,1]; those are the per-req spec-decoding step offsets that contain the correct mamba caches
                cumulative_accepted_lengths = torch.cumsum(accepted_length, dim=0)
                req_start_positions = torch.cat(
                    [
                        torch.zeros(
                            1,
                            dtype=cumulative_accepted_lengths.dtype,
                            device=cumulative_accepted_lengths.device,
                        ),
                        cumulative_accepted_lengths[:-1],
                    ]
                )
                first_token_indices_per_req = res.accepted_indices[req_start_positions]
                last_token_indices_per_req = res.accepted_indices[
                    cumulative_accepted_lengths - 1
                ]
                max_relative_indices_per_req = (
                    last_token_indices_per_req - first_token_indices_per_req
                )
            else:
                max_relative_indices_per_req = accepted_length - 1
            self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
                max_relative_indices_per_req, self.target_worker.model_runner.model
            )

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # Prepare the batch for the next draft forwards.
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
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
            num_tokens_per_batch=1,
            num_tokens_for_logprob_per_batch=1,
        )
        batch.return_hidden_states = False
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=seq_lens_cpu
        )
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.mtp_model_runner(0)
        )
        forward_batch.return_logprob = False
        forward_batch.return_hidden_states_before_norm = True
        topk_p_list = []
        topk_index_list = []
        for step in range(self.speculative_num_steps):
            logits_output, _ = self.mtp_model_runner(step).forward(forward_batch)
            if self.enable_nan_detection:
                detect_nan(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            topk_p_list.append(topk_p)
            topk_index_list.append(topk_index)
            pt = 0
            if forward_batch.extend_seq_lens is not None:
                for i, extend_len in enumerate(forward_batch.extend_seq_lens):
                    input_ids = forward_batch.input_ids[pt : pt + extend_len]
                    forward_batch.input_ids[pt : pt + extend_len] = torch.cat(
                        (input_ids[1:], topk_index[i].reshape(1))
                    )
                    pt += extend_len

        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        forward_batch.spec_info.topk_p = torch.cat(topk_p_list, dim=1)
        forward_batch.spec_info.topk_index = torch.cat(topk_index_list, dim=1)
        has_finished, unfinished_req_index = False, []
        for i, req in enumerate(batch.reqs):
            if req.finished():
                has_finished = True
            else:
                unfinished_req_index.append(i)
        if has_finished:
            unfinished_index_device = torch.tensor(
                unfinished_req_index,
                dtype=torch.int64,
                device=batch.spec_info.topk_p.device,
            )
            batch.spec_info.filter_batch(
                unfinished_index_device, has_been_filtered=False
            )

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        assert isinstance(batch.spec_info, EagleDraftInput)
        # Backup fields that will be modified in-place
        seq_lens_backup = batch.seq_lens.clone()
        seq_lens_cpu_backup = batch.seq_lens_cpu.clone()
        req_pool_indices_backup = batch.req_pool_indices
        accept_length_backup = batch.spec_info.accept_length
        return_logprob_backup = batch.return_logprob

        input_is_idle = batch.forward_mode.is_idle()

        if not input_is_idle and batch.spec_info.verified_id.numel() == 0:
            batch = batch.copy()
            batch.prepare_for_idle()
            hidden_size = (
                self.model_config.hidden_size * 3
                if self.speculative_algorithm.is_eagle3()
                else self.model_config.hidden_size
            )
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=hidden_size,
                dtype=self.model_config.dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )

        batch.spec_info.num_tokens_per_batch = self.speculative_num_steps + 1
        batch.spec_info.num_tokens_for_logprob_per_batch = 1
        batch.spec_info.prepare_extend_after_decode(
            batch,
            self.speculative_num_steps,
        )
        batch.forward_mode = (
            ForwardMode.DRAFT_EXTEND
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )

        batch.return_hidden_states = False
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.mtp_model_runner(0)
        )
        forward_batch.return_hidden_states_before_norm = True
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
        else:
            forward_batch.seq_lens_sum = batch.seq_lens.sum().item()
        topk_p_list = []
        topk_index_list = []
        # Run
        for step in range(self.speculative_num_steps):
            can_cuda_graph = len(
                self.cuda_graph_runner_for_draft_extend_list
            ) and self.cuda_graph_runner_for_draft_extend_list[step].can_run(
                forward_batch
            )
            if can_cuda_graph:
                logits_output = self.cuda_graph_runner_for_draft_extend_list[
                    step
                ].replay(forward_batch)
            else:
                forward_batch.can_run_dp_cuda_graph = False
                if not forward_batch.forward_mode.is_idle():
                    self.mtp_model_runner(step).attn_backend.init_forward_metadata(
                        forward_batch
                    )
                logits_output, _ = self.mtp_model_runner(step).forward(
                    forward_batch, skip_attn_backend_init=True
                )

            if self.enable_nan_detection:
                detect_nan(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            topk_p_list.append(topk_p)
            topk_index_list.append(topk_index)
            pt = 0
            if forward_batch.extend_seq_lens is not None:
                for i, extend_len in enumerate(forward_batch.extend_seq_lens):
                    input_ids = forward_batch.input_ids[pt : pt + extend_len]
                    forward_batch.input_ids[pt : pt + extend_len] = torch.cat(
                        (input_ids[1:], topk_index[i].reshape(1))
                    )
                    pt += extend_len

        forward_batch.spec_info.topk_p = torch.cat(topk_p_list, dim=1)
        forward_batch.spec_info.topk_index = torch.cat(topk_index_list, dim=1)

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = (
            ForwardMode.DECODE if not input_is_idle else ForwardMode.IDLE
        )
        batch.seq_lens = seq_lens_backup
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.req_pool_indices = req_pool_indices_backup
        batch.spec_info.accept_length = accept_length_backup
        batch.return_logprob = return_logprob_backup
