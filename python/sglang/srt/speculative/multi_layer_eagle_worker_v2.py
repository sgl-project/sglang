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

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.multi_layer_eagle_draft_extend_npu_graph_runner import (
    MultiLayerEagleMultiStepDraftExtendNpuGraphRunner,
)
from sglang.srt.layers.moe.utils import speculative_moe_backend_context
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import (
    default_tree_mask_mode,
    get_draft_recurrent_hidden_state_spec,
)
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    prepare_for_draft,
    prepare_for_draft_extend,
    run_eagle_verify,
)
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
    OneGraphMultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.multi_layer_eagle_utils import (
    boundary_kv_fix_enabled,
    compute_widened_draft_extend_locs_positions,
    fill_widened_draft_extend_inputs_triton,
    rotate_input_ids,
    stash_append_boundary_state_triton,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,
    get_plan_stream,
    sample_draft_proposal,
    select_top_k_tokens,
)
from sglang.srt.utils import is_cpu, is_npu, require_gathered_buffer
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import empty_context, fast_topk

_is_npu = is_npu()
_is_cpu = is_cpu()


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput


logger = logging.getLogger(__name__)


class MultiLayerEagleDraftWorker(EagleDraftWorkerBase):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.draft_extend_attn_backend_list = []
        self.model_config = target_worker.model_config

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        # Leviathan/Chen rejection sampling (temp>0): the draft samples X ~ q and
        # provides q so the verify accepts iff coin*q < p and resamples the residual.
        # Single-CG runner samples in-graph (_sample_draft_proposal); per-step
        # runner samples worker-side between replays.
        self.use_rejection_sampling = server_args.speculative_use_rejection_sampling
        assert self.speculative_num_draft_tokens == self.speculative_num_steps + 1, (
            "multi-layer EAGLE requires speculative_num_draft_tokens == "
            "speculative_num_steps + 1, "
            f"got {self.speculative_num_draft_tokens} and {self.speculative_num_steps}"
        )
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Set constant
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        # Load draft model weights only.
        with empty_context(), speculative_moe_backend_context():
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                # spec workers don't support pipeline parallelism
                ps=replace(ps, pp_rank=0),
                nccl_port=nccl_port,
                is_draft_worker=True,
                is_multi_layer_eagle=True,
            )

        # Alias for better readability
        self.draft_runner_list: List[ModelRunner] = self.draft_worker.model_runner_list
        # Match `EagleDraftWorker.draft_runner` for generic draft-runner access.
        self.draft_runner: ModelRunner = self.draft_runner_list[0]

        # Chain-style MTP: each step propagates its own output hidden states to the
        # next step.  Non-chain: each step uses the target model's hidden states.
        draft_arch = self.draft_worker.model_config.hf_config.architectures[0]
        self.chain_mtp_hidden_states = draft_arch in [
            "Step3p5MTP",
            "InklingForConditionalGenerationMTP",
            "GigaChat35ForCausalLMNextN",
        ]
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self.tree_mask_mode = default_tree_mask_mode()
        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

    @property
    def draft_runners(self) -> List[ModelRunner]:
        # One runner per draft step (len == speculative_num_steps).
        return self.draft_runner_list

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """Allocate draft KV cache pools (called by scheduler)."""
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.init_lm_head()
        self._init_boundary_kv_fix_state()

    def init_attention_backends(self):
        with (
            self.draft_tp_context(self.draft_runner_list[0].tp_group),
            speculative_moe_backend_context(),
        ):
            super().init_attention_backends()

    def init_cuda_graphs(self):
        with (
            self.draft_tp_context(self.draft_runner_list[0].tp_group),
            speculative_moe_backend_context(),
        ):
            super().init_cuda_graphs()

    def mtp_model_runner(self, step: int):
        return self.draft_runner_list[step]

    def _init_boundary_kv_fix_state(self):
        """Boundary-KV fix state: stash + widened draft-extend front. Chain topk=1 only."""
        self.draft_extend_num_front_tokens = 0
        self.draft_extend_num_warmup_tokens = 0
        self.boundary_kv_stash_tokens = None
        self.boundary_kv_stash_hiddens = None
        self.boundary_kv_stash_valid_lens = None
        if not (
            boundary_kv_fix_enabled()
            and self.topk == 1
            and self.speculative_num_steps > 1
            # The fix stashes the mamba/sconv boundary state, so it only applies
            # to hybrid models; non-hybrid MTP drafts (e.g. MiMoV2) must skip it.
            and isinstance(self.req_to_token_pool, HybridReqToTokenPool)
        ):
            return
        draft_model_runner = self.draft_runner_list[0]
        draft_hidden_size = draft_model_runner.model_config.hidden_size
        target_hidden_size = self.target_worker.model_runner.model_config.hidden_size
        if draft_hidden_size != target_hidden_size:
            logger.warning(
                "SGLANG_ENABLE_MTP_BOUNDARY_KV_FIX disabled: draft hidden size %d != "
                "target hidden size %d (the stash holds verify hiddens).",
                draft_hidden_size,
                target_hidden_size,
            )
            return
        if isinstance(self.req_to_token_pool, HybridReqToTokenPool):
            conv_state = self.req_to_token_pool.mamba_pool.mamba_cache.conv
            self.draft_extend_num_warmup_tokens = conv_state[0].shape[2]
        self.draft_extend_num_front_tokens = (
            self.speculative_num_steps - 1 + self.draft_extend_num_warmup_tokens
        )
        front = self.draft_extend_num_front_tokens
        req_pool_size = self.req_to_token_pool.req_to_token.shape[0]
        with torch.device(self.device):
            self.boundary_kv_stash_tokens = torch.zeros(
                (req_pool_size, front), dtype=torch.int64
            )
            self.boundary_kv_stash_hiddens = torch.zeros(
                (req_pool_size, front, draft_hidden_size),
                dtype=draft_model_runner.dtype,
            )
            self.boundary_kv_stash_valid_lens = torch.zeros(
                (req_pool_size,), dtype=torch.int32
            )
        logger.info(
            "SGLANG_ENABLE_MTP_BOUNDARY_KV_FIX on: draft-extend windows widened by "
            "%d front rows (%d conv warm-up).",
            front,
            self.draft_extend_num_warmup_tokens,
        )

    def _compute_boundary_kv_locs_positions(self, batch):
        if self.draft_extend_num_front_tokens == 0 or batch.forward_mode.is_idle():
            return None, None, None
        locs, positions = compute_widened_draft_extend_locs_positions(
            batch.seq_lens,
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            self.boundary_kv_stash_valid_lens,
            self.speculative_num_draft_tokens,
            self.draft_extend_num_front_tokens,
            self.draft_extend_num_warmup_tokens,
        )
        ready_event = None
        if self.plan_stream:
            ready_event = torch.get_device_module(self.device).Event()
            ready_event.record()
        return locs, positions, ready_event

    def _seed_boundary_kv_stash(self, forward_batch, target_hidden_states):
        if (
            self.draft_extend_num_front_tokens == 0
            or forward_batch.forward_mode.is_idle()
            or forward_batch.extend_seq_lens is None
            or target_hidden_states is None
        ):
            return
        extend_seq_lens = forward_batch.extend_seq_lens
        src_row_ends = (forward_batch.extend_start_loc + extend_seq_lens).to(
            torch.int64
        )
        stash_append_boundary_state_triton(
            forward_batch.input_ids,
            target_hidden_states,
            src_row_ends,
            extend_seq_lens,
            forward_batch.req_pool_indices,
            self.boundary_kv_stash_tokens,
            self.boundary_kv_stash_hiddens,
            self.boundary_kv_stash_valid_lens,
            set_valid=True,
        )

    def _fill_boundary_kv_front_and_update_stash(
        self, batch, forward_batch, predict, verify_hiddens, accept_lens
    ):
        if self.draft_extend_num_front_tokens == 0 or batch.forward_mode.is_idle():
            return
        draft_token_num = self.speculative_num_draft_tokens
        fill_widened_draft_extend_inputs_triton(
            forward_batch.input_ids,
            forward_batch.spec_info.hidden_states,
            predict,
            verify_hiddens,
            self.boundary_kv_stash_tokens,
            self.boundary_kv_stash_hiddens,
            self.boundary_kv_stash_valid_lens,
            batch.seq_lens,
            batch.req_pool_indices,
            draft_token_num=draft_token_num,
        )
        bs = len(batch.seq_lens)
        arange = torch.arange(bs, device=predict.device, dtype=torch.int64)
        src_row_ends = arange * draft_token_num + accept_lens.to(torch.int64)
        stash_append_boundary_state_triton(
            predict,
            verify_hiddens,
            src_row_ends,
            accept_lens,
            batch.req_pool_indices,
            self.boundary_kv_stash_tokens,
            self.boundary_kv_stash_hiddens,
            self.boundary_kv_stash_valid_lens,
            set_valid=False,
        )

    def init_lm_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        # Share the embedding and lm_head
        for i in range(self.speculative_num_steps):
            self.draft_runner_list[i].model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        # Create attn backends
        self.draft_extend_attn_backend_list = []
        for step in range(self.speculative_num_steps):
            draft_backend_factory = DraftBackendFactory(
                self.server_args,
                self.draft_runner_list[step],
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend_list.append(
                draft_backend_factory.create_draft_extend_backend()
            )
            if self.draft_extend_attn_backend_list[-1] is not None:
                self.draft_runner_list[step].attn_backend = (
                    self.draft_extend_attn_backend_list[-1]
                )

    def _capture_cuda_graphs(self):
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if _is_cpu or check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return

        if envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH.get():
            return

        if not _is_npu:
            # The single-CG runner replays with no Python between steps, so the
            # attn backend must fully rebuild its per-step metadata as captured
            # tensor ops; anything less gets capture-time-stale metadata (e.g.
            # SWA translations, which only the eager replay path refreshes).
            # Per-depth pools (banded MTP) mean per-depth backends — EVERY step
            # must satisfy this, not just step 0.
            draft_backend = self.draft_runner_list[0].attn_backend
            backend_supports_single_cg = all(
                runner.attn_backend.draft_extend_metadata_captured_in_graph()
                for runner in self.draft_runner_list
            )
            if envs.SGLANG_ENABLE_SINGLE_CG_DRAFT.get() and backend_supports_single_cg:
                self.cuda_graph_runner_for_draft_extend = (
                    OneGraphMultiLayerEagleMultiStepDraftExtendCudaGraphRunner(self)
                )
            else:
                if envs.SGLANG_ENABLE_SINGLE_CG_DRAFT.get():
                    logger.warning(
                        "SGLANG_ENABLE_SINGLE_CG_DRAFT is on but %s does not fully "
                        "rebuild its draft-extend metadata in-graph; falling back "
                        "to per-step draft graphs.",
                        type(draft_backend).__name__,
                    )
                self.cuda_graph_runner_for_draft_extend = (
                    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner(self)
                )
        else:
            self.cuda_graph_runner_for_draft_extend = (
                MultiLayerEagleMultiStepDraftExtendNpuGraphRunner(self)
            )

    def draft(self, batch: ScheduleBatch):
        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_cuda_graph = prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner_list[0],
            self.topk,
            self.speculative_num_steps,
        )

        # Run draft
        parent_list, top_scores_index, draft_tokens = self.draft_forward(forward_batch)

        return build_eagle_verify_input(
            batch,
            draft_input,
            parent_list,
            top_scores_index,
            draft_tokens,
            draft_input.draft_probs,
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        _, hidden_states, scores, tree_info = select_top_k_tokens(
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
                            device=tree_info[2].device,
                        )
                    )

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
        maybe_detect_oob(
            top_scores_index,
            0,
            ss_token_list.shape[1],
            "draft_forward: top_scores_index OOB for gather on ss_token_list",
        )
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def draft_extend(self):
        pass

    def _apply_deferred_mamba_init_to_draft_pools(self, forward_batch) -> None:
        if (
            self.draft_runner.model_config.hf_config.architectures[0]
            != "InklingForConditionalGenerationMTP"
        ):
            return
        fm = forward_batch.forward_mode
        if not (fm.is_extend(include_draft_extend_v2=True) or fm.is_decode()):
            return
        clear = forward_batch.mamba_clear_indices
        cow_src = forward_batch.mamba_cow_src_indices
        cow_dst = forward_batch.mamba_cow_dst_indices
        if (clear is None or len(clear) == 0) and (
            cow_src is None or len(cow_src) == 0
        ):
            return
        seen = set()
        for runner in self.draft_runner_list:
            pool = runner.req_to_token_pool.mamba_pool
            if id(pool) in seen:
                continue
            seen.add(id(pool))
            if clear is not None and len(clear) > 0:
                pool.clear_slots(clear)
            if cow_src is not None and len(cow_src) > 0:
                pool.copy_from(cow_src, cow_dst)
        forward_batch.mamba_clear_indices = None
        forward_batch.mamba_cow_src_indices = None
        forward_batch.mamba_cow_dst_indices = None

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
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
        # The draft embed clamps unconditionally (to tolerate multimodal pad
        # sentinels), so probe next_token_ids here first -- otherwise a corrupted id
        # would be clamped away instead of surfacing.
        maybe_detect_oob(
            next_token_ids,
            0,
            self.model_config.vocab_size,
            "draft_extend_for_prefill: next_token_ids before draft embed",
        )

        # Draft-extend spec_info for the extend forward; carries only
        # hidden_states + shape info.
        extend_input = EagleDraftExtendInput(
            hidden_states=target_hidden_states,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.spec_info = extend_input

        # Chain-style MTP needs FULL to get all-token hidden states;
        # non-chain only needs LAST (the target model's hidden states).
        # STANDALONE skips hidden states end-to-end.
        if self.speculative_algorithm.is_standalone():
            draft_capture_hidden_mode = CaptureHiddenMode.NULL
        elif self.chain_mtp_hidden_states:
            draft_capture_hidden_mode = CaptureHiddenMode.FULL
        else:
            draft_capture_hidden_mode = CaptureHiddenMode.LAST

        # Run forward
        forward_batch = ForwardBatch.init_new(
            batch,
            self.draft_runner_list[0],
            capture_hidden_mode=draft_capture_hidden_mode,
            return_hidden_states_before_norm=True,
        )

        self._apply_deferred_mamba_init_to_draft_pools(forward_batch)

        # Construct input_ids
        # TODO: same chunked-prefill chain divergence as PR #26329.
        if not batch.forward_mode.is_idle():
            rotate_input_ids(
                forward_batch.input_ids,
                forward_batch.extend_start_loc,
                forward_batch.extend_seq_lens,
                next_token_ids,
            )

        self._seed_boundary_kv_stash(forward_batch, target_hidden_states)

        topk_p_list = []
        topk_index_list = []
        draft_probs_list = []
        for step in range(self.speculative_num_steps):
            forward_batch.req_to_token_pool = self.draft_runner_list[
                step
            ].req_to_token_pool
            forward_batch.token_to_kv_pool = self.draft_runner_list[
                step
            ].token_to_kv_pool
            output: ModelRunnerOutput = self.draft_runner_list[step].forward(
                forward_batch
            )
            maybe_detect_nan(
                output.logits_output.next_token_logits,
                f"draft_extend_for_prefill step {step}",
            )
            maybe_detect_inf(
                output.logits_output.next_token_logits,
                f"draft_extend_for_prefill step {step}",
            )
            if self.use_rejection_sampling and self.topk == 1:
                # Rejection sampling (prefill): sample X ~ q and stash q for the first verify.
                probs, topk_p, topk_index = sample_draft_proposal(
                    output.logits_output.next_token_logits,
                    forward_batch.sampling_info.temperatures,
                )
                draft_probs_list.append(probs)
            else:
                probs = torch.softmax(output.logits_output.next_token_logits, dim=-1)
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            topk_p_list.append(topk_p)
            topk_index_list.append(topk_index)
            # Chain-style: use this step's output hidden_states as next step's input
            if (
                self.chain_mtp_hidden_states
                and step < self.speculative_num_steps - 1
                and output.logits_output.hidden_states is not None
            ):
                forward_batch.spec_info.hidden_states = (
                    output.logits_output.hidden_states
                )
            if forward_batch.extend_seq_lens is not None:
                rotate_input_ids(
                    forward_batch.input_ids,
                    forward_batch.extend_start_loc,
                    forward_batch.extend_seq_lens,
                    topk_index,
                )

        next_draft_input = EagleDraftInput(
            topk_p=torch.cat(topk_p_list, dim=1),
            topk_index=torch.cat(topk_index_list, dim=1),
            # Chain-style left the last step's hidden_states on the extend
            # input; non-chain keeps the target hidden states.
            hidden_states=extend_input.hidden_states,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        # q [bs, num_steps, vocab] for the first verify's Leviathan step (rejection only).
        next_draft_input.draft_probs = (
            torch.stack(draft_probs_list, dim=1)
            if self.use_rejection_sampling and draft_probs_list
            else None
        )

        return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        # Batch 2: Draft extend
        draft_extend_input = EagleDraftExtendInput(
            hidden_states=batch_result.logits_output.hidden_states,
            # Actual width: the multi-layer chain fills num_steps + 1 rows/req.
            num_tokens_per_req=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_req=1,
            num_front_tokens=self.draft_extend_num_front_tokens,
        )

        # Prepare for draft extend in a separate stream
        # Notice that here we use batch_result.next_token_ids as the input ids
        boundary_kv_locs, boundary_kv_positions, boundary_kv_ready_event = (
            self._compute_boundary_kv_locs_positions(batch)
        )

        with self.plan_stream_ctx:
            if boundary_kv_ready_event is not None:
                self.plan_stream.wait_event(boundary_kv_ready_event)
            forward_batch = prepare_for_draft_extend(
                draft_extend_input,
                batch,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner_list[0],
                self.cuda_graph_runner_for_draft_extend,
                return_hidden_states_before_norm=True,
                widened_out_cache_loc=boundary_kv_locs,
                widened_positions=boundary_kv_positions,
            )

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        self._apply_deferred_mamba_init_to_draft_pools(forward_batch)
        self._fill_boundary_kv_front_and_update_stash(
            batch,
            forward_batch,
            batch_result.next_token_ids,
            batch_result.logits_output.hidden_states,
            batch_result.accept_lens,
        )

        # `batch_result.accept_lens` includes the bonus token, so drafts-only
        # is accept_lens - 1. Stash on spec_info for the cuda-graph prepare().
        forward_batch.spec_info.num_correct_drafts = batch_result.accept_lens - 1
        forward_batch.spec_info.num_accept_tokens = batch_result.accept_lens

        # Run draft extend batch in the main compute stream
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run_graph(forward_batch)
        )
        ret_topk_p_list = []
        ret_topk_index_list = []
        ret_draft_probs_list = []
        ret_draft_probs = None
        next_token_ids_backup = batch_result.next_token_ids.clone()

        if can_cuda_graph:
            cgr = self.cuda_graph_runner_for_draft_extend
            # Populate the single shared buffer set once; each step replays
            # against it and the chain is advanced in place between steps.
            cgr.prepare(forward_batch)
            rotates_in_graph = cgr.rotates_in_graph
            for step in range(self.speculative_num_steps):
                _out, ret_topk_p, ret_topk_index = cgr.replay(step)
                # Rejection sampling with the per-step runner re-picks X ~ q
                # worker-side so the worker rotation carries it to step N+1; the
                # single-CG runner samples in-graph (q cloned after the loop).
                if (
                    self.use_rejection_sampling
                    and self.topk == 1
                    and not rotates_in_graph
                ):
                    if cgr.prune_draft_extend_logits:
                        step_logits = _out.next_token_logits
                    else:
                        sel = cgr.buffers.select_index[: cgr.raw_bs]
                        step_logits = _out.next_token_logits[sel]
                    probs, ret_topk_p, ret_topk_index = sample_draft_proposal(
                        step_logits,
                        forward_batch.sampling_info.temperatures,
                    )
                    ret_draft_probs_list.append(probs)
                if rotates_in_graph:
                    # Single-CG step outputs coexist until the trailing cat.
                    ret_topk_p_list.append(ret_topk_p)
                    ret_topk_index_list.append(ret_topk_index)
                else:
                    # Per-step graphs share the global graph pool; snapshot
                    # before the next step's replay can reuse the buffer.
                    ret_topk_p_list.append(ret_topk_p.clone())
                    ret_topk_index_list.append(ret_topk_index.clone())
                # Advance the draft chain by rotating the shared input_ids window
                # in place; step N+1's graph then reads the rotated values. The
                # single-CG runner rotates in-graph, so skip the worker-side rotate.
                if step < self.speculative_num_steps - 1 and not rotates_in_graph:
                    rotate_input_ids(
                        cgr.buffers.input_ids[: cgr.raw_num_tokens],
                        cgr.buffers.extend_start_loc[: cgr.raw_bs],
                        cgr.buffers.extend_seq_lens[: cgr.raw_bs],
                        ret_topk_index,
                        cgr.buffers.select_index[: cgr.raw_bs],
                    )
            if self.use_rejection_sampling and self.topk == 1 and rotates_in_graph:
                ret_draft_probs = cgr.clone_draft_probs()
        else:
            logger.warning_once(
                "can't use cuda graph for draft extend! may have correctness issue!"
            )
            select_index = (
                torch.arange(len(batch.seq_lens), device=self.device)
                * (
                    self.speculative_num_draft_tokens
                    + self.draft_extend_num_front_tokens
                )
                + self.draft_extend_num_front_tokens
                + batch_result.accept_lens
                - 1
            )
            if self.cuda_graph_runner_for_draft_extend:
                prune_logits = (
                    self.cuda_graph_runner_for_draft_extend.prune_draft_extend_logits
                )
            else:
                prune_logits = not require_gathered_buffer(self.server_args)
            if prune_logits:
                forward_batch.spec_info.select_index = select_index
            # Left unmarked on every platform: each de-tied runner has its own
            # attn backend, and only runner[0]'s was pre-planned, so each step's
            # forward must init its own metadata post-pad (mirrors NPU behavior).
            for step in range(self.speculative_num_steps):
                forward_batch.req_to_token_pool = self.draft_runner_list[
                    step
                ].req_to_token_pool
                forward_batch.token_to_kv_pool = self.draft_runner_list[
                    step
                ].token_to_kv_pool
                self.draft_runner_list[step].attn_backend.init_forward_metadata(
                    forward_batch
                )
                draft_logits_output = self.draft_runner_list[step].forward(
                    forward_batch
                )
                if prune_logits:
                    logits_sel = draft_logits_output.logits_output.next_token_logits
                else:
                    logits_sel = draft_logits_output.logits_output.next_token_logits[
                        select_index
                    ]
                if self.use_rejection_sampling and self.topk == 1:
                    probs, ret_topk_p, ret_topk_index = sample_draft_proposal(
                        logits_sel, forward_batch.sampling_info.temperatures
                    )
                    ret_draft_probs_list.append(probs)
                else:
                    probs = torch.softmax(logits_sel, dim=-1)
                    ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
                # Chain-style: use this step's output hidden_states as next step's input
                if (
                    self.chain_mtp_hidden_states
                    and step < self.speculative_num_steps - 1
                    and draft_logits_output.logits_output.hidden_states is not None
                ):
                    forward_batch.spec_info.hidden_states = (
                        draft_logits_output.logits_output.hidden_states
                    )
                if forward_batch.extend_seq_lens is not None:
                    rotate_input_ids(
                        forward_batch.input_ids,
                        forward_batch.extend_start_loc,
                        forward_batch.extend_seq_lens,
                        ret_topk_index,
                        select_index,
                    )
                ret_topk_p_list.append(ret_topk_p)
                ret_topk_index_list.append(ret_topk_index)

        batch_result.next_token_ids = next_token_ids_backup
        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            torch.cat(ret_topk_p_list, dim=1),
            torch.cat(ret_topk_index_list, dim=1),
            None,
        )
        # Under rejection sampling, carry the per-chain-step draft distributions
        # q [bs, num_steps, vocab] so the next verify runs Leviathan (accept iff
        # coin*q < p). None otherwise (default target-only tree sampling).
        if ret_draft_probs is None and ret_draft_probs_list:
            ret_draft_probs = torch.stack(ret_draft_probs_list, dim=1)
        next_draft_input.draft_probs = ret_draft_probs


class MultiLayerEagleWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.override(
            "spec_worker.match_target_context_length",
            context_length=target_worker.model_runner.model_config.context_len,
        )

        self._draft_worker = MultiLayerEagleDraftWorker(
            server_args,
            gpu_id,
            ps,
            nccl_port,
            target_worker,
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self._target_worker.model_runner.attn_backend,
            *(
                backend or runner.attn_backend
                for backend, runner in zip(
                    self._draft_worker.draft_extend_attn_backend_list,
                    self._draft_worker.draft_runner_list,
                )
            ),
        )

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # Target prefill
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch_output = self.target_worker.forward_batch_generation(
                batch, capture_hidden_mode=target_capture_mode
            )

            # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
            # Extend processed L prompt tokens; next verify iter expects same L.
            batch_output.new_seq_lens = batch.seq_lens
            # Publish before draft_extend so the fence is at target-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # Chain-style MTP needs FULL to get all-token hidden states;
            # non-chain only needs LAST (the target model's hidden states).
            batch_output.next_draft_input = self.draft_worker._draft_extend_for_prefill(
                batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
            )
            return batch_output
        else:
            if batch.spec_info is None:
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
                    self.draft_worker.draft_runner
                )
                batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=hidden_size,
                    dtype=hidden_dtype,
                    topk=self.topk * self.speculative_num_steps,
                    capture_hidden_mode=capture_mode,
                )
            verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            batch_output = self.verify(batch)
            # Publish before draft_extend so the fence is at verify-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            self.draft_worker._draft_extend_for_decode(batch, batch_output)
            return batch_output

    def verify(self, batch: ScheduleBatch):
        return run_eagle_verify(
            batch,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=False,
        )
