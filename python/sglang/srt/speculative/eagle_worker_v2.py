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
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
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
    fill_bonus_tokens,
)
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    _eagle_prefill_tail_tokens,
    build_tree_kernel_efficient,
    organize_draft_results,
    per_step_draft_out_cache_loc,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,
    generate_token_bitmask,
    load_token_map,
    record_stream_each,
    record_stream_for_v2_verify,
    select_top_k_tokens,
)
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import (
    MultiprocessingSerializer,
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    log_info_on_rank0,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()

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
        attn_cp_rank: int,
        moe_dp_rank: int,
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
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Pre-allocated constants for the topk=1 chain fast path in draft_forward.
        self._topk1_parents_prealloc = None
        self._topk1_score_indices_prealloc = None
        self._rebuild_topk1_chain_buffers()

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Init draft worker
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            # Init draft worker
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # spec workers don't support pipeline parallelism
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            eagle_config = getattr(
                self.draft_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        self.init_token_map()
        self.init_lm_head()

        # Init attention backend and cuda graphs
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.init_attention_backend()
            if server_args.enable_breakable_cuda_graph:
                self.draft_runner.init_piecewise_cuda_graphs(
                    force_for_draft_worker=True
                )
            self.init_cuda_graphs()

        if (c := self.draft_runner.canary_manager) is not None:
            c.mark_init_finished()

        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def _rebuild_topk1_chain_buffers(self) -> None:
        # For topk=1 the draft tree degenerates to a chain, so parent_list and
        # top_scores_index are runtime-invariant. Must be rebuilt after any
        # change to speculative_num_steps / speculative_num_draft_tokens.
        if self.topk != 1:
            return
        # _override_worker_state can set both directly, bypassing the hook that
        # pins this relation; the fast path is only valid when it holds.
        assert self.speculative_num_draft_tokens == self.speculative_num_steps + 1, (
            "topk=1 requires speculative_num_draft_tokens == speculative_num_steps + 1, "
            f"got {self.speculative_num_draft_tokens} and {self.speculative_num_steps}"
        )
        num_steps = self.speculative_num_steps
        sa = self.server_args
        max_bs = max(
            sa.cuda_graph_max_bs or 0,
            sa.max_running_requests or 0,
            1,
        )
        # A single-step chain has no parent entries (slow path drops the last
        # step). repeat (not expand): the kernel reads these as contiguous.
        parent_width = num_steps if num_steps > 1 else 0
        self._topk1_parents_prealloc = torch.arange(
            -1, parent_width - 1, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)
        self._topk1_score_indices_prealloc = torch.arange(
            num_steps, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)

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

        if self.server_args.model_impl == "mindspore":
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB.",
            )

        Device2ExtendCudaGraphRunner = {
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_aiter_draft_extend_graph = False
        if _is_hip:
            # Keep import local so non-HIP environments do not require aiter.
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )

            supports_hip_aiter_draft_extend_graph = isinstance(
                self.draft_attn_backend, AiterMultiStepDraftBackend
            )

        supports_cuda_draft_extend_graph = (_is_cuda or _is_musa) and isinstance(
            self.draft_extend_attn_backend,
            (
                TritonAttnBackend,
                TRTLLMMLABackend,
                TRTLLMHAAttnBackend,
                TokenspeedMLABackend,
            ),
        )
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        if self.draft_extend_attn_backend and (
            _is_npu
            or supports_cuda_draft_extend_graph
            or supports_hip_aiter_draft_extend_graph
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB.",
            )

    def draft(self, batch: ScheduleBatch):
        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        n_inner = self.speculative_num_steps - 1
        canary_outside_ctx = (
            c.with_ops_outside_graph(
                single_forward_indices=list(range(n_inner)),
                maybe_inaccurate_forward_batch=forward_batch,
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )

        with canary_outside_ctx:
            # Run draft
            if can_cuda_graph:
                parent_list, top_scores_index, draft_tokens = (
                    self.cuda_graph_runner.replay(forward_batch)
                )
            else:
                if (
                    not forward_batch.forward_mode.is_idle()
                    and self.speculative_num_steps > 1
                ):
                    # Skip attention backend init for 1-step draft,
                    # `draft_forward` only does sample in this case.
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                    forward_batch.mark_forward_metadata_ready()
                parent_list, top_scores_index, draft_tokens = self.draft_forward(
                    forward_batch
                )

        if batch.forward_mode.is_idle():
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

        # build_tree_kernel uses seq_lens_sum only to size the (non-preallocated)
        # tree mask; over-size is safe. Skip per-iter .sum().item() D2H via UB.
        seq_lens_sum = batch.seq_lens_sum
        if seq_lens_sum is None:
            if tree_mask_buf is None:
                max_context_len = (
                    self.target_worker.model_runner.attn_backend.max_context_len
                )
                seq_lens_sum = batch.seq_lens.shape[0] * max_context_len
            else:
                # tree_mask_buf preallocated -> kernel ignores seq_lens_sum.
                seq_lens_sum = 0

        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            seq_lens_sum,
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
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
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

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = per_step_draft_out_cache_loc(
            out_cache_loc,
            forward_batch.batch_size,
            self.topk,
            self.speculative_num_steps,
        )

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
            # Qwen3-MoE MTP uses a fused RoPE + KV-store path whose cache_loc
            # argument must be contiguous.
            if (
                self.draft_runner.model_config.hf_config.architectures[0]
                == "Qwen3MoeForCausalLMMTP"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            spec_info.hidden_states = hidden_states

            # Run forward under a per-step ForwardContext so the model layer
            # reads attn_backends[i] for the i-th draft step, plus a canary
            # index context so canary tracks which draft step is active.
            canary_index_ctx = (
                c.with_active_single_forward_manager(i)
                if (c := self.draft_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            with (
                forward_context(
                    ForwardContext(
                        attn_backend=self.draft_attn_backend.attn_backends[i]
                    )
                ),
                canary_index_ctx,
            ):
                logits_output = self.draft_runner.forward(forward_batch).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            maybe_detect_inf(logits_output.next_token_logits, f"draft_forward step {i}")
            if self.topk == 1 and not _is_hip:
                # topk=1 → degenerate single-path tree; `topk_p` is unused
                # downstream, so skip softmax and just argmax over logits.
                # Gated to CUDA: on ROCm the argmax tie-break diverges from
                # the softmax+max path on FP8 logits and corrupts MTP draft
                # selection (DSV3.2 MTP GSM8K, see #26358).
                topk_index = torch.argmax(
                    logits_output.next_token_logits, dim=-1, keepdim=True
                )
                topk_p = torch.ones_like(topk_index, dtype=torch.float32)
            else:
                probs = torch.softmax(logits_output.next_token_logits, dim=-1)
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states
            forward_batch.positions.add_(1)

        # Organize the results
        if (
            self.topk == 1
            and token_list[0].shape[0] <= self._topk1_parents_prealloc.shape[0]
        ):
            # Chain topology: draft_tokens = concat of per-step tokens; the
            # full-length topk/sort/gather over score_list collapses to an
            # identity. parent_list and top_scores_index are runtime-invariant
            # constants pre-allocated on the worker. Oversized batches (rare,
            # would silently truncate the slice) fall through to the slow path.
            bs = token_list[0].shape[0]
            draft_tokens = torch.cat(token_list, dim=1)
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            return parent_list, top_scores_index, draft_tokens

        return organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
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
            # Chunked-prefill-aware tail tokens (see PR #26329).
            tail_tokens = _eagle_prefill_tail_tokens(batch, next_token_ids)
            pt = 0
            for i, extend_len in enumerate(batch.extend_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (input_ids[1:], tail_tokens[i].reshape(1))
                )
                pt += extend_len

        # Construct spec_info
        next_draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            bonus_tokens=next_token_ids,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        batch.spec_info = next_draft_input

        # Run forward (LAST mode: only the final hidden state per request,
        # to feed the next draft step which expects [bs, hidden_dim]).
        # STANDALONE skips hidden states end-to-end.
        capture_hidden_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        batch.capture_hidden_mode = capture_hidden_mode
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            logits_output = self.draft_runner.forward(forward_batch).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")
        maybe_detect_inf(logits_output.next_token_logits, "draft_extend_for_prefill")

        # Update spec_info for the next draft step
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        next_draft_input.topk_p, next_draft_input.topk_index = fast_topk(
            probs, self.topk, dim=-1
        )
        next_draft_input.hidden_states = logits_output.hidden_states
        return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        # Batch 2: Draft extend
        draft_input = EagleDraftInput(
            hidden_states=batch_result.logits_output.hidden_states,
            # Draft-extend fills the whole tree width (num_draft_tokens) per req,
            # not num_steps + 1, so DP MLP-sync padding stays consistent for topk > 1.
            num_tokens_per_req=self.speculative_num_draft_tokens,
            num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
        )
        select_index = (
            torch.arange(len(batch.seq_lens), device=self.device)
            * self.speculative_num_draft_tokens
            + batch_result.accept_lens
            - 1
        )

        # Prepare for draft extend in a separate stream
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

        if forward_batch.spec_info.num_correct_drafts is None:
            # `batch_result.accept_lens` already includes the bonus token, so use it
            # directly for `num_accept_tokens` and subtract 1 for `num_correct_drafts`.
            forward_batch.spec_info.num_correct_drafts = batch_result.accept_lens - 1
            forward_batch.spec_info.num_accept_tokens = batch_result.accept_lens

        # Run draft extend batch in the main compute stream
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            if can_cuda_graph:
                draft_logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                    forward_batch
                )
            else:
                draft_logits_output = self.draft_runner.forward(
                    forward_batch
                ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
        )
        maybe_detect_inf(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
        )

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        if draft_logits_output.hidden_states is not None:
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]
        # The draft-extend graph only anchors full logits; selected-row topk is
        # owned by the worker for both graph and eager paths.
        if self.topk == 1 and not _is_hip:
            # Gated to CUDA: see #26358 — ROCm's argmax tie-break corrupts
            # MTP draft selection on FP8 logits.
            ret_topk_index = torch.argmax(
                draft_logits_output.next_token_logits, dim=-1, keepdim=True
            )
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
        else:
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
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
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
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None
        if server_args.speculative_adaptive:
            self.adaptive_controller = AdaptiveController(
                self,
                config_path=server_args.speculative_adaptive_config,
            )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

        # Build adaptive runtime states (must be after draft worker is fully initialized)
        if self.adaptive_controller is not None:
            with (
                self._draft_worker.draft_tp_context(
                    self._draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.adaptive_controller.register(
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        draft_attn_backend=self._draft_worker.draft_attn_backend,
                        cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                        target_attn_backend=self._target_worker.model_runner.attn_backend,
                        target_graph_runner=self._target_worker.model_runner.graph_runner,
                        draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                        cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
                    )
                )
                self.adaptive_controller.init_states(
                    cuda_graph_bs=(
                        None
                        if self.server_args.disable_cuda_graph
                        else self.server_args.cuda_graph_bs
                    ),
                )

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # Every attn backend a spec_v2 forward touches; consumed by
        # decide_needs_cpu_seq_lens to gate the seq_lens_cpu D2H.
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend,
        )

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # Target prefill
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch.capture_hidden_mode = target_capture_mode
            batch_output = self.target_worker.forward_batch_generation(batch)

            # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
            # Extend processed L prompt tokens; next verify iter expects same L.
            batch_output.new_seq_lens = batch.seq_lens
            # Publish before draft_extend so the fence is at target-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # Draft prefill
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
                return batch_output
        else:
            self.activate_step_by_batch(batch.seq_lens.shape[0])

            if batch.spec_info is None:
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=EagleDraftInput.hidden_size_for(self.draft_worker),
                    dtype=EagleDraftInput.dtype_for(self.draft_worker),
                    topk=self.topk,
                    capture_hidden_mode=capture_mode,
                )
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            batch_output = self.verify(batch)
            # Publish before draft_extend so the fence is at verify-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.draft_worker._draft_extend_for_decode(batch, batch_output)

            return batch_output

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(
                num_correct_drafts_per_req, batch_size=batch_size
            )

    def activate_step_by_batch(self, batch_size: int) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.activate_step_by_batch(batch_size)

    # -- Adaptive speculative decoding protocol --

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs=None,
    ) -> SpecRuntimeState:
        """Build a SpecRuntimeState for the given step configuration."""
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        with self._override_worker_state(
            speculative_num_steps,
            speculative_num_draft_tokens,
            cuda_graph_bs=cuda_graph_bs,
        ):
            self._draft_worker.init_attention_backend()
            self._draft_worker.init_cuda_graphs()

            # Build target attention backend and CUDA graph runner
            target_model_runner = self._target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                target_model_runner.init_new_workspace = backup_init

            target_graph_runner = None
            if not self.server_args.disable_cuda_graph:
                TargetGraphRunnerCls = NPUGraphRunner if _is_npu else CudaGraphRunner
                target_graph_runner = TargetGraphRunnerCls(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )

            state = SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                draft_attn_backend=self._draft_worker.draft_attn_backend,
                cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )

        return state

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        """Apply a pre-built runtime state to this worker."""
        if self.speculative_num_steps == state.speculative_num_steps:
            return

        log_info_on_rank0(
            logger,
            "Switch adaptive runtime state: "
            f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
            f"draft_tokens {self.speculative_num_draft_tokens} -> "
            f"{state.speculative_num_draft_tokens}",
        )

        # Top-level
        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens

        # Draft side
        dw = self._draft_worker
        dw.speculative_num_steps = state.speculative_num_steps
        dw.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        dw.draft_attn_backend = state.draft_attn_backend
        dw.draft_runner.draft_attn_backend = state.draft_attn_backend
        dw.cuda_graph_runner = state.cuda_graph_runner
        dw.draft_extend_attn_backend = state.draft_extend_attn_backend
        dw.cuda_graph_runner_for_draft_extend = state.cuda_graph_runner_for_draft_extend
        dw._rebuild_topk1_chain_buffers()

        # Target side
        self._target_worker.model_runner.attn_backend = state.target_attn_backend
        self._target_worker.model_runner.graph_runner = state.target_graph_runner

        # Sync server_args
        self.server_args.speculative_num_steps = state.speculative_num_steps
        self.server_args.speculative_num_draft_tokens = (
            state.speculative_num_draft_tokens
        )

    @contextlib.contextmanager
    def _override_worker_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ):
        """Temporarily override server_args and worker attributes for graph capture."""
        sa = self.server_args
        dw = self._draft_worker
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            dw.speculative_num_steps,
            dw.speculative_num_draft_tokens,
            dw.draft_attn_backend,
            dw.draft_extend_attn_backend,
            dw.draft_runner.draft_attn_backend,
            dw.cuda_graph_runner,
            dw.cuda_graph_runner_for_draft_extend,
            sa.speculative_num_steps,
            sa.speculative_num_draft_tokens,
            sa.cuda_graph_bs,
            sa.disable_cuda_graph,
        )

        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        dw.speculative_num_steps = speculative_num_steps
        dw.speculative_num_draft_tokens = speculative_num_draft_tokens
        sa.speculative_num_steps = speculative_num_steps
        sa.speculative_num_draft_tokens = speculative_num_draft_tokens
        if cuda_graph_bs is not None:
            sa.cuda_graph_bs = cuda_graph_bs
            # BS-aware adaptive spec may prune cuda_graph_bs to an empty list
            # for steps that no BS range uses (e.g. step=1). Disable graph
            # capture for those steps; restore in finally so subsequent steps
            # are not affected.
            if not cuda_graph_bs:
                sa.disable_cuda_graph = True
        dw._rebuild_topk1_chain_buffers()

        try:
            yield
        finally:
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                dw.speculative_num_steps,
                dw.speculative_num_draft_tokens,
                dw.draft_attn_backend,
                dw.draft_extend_attn_backend,
                dw.draft_runner.draft_attn_backend,
                dw.cuda_graph_runner,
                dw.cuda_graph_runner_for_draft_extend,
                sa.speculative_num_steps,
                sa.speculative_num_draft_tokens,
                sa.cuda_graph_bs,
                sa.disable_cuda_graph,
            ) = backup
            dw._rebuild_topk1_chain_buffers()

    def verify(self, batch: ScheduleBatch):
        fwd_stream = torch.get_device_module(self.device).current_stream()
        verify_input: EagleVerifyInput = batch.spec_info
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)

        verify_input.num_tokens_per_req = self.speculative_num_steps + 1
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

        # Cover post-prepare rebinds: draft_token, plan_stream-allocated out_cache_loc.
        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)

        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
            if (
                _is_npu
                and self._target_worker.model_runner.model_is_mrope
                and batch.spec_info is not None
                and getattr(batch.spec_info, "positions", None) is not None
                and not batch.forward_mode.is_idle()
            ):
                # mrope_position depends on draft output in default stream and is computed in plan stream,
                # causing errors. Compute it here for correct values.
                verify_forward_batch.compute_spec_mrope_positions(
                    self._target_worker.model_runner, batch
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
            retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrieve_next_token.shape
            ).cpu()

        # Run target verify batch in the main compute stream (GPU compute).
        # Metadata init is skipped iff cuda-graph already ran replay_prepare —
        # prepare_for_v2_verify marked the batch in exactly that case; the
        # non-cuda-graph path stays unmarked and gets forward_extend's init
        # (post-pad).
        forward_batch_output = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
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
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        # Sample
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        maybe_detect_inf(logits_output.next_token_logits, "verify: target model logits")
        (
            predict,
            accept_lens,
            accept_index,
        ) = verify_input.sample(batch, logits_output, vocab_mask)
        new_seq_lens = batch.seq_lens + accept_lens

        # Update mamba state for hybrid GDN models after verification
        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            self._mamba_verify_update(
                batch, verify_input, accept_lens, accept_index, bs
            )

        if not batch.forward_mode.is_idle():
            accept_tokens = predict[accept_index]
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
            # stride = accept_tokens per-req width = accept_index.shape[1]
            # (spec_steps + 1); NOT num_draft_tokens, wrong for topk > 1 trees.
            fill_bonus_tokens[(bs,)](
                accept_tokens,
                accept_lens,
                bonus_tokens,
                accept_index.shape[1],
            )
        else:
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not batch.forward_mode.is_idle():
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        if not batch.forward_mode.is_idle() and self.topk > 1:
            # topk == 1 needs nothing here: the accepted path is already the front
            # chain, so the whole compaction is an identity transform.
            predict = self._finalize_accepted_tree_path(
                batch, accept_index, accept_lens, predict, logits_output, bs
            )

        next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)

        # verify_forward_batch transitively holds verify-time GPU tensors
        # (draft_token / out_cache_loc / ...) that must outlive the imminent
        # batch.input_ids rebind in prepare_for_extend_to_fill_draft_kvcache.
        # Scheduler pins it in batch_record_buf for the 2-iter window.
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )

    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,
        verify_input: EagleVerifyInput,
        accept_lens: torch.Tensor,
        accept_index: torch.Tensor,
        bs: int,
    ):
        """Update mamba state for hybrid GDN models after verification."""
        # `accept_lens` already includes the bonus token (drafts + 1 per req).
        if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
            if verify_input.topk != 1:
                raise ValueError("Spec v2 currently only supports topk = 1.")

            accepted_indices_offset = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=accept_lens.dtype,
                device=accept_lens.device,
            )
            last_correct_step_indices = accept_lens - 1

            if batch.mamba_track_indices is not None:
                # If after verify, the request's seq_lens has crossed a mamba track interval,
                # we need to update the mamba state for the request at the crossing point.
                seq_lens_pre_verify = batch.seq_lens
                seq_lens_post_verify = batch.seq_lens + accept_lens
                mamba_track_interval = self.server_args.mamba_track_interval
                to_track_mask = (
                    seq_lens_pre_verify // mamba_track_interval
                    != seq_lens_post_verify // mamba_track_interval
                )
                tracking_point = (
                    seq_lens_post_verify // mamba_track_interval * mamba_track_interval
                )
                to_track_ith = torch.clamp(
                    tracking_point - seq_lens_pre_verify - 1, min=0
                ).to(torch.int64)
                req_idx = torch.arange(
                    bs,
                    dtype=torch.int64,
                    device=accept_lens.device,
                )
                candidate_track_steps = (
                    accept_index[req_idx, to_track_ith] - accepted_indices_offset
                )
                mamba_steps_to_track = torch.where(
                    to_track_mask,
                    candidate_track_steps,
                    torch.full_like(candidate_track_steps, -1),
                )
            else:
                mamba_steps_to_track = None

            self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=self.target_worker.model_runner.model,
            )

    def _finalize_accepted_tree_path(
        self,
        batch: ScheduleBatch,
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
        predict: torch.Tensor,
        logits_output,
        bs: int,
    ) -> torch.Tensor:
        """Tree drafting (topk > 1): move the accepted path -- KV slots, predict,
        hidden_states -- to the contiguous front of each per-req block, which the
        downstream chain-layout code (draft-extend select_index, committed-KV reads)
        assumes. Returns compacted predict; mutates logits_output.hidden_states
        (moved only when present)."""
        self.move_accepted_tokens_to_target_kvcache(
            batch, accept_index, accept_lens - 1
        )
        predict = self._compact_accepted_to_front(predict, accept_index, bs)
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = self._compact_accepted_to_front(
                logits_output.hidden_states, accept_index, bs
            )
        return predict

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ScheduleBatch,
        accept_index: torch.Tensor,
        num_correct_drafts: torch.Tensor,
    ):
        """
        Move accepted tokens (drafts + bonus) to the target KV cache.

        Args:
            batch: The batch to run.
            accept_index: The index of the accepted tokens (incl. bonus).
            num_correct_drafts: Per-req count of correct drafts (excludes bonus);
                seq_lens is advanced by ``num_correct_drafts + 1`` to cover the bonus slot.
        """
        bs = len(batch.seq_lens)
        # accept_index element count, NOT bs * num_draft_tokens: for topk > 1 the
        # tree exceeds the accepted chain, over-reading accept_index (illegal memory).
        size = bs * accept_index.shape[1]

        # fill_accepted_out_cache_loc reads out_cache_loc[accept_index]; -1 sentinel ok.
        maybe_detect_oob(
            accept_index,
            -1,
            batch.out_cache_loc.size(0),
            "eagle v2 move_accepted_tokens accept_index",
        )

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
            batch.seq_lens + num_correct_drafts + 1,
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

    def _compact_accepted_to_front(
        self, x: torch.Tensor, accept_index: torch.Tensor, bs: int
    ) -> torch.Tensor:
        """Gather the accepted tree path to the front of each per-req block.

        ``x`` is node-indexed over the whole tree (``[bs * num_draft_tokens, ...]``),
        ``accept_index`` is ``[bs, spec_steps + 1]`` global node indices (-1 padded).
        Padded entries clamp to node 0 but land past accept_lens (never read);
        trailing unaccepted slots stay and are freed as overshoot.
        """
        nd = self.speculative_num_draft_tokens
        s1 = accept_index.shape[1]  # spec_steps + 1
        safe = accept_index.to(torch.int64).clamp(min=0).reshape(-1)
        gathered = x[safe]
        out = x.clone()
        out.view(bs, nd, *x.shape[1:])[:, :s1] = gathered.view(bs, s1, *x.shape[1:])
        return out

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_ipc(
            recv_req
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

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
