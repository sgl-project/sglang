import contextlib
import logging
import time
from dataclasses import replace
from typing import List, Optional

import torch

from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
from sglang.srt.layers.moe.utils import (
    draft_model_build_scope,
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner import (
    DecodeCudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.runtime_context import (
    get_context,
    get_device,
    get_exec,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import (
    _eagle_prefill_tail_tokens,
    default_tree_mask_mode,
    get_draft_recurrent_hidden_state_spec,
    organize_draft_results,
    per_step_draft_out_cache_loc,
)
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    prepare_for_draft,
    prepare_for_draft_extend,
    run_eagle_verify,
)
from sglang.srt.speculative.hybrid_chain import (
    HybridChainState,
    build_hybrid_verify_input,
    parse_position_thresholds,
)
from sglang.srt.speculative.hybrid_ragged import (
    HYBRID_RAGGED_TIER_VETO,
    agree_hybrid_ragged_dp_tier,
    build_hybrid_ragged_idle_layout,
    build_hybrid_ragged_layout,
    hybrid_ragged_capture_max_bs,
    hybrid_ragged_capture_tiers,
    hybrid_ragged_dp_tier_enabled,
    plan_hybrid_ragged_tier,
    resolve_hybrid_ragged_tier,
)
from sglang.srt.speculative.hybrid_retrieval import HybridRetriever
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,
    fast_sample,
    get_plan_stream,
    load_token_map,
    renorm_draft_probs,
    sample_draft_proposal,
    select_top_k_tokens,
    spec_stage_span,
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
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    log_info_on_rank0,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_cpu = is_cpu()
_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()
_is_xpu = is_xpu()


logger = logging.getLogger(__name__)


class EagleDraftWorker(EagleDraftWorkerBase):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self.target_worker = target_worker

        # Args for easy access
        self.device = get_device().device
        self.topk = get_spec().speculative_eagle_topk
        if get_spec().speculative_use_rejection_sampling:
            assert self.topk == 1, "Chain speculative sampling supports only topk=1"
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )

        # Hybrid MTP + retrieval (hybrid retrieval). Init-static: the flags are fixed
        # for the process lifetime. When enabled, the draft chain is MTP-headed,
        # truncated per request at the first chain position whose draft prob
        # falls below that position's tau, and refilled by NGRAM retrieval up to
        # L = speculative_num_draft_tokens (> num_steps + 1). See hybrid_chain.
        self.hybrid_enabled = server_args.speculative_hybrid_retrieval
        # When on, the draft-loop step logprobs are written to a graph-safe
        # STATIC buffer instead of a Python list, so the draft-loop and
        # draft-extend cuda graphs can be captured and replayed. Off (default)
        # keeps the draft loop eager (list-based) and captures only the
        # target-verify graph.
        self.hybrid_full_cuda_graph = server_args.speculative_hybrid_full_cuda_graph
        self.hybrid_state: Optional[HybridChainState] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        if self.hybrid_enabled:
            self.hybrid_state = HybridChainState(
                thresholds=parse_position_thresholds(
                    server_args.speculative_hybrid_tau_per_pos,
                    server_args.speculative_hybrid_tau,
                    self.speculative_num_steps,
                ),
                tau_min=server_args.speculative_hybrid_tau_min,
                dynamic_tau=server_args.speculative_hybrid_dynamic_tau,
                device=self.device,
            )
            self.hybrid_retriever = HybridRetriever(server_args, self.device)
        # This step's [bs, num_steps-1] draft-loop logprobs, stashed by
        # draft_forward for the verify-input build (eager path); None on the
        # full-cuda-graph path, where they live in the static buffer below.
        self._hybrid_step_logprobs: Optional[torch.Tensor] = None
        self._hybrid_step_logprobs_buf: Optional[torch.Tensor] = None
        # One-shot warning latch for the non-greedy ragged fallback.
        self._hybrid_ragged_nongreedy_warned = False
        # Whether the graph tier has to be agreed across DP ranks each step.
        # Resolved once: every input is a frozen server arg or a parallel-state
        # constant, and re-deriving it per step would risk the ranks disagreeing
        # about whether the collective happens at all.
        self.hybrid_ragged_dp_tier = hybrid_ragged_dp_tier_enabled(
            server_args=server_args
        )
        # Advances once per step on EVERY rank (the all-gather is
        # unconditional), so it is the only key that lines dump rows up across
        # ranks for the tier-equality check.
        self._hybrid_ragged_dp_step = 0
        self._hybrid_ragged_dp_tier_num_tokens: Optional[int] = None

        # Pre-allocated constants for the topk=1 chain fast path in draft_forward.
        self._topk1_parents_prealloc = None
        self._topk1_score_indices_prealloc = None
        self._rebuild_topk1_chain_buffers()

        # Load draft model weights only.
        if (
            get_parallel().enable_dp_attention
            and self.speculative_algorithm.is_eagle3()
        ):
            ctx = draft_tp_context(get_parallel().attn_tp_group)
        else:
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context(), draft_model_build_scope():
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                # spec workers don't support pipeline parallelism
                ps=replace(ps, pp_rank=0),
                nccl_port=nccl_port,
                is_draft_worker=True,
                # The draft runs at absolute target positions.
                context_length=target_worker.model_runner.model_config.context_len,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
        self._init_dsa_index_share_state()
        # Eager draft-extend seed buffer (graph paths use their own static ones).
        self.dsa_extend_topk_buf: Optional[torch.Tensor] = None
        self.draft_tp_context = (
            draft_tp_context if get_parallel().enable_dp_attention else empty_context
        )
        self.tree_mask_mode = default_tree_mask_mode()

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

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
        self.init_token_map()
        self.init_lm_head()

        if get_spec().speculative_use_rejection_sampling:
            target_vocab_size = self.target_worker.model_config.vocab_size
            draft_vocab_size = (
                self.hot_token_id.shape[0]
                if self.hot_token_id is not None
                else target_vocab_size
            )
            # FIXME: support reduced (hot) draft vocab by scattering draft probs
            # into the target vocab via the d2t map before the sampling kernel.
            if draft_vocab_size != target_vocab_size:
                raise ValueError(
                    "--speculative-use-rejection-sampling requires the draft and "
                    f"target to share one vocab, but the draft vocab "
                    f"({draft_vocab_size}) != target vocab ({target_vocab_size})."
                )

    def init_attention_backends(self):
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker.init_attention_backends()
            self.init_attention_backend()

    def init_cuda_graphs(self):
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
            if check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE):
                self.draft_runner.init_prefill_cuda_graph(force_for_draft_worker=True)
            self._capture_cuda_graphs()

        if (c := self.draft_runner.canary_manager) is not None:
            c.mark_init_finished()

    def _init_dsa_index_share_state(self) -> None:
        # Populate DSA index-share fields from the draft runner's hf_config.
        # Reused by the attention unit-test harnesses, which skip __init__.
        hf_config = self.draft_runner.model_config.hf_config
        # Reuse the first draft step's DSA indexer topk across the rest;
        # topk == 1 only (select_top_k_tokens reorders rows, desyncing indices).
        self.index_share_for_mtp_iteration = (
            getattr(hf_config, "index_share_for_mtp_iteration", False)
            and self.topk == 1
        )
        # GLM-5.2 MTP IndexShare: seed reused indexer top-k from draft-extend
        # (last verified token), not draft-decode step 0.
        self.dsa_index_topk = getattr(hf_config, "index_topk", None)
        self.seed_dsa_topk_from_draft_extend = (
            self.index_share_for_mtp_iteration and self.dsa_index_topk is not None
        )

    def _rebuild_topk1_chain_buffers(self) -> None:
        # For topk=1 the draft tree degenerates to a chain, so parent_list and
        # top_scores_index are runtime-invariant. Must be rebuilt after any
        # change to speculative_num_steps / speculative_num_draft_tokens.
        if self.topk != 1:
            return
        # _override_worker_state can set both directly, bypassing the hook that
        # pins this relation; the fast path is only valid when it holds.
        # Hybrid MTP+retrieval intentionally runs with L = num_draft_tokens >
        # num_steps + 1 (retrieval fills the extra slots). The preallocs below
        # stay width num_steps -- the MTP draft chain is still num_steps deep --
        # and the wider verify chain is synthesized in hybrid_chain.
        if self.hybrid_enabled:
            assert (
                self.speculative_num_draft_tokens >= self.speculative_num_steps + 1
            ), (
                "hybrid topk=1 requires speculative_num_draft_tokens (L) >= "
                f"speculative_num_steps + 1, got L={self.speculative_num_draft_tokens} "
                f"and num_steps={self.speculative_num_steps}"
            )
        else:
            assert (
                self.speculative_num_draft_tokens == self.speculative_num_steps + 1
            ), (
                "topk=1 requires speculative_num_draft_tokens == speculative_num_steps + 1, "
                f"got {self.speculative_num_draft_tokens} and {self.speculative_num_steps}"
            )
        num_steps = self.speculative_num_steps
        decode_max_bs = (
            get_exec().graph.cuda_graph_config.decode.max_bs
            if get_exec().graph.cuda_graph_config is not None
            else None
        )
        max_bs = max(
            decode_max_bs or 0,
            get_schedule().max_running_requests or 0,
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
        # Graph-safe per-step draft-logprob buffer (chain positions 1..num_steps-1).
        # A fixed address so draft_forward's in-place writes are captured by the
        # draft cuda graph and re-executed on replay. Sized to the same max_bs as
        # the chain preallocs (covers eager max_running and captured graph bs).
        if self.hybrid_enabled and self.hybrid_full_cuda_graph and num_steps > 1:
            self._hybrid_step_logprobs_buf = torch.zeros(
                (max_bs, num_steps - 1), dtype=torch.float32, device=self.device
            )
        else:
            self._hybrid_step_logprobs_buf = None

    def init_token_map(self):
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if get_spec().speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif get_spec().speculative_token_map is not None:
            self.hot_token_id = load_token_map(get_spec().speculative_token_map)
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        from sglang.srt.lora.layers import unwrap_lora_layer

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        target_lm_head = unwrap_lora_layer(
            getattr(self.target_worker.model_runner.model, "lm_head", None)
        )

        def maybe_share_target_lm_head():
            if (
                target_lm_head is not None
                and self.hot_token_id is None
                and getattr(self.draft_runner.model, "hot_token_id", None) is None
                and hasattr(self.draft_runner.model, "set_lm_head_from_target")
            ):
                self.draft_runner.model.set_lm_head_from_target(target_lm_head)

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
                maybe_share_target_lm_head()
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
            maybe_share_target_lm_head()

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners

        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
            seed_dsa_topk_from_draft_extend=self.seed_dsa_topk_from_draft_extend,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        if self.draft_extend_attn_backend is not None:
            self.draft_runner.attn_backend = self.draft_extend_attn_backend
        self.tree_mask_mode = default_tree_mask_mode()

    def _capture_cuda_graphs(self):
        """Capture the draft worker's own cuda graphs (decode + draft-extend)."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if _is_cpu or check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return

        if get_model().model_impl == "mindspore":
            return

        # Hybrid MTP+retrieval keeps the DRAFT loop and the draft-extend reseed
        # EAGER so draft_forward can assemble the real per-step draft logprobs
        # the tau truncation reads (a draft-graph replay would skip the Python
        # list that carries them). The expensive TARGET-verify forward is
        # captured independently by the target worker at the fixed width L, so
        # verify still runs as a graph. --speculative-hybrid-full-cuda-graph
        # routes the logprobs through a static buffer instead, which IS
        # capturable, and re-enables both draft graphs.
        if self.hybrid_enabled and not self.hybrid_full_cuda_graph:
            log_info_on_rank0(
                logger,
                "hybrid MTP+retrieval: skipping draft / draft-extend cuda graphs "
                "(eager draft for real per-step logprobs); the target-verify graph "
                "is still captured. Pass --speculative-hybrid-full-cuda-graph to "
                "capture them too.",
            )
            return

        Device2DraftCudaGraphRunner = {
            "xpu": EAGLEDraftCudaGraphRunner,
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        decode_backend = get_exec().graph.cuda_graph_config.decode.backend
        capture_bs, _ = get_batch_sizes_to_capture(self.draft_runner)
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft decode CUDA graph begin. backend={decode_backend}, "
                f"num_tokens_per_req={self.topk}, bs={capture_bs}, "
                f"avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            capture_time = time.perf_counter() - tic
            self._specialized_graph_memory_usage["draft_decode"] = (
                self._specialized_graph_memory_usage.get("draft_decode", 0.0)
                + before_mem
                - after_mem
            )
            self._specialized_graph_time_usage["draft_decode"] = (
                self._specialized_graph_time_usage.get("draft_decode", 0.0)
                + capture_time
            )
            log_info_on_rank0(
                logger,
                "Capture draft decode CUDA graph end. "
                f"elapsed={capture_time:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, "
                f"avail mem={after_mem:.2f} GB.",
            )

        Device2ExtendCudaGraphRunner = {
            "xpu": EAGLEDraftExtendCudaGraphRunner,
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_draft_extend_graph = False
        if _is_hip:
            # Keep imports local so non-HIP environments do not require these.
            # aiter packs draft-extend support into the decode (multi-step)
            # backend; DSV4 exposes it on the draft-extend backend itself.
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4HipRadixBackend,
            )
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            supports_hip_draft_extend_graph = (
                isinstance(self.draft_attn_backend, AiterMultiStepDraftBackend)
                or isinstance(self.draft_extend_attn_backend, DeepseekV4HipRadixBackend)
                or isinstance(self.draft_extend_attn_backend, DeepseekSparseAttnBackend)
            )

        graph_supported_backend_types = [
            TritonAttnBackend,
            TRTLLMMLABackend,
            TRTLLMHAAttnBackend,
            TokenspeedMLABackend,
            FlashInferAttnBackend,
        ]
        if _is_cuda or _is_musa:
            # DSA is CUDA-only; import lazily so non-CUDA builds don't pull in
            # deep_gemm and the rest of the sparse-attention stack at import time.
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            graph_supported_backend_types.append(DeepseekSparseAttnBackend)
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4AttnBackend,
            )

            graph_supported_backend_types.append(DeepseekV4AttnBackend)
        if _is_cuda:
            # FlashMLA is CUDA-only; import lazily so CPU builds don't pull
            # sgl_kernel.flash_mla at import time.
            from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

            graph_supported_backend_types.append(FlashMLABackend)

        graph_supported_backend = isinstance(
            self.draft_extend_attn_backend,
            tuple(graph_supported_backend_types),
        )
        supports_cuda_draft_extend_graph = (
            _is_cuda or _is_musa
        ) and graph_supported_backend
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        if (
            self.draft_extend_attn_backend
            and not envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH.get()
            and (
                _is_npu
                or _is_xpu
                or supports_cuda_draft_extend_graph
                or supports_hip_draft_extend_graph
            )
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend CUDA graph begin. backend={decode_backend}, "
                f"num_tokens_per_req={self.speculative_num_draft_tokens}, "
                f"bs={capture_bs}, avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            # draft_extend is the step's last shared-buffer-reading phase; its
            # read-done event is what the scheduler's WAR barrier waits on.
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            capture_time = time.perf_counter() - tic
            self._specialized_graph_memory_usage["draft_extend"] = (
                self._specialized_graph_memory_usage.get("draft_extend", 0.0)
                + before_mem
                - after_mem
            )
            self._specialized_graph_time_usage["draft_extend"] = (
                self._specialized_graph_time_usage.get("draft_extend", 0.0)
                + capture_time
            )
            log_info_on_rank0(
                logger,
                "Capture draft extend CUDA graph end. "
                f"elapsed={capture_time:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, "
                f"avail mem={after_mem:.2f} GB.",
            )

    def draft(self, batch: ScheduleBatch):
        # Drop any stash left by a previous iteration (an idle batch never
        # reaches the hybrid build, so its [0, ...] tensor would otherwise be
        # consumed by the next real one). draft_forward re-populates it.
        self._hybrid_step_logprobs = None
        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_run_decode_cuda_graph = prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )
        if (
            can_run_decode_cuda_graph
            and not forward_batch.forward_mode.is_idle()
            and self.seed_dsa_topk_from_draft_extend
            and draft_input.dsa_topk_indices is None
        ):
            can_run_decode_cuda_graph = False

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
            if can_run_decode_cuda_graph:
                parent_list, top_scores_index, draft_tokens, draft_probs = (
                    self.cuda_graph_runner.execute(forward_batch)
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
                parent_list, top_scores_index, draft_tokens, draft_probs = (
                    self.draft_forward(forward_batch)
                )

        if self.hybrid_enabled and not batch.forward_mode.is_idle():
            verify_input = self._build_hybrid_verify_input(
                batch, draft_input, draft_tokens
            )
        else:
            verify_input = build_eagle_verify_input(
                batch,
                draft_input,
                parent_list,
                top_scores_index,
                draft_tokens,
                draft_probs,
                target_worker=self.target_worker,
                topk=self.topk,
                num_steps=self.speculative_num_steps,
                num_draft_tokens=self.speculative_num_draft_tokens,
                tree_mask_mode=self.tree_mask_mode,
                device=self.device,
            )

        # BOTH branches, including the idle one: under DP attention the graph
        # tier is a collective, and an idle rank has to join it.
        return self._finalize_hybrid_ragged_layout(batch, verify_input)

    def _build_hybrid_verify_input(
        self,
        batch: ScheduleBatch,
        draft_input: EagleDraftInput,
        mtp_draft_tokens: torch.Tensor,
    ) -> EagleVerifyInput:
        """Truncate the MTP chain by per-position tau and refill it to width L
        with an agreement-gated retrieval continuation."""
        bs = mtp_draft_tokens.shape[0]
        if self.hybrid_full_cuda_graph:
            # Written by draft_forward (captured + replayed under the graph).
            step_logprobs = (
                self._hybrid_step_logprobs_buf[:bs]
                if self._hybrid_step_logprobs_buf is not None
                else None
            )
        else:
            step_logprobs, self._hybrid_step_logprobs = self._hybrid_step_logprobs, None
            assert step_logprobs is not None or self.speculative_num_steps == 1, (
                "hybrid MTP+retrieval needs the eager topk=1 chain with real draft "
                "logprobs, but draft_forward produced none (cuda-graph replay or "
                "oversized-batch slow path). Use --speculative-hybrid-full-cuda-graph "
                "for the graph path."
            )
        assert step_logprobs is None or step_logprobs.shape[0] == bs, (
            "hybrid MTP+retrieval: draft logprob rows "
            f"({step_logprobs.shape[0]}) do not match the batch ({bs}); the "
            "draft-loop stash is stale."
        )

        # The trie walk was launched on a worker thread before draft_forward so
        # the CPU match overlaps the GPU draft; join it here. The retriever
        # reports its own continuation lengths -- they cannot be recovered from
        # the chain, because the padding value 0 is a legal token id.
        retrieval_chains, retrieval_lens = self.hybrid_retriever.collect()
        if retrieval_chains is not None:
            retrieval_chains = retrieval_chains.to(mtp_draft_tokens.dtype)

        return build_hybrid_verify_input(
            batch=batch,
            draft_input=draft_input,
            mtp_draft_tokens=mtp_draft_tokens,
            step_logprobs=step_logprobs,
            retrieval_chains=retrieval_chains,
            retrieval_lens=retrieval_lens,
            state=self.hybrid_state,
            topk=self.topk,
            verify_width=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,
            target_worker=self.target_worker,
            ragged=self._hybrid_ragged_enabled_for(batch),
        )

    def _hybrid_ragged_enabled_for(self, batch: ScheduleBatch) -> bool:
        """Whether THIS step runs the ragged (variable-length) verify forward.

        The accept cutoff is a port of DSPARK's ``CapCorrectLen``, which caps a
        GREEDY ``correct_len``. The sampling paths
        (``tree_speculative_sampling_target_only`` / rejection sampling) walk the
        tree themselves and would need a per-position draft distribution to cap
        equivalently -- which the topk=1 chain does not have (``topk_p`` is
        fused to a constant 1.0). Rather than fail the request, fall back to the
        fixed-width forward for that step: correctness is identical, only the
        token saving is lost. Startup already rejects the CONFIGURATIONS that can
        never be greedy (see ``_check_hybrid_ragged_server_args``); this handles a
        single non-greedy request landing in an otherwise greedy deployment.
        """
        if not self.server_args.speculative_hybrid_ragged:
            return False
        if batch.forward_mode.is_idle():
            return False
        sampling_info = batch.sampling_info
        if sampling_info is None or not sampling_info.is_all_greedy:
            if not self._hybrid_ragged_nongreedy_warned:
                self._hybrid_ragged_nongreedy_warned = True
                logger.warning(
                    "speculative_hybrid_ragged: batch is not all-greedy; this "
                    "verify step falls back to the fixed-width forward. Output "
                    "is unaffected; the ragged token saving is not taken."
                )
            return False
        return True

    def _finalize_hybrid_ragged_layout(
        self, batch: ScheduleBatch, verify_input: EagleVerifyInput
    ) -> EagleVerifyInput:
        """Attach the ``RaggedVerifyLayout`` once the graph tier is settled.

        Split out of ``build_hybrid_verify_input`` for one reason: the tier is a
        CROSS-RANK agreement under DP attention, and an idle rank -- which never
        builds a hybrid chain -- has to take part in it and then replay the same
        captured graph. So both of ``draft()``'s branches land here, and this is
        the single place that writes ``ragged_verify_layout``.

        Three outcomes:
          * layout with a tier      -- replayable on the token-keyed verify graph
          * layout without a tier   -- exact ``graph_num_tokens == Σ w_r``; the
                                      verify runs eager (eager mode's path, and what a
                                      ``--disable-cuda-graph`` run gets)
          * no layout               -- plain fixed-width verify
        """
        if not self.server_args.speculative_hybrid_ragged:
            return verify_input

        tier_grid = hybrid_ragged_capture_tiers(target_worker=self.target_worker)
        accept_lens_cpu = verify_input.ragged_accept_verify_lens_cpu

        if self.hybrid_ragged_dp_tier:
            tier = self._agree_hybrid_ragged_dp_tier(
                batch=batch,
                accept_lens_cpu=accept_lens_cpu,
                tier_grid=tier_grid,
            )
            if tier is None:
                # Some rank vetoed (or nobody had work). Every rank reaches this
                # from the same gathered vector, so the whole group drops to
                # fixed width together -- which is mandatory, not conservative:
                # a compact batch that does NOT replay a graph goes through
                # `prepare_mlp_sync_batch`, whose DP padding stretches input_ids
                # to `bs * num_tokens_per_req` and would not match the compact
                # attention metadata.
                return self._clear_hybrid_ragged(verify_input)
            if accept_lens_cpu is None:
                # Idle rank: no requests, but it still has to enter the graph.
                # This branch can ONLY be the idle case: a non-idle rank with no
                # accept widths (a non-greedy batch) votes VETO, which forces the
                # agreed tier to None and returns above.
                assert batch.forward_mode.is_idle(), (
                    "ragged DP tier agreed but this rank has no accept widths and "
                    "is not idle; its veto should have taken the whole group to "
                    "fixed width"
                )
                verify_input.ragged_verify_layout = build_hybrid_ragged_idle_layout(
                    tier_num_tokens=tier, target_worker=self.target_worker
                )
                return verify_input
        elif accept_lens_cpu is None:
            return self._clear_hybrid_ragged(verify_input)
        else:
            # Non-DP. Two differences from the DP branch, both because nothing
            # here GUARANTEES the graph will be admitted:
            #
            #  * the tier is capped at `bs * L`. Above that, plan_hybrid_ragged_tier
            #    cannot spend all the slack, and if the verify then falls back to
            #    eager the substrate folds the remainder into the LAST REAL
            #    request -- widening it past L and past its reserved KV window.
            #    (On the graph path that same tier is fine: captured slots exceed
            #    bs, so the remainder lands on pad requests.) When no grid member
            #    fits under the cap, tier is None and the step runs exact+eager,
            #    which costs a graph but is always safe.
            #  * if the runner would not admit this batch at all, do not build a
            #    tier -- same predicate the DP vote uses, so the two cannot drift.
            admits = self._hybrid_ragged_graph_admits(
                batch=batch, accept_lens_cpu=accept_lens_cpu, tier_grid=tier_grid
            )
            tier = (
                resolve_hybrid_ragged_tier(
                    total_verify_tokens=sum(accept_lens_cpu),
                    tier_grid=tier_grid,
                    max_tier=len(accept_lens_cpu) * self.speculative_num_draft_tokens,
                )
                if admits
                else None
            )

        plan = (
            plan_hybrid_ragged_tier(
                accept_verify_lens=accept_lens_cpu,
                verify_width=self.speculative_num_draft_tokens,
                tier_num_tokens=tier,
            )
            if tier is not None
            else None
        )
        forward_lens = plan.forward_verify_lens if plan else accept_lens_cpu
        layout = build_hybrid_ragged_layout(
            verify_lens_cpu=forward_lens,
            device=self.device,
            tier_num_tokens=tier,
        )
        verify_input.ragged_verify_layout = layout
        return verify_input

    def _clear_hybrid_ragged(self, verify_input: EagleVerifyInput) -> EagleVerifyInput:
        """Drop every ragged field so this step is byte-for-byte fixed width.

        The accept widths have to go too, not just the layout: `eagle_sample`
        keys the cutoff off the layout, so a leftover width vector would be dead
        weight that a later reader could mistake for "ragged ran".
        """
        verify_input.ragged_verify_layout = None
        verify_input.ragged_accept_verify_lens = None
        verify_input.ragged_accept_verify_lens_cpu = None
        return verify_input

    def _agree_hybrid_ragged_dp_tier(
        self,
        *,
        batch: ScheduleBatch,
        accept_lens_cpu: Optional[List[int]],
        tier_grid: Optional[List[int]],
    ) -> Optional[int]:
        """This rank's vote, then the group's tier. Runs on EVERY rank, every step.

        Unconditional participation is the whole point: the all-gather is the
        only thing keeping the ranks on one graph shape, so a rank that skipped
        it (or voted on a locally-derived predicate) would leave the others
        blocked in the collective. `is_extend_in_batch` and
        `can_run_dp_cuda_graph` are already dp-global (max- and min-reduced by
        the MLP sync), so voting them is redundant for agreement -- but it is
        free, and it keeps the "graph will run" precondition in one place instead
        of implicitly spread across the runner's admission checks.
        """
        self._hybrid_ragged_dp_step += 1
        local = self._hybrid_ragged_dp_vote(
            batch=batch, accept_lens_cpu=accept_lens_cpu, tier_grid=tier_grid
        )
        tier = agree_hybrid_ragged_dp_tier(
            local_total_verify_tokens=local,
            tier_grid=tier_grid,
            # NOT get_tp_group(): draft() runs inside draft_tp_context under DP,
            # which patches the global tp group to the draft runner's (world_size
            # 1 at attn_tp_size == 1). See agree_hybrid_ragged_dp_tier's docstring.
            cpu_group=self.target_worker.model_runner.tp_group.cpu_group,
        )
        self._hybrid_ragged_dp_tier_num_tokens = tier
        return tier

    def _hybrid_ragged_graph_admits(
        self,
        *,
        batch: ScheduleBatch,
        accept_lens_cpu: Optional[List[int]],
        tier_grid: Optional[List[int]],
    ) -> bool:
        """Will the decode runner admit this step's ragged verify to a graph?

        The PER-RANK, per-step subset of `can_run_graph` /
        `_can_run_ragged_verify_graph` that is knowable here. It has two callers
        that must never disagree:

          * the DP vote -- a rank that would not be admitted has to veto, or it
            runs its compact batch eager (through prepare_mlp_sync_batch's DP
            padding) while its peers replay the graph, and the group hangs;
          * the non-DP tier choice -- a tier above `bs * verify_width` is only
            safe when the graph runs, because only then do captured PAD requests
            exist to absorb the slack. On the eager path the substrate folds it
            into the last real request instead.

        Conditions deliberately NOT here because they are process-static or
        dp-global rather than per-rank: `supports_ragged_verify_graph`, the
        capture_hidden_mode match, `can_run_dp_cuda_graph` and
        `is_extend_in_batch` (the last two are voted separately, since they are
        dp-global by construction and belong to the DP decision).
        """
        if accept_lens_cpu is None or not tier_grid:
            return False
        if batch.replace_embeds is not None:
            # can_run_graph's FIRST test, before it ever reaches the ragged
            # branch, so the vote has to cover it or the two drift apart. It is a
            # PER-RANK property: only ranks holding a request with a
            # token-embedding override carry it, so one rank refusing the graph
            # must take the whole group to fixed width.
            #
            # `prepare_for_decode` now clears the overrides (they address prefill
            # rows), so on a decode-family batch this is normally already None.
            # Keep the test anyway: it is the union with can_run_graph, and if a
            # future field ever survives into decode again, the group still falls
            # back together instead of splitting.
            return False
        if sum(accept_lens_cpu) > tier_grid[-1]:
            return False
        if len(accept_lens_cpu) > hybrid_ragged_capture_max_bs(
            target_worker=self.target_worker
        ):
            # The runner's `batch_size <= _ragged_capture_slots(tier)` =
            # `min(tier, max_bs)`. Every w_r >= 1 makes the tier >= bs, so it
            # reduces exactly to `bs <= max_bs` -- and max_bs = max(capture_bs)
            # comes from --cuda-graph-max-bs, which is INDEPENDENT of
            # --max-running-requests.
            return False
        return True

    def _hybrid_ragged_dp_vote(
        self,
        *,
        batch: ScheduleBatch,
        accept_lens_cpu: Optional[List[int]],
        tier_grid: Optional[List[int]],
    ) -> int:
        if batch.forward_mode.is_idle():
            # Nothing to verify, no objection: an idle rank rides whatever tier
            # the busy ranks agree on.
            return 0
        if accept_lens_cpu is None:
            # Ragged is off for this step on this rank (non-greedy batch). A 0
            # here would read as "no objection" and let the peers go ragged while
            # this rank runs fixed width -- different graph, hung group.
            return HYBRID_RAGGED_TIER_VETO
        if batch.is_extend_in_batch or not batch.can_run_dp_cuda_graph:
            return HYBRID_RAGGED_TIER_VETO
        if not self._hybrid_ragged_graph_admits(
            batch=batch, accept_lens_cpu=accept_lens_cpu, tier_grid=tier_grid
        ):
            return HYBRID_RAGGED_TIER_VETO
        return sum(accept_lens_cpu)

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        if forward_batch.forward_mode.is_idle():
            return self._draft_forward_idle(forward_batch, spec_info)

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
        if get_spec().speculative_use_rejection_sampling:
            draft_probs_list: List[torch.Tensor] = [spec_info.draft_probs]

        topk1_chain_fits = (
            self.topk == 1
            and topk_index.shape[0] <= self._topk1_parents_prealloc.shape[0]
        )
        # Materialize the chain directly only when the CUDA kernel can write
        # every subsequent column. Other topk=1 paths retain the token list and
        # assemble it with one final cat instead of launching a copy per step.
        draft_tokens_topk1 = None
        if (
            topk1_chain_fits
            and _is_cuda
            and self.hot_token_id is None
            and not get_spec().speculative_use_rejection_sampling
        ):
            draft_tokens_topk1 = torch.empty(
                (topk_index.shape[0], self.speculative_num_steps),
                dtype=topk_index.dtype,
                device=topk_index.device,
            )
            draft_tokens_topk1[:, :1].copy_(topk_index)

        # Real per-step draft logprobs on the topk=1 chain fast path (production
        # fakes topk_p to 1.0 inside the fused kernel). The hybrid chain uses
        # them for confidence-based truncation.
        hybrid_on = self.hybrid_enabled and draft_tokens_topk1 is not None
        step_logprobs_buf = self._hybrid_step_logprobs_buf if hybrid_on else None
        step_lp_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        with IndexTopKShareState.mtp_iteration(
            forward_batch,
            enabled=self.index_share_for_mtp_iteration,
            keep_carry_seed=self.seed_dsa_topk_from_draft_extend,
        ):
            for i in range(self.speculative_num_steps):
                if draft_tokens_topk1 is not None:
                    input_ids = topk_index.flatten()
                else:
                    input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                        i, topk_p, topk_index, hidden_states, scores, self.topk
                    )
                    score_list.append(tree_info[0])
                    token_list.append(tree_info[1])
                    parents_list.append(tree_info[2])

                if i == self.speculative_num_steps - 1:
                    break

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
                    logits_output = self.draft_runner.forward(
                        forward_batch
                    ).logits_output
                maybe_detect_nan(
                    logits_output.next_token_logits, f"draft_forward step {i}"
                )
                maybe_detect_inf(
                    logits_output.next_token_logits, f"draft_forward step {i}"
                )
                if get_spec().speculative_use_rejection_sampling:
                    probs, topk_p, topk_index = sample_draft_proposal(
                        logits_output.next_token_logits,
                        forward_batch.sampling_info.temperatures,
                    )
                    draft_probs_list.append(probs)
                    forward_batch.positions.add_(1)
                elif self.topk == 1 and not _is_hip:
                    if _is_cuda:
                        topk_p, topk_index = draft_topk1_postprocess(
                            logits_output.next_token_logits,
                            forward_batch.positions,
                            draft_tokens_topk1,
                            i + 1,
                        )
                    else:
                        topk_index = torch.argmax(
                            logits_output.next_token_logits, dim=-1, keepdim=True
                        )
                        topk_p = torch.ones_like(topk_index, dtype=torch.float32)
                        forward_batch.positions.add_(1)
                else:
                    probs = renorm_draft_probs(
                        logits_output.next_token_logits,
                        forward_batch.sampling_info,
                        get_spec().speculative_use_rejection_sampling,
                    )
                    topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
                    forward_batch.positions.add_(1)
                if hybrid_on:
                    # log p_draft(argmax token) = gather - logsumexp; avoids
                    # materializing the full [bs, vocab] log_softmax. This
                    # forward produces chain position i + 1.
                    step_logits = logits_output.next_token_logits
                    step_lp = torch.gather(
                        step_logits, -1, topk_index
                    ) - torch.logsumexp(step_logits, dim=-1, keepdim=True)
                    if step_logprobs_buf is not None:
                        # In-place write into the graph-safe static buffer
                        # (column i = chain position i + 1), so the draft cuda
                        # graph captures it and replays it.
                        step_logprobs_buf[: step_lp.shape[0], i] = step_lp.squeeze(-1)
                    else:
                        step_lp_list.append(step_lp)
                maybe_detect_oob(
                    topk_index,
                    0,
                    logits_output.next_token_logits.shape[-1],
                    f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
                )
                if self.hot_token_id is not None:
                    topk_index = self.hot_token_id[topk_index]
                hidden_states = logits_output.hidden_states

        draft_probs = (
            torch.stack(draft_probs_list, dim=1)
            if get_spec().speculative_use_rejection_sampling
            else None
        )

        step_logprobs = torch.cat(step_lp_list, dim=1) if step_lp_list else None
        if hybrid_on and step_logprobs_buf is None:
            # Stash for the hybrid verify-input build (draft -> build is
            # sequential within this forward step).
            self._hybrid_step_logprobs = step_logprobs

        # Organize the results
        if draft_tokens_topk1 is not None:
            bs = draft_tokens_topk1.shape[0]
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            return parent_list, top_scores_index, draft_tokens_topk1, draft_probs

        if topk1_chain_fits:
            bs = token_list[0].shape[0]
            draft_tokens = torch.cat(token_list, dim=1)
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            return parent_list, top_scores_index, draft_tokens, draft_probs

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens, draft_probs

    def _draft_forward_idle(
        self, forward_batch: ForwardBatch, spec_info: EagleDraftInput
    ):
        """Run eager idle-rank collectives without materializing draft state."""
        input_ids = forward_batch.input_ids
        out_cache_loc = forward_batch.out_cache_loc
        hidden_states = spec_info.hidden_states

        # ModelRunner pads and unpads the empty batch on every call. Avoid the
        # normal tree/cache-layout path: idle outputs are discarded when the
        # verify input is built, but every rank must still enter each forward.
        for i in range(self.speculative_num_steps - 1):
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc
            spec_info.hidden_states = hidden_states
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
                self.draft_runner.forward(forward_batch)

        return None, None, None, None

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
            new_input_ids = torch.empty_like(batch.input_ids)
            pt = 0
            for i, extend_len in enumerate(batch.extend_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                new_input_ids[pt : pt + extend_len].copy_(
                    torch.cat((input_ids[1:], tail_tokens[i].reshape(1)))
                )
                pt += extend_len
            assert pt == batch.input_ids.numel()
            batch.input_ids = new_input_ids

        # Draft-extend spec_info for the extend forward; carries only
        # hidden_states + shape info.
        batch.spec_info = EagleDraftExtendInput(
            hidden_states=target_hidden_states,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        # Run forward (LAST mode: only the final hidden state per request,
        # to feed the next draft step which expects [bs, hidden_dim]).
        # STANDALONE skips hidden states end-to-end.
        capture_hidden_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        forward_batch = ForwardBatch.init_new(
            batch,
            self.draft_runner,
            capture_hidden_mode=capture_hidden_mode,
            return_hidden_states_before_norm=False,
        )
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds

        # Seed the first draft-decode loop from each request's last prefill
        # position. Gather last-per-req before the copy (prefill can be long).
        seed_from_extend = (
            self.seed_dsa_topk_from_draft_extend
            and not forward_batch.forward_mode.is_idle()
        )
        if seed_from_extend:
            bs = forward_batch.batch_size
            forward_batch.spec_info.dsa_seed_topk_capture = (
                self._get_dsa_extend_topk_buf(bs)
            )
            forward_batch.spec_info.dsa_seed_topk_select = (
                torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            ).long()

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

        prefill_dsa_topk = None
        if seed_from_extend:
            prefill_dsa_topk = self.dsa_extend_topk_buf[:bs].clone()

        # Assemble the next-iter draft spec_info from the extend output.
        use_rejection_sampling = get_spec().speculative_use_rejection_sampling
        probs = renorm_draft_probs(
            logits_output.next_token_logits,
            batch.sampling_info,
            use_rejection_sampling,
        )
        if use_rejection_sampling:
            topk_p, topk_index = fast_sample(probs, num_samples=1)
        else:
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        return EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            draft_probs=probs if use_rejection_sampling else None,
            hidden_states=logits_output.hidden_states,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            dsa_topk_indices=prefill_dsa_topk,
        )

    def _get_dsa_extend_topk_buf(self, num_tokens: int) -> torch.Tensor:
        """Lazily-grown int32 [num_tokens, index_topk] eager draft-extend seed buffer."""
        buf = self.dsa_extend_topk_buf
        if buf is None or buf.shape[0] < num_tokens:
            buf = torch.full(
                (num_tokens, self.dsa_index_topk),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self.dsa_extend_topk_buf = buf
        return buf[:num_tokens]

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        # Batch 2: Draft extend
        draft_extend_input = EagleDraftExtendInput(
            hidden_states=batch_result.logits_output.hidden_states,
            # accept_lens includes the bonus token; correct drafts exclude it.
            num_correct_drafts=batch_result.accept_lens - 1,
            num_accept_tokens=batch_result.accept_lens,
            # Draft-extend fills the whole tree width (num_draft_tokens) per req,
            # not num_steps + 1, so DP MLP-sync padding stays consistent for topk > 1.
            num_tokens_per_req=self.speculative_num_draft_tokens,
            num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
        )
        select_index = (
            torch.arange(
                0,
                len(batch.seq_lens) * self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                device=self.device,
            )
            + batch_result.accept_lens
            - 1
        )

        # Cast to int64 before entering plan stream to avoid cross-stream
        # synchronization issues with .to() inside the plan stream context.
        next_token_ids = batch_result.next_token_ids.to(torch.int64)

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = prepare_for_draft_extend(
                draft_extend_input,
                batch,
                next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
                return_hidden_states_before_norm=False,
            )

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        # Run draft extend batch in the main compute stream
        can_run_decode_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run_graph(forward_batch)
        )

        # Eager path publishes the indexer top-k into a worker buffer (the graph
        # path uses the runner's static buffer). Gathered at select_index below.
        if self.seed_dsa_topk_from_draft_extend and not can_run_decode_cuda_graph:
            forward_batch.spec_info.dsa_seed_topk_capture = (
                self._get_dsa_extend_topk_buf(forward_batch.input_ids.shape[0])
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
            if can_run_decode_cuda_graph:
                draft_logits_output = self.cuda_graph_runner_for_draft_extend.execute(
                    forward_batch
                )
            else:
                draft_logits_output = self.draft_runner.forward(
                    forward_batch
                ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
        )
        maybe_detect_inf(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
        )

        # Gather the per-request last-position indexer top-k as the next loop's
        # seed (select_index already picks the last accepted position per req).
        dsa_seed_topk_indices = None
        if self.seed_dsa_topk_from_draft_extend:
            if can_run_decode_cuda_graph:
                dsa_extend_topk_capture = (
                    self.cuda_graph_runner_for_draft_extend.buffers.dsa_seed_topk_capture
                )
            else:
                dsa_extend_topk_capture = forward_batch.spec_info.dsa_seed_topk_capture
            # Fancy indexing returns a fresh tensor (detached from the buffer).
            dsa_seed_topk_indices = dsa_extend_topk_capture[select_index]

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
        if get_spec().speculative_use_rejection_sampling:
            ret_draft_probs, ret_topk_p, ret_topk_index = sample_draft_proposal(
                draft_logits_output.next_token_logits,
                batch.sampling_info.temperatures,
            )
        elif self.topk == 1 and not _is_hip:
            # Gated to CUDA: see #26358 — ROCm's argmax tie-break corrupts
            # MTP draft selection on FP8 logits.
            ret_topk_index = torch.argmax(
                draft_logits_output.next_token_logits, dim=-1, keepdim=True
            )
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
            ret_draft_probs = None
        else:
            probs = renorm_draft_probs(
                draft_logits_output.next_token_logits,
                batch.sampling_info,
                get_spec().speculative_use_rejection_sampling,
            )
            ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
            ret_draft_probs = None
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
        if get_spec().speculative_use_rejection_sampling:
            next_draft_input.draft_probs = ret_draft_probs
        if self.seed_dsa_topk_from_draft_extend:
            next_draft_input.dsa_topk_indices = dsa_seed_topk_indices


class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        # Parse arguments
        self.server_args = server_args
        self.topk = get_spec().speculative_eagle_topk
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.ps = ps
        self.gpu_id = gpu_id
        self.device = get_device().device
        self._target_worker = target_worker
        self.page_size = get_schedule().page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )

        self._draft_worker = EagleDraftWorker(
            server_args,
            gpu_id,
            ps,
            nccl_port,
            target_worker,
        )

        # Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None
        if get_spec().speculative_adaptive:
            self.adaptive_controller = AdaptiveController(
                self,
                config_path=get_spec().speculative_adaptive_config,
            )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

    def clear_cache_pool(self):
        """Clear hybrid retrieval's process-local online retrieval state."""
        dw = self._draft_worker
        if dw.hybrid_retriever is not None:
            dw.hybrid_retriever.reset()

        state = dw.hybrid_state
        if state is not None:
            state.extension_ema = 0.0
            state.last_num_keep_drafts = None
            state.last_graft_ok = None

        # The static graph buffer is deliberately retained: it has a
        # capture-stable address and every active slice is overwritten before use.
        dw._hybrid_step_logprobs = None
        logger.info("Hybrid NGRAM corpus and retrieval state reset")

    @property
    def last_shared_read_runner(self):
        # Per the base contract: the step's last shared-buffer-reading phase is
        # draft_extend, which runs on the draft runner.
        return self._draft_worker.draft_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # Every attn backend a spec_v2 forward touches; consumed by
        # decide_needs_cpu_seq_lens to gate the seq_lens_cpu D2H.
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend
            or self._draft_worker.draft_runner.attn_backend,
        )

    def init_cuda_graphs(self):
        super().init_cuda_graphs()
        # Build adaptive runtime states after target and draft backends exist.
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
                        target_graph_runner=self._target_worker.model_runner.decode_cuda_graph_runner,
                        draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                        cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
                    )
                )
                self.adaptive_controller.init_states(
                    cuda_graph_bs=(
                        None
                        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)
                        else get_exec().graph.cuda_graph_bs_decode
                    ),
                )

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None, grammar_barrier=None
    ):
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

            # Draft prefill
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft_extend"),
            ):
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
                # Index the full prompt into the trie so hybrid retrieval can
                # copy from the prompt (RAG / long-context / code-with-context).
                # The decode-time insert only adds sliding windows of generated
                # text, so without this the prompt body is never matchable.
                if (
                    self.draft_worker.hybrid_retriever is not None
                    and self.server_args.speculative_hybrid_index_prompt
                    and not batch.forward_mode.is_idle()
                ):
                    self.draft_worker.hybrid_retriever.insert_prompt(batch.reqs)
                return batch_output
        else:
            self.activate_step_by_batch(batch.seq_lens.shape[0])

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
                    topk=self.topk,
                    capture_hidden_mode=capture_mode,
                    vocab_size=self.target_worker.model_config.vocab_size,
                )
            if self.speculative_num_steps == 0:
                # Drafting disabled (high batch size). _draft_extend below still
                # runs, keeping draft KV warm for when the batch shrinks.
                verify_input = self._build_trivial_verify_input(batch)
            else:
                # Kick off the NGRAM trie walk on a worker thread BEFORE the GPU
                # draft_forward so the CPU match overlaps it; the hybrid
                # verify-input build joins it.
                if (
                    self.draft_worker.hybrid_retriever is not None
                    and not batch.forward_mode.is_idle()
                ):
                    self.draft_worker.hybrid_retriever.launch_query(batch.reqs)
                with (
                    self.draft_worker.draft_tp_context(
                        self.draft_worker.draft_runner.tp_group
                    ),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft"),
                ):
                    verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            batch_output = self.verify(batch, grammar_barrier=grammar_barrier)
            # Publish before draft_extend so the fence is at verify-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            # Must run after on_publish (seq_lens published first) and before
            # the next launch_query drains the relay.
            if not batch.forward_mode.is_idle():
                self._hybrid_after_verify(batch, batch_output)
            if (
                self.speculative_num_steps == 0
                and envs.SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND.get()
            ):
                self._stub_skipped_draft_extend(batch, batch_output)
            else:
                with (
                    self.draft_worker.draft_tp_context(
                        self.draft_worker.draft_runner.tp_group
                    ),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft_extend"),
                ):
                    self.draft_worker._draft_extend_for_decode(batch, batch_output)

            # Fold the verified context into the trie AFTER querying it
            # (query-first / insert-after mirrors NGRAMWorker, so a query
            # matches earlier occurrences of the suffix, not the just-written
            # leaf), then drop match state for finished/retracted requests.
            if (
                self.draft_worker.hybrid_retriever is not None
                and not batch.forward_mode.is_idle()
            ):
                self.draft_worker.hybrid_retriever.insert(batch.reqs)
                self.draft_worker.hybrid_retriever.erase(batch.reqs)

            return batch_output

    def _hybrid_after_verify(
        self, batch: ScheduleBatch, batch_output: GenerationBatchResult
    ) -> None:
        """Post-verify hybrid bookkeeping: the overlap accepted-token relay and
        the retrieval-extension EMA that drives dynamic tau."""
        retriever = self.draft_worker.hybrid_retriever
        if retriever is None:
            return
        if retriever.overlap:
            # Relay this step's accepted tokens into the retriever's shadow
            # context via an async D2H, off the forward hot path.
            # next_token_ids / accept_lens are the same tensors the scheduler
            # commits from; the stride is speculative_num_draft_tokens (= L).
            retriever.publish_accepted(
                batch.reqs,
                batch_output.next_token_ids,
                batch_output.accept_lens,
                batch_output.speculative_num_draft_tokens,
            )
        state = self.draft_worker.hybrid_state
        if state is not None and state.dynamic_tau:
            state.update_extension_ema(
                num_correct_drafts=batch_output.accept_lens - 1,
                keep_on_device=retriever.overlap,
            )

    def _build_trivial_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """Build a 1-node EagleVerifyInput rooted at the previous bonus token.

        Used when ``speculative_num_steps == 0`` to skip drafting while still
        routing through the existing TARGET_VERIFY graph captured at
        ``draft_token_num=1``: the kernel always accepts the root and samples
        one new bonus token from target logits -- functionally a plain decode.
        """
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                topk=self.topk, spec_steps=0, num_verify_tokens=1, device=self.device
            )

        draft_input: EagleDraftInput = batch.spec_info
        bs = batch.seq_lens.shape[0]
        device = self.device

        retrieve_index = torch.arange(bs, dtype=torch.long, device=device).unsqueeze(1)
        retrieve_next_token = torch.full((bs, 1), -1, dtype=torch.long, device=device)
        retrieve_next_sibling = torch.full((bs, 1), -1, dtype=torch.long, device=device)

        attn_backend = self._target_worker.model_runner.attn_backend
        verify_mask = attn_backend.verify_mask
        # Every position in a 1-node tree is visible, so an all-True fill is
        # correct under either layout.
        if verify_mask is not None and verify_mask.fits(bs):
            custom_mask = verify_mask.buffer
            custom_mask.fill_(True)
        else:
            if batch.seq_lens_sum is not None:
                seq_lens_sum = batch.seq_lens_sum
            elif batch.seq_lens_cpu is not None:
                seq_lens_sum = int(batch.seq_lens_cpu.sum())
            else:
                seq_lens_sum = bs * attn_backend.max_context_len
            custom_mask = torch.ones(seq_lens_sum + bs, dtype=torch.bool, device=device)

        positions = batch.seq_lens.to(torch.int64)

        return EagleVerifyInput(
            draft_token=draft_input.bonus_tokens,
            custom_mask=custom_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=0,
            topk=self.topk,
            draft_token_num=1,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def _stub_skipped_draft_extend(
        self, batch: ScheduleBatch, batch_output: GenerationBatchResult
    ) -> None:
        """Fill shape-valid stubs on next_draft_input when draft_extend is skipped.

        ``verify`` already set ``bonus_tokens`` (the only field the next steps=0
        verify reads). The overlap FutureMap still stashes topk_p/topk_index/
        hidden_states, so provide zeroed tensors of the right shape. They are never
        consumed while at steps=0; an upshift to steps>0 would draft from this stale
        state (cold recovery), which is the documented cost of this experimental flag.
        """
        next_draft_input: EagleDraftInput = batch_output.next_draft_input
        bs = batch.seq_lens.shape[0]
        device = self.device
        next_draft_input.topk_p = torch.zeros(
            (bs, self.topk), dtype=torch.float32, device=device
        )
        next_draft_input.topk_index = torch.zeros(
            (bs, self.topk), dtype=torch.int64, device=device
        )
        hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
            self.draft_worker.draft_runner
        )
        if hidden_size is not None:
            next_draft_input.hidden_states = torch.zeros(
                (bs, hidden_size),
                dtype=hidden_dtype,
                device=device,
            )

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
            self._draft_worker._capture_cuda_graphs()

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
            if not check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
                TargetGraphRunnerCls = (
                    NPUGraphRunner if _is_npu else DecodeCudaGraphRunner
                )
                target_graph_before_mem = get_available_gpu_memory(
                    self.device, self.gpu_id
                )
                target_graph_tic = time.perf_counter()
                target_graph_runner = TargetGraphRunnerCls(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )
                target_graph_after_mem = get_available_gpu_memory(
                    self.device, self.gpu_id
                )
                target_graph_time = time.perf_counter() - target_graph_tic
                self._additional_graph_memory_usage["target_verify"] = (
                    self._additional_graph_memory_usage.get("target_verify", 0.0)
                    + target_graph_before_mem
                    - target_graph_after_mem
                )
                self._additional_graph_time_usage["target_verify"] = (
                    self._additional_graph_time_usage.get("target_verify", 0.0)
                    + target_graph_time
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
        # Keep the runner's attn_backend in step with the active draft-extend
        # backend (the draft-extend forward reads draft_runner.attn_backend);
        # mirrors init_attention_backend. When None, the runner keeps its
        # initialized backend (consistent across step configs).
        if state.draft_extend_attn_backend is not None:
            dw.draft_runner.attn_backend = state.draft_extend_attn_backend
        dw.cuda_graph_runner_for_draft_extend = state.cuda_graph_runner_for_draft_extend
        dw._rebuild_topk1_chain_buffers()

        # Target side
        self._target_worker.model_runner.attn_backend = state.target_attn_backend
        self._target_worker.model_runner.decode_cuda_graph_runner = (
            state.target_graph_runner
        )

        # Sync server_args
        get_context().override(
            "adaptive_spec.restore",
            speculative_num_steps=state.speculative_num_steps,
            speculative_num_draft_tokens=state.speculative_num_draft_tokens,
        )

    @contextlib.contextmanager
    def _override_worker_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ):
        """Temporarily override server_args and worker attributes for graph capture."""
        dw = self._draft_worker
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            dw.speculative_num_steps,
            dw.speculative_num_draft_tokens,
            dw.draft_attn_backend,
            dw.draft_extend_attn_backend,
            dw.draft_runner.draft_attn_backend,
            dw.draft_runner.attn_backend,
            dw.cuda_graph_runner,
            dw.cuda_graph_runner_for_draft_extend,
            get_spec().speculative_num_steps,
            get_spec().speculative_num_draft_tokens,
            get_exec().graph.cuda_graph_bs_decode,
            get_exec().graph.disable_cuda_graph,
        )

        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        dw.speculative_num_steps = speculative_num_steps
        dw.speculative_num_draft_tokens = speculative_num_draft_tokens
        get_context().override(
            "adaptive_spec.capture_override",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        if cuda_graph_bs is not None:
            # BS-aware adaptive spec may prune cuda_graph_bs to an empty list
            # for steps that no BS range uses (e.g. step=1). Disable graph
            # capture for those steps; restore in finally so subsequent steps
            # are not affected.
            get_context().override(
                "adaptive_spec.capture_override",
                cuda_graph_bs_decode=cuda_graph_bs,
                **({"disable_cuda_graph": True} if not cuda_graph_bs else {}),
            )
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
                dw.draft_runner.attn_backend,
                dw.cuda_graph_runner,
                dw.cuda_graph_runner_for_draft_extend,
            ) = backup[:10]
            get_context().override(
                "adaptive_spec.capture_restore",
                speculative_num_steps=backup[10],
                speculative_num_draft_tokens=backup[11],
                cuda_graph_bs_decode=backup[12],
                disable_cuda_graph=backup[13],
            )
            dw._rebuild_topk1_chain_buffers()

    def verify(self, batch: ScheduleBatch, grammar_barrier=None):
        return run_eagle_verify(
            batch,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            grammar_barrier=grammar_barrier,
        )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.ps.tp_rank]
        )
        success, message = (
            self.draft_worker.draft_runner.weight_updater.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=recv_req.load_format,
            )
        )
        if not success:
            return success, message

        success, message = (
            self.target_worker.model_runner.weight_updater.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=recv_req.load_format,
            )
        )
        return success, message
