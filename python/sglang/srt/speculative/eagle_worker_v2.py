import contextlib
import logging
import time
from typing import List, Optional, Tuple, Dict

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.attention.triton_backend import TritonMultiStepDraftBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLAMultiStepDraftBackend,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
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
)
from sglang.srt.speculative.eagle_mab import (
    MABConfig,
    MABGroupManager,
    SpeculativeResources,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    detect_nan,
    draft_tp_context,
    generate_token_bitmask,
    load_token_map,
    select_top_k_tokens,
)
from sglang.srt.utils.common import (
    MultiprocessingSerializer,
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_cuda,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()
_is_cuda = is_cuda()

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
        self.max_topk = self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.strategy_min_bs = None
        self.strategy_max_bs = None

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

        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

        # Adaptive SD configuration
        self.adaptive_spec_threshold = (
            server_args.adaptive_speculative_batch_size_threshold
        )
        self.adaptive_spec_warmup_checks = 10  # Number of consecutive checks to enable SD
        self.adaptive_spec_consecutive_checks = (
            0  # Counter for consecutive qualifying checks
        )
        self.adaptive_spec_enabled = False  # Track if speculation is currently enabled
        self.pending_transition_to_sd = (
            False  # Flag to trigger requeue before next batch
        )
        if self.adaptive_spec_threshold is not None:
            logger.info(
                f"[AdaptiveSpec] Initialized with threshold={self.adaptive_spec_threshold}. "
                f"Speculative decoding will be disabled until batch_size <= {self.adaptive_spec_threshold} "
                f"for {self.adaptive_spec_warmup_checks} consecutive decode batches."
            )

        self.use_mab = bool(server_args.speculative_eagle_mab_configs)
        if server_args.speculative_eagle_mab_configs:
            self._init_mab_configurations()
        else:
            with self.draft_tp_context(
                self.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.init_attention_backend()
                self.init_cuda_graphs()

        if self.adaptive_spec_threshold is not None:
            self.init_target_normal_decode_graph()
            # After capturing normal decode graphs, ensure spec decode backend is properly initialized
            # This is critical for V2 worker to avoid attention backend issues during first prefill
            target_runner = self.target_worker.model_runner
            target_runner.spec_algorithm = self.speculative_algorithm
            target_runner.init_attention_backend()

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
        # TODO: support draft extend cuda graph for more attention backends
        if self.draft_extend_attn_backend and (
            _is_npu
            or (
                _is_cuda
                and isinstance(self.draft_attn_backend, TritonMultiStepDraftBackend)
            )
            or (
                _is_cuda
                and isinstance(self.draft_attn_backend, TRTLLMMLAMultiStepDraftBackend)
            )
        ):
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

    def init_target_normal_decode_graph(self):
        """Initialize CUDA graph for normal decoding on the target worker."""
        if self.server_args.disable_cuda_graph:
            return

        logger.info("Capture target normal decode cuda graph begin.")
        target_runner = self.target_worker.model_runner

        # Backup
        old_spec_algo = target_runner.spec_algorithm
        old_attn_backend = target_runner.attn_backend
        old_graph_runner = target_runner.graph_runner

        try:
            # Switch to normal decode mode
            target_runner.spec_algorithm = SpeculativeAlgorithm.NONE

            # Init attention backend for normal decode
            target_runner.init_attention_backend()
            self.target_normal_decode_attn_backend = target_runner.attn_backend

            # Init cuda graph for normal decode
            # Only capture for batch sizes > threshold where we expect to run normal decode
            min_bs = (
                (self.adaptive_spec_threshold + 1)
                if self.adaptive_spec_threshold is not None
                else 1
            )
            target_runner.init_device_graphs(strategy_min_bs=min_bs)
            self.target_normal_decode_graph_runner = target_runner.graph_runner

        finally:
            # Restore
            target_runner.spec_algorithm = old_spec_algo
            target_runner.attn_backend = old_attn_backend
            target_runner.graph_runner = old_graph_runner

        logger.info("Capture target normal decode cuda graph end.")

    def reset_adaptive_spec_params(self):
        """Reset adaptive speculative decoding state."""
        self.adaptive_spec_consecutive_checks = 0
        self.adaptive_spec_enabled = False
        self.pending_transition_to_sd = False

    def check_and_trigger_transition(self):
        if self.adaptive_spec_threshold is None:
            return False

        if self.pending_transition_to_sd:
            self.pending_transition_to_sd = False
            return True
        return False

    def should_enable_sd(self, batch: ModelWorkerBatch) -> bool:
        """Check whether speculation should be enabled for this batch.

        When adaptive_spec_threshold is None, always returns True (normal speculative decoding).
        When adaptive_spec_threshold is set, implements adaptive speculation based on batch size.

        Returns:
            True if speculation should be enabled, False to skip speculation
        """
        # If no adaptive spec configured (threshold=None), always enable speculation
        if self.adaptive_spec_threshold is None:
            return True

        # If transition is pending, do not enable SD until after abort
        if self.pending_transition_to_sd:
            return False

        # If already permanently enabled, keep it enabled
        if self.adaptive_spec_enabled:
            return True

        # Check batch size
        batch_size = len(batch.seq_lens)

        # For EXTEND batches, always run (to establish draft model state)
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return True

        # For DECODE batches, check threshold with warmup
        current_enable_sd = False
        if batch_size <= self.adaptive_spec_threshold:
            self.adaptive_spec_consecutive_checks += 1

            if self.adaptive_spec_consecutive_checks >= self.adaptive_spec_warmup_checks:
                if not self.adaptive_spec_enabled:
                    logger.debug(
                        f"[AdaptiveSpec] ENABLING speculation after "
                        f"{self.adaptive_spec_consecutive_checks} checks (batch_size={batch_size}). "
                        f"Will requeue requests for EXTEND. Running this batch as non-SD."
                    )
                    self.adaptive_spec_enabled = True
                    self.pending_transition_to_sd = True
                    return False
                else:
                    return True
        else:
            # Reset counter if batch size exceeds threshold
            if self.adaptive_spec_consecutive_checks > 0:
                logger.info(
                    f"[AdaptiveSpec] Batch size {batch_size} > threshold {self.adaptive_spec_threshold}, "
                    f"resetting counter (was {self.adaptive_spec_consecutive_checks})."
                )
            self.adaptive_spec_consecutive_checks = 0
            current_enable_sd = False

        return current_enable_sd

    def get_current_mab_strategy(self):
        """Get MAB strategy string from current speculative decoding settings."""
        return MABConfig.format_config(
            self.speculative_num_steps,
            self.topk,
            self.server_args.speculative_num_draft_tokens,
        )

    def update_speculative_args(self, mab_strategy: str):
        """Update speculative settings of workers."""
        steps, topk, draft_tokens = MABConfig.parse_config(mab_strategy)
        self.speculative_num_steps = steps
        self.topk = topk
        self.speculative_num_draft_tokens = draft_tokens
        self.server_args.speculative_num_draft_tokens = draft_tokens
        self.target_worker.model_runner.server_args.speculative_num_steps = steps
        self.target_worker.model_runner.server_args.speculative_eagle_topk = topk
        self.target_worker.model_runner.server_args.speculative_num_draft_tokens = (
            draft_tokens
        )
        self.last_mab_strategy = mab_strategy

    def set_mab_strategy(self, mab_strategy: str):
        """Apply MAB strategy by updating speculative decoding settings."""
        if mab_strategy == self.last_mab_strategy:
            return

        # Get cuda graph and attn for the strategy
        resources = self.strategy_resources.get(mab_strategy)
        if resources is None:
            raise ValueError(f"No resources found for strategy: {mab_strategy}")

        self.draft_attn_backend = resources.draft_attn_backend
        self.cuda_graph_runner = resources.draft_cuda_graph_runner
        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        self.target_worker.model_runner.attn_backend = resources.target_attn_backend
        self.target_worker.model_runner.graph_runner = resources.target_cuda_graph_runner

        # Update speculative decoding settings
        self.update_speculative_args(mab_strategy)

    def select_mab_strategy(self, batch_size: int) -> str:
        """Select and apply MAB strategy for the given batch.

        Args:
            batch_size: The batch size of the current requests.

        Returns:
            Selected MAB strategy string.
        """
        if not self.use_mab or len(self.mab_strategies) == 1:
            return self.default_mab_strategy

        # Use the MAB manager to select the best strategy for this batch size
        selected_strategy = self.mab_manager.select_strategy(batch_size)

        # Apply the selected strategy
        self.set_mab_strategy(selected_strategy)

        return selected_strategy

    def record_mab_strategy_metrics(self, events, accept_length_per_req_cpu):
        """Record performance metrics for the current MAB strategy.

        Args:
            events: CUDA events dictionary.
            accept_length_per_req_cpu: Accept length per request in CPU.
        """
        # Calculate total processing time in seconds
        torch.cuda.synchronize()
        total_time = (
            events["processing_start"].elapsed_time(events["processing_end"]) / 1000.0
        )

        # Calculate metrics
        batch_size = self.mab_last_pull["batch_size"]
        mab_strategy = self.mab_last_pull["mab_strategy"]
        accept_length_avg = sum(accept_length_per_req_cpu) / batch_size + 1
        stable_accept_length = (
            self.mab_manager.get_stable_accept_length(mab_strategy)
            if len(self.mab_strategies) > 1
            else accept_length_avg
        )
        reward = stable_accept_length * batch_size / total_time

        # Update metrics in MAB manager
        self.mab_manager.record_strategy_metrics(
            batch_size, mab_strategy, reward, accept_length_avg
        )

    def _init_mab_configurations(self):
        """Initialize MAB configuration settings from server arguments."""
        self.default_mab_strategy = self.get_current_mab_strategy()
        self.mab_algorithm = self.server_args.speculative_eagle_mab_algorithm
        self.last_mab_strategy = None

        # Initialize resources for the default strategy
        self.mab_strategies = [self.default_mab_strategy]
        self.strategy_resources: Dict[str, SpeculativeResources] = {}

        # Parse additional MAB strategies if provided
        self.mab_strategies.extend(self.server_args.speculative_eagle_mab_configs)
        self.mab_strategies = sorted(list(set(self.mab_strategies)))

        # Calculate max topk needed across all strategies
        max_topk_needed = max(
            MABConfig.parse_config(strategy)[1] for strategy in self.mab_strategies
        )
        self.max_topk = max(self.max_topk, max_topk_needed)

        # Set window size for MAB metrics
        self.mab_window_size = self.server_args.speculative_mab_window_size

        # Get batch size threshold for BEG algorithm
        self.mab_bs_threshold = self.server_args.speculative_mab_bs_threshold

        self.mab_manager = MABGroupManager(
            strategies=self.mab_strategies,
            algorithm=self.mab_algorithm,
            window_size=self.mab_window_size,
            bs_threshold=self.mab_bs_threshold,
        )

        # Initialize basic data structure to store MAB pull info
        self.mab_last_pull = {
            "mab_strategy": None,
            "batch_size": None,
        }

        # Group strategies by draft_tokens for target model CUDA graph sharing
        draft_tokens_groups = {}
        for strategy in self.mab_strategies:
            _, _, draft_tokens = MABConfig.parse_config(strategy)
            if draft_tokens not in draft_tokens_groups:
                draft_tokens_groups[draft_tokens] = []
            draft_tokens_groups[draft_tokens].append(strategy)

        # Shared target model resources by draft_tokens
        target_resources_by_draft_tokens = {}
        for draft_tokens, strategies_group in draft_tokens_groups.items():
            # Temporarily set parameters for target model initialization
            original_draft_tokens = self.server_args.speculative_num_draft_tokens
            self.server_args.speculative_num_draft_tokens = draft_tokens
            self.target_worker.model_runner.server_args.speculative_num_draft_tokens = (
                draft_tokens
            )

            # Get batch size range for this group (use the union of all strategies)
            min_bs_list, max_bs_list = [], []
            for strategy in strategies_group:
                min_bs, max_bs = self.mab_manager.get_strategy_bs_range(strategy)
                min_bs_list.append(min_bs)
                max_bs_list.append(max_bs)
            group_min_bs = min(min_bs_list)
            group_max_bs = max(max_bs_list)

            # Initialize target model resources
            logger.info(
                f"Initializing target model cuda graph for draft_tokens {draft_tokens}"
            )
            self.target_worker.model_runner.init_attention_backend()
            self.target_worker.model_runner.init_device_graphs(
                strategy_min_bs=group_min_bs, strategy_max_bs=group_max_bs
            )

            # Store target resources for this draft_tokens group
            target_resources_by_draft_tokens[draft_tokens] = {
                "attn_backend": self.target_worker.model_runner.attn_backend,
                "cuda_graph_runner": self.target_worker.model_runner.graph_runner,
            }

            # Restore original draft_tokens
            self.server_args.speculative_num_draft_tokens = original_draft_tokens
            self.target_worker.model_runner.server_args.speculative_num_draft_tokens = (
                original_draft_tokens
            )

        # Set up resources for all strategies (draft model resources are strategy-specific)
        for mab_strategy in self.mab_strategies:
            logger.info(f"Initializing draft model resources for strategy {mab_strategy}")
            _, _, draft_tokens = MABConfig.parse_config(mab_strategy)

            # Temporarily update parameters for draft model initialization
            self.update_speculative_args(mab_strategy)
            self.strategy_min_bs, self.strategy_max_bs = (
                self.mab_manager.get_strategy_bs_range(mab_strategy)
            )

            logger.info(f"Initializing draft model cuda graph with strategy {mab_strategy} ")
            # Initialize draft worker resources (these are strategy-specific)
            with self.draft_tp_context(
                self.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.init_attention_backend()
                self.init_cuda_graphs()

            # Get shared target resources for this strategy's draft_tokens
            target_resources = target_resources_by_draft_tokens[draft_tokens]

            # Store resources for the strategy (draft-specific + shared target)
            self.strategy_resources[mab_strategy] = SpeculativeResources(
                draft_attn_backend=self.draft_attn_backend,
                draft_cuda_graph_runner=self.cuda_graph_runner,
                target_attn_backend=target_resources["attn_backend"],
                target_cuda_graph_runner=target_resources["cuda_graph_runner"],
            )

        self.set_mab_strategy(self.default_mab_strategy)

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
            input_ids, hidden_states, scores, tree_info, _ = select_top_k_tokens(
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
            logits_output = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
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
        
        # **FIX**: Prepare the batch for extend - this is critical!
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST

        # Run forward
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
        logits_output = self.draft_runner.forward(forward_batch).logits_output

        # Update spec_info for the next draft step
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        next_draft_input.topk_p, next_draft_input.topk_index = fast_topk(
            probs, self.topk, dim=-1
        )
        next_draft_input.hidden_states = logits_output.hidden_states
        return next_draft_input

    # def _draft_extend_for_prefill(
    #     self,
    #     batch: ModelWorkerBatch,
    #     target_hidden_states: torch.Tensor,
    #     next_token_ids: torch.Tensor,
    # ):
    #     """
    #     Run draft model extend to correctly fill the KV cache.

    #     Args:
    #         batch: The batch to run.
    #         target_hidden_states: Hidden states from the target model forward
    #         next_token_ids: Next token ids generated from the target forward.
    #     """
    #     # Construct input_ids
    #     if not batch.forward_mode.is_idle():
    #         pt = 0
    #         for i, extend_len in enumerate(batch.extend_seq_lens):
    #             input_ids = batch.input_ids[pt : pt + extend_len]
    #             batch.input_ids[pt : pt + extend_len] = torch.cat(
    #                 (input_ids[1:], next_token_ids[i].reshape(1))
    #             )
    #             pt += extend_len

    #     # Construct spec_info
    #     next_draft_input = EagleDraftInput(
    #         hidden_states=target_hidden_states,
    #         verified_id=next_token_ids,
    #         new_seq_lens=batch.seq_lens,
    #         # draft mode is same with decode mode, only 1 num token per batch
    #         num_tokens_per_batch=1,
    #         num_tokens_for_logprob_per_batch=1,
    #     )

    #     batch.spec_info = next_draft_input

    #     # Run forward
    #     forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
    #     logits_output = self.draft_runner.forward(forward_batch).logits_output

    #     # Update spec_info for the next draft step
    #     probs = torch.softmax(logits_output.next_token_logits, dim=-1)
    #     next_draft_input.topk_p, next_draft_input.topk_index = fast_topk(
    #         probs, self.topk, dim=-1
    #     )
    #     next_draft_input.hidden_states = logits_output.hidden_states
    #     return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ModelWorkerBatch, batch_result: GenerationBatchResult
    ):
        # Batch 2: Draft extend
        draft_input = EagleDraftInput(
            hidden_states=batch_result.logits_output.hidden_states,
            num_tokens_per_batch=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_batch=self.speculative_num_steps + 1,
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

        if forward_batch.spec_info.accept_length is None:
            forward_batch.spec_info.accept_length = batch_result.accept_lens

        # Run draft extend batch in the main compute stream
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            draft_logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
        else:
            draft_logits_output = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output

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
    def use_mab(self):
        """Forward use_mab from draft_worker."""
        return self._draft_worker.use_mab

    @property
    def mab_last_pull(self):
        """Forward mab_last_pull from draft_worker."""
        return self._draft_worker.mab_last_pull

    def select_mab_strategy(self, batch_size: int) -> str:
        """Forward MAB strategy selection to draft_worker."""
        return self._draft_worker.select_mab_strategy(batch_size)

    def record_mab_strategy_metrics(self, events, accept_length_per_req_cpu):
        """Forward MAB metrics recording to draft_worker."""
        return self._draft_worker.record_mab_strategy_metrics(events, accept_length_per_req_cpu)

    def should_enable_sd(self, batch):
        """Forward adaptive SD check to draft_worker."""
        return self._draft_worker.should_enable_sd(batch)

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
        # Initialize events dictionary only if MAB is enabled
        events = {}
        if self.use_mab:
            events = {
                "processing_start": torch.cuda.Event(enable_timing=True),
                "processing_end": torch.cuda.Event(enable_timing=True),
            }
            # Record the start of processing
            events["processing_start"].record()

        enable_sd = self.should_enable_sd(model_worker_batch)

        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Target prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Draft prefill - only if SD is enabled
            if enable_sd:
                model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
                with self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                    batch_output.next_draft_input = (
                        self.draft_worker._draft_extend_for_prefill(
                            model_worker_batch,
                            batch_output.logits_output.hidden_states,
                            batch_output.next_token_ids,
                        )
                    )
            return batch_output
        else:
            if not enable_sd:
                # Run normal decode without speculation
                model_worker_batch.spec_info = None
                
                # Swap resources if we have them
                use_normal_graph = (
                    self.draft_worker.adaptive_spec_threshold is not None
                    and hasattr(self.draft_worker, "target_normal_decode_graph_runner")
                    and self.draft_worker.target_normal_decode_graph_runner is not None
                )
                
                if use_normal_graph:
                    target_runner = self.target_worker.model_runner
                    old_attn = target_runner.attn_backend
                    old_graph = target_runner.graph_runner
                    old_spec_algo = target_runner.spec_algorithm
                    
                    target_runner.attn_backend = self.draft_worker.target_normal_decode_attn_backend
                    target_runner.graph_runner = self.draft_worker.target_normal_decode_graph_runner
                    target_runner.spec_algorithm = SpeculativeAlgorithm.NONE
                
                try:
                    batch_output = self.target_worker.forward_batch_generation(
                        model_worker_batch
                    )
                    batch_output.num_accepted_tokens = 0  # No speculation
                    
                    # Create draft input from target logits for overlap scheduler
                    # When SD is disabled, we still need to provide draft input for the next iteration
                    bs = len(model_worker_batch.seq_lens)
                    
                    # Get logits - in decode mode, we should have one logit per sequence
                    # But if we have more (e.g., from packed sequences), select the last one per sequence
                    logits = batch_output.logits_output.next_token_logits
                    if logits.shape[0] != bs:
                        # Select the last logit for each sequence based on seq_lens
                        indices = torch.cumsum(model_worker_batch.seq_lens, dim=0) - 1
                        logits = logits[indices]
                    
                    probs = torch.softmax(logits, dim=-1)
                    topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
                    
                    batch_output.next_draft_input = EagleDraftInput(
                        verified_id=batch_output.next_token_ids,
                        new_seq_lens=model_worker_batch.seq_lens + 1,
                        topk_p=topk_p,
                        topk_index=topk_index,
                        hidden_states=batch_output.logits_output.hidden_states,
                        num_tokens_per_batch=1,
                        num_tokens_for_logprob_per_batch=1,
                    )
                finally:
                    if use_normal_graph:
                        target_runner.attn_backend = old_attn
                        target_runner.graph_runner = old_graph
                        target_runner.spec_algorithm = old_spec_algo
                
                return batch_output

            # MAB strategy selection if enabled
            if self.use_mab:
                batch_size = len(model_worker_batch.seq_lens)
                strategy = self.select_mab_strategy(batch_size)
                self.mab_last_pull["batch_size"] = batch_size
                self.mab_last_pull["mab_strategy"] = strategy

            if model_worker_batch.spec_info is None:
                model_worker_batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=self.target_worker.model_config.hidden_size,
                    dtype=self.target_worker.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                verify_input: EagleVerifyInput = self.draft_worker.draft(
                    model_worker_batch
                )
            assert verify_input.is_verify_input()
            model_worker_batch.spec_info = verify_input
            batch_output = self.verify(model_worker_batch)
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.draft_worker._draft_extend_for_decode(
                    model_worker_batch, batch_output
                )

            # Record MAB metrics if enabled
            if self.use_mab:
                events["processing_end"].record()
                self.record_mab_strategy_metrics(
                    events, batch_output.accept_length_per_req_cpu
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
                self.speculative_num_draft_tokens,
            )
        else:
            verified_id = torch.empty((0,), device=self.device, dtype=torch.int32)

        # Construct the next draft input
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
        )

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
    ):
        """
        Move accepted tokens to the target KV cache.

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
