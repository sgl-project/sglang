import logging
import time
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_for_decode,
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_mab import (
    MABConfig,
    MABGroupManager,
    SpeculativeResources,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    detect_nan,
    draft_tp_context,
    fast_topk,
    generate_token_bitmask,
    get_last_loc_large_page_size_large_top_k,
    load_token_map,
    select_top_k_tokens,
)
from sglang.srt.utils import (
    MultiprocessingSerializer,
    empty_context,
    get_available_gpu_memory,
    is_cuda,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)


class EAGLEWorker(TpModelWorker):

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
        self.max_topk = self.topk = server_args.speculative_eagle_topk
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
        self.padded_static_len = -1
        self.strategy_min_bs = None
        self.strategy_max_bs = None

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
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
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
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            self.eagle_use_aux_hidden_state = True
            eagle_config = getattr(
                self.draft_model_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

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
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.init_attention_backend()
                self.init_cuda_graphs()

        if self.adaptive_spec_threshold is not None:
            self.init_target_normal_decode_graph()

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners
        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

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

        # Capture extend
        if self.draft_extend_attn_backend and not _is_npu:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(
                self
            )
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    @property
    def draft_model_runner(self):
        return self.model_runner

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

    def should_enable_sd(self, batch: ScheduleBatch) -> bool:
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
        batch_size = batch.batch_size()

        # For EXTEND batches, always run (to establish draft model state)
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return True

        # For DECODE batches, check threshold with warmup
        current_enable_sd = False
        if batch_size <= self.adaptive_spec_threshold:
            self.adaptive_spec_consecutive_checks += 1

            if self.adaptive_spec_consecutive_checks >= self.adaptive_spec_warmup_checks:
                if not self.adaptive_spec_enabled:
                    print(f"ENABLING SD AFTER {self.adaptive_spec_consecutive_checks} CHECKS ★★★")
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
        self.padded_static_len = self.speculative_num_steps + 1
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
        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend
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

        print(f"[MAB] Selected strategy: {selected_strategy} for batch_size={batch_size}")

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
                self.draft_model_runner.tp_group
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
        # Initialize events dictionary only if MAB is enabled
        events = {}
        if self.use_mab:
            events = {
                "processing_start": torch.cuda.Event(enable_timing=True),
                "processing_end": torch.cuda.Event(enable_timing=True),
            }
            # Record the start of processing
            events["processing_start"].record()

        enable_sd = self.should_enable_sd(batch)
        
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, seq_lens_cpu = self.forward_target_extend(
                batch
            )
            
            if enable_sd:
                with self.draft_tp_context(self.draft_model_runner.tp_group):
                    self.forward_draft_extend(batch, logits_output.hidden_states, next_token_ids, seq_lens_cpu)
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )
        else:
            if not enable_sd:
                # Temporarily disable spec to make target worker treat this as normal decode
                batch.spec_algorithm = SpeculativeAlgorithm.NONE
                batch.spec_info = None
                
                # Prepare batch for normal decode since it was prepared with skip_prepare=True
                # We need to set input_ids to the last generated token per request
                bs = len(batch.reqs)
                
                if batch.output_ids is not None and batch.output_ids.numel() == bs:
                    # Use output_ids if available and has correct shape
                    batch.input_ids = batch.output_ids
                    batch.output_ids = None
                else:
                    # Fallback: gather the last output token from each request
                    batch.input_ids = torch.tensor(
                        [req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1] 
                         for req in batch.reqs],
                        dtype=torch.int64,
                        device=batch.device,
                    )
                    batch.output_ids = None
                
                # Slice out_cache_loc for normal decode
                # In spec mode, out_cache_loc has shape [bs * spec_steps * topk]
                # In normal mode, we only need [bs]
                if batch.out_cache_loc is None:
                    batch.out_cache_loc = alloc_for_decode(batch, 1)
                    batch.seq_lens.add_(1)
                    batch.seq_lens_cpu.add_(1)

                original_out_cache_loc = batch.out_cache_loc
                if batch.out_cache_loc is not None and batch.out_cache_loc.numel() >= bs:
                    batch.out_cache_loc = batch.out_cache_loc[:bs]
                
                model_worker_batch = batch.get_model_worker_batch()
                
                # Restore original after creating model_worker_batch
                batch.out_cache_loc = original_out_cache_loc
                
                # Swap resources if we have them
                use_normal_graph = (
                    self.adaptive_spec_threshold is not None
                    and hasattr(self, "target_normal_decode_graph_runner")
                    and self.target_normal_decode_graph_runner is not None
                )
                
                if use_normal_graph:
                    old_attn = self.target_worker.model_runner.attn_backend
                    old_graph = self.target_worker.model_runner.graph_runner
                    self.target_worker.model_runner.attn_backend = (
                        self.target_normal_decode_attn_backend
                    )
                    self.target_worker.model_runner.graph_runner = (
                        self.target_normal_decode_graph_runner
                    )
                
                try:
                    batch_result = self.target_worker.forward_batch_generation(
                        model_worker_batch
                    )
                finally:
                    if use_normal_graph:
                        self.target_worker.model_runner.attn_backend = old_attn
                        self.target_worker.model_runner.graph_runner = old_graph
                
                # Increment spec_verify_ct for each request in normal decode mode
                # to ensure correct acceptance length calculation (completion_tokens / spec_verify_ct)
                # When spec is disabled, we treat each decode step as "1 token verified"
                for req in batch.reqs:
                    req.spec_verify_ct += 1
                
                return GenerationBatchResult(
                    logits_output=batch_result.logits_output,
                    next_token_ids=batch_result.next_token_ids,
                    num_accepted_tokens=0,  # No speculation
                    can_run_cuda_graph=batch_result.can_run_cuda_graph,
                )
            
            if self.use_mab:
                batch_size = batch.batch_size()
                strategy = self.select_mab_strategy(batch_size)
                self.mab_last_pull["batch_size"] = batch_size
                self.mab_last_pull["mab_strategy"] = strategy

            # Slice topk resources for spec_info
            batch.spec_info.topk_p = batch.spec_info.topk_p[:, : self.topk]
            batch.spec_info.topk_index = batch.spec_info.topk_index[:, : self.topk]
            
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                spec_info = self.draft(batch)
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )

            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                # NOTE: We should use `check_forward_draft_extend_after_decode`
                # when DP attention is enabled, but it is slow. Skip it for now.
                if (
                    self.server_args.enable_dp_attention
                    or batch.spec_info.verified_id.shape[0] > 0
                ):
                    # decode is not finished
                    self.forward_draft_extend_after_decode(batch)

            if self.use_mab:
                events["processing_end"].record()
                self.record_mab_strategy_metrics(events, verify_output.accept_length_per_req_cpu)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
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
        batch.maybe_evict_swa()
        for req in batch.reqs:
            req.decode_batch_idx += 1

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
        # Layout of the out_cache_loc
        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]
        if self.page_size == 1:
            alloc_len_per_decode = self.speculative_num_steps * self.topk
            # TODO: We only need self.speculative_num_steps - 1 * topk cache loc
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * alloc_len_per_decode,
                backup_state=True,
            )
        else:
            if self.topk == 1:
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    tok-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                #  "x" means speculative draft tokens
                #  "." means padded tokens

                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                    last_page_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens_cpu = prefix_lens_cpu % self.page_size
                num_new_pages_per_topk = (
                    last_page_lens_cpu + self.speculative_num_steps + self.page_size - 1
                ) // self.page_size
                seq_lens_cpu = (
                    prefix_lens_cpu // self.page_size * self.page_size
                    + num_new_pages_per_topk * (self.page_size * self.topk)
                )
                extend_num_tokens = torch.sum((seq_lens_cpu - prefix_lens_cpu)).item()

            out_cache_loc, token_to_kv_pool_state_backup = (
                alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        if self.page_size > 1 and self.topk > 1:
            last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
            duplicate_cache_len = torch.sum(last_page_lens_cpu).item() * (self.topk - 1)
            target_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
            source_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
        else:
            # When source_cache_loc is not needed, simply skip
            duplicate_cache_len = 0
            source_cache_loc, target_cache_loc, last_page_lens_cumsum = None, None, None

        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps + self.page_size),
        )

        if self.page_size > 1 and self.topk > 1:
            if duplicate_cache_len > 0:
                self.draft_model_runner.token_to_kv_pool.move_kv_cache(
                    target_cache_loc, source_cache_loc
                )
            # Remove padded slots
            # TODO: We only need self.speculative_num_steps - 1 cache loc
            out_cache_loc = out_cache_loc[
                : num_seqs * self.topk * self.speculative_num_steps
            ]

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=self.model_config.hidden_size,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

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
            model_worker_batch, self.draft_model_runner
        )
        can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(
            forward_batch
        )
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch
            )
        else:
            forward_batch.can_run_dp_cuda_graph = False
            if (
                not forward_batch.forward_mode.is_idle()
                and self.speculative_num_steps > 1
            ):
                # Skip attention backend init for idle mode or 1-step draft
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            # Run forward steps
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
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

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        # TODO: We only need self.speculative_num_steps - 1 cache loc
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
            input_ids, hidden_states, scores, tree_info, real_parents = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Overwrite the kv cache to keep the causal relationship of branch.
            if i > 0:
                src_cache_loc = out_cache_loc[i-1][real_parents]
                tgt_cache_loc = out_cache_loc[i-1]
                self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    tgt_cache_loc, src_cache_loc
                )


            # Set inputs
            forward_batch.input_ids = input_ids
            # This is a temporary fix for the case that the user is using standalone
            # speculative decoding and the draft model architecture is gpt-oss. gpt-oss
            # rope kernel needs cache_loc to be contiguous.
            if (
                self.server_args.speculative_algorithm == "STANDALONE"
                and self.model_config.hf_config.architectures[0] == "GptOssForCausalLM"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            if self.server_args.enable_nan_detection:
                detect_nan(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker
        pass

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        seq_lens_pre_verify = batch.seq_lens.clone()
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_batch = self.speculative_num_steps + 1
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

        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
        ):
            self._mamba_verify_update(
                batch, res, logits_output, spec_info, seq_lens_pre_verify
            )

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # Prepare the batch for the next draft forwards.
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
        spec_info: EagleVerifyInput,
        seq_lens_pre_verify: torch.Tensor,
    ):
        accepted_length = (
            torch.tensor(
                res.accept_length_per_req_cpu,
                device=logits_output.hidden_states.device,
                dtype=torch.int64,
            )
            + 1
        )
        cumulative_accepted_lengths = torch.cumsum(accepted_length, dim=0)
        # prepend 0 to the cumulative_accepted_lengths
        accepted_indices_start = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=cumulative_accepted_lengths.dtype,
                    device=cumulative_accepted_lengths.device,
                ),
                cumulative_accepted_lengths[:-1],
            ]
        )
        accepted_indices_offset = torch.arange(
            0,
            len(batch.seq_lens) * batch.spec_info.draft_token_num,
            step=batch.spec_info.draft_token_num,
            dtype=accepted_indices_start.dtype,
            device=accepted_indices_start.device,
        )

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        # res.accepted_indices.shape[0] > 0 skips DP attn idle batch
        if spec_info.topk > 1 and res.accepted_indices.shape[0] > 0:
            # accepted_indices=[0,2,3,4,5,7,9,10,11], accepted_length=[4, 3, 2], cumulative_accepted_lengths=[4, 7, 9]
            # first_token_indices_per_req=prepend(0, accepted_indices[cumulative_accepted_lengths[:-1]]) = [0, 5, 10]
            # last_token_indices_per_req=accepted_indices[cumulative_accepted_lengths - 1] = [4, 9, 11] (last token ID of each req)
            # max_relative_indices_per_req = [4,4,1]; those are the per-req spec-decoding step offsets that contain the correct mamba caches
            # first_token_indices_per_req = res.accepted_indices[accepted_indices_start]
            accepted_steps = (
                res.accepted_indices[cumulative_accepted_lengths - 1]
                - accepted_indices_offset
            )
        else:
            accepted_steps = accepted_length - 1

        if batch.mamba_track_indices is not None:
            # If after verify, the request's seq_lens has crossed a mamba track interval,
            # we need to update the mamba state for the request at the crossing point.
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            mamba_steps_to_track = torch.where(
                to_track_mask,
                res.accepted_indices[to_track_ith + accepted_indices_start]
                - accepted_indices_offset,
                -1,
            )
        else:
            mamba_steps_to_track = None

        self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

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
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.return_logprob = False
        logits_output = self.draft_model_runner.forward(forward_batch).logits_output
        if self.enable_nan_detection:
            detect_nan(logits_output)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        self.capture_for_decode(logits_output, forward_batch.spec_info)

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
                and self.eagle_use_aux_hidden_state
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
            model_worker_batch, self.draft_model_runner
        )
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
        else:
            forward_batch.seq_lens_sum = batch.seq_lens.sum().item()

        # Run
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
            forward_batch.spec_info.topk_p, forward_batch.spec_info.topk_index = (
                logits_output.topk_p,
                logits_output.topk_index,
            )
            forward_batch.spec_info.hidden_states = logits_output.hidden_states
        else:
            forward_batch.can_run_dp_cuda_graph = False
            if not forward_batch.forward_mode.is_idle():
                self.draft_model_runner.attn_backend.init_forward_metadata(
                    forward_batch
                )
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            self.capture_for_decode(logits_output, forward_batch.spec_info)

        if self.enable_nan_detection:
            detect_nan(logits_output)

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

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(
            probs, self.max_topk, dim=-1
        )
        draft_input.hidden_states = logits_output.hidden_states

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.model_runner.update_weights_from_tensor(
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


@torch.compile(dynamic=True, disable=_is_npu)
def get_last_loc_large_page_size_top_k_1(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens,
    speculative_num_steps: int,
):
    prefix_lens = seq_lens
    seq_lens = prefix_lens + speculative_num_steps
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )
    return prefix_lens, seq_lens, last_loc
