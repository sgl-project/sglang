import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.sglang.utils import (
    resolve_model_info,
    resolve_scheduler_config,
)
from sglang_simulator.simulation.utils import estimate_kv_cache_pool_capacity
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_ModelRunnerHook(BaseHook):
    HOOK_CLASS_NAME = "ModelRunner"
    HOOK_MODULE_NAME = "sglang.srt.model_executor.model_runner"

    @classmethod
    def hook(cls, target):

        def override_initialize(self, *args, **kwargs):
            class MockModel:
                def forward(self):
                    pass

            self.model = MockModel()

            self.dtype = self.model_config.dtype
            self.configure_kv_cache_dtype()

            # v0.5.16's KVCacheConfigurator consumes these fields after
            # initialize(). The real load_model() normally populates the first
            # two, but simulation deliberately skips weight loading.
            from sglang.srt.model_executor.model_runner import (
                resolve_sliding_window_size,
            )
            from sglang.srt.model_executor.model_runner_components.layer_setup import (
                resolve_layer_indices,
            )

            self.sliding_window_size = resolve_sliding_window_size(
                self.model, self.model_config
            )
            self.prefill_aware_swa = False

            if ConfigManager.get_model_info() is None:
                model = resolve_model_info(self.model_config)
                ConfigManager.set_model_info(model)

            if self.server_args.max_total_tokens is not None:
                self.max_total_num_tokens = self.server_args.max_total_tokens
            else:
                model = ConfigManager.get_model_info()
                hw = ConfigManager.get_accelerator_info()
                config = resolve_scheduler_config(
                    server_args=self.server_args,
                )

                assert model is not None and hw is not None and config is not None
                self.max_total_num_tokens = estimate_kv_cache_pool_capacity(
                    model, hw, config
                )

            if hasattr(self, "page_size") and self.page_size > 1:
                self.max_total_num_tokens = (
                    self.max_total_num_tokens // self.page_size * self.page_size
                )

            self.layer_info = resolve_layer_indices(
                model=self.model,
                model_config=self.model_config,
                is_draft_worker=self.is_draft_worker,
                spec_algorithm=self.spec_algorithm,
            )
            self.start_layer = self.layer_info.start_layer
            self.end_layer = self.layer_info.end_layer
            self.num_effective_layers = self.layer_info.num_effective_layers

            if self.is_hybrid_swa:
                self.sliding_window_size = self.model_config.sliding_window_size
                # if self.model_config.is_swa_with_compressed_attention:
                #     self.set_num_tokens_hybrid_swa_compress()
                # else:
                #     self.set_num_tokens_hybrid_swa()

                # The method names for setting parameters such as num_token vary significantly
                # across different versions. Here, we directly assign values to these parameters.
                # In the future, we may attempt to decouple or adapt to different versions
                # to ensure consistent behavior.
                from types import SimpleNamespace

                from sglang.srt.model_executor.pool_configurator import (
                    DSV4PoolConfigurator,
                )

                configurator = DSV4PoolConfigurator(
                    SimpleNamespace(
                        model_config=self.model_config,
                        layer_info=self.layer_info,
                        ps=self.ps,
                        pp_group=self.pp_group,
                        server_args=self.server_args,
                        spec_algorithm=self.spec_algorithm,
                    )
                )
                self.memory_pool_config = (
                    configurator.calculate_pool_sizes_from_max_tokens(
                        self.max_total_num_tokens,
                        self.page_size,
                    )
                )
                requested_max_reqs = self.server_args.max_running_requests
                self.memory_pool_config.max_running_requests = (
                    requested_max_reqs // self.ps.attn_dp_size
                    if requested_max_reqs is not None
                    else min(4096, max(2048, self.max_total_num_tokens // 2))
                )
                self.memory_pool_config = (
                    configurator.finalize_with_max_running_requests(
                        self.memory_pool_config
                    )
                )
                self.max_total_num_tokens = self.memory_pool_config.max_total_num_tokens
                self.max_running_requests = self.memory_pool_config.max_running_requests
                self.full_max_total_num_tokens = (
                    self.memory_pool_config.full_max_total_num_tokens
                )
                self.swa_max_total_num_tokens = (
                    self.memory_pool_config.swa_max_total_num_tokens
                )

                self.c4_max_total_num_tokens = (
                    self.memory_pool_config.c4_max_total_num_tokens
                )
                self.c128_max_total_num_tokens = (
                    self.memory_pool_config.c128_max_total_num_tokens
                )
                self.c4_state_pool_size = self.memory_pool_config.c4_state_pool_size
                self.c128_state_pool_size = self.memory_pool_config.c128_state_pool_size
                self.state_dtype = torch.float32

            max_num_reqs = min(
                max(
                    int(
                        self.max_total_num_tokens / self.model_config.context_len * 512
                    ),
                    2048,
                ),
                4096,
            )
            logger.info(
                f"Model runner initialized with {self.max_total_num_tokens} tokens. Maximum number of requests: {max_num_reqs}"
            )

            from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

            self.req_to_token_pool = ReqToTokenPool(
                size=max_num_reqs,
                max_context_len=self.model_config.context_len,
                device=self.device,
                enable_memory_saver=False,
            )

            # During simulation, the actual data in kv cache pool is not important since the MHA computation is skipped,
            # so the head_num and head_dim can be set to 1 to reduce the memory usage.
            # And the scheduler only matters about whether the token_to_kv_pool can be allocated enough space for the requests,
            # so the pool's implementation is not important and can be replaced with `MHATokenToKVPool` that only simulates the allocation logic.
            if self.is_hybrid_swa:
                assert (
                    self.page_size == 256
                ), "In paged swa mode, page_size must be 256."
                from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
                    DeepSeekV4TokenToKVPool,
                )

                print(
                    f"[override_initialize] {self.model_config.qk_nope_head_dim=}, {self.model_config.qk_rope_head_dim=}, {self.model_config.index_head_dim=}, {self.device=}, {self.swa_max_total_num_tokens=}"
                )
                if self.is_draft_worker:
                    from sglang.srt.models.deepseek_v4_nextn import (
                        COMPRESS_RATIO_NEXTN_LAYER,
                    )

                    compression_ratios = [
                        COMPRESS_RATIO_NEXTN_LAYER
                    ] * self.num_effective_layers
                else:
                    compression_ratios = self.model_config.compress_ratios
                from sglang.srt.mem_cache.kv_cache_configurator import (
                    _get_dsv4_compress_state_dtypes,
                )

                c4_state_dtype, c128_state_dtype = _get_dsv4_compress_state_dtypes()
                self.token_to_kv_pool = DeepSeekV4TokenToKVPool(
                    max_num_reqs=self.server_args.max_running_requests,
                    num_req_slots=self.req_to_token_pool.req_to_token.shape[0],
                    swa_size=self.swa_max_total_num_tokens,
                    c4_size=self.c4_max_total_num_tokens,
                    c128_size=self.c128_max_total_num_tokens,
                    c4_state_pool_size=self.c4_state_pool_size,
                    c128_state_pool_size=self.c128_state_pool_size,
                    page_size=self.page_size,
                    swa_page_size=self.page_size,
                    sliding_window=self.model_config.window_size,
                    dtype=self.kv_cache_dtype,
                    c4_state_dtype=c4_state_dtype,
                    c128_state_dtype=c128_state_dtype,
                    qk_nope_head_dim=1,  # Overwrite dim size
                    qk_rope_head_dim=1,
                    indexer_head_dim=1,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    compression_ratios=compression_ratios,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    enable_hisparse=self.enable_hisparse,
                    online_mtp_max_draft_tokens=(
                        self.server_args.max_speculative_num_draft_tokens or 0
                    ),
                )
            else:
                from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

                self.token_to_kv_pool = MHATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=1,  # Overwrite head_num and head_dim to 1.
                    head_dim=1,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    enable_alt_stream=False,
                )

            if self.is_hybrid_swa:
                from sglang.srt.mem_cache.allocator.swa import (
                    SWATokenToKVPoolAllocator,
                )

                self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                    self.full_max_total_num_tokens,
                    self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )
            elif self.page_size == 1:
                from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

                self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )
            else:
                from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

                self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )

            self.attn_backend = None
            self.graph_mem_usage = 0
            self.weight_load_mem_usage = 10

            self.max_running_requests = min(
                (
                    self.max_total_num_tokens // 2
                    if self.server_args.max_running_requests is None
                    else self.server_args.max_running_requests
                    // (
                        self.server_args.dp_size
                        if self.server_args.enable_dp_attention
                        else 1
                    )
                ),
                self.req_to_token_pool.size,
            )

            self.use_ngram_embedding = False
            self.canary_manager = None
            self.init_ngram_embedding_manager()

            return

        def wrapped_alloc_memory_pool(self, memory_pool_config=None):
            """Reuse the lightweight pools created by override_initialize."""
            if memory_pool_config is not None:
                self.memory_pool_config = memory_pool_config

        def wrapped_init_attention_backends(self):
            # Forward computation is mocked, so no real attention backend is
            # required. Keep the field expected by scheduler code.
            self.attn_backend = None

        def wrapped_init_cuda_graphs(self, capture_decode_cuda_graph=True):
            # Graph capture is intentionally disabled for simulation.
            self.graph_mem_usage = 0
            self.cuda_graph_runner = None

        def wrapped_forward(self, *args, **kwargs):
            batch = args[0]
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            output = LogitsProcessorOutput(
                next_token_logits=torch.empty(
                    size=(batch.batch_size, self.model_config.vocab_size),
                    device=self.device,
                )
            )
            from sglang.srt.model_executor.model_runner import ModelRunnerOutput

            return ModelRunnerOutput(
                logits_output=output,
                can_run_graph=False,
                expert_distribution_metrics=None,
            )

        def wrapped_sample(self, *args, **kwargs):
            logits = args[0]
            ids = torch.ones(
                size=(logits.next_token_logits.shape[0],),
                device=self.device,
                dtype=torch.int64,
            )
            return ids

        def wrapped_compute_logprobs_only(*args, **kwargs):
            return None

        target.initialize = override_initialize
        target.alloc_memory_pool = wrapped_alloc_memory_pool
        target.init_attention_backends = wrapped_init_attention_backends
        target.init_cuda_graphs = wrapped_init_cuda_graphs
        target.forward = wrapped_forward
        target.sample = wrapped_sample
        target.compute_logprobs_only = wrapped_compute_logprobs_only
