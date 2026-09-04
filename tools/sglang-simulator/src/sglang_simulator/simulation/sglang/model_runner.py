import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.sglang.utils import (
    resolve_model_info,
    resolve_scheduler_config,
)
from sglang_simulator.simulation.utils import profile_device_available_bytes


class _MockModel(torch.nn.Module):
    """Minimal model surface needed by SGLang's native runner initialization."""

    def forward(self, *args, **kwargs):
        return None


class _MockModelLoader:
    """Minimal loader state for upstream resident-weight accounting."""

    preloaded_weights_bytes = 0


def _make_mock_model_loader(model_runner_type):
    if hasattr(model_runner_type, "preloaded_weights_bytes"):
        return _MockModelLoader()
    return None


def _resolve_kv_page_size(configurator):
    return (
        getattr(configurator, "page_size", None)
        or getattr(configurator.server_args, "page_size", None)
        or 1
    )


class C_ModelRunnerHook(BaseHook):
    HOOK_CLASS_NAME = "ModelRunner"
    HOOK_MODULE_NAME = "sglang.srt.model_executor.model_runner"

    @classmethod
    def hook(cls, target):
        def override_load_model(self):
            from sglang.srt.model_executor.model_runner import (
                resolve_sliding_window_size,
            )

            self.model = _MockModel()
            self.dtype = self.model_config.dtype
            self.sliding_window_size = resolve_sliding_window_size(
                self.model, self.model_config
            )
            self.prefill_aware_swa = False
            self.weight_load_mem_usage = 0
            self.load_config = None
            self.loader = _make_mock_model_loader(type(self))

            if ConfigManager.get_model_info() is None:
                ConfigManager.set_model_info(resolve_model_info(self.model_config))

        def wrapped_forward(self, *args, **kwargs):
            batch = args[0]
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput
            from sglang.srt.model_executor.model_runner import ModelRunnerOutput

            output = LogitsProcessorOutput(
                next_token_logits=torch.empty(
                    size=(batch.batch_size, self.model_config.vocab_size),
                    device=self.device,
                )
            )
            return ModelRunnerOutput(
                logits_output=output,
                can_run_graph=False,
                expert_distribution_metrics=None,
            )

        def wrapped_sample(self, *args, **kwargs):
            logits = args[0]
            return torch.ones(
                size=(logits.next_token_logits.shape[0],),
                device=self.device,
                dtype=torch.int64,
            )

        def wrapped_compute_logprobs_only(*args, **kwargs):
            return None

        def wrapped_init_attention_backends(self):
            try:
                from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
                    resolve_attention_backend_strs,
                )
            except ImportError:
                default_backend = self.server_args.attention_backend
                self.prefill_attention_backend_str = (
                    self.server_args.prefill_attention_backend or default_backend
                )
                self.decode_attention_backend_str = (
                    self.server_args.decode_attention_backend or default_backend
                )
            else:
                resolved = resolve_attention_backend_strs(model_runner=self)
                self.prefill_attention_backend_str = resolved.prefill
                self.decode_attention_backend_str = resolved.decode

            self.attn_backend = None
            self.decode_attn_backend = None
            self.decode_attn_backend_group = None

        def wrapped_init_cuda_graphs(self, capture_decode_cuda_graph=True):
            self.graph_mem_usage = 0
            self.cuda_graph_runner = None
            self.eager_runner = None
            self.prefill_cuda_graph_runner = None
            self.decode_cuda_graph_runner = None

        # Keep SGLang's native initialize() and alloc_memory_pool() lifecycle.
        # Only operations that require model weights or GPU kernels are mocked.
        target.load_model = override_load_model
        target.forward = wrapped_forward
        target.sample = wrapped_sample
        target.compute_logprobs_only = wrapped_compute_logprobs_only
        target.init_attention_backends = wrapped_init_attention_backends
        target.init_cuda_graphs = wrapped_init_cuda_graphs


class C_KVCacheConfiguratorHook(BaseHook):
    HOOK_CLASS_NAME = "KVCacheConfigurator"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.kv_cache_configurator"

    @classmethod
    def hook(cls, target):
        original_configure = target.configure
        original_init_pools = target._init_pools
        supports_cpu_fp8_quant_method = hasattr(target, "_build_mha_quant_method")

        def wrapped_configure(self, *args, **kwargs):
            if not (
                supports_cpu_fp8_quant_method
                and getattr(self, "device", None) == "cpu"
                and getattr(self, "kv_cache_dtype", None) == torch.float8_e4m3fn
            ):
                return original_configure(self, *args, **kwargs)

            # Newer SGLang runtimes validate CPU FP8 KV-cache support and select
            # an AMX-only quant method. The simulator executes scheduler state on
            # CPU while modeling the target accelerator's FP8 cache. Suppress the
            # physical-CPU predicate only while native compact pools are built;
            # all other platform capabilities and the logical dtype stay intact.
            from sglang.srt.mem_cache.kv_cache_configurator import current_platform

            original_is_cpu = current_platform.is_cpu
            current_platform.is_cpu = lambda: False
            try:
                return original_configure(self, *args, **kwargs)
            finally:
                current_platform.is_cpu = original_is_cpu

        def override_profile_available_bytes(self, pre_model_load_memory):
            if self.server_args.max_total_tokens is not None:
                from sglang.srt.model_executor.pool_configurator import (
                    create_memory_pool_configurator,
                )

                configurator = create_memory_pool_configurator(self)
                target_tokens = self.server_args.max_total_tokens

                page_size = _resolve_kv_page_size(self)

                def resolved_tokens(budget_bytes):
                    try:
                        config = configurator.calculate_pool_sizes(
                            budget_bytes, page_size
                        )
                    except RuntimeError:
                        return 0
                    return config.max_total_num_tokens

                lower, upper = 0, 1
                while resolved_tokens(upper) < target_tokens:
                    lower, upper = upper, upper * 2
                while lower + 1 < upper:
                    middle = (lower + upper) // 2
                    if resolved_tokens(middle) < target_tokens:
                        lower = middle
                    else:
                        upper = middle
                return upper

            model = ConfigManager.get_model_info()
            if model is None:
                model = resolve_model_info(self.model_config)
                ConfigManager.set_model_info(model)
            hardware = ConfigManager.get_accelerator_info()
            scheduler_config = resolve_scheduler_config(
                server_args=self.server_args,
                model_config=self.model_config,
            )
            if hardware is None or scheduler_config is None:
                raise RuntimeError(
                    "Simulator model, accelerator, and scheduler configuration "
                    "must be resolved before KV-cache pool sizing."
                )

            available_bytes = profile_device_available_bytes(
                model=model,
                device=hardware,
                scheduler_config=scheduler_config,
            )
            if self.mambaish_config is not None:
                rest_memory_gb = self._handle_max_mamba_cache(
                    available_bytes / (1 << 30)
                )
                available_bytes = int(rest_memory_gb * (1 << 30))
            return available_bytes

        def wrapped_init_pools(self, *args, **kwargs):
            # Pool payload is never read during simulation. Preserve the native
            # pool classes and allocator wiring, but allocate minimal payload
            # dimensions and restore their logical metadata afterwards.
            compact_attrs = (
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "index_head_dim",
                "kv_lora_rank",
                "head_dim",
                "v_head_dim",
                "linear_value_head_dim",
                "linear_key_head_dim",
                "linear_conv_kernel_dim",
            )
            original_attrs = {
                name: getattr(self.model_config, name)
                for name in compact_attrs
                if hasattr(self.model_config, name)
            }
            try:
                for name in original_attrs:
                    setattr(self.model_config, name, 1)
                pools = original_init_pools(self, *args, **kwargs)
            finally:
                for name, value in original_attrs.items():
                    setattr(self.model_config, name, value)

            token_pool = pools.token_to_kv_pool
            for name, value in original_attrs.items():
                if hasattr(token_pool, name):
                    setattr(token_pool, name, value)

            if (
                hasattr(token_pool, "kv_cache_dim")
                and token_pool.kv_cache_dim == 2
                and "kv_lora_rank" in original_attrs
                and "qk_rope_head_dim" in original_attrs
            ):
                token_pool.kv_cache_dim = (
                    original_attrs["kv_lora_rank"] + original_attrs["qk_rope_head_dim"]
                )
            return pools

        target.configure = wrapped_configure
        target._profile_available_bytes = override_profile_available_bytes
        target._init_pools = wrapped_init_pools
