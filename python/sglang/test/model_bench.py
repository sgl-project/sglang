# Copyright 2023-2025 SGLang Team
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
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Callable, Optional, Self

import torch
from transformers import PretrainedConfig

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention.dummy_backend import DummyAttentionBackend
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_runner import torch_compile
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.loader import _get_quantization_config
from sglang.srt.model_loader.utils import get_model_architecture
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import add_prefix, configure_logger

logger = logging.getLogger(__name__)


@dataclass
class ModelBenchArgs:
    num_tokens: int
    forward_mode: ForwardMode
    init_device: str = "cuda"
    exec_device: str = "cuda"
    use_real_weights: bool = False
    warmup_iters: int = 0
    bench_iters: int = 1
    disable_nvtx_tracing: bool = False

    @staticmethod
    def from_args(args):
        model_bench_args = ModelBenchArgs(
            num_tokens=args.num_tokens,
            init_device=args.init_device,
            exec_device=args.exec_device,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
            forward_mode=ForwardMode(args.forward_mode),
            disable_nvtx_tracing=args.disable_nvtx_tracing,
        )
        return model_bench_args


class ModelBench(ABC):
    def __init__(
        self,
        server_args: ServerArgs,
        bench_args: ModelBenchArgs,
        initializer: Callable[
            [Self, PretrainedConfig, Optional[QuantizationConfig]], torch.nn.Module
        ],
    ) -> None:
        self._server_args = server_args
        self._bench_args = bench_args
        self._initializer = initializer
        self._prefix = "model.layers.0"

    def __enter__(self):
        set_global_server_args_for_scheduler(self._server_args)

        configure_logger(self._server_args, " Model Bench")

        logger.info("====================")
        logger.info(f"{self._server_args}")
        logger.info("====================")
        logger.info(f"{self._bench_args=}")
        logger.info("====================")

        if self._bench_args.init_device != self._bench_args.exec_device:
            logger.warning(
                "Init and exec device are different, data will be moved and may impact measurements."
            )

        torch.set_default_device(self._bench_args.init_device)

        # distributed setup required for parallel layers
        init_distributed_environment(
            backend="nccl",
            world_size=self._server_args.tp_size * self._server_args.pp_size,
            rank=0,
            local_rank=self._server_args.base_gpu_id,
            distributed_init_method=f"tcp://127.0.0.1:{self._server_args.nccl_port}",
            timeout=self._server_args.dist_timeout,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=self._server_args.tp_size,
            pipeline_model_parallel_size=self._server_args.pp_size,
            expert_model_parallel_size=self._server_args.ep_size,
            duplicate_tp_group=self._server_args.enable_pdmux,
        )

        # Pre processing required for model loader
        self._device_config = DeviceConfig(device=self._bench_args.init_device)
        self._model_config = ModelConfig.from_server_args(self._server_args)
        self._model_class, _ = get_model_architecture(self._model_config)
        self._load_config = LoadConfig(
            load_format=(
                LoadFormat.AUTO
                if self._bench_args.use_real_weights
                else LoadFormat.DUMMY
            )
        )
        self._quant_config = _get_quantization_config(
            self._model_config, self._load_config
        )
        self._hf_config = self._model_config.hf_config

        self._init_model()
        self._init_memory_pool()
        self._init_attention_backend()

        logger.info(
            f"ModelBench initialized for {self._server_args.model_path}, class: {self._model_class}, quant_config: {self._quant_config}"
        )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        destroy_model_parallel()
        destroy_distributed_environment()

    def torch_compile(self):
        torch_compile(self._model, self._server_args, self._model_config)

    # TODO: automate calling this in default exec or have warnings
    # if not called in cases where it's required like fa3
    def prepare_exec(self, forward_batch):
        if type(self._attn_backend) is FlashAttentionBackend:
            self._attn_backend.init_forward_metadata(forward_batch)

    def default_exec(self, *args):
        inputs = self._ensure_inputs_on_exec_device(*args)

        # model modifications
        if self._server_args.enable_torch_compile:
            self.torch_compile()

        # warmup
        for iter in range(self._bench_args.warmup_iters):
            _ = self._model(*inputs)

        # benchmark
        tic = time.perf_counter()
        for iter in range(self._bench_args.bench_iters):
            if not self._bench_args.disable_nvtx_tracing:
                torch.cuda.nvtx.range_push("RIG_PROFILE")
            result = self._model(*inputs)
            if not self._bench_args.disable_nvtx_tracing:
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        latency = time.perf_counter() - tic
        throughput = self._bench_args.bench_iters / latency
        logger.info(
            f"Total latency: {latency:6.5f} s, throughput: {throughput:9.2f} iters/sec"
        )
        return result

    def _init_model(self):
        """Initialize the model using the custom initializer via monkey-patching.

        Monkey-patches ``loader._initialize_model`` so that ``get_model()``
        delegates model construction to ``self._initializer`` instead of
        the default ``model_class(**kwargs)`` path.  This lets callers
        supply a partial model (e.g. just the MLP) while still going
        through the full weight-loading and device-placement pipeline.
        The original ``_initialize_model`` is always restored in the
        ``finally`` block.
        """
        import sglang.srt.model_loader.loader as loader_module

        original_initialize_model = loader_module._initialize_model
        orig_named_parameters = None

        def _custom_initialize_model(model_config, load_config, quant_config=None):
            nonlocal orig_named_parameters
            model = self._initializer(self, model_config.hf_config, quant_config)
            if self._load_config.load_format == LoadFormat.AUTO:
                weight_prefix = getattr(model, "_weight_prefix", "")

                if getattr(self._model_class, "load_weights", None) is not None:
                    model.load_weights = MethodType(
                        self._model_class.load_weights, model
                    )

                # Override named_parameters to prepend the weight prefix
                # so that params_dict keys match checkpoint weight names.
                # This is needed for stacked_params_mapping (e.g.
                # ".gate_proj" → ".gate_up_proj") which relies on the
                # leading dot from the full module path.
                # The override is restored after get_model() returns to
                # avoid confusing TorchDynamo's __func__ introspection.
                orig_named_parameters = model.named_parameters

                def _prefixed_named_parameters(prefix="", recurse=True):
                    effective = f"{weight_prefix}.{prefix}" if prefix else weight_prefix
                    return orig_named_parameters(prefix=effective, recurse=recurse)

                model.named_parameters = _prefixed_named_parameters

                model.config = self._hf_config
                model.quant_config = self._quant_config
                model.model = SimpleNamespace()

            return model

        loader_module._initialize_model = _custom_initialize_model
        try:
            self._model = get_model(
                model_config=self._model_config,
                load_config=self._load_config,
                device_config=self._device_config,
            )
        finally:
            loader_module._initialize_model = original_initialize_model
            if self._load_config.load_format == LoadFormat.AUTO:
                self._model.named_parameters = orig_named_parameters

    def _init_attention_backend(self):
        if self._server_args.attention_backend == "dummy":
            self._attn_backend = DummyAttentionBackend()
            self._attn_backend.set_out(
                torch.rand(
                    (
                        self._bench_args.num_tokens,
                        self._model_config.hf_config.hidden_size,
                    ),
                    dtype=self._model_config.dtype,
                )
            )
        elif self._server_args.attention_backend == "fa3":
            mock_model_runner = SimpleNamespace(
                mock_obj_name=ModelRunner.__name__,
                model_config=self._model_config,
                sliding_window_size=self._model_config.attention_chunk_size,
                server_args=self._server_args,
                device=self._bench_args.exec_device,
                req_to_token_pool=self._req_to_token_pool,
                kv_cache_dtype=self._kv_cache_dtype,
                is_hybrid=False,
                page_size=self._server_args.page_size,
            )
            self._attn_backend = FlashAttentionBackend(model_runner=mock_model_runner)  # type: ignore
        else:
            logger.debug(
                "No/Invalid attention backend was specified in server args hence none was initialized."
            )
            self._attn_backend = None

    # TODO: Expose more bench args controlling the memory pool initialization
    def _init_memory_pool(self):
        if self._server_args.kv_cache_dtype == "auto":
            kv_cache_quant_algo = getattr(
                self._quant_config, "kv_cache_quant_algo", None
            )
            if (
                isinstance(kv_cache_quant_algo, str)
                and kv_cache_quant_algo.upper() == "FP8"
            ):
                self._kv_cache_dtype = torch.float8_e4m3fn
            else:
                self._kv_cache_dtype = self._model_config.dtype
        elif self._server_args.kv_cache_dtype == "fp8_e5m2":
            self._kv_cache_dtype = torch.float8_e5m2
        elif self._server_args.kv_cache_dtype == "fp8_e4m3":
            self._kv_cache_dtype = torch.float8_e4m3fn
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self._server_args.kv_cache_dtype}."
            )

        logger.info(f"Using KV cache dtype: {self._kv_cache_dtype}")

        # TODO: Mock req_to_token pool if needed
        self._req_to_token_pool = ReqToTokenPool(
            size=self._bench_args.num_tokens,
            max_context_len=self._bench_args.num_tokens,
            device=self._bench_args.exec_device,
            enable_memory_saver=False,
        )

        # TODO: Mock token_to_kv_pool (KVCache) if needed
        self._token_to_kv_pool = MHATokenToKVPool(
            # Page index 0 is reserved (for something), KV cache allocation happens from page index 1
            size=(
                1 + math.ceil(self._bench_args.num_tokens / self._server_args.page_size)
            )
            * self._server_args.page_size,
            page_size=self._server_args.page_size,
            dtype=self._kv_cache_dtype,
            head_num=self.get_num_kv_heads(),
            head_dim=self._model_config.head_dim,
            # TODO: This is gonna depend on the model architecture run using the bench
            # essentially this number should be same as number of attention layers
            # should be configurable as part of the bench args
            layer_num=1,
            device=self._bench_args.exec_device,
            enable_memory_saver=False,
        )

    def _ensure_inputs_on_exec_device(self, *args) -> list:
        inputs = []
        # move to exec device if required
        if self._bench_args.exec_device != self._bench_args.init_device:
            self._model.to(self._bench_args.exec_device)

            if type(self._attn_backend) is DummyAttentionBackend:
                self._attn_backend.out_to(self._bench_args.exec_device)

            if type(self._attn_backend) is FlashAttentionBackend:
                self._attn_backend.forward_metadata.cache_seqlens_int32 = (
                    self._attn_backend.forward_metadata.cache_seqlens_int32.to(
                        self._bench_args.exec_device
                    )
                )
                self._attn_backend.forward_metadata.cu_seqlens_q = (
                    self._attn_backend.forward_metadata.cu_seqlens_q.to(
                        self._bench_args.exec_device
                    )
                )
                self._attn_backend.forward_metadata.cu_seqlens_k = (
                    self._attn_backend.forward_metadata.cu_seqlens_k.to(
                        self._bench_args.exec_device
                    )
                )

            for arg in args:
                if type(arg) is ForwardBatch or (
                    hasattr(arg, "mock_obj_name")
                    and arg.mock_obj_name == ForwardBatch.__name__
                ):
                    arg.out_cache_loc.to(self._bench_args.exec_device)
                    arg.seq_lens.to(self._bench_args.exec_device)
                    arg.req_pool_indices.to(self._bench_args.exec_device)
                    inputs.append(arg)
                    continue

                if hasattr(arg, "to"):
                    inputs.append(arg.to(self._bench_args.exec_device))
                else:
                    inputs.append(arg)
        else:
            inputs = args
        return inputs

    @property
    def model_config(self):
        return self._model_config

    @property
    def model(self):
        return self._model

    @property
    def bench_args(self):
        return self._bench_args

    @property
    def attn_backend(self):
        return self._attn_backend

    @property
    def hf_config(self):
        return self._hf_config

    @abstractmethod
    def get_num_kv_heads(self) -> int:
        pass


class LlamaBench(ModelBench):
    def __init__(
        self,
        server_args: ServerArgs,
        bench_args: ModelBenchArgs,
        initializer: Callable[
            [Self, PretrainedConfig, Optional[QuantizationConfig]], torch.nn.Module
        ],
    ) -> None:
        super().__init__(server_args, bench_args, initializer)

    def get_num_kv_heads(self):
        return self._hf_config.num_key_value_heads

    def init_attention(self):
        from sglang.srt.layers.radix_attention import RadixAttention

        return RadixAttention(
            self._hf_config.num_attention_heads,
            self._hf_config.head_dim,
            self._hf_config.head_dim**-0.5,
            num_kv_heads=self._hf_config.num_key_value_heads,
            layer_id=0,  # TODO: incerement this based on number of instantiations
            quant_config=self._quant_config,
            prefix="llama_attention",
        )

    def init_qkv_parallel_linear(self):
        from sglang.srt.layers.linear import QKVParallelLinear

        return QKVParallelLinear(
            self._hf_config.hidden_size,
            self._hf_config.head_dim,
            self._hf_config.num_attention_heads,
            self._hf_config.num_key_value_heads,
            bias=getattr(self._hf_config, "attention_bias", False)
            or getattr(self._hf_config, "bias", False),
            quant_config=self._quant_config,
            prefix="llama_qkv_proj",
        )

    def init_o_parallel_linear(self):
        from sglang.srt.layers.linear import RowParallelLinear

        return RowParallelLinear(
            self._hf_config.num_attention_heads * self._hf_config.head_dim,
            self._hf_config.hidden_size,
            bias=getattr(self._hf_config, "attention_bias", False)
            or getattr(self._hf_config, "bias", False),
            quant_config=self._quant_config,
            prefix="llama_o_proj",
        )

    def init_rope(self):
        from sglang.srt.layers.rotary_embedding import get_rope

        head_dim = self._hf_config.hidden_size // self._hf_config.num_attention_heads
        return get_rope(
            head_size=head_dim,
            rotary_dim=int(getattr(self._hf_config, "partial_rotary_factor", 1))
            * head_dim,
            max_position=getattr(self._hf_config, "max_position_embeddings", 8192),
            base=getattr(self._hf_config, "rope_theta", 10000),
            rope_scaling=getattr(self._hf_config, "rope_scaling", None),
            is_neox_style=getattr(self._hf_config, "rope_is_neox_style", True),
        )

    def init_norm(self):
        from sglang.srt.layers.layernorm import RMSNorm

        return RMSNorm(self._hf_config.hidden_size, eps=self._hf_config.rms_norm_eps)

    def init_decoder(self):
        from sglang.srt.models.llama import LlamaDecoderLayer

        return LlamaDecoderLayer(
            config=self._hf_config,  # type: ignore
            quant_config=self._quant_config,
            prefix="llama_decoder",
        )

    def init_mlp(self):
        from sglang.srt.models.llama import LlamaMLP

        prefix = add_prefix("mlp", self._prefix)
        model = LlamaMLP(
            self._hf_config.hidden_size,
            self._hf_config.intermediate_size,  # type: ignore
            self._hf_config.hidden_act,  # type: ignore
            self._quant_config,
            prefix=prefix,
        )
        model._weight_prefix = prefix
        return model

    def get_rand_input_forward_batch(self):
        forward_batch = SimpleNamespace(
            mock_obj_name=ForwardBatch.__name__,
            forward_mode=self._bench_args.forward_mode,
            attn_backend=self._attn_backend,
            out_cache_loc=torch.arange(
                start=self._server_args.page_size,
                end=self._server_args.page_size + self._bench_args.num_tokens,
                dtype=torch.int64,
            ),
            token_to_kv_pool=self._token_to_kv_pool,
            req_to_token_pool=self._req_to_token_pool,
            req_pool_indices=torch.zeros((1), dtype=torch.int64),
            seq_lens=torch.tensor([self._bench_args.num_tokens], dtype=torch.int64),
            seq_lens_cpu=torch.tensor(
                [self._bench_args.num_tokens], dtype=torch.int64, device="cpu"
            ),
            batch_size=1,  # TODO: Add support of batch size as a bench arg ?
            extend_prefix_lens_cpu=[],
            encoder_lens=None,
            spec_info=None,
        )
        return forward_batch

    def get_rand_input_positions(self):
        positions = torch.randint(
            100, (self._bench_args.num_tokens,), dtype=torch.int64
        )
        return positions

    def get_rand_input_hidden_states(self):
        hidden_states = torch.rand(
            (self._bench_args.num_tokens, self._model_config.hf_config.hidden_size),
            dtype=self._model_config.dtype,
        )
        return hidden_states

    def get_rand_input_q(self):
        num_heads = self._model_config.num_attention_heads
        head_dim = self._model_config.hidden_size // num_heads
        q = torch.rand(
            (self._bench_args.num_tokens, num_heads * head_dim),
            dtype=self._model_config.dtype,
        )
        return q

    def get_rand_input_k(self):
        num_heads = self._model_config.num_attention_heads
        num_kv_heads = self._model_config.num_key_value_heads
        head_dim = self._model_config.hidden_size // num_heads
        k = torch.rand(
            (self._bench_args.num_tokens, num_kv_heads * head_dim),
            dtype=self._model_config.dtype,
        )
        return k

    def get_rand_input_v(self):
        num_heads = self._model_config.num_attention_heads
        num_kv_heads = self._model_config.num_key_value_heads
        head_dim = self._model_config.hidden_size // num_heads
        v = torch.rand(
            (self._bench_args.num_tokens, num_kv_heads * head_dim),
            dtype=self._model_config.dtype,
        )
        return v

    def get_rand_attn_output(self):
        attn_output = torch.rand(
            (
                self._bench_args.num_tokens,
                self._model_config.hf_config.hidden_size,
            ),
            dtype=self._model_config.dtype,
        )
        return attn_output

    def get_rand_residual(self):
        residual = torch.rand(
            (self._bench_args.num_tokens, self._model_config.hf_config.hidden_size),
            dtype=self._model_config.dtype,
        )
        return residual
