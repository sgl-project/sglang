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
"""ModelRunner runs the forward passes of the models."""

import datetime
import gc
import inspect
import json
import logging
import os
import socket
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from sglang.srt.configs import (
    FalconH1Config,
    JetNemotronConfig,
    KimiLinearConfig,
    NemotronHConfig,
    Qwen3NextConfig,
)
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    get_nsa_index_head_dim,
    is_deepseek_nsa,
)
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed import (
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    attn_backend_wrapper,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_size,
    initialize_dp_attention,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator_ascend import AscendPagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    AscendMLAPagedTokenToKVPool,
    AscendTokenToKVPool,
    DoubleSparseTokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.npu_graph_runner import NPUGraphRunner
from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
    PiecewiseCudaGraphRunner,
)
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    MultiprocessingSerializer,
    cpu_has_amx_support,
    dynamic_import,
    enable_show_time_cost,
    get_available_gpu_memory,
    get_bool_env_var,
    get_cpu_ids_by_node,
    init_custom_process_group,
    is_cuda,
    is_float4_e2m1fn_x2,
    is_hip,
    is_npu,
    log_info_on_rank0,
    monkey_patch_p2p_access_check,
    set_cuda_arch,
    slow_rank_detector,
    xpu_has_xmx_support,
)
from sglang.srt.utils.offloader import (
    create_offloader_from_server_args,
    get_offloader,
    set_offloader,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

MLA_ATTENTION_BACKENDS = [
    "aiter",
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "flashmla",
    "cutlass_mla",
    "trtllm_mla",
    "ascend",
    "nsa",
]

CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = [
    "flashinfer",
    "fa3",
    "fa4",
    "flashmla",
    "cutlass_mla",
    "trtllm_mla",
]


def add_mla_attention_backend(backend_name):
    if backend_name not in MLA_ATTENTION_BACKENDS:
        MLA_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to MLA_ATTENTION_BACKENDS.")


def add_chunked_prefix_cache_attention_backend(backend_name):
    if backend_name not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS:
        CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS.append(backend_name)
        logger.info(
            f"Added {backend_name} to CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS."
        )


_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_xpu_xmx_available = xpu_has_xmx_support()

# Use a small KV cache pool size for tests in CI
SGLANG_CI_SMALL_KV_SIZE = os.getenv("SGLANG_CI_SMALL_KV_SIZE", None)

# Detect stragger ranks in model loading
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 300

# the ratio of mamba cache pool size to max_running_requests, it will be safe when it is larger than 2 (yizhang2077)
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3

logger = logging.getLogger(__name__)

if _is_npu:
    import torch_npu

    torch.npu.config.allow_internal_format = True
    torch_npu.npu.set_compile_mode(jit_compile=False)


class RankZeroFilter(logging.Filter):
    """Filter that only allows INFO level logs from rank 0, but allows all other levels from any rank."""

    def __init__(self, is_rank_zero):
        super().__init__()
        self.is_rank_zero = is_rank_zero

    def filter(self, record):
        if record.levelno == logging.INFO:
            return self.is_rank_zero
        return True


class ModelRunner:
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
    ):
        # Parse args
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.dp_size = server_args.dp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid = model_config.is_hybrid
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size
        self.forward_pass_id = 0
        self.init_new_workspace = False

        # Apply the rank zero filter to logger
        if server_args.show_time_cost:
            enable_show_time_cost()

        # Model-specific adjustment
        self.model_specific_adjustment()

        # Set the global server_args in the scheduler process
        set_global_server_args_for_scheduler(server_args)
        global_server_args = get_global_server_args()

        # FIXME: hacky set `use_mla_backend`
        global_server_args.use_mla_backend = self.use_mla_backend

        # Init OpenMP threads binding for CPU
        if self.device == "cpu":
            self.init_threads_binding()

        # Get memory before model loading
        min_per_gpu_memory = self.init_torch_distributed()

        # CPU offload
        set_offloader(create_offloader_from_server_args(server_args, dp_rank=dp_rank))

        if get_bool_env_var("SGLANG_DETECT_SLOW_RANK"):
            slow_rank_detector.execute()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # Initialize the model runner
        self.initialize(min_per_gpu_memory)

        # Temporary cached values
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

        # For weight updates
        self._model_update_group = {}
        self._weights_send_group = {}

        if (
            self.server_args.enable_piecewise_cuda_graph
            and self.can_run_piecewise_cuda_graph()
        ):
            self.attention_layers = []
            for layer in self.model.model.layers:
                if hasattr(layer, "self_attn"):
                    if hasattr(layer.self_attn, "attn"):
                        self.attention_layers.append(layer.self_attn.attn)
                    elif hasattr(layer.self_attn, "attn_mqa"):
                        # For DeepSeek model
                        self.attention_layers.append(layer.self_attn.attn_mqa)

            if len(self.attention_layers) < self.model_config.num_hidden_layers:
                # TODO(yuwei): support Non-Standard GQA
                log_info_on_rank0(
                    logger,
                    "Disable piecewise CUDA graph because some layers do not apply Standard GQA",
                )
                self.piecewise_cuda_graph_runner = None
            else:
                self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)
        else:
            self.piecewise_cuda_graph_runner = None

    def initialize(self, min_per_gpu_memory: float):
        server_args = self.server_args

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        if not self.is_draft_worker:
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(
                    server_args=server_args,
                    model_config=self.model_config,
                    moe_ep_rank=self.moe_ep_rank,
                )
            )
            if self.tp_rank == 0 and get_bool_env_var(
                "SGLANG_LOG_EXPERT_LOCATION_METADATA"
            ):
                logger.info(
                    f"Initial expert_location_metadata: {get_global_expert_location_metadata()}"
                )

            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.tp_rank,
                )
            )

        # Expert parallelism
        self.eplb_manager = (
            EPLBManager(self)
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )
        self.expert_location_updater = ExpertLocationUpdater()

        (
            ElasticEPStateManager.init(self.server_args)
            if self.server_args.elastic_ep_backend
            else None
        )
        # Load the model
        self.sampler = Sampler()
        self.load_model()

        # Check if the model is using hybrid SWA
        if (
            not self.server_args.disable_hybrid_swa_memory
            and self.sliding_window_size is not None
            and self.sliding_window_size > 0
        ):
            architectures = self.model_config.hf_config.architectures
            if architectures and not any("Llama4" in arch for arch in architectures):
                self.is_hybrid = self.model_config.is_hybrid = True

        if config := self.mamba2_config:
            class_name = config.__class__.__name__
            logger.warning(f"{class_name} model detected, disable radix cache")
            self.server_args.disable_radix_cache = True

        # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
        # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
        # determine the number of layers.
        model_has_mtp_layers = self.model_config.num_nextn_predict_layers is not None
        model_num_layers = (
            self.model_config.num_nextn_predict_layers
            if self.is_draft_worker and model_has_mtp_layers
            else max(
                self.model_config.num_hidden_layers,
                self.model_config.num_attention_layers,
            )
        )
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", model_num_layers)
        self.num_effective_layers = self.end_layer - self.start_layer
        assert (
            (not model_has_mtp_layers)
            or (self.spec_algorithm.is_none())
            or (
                (not self.spec_algorithm.is_none())
                and (self.num_effective_layers == model_num_layers)
            )
        ), "PP is not compatible with MTP models."

        # Apply torchao quantization
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            apply_torchao_config_to_model(
                self.model, get_global_server_args().torchao_config
            )

        # Apply torch TP if the model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

        # Init lora
        if server_args.enable_lora:
            self.init_lora_manager()

        # Init Double Sparsity
        if server_args.enable_double_sparsity:
            if server_args.ds_heavy_channel_type is None:
                raise ValueError(
                    "Please specify the heavy channel type for double sparsity optimization."
                )
            self.init_double_sparsity_channel_config(server_args.ds_heavy_channel_type)

        # Enable batch invariant mode
        if server_args.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

        # Init memory pool and attention backends
        self.init_memory_pool(
            min_per_gpu_memory,
            server_args.max_running_requests,
            server_args.max_total_tokens,
        )
        if self.device == "cuda":
            self.init_cublas()
            self.init_attention_backend()
            self.init_device_graphs()
        elif self.device in ["npu", "cpu"]:
            self.init_attention_backend()
            self.init_device_graphs()
        else:
            self.graph_runner = None
            self.graph_mem_usage = 0
            self.init_attention_backend()

        # auxiliary hidden capture mode. TODO: expose this to server args?
        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
            # load draft config
            draft_model_config = ModelConfig.from_server_args(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                is_draft_model=True,
            )

            try:
                # get the aux layer from draft model config
                eagle_config = getattr(
                    draft_model_config.hf_config, "eagle_config", None
                )
                eagle_aux_hidden_state_layer_ids = eagle_config[
                    "eagle_aux_hidden_state_layer_ids"
                ]
            except:
                # if there is no aux layer, set to None
                eagle_aux_hidden_state_layer_ids = None

            self.model.set_eagle3_layers_to_capture(eagle_aux_hidden_state_layer_ids)

    def model_specific_adjustment(self):
        server_args = self.server_args

        if server_args.enable_double_sparsity:
            logger.info(
                "Double sparsity optimization is turned on. Use triton backend without CUDA graph."
            )
            server_args.attention_backend = "triton"
            server_args.disable_cuda_graph = True

        if self.is_multimodal:
            if not self.is_multimodal_chunked_prefill_supported:
                server_args.chunked_prefill_size = -1
                logger.info(
                    f"Automatically turn off --chunked-prefill-size as it is not supported for "
                    f"{self.model_config.hf_config.model_type}"
                )

        if (
            not self.use_mla_backend
            or server_args.attention_backend
            not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
        ):
            server_args.disable_chunked_prefix_cache = True

        if not server_args.disable_chunked_prefix_cache:
            log_info_on_rank0(logger, "Chunked prefix cache is turned on.")

        if self.model_config.hf_config.model_type == "qwen3_vl_moe":
            if (
                quantization_config := getattr(
                    self.model_config.hf_config, "quantization_config", None
                )
            ) is not None and "weight_block_size" in quantization_config:
                weight_block_size_n = quantization_config["weight_block_size"][0]

                if self.tp_size % self.moe_ep_size != 0:
                    raise ValueError(
                        f"tp_size {self.tp_size} must be divisible by moe_ep_size {self.moe_ep_size}"
                    )
                moe_tp_size = self.tp_size // self.moe_ep_size

                moe_intermediate_size = (
                    self.model_config.hf_text_config.moe_intermediate_size
                )
                if moe_intermediate_size % moe_tp_size != 0:
                    raise ValueError(
                        f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({self.tp_size}) divided by moe_ep_size ({self.moe_ep_size})."
                    )

                if (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0:
                    raise ValueError(
                        f"For qwen3-vl-fp8 models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                        f"where moe_tp_size is equal to tp_size ({self.tp_size}) divided by moe_ep_size ({self.moe_ep_size}). "
                        f"You can fix this by setting arguments `--tp-size` and `--ep-size` correctly."
                    )

    def init_torch_distributed(self):
        logger.info("Init torch distributed begin.")

        try:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.tp_rank=} {self.tp_size=}"
            )
            raise

        if self.device == "cuda":
            if self.server_args.elastic_ep_backend == "mooncake":
                backend = "mooncake"
                if self.server_args.mooncake_ib_device:
                    mooncake_ib_device = self.server_args.mooncake_ib_device.split(",")
                    try:
                        from mooncake import ep as mooncake_ep

                        mooncake_ep.set_device_filter(mooncake_ib_device)
                    except:
                        pass  # A warning will be raised in `init_distributed_environment`
            else:
                backend = "nccl"
        elif self.device == "xpu":
            backend = "xccl"
        elif self.device == "hpu":
            backend = "hccl"
        elif self.device == "cpu":
            backend = "gloo"
        elif self.device == "npu":
            backend = "hccl"

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            monkey_patch_p2p_access_check()

        if self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
        set_torch_symm_mem_all_reduce(self.server_args.enable_torch_symm_mem)

        if not self.is_draft_worker:
            if self.device == "cpu":
                if _is_cpu_amx_available:
                    # Bind OpenMP threads to CPU cores
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

                    # Set local size to hint SGLang to use shared memory based AllReduce
                    os.environ["LOCAL_SIZE"] = str(self.tp_size)
                    torch.ops.sgl_kernel.initialize(self.tp_size, self.tp_rank)

                    @torch.library.register_fake("sgl_kernel::shm_allgather")
                    def _(data, dim):
                        return torch.cat([data] * self.tp_size, dim=dim)

                else:
                    logger.warning(
                        "init_cpu_threads_env and shared memory based AllReduce is disabled since intel amx backend is not available"
                    )

            # Only initialize the distributed environment on the target model worker.
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
            )
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                pipeline_model_parallel_size=self.pp_size,
                expert_model_parallel_size=self.moe_ep_size,
                duplicate_tp_group=self.server_args.enable_pdmux,
                torch_compile=self.server_args.enable_piecewise_cuda_graph,
            )
            initialize_dp_attention(
                server_args=self.server_args,
                model_config=self.model_config,
            )

        min_per_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1 and not self.is_draft_worker:
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                if get_bool_env_var("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"):
                    logger.warning(
                        "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                        f"{min_per_gpu_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                    )
                else:
                    raise ValueError(
                        "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                        f"{min_per_gpu_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                    )

        logger.info(
            f"Init torch distributed ends. mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return min_per_gpu_memory

    def load_model(self):
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        from sglang.srt.configs.modelopt_config import ModelOptConfig

        modelopt_config = ModelOptConfig(
            quant=self.server_args.modelopt_quant,
            checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
            checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
            export_path=self.server_args.modelopt_export_path,
            quantize_and_serve=self.server_args.quantize_and_serve,
        )

        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
            tp_rank=self.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            modelopt_config=modelopt_config,
        )
        if self.device == "cpu":
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.tp_size
            )

        if self.server_args.load_format == LoadFormat.REMOTE_INSTANCE:
            if self.tp_rank == 0:
                instance_ip = socket.gethostbyname(socket.gethostname())
                t = threading.Thread(
                    target=trigger_init_weights_send_group_for_remote_instance_request,
                    args=(
                        self.server_args.remote_instance_weight_loader_seed_instance_ip,
                        self.server_args.remote_instance_weight_loader_seed_instance_service_port,
                        self.server_args.remote_instance_weight_loader_send_weights_group_ports,
                        instance_ip,
                    ),
                )
                t.start()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        monkey_patch_vllm_parallel_state()

        with self.memory_saver_adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=self.server_args.enable_weights_cpu_backup,
        ):
            self.model = get_model(
                model_config=self.model_config,
                load_config=self.load_config,
                device_config=DeviceConfig(self.device, self.gpu_id),
            )
        monkey_patch_vllm_parallel_state(reverse=True)

        get_offloader().post_init()

        if self.server_args.kv_cache_dtype == "fp8_e4m3":
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        self.server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        self.sliding_window_size = None
        if hasattr(self.model, "get_attention_sliding_window_size"):
            self.sliding_window_size = self.model.get_attention_sliding_window_size()
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size
            logger.info(
                f"Setting sliding_window_size to be attention_chunk_size: {self.sliding_window_size}"
            )

        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        logger.info(
            f"Load weight end. "
            f"type={type(self.model).__name__}, "
            f"dtype={self.dtype}, "
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={self.weight_load_mem_usage:.2f} GB."
        )
        if self.server_args.debug_tensor_dump_output_folder is not None:
            register_forward_hook_for_model(
                self.model,
                self.server_args.debug_tensor_dump_output_folder,
                self.server_args.debug_tensor_dump_layers,
                self.tp_size,
                self.tp_rank,
                self.pp_rank,
            )

        if self.server_args.elastic_ep_backend == "mooncake":
            # Mooncake does not support `monitored_barrier`
            dist.barrier(group=get_tp_group().cpu_group)
        else:
            # Handle the case where some ranks do not finish loading.
            try:
                dist.monitored_barrier(
                    group=get_tp_group().cpu_group,
                    timeout=datetime.timedelta(
                        seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S
                    ),
                    wait_all_ranks=True,
                )
            except RuntimeError:
                raise ValueError(
                    f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
                ) from None

    def update_expert_location(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        if ElasticEPStateManager.instance() is not None:
            # TODO: refactor the weights update when elastic ep
            old_expert_location_metadata = get_global_expert_location_metadata()
            assert old_expert_location_metadata is not None
            old_expert_location_metadata.update(
                new_expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )
            self.update_weights_from_disk(
                self.server_args.model_path,
                self.server_args.load_format,
                lambda name: "mlp.experts" in name and "mlp.shared_experts" not in name,
            )
        else:
            self.expert_location_updater.update(
                self.model.routed_experts_weights_of_layer,
                new_expert_location_metadata,
                update_layer_ids=update_layer_ids,
                nnodes=self.server_args.nnodes,
                rank=self.tp_rank,
            )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            DefaultModelLoader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config

        if recapture_cuda_graph and self.device == "cuda":
            self.init_device_graphs()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_weights_send_group_for_remote_instance(
        self,
        master_address,
        ports,
        group_rank,
        world_size,
        group_name,
        backend="nccl",
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        logger.info(
            f"init custom process group: tp_rank={self.tp_rank}, gpu_id={self.gpu_id}, master_address={master_address}, master_port={group_port}, "
            f"group_rank={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        torch.cuda.empty_cache()
        success = False
        message = ""
        try:
            self._weights_send_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{group_port}",
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=torch.device("cuda", self.gpu_id),
            )
            dist.barrier(group=self._weights_send_group[group_name])
            success = True
            message = (
                f"Succeeded to init group through {master_address}:{group_port} group."
            )
        except Exception as e:
            message = f"Failed to init group: {e}."
            logger.error(message)

        torch.cuda.empty_cache()
        return success, message

    def send_weights_to_remote_instance(
        self,
        master_address,
        ports,
        group_name,
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        if self._weights_send_group[group_name] is not None:
            send_group = self._weights_send_group[group_name]
        else:
            message = f"Group {group_name} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first."
            logger.error(message)
            return False, message

        torch.cuda.empty_cache()
        success = False
        message = ""
        try:
            for _, weights in self.model.named_parameters():
                torch.distributed.broadcast(
                    weights,
                    src=0,
                    group=send_group,
                )
            success = True
            message = f"Succeeded to send weights through {master_address}:{group_port} {group_name}."
        except Exception as e:
            message = f"Failed to send weights: {e}."
            logger.error(message)

        # destroy the process group after sending weights
        del self._weights_send_group[group_name]
        torch.distributed.distributed_c10d.destroy_process_group(send_group)
        torch.cuda.empty_cache()
        return success, message

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def destroy_weights_update_group(self, group_name):
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
        load_format: Optional[str] = None,
    ):
        monkey_patch_torch_reductions()
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        self.device_module = torch.get_device_module(self.device)
        infered_device = self.device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format in self.server_args.custom_weight_loader:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
            server_args=self.server_args,
        )

    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface."""

        logger.info(
            f"LoRA adapter loading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.load_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter loading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    def unload_lora_adapter(self, lora_ref: LoRARef):
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading."""

        logger.info(
            f"LoRA adapter unloading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.unload_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter unloading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    def profile_max_num_token(self, total_gpu_memory: int):
        available_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        if self.is_draft_worker:
            num_layers = getattr(
                self.model_config.hf_config,
                "num_nextn_predict_layers",
                self.num_effective_layers,
            )
        elif config := self.mambaish_config:
            num_layers = len(config.full_attention_layer_ids)
        else:
            num_layers = self.num_effective_layers
        if self.use_mla_backend:
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * num_layers
                * torch._utils._element_size(self.kv_cache_dtype)
            )
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (
                        (
                            self.model_config.kv_lora_rank
                            + self.model_config.qk_rope_head_dim
                        )
                        // scale_block_size
                    )
                    * num_layers
                    * torch._utils._element_size(self.kv_cache_dtype)
                )

            # Add indexer KV cache overhead for NSA models (DeepSeek V3.2)
            if is_deepseek_nsa(self.model_config.hf_config):
                index_head_dim = get_nsa_index_head_dim(self.model_config.hf_config)
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // NSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    NSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                cell_size += indexer_size_per_token * num_layers * element_size
        else:
            cell_size = (
                self.model_config.get_num_kv_heads(get_attention_tp_size())
                * self.model_config.head_dim
                * num_layers
                * 2
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - self.mem_fraction_static
        )
        if self.mambaish_config is not None:
            rest_memory = self.handle_max_mamba_cache(rest_memory)
        max_num_token = int(rest_memory * (1 << 30) // cell_size)
        return max_num_token

    def handle_max_mamba_cache(self, total_rest_memory):
        config = self.mambaish_config
        server_args = self.server_args
        assert config is not None

        speculativa_ratio = (
            0
            if server_args.speculative_num_draft_tokens is None
            else server_args.speculative_num_draft_tokens
        )
        if (
            server_args.disable_radix_cache
            or config.mamba2_cache_params.mamba_cache_per_req == 0
        ):
            # with disable radix cache, sets the max_mamba_cache_size based on the max_running_requests
            if server_args.max_mamba_cache_size is None:
                if server_args.max_running_requests is not None:
                    server_args.max_mamba_cache_size = server_args.max_running_requests
                else:
                    server_args.max_mamba_cache_size = 512
        else:
            # allocate the memory based on the ratio between mamba state memory vs. full kv cache memory
            # solve the equations:
            # 1. mamba_state_memory + full_kv_cache_memory == total_rest_memory
            # 2. mamba_state_memory / full_kv_cache_memory == server_args.mamba_full_memory_ratio
            mamba_state_memory_raw = (
                total_rest_memory
                * server_args.mamba_full_memory_ratio
                / (1 + server_args.mamba_full_memory_ratio)
            )
            # calculate the max_mamba_cache_size based on the given total mamba memory
            server_args.max_mamba_cache_size = int(
                (mamba_state_memory_raw * (1 << 30))
                // config.mamba2_cache_params.mamba_cache_per_req
                // (1 + speculativa_ratio)
            )

        if self.hybrid_gdn_config is not None:
            server_args.max_mamba_cache_size = server_args.max_mamba_cache_size // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
        mamba_state_memory = (
            server_args.max_mamba_cache_size
            * config.mamba2_cache_params.mamba_cache_per_req
            * (1 + speculativa_ratio)
            / (1 << 30)
        )
        return total_rest_memory - mamba_state_memory

    @property
    def hybrid_gdn_config(self):
        config = self.model_config.hf_config
        if isinstance(config, Qwen3NextConfig | JetNemotronConfig):
            return config
        return None

    @property
    def mamba2_config(self):
        config = self.model_config.hf_config
        if isinstance(config, FalconH1Config | NemotronHConfig):
            return config
        return None

    @property
    def kimi_linear_config(self):
        config = self.model_config.hf_config
        if isinstance(config, KimiLinearConfig):
            return config
        return None

    @property
    def mambaish_config(self):
        return self.mamba2_config or self.hybrid_gdn_config or self.kimi_linear_config

    def set_num_token_hybrid(self):
        if (
            "Llama4ForConditionalGeneration"
            in self.model_config.hf_config.architectures
        ):
            temp_ratio = (
                (1 - self.is_hybrid)
                + self.is_hybrid
                * self.attention_chunk_size
                / self.model_config.context_len
            )
            self.swa_max_total_num_tokens = (
                4 * self.max_total_num_tokens * temp_ratio // (3 * temp_ratio + 1)
            )
            self.full_max_total_num_tokens = (
                4 * self.max_total_num_tokens
                - 12 * self.max_total_num_tokens * temp_ratio // (3 * temp_ratio + 1)
            )
            self.swa_max_total_num_tokens = int(
                self.swa_max_total_num_tokens
                // self.server_args.page_size
                * self.server_args.page_size
            )
            self.full_max_total_num_tokens = int(
                self.full_max_total_num_tokens
                // self.server_args.page_size
                * self.server_args.page_size
            )
            self.max_total_num_tokens = self.full_max_total_num_tokens
        else:
            assert self.sliding_window_size is not None and self.sliding_window_size > 0
            full_attention_layer_ids = []
            swa_attention_layer_ids = []

            try:
                layers = self.model.model.layers
            except:
                try:
                    layers = self.model.language_model.model.layers
                except:
                    try:
                        layers = self.model.language_model.layers
                    except:
                        self.is_hybrid = False
                        return

            for layer in layers:
                if (
                    layer.self_attn.attn.sliding_window_size is None
                    or layer.self_attn.attn.sliding_window_size == -1
                ):
                    full_attention_layer_ids.append(layer.layer_id)
                else:
                    swa_attention_layer_ids.append(layer.layer_id)
            self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
            self.model_config.full_attention_layer_ids = full_attention_layer_ids

            # Algorithm:
            # Existing max_total_num_tokens is per layer and assume all layers have the same number of tokens.
            # - Find total # of tokens available across layers.
            # - Calculate full_max_total_num_tokens and swa_max_total_num_tokens based on the given swa_full_tokens_ratio.
            total_tokens = (
                self.max_total_num_tokens * self.model_config.num_hidden_layers
            )
            full_layers_num = len(full_attention_layer_ids)
            swa_layers_num = len(swa_attention_layer_ids)
            swa_full_tokens_ratio = self.server_args.swa_full_tokens_ratio

            # Solve the equations:
            # 1. swa_max_total_num_tokens * swa_layers_num + full_max_total_num_tokens * full_layers_num == total_tokens
            # 2. full_max_total_num_tokens * swa_full_tokens_ratio == swa_max_total_num_tokens
            denominator = swa_full_tokens_ratio * swa_layers_num + full_layers_num
            self.full_max_total_num_tokens = int(total_tokens / denominator)
            self.swa_max_total_num_tokens = int(
                self.full_max_total_num_tokens * swa_full_tokens_ratio
            )
            self.max_total_num_tokens = self.full_max_total_num_tokens

            logger.info(
                f"Use Sliding window memory pool. full_layer_tokens={self.full_max_total_num_tokens}, swa_layer_tokens={self.swa_max_total_num_tokens}"
            )

    def can_run_piecewise_cuda_graph(self):
        if self.server_args.disable_cuda_graph:
            log_info_on_rank0(
                logger, "Disable piecewise CUDA graph because disable_cuda_graph is set"
            )
            return False
        if self.server_args.enable_torch_compile:
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because piecewise_cuda_graph has conflict with torch compile",
            )
            return False
        if self.pp_size > 1:
            # TODO(yuwei): support PP
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because piecewise_cuda_graph does not support PP",
            )
            return False
        return True

    def init_memory_pool(
        self,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        # Determine the kv cache dtype
        if self.server_args.kv_cache_dtype == "auto":
            quant_config = getattr(self.model, "quant_config", None)
            kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
            if (
                isinstance(kv_cache_quant_algo, str)
                and kv_cache_quant_algo.upper() == "FP8"
            ):
                if _is_hip:
                    self.kv_cache_dtype = torch.float8_e4m3fnuz
                else:
                    self.kv_cache_dtype = torch.float8_e4m3fn
            else:
                self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = torch.float8_e5m2fnuz
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = torch.float8_e4m3fnuz
            else:
                self.kv_cache_dtype = torch.float8_e4m3fn
        elif self.server_args.kv_cache_dtype in ("bf16", "bfloat16"):
            self.kv_cache_dtype = torch.bfloat16
        elif self.server_args.kv_cache_dtype == "fp4_e2m1":
            if hasattr(torch, "float4_e2m1fn_x2"):
                self.kv_cache_dtype = torch.float4_e2m1fn_x2
                logger.warning(f"FP4 (E2M1) KV Cache might lead to a accuracy drop!")
            else:
                logger.warning(
                    f"--kv-cache-dtype falls back to 'auto' because this torch version does not support torch.float4_e2m1fn_x2"
                )
                self.kv_cache_dtype = self.dtype
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

        log_info_on_rank0(logger, f"Using KV cache dtype: {self.kv_cache_dtype}")

        self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)
        if SGLANG_CI_SMALL_KV_SIZE:
            self.max_total_num_tokens = int(SGLANG_CI_SMALL_KV_SIZE)

        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(
                        self.max_total_num_tokens / self.model_config.context_len * 512
                    ),
                    2048,
                ),
                4096,
            )

        if self.mambaish_config is not None:
            ratio = (
                MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO
                if not self.server_args.disable_radix_cache
                else 1
            )
            max_num_reqs = min(
                max_num_reqs, self.server_args.max_mamba_cache_size // ratio
            )

        if self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone():
            if self.is_draft_worker:
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                max_num_reqs = self.server_args.max_num_reqs
            else:
                # We are sharing the `token_to_kv_pool`, and both verify and draft tokens
                # can be concurrently allocated, so we should give a headroom for it.
                self.server_args.draft_runner_cache_size = (
                    self.max_total_num_tokens
                    # draft
                    + max_num_reqs
                    * self.server_args.speculative_num_steps
                    * self.server_args.speculative_eagle_topk
                    # verify
                    + max_num_reqs * self.server_args.speculative_num_draft_tokens
                    # buffer
                    + 100
                )
                # Target worker and draft worker shares the same indices for the
                # token_to_kv_pool, so we should make sure to match max_total_num_tokens.
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                self.server_args.max_num_reqs = max_num_reqs

        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logging.warning(
                    f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                    f"{self.max_total_num_tokens}. "
                    f"Use the profiled value instead."
                )
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)

        self.max_total_num_tokens = (
            self.max_total_num_tokens
            // self.server_args.page_size
            * self.server_args.page_size
        )
        # different pp rank may have different num of layers, so we need to reduce the max_total_num_tokens
        if self.pp_size > 1:
            tensor = torch.tensor(self.max_total_num_tokens, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            self.max_total_num_tokens = tensor.item()

        # create token size for hybrid cache
        if self.is_hybrid:
            self.set_num_token_hybrid()

        if self.max_total_num_tokens <= 0:
            raise RuntimeError(
                f"Not enough memory. Please try to increase --mem-fraction-static. "
                f"Current value: {self.server_args.mem_fraction_static=}"
            )

        # Initialize req_to_token_pool
        if self.req_to_token_pool is None:
            # FIXME(lsyin): this is the temporary fix for the context length issue when using speculative decoding
            extra_max_context_len = 4
            if self.server_args.speculative_num_draft_tokens is not None:
                extra_max_context_len += self.server_args.speculative_num_draft_tokens

            if self.server_args.disaggregation_mode == "decode":
                from sglang.srt.disaggregation.decode import (
                    DecodeReqToTokenPool,
                    HybridMambaDecodeReqToTokenPool,
                )

                # subscribe memory for pre-allocated requests
                # if max_num_reqs <= 32, we pre-allocate 2x requests
                pre_alloc_size = max_num_reqs * 2 if max_num_reqs <= 32 else 0
                if config := self.mambaish_config:
                    self.req_to_token_pool = HybridMambaDecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        cache_params=config.mamba2_cache_params,
                        speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
                        pre_alloc_size=pre_alloc_size,
                    )
                else:
                    self.req_to_token_pool = DecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        pre_alloc_size=pre_alloc_size,
                    )
            elif config := self.mambaish_config:
                self.req_to_token_pool = HybridReqToTokenPool(
                    size=max_num_reqs,
                    mamba_size=self.server_args.max_mamba_cache_size,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    cache_params=config.mamba2_cache_params,
                    speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
                )
            else:
                self.req_to_token_pool = ReqToTokenPool(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                )
        else:
            # Draft worker shares req_to_token_pool with the target worker.
            assert self.is_draft_worker

        # Initialize token_to_kv_pool
        is_nsa_model = is_deepseek_nsa(self.model_config.hf_config)
        if self.server_args.attention_backend == "ascend":
            if self.use_mla_backend:
                self.token_to_kv_pool = AscendMLAPagedTokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=self.model_config.index_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                self.token_to_kv_pool = AscendTokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif self.use_mla_backend and is_nsa_model:
            self.token_to_kv_pool = NSATokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                index_head_dim=get_nsa_index_head_dim(self.model_config.hf_config),
            )
        elif self.use_mla_backend and not self.mambaish_config:
            assert not is_nsa_model
            self.token_to_kv_pool = MLATokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )
        elif self.server_args.enable_double_sparsity:
            self.token_to_kv_pool = DoubleSparseTokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                heavy_channel_num=self.server_args.ds_heavy_channel_num,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )
        else:
            if self.is_hybrid:
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    enable_kvcache_transpose=False,
                    device=self.device,
                )
            elif config := self.mambaish_config:
                extra_args = {}
                if self.use_mla_backend:
                    extra_args = {
                        "kv_lora_rank": self.model_config.kv_lora_rank,
                        "qk_rope_head_dim": self.model_config.qk_rope_head_dim,
                    }
                self.token_to_kv_pool = HybridLinearKVPool(
                    page_size=self.page_size,
                    size=self.max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    # if draft worker, we only need 1 attention layer's kv pool
                    full_attention_layer_ids=(
                        [0] if self.is_draft_worker else config.full_attention_layer_ids
                    ),
                    enable_kvcache_transpose=False,
                    device=self.device,
                    mamba_pool=self.req_to_token_pool.mamba_pool,
                    use_mla=self.use_mla_backend,
                    **extra_args,
                )
            else:
                self.token_to_kv_pool = MHATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    enable_alt_stream=not self.server_args.enable_pdmux,
                    enable_kv_cache_copy=(
                        self.server_args.speculative_algorithm is not None
                    ),
                )

        # Initialize token_to_kv_pool_allocator
        need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
        if self.token_to_kv_pool_allocator is None:
            if _is_npu and (
                self.server_args.attention_backend == "ascend"
                or self.hybrid_gdn_config is not None
            ):
                self.token_to_kv_pool_allocator = AscendPagedTokenToKVPoolAllocator(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=need_sort,
                )
            else:
                if self.page_size == 1:
                    if self.is_hybrid:
                        self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                            self.full_max_total_num_tokens,
                            self.swa_max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )
                    else:
                        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                            self.max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )
                else:
                    assert not self.is_hybrid
                    self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
        else:
            assert self.is_draft_worker

        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        if self.server_args.enable_pdmux:
            self.attn_backend = self._get_attention_backend(init_new_workspace=True)
            self.decode_attn_backend_group = []
            for _ in range(self.server_args.sm_group_num):
                self.decode_attn_backend_group.append(self._get_attention_backend())
            self.decode_attn_backend = self.decode_attn_backend_group[0]
        elif self.server_args.enable_two_batch_overlap and not self.is_draft_worker:
            self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
        else:
            self.attn_backend = self._get_attention_backend()

    def _get_attention_backend(self, init_new_workspace: bool = False):
        """Init attention kernel backend."""
        self.prefill_attention_backend_str, self.decode_attention_backend_str = (
            self.server_args.get_attention_backends()
        )

        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                self,
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            get_global_server_args().prefill_attention_backend,
            get_global_server_args().decode_attention_backend,
        ) = (self.prefill_attention_backend_str, self.decode_attention_backend_str)
        return attn_backend

    def _get_attention_backend_from_str(
        self, backend_str: str, init_new_workspace: bool = False
    ):
        if backend_str not in ATTENTION_BACKENDS:
            raise ValueError(f"Invalid attention backend: {backend_str}")
        self.init_new_workspace = init_new_workspace
        full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
        return attn_backend_wrapper(self, full_attention_backend)

    def init_double_sparsity_channel_config(self, selected_channel):
        selected_channel = "." + selected_channel + "_proj"
        self.sorted_channels = []
        # load channel config
        with open(self.server_args.ds_channel_config_path, "r") as f:
            channel_config = json.load(f)

        for i in range(self.start_layer, self.end_layer):
            key = "model.layers." + str(i) + ".self_attn" + selected_channel
            self.sorted_channels.append(
                torch.tensor(channel_config[key])[
                    :, : self.server_args.ds_heavy_channel_num
                ]
                .contiguous()
                .cuda()
            )

    def init_device_graphs(self):
        """Capture device graphs."""
        self.graph_runner = None
        self.graph_mem_usage = 0

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.device != "cpu" and self.server_args.disable_cuda_graph:
            return

        if self.device == "cpu" and not self.server_args.enable_torch_compile:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture {'cpu graph' if self.device == 'cpu' else 'cuda graph'} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        graph_runners = defaultdict(
            lambda: CudaGraphRunner,
            {
                "cpu": CPUGraphRunner,
                "npu": NPUGraphRunner,
            },
        )
        self.graph_runner = graph_runners[self.device](self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {'cpu graph' if self.device == 'cpu' else 'cuda graph'} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_threads_binding(self):
        omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        if omp_cpuids == "all":
            assert self.tp_size <= n_numa_node, (
                f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
                f"tp_size {self.tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
                f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
                f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
                f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
                f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
                f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "
                f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
            )
            if self.tp_size < n_numa_node:
                logger.warning(
                    f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {self.tp_size}, so only {self.tp_size} numa nodes are used."
                )
            self.local_omp_cpuid = cpu_ids_by_node[self.tp_rank]
        else:
            threads_bind_list = omp_cpuids.split("|")
            assert self.tp_size == len(threads_bind_list), (
                f"SGLANG_CPU_OMP_THREADS_BIND setting must be aligned with TP size parameter ({self.tp_size}). "
                f"Please double check your settings."
            )
            self.local_omp_cpuid = threads_bind_list[self.tp_rank]
            if self.tp_size > n_numa_node:
                logger.warning(
                    f"TP size ({self.tp_size})is larger than numa node number ({n_numa_node}), "
                    f"in this case the available memory amount of each rank cannot be determined in prior. "
                    f"Please set proper `--max-total-tokens` to avoid the out-of-memory error."
                )

    def apply_torch_tp(self):
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.layers.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def update_decode_attn_backend(self, stream_idx: int):
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            if self.server_args.enable_pdmux:
                self.decode_attn_backend.init_forward_metadata(forward_batch)
                forward_batch.attn_backend = self.decode_attn_backend
            else:
                self.attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if not self.is_generation:
            kwargs["get_embedding"] = True

        if (
            self.piecewise_cuda_graph_runner is not None
            and self.piecewise_cuda_graph_runner.can_run(forward_batch)
        ):
            return self.piecewise_cuda_graph_runner.replay(forward_batch, **kwargs)

        if not skip_attn_backend_init:
            self.attn_backend.init_forward_metadata(forward_batch)

        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> LogitsProcessorOutput:
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        ret = self.model.forward_split_prefill(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            (forward_batch.split_index, next_split_index),
        )
        forward_batch.split_index = next_split_index
        return ret

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> Tuple[Union[LogitsProcessorOutput, PPProxyTensors], bool]:
        self.forward_pass_id += 1

        with get_global_expert_distribution_recorder().with_forward_pass(
            self.forward_pass_id,
            forward_batch,
        ):
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        return output

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> Tuple[Union[LogitsProcessorOutput, PPProxyTensors], bool]:
        mode_check = (
            forward_batch.forward_mode.is_cpu_graph
            if self.device == "cpu"
            else forward_batch.forward_mode.is_cuda_graph
        )
        can_run_graph = bool(
            mode_check()
            and self.graph_runner
            and self.graph_runner.can_run(forward_batch)
        )

        if can_run_graph:
            ret = self.graph_runner.replay(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return ret, can_run_graph

        # For MLP sync
        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.prepare_mlp_sync_batch(self)

        if forward_batch.forward_mode.is_decode():
            ret = self.forward_decode(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode.is_split_prefill():
            ret = self.forward_split_prefill(
                forward_batch,
                reinit_attn_backend=reinit_attn_backend,
                forward_count=split_forward_count,
            )
        elif forward_batch.forward_mode.is_extend():
            ret = self.forward_extend(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode.is_idle():
            ret = self.forward_idle(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

        if (
            forward_batch.global_num_tokens_cpu is not None
            and self.pp_group.is_last_rank
        ):
            forward_batch.post_forward_mlp_sync_batch(ret)

        return ret, can_run_graph

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.

        # Calculate logits bias and apply it to next_token_logits.
        sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        # For duplex models with multiple output streams.
        if isinstance(logits_output, tuple):
            return torch.stack(
                [self.sample(values, forward_batch) for values in logits_output],
                axis=-1,
            )

        self._preprocess_logits(logits_output, forward_batch.sampling_info)
        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        return next_token_ids

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """
        Compute token_ids_logprobs without performing sampling.

        Optimized path for prefill-only requests that need token_ids_logprobs but don't
        require next token generation. Skips expensive sampling operations
        while still providing requested probability information.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
        """
        if not forward_batch.token_ids_logprobs:
            return

        # Preprocess logits (same as in sample method)
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        rope_scaling = getattr(self.model_config.hf_text_config, "rope_scaling", {})
        if rope_scaling is None:
            return False
        is_mrope_enabled = "mrope_section" in rope_scaling
        return is_mrope_enabled

    def save_remote_model(self, url: str):
        from sglang.srt.model_loader.loader import RemoteModelLoader

        logger.info(f"Saving model to {url}")
        RemoteModelLoader.save_model(self.model, self.model_config.model_path, url)

    def save_sharded_model(
        self, path: str, pattern: Optional[str] = None, max_size: Optional[int] = None
    ):
        from sglang.srt.model_loader.loader import ShardedStateLoader

        logger.info(
            f"Save sharded model to {path} with pattern {pattern} and max_size {max_size}"
        )
        ShardedStateLoader.save_model(self.model, path, pattern, max_size)

    def update_weights_from_ipc(self, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
