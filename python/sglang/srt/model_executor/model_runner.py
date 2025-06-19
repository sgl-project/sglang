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
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from sglang.srt import debug_utils
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.distributed import (
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_size,
    initialize_dp_attention,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization import (
    deep_gemm_wrapper,
    monkey_patch_isinstance_for_vllm_base_layer,
)
from sglang.srt.layers.sampler import Sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.layers.utils import is_sm100_supported
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.managers.eplb_manager import EPLBManager
from sglang.srt.managers.expert_distribution import (
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.managers.expert_location import (
    ExpertLocationMetadata,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.managers.schedule_batch import (
    GLOBAL_SERVER_ARGS_KEYS,
    global_server_args_dict,
)
from sglang.srt.mem_cache.memory_pool import (
    DoubleSparseTokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.paged_allocator import (
    HeapPagedTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.expert_location_updater import ExpertLocationUpdater
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.hpu_graph_runner import HPUGraphRunner
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    MultiprocessingSerializer,
    cpu_has_amx_support,
    dynamic_import,
    enable_show_time_cost,
    get_available_gpu_memory,
    get_bool_env_var,
    init_custom_process_group,
    is_cuda,
    is_fa3_default_architecture,
    is_flashinfer_available,
    is_hip,
    is_hopper_with_cuda_12_3,
    is_hpu,
    is_no_spec_infer_or_topk_one,
    monkey_patch_p2p_access_check,
    monkey_patch_vllm_gguf_config,
    set_cpu_offload_max_bytes,
    set_cuda_arch,
)

_is_hpu = is_hpu()
_is_hip = is_hip()
_is_cpu_amx_available = cpu_has_amx_support()

# Use a small KV cache pool size for tests in CI
SGLANG_CI_SMALL_KV_SIZE = os.getenv("SGLANG_CI_SMALL_KV_SIZE", None)

# Detect stragger ranks in model loading
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 300

logger = logging.getLogger(__name__)


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
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocator] = None,
    ):
        # Parse args
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id

        # Apply the rank zero filter to logger
        if not any(isinstance(f, RankZeroFilter) for f in logger.filters):
            logger.addFilter(RankZeroFilter(tp_rank == 0))
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = server_args.dp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
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
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size

        self.forward_pass_id = 0

        # Model-specific adjustment
        self.model_specific_adjustment()

        if server_args.show_time_cost:
            enable_show_time_cost()

        # Global vars
        global_server_args_dict.update(
            {k: getattr(server_args, k) for k in GLOBAL_SERVER_ARGS_KEYS}
            | {
                # TODO it is indeed not a "server args"
                "use_mla_backend": self.use_mla_backend,
                "speculative_algorithm": self.spec_algorithm,
            }
        )

        # CPU offload
        set_cpu_offload_max_bytes(int(server_args.cpu_offload_gb * 1024**3))

        # Get memory before model loading
        min_per_gpu_memory = self.init_torch_distributed()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # If it is a draft model, tp_group can be different
        self.initialize(min_per_gpu_memory)

        # temporary cached values
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

    def initialize(self, min_per_gpu_memory: float):
        server_args = self.server_args
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        if not self.is_draft_worker:
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(server_args, self.model_config)
            )
            if self.tp_rank == 0 and get_bool_env_var(
                "SGLANG_LOG_EXPERT_LOCATION_METADATA"
            ):
                logger.info(
                    f"Initial expert_location_metadata: {get_global_expert_location_metadata().debug_str()}"
                )

            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.tp_rank,
                )
            )

        self.eplb_manager = (
            EPLBManager(self)
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )
        self.expert_location_updater = ExpertLocationUpdater()

        # Load the model
        self.sampler = Sampler()
        self.load_model()

        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(
            self.model, "end_layer", self.model_config.num_hidden_layers
        )
        self.num_effective_layers = self.end_layer - self.start_layer

        # Apply torchao quantization
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            apply_torchao_config_to_model(
                self.model, global_server_args_dict["torchao_config"]
            )

        # Apply torch TP if the model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

        # Init lora
        if server_args.lora_paths is not None:
            self.init_lora_manager()

        # Init memory pool and attention backends
        self.init_memory_pool(
            min_per_gpu_memory,
            server_args.max_running_requests,
            server_args.max_total_tokens,
        )
        if self.device == "cuda":
            self.init_cublas()
            self.init_attention_backend()
            self.init_cuda_graphs()
        elif self.device == "hpu":
            self.init_attention_backend()
            self.cuda_graph_runner = HPUGraphRunner(self)
        else:
            self.cuda_graph_runner = None
            self.init_attention_backend()

        # auxiliary hidden capture mode. TODO: expose this to server args?
        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
            self.model.set_eagle3_layers_to_capture()

    def model_specific_adjustment(self):
        server_args = self.server_args

        if (
            server_args.attention_backend == "intel_amx"
            and server_args.device == "cpu"
            and not _is_cpu_amx_available
        ):
            logger.info(
                "The current platform does not support Intel AMX, will fallback to torch_native backend."
            )
            server_args.attention_backend = "torch_native"

        if server_args.attention_backend is None:
            """
            Auto select the fastest attention backend.

            1. Models with MHA Architecture (e.g: Llama, QWen)
                1.1 We will turn on FA3 on hopper unless user use spec decode with topk > 1 or page_size > 1.
                1.2 In other cases, we will use flashinfer if available, otherwise use triton.
            2. Models with MLA Architecture and using FA3
                2.1 We will use FA3 backend on hopper.
                2.2 We will use Flashinfer backend on blackwell.
                2.3 Otherwise, we will use triton backend.
            """

            if not self.use_mla_backend:
                # MHA architecture
                if (
                    is_hopper_with_cuda_12_3()
                    and is_no_spec_infer_or_topk_one(server_args)
                    and is_fa3_default_architecture(self.model_config.hf_config)
                ):
                    server_args.attention_backend = "fa3"
                elif _is_hip:
                    server_args.attention_backend = "aiter"
                else:
                    server_args.attention_backend = (
                        "flashinfer" if is_flashinfer_available() else "triton"
                    )
            else:
                # MLA architecture
                if is_hopper_with_cuda_12_3():
                    server_args.attention_backend = "fa3"
                elif is_sm100_supported():
                    server_args.attention_backend = "flashinfer"
                elif _is_hip:
                    head_num = self.model_config.get_num_kv_heads(self.tp_size)
                    # TODO current aiter only support head number 16 or 128 head number
                    if (
                        head_num == 128 or head_num == 16
                    ) and self.spec_algorithm.is_none():
                        server_args.attention_backend = "aiter"
                    else:
                        server_args.attention_backend = "triton"
                else:
                    server_args.attention_backend = "triton"
            logger.info(
                f"Attention backend not set. Use {server_args.attention_backend} backend by default."
            )
        elif self.use_mla_backend:
            if server_args.device != "cpu":
                if server_args.attention_backend in [
                    "aiter",
                    "flashinfer",
                    "fa3",
                    "triton",
                    "flashmla",
                    "cutlass_mla",
                ]:
                    logger.info(
                        f"MLA optimization is turned on. Use {server_args.attention_backend} backend."
                    )
                else:
                    raise ValueError(
                        f"Invalid attention backend for MLA: {server_args.attention_backend}"
                    )
            else:
                if server_args.attention_backend != "intel_amx":
                    raise ValueError(
                        "MLA optimization not supported on CPU except for intel_amx backend."
                    )

        if (
            server_args.attention_backend == "fa3"
            and server_args.kv_cache_dtype == "fp8_e5m2"
        ):
            logger.warning(
                "FlashAttention3 only supports fp8_e4m3 if using FP8; "
                "Setting attention backend to triton."
            )
            server_args.attention_backend = "triton"

        if server_args.enable_double_sparsity:
            logger.info(
                "Double sparsity optimization is turned on. Use triton backend without CUDA graph."
            )
            server_args.attention_backend = "triton"
            server_args.disable_cuda_graph = True
            if server_args.ds_heavy_channel_type is None:
                raise ValueError(
                    "Please specify the heavy channel type for double sparsity optimization."
                )
            self.init_double_sparsity_channel_config(server_args.ds_heavy_channel_type)

        if self.is_multimodal:
            self.mem_fraction_static *= 0.90
            logger.info(
                f"Automatically reduce --mem-fraction-static to {self.mem_fraction_static:.3f} "
                f"because this is a multimodal model."
            )
            if not self.is_multimodal_chunked_prefill_supported:
                server_args.chunked_prefill_size = -1
                logger.info(
                    f"Automatically turn of --chunked-prefill-size as it is not supported for "
                    f"{self.model_config.hf_config.model_type}"
                )

        if not self.use_mla_backend:
            server_args.disable_chunked_prefix_cache = True
        elif self.page_size > 1:
            logger.info("Disable chunked prefix cache when page size > 1.")
            server_args.disable_chunked_prefix_cache = True

        if not server_args.disable_chunked_prefix_cache:
            logger.info("Chunked prefix cache is turned on.")

        if server_args.attention_backend == "aiter":
            if self.model_config.context_len > 8192:
                self.mem_fraction_static *= 0.85

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

        if not self.is_draft_worker:
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
            )
            initialize_dp_attention(
                enable_dp_attention=self.server_args.enable_dp_attention,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_size=self.server_args.dp_size,
                moe_dense_tp_size=self.server_args.moe_dense_tp_size,
                pp_size=self.server_args.pp_size,
            )

        min_per_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        self.tp_group = get_tp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1:
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
        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
        )
        if self.server_args.load_format == "gguf":
            monkey_patch_vllm_gguf_config()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        monkey_patch_vllm_parallel_state()
        monkey_patch_isinstance_for_vllm_base_layer()

        with self.memory_saver_adapter.region():
            self.model = get_model(
                model_config=self.model_config,
                load_config=self.load_config,
                device_config=DeviceConfig(self.device),
            )
        monkey_patch_vllm_parallel_state(reverse=True)
        monkey_patch_isinstance_for_vllm_base_layer(reverse=True)

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
        self.sliding_window_size = (
            self.model.get_attention_sliding_window_size()
            if hasattr(self.model, "get_attention_sliding_window_size")
            else None
        )
        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight end. "
            f"type={type(self.model).__name__}, "
            f"dtype={self.dtype}, "
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={(before_avail_memory - after_avail_memory):.2f} GB."
        )

        # Handle the case where some ranks do not finish loading.
        try:
            dist.monitored_barrier(
                group=get_tp_group().cpu_group,
                timeout=datetime.timedelta(seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S),
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
        self.expert_location_updater.update(
            self.model.routed_experts_weights_of_layer,
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=self.server_args.nnodes,
            rank=self.tp_rank,
        )

    def update_weights_from_disk(
        self, model_path: str, load_format: str
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
        loader = get_model_loader(load_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
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

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

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
            self._model_update_group = init_custom_process_group(
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

    def update_weights_from_distributed(self, name, dtype, shape):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """
        target_dtype = (
            dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        )

        assert (
            self._model_update_group is not None
        ), "model update group must be initialized"

        try:
            weights = torch.empty(shape, dtype=target_dtype, device=self.device)
            torch.distributed.broadcast(weights, src=0, group=self._model_update_group)
            self.model.load_weights([(name, weights)])
            return True, f"Succeeded to update parameter {name} online."

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
        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank))
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
            lora_paths=self.server_args.lora_paths,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        logger.info("LoRA manager ready.")

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
        else:
            num_layers = self.num_effective_layers
        if self.use_mla_backend:
            # FIXME: pipeline parallelism is not compatible with mla backend
            assert self.pp_size == 1
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * num_layers
                * torch._utils._element_size(self.kv_cache_dtype)
            )
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
        max_num_token = int(rest_memory * (1 << 30) // cell_size)
        return max_num_token

    def init_memory_pool(
        self,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = torch.float8_e5m2fnuz
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if is_cuda():
                self.kv_cache_dtype = torch.float8_e4m3fn
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

        self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)

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

        if SGLANG_CI_SMALL_KV_SIZE:
            self.max_total_num_tokens = int(SGLANG_CI_SMALL_KV_SIZE)

        if not self.spec_algorithm.is_none():
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

        if self.max_total_num_tokens <= 0:
            raise RuntimeError(
                "Not enough memory. Please try to increase --mem-fraction-static."
            )

        if self.req_to_token_pool is None:
            if self.server_args.disaggregation_mode == "decode":
                from sglang.srt.disaggregation.decode import DecodeReqToTokenPool

                # subscribe memory for pre-allocated requests
                # if max_num_reqs <= 32, we pre-allocate 2x requests
                pre_alloc_size = max_num_reqs * 2 if max_num_reqs <= 32 else 0
                self.req_to_token_pool = DecodeReqToTokenPool(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len + 4,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    pre_alloc_size=pre_alloc_size,
                )
            else:
                self.req_to_token_pool = ReqToTokenPool(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len + 4,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                )
        else:
            # Draft worker shares req_to_token_pool with the target worker.
            assert self.is_draft_worker

        if self.use_mla_backend:
            self.token_to_kv_pool = MLATokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=(
                    self.model_config.num_hidden_layers
                    if not self.is_draft_worker
                    else self.model_config.hf_config.num_nextn_predict_layers
                ),  # PP is not compatible with mla backend
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
            self.token_to_kv_pool = MHATokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )

        if self.token_to_kv_pool_allocator is None:
            if self.page_size == 1:
                self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                    self.max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                )
            else:
                if _is_hpu:
                    self.token_to_kv_pool_allocator = HeapPagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                    )
                else:
                    self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
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
        if self.server_args.enable_two_batch_overlap:
            self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
        else:
            self.attn_backend = self._get_attention_backend()

    # TODO unify with 6338
    def _get_attention_backend(self):
        if self.server_args.attention_backend == "flashinfer":
            if not self.use_mla_backend:
                from sglang.srt.layers.attention.flashinfer_backend import (
                    FlashInferAttnBackend,
                )

                # Init streams
                if self.server_args.speculative_algorithm == "EAGLE":
                    self.plan_stream_for_flashinfer = torch.cuda.Stream()
                return FlashInferAttnBackend(self)
            else:
                from sglang.srt.layers.attention.flashinfer_mla_backend import (
                    FlashInferMLAAttnBackend,
                )

                return FlashInferMLAAttnBackend(self)
        elif self.server_args.attention_backend == "aiter":
            from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

            return AiterAttnBackend(self)
        elif self.server_args.attention_backend == "triton":
            assert not self.model_config.is_encoder_decoder, (
                "Cross attention is not supported in the triton attention backend. "
                "Please use `--attention-backend flashinfer`."
            )
            if self.server_args.enable_double_sparsity:
                from sglang.srt.layers.attention.double_sparsity_backend import (
                    DoubleSparseAttnBackend,
                )

                return DoubleSparseAttnBackend(self)
            else:
                from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

                return TritonAttnBackend(self)
        elif self.server_args.attention_backend == "torch_native":
            from sglang.srt.layers.attention.torch_native_backend import (
                TorchNativeAttnBackend,
            )

            return TorchNativeAttnBackend(self)
        elif self.server_args.attention_backend == "hpu_attn_backend":
            from sglang.srt.layers.attention.hpu_attn_backend import HPUAttnBackend

            return HPUAttnBackend(self)
        elif self.server_args.attention_backend == "flashmla":
            from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

            return FlashMLABackend(self)
        elif self.server_args.attention_backend == "fa3":
            assert (
                torch.cuda.get_device_capability()[0] == 8 and not self.use_mla_backend
            ) or torch.cuda.get_device_capability()[0] == 9, (
                "FlashAttention v3 Backend requires SM>=80 and SM<=90. "
                "Please use `--attention-backend flashinfer`."
            )
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )

            return FlashAttentionBackend(self)
        elif self.server_args.attention_backend == "cutlass_mla":
            from sglang.srt.layers.attention.cutlass_mla_backend import (
                CutlassMLABackend,
            )

            return CutlassMLABackend(self)
        elif self.server_args.attention_backend == "intel_amx":
            from sglang.srt.layers.attention.intel_amx_backend import (
                IntelAMXAttnBackend,
            )

            logger.info(f"Intel AMX attention backend is enabled.")
            return IntelAMXAttnBackend(self)
        else:
            raise ValueError(
                f"Invalid attention backend: {self.server_args.attention_backend}"
            )

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

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.disable_cuda_graph:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.cuda_graph_runner = CudaGraphRunner(self)
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def apply_torch_tp(self):
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def forward_decode(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> LogitsProcessorOutput:
        self.attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        return self.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch, **kwargs
        )

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            self.attn_backend.init_forward_metadata(forward_batch)

        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if not self.is_generation:
            kwargs["get_embedding"] = True
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

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Tuple[Union[LogitsProcessorOutput, PPProxyTensors], bool]:
        self.forward_pass_id += 1

        with get_global_expert_distribution_recorder().with_forward_pass(
            self.forward_pass_id,
            forward_batch,
        ):
            output = self._forward_raw(
                forward_batch, skip_attn_backend_init, pp_proxy_tensors
            )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        return output

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[PPProxyTensors],
    ) -> Tuple[Union[LogitsProcessorOutput, PPProxyTensors], bool]:
        can_run_cuda_graph = bool(
            forward_batch.forward_mode.is_cuda_graph()
            and self.cuda_graph_runner
            and self.cuda_graph_runner.can_run(forward_batch)
        )
        if can_run_cuda_graph:
            ret = self.cuda_graph_runner.replay(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode.is_decode():
            ret = self.forward_decode(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
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

        return ret, can_run_cuda_graph

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # Apply logit bias
        if sampling_info.sampling_info_done:
            # Overlap mode: the function update_regex_vocab_mask was executed
            # in process_batch_result of the last batch.
            if sampling_info.grammars:
                sampling_info.sampling_info_done.wait()
        else:
            # Normal mode: Put CPU-heavy tasks here. They will be overlapped with the forward pass.
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
        )
        return next_token_ids

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


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank):
    if isinstance(tensor, LocalSerializedTensor):
        monkey_patch_torch_reductions()
        tensor = tensor.get(tp_rank)
    return tensor.to(torch.cuda.current_device())


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
