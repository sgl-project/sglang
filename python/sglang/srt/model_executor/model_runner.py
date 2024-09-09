"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""ModelRunner runs the forward passes of the models."""

import gc
import importlib
import importlib.resources
import json
import logging
import pkgutil
from functools import lru_cache
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.decode import _grouped_size_compiled_for_decode_kernels
from vllm.config import DeviceConfig, LoadConfig
from vllm.config import ModelConfig as VllmModelConfig
from vllm.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import ModelRegistry

from sglang.global_config import global_config
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import SampleOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch, global_server_args_dict
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.model_config import AttentionArch, ModelConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode, InputMetadata
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    get_available_gpu_memory,
    is_generation_model,
    is_llama3_405b_fp8_head_16,
    is_multimodal_model,
    monkey_patch_vllm_dummy_weight_loader,
    monkey_patch_vllm_p2p_access_check,
    monkey_patch_vllm_qvk_linear_loader,
)

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
    ):
        # Parse args
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.nccl_port = nccl_port
        self.server_args = server_args
        self.is_multimodal_model = is_multimodal_model(
            self.model_config.hf_config.architectures
        )
        global_server_args_dict.update(
            {
                "disable_flashinfer": server_args.disable_flashinfer,
                "disable_flashinfer_sampling": server_args.disable_flashinfer_sampling,
                "triton_attention_reduce_in_fp32": server_args.triton_attention_reduce_in_fp32,
                "enable_mla": server_args.enable_mla,
                "torchao_config": server_args.torchao_config,
            }
        )

        if self.is_multimodal_model:
            logger.info(
                "Automatically turn off --chunked-prefill-size and adjust --mem-fraction-static for multimodal models."
            )
            server_args.chunked_prefill_size = None
            server_args.mem_fraction_static *= 0.95

        min_per_gpu_memory = self.init_torch_distributed()
        self.load_model()
        self.init_memory_pool(
            min_per_gpu_memory,
            server_args.max_num_reqs,
            server_args.max_total_tokens,
        )
        self.init_cublas()
        self.init_flashinfer()
        self.init_cuda_graphs()

    def init_torch_distributed(self):
        # Init torch distributed
        torch.cuda.set_device(self.gpu_id)
        logger.info("Init nccl begin.")

        if not self.server_args.enable_p2p_check:
            monkey_patch_vllm_p2p_access_check(self.gpu_id)

        if self.server_args.nccl_init_addr:
            nccl_init_method = f"tcp://{self.server_args.nccl_init_addr}"
        else:
            nccl_init_method = f"tcp://127.0.0.1:{self.nccl_port}"
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        init_distributed_environment(
            backend="nccl",
            world_size=self.tp_size,
            rank=self.tp_rank,
            local_rank=self.gpu_id,
            distributed_init_method=nccl_init_method,
        )
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        min_per_gpu_memory = get_available_gpu_memory(
            self.gpu_id, distributed=self.tp_size > 1
        )
        self.tp_group = get_tp_group()

        # Currently, there is a bug with mulit-node tensor parallelsim + padded cuda graph,
        # so we disable padding in cuda graph.
        if not all(in_the_same_node_as(self.tp_group.cpu_group, source_rank=0)):
            self.server_args.disable_cuda_graph_padding = True
            logger.info(
                "Setting disable_cuda_graph_padding to True because of multi-node tensor parallelism."
            )

        # Check memory for tensor parallelism
        if self.tp_size > 1:
            local_gpu_memory = get_available_gpu_memory(self.gpu_id)
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                raise ValueError(
                    "The memory capacity is unbalanced. Some GPUs may be occupied by other processes."
                )

        return min_per_gpu_memory

    def load_model(self):
        torch.set_num_threads(1)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.gpu_id):.2f} GB"
        )
        if torch.cuda.get_device_capability()[0] < 8:
            logger.info(
                "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
            )
            self.server_args.dtype = "float16"
            if torch.cuda.get_device_capability()[1] < 5:
                raise RuntimeError("SGLang only supports sm75 and above.")

        monkey_patch_vllm_dummy_weight_loader()
        self.device_config = DeviceConfig()
        self.load_config = LoadConfig(load_format=self.server_args.load_format)
        self.vllm_model_config = VllmModelConfig(
            model=self.server_args.model_path,
            quantization=self.server_args.quantization,
            tokenizer=None,
            tokenizer_mode=None,
            trust_remote_code=self.server_args.trust_remote_code,
            dtype=self.server_args.dtype,
            seed=42,
            skip_tokenizer_init=True,
        )

        # A temporary hack to fix the num_heads for meta-llama/Meta-Llama-3.1-405B-FP8 checkpoints
        # Drop this after Sept, 2024.
        if is_llama3_405b_fp8_head_16(self.model_config) and self.tp_size <= 8:
            self.model_config.hf_config.num_key_value_heads = 8
            self.vllm_model_config.hf_config.num_key_value_heads = 8
            monkey_patch_vllm_qvk_linear_loader()

        self.dtype = self.vllm_model_config.dtype
        if self.model_config.model_override_args is not None:
            self.vllm_model_config.hf_config.update(
                self.model_config.model_override_args
            )

        self.model = get_model(
            model_config=self.vllm_model_config,
            load_config=self.load_config,
            device_config=self.device_config,
            parallel_config=None,
            scheduler_config=None,
            lora_config=None,
            cache_config=None,
        )
        self.sliding_window_size = (
            self.model.get_attention_sliding_window_size()
            if hasattr(self.model, "get_attention_sliding_window_size")
            else None
        )
        self.is_generation = is_generation_model(
            self.model_config.hf_config.architectures, self.server_args.is_embedding
        )

        logger.info(
            f"Load weight end. "
            f"type={type(self.model).__name__}, "
            f"dtype={self.dtype}, "
            f"avail mem={get_available_gpu_memory(self.gpu_id):.2f} GB"
        )

    def update_weights(self, model_path: str, load_format: str):
        """Update weights in-place."""
        from vllm.model_executor.model_loader.loader import (
            DefaultModelLoader,
            device_loading_context,
            get_model_loader,
        )
        from vllm.model_executor.model_loader.utils import set_default_torch_dtype

        logger.info(
            f"Update weights begin. "
            f"avail mem={get_available_gpu_memory(self.gpu_id):.2f} GB"
        )

        target_device = torch.device(self.device_config.device)

        try:
            # TODO: Use a better method to check this
            vllm_model_config = VllmModelConfig(
                model=model_path,
                quantization=self.server_args.quantization,
                tokenizer=None,
                tokenizer_mode=None,
                trust_remote_code=self.server_args.trust_remote_code,
                dtype=self.server_args.dtype,
                seed=42,
                skip_tokenizer_init=True,
            )
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            return False, "Failed to update model weights"

        load_config = LoadConfig(load_format=load_format)

        # Only support vllm DefaultModelLoader for now
        loader = get_model_loader(load_config)
        if not isinstance(loader, DefaultModelLoader):
            logger.error("Failed to get weights iterator: Unsupported loader")
            return False, "Failed to update model weights"

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                config.model,
                config.revision,
                fall_back_to_pt=getattr(
                    self.model, "fall_back_to_pt_during_load", True
                ),
            )
            return iter

        def model_load_weights(model, iter):
            model.load_weights(iter)
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
            return model

        with set_default_torch_dtype(vllm_model_config.dtype):
            try:
                iter = get_weight_iter(vllm_model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}"
                logger.error(message)
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = f"Failed to update weights: {e}. \n Rolling back to original weights"
                logger.error(message)
                del iter
                gc.collect()
                iter = get_weight_iter(self.vllm_model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.vllm_model_config = vllm_model_config
        self.load_config = load_config
        self.model_config.path = model_path

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights"

    def profile_max_num_token(self, total_gpu_memory: int):
        available_gpu_memory = get_available_gpu_memory(
            self.gpu_id, distributed=self.tp_size > 1
        )
        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and self.server_args.enable_mla
        ):
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * self.model_config.num_hidden_layers
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        else:
            cell_size = (
                self.model_config.get_num_kv_heads(self.tp_size)
                * self.model_config.head_dim
                * self.model_config.num_hidden_layers
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
        max_num_reqs: int = None,
        max_total_tokens: int = None,
    ):
        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            self.kv_cache_dtype = torch.float8_e5m2
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

        self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)
        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logging.warning(
                    f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                    f"{self.max_total_num_tokens}. "
                    f"Use the profiled value instead."
                )
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)

        if self.max_total_num_tokens <= 0:
            raise RuntimeError(
                "Not enough memory. Please try to increase --mem-fraction-static."
            )

        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(
                        self.max_total_num_tokens / self.model_config.context_len * 512
                    ),
                    2048,
                ),
                5120,
            )

        self.req_to_token_pool = ReqToTokenPool(
            max_num_reqs,
            self.model_config.context_len + 8,
        )
        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and self.server_args.enable_mla
        ):
            self.token_to_kv_pool = MLATokenToKVPool(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.model_config.num_hidden_layers,
            )
            logger.info("using MLA Triton implementaion, flashinfer is disabled")
            # FIXME: temporarily only Triton MLA is supported
            self.server_args.disable_flashinfer = True
        else:
            self.token_to_kv_pool = MHATokenToKVPool(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(self.tp_size),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
            )
        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.gpu_id):.2f} GB"
        )

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_flashinfer(self):
        """Init flashinfer attention kernel wrappers."""
        if self.server_args.disable_flashinfer:
            assert (
                self.sliding_window_size is None
            ), "turn on flashinfer to support window attention"
            self.flashinfer_prefill_wrapper_ragged = None
            self.flashinfer_prefill_wrapper_paged = None
            self.flashinfer_decode_wrapper = None
            return

        if not _grouped_size_compiled_for_decode_kernels(
            self.model_config.num_attention_heads // self.tp_size,
            self.model_config.get_num_kv_heads(self.tp_size),
        ):
            use_tensor_cores = True
        else:
            use_tensor_cores = False

        if self.sliding_window_size is None:
            self.flashinfer_workspace_buffer = torch.empty(
                global_config.flashinfer_workspace_size,
                dtype=torch.uint8,
                device="cuda",
            )
            self.flashinfer_prefill_wrapper_ragged = (
                BatchPrefillWithRaggedKVCacheWrapper(
                    self.flashinfer_workspace_buffer, "NHD"
                )
            )
            self.flashinfer_prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
                self.flashinfer_workspace_buffer, "NHD"
            )
            self.flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.flashinfer_workspace_buffer,
                "NHD",
                use_tensor_cores=use_tensor_cores,
            )
        else:
            self.flashinfer_workspace_buffer = torch.empty(
                global_config.flashinfer_workspace_size,
                dtype=torch.uint8,
                device="cuda",
            )
            self.flashinfer_prefill_wrapper_ragged = None
            self.flashinfer_prefill_wrapper_paged = []
            self.flashinfer_decode_wrapper = []
            for i in range(2):
                self.flashinfer_prefill_wrapper_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.flashinfer_workspace_buffer, "NHD"
                    )
                )
                self.flashinfer_decode_wrapper.append(
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.flashinfer_workspace_buffer,
                        "NHD",
                        use_tensor_cores=use_tensor_cores,
                    )
                )

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        if self.server_args.disable_cuda_graph or self.server_args.disable_flashinfer:
            self.cuda_graph_runner = None
            return

        logger.info("Capture cuda graph begin. This can take up to several minutes.")

        if self.server_args.disable_cuda_graph_padding:
            batch_size_list = list(range(1, 32)) + [64, 128]
        else:
            batch_size_list = [1, 2, 4] + [i * 8 for i in range(1, 21)]

        self.cuda_graph_runner = CudaGraphRunner(
            self,
            max_batch_size_to_capture=max(batch_size_list),
            use_torch_compile=self.server_args.enable_torch_compile,
            disable_padding=self.server_args.disable_cuda_graph_padding,
        )
        try:
            self.cuda_graph_runner.capture(batch_size_list)
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n"
                "Possible solutions:\n"
                "1. disable cuda graph by --disable-cuda-graph\n"
                "2. set --mem-fraction-static to a smaller value\n"
                "3. disable torch compile by not using --enable-torch-compile\n"
                "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
            )

    @torch.inference_mode()
    def forward_decode(self, batch: ScheduleBatch):
        if (
            self.cuda_graph_runner
            and self.cuda_graph_runner.can_run(len(batch.reqs))
            and batch.sampling_info.can_run_in_cuda_graph()
        ):
            return self.cuda_graph_runner.replay(batch)

        input_metadata = InputMetadata.from_schedule_batch(
            self,
            batch,
            ForwardMode.DECODE,
        )

        return self.model.forward(
            batch.input_ids, input_metadata.positions, input_metadata
        )

    @torch.inference_mode()
    def forward_extend(self, batch: ScheduleBatch):
        input_metadata = InputMetadata.from_schedule_batch(
            self,
            batch,
            forward_mode=ForwardMode.EXTEND,
        )
        if self.is_generation:
            return self.model.forward(
                batch.input_ids, input_metadata.positions, input_metadata
            )
        else:
            # Only embedding models have get_embedding parameter
            return self.model.forward(
                batch.input_ids,
                input_metadata.positions,
                input_metadata,
                get_embedding=True,
            )

    @torch.inference_mode()
    def forward_extend_multi_modal(self, batch: ScheduleBatch):
        input_metadata = InputMetadata.from_schedule_batch(
            self,
            batch,
            forward_mode=ForwardMode.EXTEND,
        )
        return self.model.forward(
            batch.input_ids,
            input_metadata.positions,
            input_metadata,
            input_metadata.pixel_values,
            input_metadata.image_sizes,
            input_metadata.image_offsets,
        )

    def forward(
        self, batch: ScheduleBatch, forward_mode: ForwardMode
    ) -> Tuple[SampleOutput, LogitsProcessorOutput]:
        if self.is_multimodal_model and forward_mode == ForwardMode.EXTEND:
            return self.forward_extend_multi_modal(batch)
        elif forward_mode == ForwardMode.DECODE:
            return self.forward_decode(batch)
        elif forward_mode == ForwardMode.EXTEND:
            return self.forward_extend(batch)
        else:
            raise ValueError(f"Invaid forward mode: {forward_mode}")


@lru_cache()
def import_model_classes():
    model_arch_name_to_cls = {}
    package_name = "sglang.srt.models"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            module = importlib.import_module(name)
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(
                    entry, list
                ):  # To support multiple model classes in one module
                    for tmp in entry:
                        assert tmp.__name__ not in model_arch_name_to_cls
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    assert entry.__name__ not in model_arch_name_to_cls
                    model_arch_name_to_cls[entry.__name__] = entry

    return model_arch_name_to_cls


def load_model_cls_srt(model_arch: str) -> Optional[Type[nn.Module]]:
    model_arch_name_to_cls = import_model_classes()

    if model_arch not in model_arch_name_to_cls:
        raise ValueError(
            f"Unsupported architectures: {model_arch}. "
            f"Supported list: {list(model_arch_name_to_cls.keys())}"
        )
    return model_arch_name_to_cls[model_arch]


# Monkey patch model loader
setattr(ModelRegistry, "_try_load_model_cls", load_model_cls_srt)
