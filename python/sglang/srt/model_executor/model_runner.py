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

import importlib
import importlib.resources
import logging
import pkgutil
import warnings
from functools import lru_cache
from typing import Optional, Type

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
)
from vllm.model_executor.models import ModelRegistry

from sglang.global_config import global_config
from sglang.srt.managers.schedule_batch import (
    Batch,
    ForwardMode,
    InputMetadata,
    global_server_args_dict,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPool
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    get_available_gpu_memory,
    is_llama3_405b_fp8,
    is_multimodal_model,
    monkey_patch_vllm_dummy_weight_loader,
    monkey_patch_vllm_p2p_access_check,
    monkey_patch_vllm_qvk_linear_loader,
)

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(
        self,
        model_config,
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
        self.is_multimodal_model = is_multimodal_model(self.model_config)
        global_server_args_dict.update(
            {
                "disable_flashinfer": server_args.disable_flashinfer,
                "disable_flashinfer_sampling": server_args.disable_flashinfer_sampling,
                "attention_reduce_in_fp32": server_args.attention_reduce_in_fp32,
            }
        )

        # Init torch distributed
        torch.cuda.set_device(self.gpu_id)
        logger.info(f"[gpu={self.gpu_id}] Init nccl begin.")

        if not server_args.enable_p2p_check:
            monkey_patch_vllm_p2p_access_check(self.gpu_id)

        if server_args.nccl_init_addr:
            nccl_init_method = f"tcp://{server_args.nccl_init_addr}"
        else:
            nccl_init_method = f"tcp://127.0.0.1:{self.nccl_port}"
        init_distributed_environment(
            backend="nccl",
            world_size=self.tp_size,
            rank=self.tp_rank,
            local_rank=self.gpu_id,
            distributed_init_method=nccl_init_method,
        )
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        self.tp_group = get_tp_group()
        total_gpu_memory = get_available_gpu_memory(
            self.gpu_id, distributed=self.tp_size > 1
        )

        if self.tp_size > 1:
            total_local_gpu_memory = get_available_gpu_memory(self.gpu_id)
            if total_local_gpu_memory < total_gpu_memory * 0.9:
                raise ValueError(
                    "The memory capacity is unbalanced. Some GPUs may be occupied by other processes."
                )

        # Load the model and create memory pool
        self.load_model()
        self.init_memory_pool(
            total_gpu_memory,
            server_args.max_num_reqs,
            server_args.max_total_tokens,
        )
        self.init_cublas()
        self.init_flash_infer()

        # Capture cuda graphs
        self.init_cuda_graphs()

    def load_model(self):
        logger.info(
            f"[gpu={self.gpu_id}] Load weight begin. "
            f"avail mem={get_available_gpu_memory(self.gpu_id):.2f} GB"
        )

        monkey_patch_vllm_dummy_weight_loader()
        device_config = DeviceConfig()
        load_config = LoadConfig(load_format=self.server_args.load_format)
        vllm_model_config = VllmModelConfig(
            model=self.server_args.model_path,
            quantization=self.server_args.quantization,
            tokenizer=None,
            tokenizer_mode=None,
            trust_remote_code=self.server_args.trust_remote_code,
            dtype=self.server_args.dtype,
            seed=42,
            skip_tokenizer_init=True,
        )

        if is_llama3_405b_fp8(self.model_config) and self.tp_size <= 8:
            # A temporary hack to fix the num_heads for meta-llama/Meta-Llama-3.1-405B-FP8 checkpoints
            self.model_config.hf_config.num_key_value_heads = 8
            vllm_model_config.hf_config.num_key_value_heads = 8
            monkey_patch_vllm_qvk_linear_loader()

        self.dtype = vllm_model_config.dtype
        if self.model_config.model_overide_args is not None:
            vllm_model_config.hf_config.update(self.model_config.model_overide_args)

        if (
            self.server_args.efficient_weight_load
            and "llama" in self.server_args.model_path.lower()
            and self.server_args.quantization == "fp8"
        ):
            from sglang.srt.model_loader.model_loader import get_model
        else:
            from vllm.model_executor.model_loader import get_model

        self.model = get_model(
            model_config=vllm_model_config,
            device_config=device_config,
            load_config=load_config,
            lora_config=None,
            multimodal_config=None,
            parallel_config=None,
            scheduler_config=None,
            cache_config=None,
        )
        logger.info(
            f"[gpu={self.gpu_id}] Load weight end. "
            f"type={type(self.model).__name__}, "
            f"dtype={self.dtype}, "
            f"avail mem={get_available_gpu_memory(self.gpu_id):.2f} GB"
        )

    def profile_max_num_token(self, total_gpu_memory):
        available_gpu_memory = get_available_gpu_memory(
            self.gpu_id, distributed=self.tp_size > 1
        )
        head_dim = self.model_config.head_dim
        head_num = self.model_config.get_num_kv_heads(self.tp_size)
        cell_size = (
            head_num
            * head_dim
            * self.model_config.num_hidden_layers
            * 2
            * torch._utils._element_size(self.dtype)
        )
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - self.mem_fraction_static
        )
        max_num_token = int(rest_memory * (1 << 30) // cell_size)
        return max_num_token

    def init_memory_pool(
        self, total_gpu_memory, max_num_reqs=None, max_total_tokens=None
    ):
        self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)
        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                warnings.warn(
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
        self.token_to_kv_pool = TokenToKVPool(
            self.max_total_num_tokens,
            dtype=self.dtype,
            head_num=self.model_config.get_num_kv_heads(self.tp_size),
            head_dim=self.model_config.head_dim,
            layer_num=self.model_config.num_hidden_layers,
        )
        logger.info(
            f"[gpu={self.gpu_id}] Memory pool end. "
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

    def init_flash_infer(self):
        if self.server_args.disable_flashinfer:
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

        self.flashinfer_workspace_buffers = torch.empty(
            2, global_config.flashinfer_workspace_size, dtype=torch.uint8, device="cuda"
        )
        self.flashinfer_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.flashinfer_workspace_buffers[0], "NHD"
        )
        self.flashinfer_prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
            self.flashinfer_workspace_buffers[1], "NHD"
        )
        self.flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.flashinfer_workspace_buffers[0],
            "NHD",
            use_tensor_cores=use_tensor_cores,
        )

    def init_cuda_graphs(self):
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        if self.server_args.disable_cuda_graph or self.server_args.disable_flashinfer:
            self.cuda_graph_runner = None
            return

        logger.info(
            f"[gpu={self.gpu_id}] Capture cuda graph begin. This can take up to several minutes."
        )
        batch_size_list = [1, 2, 4] + [i * 8 for i in range(1, 17)]
        self.cuda_graph_runner = CudaGraphRunner(
            self,
            max_batch_size_to_capture=max(batch_size_list),
            use_torch_compile=self.server_args.enable_torch_compile,
        )
        try:
            self.cuda_graph_runner.capture(batch_size_list)
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}. Possible solutions:\n"
                f"1. disable cuda graph by --disable-cuda-graph\n"
                f"2. set --mem-fraction-static to a smaller value\n"
                f"Open an issue on GitHub with reproducible scripts if you need help.\n"
            )

    @torch.inference_mode()
    def forward_decode(self, batch: Batch):
        if self.cuda_graph_runner and self.cuda_graph_runner.can_run(len(batch.reqs)):
            return self.cuda_graph_runner.replay(batch)

        input_metadata = InputMetadata.create(
            self,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            prefix_lens=batch.prefix_lens,
            position_ids_offsets=batch.position_ids_offsets,
            out_cache_loc=batch.out_cache_loc,
            top_logprobs_nums=batch.top_logprobs_nums,
            return_logprob=batch.return_logprob,
        )
        return self.model.forward(
            batch.input_ids, input_metadata.positions, input_metadata
        )

    @torch.inference_mode()
    def forward_extend(self, batch: Batch):
        input_metadata = InputMetadata.create(
            self,
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            prefix_lens=batch.prefix_lens,
            position_ids_offsets=batch.position_ids_offsets,
            out_cache_loc=batch.out_cache_loc,
            top_logprobs_nums=batch.top_logprobs_nums,
            return_logprob=batch.return_logprob,
        )
        return self.model.forward(
            batch.input_ids, input_metadata.positions, input_metadata
        )

    @torch.inference_mode()
    def forward_extend_multi_modal(self, batch: Batch):
        input_metadata = InputMetadata.create(
            self,
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            prefix_lens=batch.prefix_lens,
            position_ids_offsets=batch.position_ids_offsets,
            out_cache_loc=batch.out_cache_loc,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
        )
        return self.model.forward(
            batch.input_ids,
            input_metadata.positions,
            input_metadata,
            batch.pixel_values,
            batch.image_sizes,
            batch.image_offsets,
        )

    def forward(self, batch: Batch, forward_mode: ForwardMode):
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
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    model_arch_name_to_cls[entry.__name__] = entry

            # compat: some models such as chatglm has incorrect class set in config.json
            # usage: [ tuple("From_Entry_Class_Name": EntryClass), ]
            if hasattr(module, "EntryClassRemapping") and isinstance(
                module.EntryClassRemapping, list
            ):
                for remap in module.EntryClassRemapping:
                    if isinstance(remap, tuple) and len(remap) == 2:
                        model_arch_name_to_cls[remap[0]] = remap[1]

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
setattr(ModelRegistry, "load_model_cls", load_model_cls_srt)
