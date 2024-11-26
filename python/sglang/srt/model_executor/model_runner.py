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

import gc
import importlib
import importlib.resources
import inspect
import json
import logging
import pkgutil
from functools import lru_cache
from typing import Optional, Type

import torch
import torch.nn as nn
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

from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.layers.attention.double_sparsity_backend import DoubleSparseAttnBackend
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.mem_cache.memory_pool import (
    DoubleSparseTokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    crash_on_warnings,
    enable_show_time_cost,
    get_available_gpu_memory,
    is_hip,
    monkey_patch_vllm_model_config,
    monkey_patch_vllm_p2p_access_check,
    set_cpu_offload_max_bytes,
)

logger = logging.getLogger(__name__)


class ModelRunner:
    """ModelRunner runs the forward passes of the models."""

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
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_generation = model_config.is_generation
        self.is_multimodal = model_config.is_multimodal

        # Model-specific adjustment
        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and not self.server_args.disable_mla
        ):
            logger.info("MLA optimization is turned on. Use triton backend.")
            self.server_args.attention_backend = "triton"

        if self.server_args.enable_double_sparsity:
            logger.info(
                "Double sparsity optimization is turned on. Use triton backend without CUDA graph."
            )
            self.server_args.attention_backend = "triton"
            self.server_args.disable_cuda_graph = True
            if self.server_args.ds_heavy_channel_type is None:
                raise ValueError(
                    "Please specify the heavy channel type for double sparsity optimization."
                )
            self.init_double_sparsity_channel_config(
                self.server_args.ds_heavy_channel_type
            )

        if self.is_multimodal:
            logger.info(
                "Automatically turn off --chunked-prefill-size and adjust --mem-fraction-static for multimodal models."
            )
            server_args.chunked_prefill_size = None
            self.mem_fraction_static *= 0.95
            # TODO: qwen2-vl does not support radix cache now, set disable_radix_cache=True automatically
            if self.model_config.hf_config.architectures == [
                "Qwen2VLForConditionalGeneration"
            ]:
                server_args.disable_radix_cache = True

        # Global vars
        if server_args.show_time_cost:
            enable_show_time_cost()
        if server_args.disable_disk_cache:
            from outlines.caching import disable_cache

            disable_cache()

        global_server_args_dict.update(
            {
                "attention_backend": server_args.attention_backend,
                "sampling_backend": server_args.sampling_backend,
                "triton_attention_reduce_in_fp32": server_args.triton_attention_reduce_in_fp32,
                "disable_mla": server_args.disable_mla,
                "torchao_config": server_args.torchao_config,
                "enable_nan_detection": server_args.enable_nan_detection,
                "enable_dp_attention": server_args.enable_dp_attention,
            }
        )

        set_cpu_offload_max_bytes(int(server_args.cpu_offload_gb * 1024**3))

        # Init components
        min_per_gpu_memory = self.init_torch_distributed()
        self.sampler = Sampler()
        self.load_model()

        # Apply torch TP if model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()
            self.torch_tp_applied = True
        else:
            self.torch_tp_applied = False

        if server_args.lora_paths is not None:
            self.init_lora_manager()
        self.init_memory_pool(
            min_per_gpu_memory,
            server_args.max_running_requests,
            server_args.max_total_tokens,
        )
        if self.device == "cuda":
            self.init_cublas()
            self.init_attention_backend()
            self.init_cuda_graphs()
        else:
            self.cuda_graph_runner = None
            self.init_attention_backend()

    def init_torch_distributed(self):
        logger.info("Init torch distributed begin.")
        # Init torch distributed
        torch.get_device_module(self.device).set_device(self.gpu_id)
        if self.device == "cuda":
            backend = "nccl"
        # ToDO(liangan1):Just use gloo to bypass the initilization fail
        # Need to use xccl for xpu backend in the future
        elif self.device == "xpu":
            backend = "gloo"
        elif self.device == "hpu":
            backend = "hccl"

        if not self.server_args.enable_p2p_check:
            monkey_patch_vllm_p2p_access_check(self.gpu_id)
        if self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        init_distributed_environment(
            backend=backend,
            world_size=self.tp_size,
            rank=self.tp_rank,
            local_rank=self.gpu_id,
            distributed_init_method=dist_init_method,
        )
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        min_per_gpu_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=self.tp_size > 1
        )
        self.tp_group = get_tp_group()

        # Currently, there is a bug with mulit-node tensor parallelsim + padded cuda graph,
        # so we disable padding in cuda graph.
        if self.device == "cuda" and not all(
            in_the_same_node_as(self.tp_group.cpu_group, source_rank=0)
        ):
            self.server_args.disable_cuda_graph_padding = True
            logger.info(
                "Setting disable_cuda_graph_padding to True because of multi-node tensor parallelism."
            )

        # Check memory for tensor parallelism
        if self.tp_size > 1:
            local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                raise ValueError(
                    "The memory capacity is unbalanced. Some GPUs may be occupied by other processes."
                )

        return min_per_gpu_memory

    def setup_model(self):
        try:
            from vllm.config import VllmConfig

            vllm_config = VllmConfig()
            vllm_config.model_config = self.vllm_model_config
            vllm_config.load_config = self.load_config
            vllm_config.device_config = DeviceConfig(self.device)
            vllm_config.quant_config = VllmConfig._get_quantization_config(
                vllm_config.model_config, vllm_config.load_config
            )
            return get_model(vllm_config=vllm_config)
        except ImportError:
            pass

        return get_model(
            model_config=self.vllm_model_config,
            load_config=self.load_config,
            device_config=DeviceConfig(self.device),
            parallel_config=None,
            scheduler_config=None,
            lora_config=None,
            cache_config=None,
        )

    def get_model_config_params(self):
        sig = inspect.signature(VllmModelConfig.__init__)
        params = {
            "model": self.server_args.model_path,
            "quantization": self.server_args.quantization,
            "tokenizer": None,
            "tokenizer_mode": None,
            "trust_remote_code": self.server_args.trust_remote_code,
            "dtype": self.server_args.dtype,
            "seed": self.server_args.random_seed,
            "skip_tokenizer_init": True,
        }

        if "task" in sig.parameters:
            params["task"] = ""

        return params

    def load_model(self):
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        # Prepare the vllm model config
        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
        )
        monkey_patch_vllm_model_config()
        self.vllm_model_config = VllmModelConfig(**self.get_model_config_params())
        if self.model_config.model_override_args is not None:
            self.vllm_model_config.hf_config.update(
                self.model_config.model_override_args
            )

        self.model = self.setup_model()

        self.sliding_window_size = (
            self.model.get_attention_sliding_window_size()
            if hasattr(self.model, "get_attention_sliding_window_size")
            else None
        )
        self.dtype = self.vllm_model_config.dtype

        logger.info(
            f"Load weight end. "
            f"type={type(self.model).__name__}, "
            f"dtype={self.dtype}, "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
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
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        target_device = torch.device(self.device)

        try:
            model_config_params = self.get_model_config_params()
            model_config_params["model"] = model_path
            vllm_model_config = VllmModelConfig(**model_config_params)
        except Exception as e:
            message = f"Failed to load model config: {e}."
            return False, message

        load_config = LoadConfig(load_format=load_format)

        # Only support vllm DefaultModelLoader for now
        loader = get_model_loader(load_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source(
                    config.model,
                    revision=config.revision,
                    fall_back_to_pt=getattr(
                        self.model, "fall_back_to_pt_during_load", True
                    ),
                )
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
        return True, "Succeeded to update model weights."

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            lora_paths=self.server_args.lora_paths,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
        )
        logger.info("LoRA manager ready.")

    def profile_max_num_token(self, total_gpu_memory: int):
        available_gpu_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=self.tp_size > 1
        )
        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and not self.server_args.disable_mla
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
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if is_hip():  # Using natively supported format
                self.kv_cache_dtype = torch.float8_e5m2fnuz
            else:
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
                4096,
            )

        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs + 1,
            max_context_len=self.model_config.context_len + 4,
            device=self.device,
            use_records=False,
        )
        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and not self.server_args.disable_mla
        ):
            self.token_to_kv_pool = MLATokenToKVPool(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
            )
        elif self.server_args.enable_double_sparsity:
            self.token_to_kv_pool = DoubleSparseTokenToKVPool(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(self.tp_size),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                heavy_channel_num=self.server_args.ds_heavy_channel_num,
            )
        else:
            self.token_to_kv_pool = MHATokenToKVPool(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(self.tp_size),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
            )
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
        if self.server_args.attention_backend == "flashinfer":
            self.attn_backend = FlashInferAttnBackend(self)
        elif self.server_args.attention_backend == "triton":
            assert self.sliding_window_size is None, (
                "Window attention is not supported in the triton attention backend. "
                "Please use `--attention-backend flashinfer`."
            )
            assert not self.model_config.is_encoder_decoder, (
                "Cross attention is not supported in the triton attention backend. "
                "Please use `--attention-backend flashinfer`."
            )
            if self.server_args.enable_double_sparsity:
                self.attn_backend = DoubleSparseAttnBackend(self)
            else:
                self.attn_backend = TritonAttnBackend(self)
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

        for i in range(self.model_config.num_hidden_layers):
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
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        self.cuda_graph_runner = None

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.disable_cuda_graph:
            return

        logger.info("Capture cuda graph begin. This can take up to several minutes.")
        self.cuda_graph_runner = CudaGraphRunner(self)

    def apply_torch_tp(self):
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def forward_decode(self, forward_batch: ForwardBatch):
        if self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch):
            return self.cuda_graph_runner.replay(forward_batch)

        forward_batch.positions = (forward_batch.seq_lens - 1).to(torch.int64)
        self.attn_backend.init_forward_metadata(forward_batch)
        return self.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

    def forward_extend(self, forward_batch: ForwardBatch):
        self.attn_backend.init_forward_metadata(forward_batch)
        if self.is_generation:
            if forward_batch.input_embeds is None:
                return self.model.forward(
                    forward_batch.input_ids, forward_batch.positions, forward_batch
                )
            else:
                return self.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    input_embeds=forward_batch.input_embeds.bfloat16(),
                )
        else:
            # Only embedding models have get_embedding parameter
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                get_embedding=True,
            )

    def forward_idle(self, forward_batch: ForwardBatch):
        if self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch):
            return self.cuda_graph_runner.replay(forward_batch)

        return self.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

    def forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(forward_batch)
        elif forward_batch.forward_mode.is_extend():
            return self.forward_extend(forward_batch)
        elif forward_batch.forward_mode.is_idle():
            return self.forward_idle(forward_batch)
        else:
            raise ValueError(f"Invaid forward mode: {forward_batch.forward_mode}")

    def sample(
        self, logits_output: LogitsProcessorOutput, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        sampling_info = forward_batch.sampling_info
        if sampling_info.sampling_info_done:
            # Overlap mode: the function update_regex_vocab_mask was executed
            # in process_batch_result of the last batch.
            if sampling_info.grammars:
                sampling_info.sampling_info_done.wait()
        else:
            # Normal mode: Put CPU-heavy tasks here. They will be overlapped with the forward pass.
            sampling_info.update_regex_vocab_mask()
            sampling_info.update_penalties()
        logits = self.apply_logits_bias(logits_output.next_token_logits, sampling_info)

        # Sample the next tokens.
        next_token_ids = self.sampler(logits, sampling_info)
        return next_token_ids

    def apply_logits_bias(self, logits: torch.Tensor, sampling_info: SamplingBatchInfo):
        # Apply logit_bias
        if sampling_info.logit_bias is not None:
            logits.add_(sampling_info.logit_bias)

        # min-token, presence, frequency
        if sampling_info.linear_penalties is not None:
            logits.add_(sampling_info.linear_penalties)

        # repetition
        if sampling_info.scaling_penalties is not None:
            logits = torch.where(
                logits > 0,
                logits / sampling_info.scaling_penalties,
                logits * sampling_info.scaling_penalties,
            )

        # Apply regex vocab_mask
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_mask(logits=logits, vocab_mask=sampling_info.vocab_mask)

        return logits

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        rope_scaling = getattr(self.model_config.hf_config, "rope_scaling", {})
        if rope_scaling is None:
            return False
        return rope_scaling.get("type", None) == "mrope"


@lru_cache()
def import_model_classes():
    model_arch_name_to_cls = {}
    package_name = "sglang.srt.models"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning(f"Ignore import error when loading {name}. {e}")
                if crash_on_warnings():
                    raise ValueError(f"Ignore import error when loading {name}. {e}")
                continue
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(
                    entry, list
                ):  # To support multiple model classes in one module
                    for tmp in entry:
                        assert (
                            tmp.__name__ not in model_arch_name_to_cls
                        ), f"Duplicated model implementation for {tmp.__name__}"
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    assert (
                        entry.__name__ not in model_arch_name_to_cls
                    ), f"Duplicated model implementation for {entry.__name__}"
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
setattr(ModelRegistry, "is_multimodal_model", lambda model_architectures: False)
setattr(ModelRegistry, "is_attention_free_model", lambda model_architectures: False)
setattr(ModelRegistry, "model_has_inner_state", lambda model_architectures: False)
setattr(ModelRegistry, "is_embedding_model", lambda model_architectures: False)
