import importlib
import importlib.resources
import logging
import pkgutil
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from vllm.config import DeviceConfig, LoadConfig
from vllm.config import ModelConfig as VllmModelConfig
from vllm.distributed import initialize_model_parallel
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import ModelRegistry

from sglang.srt.managers.router.infer_batch import Batch, ForwardMode
from sglang.srt.memory_pool import ReqToTokenPool, TokenToKVPool
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_available_gpu_memory, is_multimodal_model


logger = logging.getLogger("model_runner")

# for server args in model endpoints
global_server_args_dict = {}


@dataclass
class InputMetadata:
    model_runner: "ModelRunner"
    forward_mode: ForwardMode
    batch_size: int
    total_num_tokens: int
    max_seq_len: int
    req_pool_indices: torch.Tensor
    start_loc: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    positions: torch.Tensor
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: TokenToKVPool

    # for extend
    extend_seq_lens: torch.Tensor = None
    extend_start_loc: torch.Tensor = None
    max_extend_len: int = 0

    out_cache_loc: torch.Tensor = None
    out_cache_cont_start: torch.Tensor = None
    out_cache_cont_end: torch.Tensor = None

    other_kv_index: torch.Tensor = None
    return_logprob: bool = False
    top_logprobs_nums: List[int] = None

    # for flashinfer
    qo_indptr: torch.Tensor = None
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None
    kv_last_page_len: torch.Tensor = None
    prefill_wrapper = None
    decode_wrapper = None

    def init_flashinfer_args(self, tp_size):
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.kv_indptr = torch.zeros(
            (self.batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        self.kv_indptr[1:] = torch.cumsum(self.seq_lens, dim=0)
        self.kv_last_page_len = torch.ones(
            (self.batch_size,), dtype=torch.int32, device="cuda"
        )
        req_pool_indices_cpu = self.req_pool_indices.cpu().numpy()
        seq_lens_cpu = self.seq_lens.cpu().numpy()
        self.kv_indices = torch.cat(
            [
                self.req_to_token_pool.req_to_token[
                    req_pool_indices_cpu[i], : seq_lens_cpu[i]
                ]
                for i in range(self.batch_size)
            ],
            dim=0,
        ).contiguous()

        workspace_buffer = torch.empty(
            32 * 1024 * 1024, dtype=torch.int8, device="cuda"
        )
        if (
            self.forward_mode == ForwardMode.PREFILL
            or self.forward_mode == ForwardMode.EXTEND
        ):
            self.qo_indptr = torch.zeros(
                (self.batch_size + 1,), dtype=torch.int32, device="cuda"
            )
            self.qo_indptr[1:] = torch.cumsum(self.extend_seq_lens, dim=0)
            self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer, "NHD"
            )
            args = [
                self.qo_indptr,
                self.kv_indptr,
                self.kv_indices,
                self.kv_last_page_len,
                self.model_runner.model_config.num_attention_heads // tp_size,
                self.model_runner.model_config.num_key_value_heads // tp_size,
                self.model_runner.model_config.head_dim,
            ]

            self.prefill_wrapper.begin_forward(*args)
        else:
            self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer, "NHD"
            )
            self.decode_wrapper.begin_forward(
                self.kv_indptr,
                self.kv_indices,
                self.kv_last_page_len,
                self.model_runner.model_config.num_attention_heads // tp_size,
                self.model_runner.model_config.num_key_value_heads // tp_size,
                self.model_runner.model_config.head_dim,
                1,
                "NONE",
                "float16",
            )

    def init_extend_args(self):
        self.extend_seq_lens = self.seq_lens - self.prefix_lens
        self.extend_start_loc = torch.zeros_like(self.seq_lens)
        self.extend_start_loc[1:] = torch.cumsum(self.extend_seq_lens[:-1], dim=0)
        self.max_extend_len = int(torch.max(self.extend_seq_lens))

    @classmethod
    def create(
        cls,
        model_runner,
        tp_size,
        forward_mode,
        req_pool_indices,
        seq_lens,
        prefix_lens,
        position_ids_offsets,
        out_cache_loc,
        out_cache_cont_start=None,
        out_cache_cont_end=None,
        top_logprobs_nums=None,
        return_logprob=False,
    ):
        batch_size = len(req_pool_indices)
        start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
        start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)
        total_num_tokens = int(torch.sum(seq_lens))
        max_seq_len = int(torch.max(seq_lens))

        if forward_mode == ForwardMode.DECODE:
            positions = ((seq_lens - 1) + position_ids_offsets).to(torch.int64)
            other_kv_index = model_runner.req_to_token_pool.req_to_token[
                req_pool_indices[0], seq_lens[0] - 1
            ].item()
        else:
            seq_lens_cpu = seq_lens.cpu().numpy()
            prefix_lens_cpu = prefix_lens.cpu().numpy()
            position_ids_offsets_cpu = position_ids_offsets.cpu().numpy()
            positions = torch.tensor(
                np.concatenate(
                    [
                        np.arange(
                            prefix_lens_cpu[i] + position_ids_offsets_cpu[i],
                            seq_lens_cpu[i] + position_ids_offsets_cpu[i],
                        )
                        for i in range(batch_size)
                    ],
                    axis=0,
                ),
                device="cuda",
            )
            other_kv_index = None

        ret = cls(
            model_runner=model_runner,
            forward_mode=forward_mode,
            batch_size=batch_size,
            total_num_tokens=total_num_tokens,
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            start_loc=start_loc,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            positions=positions,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            out_cache_cont_start=out_cache_cont_start,
            out_cache_cont_end=out_cache_cont_end,
            other_kv_index=other_kv_index,
            return_logprob=return_logprob,
            top_logprobs_nums=top_logprobs_nums,
        )

        if forward_mode == ForwardMode.EXTEND:
            ret.init_extend_args()

        if global_server_args_dict.get("enable_flashinfer", False):
            ret.init_flashinfer_args(tp_size)

        return ret


class ModelRunner:
    def __init__(
        self,
        model_config,
        mem_fraction_static,
        tp_rank,
        tp_size,
        nccl_port,
        server_args: ServerArgs,
    ):
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.nccl_port = nccl_port
        self.server_args = server_args

        global global_server_args_dict
        global_server_args_dict = {
            "enable_flashinfer": server_args.enable_flashinfer,
            "attention_reduce_in_fp32": server_args.attention_reduce_in_fp32,
        }

        # Init torch distributed
        logger.debug("Init torch begin.")
        torch.cuda.set_device(self.tp_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=self.tp_size,
            rank=self.tp_rank,
            init_method=f"tcp://127.0.0.1:{self.nccl_port}",
        )
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        logger.debug("Init torch end.")

        total_gpu_memory = get_available_gpu_memory(
            self.tp_rank, distributed=self.tp_size > 1
        ) * (1 << 30)
        # logger.info(f"Before: {get_available_gpu_memory(self.tp_rank, False):.2f} GB")
        self.load_model()
        # logger.info(f"After: {get_available_gpu_memory(self.tp_rank, False):.2f} GB")
        self.init_memory_pool(total_gpu_memory)

        self.is_multimodal_model = is_multimodal_model(self.model_config)

    def load_model(self):
        logger.info(f"Rank {self.tp_rank}: load weight begin.")

        device_config = DeviceConfig()
        load_config = LoadConfig(load_format=self.server_args.load_format)
        vllm_model_config = VllmModelConfig(
            model=self.server_args.model_path,
            quantization=self.server_args.quantization,
            tokenizer=None,
            tokenizer_mode=None,
            trust_remote_code=self.server_args.trust_remote_code,
            dtype=torch.float16,
            seed=42,
            skip_tokenizer_init=True,
        )
        if self.model_config.model_overide_args is not None:
            vllm_model_config.hf_config.update(self.model_config.model_overide_args)

        self.model = get_model(
            model_config=vllm_model_config,
            device_config=device_config,
            load_config=load_config,
            lora_config=None,
            vision_language_config=None,
            parallel_config=None,
            scheduler_config=None,
        )
        logger.info(f"Rank {self.tp_rank}: load weight end. {type(self.model)}")

    def profile_max_num_token(self, total_gpu_memory):
        available_gpu_memory = get_available_gpu_memory(
            self.tp_rank, distributed=self.tp_size > 1
        ) * (1 << 30)
        head_dim = self.model_config.head_dim
        head_num = self.model_config.num_key_value_heads // self.tp_size
        cell_size = head_num * head_dim * self.model_config.num_hidden_layers * 2 * 2
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - self.mem_fraction_static
        )
        max_num_token = int(rest_memory // cell_size)
        return max_num_token

    def init_memory_pool(self, total_gpu_memory):
        self.max_total_num_token = self.profile_max_num_token(total_gpu_memory)

        if self.max_total_num_token <= 0:
            raise RuntimeError(
                "Not enought memory. " "Please try to increase --mem-fraction-static."
            )

        self.req_to_token_pool = ReqToTokenPool(
            int(self.max_total_num_token / self.model_config.context_len * 256),
            self.model_config.context_len + 8,
        )
        self.token_to_kv_pool = TokenToKVPool(
            self.max_total_num_token,
            dtype=torch.float16,
            head_num=self.model_config.num_key_value_heads // self.tp_size,
            head_dim=self.model_config.head_dim,
            layer_num=self.model_config.num_hidden_layers,
        )

    @torch.inference_mode()
    def forward_prefill(self, batch: Batch):
        input_metadata = InputMetadata.create(
            self,
            forward_mode=ForwardMode.PREFILL,
            tp_size=self.tp_size,
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
            tp_size=self.tp_size,
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
    def forward_decode(self, batch: Batch):
        input_metadata = InputMetadata.create(
            self,
            forward_mode=ForwardMode.DECODE,
            tp_size=self.tp_size,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            prefix_lens=batch.prefix_lens,
            position_ids_offsets=batch.position_ids_offsets,
            out_cache_loc=batch.out_cache_loc,
            out_cache_cont_start=batch.out_cache_cont_start,
            out_cache_cont_end=batch.out_cache_cont_end,
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
            tp_size=self.tp_size,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            prefix_lens=batch.prefix_lens,
            position_ids_offsets=batch.position_ids_offsets,
            out_cache_loc=batch.out_cache_loc,
            top_logprobs_nums=batch.top_logprobs_nums,
            return_logprob=batch.return_logprob,
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
        elif forward_mode == ForwardMode.PREFILL:
            return self.forward_prefill(batch)
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
                model_arch_name_to_cls[module.EntryClass.__name__] = module.EntryClass
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