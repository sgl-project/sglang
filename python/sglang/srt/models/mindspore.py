# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import logging
from typing import Any, Iterable, Optional, Tuple

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.registry import import_model_classes
from sglang.srt.utils import is_npu

_is_npu = is_npu()

if _is_npu:
    import mindspore as ms
    import numpy as np
    import torch_npu
    from mindspore import Tensor, mint, mutable

logger = logging.getLogger(__name__)


def tensor_torch2ms(x: torch.Tensor):
    if x is None or not isinstance(x, torch.Tensor):
        return x

    # torch tensor -> dlpack -> mindspore tensor
    pt_dlpack = torch.utils.dlpack.to_dlpack(x)
    ms_tensor = ms.utils.dlpack.from_dlpack(pt_dlpack)
    return ms_tensor


def tensor_ms2torch(x: ms.Tensor):
    if x is None or not isinstance(x, ms.Tensor):
        return x

    # ms tensor -> dlpack -> torch tensor
    ms_dlpack = ms.utils.dlpack.to_dlpack(x)
    torch_tensor = torch.utils.dlpack.from_dlpack(ms_dlpack)
    torch_npu.npu.synchronize()
    return torch_tensor


# Adapt from: https://gitee.com/mindspore/vllm-mindspore/blob/master/vllm_mindspore/model_executor/models/attention_mask.py
class LowerTriangularMask:
    r"""
    Provide Infer model attention mask.
    Args:
        dtype (ms dtype): The compute type of Infer model.
        max_model_len (int): The max model length of Infer model.
    """

    def __init__(self, dtype, max_model_len, decode_mask_coeff=-10000.0):
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.cached_mask_len = 8 * 1024
        self.decode_mask_coeff = decode_mask_coeff

        prefill_mask_coeff = 1.0 if self.dtype == ms.bfloat16 else -10000.0
        self.prefill_mask = Tensor(
            np.triu(np.ones(shape=(128, 128), dtype=np.float16), k=1)
            * prefill_mask_coeff,
            dtype=self.dtype,
        )

        self.hard_mask = mint.zeros((1, 1), dtype=dtype)
        self.decode_mask = (
            Tensor(
                np.triu(
                    np.ones(
                        shape=(self.cached_mask_len, self.cached_mask_len),
                        dtype=np.int8,
                    ),
                    k=1,
                ),
                dtype=self.dtype,
            )
            * self.decode_mask_coeff
        )

    def create_mask(self, query_lens_np, seq_lens_np):
        """
        when query_lens_np = [3], seq_lens_np = [6], decode_mask_coeff = 1
        init attention mask
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 0 0 0 0 0
        """
        max_seq_len = seq_lens_np.max().item()
        total_q_len = query_lens_np.sum().item()
        attention_mask = mint.zeros((total_q_len, max_seq_len), dtype=self.dtype)

        req_num = query_lens_np.shape[0]
        current_row = 0
        for i in range(req_num):
            q_len = query_lens_np[i].item()
            current_row += q_len
            # skip row when q_len <= 1, to decrease execute time
            if q_len <= 1:
                continue
            seq_len = seq_lens_np[i].item()
            context_len = seq_len - q_len
            """
            set the right half to 1
            0 0 0 1 1 1
            0 0 0 1 1 1
            0 0 0 1 1 1
            """
            attention_mask[current_row - q_len : current_row, context_len:] = (
                self.decode_mask_coeff
            )
            """
            set the lower triangle of the right half to 0
            0 0 0 0 1 1
            0 0 0 0 0 1
            0 0 0 0 0 0
            """
            right_tensor = attention_mask[
                current_row - q_len : current_row, context_len:seq_len
            ]

            # use masked_fill_ to inplace modify attention_mask
            right_tensor.masked_fill_(right_tensor.tril() == self.decode_mask_coeff, 0)

        return attention_mask

    def gen_attention_mask(
        self,
        is_prefill: bool,
        position_ids: Tensor,
        query_lens_np: np.ndarray,
        seq_lens_np: np.ndarray,
    ):
        max_query_len = query_lens_np.max()
        max_seq_len = seq_lens_np.max()
        if is_prefill:
            attention_mask = self.prefill_mask
        elif max_query_len > 1:
            if max_seq_len <= self.cached_mask_len:
                attention_mask = mint.index_select(self.decode_mask, 0, position_ids)
            else:
                attention_mask = self.create_mask(query_lens_np, seq_lens_np)
        else:
            attention_mask = self.hard_mask
        return attention_mask


class MindSporeForCausalLM(torch.nn.Module):
    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        ms.set_context(graph_kernel_flags="--disable_pass=gather_pre_rms_norm_fusion")
        ms.set_kernel_launch_capture(False)

        logger.info(
            "MindSporeForCausalLM tp size %d tp rank %d",
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
        if get_tensor_model_parallel_world_size() not in (1, 2, 4, 8):
            # MatMulAllReduce only support tp size in (1, 2, 4, 8)
            ms.set_context(graph_kernel_flags="--disable_pass=MatMulAllReduce")

        arch = self.get_arch(self.config)
        self.model = arch(config=config, quant_config=quant_config)

        self.casual_mask = LowerTriangularMask(
            self.config.param_dtype, self.config.max_position_embeddings
        )
        self.key_cache = []
        self.value_cache = []

    def get_arch(self, config):
        # Get all implemented models
        mindspore_models = import_model_classes("sgl_mindspore.models")

        # Get arch from config
        architectures = config.architectures
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            logger.warning("No model architectures are specified")

        for arch in architectures:
            if arch in mindspore_models:
                return mindspore_models[arch]
        if arch is None:
            raise ValueError(f"Unsupported arch {architectures}")

    @property
    def use_mla(self):
        return self.config.architectures[0] in ("DeepseekV3ForCausalLM")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self.model.load_weights(weights)
        for _, cell in self.model.cells_and_names():
            quant_method = getattr(cell, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(cell)

    def get_kvcache(self, forward_batch: ForwardBatch):
        def prepare_cache(cache_list, is_key_cache):
            for i in range(self.config.num_hidden_layers):
                if is_key_cache:
                    cache = forward_batch.token_to_kv_pool.get_key_buffer(i)
                else:
                    cache = forward_batch.token_to_kv_pool.get_value_buffer(i)
                cache_ms = tensor_torch2ms(cache)
                if cache_ms.ndim == 3:
                    cache_ms = mint.unsqueeze(cache_ms, 2)
                cache_list.append(cache_ms)

        if self.use_mla:
            if not self.key_cache:
                prepare_cache(self.key_cache, is_key_cache=True)
            return mutable(self.key_cache)

        if self.key_cache and self.value_cache:
            return mutable(self.key_cache), mutable(self.value_cache)

        prepare_cache(self.key_cache, is_key_cache=True)
        prepare_cache(self.value_cache, is_key_cache=False)

        return mutable(self.key_cache), mutable(self.value_cache)

    def prepare_inputs(self, input_ids, positions, forward_batch):
        if self.use_mla:
            key_cache = self.get_kvcache(forward_batch)
        else:
            key_cache, value_cache = self.get_kvcache(forward_batch)

        # Different processing for the mindspore attention operator
        # Without any prefix cache => Use FlashAttentionScore
        # With cache => Use PagedAttention, no matter the query length is 1 or not
        is_prefill = forward_batch.forward_mode.is_extend()
        is_prefill = is_prefill and forward_batch.extend_prefix_lens.sum().item() == 0

        batch_valid_length = forward_batch.seq_lens.cpu().numpy()

        if forward_batch.extend_seq_lens is not None:
            q_seq_lens = forward_batch.extend_seq_lens.cpu().numpy()
        else:
            q_seq_lens = np.ones([forward_batch.batch_size], dtype=np.int32)

        page_size = forward_batch.token_to_kv_pool.page_size
        block_tables = tensor_torch2ms(
            (
                forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : forward_batch.seq_lens.max()
                ][:, ::page_size]
                // page_size
            )
        ).to(ms.int32)

        model_inputs = {}
        model_inputs["input_ids"] = tensor_torch2ms(input_ids).to(ms.int32)
        model_inputs["batch_valid_length"] = ms.Tensor(
            batch_valid_length, dtype=ms.int32
        )
        model_inputs["position_ids"] = tensor_torch2ms(positions)
        model_inputs["q_seq_lens"] = ms.Tensor(q_seq_lens, dtype=ms.int32)
        model_inputs["attention_mask"] = self.casual_mask.gen_attention_mask(
            is_prefill, model_inputs["position_ids"], q_seq_lens, batch_valid_length
        ).contiguous()
        model_inputs["out_cache_loc"] = tensor_torch2ms(forward_batch.out_cache_loc).to(
            ms.int32
        )
        model_inputs["is_prefill"] = is_prefill
        model_inputs["key_cache"] = key_cache
        if not self.use_mla:
            model_inputs["value_cache"] = value_cache
        model_inputs["block_tables"] = block_tables
        return model_inputs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tensor:
        # prepare base inputs
        model_inputs = self.prepare_inputs(input_ids, positions, forward_batch)
        # prepare model inputs
        model_inputs = self.model.prepare_inputs(forward_batch, model_inputs)

        logits = self.model(**model_inputs)

        # TODO: npu tensor ms2torch error to be fix, remain issues of torch_npu to get tensor from dlpack
        logits_result = LogitsProcessorOutput(next_token_logits=tensor_ms2torch(logits))
        return logits_result


EntryClass = [MindSporeForCausalLM]
