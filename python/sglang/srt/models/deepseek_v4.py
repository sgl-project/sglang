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

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
"""Inference-only DeepseekV2 model."""

from __future__ import annotations

import concurrent.futures
import logging
from contextlib import nullcontext
from enum import IntEnum, auto
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch_npu
import tqdm
import triton
import triton.language as tl
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.single_batch_overlap import SboFlags, compute_overlap_args
from sglang.srt.batch_overlap.two_batch_overlap import (
    MaybeTboDeepEPDispatcher,
    model_forward_maybe_tbo,
)
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.configs.model_config import (
    get_nsa_index_head_dim,
    get_nsa_index_n_heads,
    get_nsa_index_topk,
    is_deepseek_nsa,
)
from sglang.srt.distributed import (
    divide,
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.layers.attention.nsa.nsa_indexer import Compressor, Indexer
from sglang.srt.layers.attention.nsa.utils import (
    can_cp_split,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
    prepare_input_dp_with_cp_dsa,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
    get_attn_tp_context,
)
from sglang.srt.layers.communicator_nsa_cp import NSACPLayerCommunicator
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    dp_gather_partial,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_id_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_non_idle_and_non_empty,
    is_npu,
    log_info_on_rank0,
    make_layers,
    set_weight_attrs,
    use_intel_amx_backend,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_device_sm = get_device_sm()
_is_gfx95_supported = is_gfx95_supported()
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
if _use_aiter_gfx95:
    from sglang.srt.layers.rocm_linear_utils import (
        aiter_dsv3_router_gemm,
        get_dsv3_gemm_output_zero_allocator_size,
    )
if _is_cuda:
    from sgl_kernel import dsv3_router_gemm, merge_state_v2
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    pass
elif _is_npu:
    import custom_ops as ops  # noqa: F401

    from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
        forward_dsa_core_npu,
        forward_dsa_prepare_npu,
        forward_mha_core_npu,
        forward_mha_prepare_npu,
    )
else:
    pass

logger = logging.getLogger(__name__)

# Optional quantization for DeepSeek nvfp4 checkpoint
NVFP4_CKPT_FP8_ATTN_QUANT_MODULES = ["q_b_proj"]


@triton.jit
def triton_rms_kernel(
    hidden_state_ptr,
    hidden_state_stride_bs,
    norm_output_ptr,
    variance_epsilon,
    TOTAL_BATCH: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    core_id = tl.program_id(0)
    core_num = tl.num_programs(0)
    batch_per_core = tl.cdiv(TOTAL_BATCH, core_num)
    start_batch = core_id * batch_per_core
    end_batch = tl.minimum(start_batch + batch_per_core, TOTAL_BATCH)
    offset_d = tl.arange(0, DIM)

    for row_start in tl.range(start_batch, end_batch, BLOCK_M):
        offset_row = row_start + tl.arange(0, BLOCK_M)
        mask_r = offset_row < TOTAL_BATCH
        mask_row = mask_r[:, None]
        offset_hidden = offset_row[:, None] * hidden_state_stride_bs + offset_d[None, :]

        x = tl.load(hidden_state_ptr + offset_hidden, mask=mask_row)

        variance = tl.sum(x * x, axis=-1) / DIM
        output = x * tl.rsqrt(variance[:, None] + variance_epsilon)

        tl.store(norm_output_ptr + offset_hidden, output, mask=mask_row)


def triton_q_rms(
    q,  # bs, 64, 512
    variance_epsilon,
):
    bs, head_num, dim = q.shape
    total_batch = bs * head_num
    q = q.view(total_batch, dim)

    if dim > 2048:
        raise NotImplementedError("dim > 2048 not supported")

    device_properties = triton.runtime.driver.active.utils.get_device_properties(
        q.device
    )
    num_vectorcore = device_properties.get("num_vectorcore", -1)

    ROW_BLOCK_SIZE = 16  # A safe default balancing parallelism and register pressure.
    batch_per_core = triton.cdiv(total_batch, num_vectorcore)
    BLOCK_M = min(ROW_BLOCK_SIZE, batch_per_core)

    grid = (num_vectorcore,)
    norm_output = torch.empty_like(q)

    triton_rms_kernel[grid](
        q,
        q.stride(0),
        norm_output,
        variance_epsilon,
        total_batch,
        dim,
        BLOCK_M,
    )
    return norm_output.view(bs, head_num, dim)


@triton.jit
def triton_rope_kernel_in_place(
    x_ptr,
    sin_ptr,
    cos_ptr,
    x_stride,
    cos_stride,
    hidden_size: tl.constexpr,
    rope_dim: tl.constexpr,
    head_num: tl.constexpr,
):
    cur_b = tl.program_id(0)
    dim_start = hidden_size - rope_dim
    # load x
    offset_x = cur_b * x_stride + dim_start + tl.arange(0, rope_dim)
    x = tl.load(x_ptr + offset_x).to(tl.float32)
    # load sin cos
    offset_sin_cos = cur_b // head_num * cos_stride + tl.arange(0, rope_dim)
    sin = tl.load(sin_ptr + offset_sin_cos).to(tl.float32)
    cos = tl.load(cos_ptr + offset_sin_cos).to(tl.float32)

    even = tl.extract_slice(x, [0], [rope_dim // 2], [2])
    odd = tl.extract_slice(x, [1], [rope_dim // 2], [2])
    odd = -odd

    x_rotate = tl.zeros([rope_dim], dtype=tl.float32)
    x_rotate = tl.insert_slice(x_rotate, odd, [0], [rope_dim // 2], [2])
    x_rotate = tl.insert_slice(x_rotate, even, [1], [rope_dim // 2], [2])

    out = x * cos + x_rotate * sin
    tl.store(x_ptr + offset_x, out.to(tl.bfloat16))


def triton_apply_rope_partial_in_place(x, sin, cos):
    rope_dim = sin.shape[-1]
    org_shape = x.shape
    if x.dim() == 2:
        bsz, hidden_size = x.shape
        head_num = 1
    elif x.dim() == 3:
        bsz, head_num, hidden_size = x.shape
        x = x.view(-1, hidden_size)
    else:
        raise NotImplementedError(f"x_shape={x.shape} not supported")
    cores = bsz * head_num
    assert cores < 65535
    triton_rope_kernel_in_place[(cores,)](
        x,
        sin,
        cos,
        x.stride(0),
        sin.stride(0),
        hidden_size,
        rope_dim,
        head_num,
    )
    return x.view(org_shape)


def enable_nextn_moe_bf16_cast_to_fp8(quant_config):
    return (
        envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get()
        and quant_config is not None
        and quant_config.get_name() == "modelopt_fp4"
        and get_moe_runner_backend().is_deep_gemm()
    )


@lru_cache(1)
def get_window_topk_idxs(
    num_tokens, window_size: int, seq_lens: torch.Tensor, is_prefill: bool
):
    # if not is_prefill:
    #     assert num_tokens == seq_lens.numel()
    window_topk_idxs = torch.full(
        (num_tokens, window_size), -1, dtype=torch.int32, device=seq_lens.device
    )
    seq_len_offset = 0
    for idx, seq_len in enumerate(seq_lens):
        start_pos = 0 if is_prefill else seq_len - 1
        if start_pos >= window_size - 1:
            window_topk_idxs[idx : idx + 1] = torch.arange(
                window_size, dtype=torch.int32, device=window_topk_idxs.device
            )
        elif start_pos > 0:
            window_topk_idxs[idx : idx + 1] = F.pad(
                torch.arange(
                    start_pos + 1, dtype=torch.int32, device=window_topk_idxs.device
                ),
                (0, window_size - start_pos - 1),
                value=-1,
            )
        else:
            base = torch.arange(
                seq_len, dtype=torch.int32, device=window_topk_idxs.device
            ).unsqueeze(1)
            matrix = (base - window_size + 1).clamp(0) + torch.arange(
                min(seq_len, window_size),
                dtype=torch.int32,
                device=window_topk_idxs.device,
            )
            window_topk_idxs[seq_len_offset : seq_len_offset + seq_len, :seq_len] = (
                torch.where(matrix > base, -1, matrix)
            )
        seq_len_offset += seq_len
    return window_topk_idxs


@lru_cache(2)
def get_compress_topk_idxs(
    num_tokens,
    ratio: int,
    seq_lens: torch.Tensor,
    window_size: int,
    is_prefill: bool,
    need_add_offset: bool,
):
    max_len = seq_lens.max() // ratio
    compress_topk_idxs = torch.full(
        (num_tokens, max_len), -1, dtype=torch.int32, device=seq_lens.device
    )
    seq_len_offset = 0
    for idx, seq_len in enumerate(seq_lens):
        if not is_prefill:
            topk_idxs = torch.arange(
                0, seq_len // ratio, dtype=torch.int32, device=seq_lens.device
            )
            if need_add_offset:
                topk_idxs += window_size
            compress_topk_idxs[idx : idx + 1, : topk_idxs.numel()] = topk_idxs
        else:
            matrix = torch.arange(
                seq_len // ratio, dtype=torch.int32, device=seq_lens.device
            ).repeat(seq_len, 1)
            mask = (
                matrix
                >= torch.arange(
                    1, seq_len + 1, dtype=torch.int32, device=seq_lens.device
                ).unsqueeze(1)
                // ratio
            )
            if need_add_offset:
                topk_idxs = torch.where(mask, -1, matrix + seq_len)
            else:
                topk_idxs = torch.where(mask, -1, matrix)
            compress_topk_idxs[
                seq_len_offset : seq_len_offset + seq_len, : seq_len // ratio
            ] = topk_idxs

        seq_len_offset += seq_len
    return compress_topk_idxs


class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()

    # Use multi-head attention, execute the MHA for prefix and extended kv in one shot
    # when the sequence lengths are below the threshold.
    MHA_ONE_SHOT = auto()

    # Use multi-head attention for NPU
    MHA_NPU = auto()

    # Use Deepseek V3.2 sparse multi-latent attention for NPU
    DSA_NPU = auto()


def _dispatch_mla_subtype(attn, forward_batch):
    if _is_hip:
        if attn.rocm_fused_decode_mla and forward_batch.forward_mode.is_decode():
            return AttnForwardMethod.MLA_FUSED_ROPE
        else:
            return AttnForwardMethod.MLA
    else:
        if hasattr(attn, "fused_qkv_a_proj_with_mqa") and use_intel_amx_backend(attn):
            return AttnForwardMethod.MLA_FUSED_ROPE_CPU
        else:
            return AttnForwardMethod.MLA


def hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    # mixes: [b, s, mix_hc], hc_scale: [3], hc_base: [mix_hc]
    # mix_hc = (hc + 2) * hc
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = (
        F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)) + eps
    )
    post = 2 * F.sigmoid(
        post * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult].unsqueeze(0).unsqueeze(0)
    )
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult).unsqueeze(
        0
    ).unsqueeze(0)

    comb = comb.softmax(-1) + eps
    col_sum = comb.sum(-2, keepdim=True)
    comb = comb / (col_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + eps)
    return pre, post, comb


class AttentionBackendRegistry:
    _handlers = {}

    @classmethod
    def register(cls, backend_name, handler_func):
        cls._handlers[backend_name] = handler_func

    @classmethod
    def get_handler(cls, backend_name):
        return cls._handlers.get(backend_name, cls._handlers.get("triton"))


def handle_attention_ascend(attn, forward_batch):
    return AttnForwardMethod.MHA
    if (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_target_verify()
        and not forward_batch.forward_mode.is_draft_extend()
        and not forward_batch.forward_mode.is_draft_extend_v2()
    ):
        if hasattr(attn, "indexer"):
            return AttnForwardMethod.DSA_NPU
        else:
            return AttnForwardMethod.MHA_NPU
    else:
        if hasattr(attn, "indexer"):
            return AttnForwardMethod.DSA_NPU
        else:
            return AttnForwardMethod.MHA_NPU


def _get_sum_extend_prefix_lens(forward_batch):
    return (
        sum(forward_batch.extend_prefix_lens_cpu)
        if forward_batch.extend_prefix_lens_cpu is not None
        else 0
    )


def _support_mha_one_shot(attn: DeepseekV4AttentionMLA, forward_batch, backend_name):
    attn_supported = backend_name in ["fa3", "flashinfer", "flashmla"]
    sum_seq_lens = (
        sum(forward_batch.seq_lens_cpu) if forward_batch.seq_lens_cpu is not None else 0
    )
    return attn_supported and sum_seq_lens <= forward_batch.get_max_chunk_capacity()


def _handle_attention_backend(
    attn: DeepseekV4AttentionMLA, forward_batch, backend_name
):
    if is_in_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    sum_extend_prefix_lens = _get_sum_extend_prefix_lens(forward_batch)
    disable_ragged = (
        backend_name in ["flashinfer", "flashmla"]
    ) and attn.flashinfer_mla_disable_ragged

    if (
        not disable_ragged
        and forward_batch.forward_mode.is_extend_without_speculative()
        and (
            (
                sum_extend_prefix_lens >= attn.chunked_prefix_cache_threshold
                and not attn.disable_chunked_prefix_cache
            )
            or sum_extend_prefix_lens == 0
        )
    ):
        if _support_mha_one_shot(attn, forward_batch, backend_name):
            return AttnForwardMethod.MHA_ONE_SHOT
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


def handle_attention_flashinfer(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "flashinfer")


def handle_attention_fa3(attn, forward_batch):
    # when deterministic inference is enabled, use MLA
    if get_global_server_args().enable_deterministic_inference:
        return _dispatch_mla_subtype(attn, forward_batch)
    else:
        return _handle_attention_backend(attn, forward_batch, "fa3")


def handle_attention_flashmla(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "flashmla")


def handle_attention_cutlass_mla(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "cutlass_mla")


def handle_attention_fa4(attn, forward_batch):
    # TODO(cicirori): use FA4 MHA for DeepSeekV3 for now
    return AttnForwardMethod.MHA_CHUNKED_KV


def handle_attention_trtllm_mla(attn, forward_batch):
    if is_in_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    sum_extend_prefix_lens = _get_sum_extend_prefix_lens(forward_batch)
    if forward_batch.forward_mode.is_extend_without_speculative() and (
        not attn.disable_chunked_prefix_cache or sum_extend_prefix_lens == 0
    ):
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


def handle_attention_aiter(attn, forward_batch):
    if forward_batch.forward_mode.is_extend_without_speculative():
        return AttnForwardMethod.MHA
    else:
        return AttnForwardMethod.MLA


def handle_attention_nsa(attn, forward_batch):
    """
    Dispatch logic is centralized in NativeSparseAttnBackend.set_nsa_prefill_impl and executed
    in init_forward_metadata. Read the decision from backend.use_mha.
    """

    backend = forward_batch.attn_backend
    if isinstance(backend, TboAttnBackend):  # if enable tbo, get primary backend
        backend = backend.primary
    if hasattr(backend, "use_mha") and backend.use_mha:
        return AttnForwardMethod.MHA_ONE_SHOT
    return AttnForwardMethod.MLA


def handle_attention_triton(attn, forward_batch):
    if is_in_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    # when deterministic inference is enabled, use MLA
    if get_global_server_args().enable_deterministic_inference:
        return _dispatch_mla_subtype(attn, forward_batch)

    if (
        forward_batch.forward_mode.is_extend_without_speculative()
        and sum(forward_batch.extend_prefix_lens_cpu) == 0
    ):
        return AttnForwardMethod.MHA
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w1", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("w2", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if not hasattr(self.gate_up_proj, "weight"):
            self.gate_up_proj.weight = getattr(self.gate_up_proj, "weight_packed")
        if not hasattr(self.down_proj, "weight"):
            self.down_proj.weight = getattr(self.down_proj, "weight_packed")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul(**kwargs)

    def forward(
        self,
        x,
        forward_batch=None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        if (
            gemm_output_zero_allocator is not None
            and x.shape[0] <= 256
            and self.gate_up_proj.weight.dtype == torch.uint8
        ):
            y = gemm_output_zero_allocator.allocate(
                x.shape[0] * self.gate_up_proj.output_size_per_partition
            ).view(x.shape[0], self.gate_up_proj.output_size_per_partition)
            x = (x, None, y)

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
        return x


class MoEGate(nn.Module):
    def __init__(
        self,
        config,
        quant_config,
        prefix: str = "",
        is_nextn: bool = False,
        is_hash_layer: bool = False,
    ):
        super().__init__()
        self.is_nextn = is_nextn
        self.weight = nn.Parameter(
            torch.empty(
                (config.n_routed_experts, config.hidden_size), dtype=torch.float32
            )
        )
        if config.topk_method == "noaux_tc":
            correction_bias_dtype = (
                torch.bfloat16
                if quant_config is not None
                and quant_config.get_name() == "modelopt_fp4"
                and get_moe_runner_backend().is_flashinfer_trtllm()
                else torch.float32
            )
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts), dtype=correction_bias_dtype)
            )
        else:
            self.e_score_correction_bias = None
        if _is_cpu and _is_cpu_amx_available:
            self.quant_method = PackWeightMethod(weight_names=["weight"])
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        self.is_hash_layer = is_hash_layer
        if is_hash_layer:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    config.vocab_size, config.n_activated_experts, dtype=torch.int32
                ),
                requires_grad=False,
            )
            self.e_score_correction_bias = None
        else:
            self.tid2eid = None

    def forward(
        self,
        hidden_states,
        gemm_output_zero_allocator: BumpAllocator = None,
        forward_batch: ForwardBatch = None,
    ):
        if use_intel_amx_backend(self):
            return torch.ops.sgl_kernel.weight_packed_linear(
                hidden_states.float(),
                self.weight.float(),
                None,  # bias
                True,  # is_vnni
            )

        if get_global_server_args().enable_deterministic_inference:
            return F.linear(hidden_states.float(), self.weight.float(), None)

        if forward_batch is not None and nsa_use_prefill_cp(forward_batch):
            logits = F.linear(hidden_states.float(), self.weight.float(), None)
        else:
            # NOTE: For some unknown reason, router_gemm seems degrade accept length.
            if (
                _is_cuda
                and hidden_states.shape[0] <= 16
                and hidden_states.shape[1] == 7168
                and (self.weight.shape[0] == 256 or self.weight.shape[0] == 384)
                and _device_sm >= 90
            ):

                # router gemm output float32
                logits = dsv3_router_gemm(
                    hidden_states.float(), self.weight.float(), out_dtype=torch.float32
                )
            elif _use_aiter_gfx95 and hidden_states.shape[0] <= 256:
                logits = aiter_dsv3_router_gemm(
                    hidden_states.float(),
                    self.weight.float(),
                    gemm_output_zero_allocator,
                )
            else:
                logits = F.linear(hidden_states.float(), self.weight, None)

        return logits


class DeepseekV4MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.num_fused_shared_experts = (
            0
            if get_global_server_args().disable_shared_experts_fusion
            else config.n_shared_experts
        )
        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("gate", prefix),
            is_nextn=is_nextn,
            is_hash_layer=layer_id < config.num_hash_layers and not is_nextn,
        )

        # scaling factor for fused shared experts on AMD-platform.
        fused_shared_experts_scaling_factor = None
        if self.moe_ep_size > 1 and self.num_fused_shared_experts > 0:
            # if enable_ep_moe tp_szie == ep_size, every gpu get shared experts gemm output
            # so we scale with 1 / self.moe_ep_size in ep mode which will make it equalation as in tp mode
            # with fused_shared_experts
            fused_shared_experts_scaling_factor = 1.0 / float(self.moe_ep_size)

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts
            + self.num_fused_shared_experts
            + get_global_server_args().ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_method_type=getattr(
                config, "routing_method_type", RoutingMethodType.DeepSeekV3
            ),
            prefix=add_prefix("experts", prefix),
        )
        self.experts.should_fuse_routed_scaling_factor_in_topk = True

        self.topk = TopK(
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=False,
            num_expert_group=config.n_group,
            num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=config.topk_group,
            scoring_func=config.scoring_func,
            is_hash_layer=self.gate.is_hash_layer,
            tid2eid=self.gate.tid2eid,
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
            # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized
            # and requires the output format to be standard (except trtllm). We use quant_config to determine the output format.
            output_format=(
                TopKOutputFormat.STANDARD
                if (quant_config is None)
                and (not get_moe_runner_backend().is_flashinfer_trtllm())
                else None
            ),
        )

        self.shared_experts_is_int8 = False
        self.shared_experts_is_fp8 = False
        self.shared_experts_weight_block_size = None
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # disable tp for shared experts when enable deepep moe, or with fp4 allgather
            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                swiglu_limit=(
                    config.swiglu_limit if hasattr(config, "swiglu_limit") else 0
                ),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    or get_moe_a2a_backend().is_mooncake()
                    or get_moe_a2a_backend().is_ascend_fuseep()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else {}
                ),
            )
            is_packed_weight = hasattr(
                self.shared_experts.gate_up_proj.quant_method, "quant_config"
            ) and self.shared_experts.gate_up_proj.quant_method.quant_config.get_name() in {
                "awq",
                "awq_marlin",
                "moe_wna16",
            }
            self.shared_experts_is_int8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.int8
            )
            self.shared_experts_is_fp8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn
            )
            if self.shared_experts_is_fp8:
                if (
                    _use_aiter
                    and config.quantization_config.get("quant_method")
                    == "compressed-tensors"
                ):
                    # For compressed-tensors ptpc model, don't need to check the weight_block_size
                    pass
                else:
                    assert (
                        self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
                        == self.shared_experts.down_proj.quant_method.quant_config.weight_block_size
                    )
                    self.shared_experts_weight_block_size = (
                        self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
                    )

        self.top_k = config.num_experts_per_tok

        if (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_ascend_fuseep()
        ):
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + get_global_server_args().ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

        self._enable_a2a_moe = (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_ascend_fuseep()
        )
        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids_dp_full: torch.Tensor = None,
    ) -> torch.Tensor:
        input_ids = None
        if input_ids_dp_full is not None:
            input_ids = input_ids_dp_full
        elif forward_batch is not None:
            input_ids = getattr(forward_batch, "input_ids", None)

        if not self._enable_a2a_moe:
            from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

            if (
                self.alt_stream is not None
                and self.num_fused_shared_experts == 0
                and hidden_states.shape[0] > 0
                and get_is_capture_mode()
            ):
                return self.forward_normal_dual_stream(
                    hidden_states,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                    gemm_output_zero_allocator,
                    input_ids=input_ids,
                )
            else:
                return self.forward_normal(
                    hidden_states,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                    gemm_output_zero_allocator,
                    input_ids=input_ids,
                )
        else:
            return self.forward_deepep(
                hidden_states, forward_batch, input_ids=input_ids
            )

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids: torch.Tensor = None,
    ) -> torch.Tensor:

        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(
            hidden_states, gemm_output_zero_allocator
        )

        with torch.cuda.stream(self.alt_stream):
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_output = self.topk(hidden_states, router_logits, input_ids=input_ids)
            final_hidden_states = self.experts(hidden_states, topk_output)
            if not _is_cuda or isinstance(self.experts.quant_method, KTEPWrapperMethod):
                final_hidden_states *= self.routed_scaling_factor

        current_stream.wait_stream(self.alt_stream)
        final_hidden_states += shared_output
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        if hasattr(self, "shared_experts") and use_intel_amx_backend(
            self.shared_experts.gate_up_proj
        ):
            return self.forward_cpu(hidden_states, should_allreduce_fusion)

        if hidden_states.shape[0] > 0:
            if (
                not self._fuse_shared_experts_inside_sbo
            ):  # TODO: check if it supports mtp
                shared_output = self._forward_shared_experts(
                    hidden_states, gemm_output_zero_allocator
                )
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_output = self.topk(hidden_states, router_logits, input_ids=input_ids)
        else:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        if self._fuse_shared_experts_inside_sbo:
            shared_output = None

            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):

                nonlocal shared_output
                self.alt_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.alt_stream):
                    shared_output = self._forward_shared_experts(
                        hidden_states, gemm_output_zero_allocator
                    )

                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                nonlocal shared_output
                torch.cuda.current_stream().wait_stream(self.alt_stream)
                post_combine_hook_handle.remove()

            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        final_hidden_states = self.experts(
            hidden_states,
            topk_output,
        )
        if (
            not _is_cuda
            and not _use_aiter
            or isinstance(self.experts.quant_method, KTEPWrapperMethod)
        ):
            # fused in biased_grouped_topk so we can skip here
            final_hidden_states *= self.routed_scaling_factor
        if shared_output is not None:
            final_hidden_states += shared_output
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
    ) -> torch.Tensor:
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        fused_experts_out = self.experts(
            hidden_states=hidden_states, topk_output=topk_output
        )

        assert use_intel_amx_backend(
            self.shared_experts.gate_up_proj
        ) == use_intel_amx_backend(self.shared_experts.down_proj)
        # [Note] inplace should be False in fused_experts.
        # If inplace is True in fused_experts (self.experts), hidden_states will be changed after fused_experts
        # While hidden_states is still needed in shared_expert.
        final_hidden_states = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states,
            self.shared_experts.gate_up_proj.weight,
            self.shared_experts.down_proj.weight,
            fused_experts_out,
            self.routed_scaling_factor,
            True,  # inplace
            self.shared_experts_is_int8,  # use_int8_w8a8
            self.shared_experts_is_fp8,  # use_fp8_w8a16
            (
                self.shared_experts.gate_up_proj.weight_scale
                if self.shared_experts_is_int8
                else (
                    self.shared_experts.gate_up_proj.weight_scale_inv
                    if self.shared_experts_is_fp8
                    else None
                )
            ),  # w1_scale
            (
                self.shared_experts.down_proj.weight_scale
                if self.shared_experts_is_int8
                else (
                    self.shared_experts.down_proj.weight_scale_inv
                    if self.shared_experts_is_fp8
                    else None
                )
            ),  # w2_scale
            (
                self.shared_experts_weight_block_size
                if self.shared_experts_is_fp8
                else None
            ),  # block_size
            None,  # a1_scale
            None,  # a2_scale
            True,  # is_vnni
        )
        if self.tp_size > 1 and not should_allreduce_fusion:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_deepep(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        shared_output = None
        sbo_enabled_flag = self._fuse_shared_experts_inside_sbo and not self.is_nextn
        sbo_overlap_dispatch_flag = (
            sbo_enabled_flag and SboFlags.enable_dispatch_shared_one_stream_overlap()
        )
        sbo_overlap_combine_flag = (
            sbo_enabled_flag and SboFlags.enable_combine_shared_two_stream_overlap()
        )

        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, forward_batch=forward_batch)
            if not sbo_enabled_flag:
                if self.alt_stream is not None:
                    self.alt_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.alt_stream):
                        shared_output = self._forward_shared_experts(hidden_states)
                        shared_output.record_stream(self.alt_stream)
                        shared_event = self.alt_stream.record_event()
                else:
                    shared_output = self._forward_shared_experts(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                input_ids=input_ids,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        if sbo_overlap_dispatch_flag:
            shared_output = None

            def _deepep_dispatch_hook(dispatcher: BaseDispatcher):
                nonlocal shared_output
                shared_output = self._forward_shared_experts(hidden_states)
                for handle in deepep_dispatch_hook_handle:
                    handle.remove()

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):
                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                post_dispatch_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            assert isinstance(self.experts.dispatcher, MaybeTboDeepEPDispatcher)
            deepep_dispatch_hook_handle = (
                self.experts.dispatcher.register_deepep_dispatch_hook(
                    _deepep_dispatch_hook
                )
            )
            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        elif sbo_overlap_combine_flag:
            shared_output = None

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):

                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )

                post_dispatch_hook_handle.remove()

            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):

                nonlocal shared_output

                if (
                    e := dispatcher.meta_overlap_args.get("record_event_after_down")
                ) is not None:
                    e.record()

                # TODO reduce sm for non-deepgemm
                with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                    dispatcher.meta_overlap_args["compute_num_sms"]
                ):
                    shared_output = self._forward_shared_experts(hidden_states)

                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if (
            hidden_states.shape[0] > 0
            and not sbo_enabled_flag
            and self.alt_stream is not None
        ):
            torch.cuda.current_stream().wait_event(shared_event)
        if shared_output is not None:
            x = shared_output
            if self.experts.should_fuse_routed_scaling_factor_in_topk:
                x.add_(final_hidden_states)
            else:
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            if not self.experts.should_fuse_routed_scaling_factor_in_topk:
                final_hidden_states *= self.routed_scaling_factor

        return final_hidden_states

    def _forward_shared_experts(
        self, hidden_states, gemm_output_zero_allocator: BumpAllocator = None
    ):
        if (hidden_states.shape[0] > 0) and (self.num_fused_shared_experts == 0):
            return self.shared_experts(
                hidden_states, gemm_output_zero_allocator=gemm_output_zero_allocator
            )
        else:
            return None

    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits = self.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def op_shared_experts(self, state):
        hidden_states_mlp_input = state.pop("hidden_states_mlp_input")
        if (self.num_fused_shared_experts == 0) and is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, hidden_states_mlp_input
        ):
            state.shared_output = self.shared_experts(hidden_states_mlp_input)
        else:
            state.shared_output = None

    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input

        if router_logits is not None:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.topk_output = self.topk(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_output = self.topk.empty_topk_output(hidden_states.device)

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.dispatch_a(
                hidden_states=state.hidden_states_mlp_input,
                topk_output=state.pop("topk_output"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.dispatch_output = self.experts.dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        state.combine_input = self.experts.run_moe_core(
            dispatch_output=state.dispatch_output,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.combine_a(
                combine_input=state.pop("combine_input"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = self.experts.dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_output(self, state):
        final_hidden_states = state.pop("hidden_states_after_combine")

        if (shared_output := state.pop("shared_output")) is not None:
            x = shared_output
            x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            final_hidden_states *= self.routed_scaling_factor

        state.hidden_states_mlp_output = final_hidden_states


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV4AttentionMLA(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_rope_head_dim: int,
        q_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size  # model dimension
        self.head_dim = config.head_dim  # attention dimension
        self.o_groups = config.o_groups  # Number of groups of out projections.
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = self.head_dim - qk_rope_head_dim
        self.qk_head_dim = self.head_dim
        self.v_head_dim = self.head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = None
        self.o_lora_rank = config.o_lora_rank  # Dimension for out projections.
        self.quant_config = quant_config
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.attn_tp_size = attn_tp_size
        self.use_nsa = is_deepseek_nsa(config)
        assert (
            self.o_groups >= attn_tp_size and self.o_groups % attn_tp_size == 0
        ), f"{self.o_groups=} and {attn_tp_size=} is not supported"
        self.local_o_groups = self.o_groups // attn_tp_size
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            assert self.use_nsa, "CP currently only supports deepseek v3.2 model"
        # cp reuse the attn_tp comm group but need to duplicate the weights
        if self.nsa_enable_prefill_cp and self.use_nsa:
            attn_tp_rank = 0
            attn_tp_size = 1
            self.cp_size = get_attention_tp_size()
        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.kv_cache_dtype = get_global_server_args().kv_cache_dtype
        self.window_size = config.sliding_window_size
        self.rms_norm_eps = config.rms_norm_eps
        # For tensor parallel attention
        assert self.q_lora_rank is not None
        self.attn_sink = nn.Parameter(
            torch.empty(self.num_local_heads, dtype=torch.float32)
        )
        set_weight_attrs(self.attn_sink, {"weight_loader": sharded_weight_loader(0)})

        self.wq_a = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_a", prefix),
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_norm_dummy_weight = torch.ones(
            self.head_dim,
            dtype=self.q_norm.weight.dtype,
            device=self.q_norm.weight.device,
        )
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.wkv = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wkv", prefix),
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = ColumnParallelLinear(
            self.num_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.wo_b = RowParallelLinear(
            self.o_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.compress_ratio = config.compress_ratios[layer_id]

        if not skip_rope:
            self.rotary_emb = get_rope_wrapper(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=(
                    config.compress_rope_theta
                    if self.compress_ratio > 1
                    else rope_theta
                ),
                rope_scaling=rope_scaling,
                is_neox_style=True,
                device=get_global_server_args().device,
            )

            if rope_scaling:
                pass
                # mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
                # scaling_factor = rope_scaling["factor"]
                # mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                # self.scaling = self.scaling * mscale * mscale
            else:
                self.rotary_emb.forward = self.rotary_emb.forward_native
        else:
            self.rotary_emb = None

        self.indexer = None
        self.compressor = None
        if self.use_nsa and self.compress_ratio > 1:
            self.compressor = Compressor(
                self.hidden_size,
                self.head_dim,
                qk_rope_head_dim,
                layer_id=layer_id,
                compress_ratio=self.compress_ratio,
                quant_config=quant_config,
                prefix=add_prefix("compressor", prefix),
            )
            self.compressor.rotary_emb = self.rotary_emb
            if self.compress_ratio == 4:
                self.indexer = Indexer(
                    hidden_size=hidden_size,
                    index_n_heads=get_nsa_index_n_heads(config),
                    index_head_dim=get_nsa_index_head_dim(config),
                    rope_head_dim=qk_rope_head_dim,
                    index_topk=get_nsa_index_topk(config),
                    q_lora_rank=q_lora_rank,
                    max_position_embeddings=max_position_embeddings,
                    rope_theta=config.compress_rope_theta,
                    scale_fmt="ue8m0",
                    block_size=128,
                    rope_scaling=rope_scaling,
                    prefix=add_prefix("indexer", prefix),
                    quant_config=quant_config,
                    layer_id=layer_id,
                    alt_stream=alt_stream,
                    compress_ratio=self.compress_ratio,
                    window_size=self.window_size,
                )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            sliding_window_size=self.window_size,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.alt_stream = alt_stream
        self.disable_chunked_prefix_cache = (
            get_global_server_args().disable_chunked_prefix_cache
        )
        self.current_attention_backend = (
            None  # Attention backend used by current forward batch
        )
        # TODO: Design a finer way to determine the threshold
        self.chunked_prefix_cache_threshold = (
            envs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD.get()
        )

    def dispatch_attn_forward_method(
        self, forward_batch: ForwardBatch
    ) -> AttnForwardMethod:
        # Determine attention backend used by current forward batch
        if forward_batch.forward_mode.is_decode_or_idle():
            attention_backend = get_global_server_args().decode_attention_backend
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            # Use the specified backend for speculative operations (both verify and draft extend)
            if get_global_server_args().speculative_attention_mode == "decode":
                attention_backend = get_global_server_args().decode_attention_backend
            else:  # default to prefill
                attention_backend = get_global_server_args().prefill_attention_backend
        else:
            attention_backend = get_global_server_args().prefill_attention_backend
        self.current_attention_backend = attention_backend

        handler = AttentionBackendRegistry.get_handler(attention_backend)
        return handler(self, forward_batch)

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        output = self.forward_core(s)
        return output

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        # when hidden_states is a tuple of tensors, the tuple will include quantized weight and scale tensor
        if isinstance(hidden_states, tuple):
            if (
                not get_attn_tp_context().input_scattered
                and hidden_states[0].shape[0] == 0
            ):
                return hidden_states[0]
        else:
            if not get_attn_tp_context().input_scattered and (
                hidden_states.shape[0] == 0 or forward_batch.forward_mode.is_idle()
            ):
                return hidden_states, None, forward_batch, None

        attn_forward_method = self.dispatch_attn_forward_method(forward_batch)
        if attn_forward_method == AttnForwardMethod.MHA:
            inner_state = self.forward_normal_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            inner_state = self.forward_normal_chunked_kv_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_ONE_SHOT:
            inner_state = self.forward_normal_one_shot_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_NPU:
            inner_state = forward_mha_prepare_npu(
                self, positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.DSA_NPU:
            inner_state = forward_dsa_prepare_npu(
                self, positions, hidden_states, forward_batch, zero_allocator
            )
        else:
            raise NotImplementedError
        return None, attn_forward_method, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, attn_forward_method, forward_batch, inner_state = (
            intermediate_state
        )
        if inner_state is None:
            return hidden_states

        if attn_forward_method == AttnForwardMethod.MHA:
            return self.forward_normal_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            return self.forward_normal_chunked_kv_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_ONE_SHOT:
            return self.forward_normal_one_shot_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_NPU:
            return forward_mha_core_npu(self, *inner_state)
        elif attn_forward_method == AttnForwardMethod.DSA_NPU:
            return forward_dsa_core_npu(self, *inner_state)
        else:
            raise NotImplementedError

    def forward_normal_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        x = hidden_states
        num_tokens = x.size(0)
        use_fused_mla_prolog = get_bool_env_var("USE_FUSED_MLA_PROLOG")
        if use_fused_mla_prolog:
            raise NotImplementedError("Fused MLA prolog is not supported yet.")
        else:
            qr = q = self.q_norm(self.wq_a(x)[0])
            q = self.wq_b(q)[0].unflatten(-1, (self.num_local_heads, self.head_dim))
            if get_bool_env_var("USE_FUSED_Q_RMS") and (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                q = triton_q_rms(q, self.rms_norm_eps)
            else:
                # q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.rms_norm_eps)
                q = torch_npu.npu_rms_norm(
                    q, self.q_b_norm_dummy_weight, self.rms_norm_eps
                )[0]

            kv = self.wkv(x)[0]
            kv = self.kv_norm(kv)  # [T, D]

            if self.attn_mha.layer_id in [0, 2]:  # TODO: FIX MAGIC NUMBER
                self.rotary_emb.get_cos_sin_with_position(positions)

            use_rope_partial_in_place_triton = get_bool_env_var(
                "USE_ROPE_PARTIAL_IN_PLACE_TRITON"
            ) and (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
            )
            use_rope_partial_in_place_ascendc = get_bool_env_var(
                "USE_ROPE_PARTIAL_IN_PLACE_ASCENDC"
            )
            if use_rope_partial_in_place_triton:
                q = triton_apply_rope_partial_in_place(
                    q,
                    self.rotary_emb.position_sin_layer_cache,
                    self.rotary_emb.position_cos_layer_cache,
                )
                kv = triton_apply_rope_partial_in_place(
                    kv,
                    self.rotary_emb.position_sin_layer_cache,
                    self.rotary_emb.position_cos_layer_cache,
                )
            elif use_rope_partial_in_place_ascendc:
                torch.ops.custom.inplace_partial_rotary_mul(
                    q.unsqueeze(2),  # [t, 1, n, d]
                    self.rotary_emb.position_cos_layer_cache,  # [t, 1, 1, d]
                    self.rotary_emb.position_sin_layer_cache,  # [t, 1, 1, d]
                    rotary_mode="interleave",
                    partial_slice=[self.qk_nope_head_dim, self.head_dim],  # [448, 512]
                )
                torch.ops.custom.inplace_partial_rotary_mul(
                    kv.view(-1, 1, 1, self.head_dim),
                    self.rotary_emb.position_cos_layer_cache,
                    self.rotary_emb.position_sin_layer_cache,
                    rotary_mode="interleave",
                    partial_slice=[self.qk_nope_head_dim, self.head_dim],
                )
            else:
                self.rotary_emb(
                    q[..., -self.qk_rope_head_dim :],
                    self.rotary_emb.position_sin_layer_cache,
                    self.rotary_emb.position_cos_layer_cache,
                )
                self.rotary_emb(
                    kv[..., -self.qk_rope_head_dim :],
                    self.rotary_emb.position_sin_layer_cache,
                    self.rotary_emb.position_cos_layer_cache,
                )

        is_prefill = (
            forward_batch.forward_mode.is_prefill()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
        )
        need_compute_topk_idxs = (
            is_prefill and not get_bool_env_var("USE_PA_PREFILL")
        ) or (not get_bool_env_var("USE_PA_DECODE") and not is_prefill)
        if need_compute_topk_idxs:
            topk_idxs = get_window_topk_idxs(
                num_tokens, self.window_size, forward_batch.seq_lens, is_prefill
            )
        else:
            topk_idxs = None

        if self.compress_ratio > 1:
            if self.indexer is not None:
                # NSA Indexer: cache quantized keys, auto-skip topk for sequences <= nsa_index_topk
                compress_topk_idxs = self.indexer(  # index compress/state
                    x=x,
                    q_lora=qr,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.layer_id,
                )
                # An offset is required when using the original FA implementation;
                # it can be removed once the fusion-kernel is integrated in the future.
                offset = None
                if is_prefill and not get_bool_env_var("USE_PA_PREFILL"):
                    offset = torch.cat(
                        [
                            torch.full(
                                (seq_len, 1),
                                seq_len,
                                dtype=compress_topk_idxs.dtype,
                                device=compress_topk_idxs.device,
                            )
                            for seq_len in forward_batch.seq_lens
                        ],
                        dim=0,
                    )
                if not is_prefill and not get_bool_env_var("USE_PA_DECODE"):
                    offset = self.window_size
                if offset is not None:
                    compress_topk_idxs = torch.where(
                        compress_topk_idxs != -1,
                        compress_topk_idxs + offset,
                        compress_topk_idxs,
                    )
            else:
                if need_compute_topk_idxs:
                    compress_topk_idxs = get_compress_topk_idxs(
                        num_tokens,
                        self.compress_ratio,
                        forward_batch.seq_lens,
                        window_size=self.window_size,
                        is_prefill=is_prefill,
                        need_add_offset=need_compute_topk_idxs,
                    )
                else:
                    compress_topk_idxs = None
            if is_prefill:
                if get_bool_env_var("USE_PA_PREFILL"):
                    topk_idxs = compress_topk_idxs
                else:
                    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
            else:
                if get_bool_env_var("USE_PA_DECODE"):
                    topk_idxs = compress_topk_idxs
                else:
                    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

        if topk_idxs is not None:
            topk_idxs = topk_idxs.int()

        # compress kv & attn
        if is_prefill:
            metadata = forward_batch.attn_backend.forward_metadata
            forward_batch.token_to_kv_pool.set_swa_buffer(
                self.attn_mha,
                metadata.swa_loc,
                kv[metadata.swa_loc_local],
                None,
            )
            if get_bool_env_var("USE_PA_PREFILL"):
                page_size = forward_batch.attn_backend.page_size
                num_pages = (forward_batch.seq_lens_cpu + (page_size - 1)) // page_size
                kv_pad = kv.new_zeros(
                    (num_pages.sum().item() * page_size, kv.shape[-1])
                )
                assert metadata.swa_kv_tobe_scatter_index is not None
                kv_pad[metadata.swa_kv_tobe_scatter_index] = kv
            else:
                kv_pad = kv
            if self.compress_ratio > 1:
                if (
                    kv_compress := self.compressor(
                        x, positions, forward_batch
                    )  # compress/state
                ) is not None:
                    kv_split_seq = forward_batch.seq_lens_cpu
                    kv_list = kv_pad.split(kv_split_seq.tolist(), dim=0)

                    assert isinstance(kv_compress, (list, tuple))
                    # prefill kv_cache = win_kv + compress_kv
                    if len(kv_compress) > 0:
                        assert len(kv_list) == len(kv_compress)
                        kv_pad = [
                            torch.cat(zipped_kv, dim=0)
                            for zipped_kv in zip(kv_list, kv_compress)
                        ]
                    else:
                        kv_pad = kv_list
            kv = kv_pad
        else:
            forward_batch.token_to_kv_pool.set_swa_buffer(
                self.attn_mha,
                forward_batch.attn_backend.forward_metadata.swa_loc,
                kv,
                None,
            )
            if self.compress_ratio > 1:
                self.compressor(x, positions, forward_batch)

        return q, kv, kv, forward_batch, positions, topk_idxs

    def forward_normal_core(self, q, k, v, forward_batch, positions, topk_idxs):
        attn_output = self.attn_mha(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=False,
            sinks=self.attn_sink,
            topk_indices=topk_idxs,
        )
        use_rope_partial_in_place_triton = get_bool_env_var(
            "USE_ROPE_PARTIAL_IN_PLACE_TRITON"
        ) and (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        )
        use_rope_partial_in_place_ascendc = get_bool_env_var(
            "USE_ROPE_PARTIAL_IN_PLACE_ASCENDC"
        )
        if use_rope_partial_in_place_triton:
            attn_output = triton_apply_rope_partial_in_place(
                attn_output,
                self.rotary_emb.inverse_position_sin_layer_cache,
                self.rotary_emb.position_cos_layer_cache,
            )
        elif use_rope_partial_in_place_ascendc:
            torch.ops.custom.inplace_partial_rotary_mul(
                attn_output.unsqueeze(2),
                self.rotary_emb.position_cos_layer_cache,
                self.rotary_emb.inverse_position_sin_layer_cache,
                rotary_mode="interleave",
                partial_slice=[self.qk_nope_head_dim, self.head_dim],
            )
        else:
            self.rotary_emb(
                attn_output[..., -self.qk_rope_head_dim :],
                self.rotary_emb.inverse_position_sin_layer_cache,
                self.rotary_emb.position_cos_layer_cache,
            )
        attn_output = attn_output.reshape(attn_output.shape[0], self.local_o_groups, -1)

        use_fused_transpose_batchmatmul = get_bool_env_var(
            "USE_FUSED_TRANSPOSE_BATCHMATMUL"
        )
        if use_fused_transpose_batchmatmul:
            attn_output = torch_npu.npu_transpose_batchmatmul(
                attn_output, self.wo_a.weight, perm_x1=(1, 0, 2), perm_y=(1, 0, 2)
            )
        else:
            wo_a = self.wo_a.weight.view(
                self.local_o_groups, self.o_lora_rank, -1
            )  # (G, R, D)
            attn_output = torch.einsum("sgd,grd->sgr", attn_output, wo_a)
        output = self.wo_b(attn_output.flatten(1))[0]
        return output

    def rebuild_cp_kv_cache(self, latent_cache, forward_batch, k_nope, k_pe):
        # support allgather+rerrange
        latent_cache[..., : self.kv_lora_rank] = k_nope.squeeze(1)
        latent_cache[..., self.kv_lora_rank :] = k_pe.squeeze(1)
        latent_cache_output = cp_all_gather_rerange_output(
            latent_cache.contiguous(),
            self.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )
        k_nope = latent_cache_output[..., : self.kv_lora_rank].unsqueeze(1)
        k_pe = latent_cache_output[..., self.kv_lora_rank :].unsqueeze(1)
        return k_nope, k_pe

    def _chunked_prefix_attn_mha(
        self,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            kv_indices = forward_batch.prefix_chunk_kv_indices[i]
            # Fetch latent cache from memory pool with precomputed chunked kv indices
            kv_a_normed, k_pe = self._get_mla_kv_buffer(
                kv_indices, q.dtype, forward_batch
            )
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            k = torch.empty(
                (
                    k_nope.shape[0],
                    self.num_local_heads,
                    self.qk_nope_head_dim + self.qk_rope_head_dim,
                ),
                dtype=v.dtype,
                device=v.device,
            )
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe

            output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse
            del kv, k, v, output, lse, tmp_output, tmp_lse

        return accum_output

    def forward_normal_chunked_kv_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        # In normal mha, the k and v tensors will become overly large when the prefix length is long.
        # To avoid this, we split the kv cache into chunks and process them one after another.
        # Since mha is compute friendly, the for loop induced here will not introduce significant overhead.
        # The top comments in https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
        # will be helpful for understanding the purpose of this function.

        # First do normal mha forward to get output for extended part
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def forward_normal_chunked_kv_core(self, q, k, v, forward_batch):
        has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(
            forward_batch.extend_prefix_lens_cpu
        )
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.prepare_chunked_prefix_cache_info(q.device)
            if hasattr(forward_batch.attn_backend, "init_mha_chunk_metadata"):
                forward_batch.attn_backend.init_mha_chunk_metadata(forward_batch)

        forward_batch.mha_return_lse = has_extend_prefix
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)

        # Do mha attention with chunked prefix cache if there are any sequence with prefix
        if has_extend_prefix:
            attn_output, lse = attn_output
            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_normal_one_shot_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        forward_batch.mha_one_shot = True
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def forward_normal_one_shot_core(self, q, k, v, forward_batch):
        has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.num_prefix_chunks = 0
            if hasattr(forward_batch.attn_backend, "init_mha_chunk_metadata"):
                forward_batch.attn_backend.init_mha_chunk_metadata(forward_batch)
        forward_batch.mha_return_lse = False
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        return self.forward_normal_core(q, k, v, forward_batch)


class DeepseekV4DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_global_server_args().speculative_algorithm
        )
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.self_attn = DeepseekV4AttentionMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_rope_head_dim=config.qk_rope_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            reduce_results=False,
            prefix=add_prefix("attn", prefix),
            alt_stream=alt_stream,
        )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = DeepseekV4MoE(
                config=config,
                quant_config=moe_quant_config_override or quant_config,
                prefix=add_prefix("ffn", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("ffn", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.nsa_enable_prefill_cp:
            self.layer_communicator = NSACPLayerCommunicator(
                layer_scatter_modes=self.layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=(
                    is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
                ),
            )
        else:
            self.layer_communicator = LayerCommunicator(
                layer_scatter_modes=self.layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=(
                    is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
                ),
            )

        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.norm_eps = config.rms_norm_eps
        self.hc_eps: float = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_mlp_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_mlp_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_mlp_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        torch.set_default_dtype(origin_dtype)

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        return is_nextn or (
            self.config.n_routed_experts is not None
            and layer_id >= self.config.first_k_dense_replace
            and layer_id % self.config.moe_layer_freq == 0
        )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        forward_batch,
    ):
        dtype = x.dtype
        use_fused_hc_pre_triton = get_bool_env_var("USE_FUSED_HC_PRE_TRITON")
        use_fused_hc_pre_ascendc = get_bool_env_var("USE_FUSED_HC_PRE_ASCENDC")
        if use_fused_hc_pre_ascendc:
            if forward_batch.forward_mode.is_idle():
                bs, hc, hdim = x.size()
                y = torch.empty(
                    bs,
                    hdim,
                    dtype=x.dtype,
                    device=x.device,
                )
                post = torch.empty(
                    bs,
                    hc,
                    dtype=torch.float32,
                    device=x.device,
                )
                comb = torch.empty(
                    bs,
                    hc,
                    hc,
                    dtype=torch.float32,
                    device=x.device,
                )
                return y, post, comb
            y, post, comb = torch.ops.custom.npu_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                hc_mult=self.hc_mult,
                hc_sinkhorn_iters=self.hc_sinkhorn_iters,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
            )
        else:
            # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b,s,d]
            if forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
                x = x.unsqueeze(0)  # [B*S, 4, 4096]  -> [1, S, 4 4096]
            elif forward_batch.forward_mode.is_decode_or_idle():
                x = x.unsqueeze(1)  # [B*S, 4, 4096]  -> [B, 1, 4 4096]
            else:
                raise ValueError
            shape, dtype = x.size(), x.dtype
            x = x.flatten(2).float()

            rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
            mixes = F.linear(x, hc_fn) * rsqrt

            if mixes.size(0) != 0 and use_fused_hc_pre_triton:
                raise ValueError
            else:
                pre, post, comb = hc_split_sinkhorn_torch(
                    mixes,
                    hc_scale,
                    hc_base,
                    self.hc_mult,
                    self.hc_sinkhorn_iters,
                    self.hc_eps,
                )
            y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)

            if forward_batch.forward_mode.is_extend():
                y = y.squeeze(0)  # [1, S, 4096] -> [S, 4096]
            elif forward_batch.forward_mode.is_decode_or_idle():
                y = y.squeeze(1)  # [B, 1, 4096]  -> [B, 4096]
            else:
                raise ValueError

        return y.to(dtype), post, comb

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        forward_batch,
    ):
        use_fused_hc_post_triton = get_bool_env_var("USE_FUSED_HC_POST_TRITON")
        use_fused_hc_post_ascendc = get_bool_env_var("USE_FUSED_HC_POST_ASCENDC")
        if use_fused_hc_post_triton and x.size(0) != 0:
            raise ValueError
        elif use_fused_hc_post_ascendc:
            if forward_batch.forward_mode.is_idle():
                bs, hdim = x.size()
                y = torch.empty(
                    bs,
                    self.hc_mult,
                    hdim,
                    dtype=x.dtype,
                    device=x.device,
                )
                return y

            # TODO need to check all_gather_last_layer_res()
            is_prefill = (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_draft_extend_v2()
                and not forward_batch.forward_mode.is_draft_extend()
                and not forward_batch.forward_mode.is_target_verify()
            )

            if is_prefill:
                x = x.unsqueeze(0)
                residual = residual.unsqueeze(0)
                post = post.unsqueeze(0) if post.dim() == 2 else post
                comb = comb.unsqueeze(0) if comb.dim() == 3 else comb

            if (
                forward_batch.forward_mode.is_draft_extend_v2()
                or forward_batch.forward_mode.is_draft_extend()
                or forward_batch.forward_mode.is_target_verify()
            ):
                speculative_num_draft_tokens = (
                    forward_batch.attn_backend.speculative_num_draft_tokens
                )
                x = x.view(
                    x.shape[0] // speculative_num_draft_tokens,
                    speculative_num_draft_tokens,
                    *x.shape[1:],
                )
                residual = residual.view(
                    residual.shape[0] // speculative_num_draft_tokens,
                    speculative_num_draft_tokens,
                    *residual.shape[1:],
                )
                post = post.view(-1, 4)
                comb = comb.view(-1, 4, 4)
                post = post.view(
                    post.shape[0] // speculative_num_draft_tokens,
                    speculative_num_draft_tokens,
                    post.shape[-1],
                )
                comb = comb.view(
                    comb.shape[0] // speculative_num_draft_tokens,
                    speculative_num_draft_tokens,
                    *comb.shape[-2:],
                )

            if forward_batch.forward_mode.is_decode():
                x = x.unsqueeze(1)
                residual = residual.unsqueeze(1)
                post = post.unsqueeze(1) if post.dim() == 2 else post
                comb = comb.unsqueeze(1) if comb.dim() == 3 else comb

            # input:
            # x [b, s, d] bf16
            # residual [b, s, hc, d] bf16
            # post [b, s, hc] float32
            # comb [b, s, hc, hc] float32

            # output:
            # y [b, s, hc, d]
            y = torch.ops.custom.npu_hc_post(
                x,
                residual,
                post,
                comb,
            )

            y = y.reshape(-1, y.size(2), y.size(3))
        else:
            # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
            if forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
                residual = residual.unsqueeze(
                    0
                )  # prefill [S, 4, 4096]  -> [1, S, 4, 4096];
                x = x.unsqueeze(0)  # prefill [S, 4096]  -> [1, S, 4096];
            elif forward_batch.forward_mode.is_decode_or_idle():
                residual = residual.unsqueeze(
                    1
                )  # decode [B, 4, 4096]  -> [B, 1, 4, 4096];
                x = x.unsqueeze(1)  # decode [B, 4096]  -> [B, 1, 4096];
            else:
                raise ValueError

            y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
                comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
            )

            if forward_batch.forward_mode.is_extend():
                y = y.squeeze(0)  # [1, S, 4, 4096] -> [S, 4, 4096]
            elif forward_batch.forward_mode.is_decode_or_idle():
                y = y.reshape(-1, y.size(2), y.size(3))  # [T, 4, 4096]
            else:
                raise ValueError

        return y.type_as(x)

    def scatter_first_layer_res(self, residual, post, comb, forward_batch):
        """post: [1, 16, 4] -> [1, 1, 4], comb: [1, 16, 4, 4] -> [1, 1, 4, 4]"""
        residual = residual.tensor_split(self.layer_communicator._context.attn_tp_size)[
            self.layer_communicator._context.attn_tp_rank
        ]
        if forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            post = (
                post.squeeze(0)
                .tensor_split(self.layer_communicator._context.attn_tp_size)[
                    self.layer_communicator._context.attn_tp_rank
                ]
                .unsqueeze(0)
            )  # [1, 16, 4]
            comb = (
                comb.squeeze(0)
                .tensor_split(self.layer_communicator._context.attn_tp_size)[
                    self.layer_communicator._context.attn_tp_rank
                ]
                .unsqueeze(0)
            )
        elif forward_batch.forward_mode.is_decode_or_idle():
            post = (
                post.squeeze(1)
                .tensor_split(self.layer_communicator._context.attn_tp_size)[
                    self.layer_communicator._context.attn_tp_rank
                ]
                .unsqueeze(1)
            )  # [1, 16, 4]
            comb = (
                comb.squeeze(1)
                .tensor_split(self.layer_communicator._context.attn_tp_size)[
                    self.layer_communicator._context.attn_tp_rank
                ]
                .unsqueeze(1)
            )
        else:
            raise ValueError
        return residual, post, comb

    def all_gather_last_layer_res(self, residual, post, comb, forward_batch):
        """res: [1, 4, 4096] -> [16, 4, 4096]; post: [1, 1, 4] -> [1, 16, 4]; comb: [1, 1, 4, 4] -> [1, 16, 4, 4]"""
        if forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            res_out = (
                torch.empty_like(residual)
                .squeeze(0)
                .repeat(self.layer_communicator._context.attn_tp_size, 1, 1)
            )
            post_out = (
                torch.empty_like(post)
                .squeeze(0)
                .repeat(self.layer_communicator._context.attn_tp_size, 1)
            )
            comb_out = (
                torch.empty_like(comb)
                .squeeze(0)
                .repeat(self.layer_communicator._context.attn_tp_size, 1, 1)
            )
            residual, residual_local = (res_out, residual.squeeze(0))
            post, post_local = (post_out, post.squeeze(0))
            comb, comb_local = (comb_out, comb.squeeze(0))
        elif forward_batch.forward_mode.is_decode_or_idle():
            res_out = (
                torch.empty_like(residual)
                .squeeze(1)
                .repeat(self.layer_communicator._context.attn_tp_size, 1, 1)
            )
            post_out = (
                torch.empty_like(post)
                .squeeze(1)
                .repeat(self.layer_communicator._context.attn_tp_size, 1)
            )
            comb_out = (
                torch.empty_like(comb)
                .squeeze(1)
                .repeat(self.layer_communicator._context.attn_tp_size, 1, 1)
            )
            residual, residual_local = (res_out, residual.squeeze(1))
            post, post_local = (post_out, post.squeeze(1))
            comb, comb_local = (comb_out, comb.squeeze(1))
        else:
            raise ValueError

        attn_tp_all_gather_into_tensor(
            residual,
            residual_local,
        )
        attn_tp_all_gather_into_tensor(
            post,
            post_local,
        )
        attn_tp_all_gather_into_tensor(
            comb,
            comb_local,
        )
        if forward_batch.forward_mode.is_extend():
            post = post.unsqueeze(0)
            comb = comb.unsqueeze(0)
        else:
            post = post.unsqueeze(1)
            comb = comb.unsqueeze(1)
        return residual, post, comb

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        gemm_output_zero_allocator: BumpAllocator = None,
    ) -> torch.Tensor:
        quant_format = (
            "mxfp4"
            if (
                _is_gfx95_supported
                and getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None)
                is not None
                and getattr(self.self_attn.fused_qkv_a_proj_with_mqa, "weight", None)
                is not None
                and self.self_attn.fused_qkv_a_proj_with_mqa.weight.dtype == torch.uint8
            )
            else (
                "fp8"
                if (
                    _is_gfx95_supported
                    and getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None)
                    is not None
                    and getattr(
                        self.self_attn.fused_qkv_a_proj_with_mqa, "weight", None
                    )
                    is not None
                    and self.self_attn.fused_qkv_a_proj_with_mqa.weight.dtype
                    == getattr(torch, "float8_e4m3fn", None)
                )
                else ""
            )
        )
        from sglang.srt.layers.communicator import ScatterMode

        scatter_modes = self.layer_communicator.layer_scatter_modes
        residual = hidden_states  # [B*S, 4, H]
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            forward_batch,
        )  # [16, 4096]

        hidden_states, _ = self.layer_communicator.prepare_attn(
            hidden_states,
            None,
            forward_batch,
            quant_format,
        )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        if get_moe_a2a_backend().is_deepep():
            if scatter_modes.layer_input_mode == ScatterMode.SCATTERED and (
                scatter_modes.layer_output_mode == ScatterMode.SCATTERED
                or scatter_modes.layer_output_mode == ScatterMode.TP_ATTN_FULL
            ):

                hidden_states, _ = self.layer_communicator.prepare_mlp(
                    hidden_states, None, forward_batch, skip_layernorm=True
                )  # [1, 4096]

                hidden_states = self.hc_post(
                    hidden_states, residual, post, comb, forward_batch
                )  # [B*seq, 4, 4096]

                residual = hidden_states  # [B*seq, 4, 4096]
                hidden_states, post, comb = self.hc_pre(
                    hidden_states,
                    self.hc_mlp_fn,
                    self.hc_mlp_scale,
                    self.hc_mlp_base,
                    forward_batch,
                )
                hidden_states = self.post_attention_layernorm(hidden_states)

            else:  # only first layer
                tp_size = self.layer_communicator._context.attn_tp_size
                hidden_states = self.hc_post(
                    hidden_states, residual / tp_size, post, comb, forward_batch
                )  # [B*seq, 4, 4096], TODO: residual can only added by the rank 0 in TP group

                hidden_states, _ = self.layer_communicator.prepare_mlp(
                    hidden_states, None, forward_batch, skip_layernorm=True
                )  # [1, 4096]

                residual = hidden_states  # [B*seq, 4, 4096]
                hidden_states, post, comb = self.hc_pre(
                    hidden_states,
                    self.hc_mlp_fn,
                    self.hc_mlp_scale,
                    self.hc_mlp_base,
                    forward_batch,
                )
                hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            hidden_states = self.hc_post(
                hidden_states, residual, post, comb, forward_batch
            )  # [B*seq, 4, 4096]
            residual = hidden_states  # [B*seq, 4, 4096]
            hidden_states, post, comb = self.hc_pre(
                hidden_states,
                self.hc_mlp_fn,
                self.hc_mlp_scale,
                self.hc_mlp_base,
                forward_batch,
            )
            hidden_states, _ = self.layer_communicator.prepare_mlp(
                hidden_states, None, forward_batch
            )  # [1, 4096]

        if (
            scatter_modes.layer_input_mode == ScatterMode.TP_ATTN_FULL
            and scatter_modes.mlp_mode == ScatterMode.SCATTERED
            and scatter_modes.layer_output_mode != ScatterMode.SCATTERED
        ):
            """ATTN TP -> MOE DP+EP"""
            residual, post, comb = self.scatter_first_layer_res(
                residual, post, comb, forward_batch
            )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        # Get input_ids after dp gather
        if (
            scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            and scatter_modes.mlp_mode == ScatterMode.FULL
            and self.layer_communicator._context.attn_dp_size > 1
        ):
            input_ids = forward_batch.input_ids.view(-1, 1)
            input_ids_dp_full = get_global_dp_id_buffer(dtype=input_ids.dtype)
            dp_gather_partial(input_ids_dp_full, input_ids, forward_batch)
            input_ids_dp_full = input_ids_dp_full.squeeze(-1)
            assert input_ids_dp_full.shape[0] == hidden_states.shape[0]
        elif scatter_modes.mlp_mode == ScatterMode.SCATTERED:
            input_ids = forward_batch.input_ids
            tp_rank = self.layer_communicator._context.attn_tp_rank
            token_per_rank = (
                input_ids.shape[0] // self.layer_communicator._context.attn_tp_size
            )
            input_ids_dp_full = input_ids[
                token_per_rank * tp_rank : token_per_rank * (tp_rank + 1)
            ]
        else:
            input_ids_dp_full = forward_batch.input_ids

        if isinstance(self.mlp, DeepseekV4MLP):
            gemm_output_zero_allocator = None

        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            should_allreduce_fusion,
            use_reduce_scatter,
            gemm_output_zero_allocator,
            input_ids_dp_full=input_ids_dp_full,
        )

        if not self.nsa_enable_prefill_cp and should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True

        if not should_allreduce_fusion:  # True
            hidden_states, _ = self.layer_communicator.postprocess_layer(
                hidden_states, None, forward_batch
            )

        if (
            scatter_modes.middle_residual_mode == ScatterMode.SCATTERED
            and scatter_modes.layer_output_mode == ScatterMode.TP_ATTN_FULL
        ):
            """MOE DP+EP -> ATTN TP, needs allgather here"""
            residual, post, comb = self.all_gather_last_layer_res(
                residual, post, comb, forward_batch
            )

        hidden_states = self.hc_post(hidden_states, residual, post, comb, forward_batch)

        return hidden_states, residual

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                zero_allocator=zero_allocator,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_mlp(self, state):
        hidden_states = state.pop("hidden_states_mlp_input")
        if not (
            enable_moe_dense_fully_dp()
            and (not self.is_layer_sparse)
            and hidden_states.shape[0] == 0
        ):
            state.hidden_states_mlp_output = self.mlp(
                hidden_states, state.forward_batch
            )
        else:
            state.hidden_states_mlp_output = hidden_states

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "zero_allocator",
                "tbo_subbatch_index",
            }
        )
        return output


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.pp_group = get_pp_group()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        self.hc_mult = config.hc_mult
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_tp_size()
        else:
            self.cp_size = None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = (
            torch.cuda.Stream()
            if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            else None
        )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
            offloader_kwargs=dict(
                submodule_accessor=lambda layer: (
                    layer.mlp.experts
                    if isinstance(layer.mlp, DeepseekV4MoE)
                    else layer.mlp
                ),
                whitelist_param_names_creator=lambda module: (
                    [
                        "w13_weight",
                        "w2_weight",
                        # only for nvfp4
                        *(
                            [
                                "w13_blockscale_swizzled",
                                "w2_blockscale_swizzled",
                            ]
                            if hasattr(module, "w13_blockscale_swizzled")
                            else []
                        ),
                    ]
                    if isinstance(module, FusedMoE)
                    else []
                ),
            ),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        self.gemm_output_zero_allocator_size = 0
        if (
            _use_aiter_gfx95
            and config.n_routed_experts == 256
            and self.embed_tokens.embedding_dim == 7168
        ):
            num_moe_layers = sum(
                [
                    1
                    for i in range(len(self.layers))
                    if isinstance(self.layers[i].mlp, DeepseekV4MoE)
                ]
            )

            allocate_size = 0
            for i in range(len(self.layers)):
                if isinstance(self.layers[i].mlp, DeepseekV4MoE):
                    tp_size = get_tensor_model_parallel_world_size()
                    intermediate_size = (
                        config.moe_intermediate_size * config.n_shared_experts
                    )
                    share_expert_output_size_per_partition = divide(
                        intermediate_size * 2, tp_size
                    )
                    allocate_size = share_expert_output_size_per_partition
                    break

            self.gemm_output_zero_allocator_size = (
                get_dsv3_gemm_output_zero_allocator_size(
                    config.n_routed_experts,
                    num_moe_layers,
                    allocate_size,
                    self.embed_tokens.embedding_dim,
                )
            )
        self.layers_to_capture = []
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.enable_a2a_moe = True
        else:
            self.enable_a2a_moe = False

        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.hc_eps)
        mixes = F.linear(x, hc_fn) * rsqrt  # [16, 4]
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps  # [16, 4]
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        total_num_layers = self.end_layer - self.start_layer
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        has_gemm_output_zero_allocator = hasattr(
            self, "gemm_output_zero_allocator_size"
        )

        gemm_output_zero_allocator = (
            BumpAllocator(
                buffer_size=self.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=device,
            )
            if has_gemm_output_zero_allocator
            and self.gemm_output_zero_allocator_size > 0
            else None
        )

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(
                    input_ids
                )  # [seq, hidden_size], h = self.embed(input_ids)
                hidden_states = hidden_states.unsqueeze(1).repeat(
                    1, self.hc_mult, 1
                )  # [bs*seq, 4, hidden_size]
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        if nsa_use_prefill_cp(forward_batch):
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        normal_start_layer = self.start_layer
        normal_end_layer = self.end_layer
        if forward_batch.can_run_tbo:
            if (
                self.first_k_dense_replace > normal_start_layer
                and self.first_k_dense_replace < normal_end_layer
            ):
                normal_end_layer = self.first_k_dense_replace
            elif self.first_k_dense_replace < normal_start_layer:
                normal_end_layer = normal_start_layer = 0
        aux_hidden_states = []
        for i in range(normal_start_layer, normal_end_layer):
            # NOTE: torch dynamo does not support graph break in context manager
            ctx = (
                nullcontext()
                if not get_global_server_args().disable_piecewise_cuda_graph
                else get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:
                if i in self.layers_to_capture:
                    if self.enable_a2a_moe and i > self.first_k_dense_replace:
                        aux_hidden_state = tensor_model_parallel_all_gather(
                            hidden_states + residual, dim=0
                        )
                        aux_hidden_states.append(aux_hidden_state)
                    else:
                        aux_hidden_states.append(hidden_states + residual)
                layer = self.layers[i]
                hidden_states, _ = layer(
                    positions,
                    hidden_states,  # [bs*seq, 4, hidden_size]
                    forward_batch,
                    residual,
                    zero_allocator,
                    gemm_output_zero_allocator,
                )

        if normal_end_layer != self.end_layer:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers[normal_end_layer : self.end_layer],
                enable_tbo=True,
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
                input_data_scatter_mode=self.layers[
                    normal_end_layer - 1
                ].layer_scatter_modes.layer_output_mode,
                zero_allocator=zero_allocator,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                hidden_states = self.hc_head(
                    hidden_states,
                    self.hc_head_fn,
                    self.hc_head_scale,
                    self.hc_head_base,
                )
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states = hidden_states[:, 0, :].contiguous()

        if self.pp_group.is_last_rank and nsa_use_prefill_cp(forward_batch):
            # allgather + rerrange
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class DeepseekV4ForCausalLM(nn.Module):
    # for quark model load
    packed_modules_mapping = {}

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        log_info_on_rank0(logger, f"{config.__dict__=}")
        # for quark model load
        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.pp_group = get_pp_group()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        li_kv_dtype_int8 = get_bool_env_var("LI_KV_DTYPE_INT8")
        if li_kv_dtype_int8:
            li_kv_dtype = "int8"
            quant_config.li_kv_dtype = li_kv_dtype
        else:
            li_kv_dtype = "bf16"
            if quant_config is not None:
                if hasattr(quant_config, "quant_description"):
                    quant_description = quant_config.quant_description
                else:
                    quant_description = quant_config.config
                li_cache_scheme = quant_description.get("li_cache_scheme")
                if li_cache_scheme is not None:
                    logger.info(f"{li_cache_scheme=}")
                    li_kv_dtype = str(li_cache_scheme.get("type", "")) + str(
                        li_cache_scheme.get("num_bits", "")
                    )
                quant_config.li_kv_dtype = li_kv_dtype

        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.use_nsa = is_deepseek_nsa(config)
        self.model = DeepseekV4Model(
            config, quant_config  # , prefix=add_prefix("model", prefix)
        )
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                )
                self.norm_eps: float = config.rms_norm_eps
                self.norm = RMSNorm(config.hidden_size, self.norm_eps)
                self.hc_eps: float = config.hc_eps
                self.hc_mult = hc_mult = config.hc_mult
                hc_dim = hc_mult * config.hidden_size
                origin_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
                self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
                self.hc_head_scale = nn.Parameter(torch.empty(1))
                torch.set_default_dtype(origin_dtype)
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, DeepseekV4MoE)
            }
        )
        self.capture_aux_hidden_states = False

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_rank = get_attention_tp_rank()
            self.cp_size = get_attention_tp_size()
        else:
            self.cp_rank = self.cp_size = None

        q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
        get_attn_tp_context().init_context(q_lora_rank, is_deepseek_nsa(config))

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(
        self, architecture: str = "DeepseekV3ForCausalLM"
    ):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        # Only Deepseek V3/R1 can use shared experts fusion optimization now.
        disable_reason = None
        if (
            self.config.architectures[0] != architecture
            or self.config.n_routed_experts != 256
            or self.config.n_shared_experts != 1
        ):
            disable_reason = "Config not support fused shared expert(s)."
        elif (not _is_cuda or torch.cuda.get_device_capability("cuda") < (8, 0)) and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = (
                "Only Deepseek V3/R1 on NV-platform with capability >= 80 "
                "or AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization."
            )
        elif get_moe_expert_parallel_world_size() > 1 and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = "Only Deepseek V3/R1 on AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization under expert parallelism."
        elif disable_reason is None and get_moe_a2a_backend().is_deepep():
            disable_reason = "Deepseek V3/R1 can not use shared experts fusion optimization under deepep expert parallelism."
        elif self.quant_config and self.quant_config.get_name() == "w4afp8":
            disable_reason = "Deepseek V3/R1 W4AFP8 model uses different quant method for routed experts and shared experts."

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.nsa_enable_prefill_cp:
            if can_cp_split(len(input_ids), self.cp_size, self.use_nsa, forward_batch):
                forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def post_load_weights(self, is_nextn=False, weight_names=None):

        # Perform post-processing after loading weights
        if is_nextn:
            layer_ids = [self.config.num_hidden_layers]
        else:
            if weight_names is None:
                layer_ids = range(self.model.start_layer, self.model.end_layer)
            else:
                layer_ids = set()
                for name in weight_names:
                    if "kv_b_proj" in name:
                        layer_id = int(name.split(".")[2])
                        if layer_id < self.config.num_hidden_layers:
                            layer_ids.add(layer_id)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        weights = self._maybe_quant_weights_to_fp8_ue8m0(
            weights, NVFP4_CKPT_FP8_ATTN_QUANT_MODULES, is_nextn
        )

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )
        # Params for special naming rules in mixed-precision models, for example:
        # model.layers.xx.mlp.experts.xx.w1.input_scale. For details,
        # see https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8/blob/main.
        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None
        cached_eh_proj = {}

        if is_nextn:
            nextn_layer_prefix = f"model.mtp.0"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "enorm",
                "hnorm",
                ".norm.",  # todo，把norm给shard_head.norm了
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
                "emb.tok_emb",
                ".head.",
                "e_proj",
                "h_proj",
                "eh_proj",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        params_dict = dict(self.named_parameters())
        # if torch.distributed.get_rank() == 0:
        #     for model_name, model_tensor in params_dict.items():
        #         print(f"{model_name=}, {model_tensor.shape=}, {model_tensor.dtype=}", flush=True)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            weight_names = []
            for name, loaded_weight in weights:
                # log_info_on_rank0(logger, f"{name=}, {loaded_weight.shape=}")
                # continue
                use_async_loading = should_async_load(loaded_weight)
                name = "model." + name
                name = name.replace("ffn", "mlp")
                name = name.replace("gate.bias", "gate.e_score_correction_bias")
                layer_id = get_layer_id(name)
                if (
                    layer_id is not None
                    and hasattr(self.model, "start_layer")
                    and (
                        layer_id < self.model.start_layer
                        or layer_id >= self.model.end_layer
                    )
                ):
                    continue
                if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                    name = name.replace(
                        "mlp.shared_experts",
                        f"mlp.experts.{self.config.n_routed_experts}",
                    )

                weight_names.append(name)

                if not is_nextn:
                    if hasattr(self.config, "num_nextn_predict_layers"):
                        num_nextn_layers = self.config.num_nextn_predict_layers
                        if num_nextn_layers > 0 and name.startswith("model.layers"):
                            name_list = name.split(".")
                            if (
                                len(name_list) >= 3
                                and int(name_list[2]) >= self.config.num_hidden_layers
                            ):
                                continue
                    if "mtp." in name:
                        continue
                else:
                    if not name.startswith(nextn_layer_prefix):
                        continue

                    # Use shared head and embed weights from target model
                    if ".head." in name or "emb.tok_emb." in name:
                        continue

                    is_decoder = True
                    # For nextn specific weights
                    for weight_name in nextn_spec_weight_names:
                        if weight_name in name:
                            name = name.replace(nextn_layer_prefix, "model")
                            is_decoder = False
                            break
                    # For decoder layer weights
                    if is_decoder:
                        name = name.replace(nextn_layer_prefix, "model.decoder")

                if "rotary_emb.inv_freq" in name:
                    continue

                if "experts." in name:
                    # logger.warning(f"3786 {weight_name=}, {param_name=}, {expert_id=}, {shard_id=}, {name=}")
                    if "w1.weight" in name:
                        name = name.replace("w1.", "gate_proj.")
                    elif "w2.weight" in name:
                        name = name.replace("w2.", "down_proj.")
                    elif "w3.weight" in name:
                        name = name.replace("w3.", "up_proj.")
                if "embed.weight" in name:
                    name = name.replace("embed.", "embed_tokens.")
                if is_nextn and ".norm.weight" in name:
                    name = name.replace(".norm.", ".shared_head.norm.")
                if ".attn." in name:
                    name = name.replace(".attn.", ".self_attn.")
                if ".attn_norm." in name:
                    name = name.replace(".attn_norm.", ".input_layernorm.")
                if ".mlp_norm." in name:
                    name = name.replace(".mlp_norm.", ".post_attention_layernorm.")
                if "model.head." in name:
                    name = name.replace("model.head.", "lm_head.")

                for param_name, weight_name, shard_id in stacked_params_mapping:
                    # Skip non-stacked layers and experts (experts handled below).
                    if weight_name not in name:
                        continue
                    if _is_npu:
                        name = name.replace("weight_packed", "weight")
                    # We have mlp.experts[0].gate_proj in the checkpoint.
                    # Since we handle the experts below in expert_params_mapping,
                    # we need to skip here BEFORE we update the name, otherwise
                    # name will be updated to mlp.experts[0].gate_up_proj, which
                    # will then be updated below in expert_params_mapping
                    # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                    if ("mlp.experts." in name) and name not in params_dict:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    maybe_executor_submit(
                        executor=executor,
                        futures=futures,
                        use_async=use_async_loading,
                        func=weight_loader,
                        func_args=(param, loaded_weight, shard_id),
                    )
                    break
                else:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        name = name.replace(weight_name, param_name)
                        if name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(
                                param,
                                loaded_weight,
                                name,
                            ),
                            func_kwargs={
                                "shard_id": shard_id,
                                "expert_id": expert_id,
                            },
                        )
                        break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        # Skip loading embed_tokens if not first rank in pipeline parallelism
                        if ".embed_tokens." in name and not self.pp_group.is_first_rank:
                            continue
                        # Skip loading norm if not last rank in pipeline parallelism
                        if ".norm." in name and not self.pp_group.is_last_rank:
                            continue
                        if fuse_qkv_a_proj and (
                            "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                        ):
                            cached_a_proj[name] = loaded_weight
                            q_a_proj_name = (
                                name
                                if "q_a_proj" in name
                                else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                            )
                            kv_a_proj_name = (
                                name
                                if "kv_a_proj_with_mqa" in name
                                else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                            )

                            # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                            if (
                                q_a_proj_name in cached_a_proj
                                and kv_a_proj_name in cached_a_proj
                            ):
                                q_a_proj_weight = cached_a_proj[q_a_proj_name]
                                kv_a_proj_weight = cached_a_proj[kv_a_proj_name]

                                if q_a_proj_weight.shape == torch.Size(
                                    []
                                ) and kv_a_proj_weight.shape == torch.Size([]):
                                    fused_weight = q_a_proj_weight
                                else:
                                    cat_dim = 0
                                    if self.quant_config is not None and (
                                        self.quant_config.get_name() == "awq"
                                        or self.quant_config.get_name() == "awq_marlin"
                                        or self.quant_config.get_name() == "moe_wna16"
                                    ):
                                        cat_dim = 1

                                    fused_weight = torch.cat(
                                        [q_a_proj_weight, kv_a_proj_weight], dim=cat_dim
                                    )

                                param_name = (
                                    name.replace(
                                        "q_a_proj", "fused_qkv_a_proj_with_mqa"
                                    )
                                    if "q_a_proj" in name
                                    else name.replace(
                                        "kv_a_proj_with_mqa",
                                        "fused_qkv_a_proj_with_mqa",
                                    )
                                )
                                param = params_dict[param_name]

                                weight_loader = getattr(
                                    param, "weight_loader", default_weight_loader
                                )
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, fused_weight),
                                )
                                cached_a_proj.pop(q_a_proj_name)
                                cached_a_proj.pop(kv_a_proj_name)
                        elif (
                            is_nextn
                            and self.model.eh_proj is not None
                            and (".e_proj." in name or ".h_proj." in name)
                        ):
                            cached_eh_proj[name] = loaded_weight
                            if len(cached_eh_proj) == 2:
                                cached_eh_proj = dict(sorted(cached_eh_proj.items()))
                                eh_proj_weight = torch.cat(
                                    list(cached_eh_proj.values()), dim=1
                                )
                                param = params_dict["model.eh_proj.weight"]
                                weight_loader = getattr(
                                    param, "weight_loader", default_weight_loader
                                )
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, eh_proj_weight),
                                )
                                cached_eh_proj.clear()
                        else:
                            if name not in params_dict:
                                logger.warning(f"{name} not found in params_dict.")
                                continue
                            param = params_dict[name]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(param, loaded_weight),
                            )

            # Wait for all tasks to complete and raise any exceptions.
            for future in concurrent.futures.as_completed(futures):
                future.result()

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            # we plus 1 here because in sglang, for the ith layer, it takes the output
            # of the (i-1)th layer as aux hidden state
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    # Mark the ue8m0 flag of nextn moe weights as True to avoid requantization
    def _mark_nextn_moe_weights_as_ue8m0(self):
        experts = self.model.decoder.mlp.experts
        w13_scale = (
            experts.w13_weight_scale_inv
            if hasattr(experts, "w13_weight_scale_inv")
            else experts.w13_weight_scale
        )
        w2_scale = (
            experts.w2_weight_scale_inv
            if hasattr(experts, "w2_weight_scale_inv")
            else experts.w2_weight_scale
        )
        w13_scale.format_ue8m0 = True
        w2_scale.format_ue8m0 = True

    def _maybe_quant_weights_to_fp8_ue8m0(
        self, weights, attn_quant_modules, is_nextn=False
    ):
        # Quantize some weights to fp8 ue8m0 for DeepSeek nvfp4 checkpoint
        partial_names = []
        nextn_layer_id = (
            0 if self.config.num_hidden_layers == 1 else self.config.num_hidden_layers
        )
        weights_dict = dict(weights)
        weight_block_size = [128, 128]

        if envs.SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN.get():
            layer_ids = (
                list(range(self.config.num_hidden_layers))
                if not is_nextn
                else [nextn_layer_id]
            )
            for layer_id in layer_ids:
                for stem in attn_quant_modules:
                    partial_names.append(f"model.layers.{layer_id}.self_attn.{stem}")

        if is_nextn and enable_nextn_moe_bf16_cast_to_fp8(self.quant_config):
            for expert_sub_name in [
                "shared_experts",
                *[
                    f"experts.{expert_id}"
                    for expert_id in range(self.config.n_routed_experts)
                ],
            ]:
                for stem in [
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]:
                    partial_names.append(
                        f"model.layers.{nextn_layer_id}.mlp.{expert_sub_name}.{stem}"
                    )

        if len(partial_names) > 0:
            for partial_name in tqdm.tqdm(
                partial_names,
                desc="quant weights to fp8 ue8m0",
            ):
                original_weight = weights_dict[f"{partial_name}.weight"]
                out_w, out_s = quant_weight_ue8m0(
                    original_weight, weight_block_size=weight_block_size
                )
                weights_dict[f"{partial_name}.weight"] = out_w
                weights_dict[f"{partial_name}.weight_scale_inv"] = out_s

        if is_nextn and enable_nextn_moe_bf16_cast_to_fp8(self.quant_config):
            self._mark_nextn_moe_weights_as_ue8m0()

        return list(weights_dict.items())


AttentionBackendRegistry.register("ascend", handle_attention_ascend)
# AttentionBackendRegistry.register("flashinfer", handle_attention_flashinfer)
# AttentionBackendRegistry.register("fa3", handle_attention_fa3)
# AttentionBackendRegistry.register("flashmla", handle_attention_flashmla)
# AttentionBackendRegistry.register("cutlass_mla", handle_attention_cutlass_mla)
# AttentionBackendRegistry.register("fa4", handle_attention_fa4)
# AttentionBackendRegistry.register("trtllm_mla", handle_attention_trtllm_mla)
# AttentionBackendRegistry.register("aiter", handle_attention_aiter)
# AttentionBackendRegistry.register("nsa", handle_attention_nsa)
# AttentionBackendRegistry.register("triton", handle_attention_triton)


EntryClass = [DeepseekV4ForCausalLM]
