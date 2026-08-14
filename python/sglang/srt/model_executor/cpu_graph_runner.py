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
"""Run the model with cpu torch compile."""

# The implementation of CPUGraphRunner follows the CudaGraphRunner

from __future__ import annotations

import bisect
import copy
import itertools
import logging
import os
import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Optional, Union

import psutil
import torch
import torch.nn.functional as F
import tqdm

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    enable_num_token_non_padded,
    get_required_capture_hidden_mode,
    get_server_return_hidden_states_mode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.model_loader.utils import resolve_language_model
from sglang.srt.runtime_context import get_flags, get_parallel, get_spec
from sglang.srt.utils import (
    empty_context,
    get_bool_env_var,
    log_info_on_rank0,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_compile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# skip_cross_attention capture-mode helpers (CPU graph only)
# ---------------------------------------------------------------------------
# When CPUGraphRunner captures two graphs per batch size (one with cross-
# attention, one without), it uses this context variable so that
# encoder-decoder models (e.g. mllama) receive a compile-time-constant value
# for skip_cross_attention instead of a data-dependent branch to avoid recompiles.

_capture_skip_cross_attention: Optional[bool] = None
_compile_wrapper_counter = itertools.count()
_DYNAMIC_DIM_MAX = 2**31 - 1


def _compile_wrapper_forward(self, *args, **kwargs):
    with torch.no_grad():
        if self.model_forward is None:
            return self.model(*args, **kwargs)
        return self.model_forward(*args, **kwargs)


def _make_compile_wrapper(
    model: torch.nn.Module, model_forward: Optional[Callable] = None
) -> torch.nn.Module:
    wrapper_id = next(_compile_wrapper_counter)
    forward_code = _compile_wrapper_forward.__code__.replace(
        co_firstlineno=wrapper_id + 1
    )
    forward = types.FunctionType(
        forward_code,
        globals(),
        f"cpu_graph_forward_{wrapper_id}",
    )
    wrapper_type = type(
        f"CPUGraphCompileWrapper{wrapper_id}",
        (torch.nn.Module,),
        {"forward": forward},
    )
    wrapper = wrapper_type()
    wrapper.model = model
    wrapper.model_forward = model_forward
    return wrapper


@contextmanager
def _disable_duck_shape():
    try:
        import torch.fx.experimental._config as fx_config
    except ImportError:
        yield
        return

    previous = fx_config.use_duck_shape
    fx_config.use_duck_shape = False
    try:
        yield
    finally:
        fx_config.use_duck_shape = previous


@contextmanager
def _disable_cpu_graph_custom_op_aliasing_check():
    try:
        import torch._functorch.config as functorch_config
    except ImportError:
        yield
        return

    if not hasattr(functorch_config, "check_custom_op_aliasing"):
        yield
        return

    with functorch_config.patch(check_custom_op_aliasing=False):
        yield


def get_capture_skip_cross_attention() -> Optional[bool]:
    """Return the active skip_cross_attention override, or None if not set."""
    return _capture_skip_cross_attention


@contextmanager
def capture_with_skip_cross_attention(skip: bool):
    """Pin skip_cross_attention to *skip* for the duration of the context."""
    global _capture_skip_cross_attention
    previous = _capture_skip_cross_attention
    _capture_skip_cross_attention = skip
    try:
        yield
    finally:
        _capture_skip_cross_attention = previous


def _mark_dynamic(*tensors: Optional[torch.Tensor]) -> None:
    """Mark dim 0 of each tensor dynamic."""
    for t in tensors:
        if t is not None:
            if t.ndim > 0 and t.shape[0] == 1:
                torch._dynamo.maybe_mark_dynamic(t, 0)
            else:
                torch._dynamo.mark_dynamic(t, 0, min=1, max=_DYNAMIC_DIM_MAX)


def _mark_dynamic_axis(tensor: Optional[torch.Tensor], dim: int) -> None:
    if tensor is not None:
        if tensor.ndim > 0 and tensor.shape[dim] == 1:
            torch._dynamo.maybe_mark_dynamic(tensor, dim)
        else:
            torch._dynamo.mark_dynamic(tensor, dim, min=1, max=_DYNAMIC_DIM_MAX)


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
    dynamic: bool = False,
    model_forward: Optional[Callable] = None,
):
    """Patch the model to make it compatible with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            with _disable_duck_shape(), _disable_cpu_graph_custom_op_aliasing_check():
                compile_backend = os.environ.get(
                    "SGLANG_CPU_GRAPH_COMPILE_BACKEND", "inductor"
                )
                yield torch.compile(
                    _make_compile_wrapper(model, model_forward),
                    backend=compile_backend,
                    dynamic=dynamic,
                )
        else:
            yield model.forward
    finally:
        if enable_compile:
            tp_group.ca_comm = backup_ca_comm


def _install_cpu_compile_layer_caches(model_runner: ModelRunner):
    if model_runner is None:
        return
    mamba_pool = getattr(model_runner.req_to_token_pool, "mamba_pool", None)
    mamba_cache = getattr(mamba_pool, "mamba_cache", None)
    if mamba_pool is None or mamba_cache is None:
        return

    layer_caches = [
        mamba_cache.at_layer_idx(layer_idx)
        for layer_idx in range(mamba_pool.num_mamba_layers)
    ]

    def get_layer_cache(layer_id: int):
        return layer_caches[layer_id]

    mamba_pool.mamba2_layer_cache = get_layer_cache


def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.freezing = True
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.assume_static_by_default = False
    register_inductor_fallback_ops()
    monkey_patch_torch_compile()


def get_batch_sizes_to_capture(model_runner: ModelRunner):
    # torch compile speeds up decoding by reducing python overhead on CPU
    server_args = model_runner.server_args
    # Reuse cuda_graph_config[decode].bs here.
    # Users can customize the batch sizes supported by cpu_graph, such as:
    # --cuda-graph-bs-decode 1 2 4 8 16
    capture_bs = server_args.cuda_graph_config.decode.bs
    assert (
        max(capture_bs) <= server_args.torch_compile_max_bs
    ), f"{capture_bs=}, {server_args.torch_compile_max_bs=}"
    capture_bs = [bs for bs in capture_bs if bs <= model_runner.req_to_token_pool.size]
    capture_bs = list(sorted(set(capture_bs)))
    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    return capture_bs


_CPU_COMPILE_FAKE_OPS: set[str] = set()


def register_cpu_compile_fake(op_name: str):
    _CPU_COMPILE_FAKE_OPS.add(op_name)
    return torch.library.register_fake(f"sgl_kernel::{op_name}")


def register_inductor_fallback_ops():
    from torch._inductor.lowering import lowerings, make_fallback

    sgl_kernel_ops = torch.ops.sgl_kernel
    for op_name in sorted(_CPU_COMPILE_FAKE_OPS):
        try:
            op = getattr(getattr(sgl_kernel_ops, op_name), "default")
        except AttributeError:
            continue
        if op not in lowerings:
            make_fallback(op, warn=False)


def register_fake_ops(tp_size: int):
    """
    Registers fake/meta implementations for all custom sgl_kernel CPU operators
    using torch.library.register_fake to support torch.compile
    """

    def add_position_dependency(output, positions):
        position_zero = positions.to(output.dtype).sum() * 0
        return output + position_zero

    none_return_ops = {
        "shm_allreduce",
        "shm_allgather_into_tensor",
        "shm_reduce_scatter_tensor",
        "bmm_cpu",
        "fused_add_rmsnorm_cpu",
        "decode_attention_cpu",
        "extend_attention_cpu",
        "gemma_fused_add_rmsnorm_cpu",
        "fused_sigmoid_mul_cpu",
        "store_cache_cpu",
    }
    for op in none_return_ops:

        @register_cpu_compile_fake(op)
        def _(*args, **kwargs):
            return

    for op in [
        "rmsnorm_cpu",
        "layernorm_cpu",
        "fused_add_layernorm_cpu",
        "l2norm_cpu",
        "fused_experts_cpu",
        "fused_rmsnorm_gated_cpu",
        "shared_expert_cpu",
        "gemma_rmsnorm_cpu",
        "gemma3_rmsnorm_cpu",
        "gemma4_rmsnorm_cpu",
    ]:

        @register_cpu_compile_fake(op)
        def _(input, *args, **kwargs):
            return torch.empty_like(input)

    @register_cpu_compile_fake("fused_qk_rmsnorm_cpu")
    def _(q, k, *args, **kwargs):
        return torch.empty_like(q), torch.empty_like(k)

    @register_cpu_compile_fake("fused_qk_rmsnorm_sumsq_cpu")
    def _(q, k):
        return torch.empty((q.shape[0], 2), dtype=torch.float32, device=q.device)

    @register_cpu_compile_fake("fused_qk_rmsnorm_apply_from_stats_cpu")
    def _(q, k, *args, **kwargs):
        return torch.empty_like(q), torch.empty_like(k)

    @register_cpu_compile_fake("causal_conv1d_fwd_cpu")
    def _(
        input,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        silu_activation,
        pad_slot_id,
        is_vnni,
    ):
        return torch.empty_like(input)

    @register_cpu_compile_fake("causal_conv1d_update_cpu")
    def _(
        input,
        conv_states,
        weight,
        bias,
        silu_activation,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
        is_vnni,
    ):
        return torch.empty_like(input)

    @register_cpu_compile_fake("rotary_embedding_cpu")
    def _(positions, query, key, head_size, cos_sin_cache, is_neox):
        return None

    @register_cpu_compile_fake("apply_rotary_pos_emb_cpu")
    def _(query, key, cos, sin):
        return None

    @register_cpu_compile_fake("shm_allgather")
    def _(data, dim):
        return torch.cat([data] * tp_size, dim=dim)

    @register_cpu_compile_fake("qkv_proj_with_rope")
    def _(
        hidden_states,
        q_a_proj_weight,
        q_b_proj_weight,
        kv_a_proj_weight,
        w_kc,
        q_a_layernorm_weight,
        kv_a_layernorm_weight,
        positions,
        cos_sin_cache,
        eps,
        use_int8_w8a8,
        use_fp8_w8a16,
        q_a_proj_scale,
        q_b_proj_scale,
        kv_a_proj_scale,
        is_vnni,
        block_size,
    ):
        num_seqs = hidden_states.shape[0]
        num_heads = w_kc.shape[0]
        kv_lora_rank = w_kc.shape[1]
        qk_rope_head_dim = kv_a_proj_weight.shape[0] - kv_lora_rank
        q_input = torch.empty(
            num_seqs,
            num_heads,
            kv_lora_rank + qk_rope_head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        k_input = torch.empty(
            num_seqs,
            1,
            kv_lora_rank + qk_rope_head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_input = add_position_dependency(q_input, positions)
        k_input = add_position_dependency(k_input, positions)
        v_input = k_input.narrow(-1, 0, kv_lora_rank)
        return q_input, k_input, v_input

    @register_cpu_compile_fake("multimodal_rotary_embedding_cpu")
    def _(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        mrope_section,
        mrope_interleaved,
        is_neox,
    ):
        position_zero = positions.to(query.dtype).sum() * 0
        query.add_(position_zero)
        key.add_(position_zero)

    @register_cpu_compile_fake("qkv_proj_with_rope_fused_weight")
    def _(
        hidden_states,
        q_a_proj_weight,
        q_b_proj_weight,
        w_kc,
        q_a_layernorm_weight,
        kv_a_layernorm_weight,
        positions,
        cos_sin_cache,
        eps,
        use_int8_w8a8,
        use_fp8_w8a16,
        qkv_a_proj_scale,
        q_b_proj_scale,
        w_scale,
        is_vnni,
        block_size,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
    ):
        num_seqs = hidden_states.shape[0]
        num_heads = w_kc.shape[0]
        kv_lora_rank = w_kc.shape[1]
        weight_chunks = torch.split(
            q_a_proj_weight, [q_lora_rank, kv_lora_rank + qk_rope_head_dim], dim=0
        )
        qk_rope_head_dim = weight_chunks[1].shape[0] - kv_lora_rank
        q_input = torch.empty(
            num_seqs,
            num_heads,
            kv_lora_rank + qk_rope_head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        k_input = torch.empty(
            num_seqs,
            1,
            kv_lora_rank + qk_rope_head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_input = add_position_dependency(q_input, positions)
        k_input = add_position_dependency(k_input, positions)
        v_input = k_input.narrow(-1, 0, kv_lora_rank)
        return q_input, k_input, v_input

    def get_n_size(mat2, is_vnni):
        tile_n = 16
        if mat2.dtype == torch.float32:
            return mat2.shape[1]
        if not is_vnni and mat2.dim() == 2 and mat2.shape[0] < tile_n:
            return mat2.shape[1]
        return mat2.shape[0]

    @register_cpu_compile_fake("weight_packed_linear")
    def _(mat1, mat2, bias, is_vnni):
        M = mat1.shape[0]
        N = get_n_size(mat2, is_vnni)
        return mat1.new_empty(M, N)

    @register_cpu_compile_fake("per_token_quant_int8_cpu")
    def _(input):
        M = input.shape[0]
        K = input.shape[1]
        Aq = input.new_empty(M, K, dtype=torch.int8)
        As = input.new_empty(M, dtype=torch.float32)
        return Aq, As

    @register_cpu_compile_fake("int8_scaled_mm_cpu")
    def _(mat1, mat2, scales1, scales2, bias, out_dtype, is_vnni):
        M = mat1.shape[0]
        N = mat2.shape[0]
        out = mat1.new_empty(M, N, dtype=out_dtype)
        return out

    @register_cpu_compile_fake("grouped_topk_cpu")
    def _(
        hidden_states,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        num_token_non_padded,
    ):
        num_tokens = hidden_states.shape[0]
        shape = (num_tokens, topk)
        device = hidden_states.device
        topk_weights = torch.empty(shape, device=device, dtype=torch.float32)
        topk_ids = torch.empty(shape, device=device, dtype=torch.int)
        return topk_weights, topk_ids

    @register_cpu_compile_fake("biased_grouped_topk_cpu")
    def _(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        num_token_non_padded,
    ):
        num_tokens = hidden_states.shape[0]
        shape = (num_tokens, topk)
        device = hidden_states.device
        topk_weights = torch.empty(shape, device=device, dtype=torch.float32)
        topk_ids = torch.empty(shape, device=device, dtype=torch.int)
        return topk_weights, topk_ids

    @register_cpu_compile_fake("topk_sigmoid_cpu")
    def _(hidden_states, gating_output, topk, renormalize, correction_bias=None):
        num_tokens = hidden_states.shape[0]
        shape = (num_tokens, topk)
        return (
            torch.empty(shape, device=hidden_states.device, dtype=torch.float),
            torch.empty(shape, device=hidden_states.device, dtype=torch.int),
        )

    @register_cpu_compile_fake("topk_softmax_cpu")
    def _(
        hidden_states,
        gating_output,
        topk,
        renormalize,
        correction_bias=None,
    ):
        num_tokens = hidden_states.shape[0]
        shape = (num_tokens, topk)
        return (
            torch.empty(shape, device=hidden_states.device, dtype=torch.float),
            torch.empty(shape, device=hidden_states.device, dtype=torch.int),
        )

    for act_op in [
        "silu_and_mul_cpu",
        "gelu_tanh_and_mul_cpu",
        "gelu_and_mul_cpu",
    ]:

        @register_cpu_compile_fake(act_op)
        def _(input):
            sizes = list(input.shape)
            last_dim = input.dim() - 1
            d = sizes[last_dim] // 2
            sizes[last_dim] = d
            return input.new_empty(sizes)

    @register_cpu_compile_fake("int8_scaled_mm_with_quant")
    def _(
        mat1,
        mat2,
        scales2,
        bias,
        out_dtype,
        is_vnni,
    ):
        M = mat1.shape[0]
        N = mat2.shape[0]
        return mat1.new_empty(M, N, dtype=out_dtype)

    @register_cpu_compile_fake("fp8_scaled_mm_cpu")
    def _(
        mat1,
        mat2,
        scales2,
        block_size,
        bias,
        out_dtype,
        is_vnni,
    ):
        M = mat1.shape[0]
        N = mat2.shape[0]
        return mat1.new_empty(M, N, dtype=out_dtype)

    @register_cpu_compile_fake("mxfp4_scaled_mm_cpu")
    def _(mat1, mat2, scales2, bias, is_vnni):
        sizes = list(mat1.shape)
        sizes[-1] = mat2.shape[0]
        return mat1.new_empty(sizes)

    @register_cpu_compile_fake("int4_scaled_mm_cpu")
    def _(x, w, w_zeros, w_scales, bias):
        sizes = list(x.shape)
        sizes[-1] = w_scales.shape[0] * w_scales.shape[-1]
        return x.new_empty(sizes)

    @register_cpu_compile_fake("fused_linear_sigmoid_mul")
    def _(
        mat1,
        mat2,
        bias,
        is_vnni,
        post_mul_mat,
    ):
        M = mat1.shape[0]
        N = post_mul_mat.shape[1]
        return mat1.new_empty(M, N)

    @register_cpu_compile_fake("fused_qkvzba_split_reshape_cat_cpu")
    def _(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
        batch = mixed_qkvz.shape[0]
        qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
        mixed_qkv = mixed_qkvz.new_empty(batch, qkv_dim)
        z = mixed_qkvz.new_empty(batch, num_heads_v, head_v)
        b = mixed_ba.new_empty(batch, num_heads_v)
        a = mixed_ba.new_empty(batch, num_heads_v)
        return mixed_qkv, z, b, a

    @register_cpu_compile_fake("fused_qkvzba_split_reshape_cat_contiguous_cpu")
    def _(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
        batch = mixed_qkvz.shape[0]
        qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
        mixed_qkv = mixed_qkvz.new_empty(batch, qkv_dim)
        z = mixed_qkvz.new_empty(batch, num_heads_v, head_v)
        b = mixed_ba.new_empty(batch, num_heads_v)
        a = mixed_ba.new_empty(batch, num_heads_v)
        return mixed_qkv, z, b, a

    @register_cpu_compile_fake("fused_input_proj_cpu")
    def _(hidden_states, qkvz_weight, ba_weight, is_vnni):
        batch = hidden_states.shape[0]
        qkvz_dim = qkvz_weight.shape[0]
        ba_dim = ba_weight.shape[0]
        return hidden_states.new_empty(batch, qkvz_dim), hidden_states.new_empty(
            batch, ba_dim
        )

    @register_cpu_compile_fake("fused_sigmoid_gating_delta_rule_update_cpu")
    def _(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    ):
        assert q.dim() == 4
        assert v.dim() == 4
        batch_size = q.shape[1]
        seq_len = q.shape[0]
        v_num_heads = v.shape[2]
        v_head_dim = v.shape[3]
        return q.new_empty(batch_size, seq_len, v_num_heads, v_head_dim)

    @register_cpu_compile_fake("fused_gdn_gating_cpu")
    def _(A_log, a, b, dt_bias):
        batch = a.shape[0]
        num_heads = a.shape[1]
        out = a.new_empty(1, batch, num_heads, dtype=torch.float)
        beta = b.new_empty(1, batch, num_heads)
        return out, beta

    @register_cpu_compile_fake("chunk_gated_delta_rule_cpu")
    def _(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel,
        initial_state_indices,
        eps=1e-6,
    ):
        output = torch.empty_like(value)
        assert initial_state is not None
        return output

    @register_cpu_compile_fake("fused_qk_gemma_rmsnorm_cpu")
    def _(q, k, q_norm_weight, k_norm_weight, eps, head_dim):
        return torch.empty_like(q), torch.empty_like(k)

    @register_cpu_compile_fake("fused_qk_gemma_rmsnorm_with_gate_cpu")
    def _(q_gate, k, q_norm_weight, k_norm_weight, eps, head_dim, num_head):
        q_flat = q_gate.new_empty(q_gate.shape[0] * num_head, head_dim)
        k_flat = k.new_empty(k.shape[0] * (k.shape[1] // head_dim), head_dim)
        gate_flat = q_gate.new_empty(q_gate.shape[0] * num_head, head_dim)
        return q_flat, k_flat, gate_flat

    # @register_cpu_compile_fake("hc_pre_fused_cpu")
    # def _(x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_norm_eps, hc_eps):
    #     T = x.shape[0]
    #     hc = x.shape[1] if x.ndim == 3 else hc_base.shape[0]
    #     d = x.shape[2] if x.ndim == 3 else x.shape[1] // hc
    #     y = x.new_empty(T, d)
    #     post = x.new_empty(T, hc, dtype=torch.float32)
    #     comb = x.new_empty(T, hc, hc, dtype=torch.float32)
    #     return y, post, comb

    # @register_cpu_compile_fake("hc_post_fused_cpu")
    # def _(x, residual, post, comb):
    #     return torch.empty_like(residual)

    # @register_cpu_compile_fake("hc_head_fused_cpu")
    # def _(x, hc_fn, hc_scale, hc_base, hc_eps, norm_eps):
    #     T = x.shape[0]
    #     d = x.shape[2] if x.ndim == 3 else x.shape[1]
    #     return x.new_empty(T, d)

    @register_cpu_compile_fake("conv3d_embed_cpu")
    def _(x, weight, bias=None, is_vnni=False):
        c_out = bias.shape[0] if bias is not None else weight.shape[0]
        out_shape = (x.shape[0], c_out) + x.shape[2:]
        return x.new_empty(out_shape)


class CPUGraphRunner:
    """CPU torch.compile runner for decode and prefill buckets.

    Decode uses batch-size buckets. Prefill uses total-token buckets. Static
    padding follows the server setting; ``SGLANG_TORCH_DYNAMIC_SHAPE=1`` adds
    one dynamic fallback per phase, while the default is eager fallback.
    """

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        self.device = model_runner.device
        # Decode steps (and hence decode graphs) only exist for generation
        # models -- embedding/reward/classification models only ever run
        # EXTEND. Gate the decode-only setup below on this instead of
        # skipping construction entirely (that would also deny non-generation
        # models the prefill graph, which *is* meaningful for them).
        self.is_generation = model_runner.is_generation
        self.return_hidden_states_mode = (
            CaptureHiddenMode.NULL
            if model_runner.is_draft_worker
            else get_server_return_hidden_states_mode(model_runner.server_args)
        )
        self.enable_return_hidden_states = self.return_hidden_states_mode.need_capture()
        # bs -> compiled fn (text-only / skip_cross_attention=True)
        self.graphs = {}
        # bs -> compiled fn (cross-attention / skip_cross_attention=False, enc-dec only)
        self.graphs_cross = {}
        # Dynamic decode fallback keyed by skip_cross_attention.
        self.decode_dynamic_graphs = {}
        self.decode_dynamic_batch_templates = {}
        self.output_buffers = {}
        self.enable_torch_compile = get_flags().capture.enable_torch_compile
        if self.enable_torch_compile:
            _install_cpu_compile_layer_caches(self.model_runner)
        self.enable_dynamic_graph = get_bool_env_var(
            "SGLANG_TORCH_DYNAMIC_SHAPE", "false"
        )
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.enable_two_batch_overlap = (
            model_runner.server_args.enable_two_batch_overlap
        )
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = get_parallel().dp_size
        self.pp_size = model_runner.server_args.pp_size

        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = self.return_hidden_states_mode
        self._prefill_graph_model = self._get_prefill_graph_model()
        self._prefill_uses_eager_logits_tail = (
            self._prefill_graph_model is not self.model_runner.model
        )
        # Static capture width: CPU graphs are decode-only.
        self.captured_req_width = 1

        assert (
            not self.model_runner.server_args.enable_lora
        ), "CPUGraphRunner does not support LoRA yet."
        assert (
            not self.enable_two_batch_overlap
        ), "CPUGraphRunner does not support two batch overlap yet."
        assert (
            not self.require_mlp_tp_gather
        ), "CPUGraphRunner does not support MLP TP gather yet."
        assert (
            not self.require_mlp_sync
        ), "CPUGraphRunner does not support MLP sync yet."
        assert (
            not self.require_gathered_buffer
        ), "CPUGraphRunner does not support gathered buffer yet."
        assert (
            model_runner.spec_algorithm.is_none()
        ), "CPUGraphRunner does not support speculative inference yet."

        assert self.dp_size == 1, "CPUGraphRunner does not support DP yet."
        assert self.pp_size == 1, "CPUGraphRunner does not support PP yet."

        if self.is_generation:
            # Batch sizes to capture
            self.capture_bs = get_batch_sizes_to_capture(model_runner)
            log_info_on_rank0(logger, f"Capture cpu graph bs {self.capture_bs}")
            # bs -> ForwardBatch (text-only / skip_cross_attention=True)
            self.captured_forward_batches = {}
            # bs -> ForwardBatch (cross-attention / skip=False, enc-dec only)
            self.captured_forward_batches_cross = {}
            # Attention backend
            self.max_bs = max(self.capture_bs)
            self.max_num_token = self.max_bs * self.captured_req_width
            if self.enable_dynamic_graph:
                self.dynamic_capture_bs = min(
                    self.max_bs + 1, model_runner.req_to_token_pool.size
                )
            else:
                self.dynamic_capture_bs = self.max_bs
            self.capture_buffer_bs = max(self.max_bs, self.dynamic_capture_bs)
            self.capture_buffer_num_token = (
                self.capture_buffer_bs * self.captured_req_width
            )
            self.model_runner.attn_backend.init_cpu_graph_state(
                self.capture_buffer_bs, self.capture_buffer_num_token
            )

            self.encoder_len_fill_value = 0
            self.seq_len_fill_value = (
                self.model_runner.attn_backend.get_cpu_graph_seq_len_fill_value()
            )
        else:
            # No decode steps for non-generation (embedding/reward/classification)
            # models -- leave decode state empty; can_run_graph rejects DECODE
            # mode below and everything falls through to the prefill path.
            self.capture_bs = []
            self.captured_forward_batches = {}
            self.captured_forward_batches_cross = {}
            self.max_bs = 0
            self.max_num_token = 0
            self.encoder_len_fill_value = 0
            self.seq_len_fill_value = 0

        if self.enable_torch_compile:
            register_fake_ops(self.tp_size)
            set_torch_compile_config()

        if self.is_generation:
            # Graph inputs
            with torch.device(self.device):
                self.input_ids = torch.zeros(
                    (self.capture_buffer_num_token,), dtype=torch.int64
                )
                self.req_pool_indices = torch.zeros(
                    (self.capture_buffer_bs,), dtype=torch.int64
                )
                self.seq_lens = torch.full(
                    (self.capture_buffer_bs,),
                    self.seq_len_fill_value,
                    dtype=torch.int64,
                )
                self.out_cache_loc = torch.zeros(
                    (self.capture_buffer_num_token,), dtype=torch.int64
                )
                self.positions = torch.zeros(
                    (self.capture_buffer_num_token,), dtype=torch.int64
                )
                self.mrope_positions = torch.zeros(
                    (3, self.capture_buffer_num_token), dtype=torch.int64
                )
                self.num_token_non_padded = torch.zeros((1,), dtype=torch.int64)
                self.custom_mask = torch.ones(
                    (
                        (
                            self.seq_lens[: self.capture_buffer_bs].sum().item()
                            + self.capture_buffer_num_token
                        )
                        * self.captured_req_width
                    ),
                    dtype=torch.bool,
                    device=self.device,
                )
                if self.is_encoder_decoder:
                    self.encoder_lens = torch.full(
                        (self.capture_buffer_bs,),
                        self.encoder_len_fill_value,
                        dtype=torch.int64,
                    )
                else:
                    self.encoder_lens = None
        else:
            self.encoder_lens = None

        # --- Prefill (EXTEND) graph state -------------------------------
        # bs is NOT the bucket axis here (num_tokens is) -- see class
        # docstring section on prefill design. Disabled (empty bucket list)
        # for encoder-decoder models (skip_cross_attention dual-graph capture
        # is not implemented for prefill), when cuda_graph_config.prefill.bs
        # isn't configured, or when prefill.backend is explicitly 'disabled' --
        # can_run_graph then simply rejects EXTEND batches and callers fall
        # back to eager, exactly like today.
        prefill_disabled = check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)
        self.capture_num_tokens = (
            sorted(model_runner.server_args.cuda_graph_config.prefill.bs or [])
            if not self.is_encoder_decoder and not prefill_disabled
            else []
        )
        if (
            model_runner.server_args.cuda_graph_config.prefill.bs
            and self.is_encoder_decoder
        ):
            log_info_on_rank0(
                logger,
                "Skipping CPU prefill graph for encoder-decoder model "
                "(unsupported in v1).",
            )
        # num_tokens bucket -> compiled callable (bs dynamic inside).
        self.prefill_graphs: dict = {}
        self.prefill_graphs_bs1: dict = {}
        # Single fallback callable (num_tokens *and* bs both dynamic), used for
        # any num_tokens that doesn't exactly match a configured bucket.
        self.prefill_dynamic_graph: Optional[Callable] = None
        self.prefill_dynamic_graph_bs1: Optional[Callable] = None
        self.prefill_dynamic_batch_template: Optional[ForwardBatch] = None
        if self.capture_num_tokens:
            self.prefill_max_num_tokens = self.capture_num_tokens[-1]
            self.prefill_max_bs = model_runner.req_to_token_pool.size
            # bs values used to warm up every bucket / the dynamic fallback so
            # the request axis is established as a symbolic range before
            # serving starts (design notes doc, section 9.4). Two
            # well-separated values (>=2) force torch.compile to generate a
            # symbolic shape graph immediately without specializing on size 1.
            start_bs = 2 if self.prefill_max_bs >= 2 else 1
            self.prefill_warmup_bs_values = list(
                dict.fromkeys([start_bs, min(4, self.prefill_max_bs), 1])
            )

        # Capture
        try:
            # use model_capture_mode for encoder-decoder models to
            # set skip_cross_attention to avoid
            # "Graph Break Reason: Data-dependent branching" caused by
            # skip_cross_attention = forward_batch.encoder_lens.max() == 0
            capture_context = (
                model_capture_mode if self.is_encoder_decoder else empty_context
            )
            with capture_context():
                self.capture()
            if self.capture_num_tokens:
                self.capture_prefill()
        except RuntimeError as e:
            raise Exception(
                f"Capture CPU graph failed: {e}\n{CPU_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _get_skip_cross_attention(self, forward_batch: ForwardBatch) -> bool:
        """Return True when cross-attention layers should be skipped.

        Non-encoder-decoder models have no cross-attention at all, so they
        always use self.graphs (the skip=True / text-only graph dict).
        For encoder-decoder models, skip when no request in the batch has
        encoder output (i.e. no images).
        """
        if not self.is_encoder_decoder:
            return True
        return bool(forward_batch.encoder_lens.max() == 0)

    def can_run_graph(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_extend():
            return self._can_run_prefill_graph(forward_batch)

        skip_cross_attention = self._get_skip_cross_attention(forward_batch)
        graphs = self.graphs if skip_cross_attention else self.graphs_cross
        if forward_batch.batch_size in graphs:
            return True
        if not self.disable_padding and forward_batch.batch_size <= self.max_bs:
            return True
        # PyTorch's dynamic symbols use a default lower bound of 2.  Keep a
        # size-one request on the static/padding path instead of tracing a
        # dynamic decode graph with an invalid lower bound.
        if forward_batch.batch_size == 1:
            return False
        return self.enable_dynamic_graph and (
            self.decode_dynamic_graphs.get(skip_cross_attention) is not None
        )

    def capture(self) -> None:
        capture_range = (
            tqdm.tqdm(list(reversed(self.capture_bs)))
            if get_parallel().tp_rank == 0
            else reversed(self.capture_bs)
        )
        for bs in capture_range:
            if get_parallel().tp_rank == 0:
                avail_mem = psutil.virtual_memory().available / (1 << 30)
                capture_range.set_description(
                    f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                )

            with patch_model(
                self.model_runner.model,
                bs in self.capture_bs,
                num_tokens=bs * self.captured_req_width,
                tp_group=self.model_runner.tp_group,
            ) as forward:
                graph, output_buffers = self.capture_one_batch_size(
                    bs, forward, skip_cross_attention=True
                )
                self.graphs[bs] = graph
                self.output_buffers[bs] = output_buffers
                if self.is_encoder_decoder:
                    # Capture a second graph with cross-attention enabled
                    # (used when the batch contains images).
                    graph_cross, _ = self.capture_one_batch_size(
                        bs, forward, skip_cross_attention=False
                    )
                    self.graphs_cross[bs] = graph_cross

        # Re-init states for qwen3-next as
        # torch.compile may change the states
        self._reset_mamba_cache_if_needed()
        if not self.capture_num_tokens:
            self._compile_decode_dynamic_graphs()

    def _reset_mamba_cache_if_needed(self) -> None:

        mamba_pool = getattr(self.model_runner.req_to_token_pool, "mamba_pool", None)
        if mamba_pool is None:
            return
        mamba_cache = getattr(mamba_pool, "mamba_cache", None)
        if mamba_cache is None:
            return

        def _zero_nested(obj):
            if isinstance(obj, torch.Tensor):
                obj.zero_()
            elif isinstance(obj, (list, tuple)):
                for it in obj:
                    _zero_nested(it)

        for v in vars(mamba_cache).values():
            _zero_nested(v)

        for name in (
            "replayssm_write_pos",
            "replayssm_cache_base",
            "replayssm_is_flush",
            "req_index_to_mamba_index_mapping",
            "req_index_to_mamba_ping_pong_track_buffer_mapping",
        ):
            value = getattr(mamba_pool, name, None)
            if isinstance(value, torch.Tensor):
                value.zero_()

    def capture_one_batch_size(
        self,
        bs: int,
        forward: Callable,
        skip_cross_attention: bool = False,
        mark_dynamic: bool = False,
        save_captured_batch: bool = True,
    ):
        num_tokens = bs * self.captured_req_width

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        if mark_dynamic:
            # A dynamic slice must not retain the larger capture buffer as its
            # view base. Dynamo otherwise guards on base.size() == view.size()
            # and specializes the dynamic callable for every warmup bucket.
            input_ids = input_ids.clone()
            req_pool_indices = req_pool_indices.clone()
            seq_lens = seq_lens.clone()
            out_cache_loc = out_cache_loc.clone()
            positions = positions.clone()
        # Runtime mRoPE positions are [3, token_count] contiguous tensors.
        # Do not trace a view with the parent buffer's max-batch stride, and do
        # not introduce a Tensor input for models that never use mRoPE.
        mrope_positions = (
            self.mrope_positions[:, :num_tokens].contiguous()
            if self.model_runner.model_config.model_is_mrope
            else None
        )
        self.num_token_non_padded[...] = num_tokens
        if self.is_encoder_decoder:
            encoder_lens = self.encoder_lens[:bs]
            if mark_dynamic:
                encoder_lens = encoder_lens.clone()
        else:
            encoder_lens = None

        spec_info = self.get_spec_info(num_tokens)
        self.capture_hidden_mode = get_required_capture_hidden_mode(
            self.capture_hidden_mode,
            spec_info,
        )

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            encoder_lens=encoder_lens,
            encoder_lens_cpu=encoder_lens,
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=self.num_token_non_padded,
            global_forward_mode=self.capture_forward_mode,
        )
        # Wrap all forward calls with capture_with_skip_cross_attention so that
        # mllama (and any other encoder-decoder model) sees the correct compile-
        # time constant for skip_cross_attention during tracing.
        skip_ctx = (
            capture_with_skip_cross_attention(skip_cross_attention)
            if self.is_encoder_decoder
            else empty_context()
        )
        with skip_ctx:
            with forward_context(
                ForwardContext(attn_backend=self.model_runner.attn_backend)
            ):
                self.model_runner.attn_backend.init_forward_metadata_capture_cpu_graph(
                    bs,
                    num_tokens,
                    req_pool_indices,
                    seq_lens,
                    None,
                    forward_batch.forward_mode,
                    forward_batch.spec_info,
                )
                if mark_dynamic:
                    self._mark_dynamic_decode_inputs(forward_batch)
                with torch.no_grad():
                    self.model_runner.tp_group.barrier()
                    self.model_runner.model.forward(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                    )

                # Run and capture
                def run_once():
                    # Clean intermediate result cache for DP attention
                    forward_batch.dp_local_start_pos = (
                        forward_batch.dp_local_num_tokens
                    ) = None
                    logits_output_or_pp_proxy_tensors = forward(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                    )
                    return logits_output_or_pp_proxy_tensors

                with torch.no_grad():
                    for _ in range(2):
                        self.model_runner.tp_group.barrier()
                        out = run_once()
                    # Save the captured forward_batch in the appropriate dict
                    if save_captured_batch:
                        if skip_cross_attention:
                            self.captured_forward_batches[bs] = forward_batch
                        else:
                            self.captured_forward_batches_cross[bs] = forward_batch
                    if mark_dynamic:
                        self.decode_dynamic_batch_templates[skip_cross_attention] = (
                            forward_batch
                        )
                    return forward, out

    def _mark_dynamic_decode_inputs(self, forward_batch: ForwardBatch) -> None:
        for name, value in vars(forward_batch).items():
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                continue
            dim = value.ndim - 1 if name == "mrope_positions" else 0
            _mark_dynamic_axis(value, dim)

    def _refresh_dynamic_forward_batch_metadata(
        self,
        dynamic_forward_batch: ForwardBatch,
        forward_batch: ForwardBatch,
        include_extend_metadata: bool,
    ) -> None:
        metadata_fields = [
            "num_token_non_padded_cpu",
            "encoder_lens_cpu",
            "encoder_cached",
        ]
        if include_extend_metadata:
            metadata_fields.extend(
                [
                    "extend_num_tokens",
                    "extend_seq_lens_cpu",
                    "extend_prefix_lens_cpu",
                    "extend_logprob_start_lens_cpu",
                    "top_logprobs_nums",
                    "token_ids_logprobs",
                    "multi_item_delimiter_indices",
                ]
            )
        for name in metadata_fields:
            value = getattr(forward_batch, name)
            setattr(dynamic_forward_batch, name, copy.copy(value))

    def _compile_decode_dynamic_graph(self, skip_cross_attention: bool) -> Callable:
        with patch_model(
            self.model_runner.model,
            self.enable_torch_compile,
            num_tokens=self.capture_buffer_num_token,
            tp_group=self.model_runner.tp_group,
            dynamic=True,
        ) as forward:
            dynamic_warmup_bs = [
                bs
                for bs in sorted(set(self.capture_bs + [self.dynamic_capture_bs]))
                if bs >= 2
            ]
            for bs in dynamic_warmup_bs:
                self.capture_one_batch_size(
                    bs,
                    forward,
                    skip_cross_attention=skip_cross_attention,
                    mark_dynamic=True,
                    save_captured_batch=False,
                )
        return forward

    def _prepare_dynamic_decode_batch(
        self, forward_batch: ForwardBatch, skip_cross_attention: bool
    ) -> ForwardBatch:
        template = self.decode_dynamic_batch_templates.get(skip_cross_attention)
        if template is None:
            return forward_batch

        dynamic_forward_batch = template
        for name, value in vars(forward_batch).items():
            if isinstance(value, torch.Tensor):
                setattr(dynamic_forward_batch, name, value)
        dynamic_forward_batch.forward_mode = self.capture_forward_mode
        dynamic_forward_batch.global_forward_mode = self.capture_forward_mode
        dynamic_forward_batch.batch_size = forward_batch.batch_size
        dynamic_forward_batch.seq_lens_sum = forward_batch.seq_lens_sum
        if (
            forward_batch.mm_inputs is not None
            and not forward_batch.contains_mm_inputs()
        ):
            dynamic_forward_batch.mm_inputs = None
        else:
            dynamic_forward_batch.mm_inputs = forward_batch.mm_inputs
        dynamic_forward_batch.encoder_out_cache_loc = None
        self._refresh_dynamic_forward_batch_metadata(
            dynamic_forward_batch,
            forward_batch,
            include_extend_metadata=False,
        )
        return dynamic_forward_batch

    def _validate_capture_hidden_mode(self, forward_batch: ForwardBatch) -> None:
        if self.capture_hidden_mode < forward_batch.capture_hidden_mode:
            raise RuntimeError(
                "The runtime hidden-state mode exceeds the fixed CPU graph "
                f"capture mode ({self.capture_hidden_mode.name})."
            )

    def prepare_replay(
        self,
        forward_batch: ForwardBatch,
        skip: bool = False,
    ):
        self._validate_capture_hidden_mode(forward_batch)

        graphs = self.graphs_cross if not skip else self.graphs
        cfbs = (
            self.captured_forward_batches_cross
            if not skip
            else self.captured_forward_batches
        )

        raw_bs = forward_batch.batch_size
        if raw_bs in graphs:
            # Use the same ForwardBatch object and storage that were traced.
            # Passing the live scheduler tensors for an exact bucket makes
            # Dynamo observe a different dynamic-marker/storage state than it
            # saw during capture, even when all visible shapes match.
            captured_forward_batch = cfbs[raw_bs]
            assert captured_forward_batch is not None
            captured_forward_batch.input_ids.copy_(forward_batch.input_ids)
            captured_forward_batch.req_pool_indices.copy_(
                forward_batch.req_pool_indices
            )
            captured_forward_batch.seq_lens.copy_(forward_batch.seq_lens)
            captured_forward_batch.out_cache_loc.copy_(forward_batch.out_cache_loc)
            captured_forward_batch.positions.copy_(forward_batch.positions)
            if captured_forward_batch.mrope_positions is not None:
                assert forward_batch.mrope_positions is not None
                captured_forward_batch.mrope_positions.copy_(
                    forward_batch.mrope_positions
                )
            # Keep encoder_out_cache_loc consistent with the captured graph (None).
            if self.is_encoder_decoder:
                # encoder_out_cache_loc is never accessed during decode (k/v are
                # None so the KV-write path is skipped in the kernel).  Use None
                # consistently at both capture time and runtime.
                captured_forward_batch.encoder_lens.copy_(forward_batch.encoder_lens)
                forward_batch.encoder_out_cache_loc = None
                captured_forward_batch.encoder_out_cache_loc = None
            if enable_num_token_non_padded():
                captured_forward_batch.num_token_non_padded.copy_(
                    forward_batch.num_token_non_padded
                )
            self._refresh_dynamic_forward_batch_metadata(
                captured_forward_batch,
                forward_batch,
                include_extend_metadata=False,
            )
            replay_init = getattr(
                self.model_runner.attn_backend,
                "init_forward_metadata_cpu_graph_replay",
                None,
            )
            if replay_init is None:
                self.model_runner.attn_backend.init_forward_metadata(
                    captured_forward_batch
                )
            else:
                replay_init(captured_forward_batch)
            return captured_forward_batch

        raw_num_token = raw_bs * self.captured_req_width
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        assert bs > raw_bs
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs

        captured_forward_batch = cfbs[bs]
        assert captured_forward_batch is not None
        captured_forward_batch.seq_lens.fill_(self.seq_len_fill_value)
        captured_forward_batch.out_cache_loc.zero_()
        # Pair with seq_lens fill: padded rows must point at reserved
        # req_pool slot 0 (req_to_token[0, :] is all zeros from init).
        captured_forward_batch.req_pool_indices.zero_()
        captured_forward_batch.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        captured_forward_batch.req_pool_indices[:raw_bs].copy_(
            forward_batch.req_pool_indices
        )
        captured_forward_batch.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        captured_forward_batch.out_cache_loc[:raw_num_token].copy_(
            forward_batch.out_cache_loc
        )
        captured_forward_batch.positions[:raw_num_token].copy_(forward_batch.positions)
        if captured_forward_batch.mrope_positions is not None:
            assert forward_batch.mrope_positions is not None
            captured_forward_batch.mrope_positions[:, raw_num_token:].zero_()
            captured_forward_batch.mrope_positions[:, :raw_num_token].copy_(
                forward_batch.mrope_positions
            )

        if self.is_encoder_decoder:
            captured_forward_batch.encoder_lens[:raw_bs].copy_(
                forward_batch.encoder_lens
            )
            captured_forward_batch.encoder_out_cache_loc = None
        if enable_num_token_non_padded():
            captured_forward_batch.num_token_non_padded.copy_(
                forward_batch.num_token_non_padded
            )
        self._refresh_dynamic_forward_batch_metadata(
            captured_forward_batch,
            forward_batch,
            include_extend_metadata=False,
        )

        replay_init = getattr(
            self.model_runner.attn_backend,
            "init_forward_metadata_cpu_graph_replay",
            None,
        )
        if replay_init is None:
            self.model_runner.attn_backend.init_forward_metadata(captured_forward_batch)
        else:
            replay_init(captured_forward_batch)
        return captured_forward_batch

    def execute(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if forward_batch.forward_mode.is_extend():
            return self._execute_prefill_graph(forward_batch, pp_proxy_tensors)

        assert (
            pp_proxy_tensors is None
        ), "PPProxyTensors is not supported in CPUGraphRunner yet."

        replay_context = (
            model_capture_mode if self.is_encoder_decoder else empty_context
        )
        # Determine which compiled graph to use and pin skip_cross_attention so
        # that any torch.compile re-tracing sees the same compile-time constant.
        skip = self._get_skip_cross_attention(forward_batch)
        graphs = self.graphs_cross if not skip else self.graphs
        skip_ctx = (
            capture_with_skip_cross_attention(skip)
            if self.is_encoder_decoder
            else empty_context()
        )
        with replay_context():
            with skip_ctx:
                dynamic_graph = self.decode_dynamic_graphs.get(skip)
                can_pad = (
                    not self.disable_padding and forward_batch.batch_size <= self.max_bs
                )
                if (
                    forward_batch.batch_size not in graphs
                    and not can_pad
                    and dynamic_graph is not None
                ):
                    dynamic_forward_batch = self._prepare_dynamic_decode_batch(
                        forward_batch, skip
                    )
                    self._mark_dynamic_decode_inputs(dynamic_forward_batch)
                    self.model_runner.attn_backend.init_forward_metadata(
                        dynamic_forward_batch
                    )
                    return dynamic_graph(
                        dynamic_forward_batch.input_ids,
                        dynamic_forward_batch.positions,
                        dynamic_forward_batch,
                    )
                prepared_forward_batch = self.prepare_replay(forward_batch, skip=skip)
                output = graphs[prepared_forward_batch.batch_size](
                    prepared_forward_batch.input_ids,
                    prepared_forward_batch.positions,
                    prepared_forward_batch,
                )
        if forward_batch.batch_size in graphs:
            return output

        assert isinstance(output, LogitsProcessorOutput)
        return LogitsProcessorOutput(
            next_token_logits=output.next_token_logits[: self.raw_num_token],
            hidden_states=(
                output.hidden_states[: self.raw_num_token]
                if output.hidden_states is not None
                else None
            ),
        )

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if (
            self.model_runner.spec_algorithm.is_eagle()
            or self.model_runner.spec_algorithm.is_standalone()
        ):
            from sglang.srt.speculative.eagle_info import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=self.custom_mask,
                    positions=None,
                    retrieve_index=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    retrieve_cum_len=None,
                    spec_steps=get_spec().speculative_num_steps,
                    topk=self.model_runner.server_args.speculative_eagle_topk,
                    draft_token_num=get_spec().speculative_num_draft_tokens,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )

        return spec_info

    # ------------------------------------------------------------------
    # Prefill (EXTEND) graph: dummy batch construction
    # ------------------------------------------------------------------

    def _build_prefill_dummy_forward_batch(
        self, num_tokens: int, bs: int
    ) -> ForwardBatch:
        """Build a dummy EXTEND ForwardBatch with `bs` requests sharing
        `num_tokens` total tokens (split as evenly as possible), no cached
        prefix. Simplified vs. PrefillCudaGraphRunner.capture_prepare: no mamba
        track / multimodal / pp_proxy support (v1 scope, see class docstring).
        """
        assert bs <= num_tokens, f"{bs=} must be <= {num_tokens=}"
        base = num_tokens // bs
        lens_cpu = [base] * bs
        lens_cpu[-1] += num_tokens - base * bs  # remainder goes to the last request
        start_loc_cpu = [0]
        for l in lens_cpu[:-1]:
            start_loc_cpu.append(start_loc_cpu[-1] + l)

        with torch.device(self.device):
            req_pool_indices = torch.arange(bs, dtype=torch.int64)
            extend_seq_lens = torch.tensor(lens_cpu, dtype=torch.int32)
            extend_prefix_lens = torch.zeros((bs,), dtype=torch.int32)
            extend_start_loc = torch.tensor(start_loc_cpu, dtype=torch.int32)
            seq_lens = torch.tensor(
                lens_cpu, dtype=torch.int64
            )  # no cached prefix in the dummy batch
            input_ids = torch.zeros((num_tokens,), dtype=torch.int64)
            out_cache_loc = torch.zeros((num_tokens,), dtype=torch.int64)
            positions = torch.cat(
                [torch.arange(l, dtype=torch.int64) for l in lens_cpu]
            )
            mrope_positions = (
                positions.unsqueeze(0).repeat(3, 1)
                if getattr(self.model_runner.model, "is_mrope_enabled", False)
                else None
            )
            num_token_non_padded = torch.tensor([num_tokens], dtype=torch.int64)

        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=torch.tensor(lens_cpu, dtype=torch.int64),
            out_cache_loc=out_cache_loc,
            seq_lens_sum=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_seq_lens_cpu=lens_cpu,
            extend_prefix_lens_cpu=[0] * bs,
            extend_logprob_start_lens_cpu=lens_cpu,
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=num_token_non_padded,
            global_forward_mode=ForwardMode.EXTEND,
        )

    def _mark_dynamic_request_axis(self, forward_batch: ForwardBatch) -> None:
        _mark_dynamic(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_start_loc,
        )

    def _mark_dynamic_token_axis(self, forward_batch: ForwardBatch) -> None:
        _mark_dynamic(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch.out_cache_loc,
        )
        _mark_dynamic_axis(forward_batch.mrope_positions, 1)

    def _mark_dynamic_prefill_metadata(self) -> None:
        backend = self.model_runner.attn_backend
        for candidate in (backend, getattr(backend, "linear_attn_backend", None)):
            metadata = getattr(candidate, "forward_metadata", None)
            if metadata is not None:
                _mark_dynamic(
                    getattr(metadata, "query_start_loc", None),
                    getattr(metadata, "mamba_cache_indices", None),
                )

    # ------------------------------------------------------------------
    # Prefill (EXTEND) graph: capture / warmup
    # ------------------------------------------------------------------

    def _get_prefill_graph_model(self) -> torch.nn.Module:
        model = self.model_runner.model
        if (
            self.is_generation
            and getattr(model, "pp_group", None) is not None
            and model.pp_group.is_last_rank
        ):
            try:
                layer_model = resolve_language_model(model)
            except AttributeError:
                layer_model = getattr(model, "language_model", model)

            seen = set()
            while not hasattr(layer_model, "layers") and hasattr(layer_model, "model"):
                model_id = id(layer_model)
                if model_id in seen:
                    return model
                seen.add(model_id)
                layer_model = layer_model.model

            if hasattr(layer_model, "layers"):
                return layer_model
        return model

    def _run_prefill_once(self, compiled_fn: Callable, forward_batch: ForwardBatch):
        # Metadata prep happens *outside* the compiled callable, exactly like
        # decode already does above -- CPU never needs a separate "capture-time"
        # metadata initializer the way CUDA graphs do (see class docstring).
        with forward_context(
            ForwardContext(attn_backend=self.model_runner.attn_backend)
        ):
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)
            if self.enable_torch_compile:
                self._mark_dynamic_prefill_metadata()
            with torch.no_grad():
                self.model_runner.tp_group.barrier()
                if self._prefill_uses_eager_logits_tail:
                    original_forward = self._prefill_graph_model.forward

                    def replay_forward(*args, **kwargs):
                        return compiled_fn(*args, **kwargs)

                    self._prefill_graph_model.forward = replay_forward
                    try:
                        return self.model_runner.model.forward(
                            forward_batch.input_ids,
                            forward_batch.positions,
                            forward_batch,
                        )
                    finally:
                        self._prefill_graph_model.forward = original_forward
                return compiled_fn(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                )

    def _warmup_prefill_callable(
        self,
        compiled_fn: Callable,
        num_tokens: int,
        mark_token_axis_dynamic: bool,
        warmup_bs_values: Optional[list[int]] = None,
        mark_request_axis_dynamic: bool = True,
        save_dynamic_template: bool = True,
    ) -> None:
        for bs in warmup_bs_values or self.prefill_warmup_bs_values:
            if bs > num_tokens:
                continue
            fb = self._build_prefill_dummy_forward_batch(num_tokens, bs)
            if self.enable_torch_compile and mark_request_axis_dynamic:
                self._mark_dynamic_request_axis(fb)
            if self.enable_torch_compile and mark_token_axis_dynamic:
                self._mark_dynamic_token_axis(fb)
            if mark_token_axis_dynamic and save_dynamic_template:
                self.prefill_dynamic_batch_template = fb
            # Run twice: the first call triggers tracing/compilation, the second
            # exercises the already-compiled artifact (mirrors
            # capture_one_batch_size above).
            for _ in range(2):
                self._run_prefill_once(compiled_fn, fb)

    def _compile_prefill_bucket(self, num_tokens: int) -> Callable:
        with patch_model(
            self._prefill_graph_model,
            self.enable_torch_compile,
            num_tokens=num_tokens,
            tp_group=self.model_runner.tp_group,
            dynamic=True,
            model_forward=(
                self._prefill_graph_model.forward
                if self._prefill_uses_eager_logits_tail
                else None
            ),
        ) as forward:
            dynamic_warmup_bs = [
                bs for bs in self.prefill_warmup_bs_values if bs >= 2
            ] or [1]
            self._warmup_prefill_callable(
                forward,
                num_tokens,
                mark_token_axis_dynamic=False,
                warmup_bs_values=dynamic_warmup_bs,
            )
        return forward

    def _compile_prefill_bucket_bs1(self, num_tokens: int) -> Callable:
        with patch_model(
            self._prefill_graph_model,
            self.enable_torch_compile,
            num_tokens=num_tokens,
            tp_group=self.model_runner.tp_group,
            dynamic=False,
            model_forward=(
                self._prefill_graph_model.forward
                if self._prefill_uses_eager_logits_tail
                else None
            ),
        ) as forward:
            self._warmup_prefill_callable(
                forward,
                num_tokens,
                mark_token_axis_dynamic=False,
                warmup_bs_values=[1],
                mark_request_axis_dynamic=False,
            )
        return forward

    def _prepare_dynamic_prefill_batch(
        self, forward_batch: ForwardBatch
    ) -> ForwardBatch:
        template = self.prefill_dynamic_batch_template
        if template is None:
            return forward_batch

        dynamic_forward_batch = template
        for name, value in vars(forward_batch).items():
            if isinstance(value, torch.Tensor):
                setattr(dynamic_forward_batch, name, value)
        dynamic_forward_batch.forward_mode = ForwardMode.EXTEND
        dynamic_forward_batch.global_forward_mode = ForwardMode.EXTEND
        dynamic_forward_batch.batch_size = forward_batch.batch_size
        dynamic_forward_batch.seq_lens_sum = forward_batch.seq_lens_sum
        dynamic_forward_batch.mm_inputs = None
        self._refresh_dynamic_forward_batch_metadata(
            dynamic_forward_batch,
            forward_batch,
            include_extend_metadata=True,
        )
        return dynamic_forward_batch

    def _compile_prefill_dynamic_fallback(self) -> Callable:
        with patch_model(
            self._prefill_graph_model,
            self.enable_torch_compile,
            num_tokens=self.prefill_max_num_tokens,
            tp_group=self.model_runner.tp_group,
            dynamic=True,
            model_forward=(
                self._prefill_graph_model.forward
                if self._prefill_uses_eager_logits_tail
                else None
            ),
        ) as forward:
            # Warm up at two distinct num_tokens (as well as two distinct bs) so
            # both axes are established as symbolic, not just bs.
            for num_tokens in sorted(
                {self.prefill_max_num_tokens, max(self.prefill_max_num_tokens // 2, 1)}
            ):
                dynamic_warmup_bs = [
                    bs for bs in self.prefill_warmup_bs_values if bs >= 2
                ] or [1]
                self._warmup_prefill_callable(
                    forward,
                    num_tokens,
                    mark_token_axis_dynamic=True,
                    warmup_bs_values=dynamic_warmup_bs,
                )
        return forward

    def _compile_prefill_dynamic_batch_one(self) -> Callable:
        with patch_model(
            self._prefill_graph_model,
            self.enable_torch_compile,
            num_tokens=self.prefill_max_num_tokens,
            tp_group=self.model_runner.tp_group,
            dynamic=True,
            model_forward=(
                self._prefill_graph_model.forward
                if self._prefill_uses_eager_logits_tail
                else None
            ),
        ) as forward:
            for num_tokens in sorted(
                {self.prefill_max_num_tokens, max(self.prefill_max_num_tokens // 2, 1)}
            ):
                self._warmup_prefill_callable(
                    forward,
                    num_tokens,
                    mark_token_axis_dynamic=True,
                    warmup_bs_values=[1],
                    mark_request_axis_dynamic=False,
                    save_dynamic_template=False,
                )
        return forward

    def _rewarm_decode_dynamic_graphs(self) -> None:
        """Re-establish decode variants after prefill warmup.

        Decode and prefill use separate compiled callables, but Dynamo guards
        the shared model forward code by ``ForwardMode``. Prefill capture runs
        after the initial decode capture, so warm one representative decode
        shape once more at the end of startup before requests can switch back
        to DECODE. The initial decode capture already covers every bucket.
        """
        representative_bs = self.dynamic_capture_bs
        if representative_bs < 2:
            return
        for skip_cross_attention, compiled_fn in self.decode_dynamic_graphs.items():
            self.capture_one_batch_size(
                representative_bs,
                compiled_fn,
                skip_cross_attention=skip_cross_attention,
                mark_dynamic=True,
                save_captured_batch=False,
            )

    def _compile_decode_dynamic_graphs(self) -> None:
        if not (
            self.is_generation
            and self.enable_dynamic_graph
            and self.enable_torch_compile
        ):
            return

        log_info_on_rank0(logger, "Compiling CPU decode dynamic fallback graph last")
        self.decode_dynamic_graphs[True] = self._compile_decode_dynamic_graph(
            skip_cross_attention=True
        )
        if self.is_encoder_decoder:
            self.decode_dynamic_graphs[False] = self._compile_decode_dynamic_graph(
                skip_cross_attention=False
            )
        self._rewarm_decode_dynamic_graphs()
        self._reset_mamba_cache_if_needed()

    def capture_prefill(self) -> None:
        capture_range = (
            tqdm.tqdm(list(reversed(self.capture_num_tokens)))
            if get_parallel().tp_rank == 0
            else reversed(self.capture_num_tokens)
        )
        for num_tokens in capture_range:
            if get_parallel().tp_rank == 0:
                capture_range.set_description(
                    f"Capturing prefill num_tokens ({num_tokens=})"
                )
            self.prefill_graphs[num_tokens] = self._compile_prefill_bucket(num_tokens)
            self.prefill_graphs_bs1[num_tokens] = self._compile_prefill_bucket_bs1(
                num_tokens
            )

        if self.enable_dynamic_graph:
            log_info_on_rank0(logger, "Compiling CPU prefill dynamic fallback graph")
            self.prefill_dynamic_graph = self._compile_prefill_dynamic_fallback()
            self.prefill_dynamic_graph_bs1 = self._compile_prefill_dynamic_batch_one()

        self._compile_decode_dynamic_graphs()
        # Prefill warmups run after the decode capture reset and mutate the
        # recurrent cache again. Clear all capture-time state before serving.
        self._reset_mamba_cache_if_needed()

    # ------------------------------------------------------------------
    # Prefill (EXTEND) graph: runtime
    # ------------------------------------------------------------------

    def _can_run_prefill_graph(self, forward_batch: ForwardBatch) -> bool:
        if not self.capture_num_tokens:
            return False
        if forward_batch.return_logprob:
            return False
        if forward_batch.input_embeds is not None:
            return False
        if getattr(forward_batch, "replace_embeds", None) is not None:
            return False
        if forward_batch.contains_mm_inputs():
            return False
        if forward_batch.batch_size > self.prefill_max_bs:
            return False

        raw_num_tokens = forward_batch.input_ids.shape[0]
        if raw_num_tokens > self.prefill_max_num_tokens:
            return self.prefill_dynamic_graph is not None
        if raw_num_tokens in self.prefill_graphs:
            return True
        return (
            self._get_prefill_padding_bucket(raw_num_tokens) is not None
            or self.prefill_dynamic_graph is not None
        )

    def _get_prefill_padding_bucket(self, raw_num_tokens: int) -> Optional[int]:
        if self.disable_padding or raw_num_tokens > self.prefill_max_num_tokens:
            return None

        idx = bisect.bisect_left(self.capture_num_tokens, raw_num_tokens)
        if idx >= len(self.capture_num_tokens):
            return None

        candidate = self.capture_num_tokens[idx]
        diff = candidate - raw_num_tokens
        if diff / float(raw_num_tokens) <= 0.20 or diff <= 64:
            return candidate
        return None

    def _execute_prefill_graph(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        assert (
            pp_proxy_tensors is None
        ), "PPProxyTensors is not supported in CPUGraphRunner's prefill path yet."

        if (
            forward_batch.mm_inputs is not None
            and not forward_batch.contains_mm_inputs()
        ):
            forward_batch = copy.copy(forward_batch)
            forward_batch.mm_inputs = None

        raw_num_tokens = forward_batch.input_ids.shape[0]
        use_batch_one_graph = forward_batch.batch_size == 1
        graph_map = (
            self.prefill_graphs_bs1 if use_batch_one_graph else self.prefill_graphs
        )
        compiled_fn = graph_map.get(raw_num_tokens)
        using_dynamic_fallback = False

        if compiled_fn is None:
            padded_num_tokens = self._get_prefill_padding_bucket(raw_num_tokens)

            if padded_num_tokens is not None and padded_num_tokens in graph_map:
                compiled_fn = graph_map[padded_num_tokens]
                pad_len = padded_num_tokens - raw_num_tokens
                padded_fb = copy.copy(forward_batch)
                padded_fb.input_ids = F.pad(
                    forward_batch.input_ids, (0, pad_len), value=0
                )
                padded_fb.positions = F.pad(
                    forward_batch.positions, (0, pad_len), value=0
                )
                if forward_batch.mrope_positions is not None:
                    padded_fb.mrope_positions = F.pad(
                        forward_batch.mrope_positions, (0, pad_len), value=0
                    )
                if forward_batch.out_cache_loc is not None:
                    padded_fb.out_cache_loc = F.pad(
                        forward_batch.out_cache_loc, (0, pad_len), value=0
                    )
                forward_batch = padded_fb
            else:
                using_dynamic_fallback = True
                if use_batch_one_graph and self.prefill_dynamic_graph_bs1 is not None:
                    compiled_fn = self.prefill_dynamic_graph_bs1
                else:
                    compiled_fn = self.prefill_dynamic_graph
                    forward_batch = self._prepare_dynamic_prefill_batch(forward_batch)

        if self.enable_torch_compile and not use_batch_one_graph:
            self._mark_dynamic_request_axis(forward_batch)
        if self.enable_torch_compile and using_dynamic_fallback:
            self._mark_dynamic_token_axis(forward_batch)

        return self._run_prefill_once(compiled_fn, forward_batch)


CPU_GRAPH_CAPTURE_FAILED_MSG = (
    "Possible solutions:\n"
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "2. set --torch-compile-max-bs to a smaller value (e.g., 8)\n"
    "3. disable torch compile by not using --enable-torch-compile\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)
