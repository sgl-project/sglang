"""ROCm: the Inkling2 MTP draft's context K/V store in one launch per layer, its
Markov RNN step in two launches from a persistent zero initial state, and the f_rnn
store after verify in one launch."""

import torch

from sglang.kernels.ops.inkling2.mtp_hip import (
    ctx_kv_store,
    mix_rmsnorm_last,
    rnn_step,
    rnn_step_fits,
)
from sglang.kernels.ops.inkling2.rmsnorm_triton import rmsnorm as triton_rmsnorm
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers import layernorm
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool


def _ctx_kv_write_target(pool, attn, cache_loc_2d, swa_loc, dtype):
    """(k_buffer, v_buffer, loc_2d) where ctx_kv_store writes a draft layer's context
    rows as k_norm_ctx, apply_k_rope and set_kv_buffer_prefix_valid would (ROCm), or
    None."""
    rope = attn.rotary_emb
    if not (
        isinstance(attn.k_norm_ctx, TritonRMSNorm)
        and attn.k_norm_ctx.uses_triton()
        and type(rope) is RotaryEmbedding
        and rope.use_fallback_kernel
        and rope.is_neox_style
        and rope.rotary_dim == rope.head_size == attn.head_dim
        and rope.cos_sin_cache.dtype == torch.float32
    ):
        return None
    # the sub-pool and locations write_target_hidden_kv's set_kv_buffer_prefix_valid
    # would use
    layer_id, loc = attn.attn.layer_id, cache_loc_2d
    if isinstance(pool, SWAKVPool):
        layer_id, is_local = pool.layers_mapping[layer_id]
        pool = pool.swa_kv_pool if is_local else pool.full_kv_pool
        if is_local:
            loc = swa_loc
    if (
        loc is None
        or type(pool) is not MHATokenToKVPool
        or pool.is_quantized_kv_cache
        or pool.use_hnd
        or pool.dtype != dtype
        or pool.store_dtype != dtype
    ):
        return None
    index = layer_id - pool.start_layer
    k_buffer, v_buffer = pool.k_buffer[index], pool.v_buffer[index]
    rows = (attn.num_kv_heads, attn.head_dim)
    if not (
        k_buffer.dim() == v_buffer.dim() == 3
        and k_buffer.shape[1:] == v_buffer.shape[1:] == rows
        and k_buffer.stride()[1:] == v_buffer.stride()[1:] == (attn.head_dim, 1)
    ):
        return None
    return k_buffer, v_buffer, loc.view(cache_loc_2d.shape)


def store_context_kv(pool, attn, kv, positions, cache_loc_2d, swa_loc, commit_lens):
    """ctx_kv_store of a draft layer's context rows kv; False when the pool or the
    layer's norm and rope rule it out."""
    target = _ctx_kv_write_target(pool, attn, cache_loc_2d, swa_loc, kv.dtype)
    if target is None:
        return False
    norm = attn.k_norm_ctx
    ctx_kv_store(
        kv,
        positions,
        norm.weight,
        norm.variance_epsilon,
        attn.rotary_emb.cos_sin_cache,
        *target,
        commit_lens,
    )
    return True


def run_rnn_step(head, state, prev_tokens, hidden, f_rnn):
    """Inkling2MTPHead._rnn_step in two launches (rnn_step) instead of rnn_inputs,
    hipBLASLt's split-K GEMM and its reduction, and rnn_gates; None when
    rnn_step_fits rules it out (more than a few rows)."""
    proj = head.rnn_proj
    if not rnn_step_fits(state, hidden, proj.weight, proj.bias):
        return None
    return rnn_step(
        state, prev_tokens, head.markov_w1.weight, hidden, f_rnn, proj.weight, proj.bias
    )


# Rows of the persistent zero initial RNN state (larger draft batches fill their own).
_ZERO_STATE_ROWS = 1024


def attach_zero_state(head, weight):
    """Makes Inkling2MTPHead's initial RNN state once, when the fold is attached,
    rather than filling it in every draft (no step writes into its input state)."""
    head._hip_zero_state = weight.new_zeros((_ZERO_STATE_ROWS, head.markov_rank))


def zero_state(head, hidden_states):
    """The initial RNN state for hidden_states' rows from attach_zero_state's
    buffer, or None when it does not cover them."""
    zeros = getattr(head, "_hip_zero_state", None)
    rows = hidden_states.shape[0]
    if (
        zeros is None
        or rows > zeros.shape[0]
        or zeros.dtype != hidden_states.dtype
        or zeros.device != hidden_states.device
    ):
        return None
    return zeros[:rows]


def store_last_f_rnn(model, req_pool_indices, target_hidden, commit_lens):
    """store_f_rnn of row commit_lens[r] - 1 of each request's block of
    target_hidden (requests, block, features): the gather, the norm and the copy."""
    assert model._f_rnn_buf is not None and model._fuse_alpha is not None
    reqs, width = target_hidden.shape[:2]
    targets = target_hidden.view(
        reqs * width, model.num_context_features, model.config.hidden_size
    )
    norm = model.markov_head.dflare_norm
    mix_rmsnorm_last(
        targets,
        width,
        commit_lens,
        req_pool_indices.view(-1),
        model._fuse_alpha[-1].to(targets.dtype),
        norm.weight,
        norm.variance_epsilon,
        model._f_rnn_buf,
    )


class TritonRMSNorm(RMSNorm):
    """The DSpark draft's RMSNorm on ROCm without vLLM's or aiter's norms: the plain
    and fused-residual norms run rmsnorm_triton (forward_native's semantics), which
    ctx_kv_store's K norm reproduces bitwise; the rest take RMSNorm.forward_hip."""

    def uses_triton(self, residual=None, post_residual_addition=None) -> bool:
        return (
            not layernorm._has_vllm_rms_norm
            and post_residual_addition is None
            and not self.cast_x_before_out_mul
            and self.variance_size_override is None
            and self.override_orig_dtype is None
            and not (residual is not None and self.fp32_residual)
            and not is_batch_invariant_mode_enabled()
        )

    def forward_hip(
        self, x, residual=None, post_residual_addition=None, quant_linear=None
    ):
        if (
            self.uses_triton(residual, post_residual_addition)
            and x.shape[-1] == self.hidden_size
            and x.numel()
        ):
            return triton_rmsnorm(x, self.weight.data, self.variance_epsilon, residual)
        return super().forward_hip(x, residual, post_residual_addition, quant_linear)


def use_triton_norms(model):
    """The draft's RMSNorms become TritonRMSNorm (other models keep RMSNorm)."""
    for module in model.modules():
        if type(module) is RMSNorm:
            module.__class__ = TritonRMSNorm
