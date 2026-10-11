"""Warm selected linear-attention kernels with the initialized backend's layout.

Like model-local precompile helpers, this calls the real operator on scratch
inputs. It runs after pool/backend construction because state strides and the
selected dispatcher are not available at the earlier model-load hook.
"""

import logging
import time

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

logger = logging.getLogger(__name__)


def _sizes(value):
    sizes = tuple(dict.fromkeys(int(item) for item in value.split(",") if item.strip()))
    if any(size <= 0 for size in sizes):
        raise ValueError("Linear attention precompile sizes must be positive")
    return sizes


@torch.no_grad()
def _precompile(kernel, layer, pool, dtype, mode, size, checkpoint_interval):
    is_kda = isinstance(kernel, TritonKDAKernel)
    batch = size if mode == "decode" else 1
    qkv = torch.full(
        (size, layer.q_dim + layer.k_dim + layer.v_dim),
        0.1,
        dtype=dtype,
        device=pool.device,
    )
    q, k, v = qkv.split((layer.q_dim, layer.k_dim, layer.v_dim), dim=-1)
    state = torch.empty_strided(
        (batch, *pool.shape[1:]), pool.stride(), dtype=pool.dtype, device=pool.device
    ).zero_()
    kwargs = {
        "q": q.view(1, size, layer.num_q_heads, layer.head_k_dim),
        "k": k.view(1, size, layer.num_k_heads, layer.head_k_dim),
        "v": v.view(1, size, layer.num_v_heads, layer.head_v_dim),
        "ssm_states": state,
        "cache_indices": torch.arange(batch, dtype=torch.int32, device=pool.device),
        "query_start_loc": torch.tensor(
            list(range(batch + 1)) if mode == "decode" else [0, size],
            dtype=torch.int32,
            device=pool.device,
        ),
        "A_log": layer.A_log,
        "dt_bias": layer.dt_bias,
    }
    gate_shape = (1, size, layer.num_v_heads)
    zeros = lambda shape: torch.zeros(shape, dtype=dtype, device=pool.device)
    if mode == "decode":
        kernel.decode(**kwargs, a=zeros(gate_shape), b=zeros(gate_shape))
        return 1
    if is_kda:
        kwargs.update(
            g=zeros((*gate_shape, layer.head_k_dim)),
            beta=zeros(gate_shape),
            lower_bound=layer.lower_bound,
            beta_is_raw=True,
        )
    else:
        kwargs.update(g=zeros(gate_shape).float() - 1, beta=zeros(gate_shape) + 0.5)
    kernel.extend(**kwargs)
    state.zero_()
    if is_kda:
        kernel.extend(
            **kwargs,
            return_intermediate_states=True,
            track_state=torch.zeros_like(state, dtype=torch.float32),
            track_chunk_idx=torch.tensor([0], dtype=torch.int32, device=pool.device),
        )
        return 2
    if size >= checkpoint_interval:
        checkpoints = size // checkpoint_interval
        kernel.extend(
            **kwargs,
            state_checkpoint_cu_starts=torch.tensor(
                [0, checkpoints], dtype=torch.int64, device=pool.device
            ),
            num_state_checkpoints=checkpoints,
            state_checkpoint_every_n_tokens=checkpoint_interval,
        )
        return 2
    return 1


@torch.no_grad()
def precompile_linear_attention(model_runner, backend):
    """Called by KDA/GDN backend constructors; identical layer shapes run once."""
    if torch.device(model_runner.device).type != "cuda":
        return []
    selections = (
        ("prefill", _sizes(envs.SGLANG_LINEAR_ATTN_PREPARE_PREFILL_TOKENS.get())),
        ("decode", _sizes(envs.SGLANG_LINEAR_ATTN_PREPARE_DECODE_BATCH_SIZES.get())),
    )
    dispatcher = backend.kernel_dispatcher
    seen, results = set(), []
    for layer in model_runner.model.modules():
        if not isinstance(layer, RadixLinearAttention):
            continue
        pool = model_runner.req_to_token_pool.mamba2_layer_cache(
            layer.layer_id
        ).temporal
        signature = (
            layer.num_q_heads,
            layer.num_k_heads,
            layer.num_v_heads,
            layer.head_k_dim,
            layer.head_v_dim,
            layer.lower_bound,
            pool.dtype,
            tuple(pool.shape[1:]),
            pool.stride(),
            layer.A_log.dtype,
            tuple(layer.A_log.shape),
            layer.dt_bias.dtype,
            tuple(layer.dt_bias.shape),
        )
        for mode, sizes in selections:
            for size in sizes:
                if mode == "prefill":
                    effective = getattr(dispatcher, "effective_extend_kernel", None)
                    kernel = (
                        effective(layer.lower_bound, size)
                        if effective
                        else dispatcher.extend_kernel
                    )
                else:
                    kernel = dispatcher.decode_kernel
                if not (
                    isinstance(kernel, FlashInferGDNKernel)
                    or (mode == "prefill" and isinstance(kernel, TritonKDAKernel))
                ):
                    continue
                key = (id(kernel), mode, size, signature)
                if key in seen:
                    continue
                seen.add(key)
                torch.cuda.synchronize(pool.device)
                start = time.perf_counter()
                cases = _precompile(
                    kernel,
                    layer,
                    pool,
                    model_runner.dtype,
                    mode,
                    size,
                    backend.mamba_chunk_size,
                )
                torch.cuda.synchronize(pool.device)
                result = {
                    "kernel": type(kernel).__name__,
                    "mode": mode,
                    "size": size,
                    "cases": cases,
                    "seconds": time.perf_counter() - start,
                }
                results.append(result)
                logger.info("Precompiled linear attention: %s", result)
    return results
