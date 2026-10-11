"""Operator-only startup preparation for eager linear attention.

This intentionally executes kernels during init, using private scratch state.
It does not run a model forward, capture graphs, patch a compiler, or write to
the request pool. SGLang also persists supported GDN prefill cache entries;
other compiler/autotuner caches remain owned by their libraries.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


def parse_sizes(value: str) -> tuple[int, ...]:
    """An empty list disables preparation for that mode."""
    sizes = tuple(
        dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip())
    )
    if any(size <= 0 for size in sizes):
        raise ValueError("Linear attention preparation sizes must be positive")
    return sizes


def linear_backends(backend, mode):
    """Resolve mode, hybrid and overlap wrappers without constructing backends."""
    if backend is None:
        return
    if hasattr(backend, "kernel_dispatcher"):
        yield backend
    elif hasattr(backend, "linear_attn_backend"):
        yield from linear_backends(backend.linear_attn_backend, mode)
    elif hasattr(backend, "prefill_backend"):
        yield from linear_backends(getattr(backend, f"{mode}_backend"), mode)
    elif hasattr(backend, "primary"):
        yield from linear_backends(backend.primary, mode)
        if mode == "prefill":
            for child in backend.children:
                yield from linear_backends(child, mode)


@dataclass
class LinearAttentionPrepareSpec:
    num_q_heads: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    dtype: torch.dtype
    state: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    lower_bound: float | None = None
    checkpoint_interval: int = 64

    def signature(self) -> tuple:
        def layout(tensor):
            return (tensor.dtype, tuple(tensor.shape), tensor.stride())

        return (
            self.num_q_heads,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.dtype,
            self.state.device,
            self.state.dtype,
            tuple(self.state.shape[1:]),
            self.state.stride(),
            layout(self.A_log),
            layout(self.dt_bias),
            self.lower_bound,
            self.checkpoint_interval,
        )


def make_prepare_inputs(
    spec: LinearAttentionPrepareSpec, mode: str, size: int, *, kind: str
):
    """Build one bounded case with the serving QKV view layout and state strides.

    Parameters are read-only. Only the newly allocated state is mutated, even
    with a strided/unified serving pool. Prefill uses one packed sequence.
    """
    if mode not in ("prefill", "decode") or kind not in ("kda", "gdn") or size <= 0:
        raise ValueError("Unsupported linear attention preparation case")
    device = spec.state.device
    batch = size if mode == "decode" else 1
    tokens = size
    dims = (
        spec.num_q_heads * spec.head_k_dim,
        spec.num_k_heads * spec.head_k_dim,
        spec.num_v_heads * spec.head_v_dim,
    )
    qkv = torch.full((tokens, sum(dims)), 0.1, dtype=spec.dtype, device=device)
    q, k, v = qkv.split(dims, dim=-1)
    q = q.view(1, tokens, spec.num_q_heads, spec.head_k_dim)
    k = k.view(1, tokens, spec.num_k_heads, spec.head_k_dim)
    v = v.view(1, tokens, spec.num_v_heads, spec.head_v_dim)
    state = torch.empty_strided(
        (batch, *spec.state.shape[1:]),
        spec.state.stride(),
        dtype=spec.state.dtype,
        device=device,
    ).zero_()
    ends = list(range(batch + 1)) if mode == "decode" else [0, tokens]
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "ssm_states": state,
        "cache_indices": torch.arange(batch, dtype=torch.int32, device=device),
        "query_start_loc": torch.tensor(ends, dtype=torch.int32, device=device),
        "A_log": spec.A_log,
        "dt_bias": spec.dt_bias,
    }
    gate_shape = (1, tokens, spec.num_v_heads)
    if mode == "decode":
        kwargs.update(
            a=torch.zeros(gate_shape, dtype=spec.dtype, device=device),
            b=torch.zeros(gate_shape, dtype=spec.dtype, device=device),
        )
    elif kind == "kda":
        kwargs.update(
            g=torch.zeros(
                (*gate_shape, spec.head_k_dim), dtype=spec.dtype, device=device
            ),
            beta=torch.zeros(gate_shape, dtype=spec.dtype, device=device),
            lower_bound=spec.lower_bound,
            beta_is_raw=True,
        )
    else:
        # GDN extend accepts activated log-gate / beta, unlike decode.
        kwargs.update(
            g=torch.full(gate_shape, -1.0, dtype=torch.float32, device=device),
            beta=torch.full(gate_shape, 0.5, dtype=spec.dtype, device=device),
        )
    return kwargs


@torch.no_grad()
def prepare_kernel(kernel, spec, mode, sizes, *, kind):
    count = 0
    for size in sizes:
        kwargs = make_prepare_inputs(spec, mode, size, kind=kind)
        call = kernel.decode if mode == "decode" else kernel.extend
        call(**kwargs)
        count += 1
        if mode == "prefill" and kind == "kda":
            # Prefix tracking changes constexprs even when shape stays the same.
            kwargs["ssm_states"].zero_()
            kwargs.update(
                return_intermediate_states=True,
                track_state=torch.zeros_like(kwargs["ssm_states"], dtype=torch.float32),
                track_chunk_idx=torch.tensor(
                    [0], dtype=torch.int32, device=spec.state.device
                ),
            )
            call(**kwargs)
            count += 1
        elif mode == "prefill" and kind == "gdn" and size >= spec.checkpoint_interval:
            interval = spec.checkpoint_interval
            checkpoints = size // interval
            kwargs["ssm_states"].zero_()
            kwargs.update(
                state_checkpoint_cu_starts=torch.tensor(
                    [0, checkpoints], dtype=torch.int64, device=spec.state.device
                ),
                num_state_checkpoints=checkpoints,
                state_checkpoint_every_n_tokens=interval,
            )
            call(**kwargs)
            count += 1
    torch.cuda.synchronize(spec.state.device)
    return count


@torch.no_grad()
def prepare_model_linear_attention(model_runner):
    """Called after backend construction, before graph setup or user forward."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

    prefill_sizes = parse_sizes(envs.SGLANG_LINEAR_ATTN_PREPARE_PREFILL_TOKENS.get())
    decode_sizes = parse_sizes(envs.SGLANG_LINEAR_ATTN_PREPARE_DECODE_BATCH_SIZES.get())
    if torch.device(model_runner.device).type != "cuda":
        raise ValueError("Linear attention startup preparation currently requires CUDA")
    # ModelRunner can hold distinct hybrid backends for prefill and decode.
    decode_backends = getattr(model_runner, "decode_attn_backend_group", None) or [
        model_runner.decode_attn_backend or model_runner.attn_backend
    ]
    selections = [(model_runner.attn_backend, "prefill", prefill_sizes)]
    selections.extend((backend, "decode", decode_sizes) for backend in decode_backends)
    resolved = [
        (linear, mode, sizes)
        for backend, mode, sizes in selections
        for linear in linear_backends(backend, mode)
    ]
    seen = set()
    result = []
    for layer in model_runner.model.modules():
        if not isinstance(layer, RadixLinearAttention):
            continue
        for linear, mode, sizes in resolved:
            if not sizes:
                continue
            dispatcher = linear.kernel_dispatcher
            state = model_runner.req_to_token_pool.mamba2_layer_cache(
                layer.layer_id
            ).temporal
            spec = LinearAttentionPrepareSpec(
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_k_dim,
                layer.head_v_dim,
                model_runner.dtype,
                state,
                layer.A_log,
                layer.dt_bias,
                layer.lower_bound,
                linear.mamba_chunk_size,
            )
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
                key = (id(kernel), mode, size, spec.signature())
                if key in seen:
                    continue
                seen.add(key)
                torch.cuda.synchronize(state.device)
                start = time.perf_counter()
                cases = kernel.prepare_for_inference(spec, mode, (size,))
                elapsed = time.perf_counter() - start
                item = {
                    "kernel": type(kernel).__name__,
                    "mode": mode,
                    "size": size,
                    "cases": cases,
                    "seconds": elapsed,
                }
                result.append(item)
                if cases:
                    logger.info("Linear attention init preparation: %s", item)
                else:
                    logger.warning(
                        "Linear attention init preparation unsupported: %s", item
                    )
    return result
