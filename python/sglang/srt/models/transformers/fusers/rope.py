# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import ast
import inspect
import types
from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch import fx, nn
from triton.language.extra import libdevice

from sglang.srt.utils.custom_op import register_custom_op

from .source import ReplaceNodes, compile_forward, read_forward


def _rotate_half(x):
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _pair_reference(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _graph_signature(function):
    graph = fx.symbolic_trace(function, concrete_args={"unsqueeze_dim": 1}).graph
    signatures = {}
    for node in graph.nodes:
        signatures[node] = (
            node.op,
            node.target if node.op != "placeholder" else len(signatures),
            fx.map_arg(node.args, lambda value: signatures[value]),
            fx.map_arg(node.kwargs, lambda value: signatures[value]),
        )
    return next(signatures[node] for node in graph.nodes if node.op == "output")


@lru_cache(maxsize=None)
def _matches_pair(function):
    try:
        function = inspect.unwrap(function)
        parameters = list(inspect.signature(function).parameters.values())
        if (
            len(parameters) != 5
            or parameters[-1].name != "unsqueeze_dim"
            or parameters[-1].default != 1
        ):
            return False
        return _graph_signature(function) == _graph_signature(_pair_reference)
    except (OSError, TypeError, ValueError, fx.proxy.TraceError):
        return False


@triton.jit
def _pair_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    qo_ptr,
    ko_ptr,
    qb: tl.constexpr,
    qh: tl.constexpr,
    qt: tl.constexpr,
    kb: tl.constexpr,
    kh: tl.constexpr,
    kt: tl.constexpr,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    tokens: tl.constexpr,
    width: tl.constexpr,
    broadcast_batch: tl.constexpr,
    block: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    batch, position = token // tokens, token % tokens
    columns = tl.arange(0, block)
    mask = columns < width
    paired = (columns + width // 2) % width
    angle_offset = (position if broadcast_batch else token) * width + columns
    cosine = tl.load(cos_ptr + angle_offset, mask, 0).to(tl.float32)
    sine = tl.load(sin_ptr + angle_offset, mask, 0).to(tl.float32)
    if head < q_heads:
        offset = batch * qb + head * qh + position * qt
        value = tl.load(q_ptr + offset + columns, mask, 0).to(tl.float32)
        partner = tl.load(q_ptr + offset + paired, mask, 0).to(tl.float32)
        partner = tl.where(columns < width // 2, -partner, partner)
        first = (value * cosine).to(qo_ptr.dtype.element_ty).to(tl.float32)
        second = (partner * sine).to(qo_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            qo_ptr + ((batch * tokens + position) * q_heads + head) * width + columns,
            first + second,
            mask,
        )
    else:
        head = head - q_heads
        offset = batch * kb + head * kh + position * kt
        value = tl.load(k_ptr + offset + columns, mask, 0).to(tl.float32)
        partner = tl.load(k_ptr + offset + paired, mask, 0).to(tl.float32)
        partner = tl.where(columns < width // 2, -partner, partner)
        first = (value * cosine).to(ko_ptr.dtype.element_ty).to(tl.float32)
        second = (partner * sine).to(ko_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            ko_ptr + ((batch * tokens + position) * k_heads + head) * width + columns,
            first + second,
            mask,
        )


def _token_major(x):
    batch, heads, tokens, width = x.shape
    return x.new_empty((batch, tokens, heads, width)).transpose(1, 2)


def _fake_pair(q, k, cos, sin):
    return _token_major(q), _token_major(k)


@register_custom_op(fake_impl=_fake_pair)
def transformers_rotary_pair(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output_q, output_k = _fake_pair(q, k, cos, sin)
    batch, heads, tokens, width = q.shape
    if batch * tokens:
        _pair_kernel[(batch * tokens, heads + k.shape[1])](
            q,
            k,
            cos,
            sin,
            output_q,
            output_k,
            *q.stride()[:3],
            *k.stride()[:3],
            heads,
            k.shape[1],
            tokens,
            width,
            cos.shape[0] == 1,
            triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output_q, output_k


class PairedRotaryEmbedding(nn.Module):
    def forward(self, q, k, cos, sin):
        if (
            q.is_cuda
            and k.is_cuda
            and cos.is_cuda
            and sin.is_cuda
            and q.device == k.device == cos.device == sin.device
            and q.ndim == k.ndim == 4
            and cos.ndim == sin.ndim == 3
            and q.shape[0] == k.shape[0]
            and q.shape[2:] == k.shape[2:]
            and cos.shape == sin.shape
            and cos.shape[1:] == q.shape[2:]
            and cos.shape[0] in (1, q.shape[0])
            and q.stride(-1) == k.stride(-1) == 1
            and 0 < q.shape[-1] <= 512
            and q.shape[-1] % 2 == 0
            and q.dtype == k.dtype == cos.dtype == sin.dtype
            and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            return transformers_rotary_pair(q, k, cos.contiguous(), sin.contiguous())
        return _pair_reference(q, k, cos, sin)


def fuse_rotary_embedding(module):
    if hasattr(module, "_sglang_rotary"):
        return False
    config = getattr(module, "config", None)
    if config is not None:
        parameters = getattr(config, "rope_parameters", None) or {}
        if getattr(config, "partial_rotary_factor", 1.0) != 1.0:
            return False
        if parameters.get("rope_type", "default") not in {
            "default",
            "linear",
            "llama3",
        }:
            return False
        if (
            any(isinstance(value, dict) for value in parameters.values())
            or "mrope_section" in parameters
        ):
            return False
    try:
        function, original = read_forward(module)
    except (OSError, TypeError, SyntaxError):
        return False
    matches = []
    for node in ast.walk(function):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or len(node.args) != 4
            or node.keywords
        ):
            continue
        target = original.__globals__.get(node.func.id)
        if callable(target) and _matches_pair(target):
            matches.append(node)
    if len(matches) != 1:
        return False
    call = matches[0]
    replacement = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr="_sglang_rotary",
            ctx=ast.Load(),
        ),
        args=call.args,
        keywords=[],
    )
    ReplaceNodes({id(call): replacement}).visit(function)
    try:
        forward = compile_forward(function, original)
    except (TypeError, ValueError, SyntaxError):
        return False
    module._sglang_rotary = PairedRotaryEmbedding()
    module.forward = types.MethodType(forward, module)
    return True


@triton.jit
def _cached_angles_kernel(
    positions_ptr,
    cache_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    half_width: tl.constexpr,
    cache_length: tl.constexpr,
    scale: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    position = tl.load(positions_ptr + token)
    columns = tl.arange(0, block)
    mask = columns < half_width
    if (position >= 0) & (position < cache_length):
        cosine = tl.load(cache_ptr + position * half_width * 2 + columns, mask, 0)
        sine = tl.load(
            cache_ptr + position * half_width * 2 + half_width + columns, mask, 0
        )
    else:
        angles = position.to(tl.float32) * tl.load(inv_freq_ptr + columns, mask, 0)
        cosine, sine = libdevice.cos(angles) * scale, libdevice.sin(angles) * scale
    offsets = token * half_width * 2 + columns
    tl.store(cos_ptr + offsets, cosine, mask)
    tl.store(cos_ptr + offsets + half_width, cosine, mask)
    tl.store(sin_ptr + offsets, sine, mask)
    tl.store(sin_ptr + offsets + half_width, sine, mask)


def _fake_angles(x, positions, cache, inv_freq, scale):
    shape = (*positions.shape, inv_freq.numel() * 2)
    return x.new_empty(shape), x.new_empty(shape)


@register_custom_op(fake_impl=_fake_angles)
def transformers_cached_rotary_angles(
    x: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    inv_freq: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine, sine = _fake_angles(x, positions, cache, inv_freq, scale)
    if positions.numel():
        _cached_angles_kernel[(positions.numel(),)](
            positions,
            cache,
            inv_freq,
            cosine,
            sine,
            inv_freq.numel(),
            cache.shape[0],
            scale,
            triton.next_power_of_2(inv_freq.numel()),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return cosine, sine


class StaticRotaryEmbedding(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.config = original.config
        self.attention_scaling = float(original.attention_scaling)
        self.register_buffer(
            "inv_freq", original.inv_freq.detach().float().clone(), persistent=False
        )
        length = min(original.config.max_position_embeddings, 8192)
        positions = torch.arange(
            length, device=self.inv_freq.device, dtype=torch.float32
        )
        angles = positions[:, None] * self.inv_freq[None, :]
        cache = torch.cat((angles.cos(), angles.sin()), -1) * self.attention_scaling
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _apply(self, fn, recurse=True):
        buffers = dict(self._buffers)
        result = super()._apply(fn, recurse=recurse)
        for name, original in buffers.items():
            converted = self._buffers[name]
            if converted.dtype != torch.float32:
                self._buffers[name] = original.to(
                    device=converted.device, dtype=torch.float32
                )
        return result

    def forward(self, x, position_ids):
        if (
            x.is_cuda
            and position_ids.is_cuda
            and self.cos_sin_cache.is_cuda
            and self.cos_sin_cache.dtype == self.inv_freq.dtype == torch.float32
            and position_ids.ndim == 2
            and position_ids.dtype in (torch.int32, torch.int64)
        ):
            return transformers_cached_rotary_angles(
                x,
                position_ids.contiguous(),
                self.cos_sin_cache,
                self.inv_freq,
                self.attention_scaling,
            )
        if (
            not torch.compiler.is_compiling()
            and position_ids.numel()
            and position_ids.min() >= 0
            and position_ids.max() < self.cos_sin_cache.shape[0]
        ):
            angles = self.cos_sin_cache[position_ids]
            cosine, sine = angles.chunk(2, dim=-1)
        else:
            angles = position_ids.float()[..., None] * self.inv_freq.float()
            cosine, sine = (
                angles.cos() * self.attention_scaling,
                angles.sin() * self.attention_scaling,
            )
        return torch.cat((cosine, cosine), -1).to(x.dtype), torch.cat(
            (sine, sine), -1
        ).to(x.dtype)


def replace_rotary_embedding(module):
    if not type(module).__module__.startswith("transformers.models.") or not hasattr(
        module, "inv_freq"
    ):
        return module
    if getattr(module, "rope_type", None) not in {"default", "linear", "llama3"}:
        return module
    config = getattr(module, "config", None)
    if config is None or getattr(config, "partial_rotary_factor", 1.0) != 1.0:
        return module
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    if (
        module.inv_freq.ndim != 1
        or module.inv_freq.numel() * 2 != head_dim
        or module.inv_freq.device.type == "meta"
    ):
        return module
    if not isinstance(getattr(module, "attention_scaling", None), (int, float)):
        return module
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    try:
        function, _ = read_forward(module)
        source = ast.dump(ast.Module(body=function.body, type_ignores=[]))
        for reference_type in (Qwen3RotaryEmbedding, LlamaRotaryEmbedding):
            reference, _ = read_forward(reference_type.__new__(reference_type))
            if source == ast.dump(ast.Module(body=reference.body, type_ignores=[])):
                return StaticRotaryEmbedding(module)
    except (OSError, TypeError, SyntaxError):
        return module
    return module
