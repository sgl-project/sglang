# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_SOL_ATTN_HEAD_DIM = 128


def _parse_layer_ranges(spec: str | int | None) -> frozenset[int]:
    if spec is None:
        return frozenset()
    if isinstance(spec, int):
        return frozenset({spec})
    layers: set[int] = set()
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(item))
    return frozenset(layers)


def _resolve_kv_splits(q: torch.Tensor, kv_splits: int | str | None) -> int:
    if kv_splits not in (None, "auto"):
        return int(kv_splits)
    arch = tuple(torch.cuda.get_device_capability(q.device))
    if arch == (9, 0) and q.shape[1] >= 65536:
        try:
            import cuda.bindings.driver  # noqa: F401
            import cutlass.cute  # noqa: F401

            return 4
        except ImportError:
            pass
    return 1


def _get_sol_attn_runtime_config() -> dict:
    server_args = get_global_server_args()
    cfg = getattr(server_args, "attention_backend_config", None) or {}
    dense_layers = cfg.get("dense_layers", "0,1")
    sink_start = cfg.get("sink_start", 0)
    return {
        "tau": float(cfg.get("tau", 1.0)),
        "thresh_type": str(cfg.get("thresh_type", "diag")),
        "kv_splits": cfg.get("kv_splits", "auto"),
        "sink_tokens": int(cfg.get("sink_tokens", 0)),
        "sink_start": None if sink_start is None else int(sink_start),
        "dense_steps": int(cfg.get("dense_steps", 10)),
        "dense_layers": _parse_layer_ranges(dense_layers),
    }


class SolAttnBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [_SOL_ATTN_HEAD_DIM]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SOL_ATTN

    @staticmethod
    def get_impl_cls() -> type[SolAttnImpl]:
        return SolAttnImpl


class SolAttnImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        del num_heads, num_kv_heads, extra_impl_args
        if head_size != _SOL_ATTN_HEAD_DIM:
            raise ValueError(
                f"Sol-Attn requires head_size={_SOL_ATTN_HEAD_DIM}, got {head_size}"
            )
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.prefix = prefix
        self.layer_idx = self._parse_layer_idx(prefix)

    @staticmethod
    def _parse_layer_idx(prefix: str) -> int | None:
        match = re.search(r"blocks\.(\d+)", prefix)
        if match is None:
            return None
        return int(match.group(1))

    def _should_use_dense(self) -> bool:
        cfg = _get_sol_attn_runtime_config()
        try:
            from sglang.multimodal_gen.runtime.managers.forward_context import (
                get_forward_context,
            )

            step = int(get_forward_context().current_timestep)
        except AssertionError:
            step = 0
        if step < cfg["dense_steps"]:
            return True
        if self.layer_idx is not None and self.layer_idx in cfg["dense_layers"]:
            return True
        return False

    def _dense_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return output[0] if isinstance(output, tuple) else output

    def _run_sol_attn_thd(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        from sol_attn import sol_attn

        cfg = _get_sol_attn_runtime_config()
        q = query.unsqueeze(0).contiguous()
        k = key.unsqueeze(0).contiguous()
        v = value.unsqueeze(0).contiguous()
        if q.dtype != torch.bfloat16:
            raise TypeError(f"Sol-Attn requires bfloat16 activations, got {q.dtype}")
        out = sol_attn(
            q,
            k,
            v,
            tau=cfg["tau"],
            thresh_type=cfg["thresh_type"],
            kv_splits=_resolve_kv_splits(q, cfg["kv_splits"]),
            sink_start=cfg["sink_start"],
            sink_tokens=cfg["sink_tokens"],
        )
        return out.squeeze(0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        del attn_metadata
        if self._should_use_dense():
            q = query.transpose(1, 2).reshape(
                query.shape[0] * query.shape[1], query.shape[2], query.shape[3]
            )
            k = key.transpose(1, 2).reshape(
                key.shape[0] * key.shape[1], key.shape[2], key.shape[3]
            )
            v = value.transpose(1, 2).reshape(
                value.shape[0] * value.shape[1], value.shape[2], value.shape[3]
            )
            cu_seqlens = torch.arange(
                0,
                (query.shape[0] + 1) * query.shape[1],
                query.shape[1],
                device=query.device,
                dtype=torch.int32,
            )
            out = self._dense_varlen(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                max_seqlen=query.shape[1],
            )
            return out.reshape(query.shape[0], query.shape[1], query.shape[2], -1)
        q = query.reshape(query.shape[0] * query.shape[1], query.shape[2], -1)
        k = key.reshape(key.shape[0] * key.shape[1], key.shape[2], -1)
        v = value.reshape(value.shape[0] * value.shape[1], value.shape[2], -1)
        out = self._run_sol_attn_thd(q, k, v)
        return out.reshape(query.shape[0], query.shape[1], query.shape[2], -1)

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        del cu_seqlens_host
        if self._should_use_dense():
            return self._dense_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        return self._run_sol_attn_thd(query, key, value)
