# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionImpl,
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
        if head_size != _SOL_ATTN_HEAD_DIM:
            raise ValueError(
                f"Sol-Attn requires head_size={_SOL_ATTN_HEAD_DIM}, got {head_size}"
            )
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.prefix = prefix
        self.layer_idx = self._parse_layer_idx(prefix)
        # Dense guards must use the platform-selected FlashAttention version.
        # Calling the low-level wrapper directly defaults to FA3, which is not
        # available on Blackwell; the CUDA resolver selects FA4 for this impl.
        self.dense_impl = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.dense",
            **extra_impl_args,
        )

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
        return self.dense_impl.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

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
        if self._should_use_dense():
            return self.dense_impl.forward(query, key, value, attn_metadata)
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
        if self._should_use_dense():
            return self._dense_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(value) for value in cu_seqlens.tolist())
        )
        segments = [
            (start, stop)
            for start, stop in zip(bounds[:-1], bounds[1:])
            if stop > start
        ]
        if not segments:
            return torch.zeros_like(query)

        # Sparse kernels operate on one document. Running once on the packed
        # buffer would let documents attend across boundaries; in MiniMax-H3 it
        # would also mix the aligned padding tail into every real query.
        output = torch.zeros_like(query)
        for start, stop in segments:
            # H3 exposes its one live document as (0, used, total), with
            # max_seqlen=used. The trailing document is alignment padding and
            # must stay zero rather than becoming an independent sparse call.
            trailing_padding = (
                len(bounds) == 3
                and start == bounds[1]
                and stop == query.shape[0]
                and bounds[1] == max_seqlen
            )
            if not trailing_padding:
                output[start:stop] = self._run_sol_attn_thd(
                    query[start:stop], key[start:stop], value[start:stop]
                )
        return output
