# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
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
_DENSE_BACKENDS = {"fa", "sage_attn"}


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
    dense_backend = (
        str(cfg.get("dense_backend", "fa")).strip().lower().replace("-", "_")
    )
    if dense_backend in {"sage", "sageattention"}:
        dense_backend = "sage_attn"
    if dense_backend not in _DENSE_BACKENDS:
        raise ValueError(
            f"Unsupported sol_attn dense_backend={dense_backend!r}; "
            f"expected one of {sorted(_DENSE_BACKENDS)}"
        )
    sink_start = cfg.get("sink_start", 0)
    return {
        "tau": float(cfg.get("tau", 1.0)),
        "thresh_type": str(cfg.get("thresh_type", "diag")),
        "kv_splits": cfg.get("kv_splits", "auto"),
        "sink_tokens": int(cfg.get("sink_tokens", 0)),
        "sink_start": None if sink_start is None else int(sink_start),
        "dense_steps": int(cfg.get("dense_steps", 10)),
        "dense_layers": _parse_layer_ranges(cfg.get("dense_layers", "0,1")),
        "dense_backend": dense_backend,
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
        self._sol_params: frozenset[str] | None = None

    @staticmethod
    def _parse_layer_idx(prefix: str) -> int | None:
        match = re.search(r"blocks\.(\d+)", prefix)
        return int(match.group(1)) if match else None

    def _should_use_dense(self) -> bool:
        cfg = _get_sol_attn_runtime_config()
        try:
            from sglang.multimodal_gen.runtime.managers.forward_context import (
                get_forward_context,
            )

            step = int(get_forward_context().current_timestep)
        except AssertionError:
            step = 0
        return step < cfg["dense_steps"] or (
            self.layer_idx is not None and self.layer_idx in cfg["dense_layers"]
        )

    def _dense_fa(
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

    def _dense_sage(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SageAttention dense path.

        - batched NHD ``[B, T, H, D]`` when ``cu_seqlens is None``
        - packed ``[total, H, D]`` when ``cu_seqlens`` is provided
        """
        from sageattention import sageattn

        if cu_seqlens is None:
            return sageattn(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                tensor_layout="NHD",
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
            )

        bounds = [int(x) for x in cu_seqlens.tolist()]
        output = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            output[start:stop] = sageattn(
                query[start:stop].unsqueeze(0).contiguous(),
                key[start:stop].unsqueeze(0).contiguous(),
                value[start:stop].unsqueeze(0).contiguous(),
                tensor_layout="NHD",
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
            )[0]
        return output

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

        if self._sol_params is None:
            self._sol_params = frozenset(inspect.signature(sol_attn).parameters)

        kwargs = {
            "tau": cfg["tau"],
            "thresh_type": cfg["thresh_type"],
            "kv_splits": _resolve_kv_splits(q, cfg["kv_splits"]),
            "sink_start": cfg["sink_start"],
            "sink_tokens": cfg["sink_tokens"],
        }
        # Wan2GP Ada port: INT8-QK Triton; official NVlabs API has no int8_qk.
        if "int8_qk" in self._sol_params and tuple(
            torch.cuda.get_device_capability(q.device)
        ) >= (8, 9):
            kwargs["int8_qk"] = True
        kwargs = {k: v for k, v in kwargs.items() if k in self._sol_params}
        return sol_attn(q, k, v, **kwargs).squeeze(0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        del attn_metadata
        # ``query`` is NHD: [B, T, H, D]
        if self._should_use_dense():
            if _get_sol_attn_runtime_config()["dense_backend"] == "sage_attn":
                return self._dense_sage(query, key, value)
            # NHD [B, T, H, D] → packed THD [B*T, H, D] (plain reshape; do not
            # transpose — that would scramble token order for flash_attn_varlen).
            q = query.reshape(
                query.shape[0] * query.shape[1], query.shape[2], query.shape[3]
            )
            k = key.reshape(key.shape[0] * key.shape[1], key.shape[2], key.shape[3])
            v = value.reshape(
                value.shape[0] * value.shape[1], value.shape[2], value.shape[3]
            )
            cu_seqlens = torch.arange(
                0,
                (query.shape[0] + 1) * query.shape[1],
                query.shape[1],
                device=query.device,
                dtype=torch.int32,
            )
            out = self._dense_fa(
                q, k, v, cu_seqlens=cu_seqlens, max_seqlen=query.shape[1]
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
            if _get_sol_attn_runtime_config()["dense_backend"] == "sage_attn":
                return self._dense_sage(query, key, value, cu_seqlens=cu_seqlens)
            return self._dense_fa(
                query, key, value, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
        return self._run_sol_attn_thd(query, key, value)
