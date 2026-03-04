# SPDX-License-Identifier: Apache-2.0
"""Piecewise CUDA graph runner for diffusion DiT models.

This runner mirrors LLM piecewise CUDA graph behavior:
1. enable piecewise graph splitting via torch.compile backend hooks
2. bucket requests to capture sizes
3. apply padding to bucket shape
4. capture and replay CUDA graph with stable tensor addresses
"""

from __future__ import annotations

import bisect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Hashable

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compile import install_torch_compiled
from sglang.srt.compilation.piecewise_context_manager import (
    enable_piecewise_cuda_graph,
    enable_piecewise_cuda_graph_compile,
    set_pcg_capture_stream,
)

logger = init_logger(__name__)

_RUNTIME_VALUE = object()
_GLOBAL_GRAPH_POOL = None


def _set_torch_compile_config() -> None:
    import torch._dynamo.config

    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024


@contextmanager
def _graph_capture_stream():
    if not current_platform.is_cuda_alike():
        yield None
        return

    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    if current != stream:
        stream.wait_stream(current)
    with torch.cuda.stream(stream):
        yield stream


def _is_tensor(obj: Any) -> bool:
    return isinstance(obj, torch.Tensor)


def _join_path(parent: str, key: str) -> str:
    if not parent:
        return key
    return f"{parent}.{key}"


def _supports_scalar(obj: Any) -> bool:
    return isinstance(obj, (int, float, str, bool))


def _pad_dim_for_path(path: str, tensor: torch.Tensor, raw_seq_len: int) -> int | None:
    # Mandatory for all sequence DiTs.
    if path.endswith("hidden_states") and tensor.ndim >= 2:
        if tensor.shape[1] == raw_seq_len:
            return 1

    # Qwen-Image rotary cache for image tokens.
    if path == "freqs_cis.0" and tensor.ndim >= 1:
        if tensor.shape[0] == raw_seq_len:
            return 0

    return None


def _signature(
    obj: Any,
    *,
    raw_seq_len: int,
    static_seq_len: int,
    path: str = "",
) -> Hashable:
    if _is_tensor(obj):
        shape = list(obj.shape)
        pad_dim = _pad_dim_for_path(path, obj, raw_seq_len)
        if pad_dim is not None:
            shape[pad_dim] = static_seq_len
        return (
            "tensor",
            tuple(shape),
            str(obj.dtype),
            str(obj.device),
        )

    if isinstance(obj, list):
        return (
            "list",
            tuple(
                _signature(
                    x,
                    raw_seq_len=raw_seq_len,
                    static_seq_len=static_seq_len,
                    path=_join_path(path, str(i)),
                )
                for i, x in enumerate(obj)
            ),
        )
    if isinstance(obj, tuple):
        return (
            "tuple",
            tuple(
                _signature(
                    x,
                    raw_seq_len=raw_seq_len,
                    static_seq_len=static_seq_len,
                    path=_join_path(path, str(i)),
                )
                for i, x in enumerate(obj)
            ),
        )
    if isinstance(obj, dict):
        items = tuple(
            sorted(
                (
                    k,
                    _signature(
                        v,
                        raw_seq_len=raw_seq_len,
                        static_seq_len=static_seq_len,
                        path=_join_path(path, str(k)),
                    ),
                )
                for k, v in obj.items()
            )
        )
        return ("dict", items)
    if obj is None:
        return ("none",)
    if _supports_scalar(obj):
        return ("scalar", obj)
    return ("repr", repr(obj))


@dataclass
class _TensorSlot:
    tensor: torch.Tensor
    pad_dim: int | None = None


def _build_slots(
    obj: Any,
    *,
    raw_seq_len: int,
    static_seq_len: int,
    path: str = "",
) -> Any:
    if _is_tensor(obj):
        pad_dim = _pad_dim_for_path(path, obj, raw_seq_len)
        if pad_dim is None:
            return _TensorSlot(tensor=torch.empty_like(obj), pad_dim=None)
        new_shape = list(obj.shape)
        new_shape[pad_dim] = static_seq_len
        return _TensorSlot(
            tensor=torch.empty(
                tuple(new_shape), dtype=obj.dtype, device=obj.device
            ),
            pad_dim=pad_dim,
        )

    if isinstance(obj, list):
        return [
            _build_slots(
                x,
                raw_seq_len=raw_seq_len,
                static_seq_len=static_seq_len,
                path=_join_path(path, str(i)),
            )
            for i, x in enumerate(obj)
        ]
    if isinstance(obj, tuple):
        return tuple(
            _build_slots(
                x,
                raw_seq_len=raw_seq_len,
                static_seq_len=static_seq_len,
                path=_join_path(path, str(i)),
            )
            for i, x in enumerate(obj)
        )
    if isinstance(obj, dict):
        return {
            k: _build_slots(
                v,
                raw_seq_len=raw_seq_len,
                static_seq_len=static_seq_len,
                path=_join_path(path, str(k)),
            )
            for k, v in obj.items()
        }
    return _RUNTIME_VALUE


def _materialize_call_kwargs(slots: Any, values: Any) -> Any:
    if isinstance(slots, _TensorSlot):
        src = values
        dst = slots.tensor

        if slots.pad_dim is None:
            if dst.shape != src.shape:
                raise ValueError(
                    f"Tensor shape changed for CUDA graph replay: {dst.shape} vs {src.shape}"
                )
            dst.copy_(src)
            return dst

        if src.shape[slots.pad_dim] > dst.shape[slots.pad_dim]:
            raise ValueError(
                "Input sequence length exceeds padded slot shape: "
                f"{src.shape[slots.pad_dim]} > {dst.shape[slots.pad_dim]}"
            )

        dst.zero_()
        indices = [slice(None)] * dst.ndim
        indices[slots.pad_dim] = slice(0, src.shape[slots.pad_dim])
        dst[tuple(indices)].copy_(src)
        return dst

    if slots is _RUNTIME_VALUE:
        return values

    if isinstance(slots, list):
        return [
            _materialize_call_kwargs(s, v) for s, v in zip(slots, values, strict=True)
        ]
    if isinstance(slots, tuple):
        return tuple(
            _materialize_call_kwargs(s, v) for s, v in zip(slots, values, strict=True)
        )
    if isinstance(slots, dict):
        return {k: _materialize_call_kwargs(slots[k], values[k]) for k in slots.keys()}

    raise TypeError(f"Unsupported slot type: {type(slots)}")


def _slice_output_to_raw_seq(output: Any, raw_seq_len: int, static_seq_len: int) -> Any:
    if raw_seq_len == static_seq_len:
        return output

    if isinstance(output, torch.Tensor):
        if output.ndim >= 2 and output.shape[1] == static_seq_len:
            return output[:, :raw_seq_len, ...]
        return output
    if isinstance(output, list):
        return [_slice_output_to_raw_seq(x, raw_seq_len, static_seq_len) for x in output]
    if isinstance(output, tuple):
        return tuple(_slice_output_to_raw_seq(x, raw_seq_len, static_seq_len) for x in output)
    if isinstance(output, dict):
        return {
            k: _slice_output_to_raw_seq(v, raw_seq_len, static_seq_len)
            for k, v in output.items()
        }
    return output


def _get_graph_pool(device: torch.device):
    global _GLOBAL_GRAPH_POOL
    if _GLOBAL_GRAPH_POOL is None:
        _GLOBAL_GRAPH_POOL = torch.get_device_module(device).graph_pool_handle()
    return _GLOBAL_GRAPH_POOL


@dataclass
class _GraphEntry:
    signature: Hashable
    static_seq_len: int
    slots: dict[str, Any]
    captured: bool = False


class DiffusionPiecewiseCudaGraphRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        capture_sizes: list[int],
        *,
        compiler: str = "eager",
        enable_debug: bool = False,
    ) -> None:
        self.model = model
        self.capture_sizes = sorted(set(int(x) for x in capture_sizes))
        self.capture_sizes_set = set(self.capture_sizes)
        self.compiler = compiler
        self.enable_debug = enable_debug

        self._entries: dict[Hashable, _GraphEntry] = {}
        self._installed = False
        self._compiled = False
        self._eager_warmup_done = False

        device = next(model.parameters()).device
        self.device = device
        self.graph_pool = _get_graph_pool(device)

        self.compile_config = CompilationConfig(
            self.capture_sizes,
            compiler=self.compiler,
            enable_debug_mode=self.enable_debug,
        )
        _set_torch_compile_config()

    def _install_compiled(self) -> None:
        if self._installed:
            return
        install_torch_compiled(
            self.model,
            dynamic_arg_dims={"hidden_states": [1]},
            compile_config=self.compile_config,
            graph_pool=self.graph_pool,
            fullgraph=True,
        )
        self._installed = True

    def can_run(
        self, hidden_states: torch.Tensor, seq_len_override: int | None = None
    ) -> bool:
        if not current_platform.is_cuda_alike():
            return False
        if not torch.cuda.is_available():
            return False
        if hidden_states.device.type != "cuda":
            return False
        if hidden_states.ndim < 2:
            return False
        if seq_len_override is None:
            return False
        if hidden_states.shape[1] != int(seq_len_override):
            return False
        if not self.capture_sizes:
            return False
        if seq_len_override > self.capture_sizes[-1]:
            return False
        return True

    def _select_static_seq_len(self, seq_len: int) -> int | None:
        idx = bisect.bisect_left(self.capture_sizes, seq_len)
        if idx >= len(self.capture_sizes):
            return None
        return self.capture_sizes[idx]

    def _ensure_compiled(self, call_kwargs: dict[str, Any]) -> None:
        if self._compiled:
            return
        # Warm up lazy custom kernels (e.g. JIT kernels that touch filesystem)
        # outside torch.compile to avoid Dynamo tracing unsupported Python/C APIs.
        if not self._eager_warmup_done:
            with torch.no_grad():
                self.model(**call_kwargs)
            self._eager_warmup_done = True
        with enable_piecewise_cuda_graph():
            with enable_piecewise_cuda_graph_compile():
                _ = self.model(**call_kwargs)
        self._compiled = True

    def _capture(self, call_kwargs: dict[str, Any]) -> None:
        with enable_piecewise_cuda_graph():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

            with _graph_capture_stream() as stream:
                if stream is not None:
                    with set_pcg_capture_stream(stream):
                        self.model(**call_kwargs)
                        self.model(**call_kwargs)
                else:
                    self.model(**call_kwargs)
                    self.model(**call_kwargs)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

    def run(self, *, seq_len_override: int | None = None, **kwargs) -> Any | None:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None or not self.can_run(
            hidden_states, seq_len_override=seq_len_override
        ):
            return None

        raw_seq_len = int(seq_len_override)
        static_seq_len = self._select_static_seq_len(raw_seq_len)
        if static_seq_len is None:
            return None

        self._install_compiled()

        sig = (
            static_seq_len,
            _signature(
                kwargs,
                raw_seq_len=raw_seq_len,
                static_seq_len=static_seq_len,
                path="",
            ),
        )
        entry = self._entries.get(sig)
        if entry is None:
            entry = _GraphEntry(
                signature=sig,
                static_seq_len=static_seq_len,
                slots=_build_slots(
                    kwargs,
                    raw_seq_len=raw_seq_len,
                    static_seq_len=static_seq_len,
                    path="",
                ),
            )
            self._entries[sig] = entry
            logger.info(
                "Diffusion PCG init for %s (raw_seq=%d, static_seq=%d)",
                self.model.__class__.__name__,
                raw_seq_len,
                static_seq_len,
            )

        call_kwargs = _materialize_call_kwargs(entry.slots, kwargs)

        if not self._compiled:
            self._ensure_compiled(call_kwargs)

        if not entry.captured:
            self._capture(call_kwargs)
            entry.captured = True

        with enable_piecewise_cuda_graph():
            output = self.model(**call_kwargs)

        return _slice_output_to_raw_seq(output, raw_seq_len, static_seq_len)


def resolve_capture_sizes(
    *,
    seq_len: int,
    explicit_sizes: list[int] | None,
    max_tokens: int,
) -> list[int]:
    if explicit_sizes:
        sizes = sorted(set(int(x) for x in explicit_sizes if int(x) > 0))
        if seq_len <= max_tokens and seq_len not in sizes:
            sizes.append(int(seq_len))
        return sorted(set(sizes))

    capture_sizes = (
        list(range(4, 33, 4))
        + list(range(48, 257, 16))
        + list(range(288, 513, 32))
        + list(range(576, 1024 + 1, 64))
        + list(range(1280, 4096 + 1, 256))
        + list(range(4608, int(max_tokens) + 1, 512))
    )
    capture_sizes = [s for s in capture_sizes if s <= int(max_tokens)]
    if seq_len <= max_tokens and seq_len not in capture_sizes:
        capture_sizes.append(int(seq_len))
    return sorted(set(capture_sizes))
