# SPDX-License-Identifier: Apache-2.0
"""cuDNN SDPA forward pinned to the COMPOSITE implementation.

PyTorch's cuDNN SDPA uses the AUTO implementation, which resolves to the
UNIFIED engine on sm90 (cuDNN 9.24). Building that plan is not reproducible:
some builds give a kernel whose bf16 output differs in the last bits, and the
plan is cached per thread, so Qwen3-VL image embeddings differed between
processes and TP ranks. COMPOSITE gives the same result as the usual UNIFIED
build and builds deterministically. Graphs are cached per shape/stride/scale.
"""

from __future__ import annotations

import threading

import torch

try:  # nvidia-cudnn-frontend python bindings
    import cudnn as _cudnn
except Exception:  # pragma: no cover - optional dependency
    _cudnn = None

_tls = threading.local()
_DTYPES = {}
if _cudnn is not None:
    _DTYPES = {
        torch.bfloat16: _cudnn.data_type.BFLOAT16,
        torch.float16: _cudnn.data_type.HALF,
    }


def available() -> bool:
    return _cudnn is not None and hasattr(_cudnn, "attention_implementation")


def _aligned(t: torch.Tensor) -> bool:
    return (
        t.stride(-1) == 1
        and t.data_ptr() % 16 == 0
        and all(s % 8 == 0 for s, n in zip(t.stride()[:-1], t.shape[:-1]) if n > 1)
    )


def _handle(device: torch.device):
    handles = getattr(_tls, "handles", None)
    if handles is None:
        handles = _tls.handles = {}
    h = handles.get(device.index)
    if h is None:
        with torch.cuda.device(device):
            h = handles[device.index] = _cudnn.create_handle()
    return h


def _graph(q, k, v, out, scale: float, handle):
    key = (
        q.device.index,
        q.dtype,
        tuple(q.shape),
        q.stride(),
        tuple(k.shape),
        k.stride(),
        tuple(v.shape),
        v.stride(),
        out.stride(),
        float(scale),
    )
    cache = getattr(_tls, "graphs", None)
    if cache is None:
        cache = _tls.graphs = {}
    entry = cache.get(key)
    if entry is None:
        g = _cudnn.pygraph(
            io_data_type=_DTYPES[q.dtype],
            intermediate_data_type=_cudnn.data_type.FLOAT,
            compute_data_type=_cudnn.data_type.FLOAT,
            handle=handle,
        )
        qt, kt, vt = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
        o, _ = g.sdpa(
            name="sdpa",
            q=qt,
            k=kt,
            v=vt,
            generate_stats=False,
            attn_scale=scale,
            use_causal_mask=False,
            implementation=_cudnn.attention_implementation.COMPOSITE,
        )
        o.set_output(True).set_dim(list(out.shape)).set_stride(list(out.stride()))
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([_cudnn.heur_mode.A])
        g.check_support()
        g.build_plans()
        entry = cache[key] = (g, (qt, kt, vt, o), max(1, g.get_workspace_size()))
    return entry


def cudnn_composite_sdpa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float
) -> torch.Tensor | None:
    """Non-causal, unmasked SDPA on [B, H, S, D] tensors; None if unsupported."""
    if not (
        available()
        and query.is_cuda
        and query.dtype in _DTYPES
        and query.dtype == key.dtype == value.dtype
        and query.dim() == key.dim() == value.dim() == 4
        and query.shape[1] == key.shape[1] == value.shape[1]
        and query.shape[-1] == key.shape[-1] == value.shape[-1]
        and query.shape[-1] % 8 == 0
        and query.shape[-1] <= 256
        and _aligned(query)
        and _aligned(key)
        and _aligned(value)
    ):
        return None
    b, h, s, d = query.shape
    # BSHD storage, the layout PyTorch's cuDNN path writes
    out = torch.empty(b, s, h, d, device=query.device, dtype=query.dtype).transpose(
        1, 2
    )
    handle = _handle(query.device)
    try:
        g, (qt, kt, vt, out_t), ws_bytes = _graph(query, key, value, out, scale, handle)
    except Exception:
        return None
    stream = torch.cuda.current_stream(query.device)
    _cudnn.set_stream(handle=handle, stream=stream.cuda_stream)
    workspace = torch.empty(ws_bytes, device=query.device, dtype=torch.uint8)
    g.execute({qt: query, kt: key, vt: value, out_t: out}, workspace, handle=handle)
    return out
