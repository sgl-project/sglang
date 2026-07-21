"""Two-stream LoRA overlap (O1 + O7 + O8 + O9) — installed as a monkey-patch.

Activates when env ``SGLANG_LORA_TWO_STREAM=1``. Triggered exactly once via
:func:`install_two_stream_overrides` (called at end of ``sglang/srt/lora/layers.py``).

When enabled, these call sites are redirected to side-stream-overlapped versions
defined entirely in this package:

  * ``QKVParallelLinearWithLoRA.forward``  → :mod:`.attention.qkv_proj_lora_forward`
  * ``RowParallelLinearWithLoRA.forward``  → :mod:`.attention.row_parallel_lora_forward`
  * ``MergedColumnParallelLinearWithLoRA.forward`` → :mod:`.merged_column.merged_column_lora_forward`
  * ``fused_experts_none_to_experimental_sgl_trtllm_fp8_lora`` →
    :mod:`.moe_overlap.fused_experts_none_to_experimental_sgl_trtllm_fp8_lora_two_stream`

When disabled (env unset), ``install_two_stream_overrides`` is a no-op and all
the original functions / methods in ``sglang/srt/lora/layers.py`` and
``sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`` run unchanged.

Per-batch gating still happens inside the patched callables — they fall back
to the saved-original implementation for non-decode batches (token count above
``SGLANG_TWO_STREAM_MAX_TOKENS`` default 256), so prefill stays on the serial
path even with the patch installed.
"""

from typing import Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs


def is_two_stream_active(x: torch.Tensor) -> bool:
    """Per-batch gate (two-stream now always-on). True iff batch is decode-shaped (<= SGLANG_TWO_STREAM_MAX_TOKENS)."""
    return x.shape[0] <= lora_envs.SGLANG_TWO_STREAM_MAX_TOKENS.get()


def supports_two_stream_dense_lora(lora_a: torch.Tensor, lora_b: torch.Tensor) -> bool:
    """Keep the temporary shrink kernel within its safe combined-rank tile."""
    return lora_a.shape[-2] <= 128 and lora_b.shape[-1] <= 64


# One side stream per consumer stream: routed (main/capture) and InklingMoE's sink (alt
# stream) run concurrently; sharing one side stream is a premature-reuse WAR -> IMA.
_LORA_SIDE_STREAMS: dict[torch.cuda.Stream, torch.cuda.Stream] = {}


def get_lora_side_stream() -> torch.cuda.Stream:
    # Lazy creation is capture-safe: graph warmup runs on the capture stream
    # (graph_capture() sets it), so every key exists before any capture region.
    consumer = torch.cuda.current_stream()
    if consumer not in _LORA_SIDE_STREAMS:
        _LORA_SIDE_STREAMS[consumer] = torch.cuda.Stream()
    return _LORA_SIDE_STREAMS[consumer]


def init_lora_two_stream_resources(device: Optional[torch.device] = None) -> None:
    """Eagerly create the side stream before cuda-graph capture begins.

    ``torch.cuda.Stream()`` is a driver call that must not run inside a
    cuda-graph capture region. Since :func:`get_lora_side_stream` is otherwise
    lazy, the first eligible decode forward would create it — which can fall
    inside capture if warmup didn't happen to exercise a two-stream batch.
    Calling this from a pre-capture hook pins creation to init/warmup on the
    correct device.
    """
    if device is not None:
        with torch.cuda.device(device):
            get_lora_side_stream()
    else:
        get_lora_side_stream()


def lora_overlap_alloc_stream() -> Optional[torch.cuda.Stream]:
    """Stream to allocate side-stream LoRA-shrink OUTPUT buffers on, or None for default behavior.

    A buffer allocated *inside* ``with torch.cuda.stream(side)`` is tagged to the side stream, so the
    caching allocator may free/reuse it on the side stream's schedule — before the MAIN stream (the real
    consumer, via the LoRA-B expand) is done. Under cuda-graph replay that's a premature-reuse WAR ->
    qwen3.5 mamba decode garbage. With ``SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC`` this returns the MAIN
    stream so the op allocates the output on the consumer stream (like the MoE O1 ``gate_up_delta``),
    making a single shared side stream graph-safe. Call on the MAIN stream BEFORE forking to the side.
    """
    if lora_envs.SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC.get():
        return torch.cuda.current_stream()
    return None


# References to the original implementations, captured at install time so the
# patched callables can defer to them for non-decode batches.
_ORIGINAL_QKV_FORWARD: Optional[Callable] = None
_ORIGINAL_ROW_FORWARD: Optional[Callable] = None
_ORIGINAL_MERGED_FORWARD: Optional[Callable] = None
_ORIGINAL_COLUMN_FORWARD: Optional[Callable] = None
_ORIGINAL_REPLICATED_FORWARD: Optional[Callable] = None
_ORIGINAL_MOE_LORA_FUNC: Optional[Callable] = None
_ORIGINAL_FP4_MOE_LORA_FUNC: Optional[Callable] = None
_ORIGINAL_BF16_MOE_LORA_FUNC: Optional[Callable] = None
_INSTALLED: bool = False


def get_original_qkv_forward() -> Callable:
    return _ORIGINAL_QKV_FORWARD


def get_original_row_forward() -> Callable:
    return _ORIGINAL_ROW_FORWARD


def get_original_merged_column_forward() -> Callable:
    return _ORIGINAL_MERGED_FORWARD


def get_original_column_forward() -> Callable:
    return _ORIGINAL_COLUMN_FORWARD


def get_original_replicated_forward() -> Callable:
    return _ORIGINAL_REPLICATED_FORWARD


def get_original_moe_lora_func() -> Callable:
    return _ORIGINAL_MOE_LORA_FUNC


def get_original_fp4_moe_lora_func() -> Callable:
    return _ORIGINAL_FP4_MOE_LORA_FUNC


def get_original_bf16_moe_lora_func() -> Callable:
    return _ORIGINAL_BF16_MOE_LORA_FUNC


def install_two_stream_overrides() -> None:
    """Install the side-stream overlapped overrides if ``SGLANG_LORA_TWO_STREAM=1``.

    Idempotent: subsequent calls are a no-op. Patches:

      1. ``QKVParallelLinearWithLoRA.forward`` (O7 — qkv LoRA shrink overlap)
      2. ``RowParallelLinearWithLoRA.forward`` (O8 — o_proj LoRA shrink overlap)
      3. ``MergedColumnParallelLinearWithLoRA.forward`` (O9 — merged-column LoRA
         shrink overlap: dense gate_up + mamba in_proj_qkvz)
      4. ``lora_dispatch.fused_experts_none_to_experimental_sgl_trtllm_fp8_lora``
         (O1 — MoE gate_up LoRA overlap), plus its fp4 (O1-fp4) and bf16
         (O1-bf16) siblings

    The saved originals are exposed via :func:`get_original_qkv_forward`,
    :func:`get_original_row_forward`, :func:`get_original_moe_lora_func` so the
    new versions can fall back when their per-batch gate says single-stream.
    """
    global _INSTALLED, _ORIGINAL_QKV_FORWARD, _ORIGINAL_ROW_FORWARD, _ORIGINAL_MERGED_FORWARD, _ORIGINAL_COLUMN_FORWARD, _ORIGINAL_REPLICATED_FORWARD, _ORIGINAL_MOE_LORA_FUNC, _ORIGINAL_FP4_MOE_LORA_FUNC, _ORIGINAL_BF16_MOE_LORA_FUNC

    if _INSTALLED:
        return

    from sglang.srt.lora.layers import (
        ColumnParallelLinearWithLoRA,
        MergedColumnParallelLinearWithLoRA,
        QKVParallelLinearWithLoRA,
        ReplicatedLinearWithLoRA,
        RowParallelLinearWithLoRA,
    )
    from sglang.srt.lora.trtllm_lora_temp.attention import (
        column_parallel_lora_forward,
        qkv_proj_lora_forward,
        replicated_lora_forward,
        row_parallel_lora_forward,
    )
    from sglang.srt.lora.trtllm_lora_temp.merged_column import (
        merged_column_lora_forward,
    )

    # Capture all originals before patching: QKV / MergedColumn subclass
    # ColumnParallel, so the plain-Column O10 patch must not clobber the
    # subclasses' own (3-/2-slice) forwards captured here as their fallbacks.
    _ORIGINAL_QKV_FORWARD = QKVParallelLinearWithLoRA.forward
    _ORIGINAL_ROW_FORWARD = RowParallelLinearWithLoRA.forward
    _ORIGINAL_MERGED_FORWARD = MergedColumnParallelLinearWithLoRA.forward
    _ORIGINAL_COLUMN_FORWARD = ColumnParallelLinearWithLoRA.forward
    _ORIGINAL_REPLICATED_FORWARD = ReplicatedLinearWithLoRA.forward
    QKVParallelLinearWithLoRA.forward = qkv_proj_lora_forward
    RowParallelLinearWithLoRA.forward = row_parallel_lora_forward
    MergedColumnParallelLinearWithLoRA.forward = merged_column_lora_forward
    # O10 (MLA q_b_proj / kv_b_proj) + O11 (MLA fused_qkv_a_proj_with_mqa).
    ColumnParallelLinearWithLoRA.forward = column_parallel_lora_forward
    ReplicatedLinearWithLoRA.forward = replicated_lora_forward

    import sglang.srt.lora.trtllm_lora_temp.lora_dispatch as ft
    from sglang.srt.lora.trtllm_lora_temp.moe_overlap import (
        fused_experts_none_to_experimental_sgl_trtllm_bf16_lora_two_stream,
        fused_experts_none_to_experimental_sgl_trtllm_fp4_lora_two_stream,
        fused_experts_none_to_experimental_sgl_trtllm_fp8_lora_two_stream,
    )

    # O1 (FP8 Qwen) + O1-fp4 (NVFP4 Kimi) + O1-bf16 (unquantized Qwen): MoE gate_up
    # LoRA overlap. Each patched fn falls back to its saved single-stream original
    # for non-decode batches.
    _ORIGINAL_MOE_LORA_FUNC = ft.fused_experts_none_to_experimental_sgl_trtllm_fp8_lora
    _ORIGINAL_FP4_MOE_LORA_FUNC = (
        ft.fused_experts_none_to_experimental_sgl_trtllm_fp4_lora
    )
    _ORIGINAL_BF16_MOE_LORA_FUNC = (
        ft.fused_experts_none_to_experimental_sgl_trtllm_bf16_lora
    )
    ft.fused_experts_none_to_experimental_sgl_trtllm_fp8_lora = (
        fused_experts_none_to_experimental_sgl_trtllm_fp8_lora_two_stream
    )
    ft.fused_experts_none_to_experimental_sgl_trtllm_fp4_lora = (
        fused_experts_none_to_experimental_sgl_trtllm_fp4_lora_two_stream
    )
    ft.fused_experts_none_to_experimental_sgl_trtllm_bf16_lora = (
        fused_experts_none_to_experimental_sgl_trtllm_bf16_lora_two_stream
    )

    _INSTALLED = True


__all__ = [
    "is_two_stream_active",
    "supports_two_stream_dense_lora",
    "get_lora_side_stream",
    "init_lora_two_stream_resources",
    "get_original_qkv_forward",
    "get_original_row_forward",
    "get_original_merged_column_forward",
    "get_original_column_forward",
    "get_original_replicated_forward",
    "get_original_moe_lora_func",
    "get_original_fp4_moe_lora_func",
    "get_original_bf16_moe_lora_func",
    "install_two_stream_overrides",
]
