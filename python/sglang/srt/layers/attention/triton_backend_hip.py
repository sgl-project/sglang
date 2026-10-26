"""ROCm glue of triton_backend: TritonAttnBackend's state for the gfx950 M3.1 DSpark
draft, bound there under _is_hip as _hip. The draft's grouped-head verify kernel reads a
sliding window, its target verify reads only the window KV indices, and its KV store
skips the defensive k/v clone when the pool stores them as given."""

from __future__ import annotations

import functools

import torch

from sglang.srt.configs.model_config import is_dspark_draft
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.runtime_context import get_model
from sglang.srt.utils import is_gfx95_supported


def _windows_every_layer(hf_config) -> bool:
    from sglang.srt.speculative.dflash_utils import get_dflash_layer_types

    layer_types = get_dflash_layer_types(hf_config)
    return bool(layer_types) and all(t == "sliding_attention" for t in layer_types)


def is_minimax_m31_dspark_draft(hf_config) -> bool:
    """The DSpark draft of an M3.1 target; the MXFP8 M3 base target shares its arch."""
    # is_dspark_draft first, so other models never call get_model() here
    return (
        is_dspark_draft(hf_config)
        and hf_config.architectures[0] == "DSparkMiniMaxDraftModel"
        and get_model().quantization == "minimax_m3_nvfp4"
    )


def init_m31_draft(backend, hf_config) -> None:
    """The ROCm state of TritonAttnBackend.__init__; reads backend.sliding_window_size."""
    from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd

    m31_draft = is_minimax_m31_dspark_draft(hf_config)
    # The sliding-window path is validated only on the M3.1 draft (window 4095).
    backend.verify_shared_kv_fwd = torch.compiler.disable(
        functools.partial(verify_shared_kv_fwd, allow_sliding_window=m31_draft)
    )
    backend.is_gfx950_m31_draft = is_gfx95_supported() and m31_draft
    # The M3.1 draft windows every layer and reads only window_kv_indices; its
    # target-verify steps skip packing the full-context index stream.
    backend.verify_reads_window_kv_only = (
        backend.is_gfx950_m31_draft
        and backend.sliding_window_size is not None
        and backend.sliding_window_size > 0
        and _windows_every_layer(hf_config)
    )


def stores_kv_as_given(backend, k: torch.Tensor) -> bool:
    """Whether set_kv_buffer stores k and v as given: a plain MHA pool of their
    dtype ignores the layer's scales."""
    pool = backend.token_to_kv_pool
    return (
        type(pool) is MHATokenToKVPool
        and pool.dtype == k.dtype
        and not pool.is_quantized_kv_cache
    )
