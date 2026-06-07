"""b12x CuTe-DSL GLM_NSA sparse-MLA kernel as a DSA decode/prefill impl.

Selected via `--attention-backend dsa --dsa-decode-backend cutedsl
--dsa-prefill-backend cutedsl`. Reuses the DSA KV pool (packed layout
nope_fp8(512) | scales(16) | rope_bf16(128) = 656B), indexer top-k, FP8 KV
quant, and bf16 RoPE-in-prepare q. The kernel matches the flashinfer trtllm-gen
sparse MLA ground truth (test/registered/kernels/test_cutedsl_sparse_mla.py).

The decode workspace is pre-allocated (use_cuda_graph=True) so no device memory
is allocated during CUDA graph capture/replay.
"""

from __future__ import annotations

import torch

_B12X = None


def _b12x():
    global _B12X
    if _B12X is None:
        import b12x.cute.fp4 as fp4

        fp4.get_sm_version = lambda device=None: 120
        from b12x.attention.workspace import B12XAttentionWorkspace
        from b12x.integration.mla import (
            sparse_mla_decode_forward,
            sparse_mla_extend_forward,
        )

        _B12X = (
            B12XAttentionWorkspace,
            sparse_mla_decode_forward,
            sparse_mla_extend_forward,
        )
    return _B12X


def _get_workspace(backend, mode):
    cache = getattr(backend, "_b12x_ws", None)
    if cache is None:
        cache = backend._b12x_ws = {}
    ws = cache.get(mode)
    if ws is not None:
        return ws
    ws_cls = _b12x()[0]
    if mode == "decode":
        max_q = max_batch = backend.b12x_decode_max_bs
    else:
        max_q = backend.b12x_extend_max_q
        max_batch = backend.b12x_extend_max_batch
    # Allocate as normal (non-inference) tensors so b12x can mutate its split-config
    # control tensors in-place during CUDA graph capture.
    with torch.inference_mode(False), torch.no_grad():
        ws = ws_cls.for_fixed_capacity(
            mode=mode,
            device=backend.device,
            dtype=torch.bfloat16,
            kv_dtype=backend.kv_cache_dtype,
            num_q_heads=backend.num_q_heads,
            head_dim=backend.kv_lora_rank + backend.qk_rope_head_dim,
            v_head_dim=backend.kv_lora_rank,
            topk=backend.dsa_index_topk,
            max_total_q=max_q,
            max_batch=max_batch,
            page_size=backend.real_page_size,
            use_cuda_graph=(mode == "decode"),
        )
    cache[mode] = ws
    return ws


def prealloc_decode_workspace(backend):
    _get_workspace(backend, "decode")


def forward_b12x_sparse_mla(
    backend,
    *,
    is_prefill,
    q_all,
    kv_cache,
    page_table_1,
    cache_seqlens,
    nsa_seqlens,
    sm_scale,
    v_head_dim,
):
    _, decode_fwd, extend_fwd = _b12x()
    kv_cache = kv_cache.view(-1, 1, backend.kv_cache_dim)
    ws = _get_workspace(backend, "extend" if is_prefill else "decode")
    if is_prefill:
        return extend_fwd(
            q_all=q_all,
            kv_cache=kv_cache,
            selected_token_offsets=page_table_1,
            cache_seqlens_int32=cache_seqlens,
            nsa_cache_seqlens_int32=nsa_seqlens,
            workspace=ws,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
        )
    # b12x picks num_splits from the batch size and only re-fills its split-count
    # control tensor when the cached value changes. Under CUDA graphs the per-bs
    # graphs share one workspace, so force the fill to be recorded in every
    # captured graph; otherwise a graph that skipped it reads a stale split count
    # at replay and the split-merge sums the wrong number of partials.
    ws.num_chunks_value = -1
    return decode_fwd(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_seqlens,
        workspace=ws,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
        backend="sm120_unified",
    )
