# SPDX-License-Identifier: Apache-2.0
"""Native MLA adapter for the K3 whole-model decode plugin.

The plugin owns Q/KV projection and the value/gate kernel. SGLang owns local
attention metadata, cache restore barriers, DCP communication and LSE semantics.
No private Lamport DCP payload or cache policy is introduced here.
"""

from __future__ import annotations

from typing import Any

import torch

# Admission can distinguish this helper from the older TRT-only API-v1 module.
NATIVE_MLA_DECODE_BACKENDS = frozenset(("trtllm_mla", "cutedsl_mla"))


def _cute_dcp_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    backend: Any,
    layer: Any,
    metadata: Any,
    forward_batch: Any,
    parallel: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse the packed-query half of CuteDslMLABackend.forward_decode.

    Calling the TRT base's ``_forward_decode_dcp`` on the Cute subclass drops
    its global causal bound and CP placement. Calling ``forward_decode`` would
    instead quantize and insert the plugin's already-packed Q/KV a second time.
    Keep this invocation aligned with the native Cute decode, including its
    zero-local-KV neutralization, without changing either kernel implementation.
    """
    from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows

    batch = query.shape[0]
    if forward_batch.batch_size != batch:
        raise ValueError("K3 Cute MLA query batch differs from native decode metadata")
    if metadata.seq_lens_k is not None and metadata.global_seq_lens_k is not None:
        local_seq_lens = metadata.seq_lens_k[:batch]
        global_seq_lens = metadata.global_seq_lens_k[:batch]
    else:
        global_lengths = getattr(forward_batch, "seq_lens", None)
        if global_lengths is None:
            raise ValueError("K3 Cute DCP MLA requires native global causal lengths")
        global_seq_lens = global_lengths[:batch]
        local_seq_lens = backend._get_dcp_local_seq_lens(global_seq_lens)
    raw_out, lse = backend._run_decode_kernel(
        query=query,
        kv_cache=kv_cache,
        block_tables=metadata.block_kv_indices,
        seq_lens=local_seq_lens,
        max_seq_len=metadata.max_seq_len_k,
        layer=layer,
        causal_seqs=global_seq_lens,
        cp_world=parallel.dcp_size,
        cp_rank=parallel.dcp_rank,
        return_lse=True,
    )
    partial = raw_out.view(-1, layer.tp_q_head_num, layer.v_head_dim)
    lse = lse.view(-1, layer.tp_q_head_num)
    fixup_zero_kv_rows(
        partial,
        lse,
        local_seq_lens,
        backend.q_indptr_decode[: batch + 1],
        1,
    )
    return partial.flatten(1), lse


def native_mla_decode(
    query: torch.Tensor,
    output: torch.Tensor,
    *,
    backend: Any,
    cache: torch.Tensor,
    attention: Any,
    metadata: Any,
    forward_batch: Any,
) -> None:
    """Consume packed FP8 local queries after the plugin's owner-only KV insert.

    Ordinary decode only. Reuse the same native backend and DCP merge selected
    by the standard model. This function never writes KV or chooses KDA slots.
    Allocation by the native collectives is captured just as on the stock path.
    """
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if not forward_batch.forward_mode.is_decode():
        raise ValueError("K3 native MLA adapter serves ordinary decode only")
    batch, heads = query.shape[:2]
    if query.shape != (batch, 12, 576) or query.dtype != torch.uint8:
        raise ValueError("K3 native MLA adapter needs packed [B, 12, 576] queries")
    if output.shape != (batch, heads, 512) or output.dtype != torch.bfloat16:
        raise ValueError("K3 native MLA adapter needs BF16 [B, 12, 512] output")
    if parallel.dcp_enabled:
        from sglang.srt.layers.logits_processor import get_in_autotune_dummy_run

        if get_in_autotune_dummy_run():
            # Match the native model: synthetic FlashInfer tuning discards
            # attention outputs and must not introduce DCP collectives.
            output.zero_()
            return
    kv_cache = cache.view(torch.float8_e4m3fn).unsqueeze(1)
    if not parallel.dcp_enabled:
        result = backend._run_decode_kernel(
            query=query.view(torch.float8_e4m3fn).unsqueeze(1),
            kv_cache=kv_cache,
            block_tables=metadata.block_kv_indices,
            seq_lens=metadata.seq_lens_k,
            max_seq_len=metadata.max_seq_len_k,
            layer=attention.attn_mqa,
        )
        output.copy_(result.view_as(output))
        return

    from sglang.srt.layers.dcp import (
        all_gather_q_for_mla_decode,
        cp_lse_ag_out_rs_mla,
        dcp_a2a_lse_reduce,
    )

    # Gather bytes using the authoritative head ordering. FP8 values were
    # quantized by the plugin; no second cast/quantization changes them here.
    # These are the plugin's original TP-sharded projection weights, even when
    # the ordinary model has extra replicated-Q buffers. Always gather the 12
    # local heads here; the config's replicated-Q flag is not this tensor's
    # layout and must not cause the gather to be skipped.
    nope, rope = all_gather_q_for_mla_decode(query[..., :512], query[..., 512:])
    gathered = torch.cat((nope, rope), dim=-1).view(torch.float8_e4m3fn)
    is_cute_dsl = getattr(backend, "backend", None) == "cute-dsl"
    if is_cute_dsl:
        partial, lse = _cute_dcp_decode(
            gathered.unsqueeze(1),
            kv_cache,
            backend=backend,
            layer=attention.attn_mqa_for_dcp_decode,
            metadata=metadata,
            forward_batch=forward_batch,
            parallel=parallel,
        )
    else:
        partial, lse = backend._forward_decode_dcp(
            gathered.unsqueeze(1),
            kv_cache,
            metadata,
            attention.attn_mqa_for_dcp_decode,
            forward_batch,
        )
    partial = partial.view(batch, heads * parallel.attn_dcp_size, 512)
    # Match native forward_mla.is_mla_dcp_lse_base_on_e: Cute DSL returns natural
    # log LSE; TRT returns log2. Both paths neutralize empty local shards before
    # the same native merge. Never reinterpret Cute LSE as a base-two value.
    if parallel.dcp_comm_backend in ("a2a", "fi_a2a"):
        merged = dcp_a2a_lse_reduce(
            partial.contiguous(),
            lse.contiguous(),
            parallel.dcp_group,
            is_lse_base_on_e=is_cute_dsl,
            comm_backend=parallel.dcp_comm_backend,
        )
    else:
        merged = cp_lse_ag_out_rs_mla(
            partial, lse, parallel.dcp_group, is_lse_base_on_e=is_cute_dsl
        ).transpose(0, 1)
    output.copy_(merged.view_as(output))
