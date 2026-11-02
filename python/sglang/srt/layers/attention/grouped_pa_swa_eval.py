"""Opt-in MiMo target SWA using AITER PA and one shared plan per forward."""

import torch
import triton
import triton.language as tl


@triton.jit
def _csr_to_dense(indptr, indices, dense, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    begin = tl.load(indptr + row)
    end = tl.load(indptr + row + 1)
    values = tl.load(
        indices + begin + columns,
        mask=(columns < WIDTH) & (begin + columns < end),
        other=0,
    )
    tl.store(dense + row * WIDTH + columns, values, mask=columns < WIDTH)


def forward_grouped_pa_swa(backend, q, layer, batch, k, v, out, sinks):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    bs = int(batch.batch_size)
    qlen = int(backend.forward_metadata.max_q_len)
    if (
        layer.tp_q_head_num,
        layer.tp_k_head_num,
        layer.qk_head_dim,
        layer.v_head_dim,
    ) != (16, 2, 192, 128):
        raise ValueError("Grouped PA SWA requires Flash TP4 Q16/KV2/QK192/V128")
    if (
        backend.page_size != 64
        or int(layer.sliding_window_size) != 128
        or qlen not in (1, 4, 8)
    ):
        raise ValueError(
            "Grouped PA SWA requires page64, window128 and query length1/4/8"
        )
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise ValueError("Grouped PA SWA requires BF16 Q/K/V")
    if sinks is None or sinks.dtype != torch.float32:
        raise ValueError("Grouped PA SWA requires the model FP32 sink tensor")
    if bs == 0:
        return

    metadata = backend.forward_metadata
    if any(
        getattr(metadata, name, None) is None
        for name in (
            "flydsl_swa_context_lens",
            "flydsl_swa_kv_indptr",
            "flydsl_swa_kv_indices",
        )
    ):
        raise RuntimeError(
            "Grouped PA SWA metadata must be prepared before layer forward"
        )
    width = int(backend._flydsl_dflash_swa_max_pages)
    cache = getattr(backend, "_grouped_pa_swa_workspace", None)
    if cache is None:
        cache = backend._grouped_pa_swa_workspace = {}
    key = (bs, qlen, width)
    if key not in cache:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Grouped PA SWA workspace must be warmed before graph capture"
            )
        if layer.layer_id != backend._grouped_pa_swa_first_layer:
            raise RuntimeError("Grouped PA SWA must start at the first SWA layer")
        lengths = torch.zeros(bs, device=q.device, dtype=torch.int32)
        dense = torch.zeros((bs, width), device=q.device, dtype=torch.int32)
        plan = plan_pa_decode(
            lengths, 2, max_partitions=1, sliding_window=128, query_length=qlen
        )
        shape = (2, plan.capacity, qlen * 8)
        pmax = torch.empty(shape, device=q.device, dtype=torch.float32)
        psum = torch.empty_like(pmax)
        pout = torch.empty(*shape, 128, device=q.device, dtype=torch.bfloat16)
        cache[key] = (lengths, dense, plan, pmax, psum, pout)
    lengths, dense, plan, pmax, psum, pout = cache[key]
    if layer.layer_id == backend._grouped_pa_swa_first_layer:
        # Captured once per target forward; each replay refreshes the shared
        # metadata before any SWA layer consumes its own Q/K/V and sinks.
        lengths.copy_(metadata.flydsl_swa_context_lens)
        _csr_to_dense[(bs,)](
            metadata.flydsl_swa_kv_indptr,
            metadata.flydsl_swa_kv_indices,
            dense,
            WIDTH=width,
            BLOCK=triton.next_power_of_2(width),
        )
        plan_pa_decode(lengths, 2, plan=plan, sliding_window=128, query_length=qlen)
    pa_decode(
        out.view(bs * qlen, 16, 128),
        q.view(bs * qlen, 16, 192),
        k,
        v,
        lengths,
        dense,
        layer.scaling,
        qlen,
        sliding_window=128,
        sinks=sinks,
        work_plan=plan,
        max_context_length=width * 64,
        exp_sums=psum,
        max_logits=pmax,
        temporary_output=pout,
    )
