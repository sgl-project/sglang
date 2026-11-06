"""CP attention, token redistribution, cache continuity and TP MoE reduction.

Run with python FILE; NCCL workers are spawned by the tests."""

import json
import socket
import tempfile
import traceback
import unittest
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_varlen_func

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.merge_state import merge_state
from sglang.srt.layers.cp.utils import (
    ShardedContextParallelMetadata as ContextParallelMetadata,
)
from sglang.srt.layers.cp.utils import (
    TokenExchangeStream,
    contiguous_chunk_indices,
    contiguous_token_indices,
    copy_token_rows,
    cp_gather_full_sequence_states,
)
from sglang.srt.layers.cp.utils import make_cp_attention_layout as _make_ring_layout
from sglang.srt.layers.cp.utils import (
    make_cp_attention_segments,
    make_token_exchange,
    make_token_gather_scatter,
    make_token_reduce_scatter,
)
from sglang.srt.layers.layer_boundary.prepare import (
    prepare_sharded_cp_attention,
    prepare_sharded_cp_ffn,
)
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.layers.layer_boundary.row_transport import (
    finish_sharded_cp_ffn,
    launch_cp_residual_exchange,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, MHATokenToKVPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, mock_parallel_groups

register_cuda_ci(est_time=200, stage="stage-b", runner_config="2-gpu-large")


_test_layout = SimpleNamespace(query_layout="zigzag")


def make_cp_attention_layout(**kwargs):
    return _make_ring_layout(query_layout=_test_layout.query_layout, **kwargs)


def _prepare_attention_layout(frame, hidden, residual, batch, layout, norm=None):
    """Exercise the production residual boundary while isolating token movement."""
    batch.prefill_cp_metadata = frame
    batch.layer_cp_metadata = frame
    stream = ResidualStream(residual)
    if residual is not None:
        hidden = stream.record(hidden, PLAIN_ADD)
    if norm is None:
        norm = lambda value, residual=None: (
            value if residual is None else (value, residual)
        )
    hidden = prepare_sharded_cp_attention(hidden, stream, batch, norm, layout=layout)
    return hidden, stream.residual


class ShardedCache:
    def __init__(self, *, maximum, rank, world, kv_heads, dim, dtype):
        self.rank, self.world = rank, world
        self.k_scale = (
            torch.tensor(0.75, device="cuda") if dtype == torch.float8_e4m3fn else None
        )
        self.v_scale = (
            torch.tensor(1.25, device="cuda") if dtype == torch.float8_e4m3fn else None
        )
        self.page_size = 16
        pages = [(length + 16 * world - 1) // (16 * world) for length in maximum]
        generator = torch.Generator().manual_seed(2718)
        shuffled = (torch.randperm(sum(pages), generator=generator) + 1).tolist()
        self.slots = []
        start = 0
        for length, count in zip(maximum, pages):
            table = shuffled[start : start + count]
            self.slots.append(
                tuple(
                    table[pos // (16 * world)] * 16 * world + pos % (16 * world)
                    for pos in range(length)
                )
            )
            start += count
        self.pool = MHATokenToKVPool(
            size=(sum(pages) + 1) * 16,
            page_size=16,
            dtype=dtype,
            head_num=kv_heads,
            head_dim=dim,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        self.pool.get_key_buffer(0).fill_(float("nan"))
        self.pool.get_value_buffer(0).fill_(float("nan"))

    def write(self, *, local_kv, layout):
        new_slots = [
            slot
            for slots, s, q in zip(self.slots, layout.seq_lens, layout.query_lens)
            for slot in slots[s - q : s]
        ]
        owned = [new_slots[i] for i in layout.new_kv_indices[self.rank]]
        assert all(slot % self.world == self.rank for slot in owned)
        loc = torch.tensor([slot // self.world for slot in owned], device="cuda")
        if owned:
            self.pool.set_kv_buffer(
                layer=None,
                loc_info=KVWriteLoc(loc, physical=True),
                cache_k=local_kv[0],
                cache_v=local_kv[1],
                k_scale=self.k_scale,
                v_scale=self.v_scale,
                layer_id_override=0,
            )

    def read(self, *, layout):
        loc = torch.tensor(
            [
                slot // self.world
                for slots, s in zip(self.slots, layout.seq_lens)
                for slot in slots[self.rank : s : self.world]
            ],
            device="cuda",
            dtype=torch.long,
        )
        return torch.stack(
            (
                self.pool.get_key_buffer(0).index_select(0, loc),
                self.pool.get_value_buffer(0).index_select(0, loc),
            )
        )


def _reference(*, queries, keys, values, seq_lens, query_lens, cache=None):
    outputs, lses, fa_outputs, fa_lses = [], [], [], []
    for q, k, v, size, count in zip(queries, keys, values, seq_lens, query_lens):
        if not count:
            continue
        q, k, v = q[size - count : size], k[:size], v[:size]
        k_descale = v_descale = None
        if cache is not None and cache.pool.dtype == torch.float8_e4m3fn:
            q = q.to(cache.pool.dtype)
            k = (k / cache.k_scale).to(cache.pool.dtype)
            v = (v / cache.v_scale).to(cache.pool.dtype)
            k_descale = cache.k_scale.expand(1, k.shape[1])
            v_descale = cache.v_scale.expand(1, v.shape[1])
        cu_q = torch.tensor([0, count], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, size], device="cuda", dtype=torch.int32)
        fa_q, fa_k, fa_v = q, k, v
        fa_k_descale, fa_v_descale = k_descale, v_descale
        if (
            q.dtype == torch.float8_e4m3fn
            and torch.cuda.get_device_capability()[0] == 10
        ):
            fa_q = q.to(torch.bfloat16)
            fa_k = (k.float() * cache.k_scale).to(torch.bfloat16)
            fa_v = (v.float() * cache.v_scale).to(torch.bfloat16)
            fa_k_descale = fa_v_descale = None
        fa_out, fa_lse, *_ = flash_attn_varlen_func(
            q=fa_q,
            k=fa_k,
            v=fa_v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=count,
            max_seqlen_k=size,
            causal=True,
            k_descale=fa_k_descale,
            v_descale=fa_v_descale,
            return_softmax_lse=True,
            ver=3 if torch.cuda.get_device_capability()[0] == 9 else 4,
        )
        fa_outputs.append(fa_out)
        fa_lses.append(fa_lse.T.contiguous())
        repeat = q.shape[1] // k.shape[1]
        kt = k.float().repeat_interleave(repeat, 1).transpose(0, 1).transpose(1, 2)
        vt = v.float().repeat_interleave(repeat, 1).transpose(0, 1)
        if k_descale is not None:
            kt = kt * cache.k_scale
            vt = vt * cache.v_scale
        key_positions = torch.arange(size, device="cuda")
        for start in range(0, count, 128):
            stop = min(count, start + 128)
            logits = (q[start:stop].float().transpose(0, 1) @ kt) * q.shape[-1] ** -0.5
            positions = torch.arange(
                size - count + start, size - count + stop, device="cuda"
            )
            logits.masked_fill_(key_positions[None, :] > positions[:, None], -torch.inf)
            lses.append(torch.logsumexp(logits, dim=-1).T.contiguous())
            outputs.append((torch.softmax(logits, dim=-1) @ vt).transpose(0, 1))
    if not outputs:
        shape = (0, *queries[0].shape[1:])
        empty = torch.empty(shape, device="cuda", dtype=torch.float32)
        return empty, empty[:, :, 0], empty.to(queries[0].dtype), empty[:, :, 0]
    return (
        torch.cat(outputs),
        torch.cat(lses),
        torch.cat(fa_outputs),
        torch.cat(fa_lses),
    )


def _check(*, actual, expected, actual_lse, expected_lse, label):
    if not actual.numel():
        assert actual.shape == expected.shape
        return {
            "label": label,
            "relative_rms": 0.0,
            "max_over_row_rms": 0.0,
            "lse_max_abs": 0.0,
        }
    delta = actual.float() - expected.float()
    rms = expected.float().square().mean().sqrt().clamp_min(1e-8)
    # Early causal rows have much larger outputs than late rows. A global RMS
    # denominator makes the max-error bound depend on sequence length.
    row_rms = expected.float().square().mean(-1).sqrt().clamp_min(1e-8)
    metrics = torch.stack(
        (
            delta.square().mean().sqrt() / rms,
            (delta.abs().amax(-1) / row_rms).max(),
            (actual_lse - expected_lse).abs().max(),
        )
    ).tolist()
    assert all(torch.isfinite(torch.tensor(metrics))), (label, metrics)
    assert metrics[0] < 0.005 and metrics[1] < 0.04 and metrics[2] < 2e-4, (
        label,
        metrics,
    )
    return {
        "label": label,
        "relative_rms": metrics[0],
        "max_over_row_rms": metrics[1],
        "lse_max_abs": metrics[2],
    }


def _check_fp8_output(actual, tp_output, reference):
    """Bound partitioned FP8 error by TP FP8 error on identical quantized inputs."""
    if not actual.numel():
        assert actual.shape == tp_output.shape == reference.shape
        return {
            "cp_relative_rms": 0.0,
            "tp_relative_rms": 0.0,
            "cp_tp_relative_rms": 0.0,
        }
    reference = reference.float()
    denominator = reference.square().mean().clamp_min(1e-16)
    cp_error = (
        ((actual.float() - reference).square().mean() / denominator).sqrt().item()
    )
    tp_error = (
        ((tp_output.float() - reference).square().mean() / denominator).sqrt().item()
    )
    assert cp_error <= 1.25 * tp_error + 0.003 and cp_error < 0.04, (cp_error, tp_error)
    assert torch.isfinite(actual).all()
    difference = (
        ((actual.float() - tp_output.float()).square().mean() / denominator)
        .sqrt()
        .item()
    )
    assert difference < 0.05
    return {
        "cp_relative_rms": cp_error,
        "tp_relative_rms": tp_error,
        "cp_tp_relative_rms": difference,
    }


def _no_gather():
    return patch.multiple(
        dist,
        all_gather=unittest.mock.Mock(side_effect=AssertionError("KV all-gather")),
        all_gather_into_tensor=unittest.mock.Mock(
            side_effect=AssertionError("KV all-gather")
        ),
    )


def _run_step(*, ring, cache, queries, keys, values, seq_lens, query_lens, label):
    rank, group = ring.rank, ring.exchange.group
    layout = make_cp_attention_layout(
        seq_lens=seq_lens, query_lens=query_lens, world_size=ring.world
    )
    contiguous = contiguous_token_indices(tokens=sum(query_lens), world_size=ring.world)
    ep_to_zigzag = make_token_exchange(
        source=contiguous,
        target=layout.query_indices,
        group=group,
        device=queries[0].device,
    )
    zigzag_to_kv = make_token_exchange(
        source=layout.query_indices,
        target=layout.new_kv_indices,
        group=group,
        device=queries[0].device,
    )
    restore = make_token_exchange(
        source=layout.query_indices,
        target=contiguous,
        group=group,
        device=queries[0].device,
    )
    q_heads, kv_heads, dim = queries[0].shape[1], keys[0].shape[1], queries[0].shape[2]
    packed = torch.cat(
        [
            torch.cat((q[s - n : s], k[s - n : s], v[s - n : s]), dim=1)
            for q, k, v, s, n in zip(queries, keys, values, seq_lens, query_lens)
        ]
    )
    local_input = packed[list(contiguous[rank])].contiguous()
    segments = make_cp_test_segments(layout=layout, rank=rank, device=queries[0].device)
    with _no_gather():
        local = ep_to_zigzag.apply(local_input)
        q = local[:, :q_heads].contiguous()
        new_kv = (
            zigzag_to_kv.apply(local[:, q_heads:])
            .view(-1, 2, kv_heads, dim)
            .transpose(0, 1)
            .contiguous()
        )
        cache.write(local_kv=new_kv, layout=layout)
        shard = cache.read(layout=layout)
        assert shard.shape[1] == sum(layout.kv_lens[rank])
        shape = (segments[0].cu_q.numel() - 1, kv_heads)
        output, lse = ring.forward(
            q=q.to(shard.dtype),
            local_kv=shard,
            layout=layout,
            segments=segments,
            k_descale=(
                cache.k_scale.expand(shape) if cache.k_scale is not None else None
            ),
            v_descale=(
                cache.v_scale.expand(shape) if cache.v_scale is not None else None
            ),
        )
        assert output.dtype == queries[0].dtype and lse.dtype == torch.float32
        assert ring.exchange.buffers.dtype == cache.pool.dtype
        restored = restore.apply(output)
    expected, expected_lse, fa_output, fa_lse = _reference(
        queries=queries,
        keys=keys,
        values=values,
        seq_lens=seq_lens,
        query_lens=query_lens,
        cache=cache,
    )
    _check_backend(cache, packed, seq_lens, query_lens, fa_output, expected)
    index = list(layout.query_indices[rank])
    if cache.k_scale is not None:
        result = _check_fp8_output(output, fa_output[index], expected[index])
        _check_fp8_output(
            restored,
            fa_output[list(contiguous[rank])],
            expected[list(contiguous[rank])],
        )
        lse_error = (lse - fa_lse[index]).abs().max().item() if lse.numel() else 0.0
        assert lse_error < 2e-4
        fa_result = {"relative_rms": result["cp_tp_relative_rms"]}
        result.update(label=label, lse_max_abs=lse_error)
    else:
        result = _check(
            actual=output,
            expected=expected[index],
            actual_lse=lse,
            expected_lse=expected_lse[index],
            label=label + "/fp32",
        )
        fa_result = _check(
            actual=output,
            expected=fa_output[index],
            actual_lse=lse,
            expected_lse=fa_lse[index],
            label=label + "/fa3",
        )
        torch.testing.assert_close(
            restored, fa_output[list(contiguous[rank])], atol=0.008, rtol=0.04
        )
    result.update(
        fa3_relative_rms=fa_result["relative_rms"],
        rank=rank,
        cache_tokens=shard.shape[1],
        total_kv_tokens=sum(seq_lens),
    )
    if (
        sum(query_lens)
        and sum(query_lens) % (2 * ring.world) == 0
        and len(seq_lens) == 1
    ):
        prefix = seq_lens[0] - query_lens[0]
        costs = [
            sum(prefix + i + 1 for i in indices) for indices in layout.query_indices
        ]
        if _test_layout.query_layout == "zigzag":
            assert max(costs) == min(costs), costs
        else:
            assert max(costs) - min(costs) == query_lens[0] // ring.world * (
                ring.world - 1
            ), costs
    return result


def _check_backend(cache, packed, seq_lens, query_lens, expected, reference):
    """Exercise the serving backend and actual sharded pool, including ragged rows."""
    table = torch.zeros(
        len(cache.slots), max(map(len, cache.slots)), device="cuda", dtype=torch.int64
    )
    for i, slots in enumerate(cache.slots):
        table[i, : len(slots)] = torch.tensor(slots, device="cuda")
    out_slots = [
        slot
        for slots, s, n in zip(cache.slots, seq_lens, query_lens)
        for slot in slots[s - n : s]
    ]
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        prefill_cp_metadata=None,
        extend_seq_lens_cpu=query_lens,
        extend_prefix_lens_cpu=tuple(s - n for s, n in zip(seq_lens, query_lens)),
        req_pool_indices=torch.arange(len(seq_lens), device="cuda"),
        out_cache_loc=torch.tensor(out_slots, device="cuda", dtype=torch.long),
        num_token_non_padded=torch.tensor(sum(query_lens), device="cuda"),
    )
    kv_heads, dim = cache.pool.get_key_buffer(0).shape[1:]
    q_heads = packed.shape[1] - 2 * kv_heads
    cache.pool.full_attention_cp_size = dist.get_world_size()
    backend = create_cp_test_backend(
        SimpleNamespace(
            device=str(packed.device),
            server_args=SimpleNamespace(
                prefill_cp_mode=(
                    "in-seq-split"
                    if _test_layout.query_layout == "zigzag"
                    else "round-robin-split"
                )
            ),
            dtype=packed.dtype,
            token_to_kv_pool=cache.pool,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            model_config=SimpleNamespace(
                get_num_attention_heads=lambda _: q_heads, context_len=table.shape[1]
            ),
            kv_index_translator=None,
            page_size=cache.pool.page_size * dist.get_world_size(),
            kv_cache_dtype=cache.pool.dtype,
            cp_kv_exchange=CPAttentionTestRunner(
                group=dist.group.WORLD,
                capacity=max(1, max(sum(seq_lens), sum(query_lens))),
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=dim,
                device=packed.device,
                dtype=cache.pool.dtype,
            ).exchange,
        )
    )
    # Follow the serving order: CPU plan and upload before backend binding.
    batch.reqs = [
        SimpleNamespace(kv=SimpleNamespace(req_pool_idx=i))
        for i in range(len(seq_lens))
    ]
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    hybrid = object.__new__(HybridLinearAttnBackend)
    hybrid.full_attn_backend = backend
    hybrid.linear_attn_backend = SimpleNamespace(
        device=backend.device, _cp_residual_exchange=None
    )
    hybrid.req_to_token_pool = backend.req_to_token_pool
    hybrid._cp_metadata_plan = None
    hybrid.prepare_cp_metadata(
        batch, lengths=query_lens, prefixes=batch.extend_prefix_lens_cpu
    )
    backend.init_forward_metadata(batch)
    frame = batch.prefill_cp_metadata
    layer = SimpleNamespace(
        tp_q_head_num=q_heads,
        tp_k_head_num=kv_heads,
        tp_v_head_num=kv_heads,
        qk_head_dim=dim,
        v_head_dim=dim,
        layer_id=0,
        k_scale=cache.k_scale,
        v_scale=cache.v_scale,
        scaling=dim**-0.5,
    )
    with _no_gather():
        # Emulate GDN -> EP combine -> full attention -> EP combine -> GDN.
        local, residual = _prepare_attention_layout(
            frame, packed, packed.clone(), batch, "contiguous"
        )
        local, residual = _prepare_attention_layout(
            frame, local, residual, batch, "attention"
        )
        torch.testing.assert_close(local, residual, atol=0, rtol=0)
        actual = backend.forward_extend(
            local[:, :q_heads],
            local[:, q_heads : q_heads + kv_heads],
            local[:, q_heads + kv_heads :],
            layer,
            batch,
        )
        actual, _ = _prepare_attention_layout(frame, actual, None, batch, "contiguous")
        # A shorter checkpoint must drop rows and redistribute to empty ranks.
        source = contiguous_chunk_indices((sum(query_lens),), cache.world)
        target = contiguous_chunk_indices((min(3, sum(query_lens)),), cache.world)
        probe = torch.tensor(
            source[cache.rank], device="cuda", dtype=torch.float32
        ).reshape(-1, 1)
        subset = make_token_exchange(
            source=source,
            target=target,
            group=frame.group,
            device=probe.device,
            allow_subset=True,
        )
        reduced = subset.apply(probe)
        torch.testing.assert_close(
            reduced.flatten(),
            torch.tensor(target[cache.rank], device="cuda", dtype=torch.float32),
        )
    gathered = cp_gather_full_sequence_states(
        actual, batch, torch.cuda.current_stream()
    ).reshape_as(expected)
    if cache.k_scale is not None:
        _check_fp8_output(gathered, expected, reference)
        assert backend.cp_exchange.buffers.dtype == torch.float8_e4m3fn
        assert gathered.dtype == torch.bfloat16
    else:
        torch.testing.assert_close(gathered, expected, atol=0.008, rtol=0.04)
    assert batch.num_token_non_padded.item() == sum(query_lens)


def _check_empty_merge():
    output = torch.zeros(3, 2, 64, device="cuda", dtype=torch.float32)
    lse = torch.full((3, 2), -torch.inf, device="cuda")
    partial = torch.full_like(output, float("nan"), dtype=torch.bfloat16)
    partial_lse = torch.full((2, 3), -torch.inf, device="cuda")
    partial_lse[:, 1] = torch.inf
    merge_state(
        prefix_output=output,
        prefix_lse=lse,
        suffix_output=partial,
        suffix_lse=partial_lse.T,
        output=output,
        output_lse=lse,
    )
    assert torch.equal(output, torch.zeros_like(output)) and torch.isneginf(lse).all()
    partial.fill_(2)
    partial_lse.fill_(1000)
    merge_state(
        prefix_output=output,
        prefix_lse=lse,
        suffix_output=partial,
        suffix_lse=partial_lse.T,
        output=output,
        output_lse=lse,
    )
    assert torch.equal(output, partial.float()) and torch.equal(
        lse, torch.full_like(lse, 1000)
    )


def _check_buffer_reuse(*, ring, cache, queries, seq_lens, query_lens):
    # Back-to-back reuse must preserve receive-buffer lifetime without a host fence.
    layout = make_cp_attention_layout(
        seq_lens=seq_lens, query_lens=query_lens, world_size=ring.world
    )
    segments = make_cp_test_segments(
        layout=layout, rank=ring.rank, device=queries[0].device
    )
    packed_q = torch.cat(
        [q[s - n : s] for q, s, n in zip(queries, seq_lens, query_lens)]
    )
    local_q = packed_q[list(layout.query_indices[ring.rank])].contiguous()
    shard = cache.read(layout=layout)
    local_q = local_q.to(shard.dtype)
    shape = (segments[0].cu_q.numel() - 1, shard.shape[2])
    descales = dict(
        k_descale=cache.k_scale.expand(shape) if cache.k_scale is not None else None,
        v_descale=cache.v_scale.expand(shape) if cache.v_scale is not None else None,
    )
    baseline, baseline_lse = ring.forward(
        q=local_q, local_kv=shard, layout=layout, segments=segments, **descales
    )
    with _no_gather():
        repeated = []
        for _ in range(8):
            torch.cuda._sleep(100000)
            repeated.append(
                ring.forward(
                    q=local_q,
                    local_kv=shard,
                    layout=layout,
                    segments=segments,
                    **descales,
                )
            )
        for actual, actual_lse in repeated:
            torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
            torch.testing.assert_close(actual_lse, baseline_lse, rtol=0, atol=0)
    serial = CPAttentionTestRunner(
        group=ring.exchange.group,
        capacity=ring.exchange.capacity,
        q_heads=local_q.shape[1],
        kv_heads=ring.exchange.kv_heads,
        head_dim=ring.exchange.head_dim,
        device=queries[0].device,
        overlap=False,
        transport=ring.exchange.transport,
        dtype=local_q.dtype,
    )
    actual, actual_lse = serial.forward(
        q=local_q, local_kv=shard, layout=layout, segments=segments, **descales
    )
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    torch.testing.assert_close(actual_lse, baseline_lse, rtol=0, atol=0)


def _run_cases(*, rank, group):
    reports = []
    world = dist.get_world_size(group)
    cases = (
        (
            "prefill_16k",
            (16392,),
            [
                ((16384,), (16384,)),
                ((16391,), (7,)),
                ((16392,), (1,)),
                ((16384,), (129,)),
            ],
            8,
            2,
            128,
            torch.bfloat16,
        ),
        ("head256", (513,), [((513,), (513,))], 24, 2, 256, torch.bfloat16),
        (
            "qwen397b_head256",
            (1025, 129, 0),
            [((1024, 128, 0), (1024, 128, 0)), ((1025, 129, 0), (1, 1, 0))],
            32,
            2,
            256,
            torch.bfloat16,
        ),
        (
            "qwen397b_fp8_cache",
            (513, 17, 0),
            [((257, 1, 0), (257, 1, 0)), ((513, 17, 0), (256, 16, 0))],
            32,
            2,
            256,
            torch.bfloat16,
        ),
        (
            "fp8_cache",
            (513, 17, 0),
            [
                ((257, 1, 0), (257, 1, 0)),
                ((513, 17, 0), (256, 16, 0)),
                ((513, 17, 0), (1, 0, 0)),
            ],
            32,
            2,
            256,
            torch.bfloat16,
        ),
        ("fp16", (259,), [((259,), (259,))], 12, 3, 128, torch.float16),
        ("mha", (127,), [((127,), (127,))], 4, 4, 64, torch.bfloat16),
        ("one_token_mqa", (1,), [((1,), (1,))], 8, 1, 128, torch.bfloat16),
        (
            "cache",
            (520, 264, 6, 0),
            [
                ((193, 64, 0, 0), (193, 64, 0, 0)),
                ((515, 259, 1, 0), (322, 195, 1, 0)),
                *[((n, n - 256, n - 514, 0), (1, 1, 1, 0)) for n in range(516, 521)],
                ((130, 70, 1, 0), (1, 3, 1, 0)),
                ((130, 70, 1, 0), (0, 0, 0, 0)),
            ],
            8,
            2,
            128,
            torch.bfloat16,
        ),
    )
    _check_empty_merge()
    for name, maximum, steps, q_heads, kv_heads, dim, dtype in cases:
        if torch.cuda.get_device_capability()[0] == 10 and (
            dim != 256 or q_heads != 16 * kv_heads
        ):
            continue
        torch.manual_seed(314159)
        queries = [
            torch.randn(n, q_heads, dim, device="cuda", dtype=dtype) for n in maximum
        ]
        keys = [
            torch.randn(n, kv_heads, dim, device="cuda", dtype=dtype) for n in maximum
        ]
        values = [torch.randn_like(k) for k in keys]
        cache = ShardedCache(
            maximum=maximum,
            rank=rank,
            world=world,
            kv_heads=kv_heads,
            dim=dim,
            dtype=torch.float8_e4m3fn if name.endswith("fp8_cache") else dtype,
        )
        capacity = sum((n + world - 1) // world for n in maximum)
        ring = CPAttentionTestRunner(
            group=group,
            capacity=capacity,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=dim,
            device=queries[0].device,
            dtype=cache.pool.dtype,
        )
        for step, (seq_lens, query_lens) in enumerate(steps):
            reports.append(
                _run_step(
                    ring=ring,
                    cache=cache,
                    queries=queries,
                    keys=keys,
                    values=values,
                    seq_lens=seq_lens,
                    query_lens=query_lens,
                    label=f"{name}/{step}",
                )
            )
        _check_buffer_reuse(
            ring=ring,
            cache=cache,
            queries=queries,
            seq_lens=steps[0][0],
            query_lens=steps[0][1],
        )
    return reports


def _check_rma_overlap(
    group, tokens=1024, q_heads=32, head_dim=256, dtype=torch.bfloat16
):
    """Changing payloads + rank skew expose stale signals and premature reuse."""
    import time

    torch.manual_seed(5678)
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    layout = make_cp_attention_layout(
        seq_lens=(tokens,), query_lens=(tokens,), world_size=world
    )
    segments = make_cp_test_segments(
        layout=layout, rank=rank, device=torch.device("cuda")
    )
    q = torch.randn(tokens, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)[
        list(layout.query_indices[rank])
    ].contiguous()
    shard = torch.randn(
        2, sum(layout.kv_lens[rank]), 2, head_dim, device="cuda", dtype=torch.bfloat16
    )
    shard = shard + rank * 0.25
    q = q.to(dtype)
    args = dict(
        group=group,
        capacity=layout.capacity,
        q_heads=q_heads,
        kv_heads=2,
        head_dim=head_dim,
        device=torch.device("cuda"),
        dtype=dtype,
    )
    ordinary = CPAttentionTestRunner(**args, transport="nccl")
    rma = CPAttentionTestRunner(**args, transport="symm_rma")
    values = [(shard * (0.5 + i / 8)).to(dtype) for i in range(12)]
    shard = shard.to(dtype)
    reference = [
        ordinary.forward(q=q, local_kv=x, layout=layout, segments=segments)
        for x in values
    ]
    actual = []
    with (
        patch.object(
            dist,
            "batch_isend_irecv",
            side_effect=AssertionError("RMA must not use two-sided P2P"),
        ),
        _no_gather(),
    ):
        for i, x in enumerate(values):
            if (i + rank) % world == 0:
                torch.cuda._sleep(500000)
            actual.append(
                rma.forward(q=q, local_kv=x, layout=layout, segments=segments)
            )
        for (out, lse), (expected, expected_lse) in zip(actual, reference):
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)
    # Time both transports with the same compute, warmed workspaces and inputs.
    timings = {}
    for name, ring in (("nccl", ordinary), ("symm_rma", rma)):
        torch.cuda.synchronize()
        begin = time.monotonic()
        for _ in range(20):
            ring.forward(q=q, local_kv=shard, layout=layout, segments=segments)
        torch.cuda.synchronize()
        timings[name] = (time.monotonic() - begin) * 1000 / 20
    print(
        json.dumps(
            dict(
                rma_overlap_pass=True,
                dtype=str(dtype),
                rank=rank,
                tokens=tokens,
                q_heads=q_heads,
                head_dim=head_dim,
                ms=timings,
            )
        ),
        flush=True,
    )


def _check_symm_workspace_isolation(group):
    """Shared communicator, concurrent independent signals, and all-reduce reuse."""
    from sglang.srt.distributed.device_communicators.torch_symm_mem import (
        TorchSymmMemCommunicator,
    )
    from sglang.srt.layers.cp.utils import RingKVExchange, RingTensorExchange

    rank, world = dist.get_rank(group), dist.get_world_size(group)
    device = torch.device("cuda", rank)
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        with torch.inference_mode():
            kv = RingKVExchange(
                group=group,
                capacity=7,
                kv_heads=1,
                head_dim=8,
                device=device,
                dtype=dtype,
                overlap=True,
                transport="symm_rma",
            )
            tail = RingTensorExchange(
                group=group,
                shape=(3, 8),
                device=device,
                dtype=torch.bfloat16,
                transport="symm_rma",
            )
        communicator = kv.symm_mem_comm
        assert communicator is tail.symm_mem_comm
        assert communicator is TorchSymmMemCommunicator.get(group, device)
        assert kv.rma.generations is not tail.rma.generations
        assert kv.rma.signals.data_ptr() != tail.rma.signals.data_ptr()
        assert not kv.buffers.is_inference() and not tail.buffers.is_inference()
        # Exercise uint32 wrap on KV while the tail has fresh zero generations.
        kv.rma.generations[:] = [0xFFFFFFFE, 0xFFFFFFFE]
        kv.rma.signals.fill_(-2)
        torch.cuda.synchronize()
        dist.barrier(group)
        for step in range(8):
            tokens = step % 4  # Includes zero payloads and short K/V planes.
            value = torch.full((2, tokens, 1, 8), rank + step, device=device).to(dtype)
            value[1].fill_(rank + step + 8)
            pending = tail.prepare(
                torch.full(
                    (3, 8), rank + step * world, device=device, dtype=torch.bfloat16
                )
            )
            pending.start()
            if (step + rank) % world == 0:
                torch.cuda._sleep(100000)
            for owner, received in kv.shards(value, active_tokens=tokens):
                expected = torch.full_like(received, owner + step)
                expected[1].fill_(owner + step + 8)
                torch.testing.assert_close(
                    received.float(), expected.float(), atol=0, rtol=0
                )
            gathered = pending.wait()
            for owner in range(world):
                torch.testing.assert_close(
                    gathered[owner],
                    torch.full_like(gathered[owner], owner + step * world),
                    atol=0,
                    rtol=0,
                )
        torch.cuda.synchronize()
        communicator.initialize_all_reduce()
        assert not communicator.disabled
        inp = torch.full((256,), rank + 1, device=device, dtype=torch.bfloat16)
        reduced = communicator.all_reduce(inp)
        torch.testing.assert_close(
            reduced, torch.full_like(inp, world * (world + 1) / 2), atol=0, rtol=0
        )
        # Enabling all-reduce must not replace another operation's payload/signals.
        pending = tail.prepare(torch.full_like(gathered[rank], rank + 32))
        pending.start()
        for owner, received in enumerate(pending.wait()):
            torch.testing.assert_close(
                received, torch.full_like(received, owner + 32), atol=0, rtol=0
            )
        # Multicast uses its own readiness/arrival protocol on the same communicator.
        with torch.inference_mode():
            broadcast = communicator.allocate_broadcast_workspace(4096)
        assert not broadcast.buffers.is_inference()
        assert broadcast.signals.data_ptr() != tail.rma.signals.data_ptr()
        broadcast.generations[0] = 0xFFFFFFFE
        broadcast.signals.fill_(-2)
        torch.cuda.synchronize()
        dist.barrier(group)
        for step in range(world * 2):
            owner = step % world
            payload_dtype = torch.float32 if step % 2 else torch.bfloat16
            payload = torch.full(
                (256,), rank + step * world, device=device, dtype=payload_dtype
            )
            communicator.broadcast(broadcast, payload, owner)
            torch.testing.assert_close(
                payload, torch.full_like(payload, owner + step * world), atol=0, rtol=0
            )
        broadcast.close()
        tail.rma.close()
        torch.cuda.synchronize()
        communicator.release_workspace(kv.rma)
        with unittest.TestCase().assertRaisesRegex(ValueError, "closed"):
            communicator.exchange_ring(kv.rma, send_slot=0, recv_slot=1, signal_slot=1)
    print(
        json.dumps({"symm_workspace_isolation": True, "rank": rank, "world": world}),
        flush=True,
    )


def _worker(rank, result_dir, world_size):
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    get_context().set_server_args(
        ServerArgs(
            model_path="dummy",
            tp_size=world_size,
            attn_cp_size=world_size,
            linear_attn_prefill_backend="flashqla",
            disaggregation_mode="prefill",
        )
    )
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        init_method=Path(result_dir, "rendezvous").as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    try:
        dist.barrier()
        _check_symm_workspace_isolation(dist.group.WORLD)
        with mock_parallel_groups(
            attn_cp_size=world_size,
            attn_cp_rank=rank,
            attn_tp_size=1,
            moe_tp_size=1,
            attn_cp_group=SimpleNamespace(device_group=dist.group.WORLD),
        ):
            reports = []
            for partition in ("zigzag", "interleaved"):
                with (
                    patch.object(_test_layout, "query_layout", partition),
                    get_context().parallel.override(
                        cp_strategy="zigzag" if partition == "zigzag" else "interleave"
                    ),
                ):
                    results = _run_cases(rank=rank, group=dist.group.WORLD)
                    for row in results:
                        row["query_layout"] = partition
                    reports.extend(results)
                    _check_rma_overlap(dist.group.WORLD)
                    _check_rma_overlap(dist.group.WORLD, dtype=torch.float8_e4m3fn)
        Path(result_dir, f"rank{rank}.json").write_text(json.dumps(reports, indent=2))
    except Exception:
        traceback.print_exc()
        # Let spawn terminate the peer; collective teardown can hang when that
        # peer has already entered the next test's P2P operation.
        raise
    dist.destroy_process_group()


@unittest.skipUnless(
    torch.cuda.device_count() >= 2
    and torch.cuda.get_device_capability() in ((9, 0), (10, 0), (10, 3)),
    "requires two Hopper or datacenter Blackwell GPUs",
)
class TestCPAttentionTestRunner(CustomTestCase):
    def test_sharded_cache_causality_and_ring_lifetimes(self):
        """Guard causal offsets, empty shards, cache continuation and buffer reuse."""
        self._check_sharded_cache_causality_and_ring_lifetimes()

    @unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four GPUs")
    def test_cp4_sharded_cache_causality_and_ring_lifetimes(self):
        self._check_sharded_cache_causality_and_ring_lifetimes(world_size=4)

    def _check_sharded_cache_causality_and_ring_lifetimes(self, world_size=2):
        with tempfile.TemporaryDirectory() as result_dir:
            mp.spawn(
                _worker, args=(result_dir, world_size), nprocs=world_size, join=True
            )
            results = [
                json.loads(Path(result_dir, f"rank{r}.json").read_text())
                for r in range(world_size)
            ]
            print(json.dumps({"zigzag_ring_results": results}), flush=True)


class _Norm(torch.nn.Module):
    def forward(self, hidden, residual=None):
        summed = hidden if residual is None else hidden + residual
        normalized = (
            summed.float() * torch.rsqrt(summed.float().square().mean(-1, True) + 1e-6)
        ).to(hidden.dtype)
        return normalized if residual is None else (normalized, summed)


def _experts(hidden, weights, channel_slice, *, launch_residual=None):
    """Four routed experts, top-2 per token, plus a sigmoid-gated shared expert."""
    gate, up, down, shared_up, shared_down, shared_gate = weights
    probabilities, ids = (hidden.float() @ gate).softmax(-1).topk(2, dim=-1)
    probabilities /= probabilities.sum(-1, keepdim=True)
    if launch_residual is not None:
        launch_residual()
    result = torch.zeros_like(hidden)
    for expert in range(4):
        mass = ((ids == expert) * probabilities).sum(-1, keepdim=True)
        activation = F.silu(hidden @ up[expert, :, channel_slice])
        partial = activation @ down[expert, channel_slice]
        result += (partial.float() * mass).to(hidden.dtype)
    shared = F.silu(hidden @ shared_up[:, channel_slice]) @ shared_down[channel_slice]
    return result + (shared.float() * (hidden.float() @ shared_gate).sigmoid()).to(
        hidden.dtype
    )


def _run_case(
    rank,
    world,
    lengths,
    dtype,
    *,
    moe_tp=True,
    residual_exchange=None,
    defer_residual=False,
    previous=None,
):
    device = torch.device("cuda", rank)
    torch.manual_seed(619)
    hidden_size, intermediate = 64, 128 * world
    tokens = sum(lengths)
    original = torch.randn(tokens, hidden_size, device=device, dtype=dtype) * 0.1
    weights = (
        torch.randn(hidden_size, 4, device=device) * 0.1,
        torch.randn(4, hidden_size, intermediate, device=device, dtype=dtype) * 0.02,
        torch.randn(4, intermediate, hidden_size, device=device, dtype=dtype) * 0.02,
        torch.randn(hidden_size, intermediate, device=device, dtype=dtype) * 0.02,
        torch.randn(intermediate, hidden_size, device=device, dtype=dtype) * 0.02,
        torch.randn(hidden_size, 1, device=device) * 0.1,
    )
    original_count = torch.tensor(tokens, device=device, dtype=torch.int32)
    batch = SimpleNamespace(
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens_cpu=tuple(64 * i for i in range(len(lengths))),
        num_token_non_padded=original_count,
    )
    from sglang.srt.layers.moe import MoeA2ABackend
    from sglang.srt.runtime_context import get_flags

    with (
        get_flags().moe.override(
            a2a_backend=MoeA2ABackend.NONE if moe_tp else MoeA2ABackend.DEEPEP
        ),
        mock_parallel_groups(
            attn_cp_size=world,
            attn_cp_rank=rank,
            moe_tp_size=world if moe_tp else 1,
            attn_cp_group=SimpleNamespace(device_group=dist.group.WORLD),
        ),
    ):
        batch.prefill_cp_metadata = ContextParallelMetadata.from_prefill(
            batch, device, residual_exchange=residual_exchange, previous=previous
        )
    frame = batch.prefill_cp_metadata
    hidden, residual = original, None
    reference, reference_residual = original.clone(), None
    norm = _Norm()
    layouts = ("contiguous", "contiguous", "attention", "contiguous")
    for layer_index, layout in enumerate(layouts):
        next_layout = (
            layouts[layer_index + 1] if layer_index + 1 < len(layouts) else None
        )
        hidden, residual = _prepare_attention_layout(
            frame, hidden, residual, batch, layout, norm
        )
        if reference_residual is None:
            reference_residual = reference
            reference = norm(reference)
        else:
            reference, reference_residual = norm(reference, reference_residual)
        # A token-local attention stand-in isolates the new MoE boundary from
        # GDN/FA arithmetic, which has separate operator accuracy tests.
        hidden = hidden * 0.25
        reference = reference * 0.25
        stream = ResidualStream(residual)
        hidden = stream.record(hidden, PLAIN_ADD)
        hidden = prepare_sharded_cp_ffn(
            hidden,
            stream,
            batch,
            norm,
            next_layout=next_layout,
            defer_cp_residual=defer_residual,
        )
        residual = stream.residual
        if defer_residual:
            assert frame.pending_residual is None
            needs_exchange = (
                residual_exchange is not None
                and next_layout is not None
                and frame.layout_names[next_layout] != frame.current
            )
            assert (frame.queued_residual is not None) == needs_exchange
        reference, reference_residual = norm(reference, reference_residual)
        rows = list(frame.token_layouts[frame.current][rank])
        torch.testing.assert_close(residual, reference_residual[rows], rtol=0, atol=0)
        assert hidden.shape[0] == (tokens if moe_tp else len(rows))
        assert batch.num_token_non_padded.item() == hidden.shape[0]
        with nullcontext():
            channels = slice(rank * 128, (rank + 1) * 128) if moe_tp else slice(None)
            hidden = _experts(
                hidden,
                weights,
                channels,
                launch_residual=(
                    (lambda: launch_cp_residual_exchange(frame))
                    if defer_residual
                    else None
                ),
            )
        assert frame.queued_residual is None
        reference = _experts(reference, weights, slice(None))
        hidden, residual = finish_sharded_cp_ffn(hidden, residual, forward_batch=batch)
        # A fused RS already owns the successor's rows. The residual is joined
        # by prepare_attn on that layer, before any read or in-place norm update.
        rows = list(frame.token_layouts[frame.current][rank])
        assert batch.num_token_non_padded.item() == len(rows)
        torch.testing.assert_close(
            hidden,
            reference[rows],
            rtol=0.03 if dtype == torch.bfloat16 else 2e-5,
            atol=0.002 if dtype == torch.bfloat16 else 1e-6,
        )
        # Avoid accumulating BF16 reduction-order drift into the next oracle.
        reference = cp_gather_full_sequence_states(
            hidden, batch, torch.cuda.current_stream()
        )
    actual = cp_gather_full_sequence_states(hidden, batch, torch.cuda.current_stream())
    assert frame.pending_residual is None
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert batch.num_token_non_padded is original_count
    if previous is not None:
        assert frame.query_indices is previous.query_indices
    return frame


def _moe_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    exchange = TokenExchangeStream(
        group=dist.group.WORLD, device=torch.device("cuda", rank)
    )
    assert exchange.group is dist.group.WORLD
    for residual_exchange, defer_residual in (
        (None, False),
        (exchange, False),
        (exchange, True),
    ):
        for dtype in (torch.float32, torch.bfloat16):
            for lengths in (
                (0,),
                (1,),
                (129, 1, 0),
                (513, 65),
                (16384,),
                (31121,),
                (32768,),
            ):
                frame = _run_case(
                    rank,
                    2,
                    lengths,
                    dtype,
                    residual_exchange=residual_exchange,
                    defer_residual=defer_residual,
                )
                if lengths == (16384,):
                    _run_case(
                        rank,
                        2,
                        lengths,
                        dtype,
                        residual_exchange=residual_exchange,
                        defer_residual=defer_residual,
                        previous=frame,
                    )
            for lengths in ((0,), (1,), (129, 1, 0), (513, 65), (16384,)):
                _run_case(
                    rank,
                    2,
                    lengths,
                    dtype,
                    moe_tp=False,
                    residual_exchange=residual_exchange,
                    defer_residual=defer_residual,
                )
    torch.cuda.synchronize()
    dist.destroy_process_group()


class TestRingCPMoETP(CustomTestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two CUDA GPUs")
    def test_moe_tp_layout_and_reduction(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = f"file://{directory}/rendezvous"
            mp.spawn(_moe_worker, args=(rendezvous,), nprocs=2, join=True)


class TestContextParallelMetadata(unittest.TestCase):
    def setUp(self):
        from sglang.srt.runtime_context import get_context

        config = get_context().override_server_args(tp_size=2, attn_cp_size=2)
        config.install()
        self.addCleanup(config.restore)
        self.exchange = object()
        self.parallel = SimpleNamespace(
            attn_cp_group=SimpleNamespace(
                device_group=object(), rank_in_group=0, world_size=2
            ),
            attn_cp_rank=0,
            attn_cp_size=2,
            moe_tp_size=2,
        )
        for context in (
            patch(
                "sglang.srt.distributed.get_attn_cp_group",
                side_effect=lambda: self.parallel.attn_cp_group,
            ),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def frame(
        self,
        lengths=(129, 65),
        prefixes=(0, 0),
        previous=None,
        cp_mode="round-robin-split",
    ):
        batch = SimpleNamespace(
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens_cpu=prefixes,
            num_token_non_padded=torch.tensor(sum(lengths)),
        )
        from sglang.srt.layers.moe import MoeA2ABackend
        from sglang.srt.runtime_context import get_flags

        with get_flags().moe.override(
            a2a_backend=(
                MoeA2ABackend.NONE
                if self.parallel.moe_tp_size > 1
                else MoeA2ABackend.DEEPEP
            )
        ):
            return ContextParallelMetadata.from_prefill(
                batch,
                torch.device("cpu"),
                residual_exchange=self.exchange,
                previous=previous,
                cp_mode=cp_mode,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gdn_metadata_does_not_read_device_state_on_host(self):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            GDNAttnBackend,
            make_gdn_cp_conv_metadata,
        )

        backend = object.__new__(GDNAttnBackend)
        backend.kernel_dispatcher = SimpleNamespace(use_context_parallel=True)
        mapping = torch.tensor([5, 2, 7], dtype=torch.int32, device="cuda")
        backend.req_to_token_pool = SimpleNamespace(
            get_mamba_indices=lambda indices: mapping[indices],
            mamba_pool=SimpleNamespace(synchronize=unittest.mock.Mock()),
        )
        checkpoints = (((256, mapping[:1]),), (), ((0, mapping[1:2]),))
        from sglang.srt.model_executor.forward_mode import ForwardMode

        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            prefill_cp_metadata=SimpleNamespace(
                lengths=(513, 1, 65), world=2, rank=0, device=torch.device("cuda")
            ),
            req_pool_indices=torch.tensor([2, 0, 1], device="cuda"),
            batch_size=3,
            mamba_track_checkpoints=checkpoints,
            mamba_track_mask=torch.tensor([True, False, True], device="cuda"),
        )
        batch.prefill_cp_metadata.gdn_conv_metadata = make_gdn_cp_conv_metadata(
            batch.prefill_cp_metadata
        )
        torch.cuda.synchronize()
        mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            backend.init_forward_metadata(batch)
            self.assertEqual(
                backend.req_to_token_pool.mamba_pool.synchronize.call_count, 1
            )
        finally:
            torch.cuda.set_sync_debug_mode(mode)
        self.assertIs(backend.cp_checkpoints, checkpoints)
        self.assertEqual(backend.cp_state_slots.tolist(), [7, 5, 2])
        self.assertEqual(
            [m["query_start_loc"].tolist() for m in backend.cp_conv_metadata],
            [[0, 320], [0, 1], [0, 64]],
        )
        # The next chunk must use its own checkpoint snapshot, even when the
        # token geometry and device plans can be reused.
        next_checkpoints = ((), ((1, mapping[2:3]),), ())
        batch.mamba_track_checkpoints = next_checkpoints
        backend.init_forward_metadata(batch)
        self.assertIs(backend.cp_checkpoints, next_checkpoints)
        batch.mamba_track_checkpoints = None
        with self.assertRaisesRegex(ValueError, "scheduler checkpoint"):
            backend.init_forward_metadata(batch)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cpu_planning_and_batched_upload_across_streams(self):
        device = torch.device("cuda", torch.cuda.current_device())
        schedule, forward = torch.cuda.Stream(), torch.cuda.Stream()
        previous = None
        for lengths, prefixes in (
            ((129, 65), (7, 3)),
            ((129, 65), (9, 5)),
            ((1, 0), (0, 0)),
        ):
            # Planning may describe CUDA execution but must not initialize or
            # synchronize CUDA, launch kernels, or upload individual indices.
            with patch(
                "torch.Tensor.to",
                side_effect=AssertionError("CPU plan uploaded a tensor"),
            ):
                frame = ContextParallelMetadata.plan_prefill(
                    lengths,
                    prefixes,
                    device,
                    request_indices=(2, 0),
                    request_stride=512,
                    previous=previous,
                    cp_mode="round-robin-split",
                )
            with torch.cuda.stream(schedule):
                from sglang.srt.layers.attention.linear.gdn_backend import (
                    make_gdn_cp_conv_metadata,
                )

                frame.gdn_conv_metadata = tuple(
                    make_gdn_cp_conv_metadata(frame, device="cpu")
                )
                frame.upload()
                self.assertEqual(len(frame.device_buffers), 1)
                self.assertEqual(
                    frame.gdn_conv_metadata[0]["has_initial_state"].dtype, torch.bool
                )
                # Preserve warmup's pointer specialization even for the bool
                # metadata of a one-token request with empty CP peers.
                for metadata in frame.gdn_conv_metadata:
                    for name in ("has_initial_state", "query_start_loc"):
                        self.assertEqual(metadata[name].data_ptr() % 16, 0)
                table = (
                    torch.arange(4 * 512, device=device, dtype=torch.int32).reshape(
                        4, 512
                    )
                    * 2
                )
                locations = torch.arange(sum(lengths), device=device) * 2
                frame.resolve_kv_slots(table, locations)
            with torch.cuda.stream(forward):
                forward.wait_stream(schedule)
                frame.record_stream(forward)
                actual = frame.kv_read_slots.clone()
                writes = frame.kv_write_slots.clone()
            forward.synchronize()
            expected = torch.cat(
                [
                    torch.arange(n + p)[frame.rank :: frame.world] + slot * 512
                    for n, p, slot in zip(lengths, prefixes, (2, 0))
                ]
            )
            torch.testing.assert_close(actual.cpu(), expected.long())
            torch.testing.assert_close(
                writes.cpu(),
                torch.tensor(frame.layout.new_kv_indices[frame.rank], dtype=torch.long),
            )
            self.assertTrue(
                all(plan.send_indices.is_cuda for plan in frame.routes.values())
            )
            previous = frame

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_runner_prepares_then_backend_only_binds_metadata(self):
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.model_executor.model_runner import ModelRunner

        table = torch.arange(1024, dtype=torch.int32, device="cuda").reshape(2, 512) * 2
        pool = SimpleNamespace(req_to_token=table)
        full = SimpleNamespace(
            device=table.device,
            prefill_cp_mode="in-seq-split",
            residual_exchange=None,
            forward_metadata=None,
            cp_packed_kv=False,
            kv_cache_dtype=torch.bfloat16,
            cp_exchange=SimpleNamespace(capacity=512),
            req_to_token_pool=pool,
        )
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        hybrid = object.__new__(HybridLinearAttnBackend)
        hybrid.full_attn_backend = full
        hybrid.linear_attn_backend = SimpleNamespace(
            device=table.device, _cp_residual_exchange=None
        )
        hybrid.req_to_token_pool = pool
        hybrid._cp_metadata_plan = None
        runner = SimpleNamespace(
            server_args=SimpleNamespace(
                attn_cp_size=2,
                linear_attn_prefill_backend="flashqla",
                disaggregation_mode="prefill",
                encoder_only=False,
            ),
            attn_backend=hybrid,
            req_to_token_pool=pool,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            prefill_cp_metadata=None,
            extend_lens=[129, 65],
            prefix_lens=[7, 3],
            reqs=[
                SimpleNamespace(kv=SimpleNamespace(req_pool_idx=1)),
                SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0)),
            ],
            out_cache_loc=torch.arange(194, device="cuda") * 2,
            num_token_non_padded=torch.tensor(194, device="cuda", dtype=torch.int32),
        )
        ModelRunner.prepare_cp_metadata(runner, batch)
        frame = batch.prefill_cp_metadata
        self.assertIsNone(full.forward_metadata)
        self.assertEqual(len(frame.device_buffers), 1)
        self.assertEqual(len(frame.gdn_conv_metadata), 2)
        with patch.object(
            ContextParallelMetadata,
            "plan_prefill",
            side_effect=AssertionError("Late CPU plan"),
        ):
            ModelRunner.prepare_cp_metadata(runner, batch)
            AttentionBackend._init_cp_forward_metadata(full, batch)
        self.assertIs(full.forward_metadata, frame)
        torch.testing.assert_close(
            frame.kv_read_slots.cpu(),
            torch.cat((torch.arange(0, 136, 2) + 512, torch.arange(0, 68, 2))),
        )
        self.assertTrue(frame.gdn_conv_metadata[0]["has_initial_state"].item())

    def test_split_mode_changes_fa_layout_but_preserves_gdn_order(self):
        for lengths, prefixes in (
            ((513,), (7,)),
            ((129, 65), (7, 3)),
            ((0, 1, 257), (0, 8, 2)),
        ):
            with self.subTest(lengths=lengths, prefixes=prefixes):
                zigzag = self.frame(lengths, prefixes, cp_mode="in-seq-split")
                round_robin = self.frame(lengths, prefixes, previous=zigzag)
                self.assertEqual(zigzag.contiguous, round_robin.contiguous)
                self.assertEqual(
                    zigzag.layout.new_kv_indices, round_robin.layout.new_kv_indices
                )
                self.assertEqual(
                    round_robin.attention, round_robin.layout.new_kv_indices
                )
                self.assertNotEqual(zigzag.attention, round_robin.attention)
                self.assertIsNot(zigzag.query_indices, round_robin.query_indices)
                for frame in (zigzag, round_robin):
                    for layout in (frame.contiguous, frame.attention):
                        self.assertEqual(
                            sorted(i for shard in layout for i in shard),
                            list(range(sum(lengths))),
                        )
                    for shard in frame.contiguous:
                        self.assertEqual(list(shard), sorted(shard))

    def test_default_is_existing_in_sequence_split(self):
        self.exchange = None
        batch = SimpleNamespace(
            extend_seq_lens_cpu=[128],
            extend_prefix_lens_cpu=[0],
            num_token_non_padded=torch.tensor(128),
        )
        default = ContextParallelMetadata.from_prefill(batch, torch.device("cpu"))
        explicit = self.frame((128,), (0,), cp_mode="in-seq-split")
        self.assertEqual(default.attention, explicit.attention)
        self.assertEqual(
            default.attention,
            (tuple(range(32)) + tuple(range(96, 128)), tuple(range(32, 96))),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported prefill CP mode"):
            self.frame(cp_mode="bad")

    def test_position_split_preserves_legacy_and_hybrid_layouts(self):
        from sglang.srt.layers.cp.utils import cp_shard_position_ids

        positions = torch.arange(32).reshape(2, 16)
        from sglang.srt.layers.cp.zigzag import (
            ZigzagContextParallelMetadata,
            ZigzagCPStrategy,
        )

        legacy = ZigzagContextParallelMetadata(
            split_list=[4, 4, 4, 4], zigzag_index=[0, 3]
        )
        legacy.total_seq_lens = 16
        batch = SimpleNamespace(attn_cp_metadata=legacy, prefill_cp_metadata=None)
        with (
            patch("sglang.srt.layers.cp.utils.is_cp_active", return_value=True),
            patch(
                "sglang.srt.layers.cp.utils.get_cp_strategy",
                return_value=object.__new__(ZigzagCPStrategy),
            ),
        ):
            torch.testing.assert_close(
                cp_shard_position_ids(positions, batch),
                torch.cat((positions[:, :4], positions[:, 12:]), dim=-1),
            )
        for mode in ("in-seq-split", "round-robin-split"):
            frame = self.frame((16,), (0,), cp_mode=mode)
            batch.prefill_cp_metadata = frame
            torch.testing.assert_close(
                cp_shard_position_ids(positions, batch),
                positions[:, list(frame.attention[frame.rank])],
            )

    def test_reuse_updates_causality_and_resets_execution_state(self):
        first = self.frame()
        first.current = first.layout_names["contiguous"]
        gather = first.token_collectives()
        first.pending_residual = object()
        first.queued_residual = object()
        first.mlp_destination = first.layout_names["attention"]
        second = self.frame(prefixes=(130, 66), previous=first)
        self.assertEqual(second.layout.seq_lens, (259, 131))
        self.assertEqual(second.prefixes, (130, 66))
        self.assertIsNone(second.current)
        self.assertIsNone(second.pending_residual)
        self.assertIsNone(second.queued_residual)
        self.assertIsNone(second.mlp_destination)
        self.assertIsNot(second.original_non_padded, first.original_non_padded)
        self.assertIs(second.query_indices, first.query_indices)
        self.assertIs(
            second.gather_scatter_plans[second.layout_names["contiguous"]], gather
        )
        self.assertIsNot(second.routes, first.routes)
        for key, plan in first.routes.items():
            self.assertIs(second.routes[key], plan)
        for key, plan in first.moe_output_plans.items():
            self.assertIs(second.moe_output_plans[key], plan)

    def test_ownership_changes_invalidate_all_plans(self):
        first = self.frame()
        for lengths, prefixes in (((129, 65), (1, 0)), ((128, 66), (0, 0))):
            with self.subTest(lengths=lengths, prefixes=prefixes):
                second = self.frame(lengths, prefixes, previous=first)
                self.assertIsNot(second.query_indices, first.query_indices)
                self.assertEqual(
                    set(second.gather_scatter_plans),
                    {second.layout_names[name] for name in second.token_layouts},
                )
                self.assertFalse(
                    {id(p) for p in second.gather_scatter_plans.values()}
                    & {id(p) for p in first.gather_scatter_plans.values()}
                )
                self.assertFalse(
                    {id(p) for p in second.routes.values()}
                    & {id(p) for p in first.routes.values()}
                )

    def test_layer_execution_uses_prepared_indices(self):
        for mode in ("in-seq-split", "round-robin-split"):
            for lengths in ((0,), (1,), (129, 1, 0), (513, 65), (16384,)):
                for moe_tp in (1, 2):
                    with self.subTest(mode=mode, lengths=lengths, moe_tp=moe_tp):
                        self.parallel.moe_tp_size = moe_tp
                        frame = self.frame(lengths, (0,) * len(lengths), cp_mode=mode)
                        original = torch.arange(sum(lengths) * 3).reshape(-1, 3)
                        batch = SimpleNamespace(num_token_non_padded=None)
                        with ExitStack() as stack:
                            for factory in (
                                "make_token_gather_scatter",
                                "make_token_exchange",
                                "make_token_reduce_scatter",
                                "_cp_metadata_tensor",
                            ):
                                stack.enter_context(
                                    patch(
                                        "sglang.srt.layers.cp.utils." + factory,
                                        side_effect=AssertionError(
                                            "Forward must not build CP indices"
                                        ),
                                    )
                                )
                            for name, indices in frame.token_layouts.items():
                                frame.current = None
                                hidden, residual = _prepare_attention_layout(
                                    frame, original, original, batch, name
                                )
                                expected = original[list(indices[frame.rank])]
                                torch.testing.assert_close(hidden, expected)
                                torch.testing.assert_close(residual, expected)
                                plan = frame.token_collectives()
                                self.assertEqual(plan.local_rows, len(expected))
                                self.assertEqual(
                                    batch.num_token_non_padded.item(), len(expected)
                                )
                            for source, target in (
                                ("contiguous", "attention"),
                                ("attention", "contiguous"),
                                ("attention", "kv"),
                            ):
                                frame.exchange_plan(source, target)

    def test_group_change_and_missing_previous_do_not_reuse_device_plans(self):
        first = self.frame()
        second = self.frame(previous=None)
        self.assertIsNot(second.query_indices, first.query_indices)
        self.parallel.attn_cp_group.device_group = object()
        third = self.frame(previous=first)
        self.assertIsNot(third.query_indices, first.query_indices)

    def test_exchange_rejects_invalid_token_ownership(self):
        for source, target, subset in (
            (((0, 0), (1,)), ((0,), (1,)), False),
            (((0,), (1,)), ((0, 1), (1,)), False),
            (((0,), (1,)), ((0,), (2,)), False),
            (((0,), (1,)), ((2,), ()), True),
        ):
            with self.subTest(source=source, target=target, subset=subset):
                with self.assertRaisesRegex(ValueError, "unique tokens"):
                    make_token_exchange(
                        source=source,
                        target=target,
                        group=self.parallel.attn_cp_group.device_group,
                        device=torch.device("cpu"),
                        allow_subset=subset,
                    )

    def test_combine_destinations_preserve_token_identity(self):
        for lengths in ((0,), (1,), (129, 1, 0), (513, 65), (16384,)):
            contiguous = contiguous_chunk_indices(lengths, 2)
            interleaved = tuple(tuple(range(r, sum(lengths), 2)) for r in range(2))
            for source, target in (
                (contiguous, interleaved),
                (interleaved, contiguous),
                (contiguous, tuple(rows[::2] for rows in interleaved)),
            ):
                expected = {
                    token: (rank, row)
                    for rank, rows in enumerate(target)
                    for row, token in enumerate(rows)
                }
                for rank in range(2):
                    with (
                        self.subTest(lengths=lengths, rank=rank),
                        patch("torch.distributed.get_rank", return_value=rank),
                    ):
                        plan = make_token_exchange(
                            source=source,
                            target=target,
                            group=self.parallel.attn_cp_group.device_group,
                            device=torch.device("cpu"),
                            allow_subset=True,
                        )
                        destinations = plan.output_destinations.tolist()
                        self.assertEqual(len(destinations), len(source[rank]))
                        for token, (peer, row) in zip(source[rank], destinations):
                            if token in expected:
                                self.assertEqual((peer, row), expected[token])
                            else:
                                self.assertEqual(peer, -1)


def _collective_cases(rank, world):
    device = torch.device("cuda", rank)
    for dtype in (torch.float32, torch.bfloat16):
        for lengths in ((0,), (1,), (129, 1, 0), (513, 65), (31121,), (32768,)):
            tokens = sum(lengths)
            contiguous = contiguous_chunk_indices(lengths, world)
            # A deliberately different permutation with uneven/empty ranks.
            interleaved = tuple(tuple(range(r, tokens, world)) for r in range(world))
            reference = (
                torch.arange(tokens * 17, device=device).reshape(tokens, 17) % 13
            ).to(dtype)
            for source in (contiguous, interleaved):
                plan = make_token_gather_scatter(
                    indices=source, group=dist.group.WORLD, device=device
                )
                local = reference[list(source[rank])]
                for rank_ordered in (False, True):
                    gathered = plan.gather(local, rank_ordered=rank_ordered)
                    balanced = tokens == world * max(map(len, source))
                    order = (
                        [t for rows in source for t in rows]
                        if balanced and rank_ordered
                        else list(range(tokens))
                    )
                    torch.testing.assert_close(
                        gathered, reference[order], rtol=0, atol=0
                    )
                    partial = gathered * (rank + 1)
                    result = plan.reduce_scatter(partial, rank_ordered=rank_ordered)
                    expected = local * (world * (world + 1) // 2)
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                for target in (contiguous, interleaved):
                    exchange = make_token_exchange(
                        source=source,
                        target=target,
                        group=dist.group.WORLD,
                        device=device,
                    )
                    torch.testing.assert_close(
                        exchange.apply(local),
                        reference[list(target[rank])],
                        rtol=0,
                        atol=0,
                    )
                    if dtype == torch.bfloat16 and tokens <= 1024:
                        from sglang.srt.layers.layernorm import GemmaRMSNorm

                        norm = GemmaRMSNorm(17).to(device=device, dtype=dtype)
                        count = torch.tensor(len(target[rank]), device=device)
                        frame = SimpleNamespace(
                            layout_names={"source": "source", "target": "target"},
                            token_layouts={"source": source, "target": target},
                            rank=rank,
                            current="source",
                            pending_residual=None,
                            exchange_plan=lambda source, target: exchange,
                            local_token_count=lambda rows: count,
                        )
                        batch = SimpleNamespace(prefill_cp_metadata=frame)
                        with torch.inference_mode():
                            actual = _prepare_attention_layout(
                                frame, local, local.clone(), batch, "target", norm
                            )
                            expected_input = reference[list(target[rank])]
                            expected = (
                                norm(expected_input.clone(), expected_input.clone())
                                if expected_input.numel()
                                else (expected_input, expected_input)
                            )
                        torch.testing.assert_close(
                            actual[1], expected[1], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            actual[0], expected[0], rtol=8e-3, atol=1e-6
                        )
                        assert frame.current == "target"
                        assert batch.num_token_non_padded is count
                    reducer = make_token_reduce_scatter(
                        source=source,
                        target=target,
                        group=dist.group.WORLD,
                        device=device,
                    )
                    gathered = plan.gather(local, rank_ordered=True)
                    result = reducer.apply(gathered * (rank + 1))
                    expected = reference[list(target[rank])] * (
                        world * (world + 1) // 2
                    )
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
            # Exercise subset exchange and identity send/restore independently.
            subset = tuple(rows[::2] for rows in contiguous)
            exchange = make_token_exchange(
                source=contiguous,
                target=subset,
                group=dist.group.WORLD,
                device=device,
                allow_subset=True,
            )
            torch.testing.assert_close(
                exchange.apply(reference[list(contiguous[rank])]),
                reference[list(subset[rank])],
                rtol=0,
                atol=0,
            )


def _token_worker(rank, world, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        _collective_cases(rank, world)
        _symmetric_residual_cases(rank, world)
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


def _symmetric_residual_cases(rank, world):
    device = torch.device("cuda", rank)
    for dtype in (torch.bfloat16, torch.float32):
        exchange = TokenExchangeStream(
            group=dist.group.WORLD,
            device=device,
            capacity=16384,
            features=4 * 17,
            dtype=dtype,
            transport="symm_rma",
        )
        storage = exchange.workspace.buffers.data_ptr()
        for lengths in ((0,), (1,), (3, 5), (129, 1, 0), (8192,), (16384,)):
            tokens = sum(lengths)
            reference = (torch.arange(tokens * 68, device=device) % 79).to(dtype)
            reference = reference.view(tokens, 4, 17)
            layouts = (
                contiguous_chunk_indices(lengths, world),
                tuple(tuple(range(peer, tokens, world)) for peer in range(world)),
            )
            if tokens <= 130:
                layouts += (tuple(tuple(reversed(rows)) for rows in layouts[1]),)
            local = reference[list(layouts[0][rank])]
            # Consume a workspace view as the next sender, reusing both slots.
            # This catches premature reuse and missing producer/consumer events.
            with torch.inference_mode():
                for step in range(6):
                    source = layouts[step % len(layouts)]
                    target = layouts[(step + 1) % len(layouts)]
                    plan = make_token_exchange(
                        source=source,
                        target=target,
                        group=dist.group.WORLD,
                        device=device,
                    )
                    ready = torch.cuda.current_stream().record_event()
                    with patch.object(
                        type(plan), "apply", side_effect=AssertionError("NCCL fallback")
                    ):
                        pending = exchange.submit(
                            plan, local, ready=ready, restore_order=False
                        )
                    local = pending.wait()
                    torch.testing.assert_close(
                        local, reference[list(target[rank])], rtol=0, atol=0
                    )
                    assert exchange.workspace.buffers.data_ptr() == storage
        exchange.stream.synchronize()
        exchange.workspace.close()


class TestCPTokenLayout(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two CUDA GPUs")
    def test_collective_layouts(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(_token_worker, args=(2, port), nprocs=2, join=True)


def _flashqla_context_worker(rank, world, port):
    from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE
    from flash_qla.utils.index import prepare_chunk_indices, prepare_chunk_offsets

    from sglang.srt.layers.attention.linear.kernels.gdn_flashqla import (
        FlashQLAGDNKernel,
    )

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        kernel = FlashQLAGDNKernel(
            fallback=None, cp_group=dist.group.WORLD, symm_state=False
        )
        kernel.warmup_cp_contexts(device="cuda", max_tokens=16384, value_heads=64)
        contexts = dict(kernel._cp_contexts)
        assert len(contexts) == 16384 // (64 * world)
        expected = {}
        for context in contexts.values():
            tensors = [context.cu_seqlens]
            if context.is_intra:
                tensors.append(context.intra_cp_cu_seqlens)
            for tensor in tensors:
                counts = (tensor.cpu().diff() + CHUNK_SIZE - 1) // CHUNK_SIZE
                expected[id(tensor)] = int(counts.sum())
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            with patch.object(
                kernel, "_build_context", side_effect=AssertionError("Lazy CP context")
            ):
                # More than eight alternating lengths must retain tensor identity;
                # both implicit and explicit CUDA device names hit the same cache.
                for (device, length, heads), context in reversed(contexts.items()):
                    for target in (device, "cuda"):
                        actual = kernel.prepare_cp_context(
                            device=target, padded_length=length, value_heads=heads
                        )
                        assert actual is context
                        if kernel._prepare_varlen_metadata is not None:
                            tensors = [actual.cu_seqlens]
                            if actual.is_intra:
                                tensors.append(actual.intra_cp_cu_seqlens)
                            for tensor in tensors:
                                indices = prepare_chunk_indices(tensor, CHUNK_SIZE)
                                _, count = prepare_chunk_offsets(tensor, CHUNK_SIZE)
                                assert count == expected[id(tensor)]
                                assert indices.shape == (count, 2)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    finally:
        dist.destroy_process_group()


class TestCPDecoderParticipation(unittest.TestCase):
    def test_moe_submits_residual_before_experts_on_empty_and_active_shards(self):
        from types import MethodType
        from unittest.mock import Mock

        from sglang.srt.models import qwen2_moe

        for rows in (0, 1, 17):
            for cp, mapped in ((False, False), (True, False), (True, True)):
                with self.subTest(rows=rows, cp=cp, mapped=mapped):
                    calls = []
                    hidden = torch.ones((rows, 4))
                    pending = object()
                    plan, ready = object(), object()

                    def submit(actual_plan, residual, *, ready, restore_order):
                        self.assertIs(actual_plan, plan)
                        self.assertIs(residual, hidden)
                        self.assertFalse(restore_order)
                        calls.append("residual")
                        return pending

                    metadata = SimpleNamespace(
                        queued_residual=("attention", plan, hidden, ready),
                        pending_residual=None,
                        residual_exchange=SimpleNamespace(submit=submit),
                        moe_tp=False,
                        current="contiguous",
                        mlp_destination="attention" if mapped else "contiguous",
                    )

                    def experts(value, topk, **kwargs):
                        self.assertIs(
                            kwargs.get("output_layout"), metadata if mapped else None
                        )
                        self.assertEqual(calls, ["residual"] if cp else [])
                        if cp:
                            self.assertIsNone(metadata.queued_residual)
                            self.assertEqual(
                                metadata.pending_residual, ("attention", pending)
                            )
                        calls.append("experts")
                        return value.new_ones((rows + int(mapped), 4))

                    def shared_experts(value, **kwargs):
                        if mapped:
                            self.assertEqual(calls, ["residual"])
                            self.assertIsNone(metadata.queued_residual)
                        return None

                    def router(value):
                        if mapped:
                            self.assertEqual(calls, ["residual"])
                        return value, None

                    layer = SimpleNamespace(
                        _use_mega_moe=False,
                        shared_expert_gate=None,
                        alt_stream=None,
                        reduce_results=False,
                        enable_shared_expert_fusion=False,
                        gate=router,
                        topk=Mock(return_value=object()),
                        experts=experts,
                        _forward_shared_experts=shared_experts,
                    )
                    layer._forward_router_experts = MethodType(
                        qwen2_moe.Qwen2MoeSparseMoeBlock._forward_router_experts, layer
                    )
                    backend = SimpleNamespace(
                        **{
                            "is_" + name: lambda: False
                            for name in ("deepep", "deepep_v2", "mori", "ascend_fuseep")
                        }
                    )
                    with (
                        patch.object(
                            qwen2_moe, "get_moe_a2a_backend", return_value=backend
                        ),
                        patch.object(
                            qwen2_moe,
                            "get_exec",
                            return_value=SimpleNamespace(
                                moe=SimpleNamespace(enable_mxfp4_mega=mapped)
                            ),
                        ),
                    ):
                        output = qwen2_moe.Qwen2MoeSparseMoeBlock.forward(
                            layer,
                            hidden,
                            SimpleNamespace(
                                prefill_cp_metadata=metadata if cp else None,
                                layer_cp_metadata=metadata if cp else None,
                                moe_num_token_non_padded=lambda: torch.tensor(rows),
                            ),
                        )
                    torch.testing.assert_close(
                        output, hidden.new_ones((rows + int(mapped), 4))
                    )
                    self.assertEqual(
                        calls, ["residual", "experts"] if cp else ["experts"]
                    )

    def test_empty_shards_enter_attention_only_when_cp_is_active(self):
        from contextlib import contextmanager

        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.models.qwen3_5 import (
            Qwen3_5AttentionDecoderLayer,
            Qwen3_5LinearDecoderLayer,
        )

        class Boundary:
            def prepare(self, hidden, batch, **kwargs):
                return hidden

            def finish(self, hidden, batch=None):
                return hidden

            @contextmanager
            def exit(self, batch):
                yield self

        for decoder_type in (Qwen3_5AttentionDecoderLayer, Qwen3_5LinearDecoderLayer):
            for mode, cp, rows, expected in (
                (ForwardMode.EXTEND, True, 0, 1),
                (ForwardMode.EXTEND, True, 1, 1),
                (ForwardMode.EXTEND, False, 0, 0),
                (ForwardMode.EXTEND, False, 1, 1),
                (ForwardMode.IDLE, False, 0, 0),
            ):
                with self.subTest(
                    decoder=decoder_type.__name__, mode=mode, cp=cp, rows=rows
                ):
                    calls = []

                    def attention(hidden_states, forward_batch, **kwargs):
                        calls.append(hidden_states.shape[0])
                        return hidden_states

                    layer = SimpleNamespace(
                        attn_boundary=Boundary(),
                        ffn_boundary=Boundary(),
                        linear_attn=attention,
                        self_attention=attention,
                        mlp=torch.nn.Identity(),
                    )
                    batch = SimpleNamespace(
                        forward_mode=mode,
                        prefill_cp_metadata=object() if cp else None,
                    )
                    hidden = torch.empty((rows, 4))
                    output = decoder_type.forward(
                        layer,
                        hidden_states=hidden,
                        forward_batch=batch,
                        positions=torch.arange(rows),
                    )
                    self.assertEqual(len(calls), expected)
                    self.assertIs(output, hidden)


class TestCPWarmup(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_gates_reuse_loaded_kernels_after_startup(self):
        import triton

        from sglang.kernels.ops.elementwise.elementwise import (
            fused_gate_sigmoid_mul,
            fused_gate_sigmoid_mul_add,
            fused_sigmoid_mul,
            warmup_fused_gate_sigmoid_mul,
            warmup_fused_sigmoid_mul,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        for dtype in (torch.bfloat16, torch.float16):
            for width in (4096, 8192):
                warmup_fused_gate_sigmoid_mul(
                    device=device, dtype=dtype, weight_dtype=dtype, hidden_dim=width
                )
                warmup_fused_sigmoid_mul(device=device, dtype=dtype, hidden_dim=width)
                weight = torch.randn(width, device=device, dtype=dtype) * 0.01
                for rows in (1, 17, 1023, 1024, 4096):
                    with self.subTest(dtype=dtype, width=width, rows=rows):
                        hidden = torch.randn((rows, width), device=device, dtype=dtype)
                        shared = torch.randn_like(hidden)
                        routed = torch.randn_like(hidden)
                        gate = torch.randn_like(hidden)
                        expected_gate = torch.sigmoid(
                            (hidden.float() * weight.float()).sum(-1)
                        )[:, None]
                        expected_add = torch.addcmul(
                            routed.float(), expected_gate, shared.float()
                        ).to(dtype)
                        with patch.object(
                            triton.runtime.driver.active.utils,
                            "load_binary",
                            side_effect=AssertionError(
                                "CP gate loaded a new CUDA module"
                            ),
                        ):
                            gated = fused_gate_sigmoid_mul(hidden, weight, shared)
                            fused_gate_sigmoid_mul_add(hidden, weight, shared, routed)
                            output = fused_sigmoid_mul(hidden, gate, inplace=False)
                        torch.testing.assert_close(
                            gated,
                            (expected_gate * shared.float()).to(dtype),
                            atol=0.008,
                            rtol=0.008,
                        )
                        torch.testing.assert_close(
                            routed, expected_add, atol=0.008, rtol=0.008
                        )
                        torch.testing.assert_close(
                            output,
                            (hidden.float() * torch.sigmoid(gate.float())).to(dtype),
                            atol=0.008,
                            rtol=0.008,
                        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_small_unquantized_projections_after_startup_warmup(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.model_executor.model_runner_components.load_model_utils import (
            warmup_cp_unquantized_linears,
        )
        from sglang.test.test_utils import published_topology

        with published_topology(), torch.no_grad(), torch.device("cuda"):
            projections = torch.nn.ModuleList(
                [
                    ReplicatedLinear(
                        4096, width, bias=False, params_dtype=torch.bfloat16
                    )
                    for width in (128, 512)
                ]
                + [torch.nn.Linear(4096, 1, bias=False, dtype=torch.bfloat16)]
            )
            for module in projections:
                module.weight.normal_(std=0.01)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                warmup_cp_unquantized_linears(projections)
            for module in projections:
                for rows in (1, 17, 64):
                    with self.subTest(width=module.weight.shape[0], rows=rows):
                        x = torch.randn((rows, 4096), dtype=torch.bfloat16)
                        expected = x.cpu().float() @ module.weight.cpu().float().T
                        stream.wait_stream(torch.cuda.current_stream())
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            output = module(x)
                            if isinstance(output, tuple):
                                output = output[0]
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            output.cpu().float(), expected, atol=0.015, rtol=0.01
                        )

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two CUDA GPUs")
    def test_flashqla_contexts_survive_varying_tail_lengths(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(_flashqla_context_worker, args=(2, port), nprocs=2, join=True)

    def test_parameter_gating_and_startup_phase(self):
        from unittest.mock import Mock

        from sglang.srt.layers.cp.utils import is_sharded_cp_enabled
        from sglang.srt.model_executor.runner.base_runner import BaseRunner

        for cp_size, encoder, role, expected in (
            (4, False, "prefill", True),
            (1, False, "prefill", False),
            (4, True, "prefill", False),
            (4, False, "decode", False),
        ):
            args = SimpleNamespace(
                attn_cp_size=cp_size,
                linear_attn_prefill_backend="flashqla",
                disaggregation_mode=role,
                encoder_only=encoder,
            )
            self.assertEqual(is_sharded_cp_enabled(args), expected)
            mr = SimpleNamespace(
                device="cuda",
                attn_backend=Mock(),
                token_to_kv_pool_allocator=None,
                model=SimpleNamespace(),
            )
            runner = SimpleNamespace(
                model_runner=mr,
                _pre_initialize_flashinfer_allreduce_workspace=Mock(),
                _pre_initialize_fi_a2a_workspace=Mock(),
                _pre_initialize_pcie_ipc_workspace=Mock(),
            )
            with (
                patch(
                    "sglang.srt.layers.cp.utils.is_sharded_cp_enabled",
                    return_value=expected,
                ),
                patch(
                    "sglang.srt.model_executor.runner.base_runner.should_run_flashinfer_autotune",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.model_executor.runner.base_runner.deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM",
                    False,
                ),
            ):
                BaseRunner.warmup(runner)
                BaseRunner.warmup(runner)
            if expected:
                mr.attn_backend.warmup.assert_called_once_with(mr)
            else:
                mr.attn_backend.warmup.assert_not_called()

    def test_gdn_context_warmup_uses_per_sequence_limit(self):
        from unittest.mock import Mock

        from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

        backend = GDNAttnBackend.__new__(GDNAttnBackend)
        kernel = Mock()
        backend.req_to_token_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(cp_size=1)
        )
        backend.kernel_dispatcher = SimpleNamespace(
            use_context_parallel=True, extend_kernel=kernel
        )
        for chunk, dynamic, expected in (
            (16384, False, 16384),
            (16384, True, 131072),
            (-1, False, 131072),
        ):
            runner = SimpleNamespace(
                device="cuda",
                server_args=SimpleNamespace(
                    chunked_prefill_size=chunk, enable_dynamic_chunking=dynamic
                ),
                model_config=SimpleNamespace(
                    context_len=131072,
                    hf_text_config=SimpleNamespace(linear_num_value_heads=64),
                ),
            )
            with patch(
                "sglang.srt.layers.attention.linear.gdn_backend.get_schedule",
                return_value=runner.server_args,
            ):
                backend.warmup(runner)
            kernel.warmup_cp_contexts.assert_called_with(
                device=torch.device("cuda"), max_tokens=expected, value_heads=64
            )
        kernel.reset_mock()
        backend.kernel_dispatcher.use_context_parallel = False
        backend.warmup(runner)
        kernel.warmup_cp_contexts.assert_not_called()


class CPAttentionTestRunner:
    """Exercise production exchange and existing backend kernels in GPU tests."""

    def __init__(self, *, q_heads, **kwargs):
        import torch

        from sglang.srt.layers.cp.utils import RingKVExchange

        kwargs.setdefault("dtype", torch.bfloat16)
        kwargs.setdefault("overlap", True)
        self.exchange = RingKVExchange(**kwargs)
        self.exchange.buffers.zero_()
        self.rank, self.world = self.exchange.rank, self.exchange.world
        if torch.cuda.get_device_capability(kwargs["device"])[0] == 9:
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )

            self.backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
        else:
            from sglang.srt.layers.attention.trtllm_mha_backend import (
                TRTLLMHAAttnBackend,
                get_trtllm_workspace,
                validate_cp_attention_support,
            )

            validate_cp_attention_support(
                head_dim=kwargs["head_dim"],
                q_heads=q_heads,
                kv_heads=kwargs["kv_heads"],
                dtype=kwargs["dtype"],
            )
            self.backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
            self.backend.workspace_buffer = get_trtllm_workspace(kwargs["device"])

    def forward(self, **kwargs):
        self.backend.cp_exchange = self.exchange
        return self.backend._forward_cp_attention(**kwargs)


def make_cp_test_segments(**kwargs):
    import torch

    from sglang.srt.layers.cp.utils import make_cp_attention_segments

    kwargs["packed_kv"] = torch.cuda.get_device_capability(kwargs["device"])[0] == 10
    return make_cp_attention_segments(**kwargs)


def create_cp_test_backend(runner):
    import torch

    if torch.cuda.get_device_capability(runner.device)[0] == 9:
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        return FlashAttentionBackend(runner)
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

    return TRTLLMHAAttnBackend(runner)


class TestCPTokenCopy(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    @torch.inference_mode()
    def test_reorder_residual_norm(self):
        from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm

        for norm_type in (RMSNorm, GemmaRMSNorm):
            dtypes = (
                (torch.float16, torch.bfloat16)
                if norm_type is GemmaRMSNorm
                else (torch.float16, torch.bfloat16, torch.float32)
            )
            for dtype in dtypes:
                for width in (129, 4096):
                    norm = norm_type(width).to(device="cuda", dtype=dtype)
                    norm.weight.copy_(torch.randn_like(norm.weight) * 0.1)
                    for stride_padding in (0, 17):
                        x = torch.randn(
                            7, width + stride_padding, device="cuda", dtype=dtype
                        )[:, :width]
                        residual = torch.randn_like(x)
                        saved_x, saved_residual = x.clone(), residual.clone()
                        for rows in ([6, 0, 3, 3], []):
                            indices = torch.tensor(
                                rows, device="cuda", dtype=torch.int64
                            )
                            selected = (
                                x.index_select(0, indices),
                                residual.index_select(0, indices),
                            )
                            # The layer boundary skips norm on empty local shards.
                            expected = norm(*selected) if rows else selected
                            actual = norm(x, residual, row_indices=indices)
                            torch.testing.assert_close(
                                actual[1], expected[1], rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                actual[0],
                                expected[0],
                                rtol=(
                                    8e-3
                                    if dtype == torch.bfloat16
                                    else 1e-3
                                    if dtype == torch.float16
                                    else 1e-6
                                ),
                                atol=1e-6,
                            )
                        torch.testing.assert_close(x, saved_x, rtol=0, atol=0)
                        torch.testing.assert_close(
                            residual, saved_residual, rtol=0, atol=0
                        )

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_copy_rows(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.int64):
            for features in (1, 17, 129, 4096):
                storage = (torch.arange(7 * 2 * features, device="cuda") % 29).to(dtype)
                strided = storage.view(7, 2 * features)[:, :features]
                for value in (strided, strided.contiguous()):
                    indices = torch.tensor([6, 0, -1, 3, 3, -1], device="cuda")
                    expected = value[indices.clamp_min(0)].clone()
                    expected[indices < 0] = 0
                    actual = copy_token_rows(value, indices)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        copy_token_rows(value, indices[::2]),
                        expected[::2],
                        rtol=0,
                        atol=0,
                    )
                    padded = copy_token_rows(value, rows=10)
                    torch.testing.assert_close(padded[:7], value, rtol=0, atol=0)
                    assert torch.count_nonzero(padded[7:]).item() == 0
                    assert copy_token_rows(value, rows=0).shape == (0, features)
                value = storage.view(7, 2, features).transpose(1, 2)
                torch.testing.assert_close(
                    copy_token_rows(value), value, rtol=0, atol=0
                )
            empty = torch.empty((0, 17), device="cuda", dtype=dtype)
            torch.testing.assert_close(
                copy_token_rows(empty, rows=3), empty.new_zeros(3, 17), rtol=0, atol=0
            )
        # FP8 values are copied without dequantization or a BF16 staging buffer.
        value = torch.randn(7, 17, device="cuda").to(torch.float8_e4m3fn)
        indices = torch.tensor([3, -1, 6], device="cuda")
        actual = copy_token_rows(value, indices).float()
        expected = value.float()[[3, 0, 6]]
        expected[1] = 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _cp_shard_inputs(
    seq_lens, query_lens, rank, dtype, *, heads=32, kv_heads=2, dim=256
):
    torch.manual_seed(20261015)
    layout = make_cp_attention_layout(
        seq_lens=seq_lens, query_lens=query_lens, world_size=4
    )
    source_dtype = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    keys = [
        torch.randn(s, kv_heads, dim, device="cuda", dtype=source_dtype)
        for s in seq_lens
    ]
    values = [torch.randn_like(k) for k in keys]
    queries = [
        torch.randn(n, heads, dim, device="cuda", dtype=source_dtype)
        for n in query_lens
    ]
    queries, keys, values = (
        [x.to(dtype) for x in xs] for xs in (queries, keys, values)
    )
    q = (
        torch.cat(queries)
        .index_select(
            0, torch.tensor(layout.query_indices[rank], device="cuda", dtype=torch.long)
        )
        .contiguous()
    )
    shards = []
    for owner in range(4):
        kv = torch.zeros(
            2, max(1, layout.capacity), kv_heads, dim, device="cuda", dtype=dtype
        )
        for i, tensors in enumerate((keys, values)):
            packed = torch.cat([x[owner::4] for x in tensors])
            kv[i, : packed.shape[0]] = packed
        shards.append(kv)
    plans = make_cp_attention_segments(layout=layout, rank=rank, device=q.device)
    return layout, q, shards, plans, queries, keys, values


def _cp_shard_attention(q, shards, plans, *, k_scale=None, v_scale=None):
    packed = q.index_select(0, plans[0].query_indices)
    output = torch.zeros_like(packed, dtype=torch.float32)
    lse = torch.full(packed.shape[:2], -torch.inf, device=q.device)
    backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
    for kv, plan in zip(shards, plans):
        shape = (plan.cu_q.numel() - 1, kv.shape[2])
        part, plse = backend._forward_cp_shard(
            q=packed,
            kv=kv,
            segments=plan,
            scale=q.shape[-1] ** -0.5,
            k_descale=k_scale.expand(shape) if k_scale is not None else None,
            v_descale=v_scale.expand(shape) if v_scale is not None else None,
        )
        merge_state(output, lse, part, plse, output, lse)
    restore = plans[0].restore_indices
    dtype = torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
    return output.to(dtype).index_select(0, restore), lse.index_select(0, restore)


def _cp_attention_reference(
    queries, keys, values, seq_lens, query_lens, *, fp32, k_scale=None, v_scale=None
):
    outputs, lses = [], []
    for q, k, v, size, n in zip(queries, keys, values, seq_lens, query_lens):
        if not n:
            continue
        if fp32:
            repeat = q.shape[1] // k.shape[1]
            logits = (
                q.float().transpose(0, 1)
                @ k.float().repeat_interleave(repeat, 1).permute(1, 2, 0)
            ) * q.shape[-1] ** -0.5
            mask = (
                torch.arange(size, device=q.device)[None, :]
                > torch.arange(size - n, size, device=q.device)[:, None]
            )
            logits.masked_fill_(mask, -torch.inf)
            out = (
                logits.softmax(-1)
                @ v.float().repeat_interleave(repeat, 1).transpose(0, 1)
            ).transpose(0, 1)
            lse = logits.logsumexp(-1).T
        else:
            out, lse, *_ = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.tensor([0, n], device=q.device, dtype=torch.int32),
                cu_seqlens_k=torch.tensor(
                    [0, size], device=q.device, dtype=torch.int32
                ),
                max_seqlen_q=n,
                max_seqlen_k=size,
                k_descale=(
                    k_scale.expand(1, k.shape[1]) if k_scale is not None else None
                ),
                v_descale=(
                    v_scale.expand(1, k.shape[1]) if v_scale is not None else None
                ),
                causal=True,
                num_splits=0,
                return_softmax_lse=True,
            )
            lse = lse.T
        outputs.append(out)
        lses.append(lse)
    if not outputs:
        return queries[0].float(), queries[0][:, :, 0].float()
    return torch.cat(outputs), torch.cat(lses)


def _check_cp_attention(actual, expected):
    out, lse = actual
    ref, ref_lse = expected
    if not out.numel():
        assert out.shape == ref.shape
        return {"relative_rms": 0, "lse_max_abs": 0}
    error = out.float() - ref.float()
    rms = ref.float().square().mean().sqrt().clamp_min(1e-8)
    relative = (error.square().mean().sqrt() / rms).item()
    lse_error = (lse - ref_lse).abs().max().item()
    assert relative < 0.005 and lse_error < 2e-4, (relative, lse_error)
    return {"relative_rms": relative, "lse_max_abs": lse_error}


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "Requires Hopper",
)
class TestContextParallelFA3(unittest.TestCase):
    @torch.inference_mode()
    def test_causal_boundaries_and_query_permutation(self):
        for dtype in (torch.bfloat16, torch.float16):
            for seq_lens, query_lens in [
                ((0,), (0,)),
                ((1,), (1,)),
                ((7,), (7,)),
                ((129, 33, 0), (65, 1, 0)),
                ((1025, 513), (257, 65)),
            ]:
                for rank in range(4):
                    with self.subTest(
                        dtype=dtype, lengths=seq_lens, query_lens=query_lens, rank=rank
                    ):
                        layout, q, shards, plans, qs, ks, vs = _cp_shard_inputs(
                            seq_lens,
                            query_lens,
                            rank,
                            dtype,
                            heads=4,
                            kv_heads=1,
                            dim=64,
                        )
                        ref, ref_lse = _cp_attention_reference(
                            qs, ks, vs, seq_lens, query_lens, fp32=True
                        )
                        rows = list(layout.query_indices[rank])
                        _check_cp_attention(
                            _cp_shard_attention(q, shards, plans),
                            (ref[rows], ref_lse[rows]),
                        )

    @torch.inference_mode()
    def test_qwen_long_prefix_and_uneven_tail(self):
        for seq_lens, query_lens in [
            ((16384,), (16384,)),
            ((81920,), (16384,)),
            ((82346,), (426,)),
        ]:
            for rank in (0, 3):
                layout, q, shards, plans, qs, ks, vs = _cp_shard_inputs(
                    seq_lens, query_lens, rank, torch.bfloat16
                )
                ref, ref_lse = _cp_attention_reference(
                    qs, ks, vs, seq_lens, query_lens, fp32=False
                )
                rows = list(layout.query_indices[rank])
                _check_cp_attention(
                    _cp_shard_attention(q, shards, plans),
                    (ref[rows], ref_lse[rows]),
                )

    @torch.inference_mode()
    def test_fp8_matches_tp_dtype_and_descales(self):
        # The _cp_attention_reference uses the same quantized Q/K/V and scales as TP FA3.
        # This isolates ring decomposition from input quantization error.
        reports = []
        for seq_lens, query_lens, heads, kv_heads, dim in [
            ((1,), (1,), 4, 1, 64),
            ((129, 33, 0), (65, 1, 0), 8, 2, 128),
            ((16384,), (16384,), 32, 2, 256),
            ((81920,), (16384,), 32, 2, 256),
            ((82346,), (426,), 32, 2, 256),
        ]:
            for rank in (0, 3):
                for scales in (None, (0.75, 1.25)):
                    kwargs = (
                        {}
                        if scales is None
                        else dict(
                            k_scale=torch.tensor(scales[0], device="cuda"),
                            v_scale=torch.tensor(scales[1], device="cuda"),
                        )
                    )
                    layout, q, shards, plans, qs, ks, vs = _cp_shard_inputs(
                        seq_lens,
                        query_lens,
                        rank,
                        torch.float8_e4m3fn,
                        heads=heads,
                        kv_heads=kv_heads,
                        dim=dim,
                    )
                    ref, ref_lse = _cp_attention_reference(
                        qs, ks, vs, seq_lens, query_lens, fp32=False, **kwargs
                    )
                    actual = _cp_shard_attention(q, shards, plans, **kwargs)
                    assert actual[0].dtype == torch.bfloat16
                    assert actual[1].dtype == torch.float32
                    rows = list(layout.query_indices[rank])
                    # FP8 FA3 also rounds softmax probabilities before P @ V.
                    # Partitioning changes that rounding. Compare both paths
                    # to BF16 FA3 on exactly the same quantized inputs, rather
                    # than applying a BF16 tolerance to FP8 arithmetic.
                    wide_q = [x.to(torch.bfloat16) for x in qs]
                    wide_k = [
                        x.to(torch.bfloat16) * (scales[0] if scales else 1) for x in ks
                    ]
                    wide_v = [
                        x.to(torch.bfloat16) * (scales[1] if scales else 1) for x in vs
                    ]
                    high, _ = _cp_attention_reference(
                        wide_q, wide_k, wide_v, seq_lens, query_lens, fp32=False
                    )
                    high, tp = high[rows].float(), ref[rows].float()
                    if not high.numel():
                        metrics = _check_cp_attention(
                            actual, (ref[rows], ref_lse[rows])
                        )
                    else:
                        denominator = high.square().mean().clamp_min(1e-16)
                        cp_error = (
                            ((actual[0].float() - high).square().mean() / denominator)
                            .sqrt()
                            .item()
                        )
                        tp_error = (
                            ((tp - high).square().mean() / denominator).sqrt().item()
                        )
                        difference = (
                            ((actual[0].float() - tp).square().mean() / denominator)
                            .sqrt()
                            .item()
                        )
                        lse_error = (actual[1] - ref_lse[rows]).abs().max().item()
                        # The 0.003 floor covers BF16 reference/output rounding;
                        # otherwise CP must stay within TP's FP8 error budget.
                        assert cp_error <= 1.25 * tp_error + 0.003, (cp_error, tp_error)
                        assert (
                            cp_error < 0.04 and difference < 0.05 and lse_error < 2e-4
                        )
                        metrics = dict(
                            cp_relative_rms=cp_error,
                            tp_relative_rms=tp_error,
                            cp_tp_relative_rms=difference,
                            lse_max_abs=lse_error,
                        )
                    reports.append(
                        dict(seq_lens=seq_lens, rank=rank, scales=scales, **metrics)
                    )
        print(json.dumps({"fp8_tp_reference": reports}), flush=True)


if __name__ == "__main__":
    unittest.main()
