"""Native-bitwise, independent-reference and replay tests for DFlash SWA MLA."""

import math
import sys

import pytest
import torch

from sglang.kernels.ops.attention.dflash_swa_mla import (
    dflash_swa_mla_fwd,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-c", runner_config="2-gpu-gb300")

SCALE = 192**-0.5 * (1.0 + 0.1 * math.log(32.0)) ** 2
WINDOW = 4095
requires_gb300 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="qualified on GB300 (SM103)",
)


@pytest.fixture(autouse=True)
def _full_precision_reference():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _fixture(prefixes, heads=12, query_lengths=None, *, clip_prefix=True):
    torch.manual_seed(20817)
    batch = len(prefixes)
    query_lengths = (8,) * batch if query_lengths is None else query_lengths
    assert len(query_lengths) == batch and all(0 < n <= 8 for n in query_lengths)
    tokens = sum(query_lengths)
    q = torch.randn(tokens, heads, 576, device="cuda", dtype=torch.bfloat16) * 0.4
    k = torch.randn(tokens, 1, 576, device="cuda", dtype=torch.bfloat16) * 0.4
    v = k[..., :512].contiguous()
    slots = max(64, sum(prefixes) + 64)
    cache = torch.randn(slots, 1, 576, device="cuda", dtype=torch.bfloat16) * 0.4
    physical = torch.randperm(slots, device="cuda")
    rails = []
    lengths = [0]
    cursor = 0
    for length in prefixes:
        visible = min(length, WINDOW) if clip_prefix else length
        rails.append(physical[cursor + length - visible : cursor + length])
        lengths.append(lengths[-1] + visible)
        cursor += length
    indices = torch.cat(rails).contiguous()
    query_offsets = [0]
    for length in query_lengths:
        query_offsets.append(query_offsets[-1] + length)
    qo = torch.tensor(query_offsets, device="cuda", dtype=torch.int32)
    kv = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    output = torch.empty(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    return [q, k, v, output, cache, cache[..., :512], qo, kv, indices]


def _reference(args):
    q, k, v, _, kb, vb, qo, kv, indices = args
    outputs = []
    for batch in range(qo.numel() - 1):
        start, stop = qo[batch : batch + 2].tolist()
        left, right = kv[batch : batch + 2].tolist()
        slots = indices[left:right].long()
        keys = torch.cat((kb[slots, 0], k[start:stop, 0])).float()
        values = torch.cat((vb[slots, 0], v[start:stop, 0])).float()
        query = q[start:stop].float()
        scores = torch.einsum("qhd,kd->hqk", query, keys) * SCALE
        positions = (right - left) + torch.arange(stop - start, device=q.device)
        key_positions = torch.arange(keys.shape[0], device=q.device)
        legal = (key_positions[None] <= positions[:, None]) & (
            key_positions[None] >= positions[:, None] - WINDOW
        )
        probabilities = scores.masked_fill(~legal[None], -torch.inf).softmax(-1)
        outputs.append(torch.einsum("hqk,kd->qhd", probabilities, values))
    return torch.cat(outputs)


def _assert_reference(actual, expected):
    assert bool(torch.isfinite(actual).all())
    error = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-8)
    assert float(error) < 0.01
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.004)


def _native(args, page_size=64):
    # Import lazily: the native module detects CUDA capability at import time.
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

    native_args = list(args)
    native_args[3] = torch.empty_like(args[3], memory_format=torch.contiguous_format)
    extend_attention_fwd(
        *native_args,
        custom_mask=None,
        is_causal=True,
        mask_indptr=None,
        max_len_extend=8,
        k_scale=1.0,
        v_scale=1.0,
        sm_scale=SCALE,
        sliding_window_size=WINDOW,
        page_size=page_size,
    )
    return native_args[3]


def _assert_native_and_reference(args):
    expected = _reference(args)
    native = _native(args)
    # CSR can shrink during graph replay without resizing the allocations.
    active = expected.shape[0]
    actual = args[3][:active]
    native = native[:active]
    _assert_reference(native, expected)
    _assert_reference(actual, expected)
    assert torch.equal(actual, native), "value kernel differs from native BF16 output"


@requires_gb300
@pytest.mark.parametrize(
    "prefixes", [(0, 31), (4094, 4095), (4096, 4103), (8192, 32768)]
)
@pytest.mark.parametrize("num_splits", [2, 4, 8])
def test_causal_swa_reference_and_strided_cache(prefixes, num_splits):
    args = _fixture(prefixes)
    dflash_swa_mla_fwd(*args, sm_scale=SCALE, num_splits=num_splits, page_size=64)
    _assert_native_and_reference(args)


@requires_gb300
@pytest.mark.parametrize("num_splits", [2, 4, 8])
def test_ragged_query_lengths_and_split_counts(num_splits):
    # Exercise partial query rows, a prefix/current-block tile boundary and a
    # full SWA window together. Each value partition must write every live row.
    args = _fixture((0, 63, 4095), query_lengths=(1, 7, 8))
    dflash_swa_mla_fwd(*args, sm_scale=SCALE, num_splits=num_splits)
    _assert_native_and_reference(args)


@requires_gb300
@pytest.mark.parametrize("num_splits", [2, 4, 8])
def test_prefix_tile_boundaries_without_host_window_clipping(num_splits):
    # Exercise the primitive itself at the prefix tail/current-token boundary.
    # In particular, prefix4096 must mask out key0 for query0, while prefix4095
    # keeps key0. The production metadata normally clips prefixes to4095.
    args = _fixture((63, 64, 65, 4096), query_lengths=(1, 7, 8, 8), clip_prefix=False)
    dflash_swa_mla_fwd(*args, sm_scale=SCALE, num_splits=num_splits, page_size=64)
    _assert_native_and_reference(args)


@requires_gb300
@pytest.mark.parametrize("num_splits", [2, 4, 8])
def test_primitive_supports_full_head_count(num_splits):
    # The backend dispatch is deliberately narrow, while the primitive remains
    # usable by full-head numerical qualification harnesses.
    args = _fixture((65,), heads=96)
    dflash_swa_mla_fwd(*args, sm_scale=SCALE, num_splits=num_splits)
    _assert_native_and_reference(args)


@requires_gb300
@pytest.mark.parametrize("num_splits", [2, 4, 8])
def test_strided_query_and_output_with_storage_offsets(num_splits):
    args = _fixture((31, 4095), heads=24, query_lengths=(7, 8))
    tokens, heads, _ = args[0].shape
    q_storage = torch.full(
        (tokens * 2, heads * 2, 576),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_view = q_storage[1::2, ::2]
    q_view.copy_(args[0])
    out_storage = torch.full(
        (tokens * 2, heads * 2, 512),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    args[0] = q_view
    args[3] = out_storage[::2, 1::2]
    assert not args[0].is_contiguous() and not args[3].is_contiguous()
    dflash_swa_mla_fwd(*args, sm_scale=SCALE, num_splits=num_splits)
    _assert_native_and_reference(args)
    # Output padding is not an allowed destination, even for masked query rows.
    assert bool(torch.isnan(out_storage[1::2]).all())
    assert bool(torch.isnan(out_storage[::2, ::2]).all())


@requires_gb300
@pytest.mark.parametrize("num_splits", [2, 4, 8])
def test_future_rows_and_expired_window_keys_do_not_leak(num_splits):
    args = _fixture((4095,))
    args[0].zero_()  # Uniform legal weights make the positive control unambiguous.
    dflash_swa_mla_fwd(*args, sm_scale=SCALE, num_splits=num_splits)
    original = args[3].clone()
    # Last current-block row is future context for query rows0..6.
    args[1][7].add_(128)
    args[2][7].add_(128)
    dflash_swa_mla_fwd(*args, sm_scale=SCALE, num_splits=num_splits)
    assert torch.equal(original[:7], args[3][:7])
    assert not torch.equal(original[7], args[3][7])
    # The first seven cached rows have left query7's window, but remain legal
    # for earlier queries. Cache V aliases K, matching the absorbed MLA pool.
    previous = args[3].clone()
    expired = args[8][:7].long()
    args[4][expired] += 128
    dflash_swa_mla_fwd(*args, sm_scale=SCALE, num_splits=num_splits)
    assert torch.equal(previous[7], args[3][7])
    assert not torch.equal(previous[0], args[3][0])
    _assert_native_and_reference(args)


@requires_gb300
@pytest.mark.parametrize("num_splits", [2, 4, 8])
def test_graph_replay_updates_prefix_and_query_lengths(num_splits):
    args = _fixture((4095, 31), heads=24)
    for _ in range(3):
        dflash_swa_mla_fwd(*args, sm_scale=SCALE, num_splits=num_splits)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dflash_swa_mla_fwd(*args, sm_scale=SCALE, num_splits=num_splits)
    graph.replay()
    _assert_native_and_reference(args)
    # Reuse every allocation. Metadata is read from device memory on each
    # replay even though the graph is unchanged.
    args[7].copy_(torch.tensor([0, 0, 17], device="cuda", dtype=torch.int32))
    graph.replay()
    _assert_native_and_reference(args)
    # Query CSR changes are graph inputs too. Shrink to lengths1/7 without
    # reallocating tensors, and poison outputs so a stale row cannot pass.
    args[6].copy_(torch.tensor([0, 1, 8], device="cuda", dtype=torch.int32))
    args[3].fill_(float("nan"))
    graph.replay()
    _assert_native_and_reference(args)
    assert bool(torch.isnan(args[3][8:]).all())
    # A request can become empty without moving the other request's buffers.
    args[6].copy_(torch.tensor([0, 0, 7], device="cuda", dtype=torch.int32))
    args[3].fill_(float("nan"))
    graph.replay()
    _assert_native_and_reference(args)
    assert bool(torch.isnan(args[3][7:]).all())
    # Grow back to the original8/8 rows and prefixes in the same graph.
    args[6].copy_(torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32))
    args[7].copy_(torch.tensor([0, 4095, 4126], device="cuda", dtype=torch.int32))
    graph.replay()
    _assert_native_and_reference(args)


def test_unsupported_window_and_block_fail_before_launch():
    q = torch.empty(8, 12, 576, device="meta", dtype=torch.bfloat16)
    k = torch.empty(8, 1, 576, device="meta", dtype=torch.bfloat16)
    v = torch.empty(8, 1, 512, device="meta", dtype=torch.bfloat16)
    out = torch.empty(8, 12, 512, device="meta", dtype=torch.bfloat16)
    ptr = torch.empty(2, device="meta", dtype=torch.int32)
    indices = torch.empty(0, device="meta", dtype=torch.int64)
    args = (q, k, v, out, k, v, ptr, ptr, indices)
    with pytest.raises(ValueError, match="block8 and window_left4095"):
        dflash_swa_mla_fwd(*args, sm_scale=SCALE, sliding_window_size=4096)
    with pytest.raises(ValueError, match="block8 and window_left4095"):
        dflash_swa_mla_fwd(*args, sm_scale=SCALE, max_len_extend=9)
    with pytest.raises(ValueError, match="finite and positive"):
        dflash_swa_mla_fwd(*args, sm_scale=float("nan"))
    for num_splits in (1, 3, 16):
        with pytest.raises(ValueError, match="num_splits"):
            dflash_swa_mla_fwd(*args, sm_scale=SCALE, num_splits=num_splits)
    with pytest.raises(ValueError, match="same-device CUDA BF16"):
        dflash_swa_mla_fwd(*args, sm_scale=SCALE)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
