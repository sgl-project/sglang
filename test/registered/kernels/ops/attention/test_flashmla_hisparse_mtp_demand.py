from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
from sgl_kernel.flashmla_hisparse_demand import HiSparseDemandInputs

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _sm90_with_cuda_124() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    cuda_version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
    return torch.cuda.get_device_capability() == (9, 0) and cuda_version >= (12, 4)


pytestmark = pytest.mark.skipif(
    not _sm90_with_cuda_124(), reason="requires SM90 and CUDA 12.4+"
)

TOPK = 2048
HOST_ROWS = 8192
CACHE_ROWS = 4096
VERIFY_ROWS = 4
READY = 2


def _pack_v32_rows(rows: torch.Tensor) -> torch.Tensor:
    """Pack BF16 [row, 1, 576] values into FlashMLA's 656-byte V32 rows."""
    row_count = rows.shape[0]
    assert row_count % 64 == 0
    packed = torch.empty(
        (row_count, 1, 656), dtype=torch.float8_e4m3fn, device=rows.device
    )
    nope = packed[..., :512]
    scales = packed[..., 512:528].view(torch.float32)
    packed[..., 528:].view(torch.bfloat16).copy_(rows[..., 512:])
    for tile_idx in range(4):
        tile = rows[..., tile_idx * 128 : (tile_idx + 1) * 128].float()
        scale = tile.abs().amax(dim=-1) / 448.0
        scale = torch.pow(2, scale.clamp_min(1e-4).log2().ceil())
        scales[..., tile_idx].copy_(scale)
        nope[..., tile_idx * 128 : (tile_idx + 1) * 128].copy_(
            (tile / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        )
    return packed.view(row_count // 64, 64, 1, 656)


def _assert_bytes_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(
        actual.contiguous().reshape(-1).view(torch.uint8),
        expected.contiguous().reshape(-1).view(torch.uint8),
    )


@torch.inference_mode()
def _make_demand_fixture(batch_size, cache_rows, host_stride, count_sources=False):
    CACHE_ROWS = cache_rows
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260821)
    committed_len = 131068

    logical_rows = torch.randn(
        (HOST_ROWS, 1, 576),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    packed_rows = _pack_v32_rows(logical_rows)
    packed_flat = packed_rows.view(HOST_ROWS, 1, 656)
    host_kv = torch.empty(
        (*packed_rows.shape[:-1], host_stride),
        dtype=packed_rows.dtype,
        device="cpu",
        pin_memory=True,
    )
    host_kv.view(torch.uint8).fill_(0xA5)
    host_kv[..., :656].copy_(packed_rows)

    history_positions = list(range(TOPK - VERIFY_ROWS - 1)) + [committed_len - 1]
    history_host_rows = torch.randperm(
        HOST_ROWS - VERIFY_ROWS, device=device, generator=generator
    )[: TOPK - VERIFY_ROWS].to(torch.int32)
    overlay_positions = list(range(committed_len, committed_len + VERIFY_ROWS))
    logical_indices = torch.tensor(
        [history_positions + overlay_positions] * VERIFY_ROWS,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(1)
    for verify_row in range(VERIFY_ROWS):
        logical_indices[verify_row, 0, TOPK - VERIFY_ROWS + verify_row + 1 :] = -1

    physical_indices = torch.full_like(logical_indices, -1)
    for source_ordinal, logical_position in enumerate(history_positions):
        physical_indices[logical_indices == logical_position] = history_host_rows[
            source_ordinal
        ]
    for offset, logical_position in enumerate(overlay_positions):
        physical_indices[logical_indices == logical_position] = (
            HOST_ROWS - VERIFY_ROWS + offset
        )
    logical_indices = logical_indices.repeat(batch_size, 1, 1)
    physical_indices = physical_indices.repeat(batch_size, 1, 1)
    query_rows = batch_size * VERIFY_ROWS

    rows_per_request = ((CACHE_ROWS + 6 + 63) // 64) * 64
    hot_device_kv = torch.zeros(
        (batch_size * rows_per_request // 64, 64, 1, 656),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    hot_flat = hot_device_kv.view(-1, 1, 656)
    device_locs = torch.zeros(
        (batch_size + 1, CACHE_ROWS + 6), dtype=torch.int64, device=device
    )
    local_rows = torch.arange(CACHE_ROWS + 6, dtype=torch.int64, device=device)
    for req_slot in range(1, batch_size + 1):
        device_locs[req_slot] = (req_slot - 1) * rows_per_request + local_rows
    hot_flat[device_locs[1:, CACHE_ROWS + 2 :]] = packed_flat[
        HOST_ROWS - VERIFY_ROWS : HOST_ROWS
    ].unsqueeze(0)

    host_locs = torch.full((query_rows, TOPK), -1, dtype=torch.int32, device=device)
    host_locs[:, : TOPK - VERIFY_ROWS] = history_host_rows.to(torch.int32)
    cache_tags = torch.zeros(
        (batch_size + 1, CACHE_ROWS), dtype=torch.int64, device=device
    )
    source_counts = (
        torch.zeros((batch_size + 1, 8), dtype=torch.int64, device=device)
        if count_sources
        else None
    )
    decode_calls = torch.tensor(
        [0] + [2] * batch_size, dtype=torch.int32, device=device
    )
    req_pool_indices = torch.arange(
        1, batch_size + 1, dtype=torch.int64, device=device
    ).repeat_interleave(VERIFY_ROWS)

    q = torch.randn(
        (query_rows, 1, 64, 576),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    kernel_seq_lens = torch.full((query_rows,), TOPK, dtype=torch.int32, device=device)
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        kernel_seq_lens,
        num_q_tokens_per_head_k=64,
        num_heads_k=1,
        num_heads_q=64,
        is_fp8_kvcache=True,
        topk=TOPK,
    )
    common = dict(
        q=q,
        block_table=torch.empty((query_rows, 0), dtype=torch.int32, device=device),
        cache_seqlens=kernel_seq_lens,
        head_dim_v=512,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=1.0 / math.sqrt(576),
        is_fp8_kvcache=True,
    )
    base_out, base_lse = flash_mla_with_kvcache(
        k_cache=packed_rows, indices=physical_indices, **common
    )

    host_kv[..., :656].copy_(packed_rows)
    hot_flat.zero_()
    hot_flat[device_locs[1:, CACHE_ROWS + 2 :]] = packed_flat[
        HOST_ROWS - VERIFY_ROWS : HOST_ROWS
    ].unsqueeze(0)
    cache_tags.zero_()
    decode_calls[1:] = 2
    effective_seq_lens = torch.arange(
        committed_len + 1,
        committed_len + VERIFY_ROWS + 1,
        dtype=torch.int32,
        device=device,
    ).repeat(batch_size)
    demand_kwargs = dict(
        k_cache=hot_device_kv,
        indices=logical_indices,
        hisparse_demand=HiSparseDemandInputs(
            host_kv=host_kv,
            host_locs=host_locs,
            device_locs=device_locs,
            cache_tags=cache_tags,
            decode_calls=decode_calls,
            num_real_reqs=torch.tensor([query_rows], dtype=torch.int32, device=device),
            req_pool_indices=req_pool_indices,
            seq_lens=effective_seq_lens,
            mtp_committed_lens=torch.full(
                (query_rows,), committed_len, dtype=torch.int32, device=device
            ),
            cache_rows=CACHE_ROWS,
            source_counts=source_counts,
        ),
        **common,
    )
    return dict(
        locals(),
        TOPK=TOPK,
        HOST_ROWS=HOST_ROWS,
        flash_mla_with_kvcache=flash_mla_with_kvcache,
        _assert_bytes_equal=_assert_bytes_equal,
        _pack_v32_rows=_pack_v32_rows,
    )


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("count_sources", [False, True])
@pytest.mark.parametrize("cache_rows", [4096, 8192])
@pytest.mark.parametrize("host_stride", [656, 768])
@torch.inference_mode()
def test_mtp_direct_demand_matches_hbm_and_promotes_hits(
    batch_size: int, count_sources: bool, cache_rows: int, host_stride: int
):
    v = _make_demand_fixture(batch_size, cache_rows, host_stride, count_sources)
    CACHE_ROWS = v["CACHE_ROWS"]
    device = v["device"]
    base_out = v["base_out"]
    base_lse = v["base_lse"]
    cache_tags = v["cache_tags"]
    source_counts = v["source_counts"]
    history_host_rows = v["history_host_rows"]
    hot_flat = v["hot_flat"]
    device_locs = v["device_locs"]
    packed_flat = v["packed_flat"]
    packed_rows = v["packed_rows"]
    host_kv = v["host_kv"]
    decode_calls = v["decode_calls"]
    demand_kwargs = v["demand_kwargs"]
    demand_out, demand_lse = flash_mla_with_kvcache(**demand_kwargs)
    if count_sources:
        # Each valid TopK occurrence is resolved once, including ten causal
        # overlay occurrences across the four verify queries, per request.
        counts = source_counts.cpu()
        assert counts[0].sum() == 0
        assert torch.all(counts[1:, 1] == 10)
        assert torch.all(counts[1:, 1:6].sum(dim=1) == 4 * (TOPK - VERIFY_ROWS) + 10)
        assert torch.all(counts[1:, 3] > 0)
        assert torch.all(counts[1:, 6] == TOPK - VERIFY_ROWS)
        assert torch.all(counts[1:, 7] == 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out, graph_lse = flash_mla_with_kvcache(**demand_kwargs)
        graph.replay()
        _assert_bytes_equal(graph_out, base_out)
        _assert_bytes_equal(graph_lse, base_lse)
        assert torch.all(source_counts[1:, 7] == 2)

    ready_mask = (cache_tags & 0x3) == READY
    ready_rows = (cache_tags[ready_mask] >> 26).to(torch.int64)
    history_host_row_mask = torch.zeros(HOST_ROWS, dtype=torch.bool, device=device)
    history_host_row_mask[history_host_rows] = True
    unexpected = ready_rows[
        (ready_rows < 0)
        | (ready_rows >= HOST_ROWS)
        | ~history_host_row_mask[ready_rows.clamp(0, HOST_ROWS - 1)]
    ]
    # Two independent candidates avoid most direct-map collisions while
    # keeping every lookup bounded. The deterministic fixture retains at
    # least 94% of the historical TopK rows after the cold fill.
    min_resident_rows = math.floor((TOPK - VERIFY_ROWS) * 0.94)
    assert ready_mask.sum().item() >= batch_size * min_resident_rows
    assert torch.all(ready_mask[1:].sum(dim=1) >= min_resident_rows)
    assert unexpected.numel() == 0
    assert not torch.any(ready_mask[0])
    resident_row_sets = []
    for req_slot in range(1, batch_size + 1):
        slots = torch.nonzero(ready_mask[req_slot], as_tuple=False).flatten()
        tagged_host_rows = (cache_tags[req_slot, slots] >> 26).to(torch.int64)
        resident_row_sets.append(set(tagged_host_rows.cpu().tolist()))
        cached_rows = hot_flat[device_locs[req_slot, slots]]
        expected_rows = packed_flat[tagged_host_rows]
        _assert_bytes_equal(cached_rows, expected_rows)

    _assert_bytes_equal(demand_out, base_out)
    _assert_bytes_equal(demand_lse, base_lse)
    assert not torch.any((cache_tags & 0x3) == 1)

    # Replaying the same generation must consume resident rows without Host
    # fallback or tag mutation. Duplicate fills can occupy both candidate
    # slots, so corrupt only rows confirmed resident for every request and
    # leave legitimate collision fallbacks intact.
    same_generation_tags = cache_tags.clone()
    common_resident_rows = sorted(set.intersection(*resident_row_sets))
    assert len(common_resident_rows) >= TOPK // 2
    host_kv.view(HOST_ROWS, host_stride).view(torch.uint8).index_fill_(
        0,
        torch.tensor(common_resident_rows, dtype=torch.int64),
        0,
    )
    hit_out, hit_lse = flash_mla_with_kvcache(**demand_kwargs)
    _assert_bytes_equal(hit_out, base_out)
    _assert_bytes_equal(hit_lse, base_lse)
    assert torch.equal(cache_tags[ready_mask], same_generation_tags[ready_mask])
    assert ((cache_tags & 0x3) == READY).sum() >= ready_mask.sum()
    assert not torch.any((cache_tags & 0x3) == 1)

    # Across decode calls, a set-associative cache may legitimately fall back to the
    # authoritative Host row after a collision. The output must remain exact.
    host_kv[..., :656].copy_(packed_rows)
    decode_calls[1:] = 3
    retained_out, retained_lse = flash_mla_with_kvcache(**demand_kwargs)
    _assert_bytes_equal(retained_out, base_out)
    _assert_bytes_equal(retained_lse, base_lse)

    decode_calls[1:] = 4
    promoted_out, promoted_lse = flash_mla_with_kvcache(**demand_kwargs)
    _assert_bytes_equal(promoted_out, base_out)
    _assert_bytes_equal(promoted_lse, base_lse)

    # The 24-bit generation wraps lazily. Existing rows may be promoted or
    # replaced, but wrapped epochs must never expose a stale or partial row.
    decode_calls[1:] = (1 << 24) + 2
    wrapped_out, wrapped_lse = flash_mla_with_kvcache(**demand_kwargs)
    _assert_bytes_equal(wrapped_out, base_out)
    _assert_bytes_equal(wrapped_lse, base_lse)
    assert not torch.any((cache_tags & 0x3) == 1)


def validate_plan(plan, adapter, logical):
    plans, hosts, tags = (
        plan.cpu().tolist(),
        adapter.host_locs.cpu().tolist(),
        adapter.cache_tags.cpu().tolist(),
    )
    reqs, limits = (
        adapter.req_pool_indices.cpu().tolist(),
        adapter.mtp_committed_lens.cpu().tolist(),
    )
    logical_cpu, seen = logical[:, 0].cpu().tolist(), {}
    for query, entries in enumerate(plans):
        for ordinal, s in enumerate(entries):
            if s < 0:
                assert s == -1
                continue
            assert s < adapter.cache_rows
            assert 0 <= logical_cpu[query][ordinal] < limits[query]
            key, row = (reqs[query], s), hosts[query][ordinal]
            assert seen.setdefault(key, row) == row, "contradictory slot assignment"
            tag = tags[reqs[query]][s]
            assert tag & 3 == 2 and tag >> 26 == row


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("cache_rows", [4096, 8192])
@pytest.mark.parametrize("host_stride", [656, 768])
@torch.inference_mode()
def test_group_plan(batch_size, cache_rows, host_stride):
    v = _make_demand_fixture(batch_size, cache_rows, host_stride)
    call, equal = v["flash_mla_with_kvcache"], v["_assert_bytes_equal"]
    common, logical = v["common"], v["logical_indices"]
    adapter = v["demand_kwargs"]["hisparse_demand"]
    plan = torch.full_like(adapter.host_locs, 123456)
    first = replace(adapter, group_slots=plan, group_role=1)
    packed = v["_pack_v32_rows"](-v["logical_rows"])
    host = torch.empty_like(adapter.host_kv, device="cpu", pin_memory=True)
    host.view(torch.uint8).fill_(0xA5)
    host[..., :656].copy_(packed)
    local_tags = torch.full_like(adapter.cache_tags, ((v["HOST_ROWS"] - 1) << 26) | 6)
    local_tags[0].zero_()
    second = replace(
        adapter, host_kv=host, cache_tags=local_tags, group_slots=plan, group_role=2
    )
    hot_first, hot_second = v["hot_device_kv"], torch.zeros_like(v["hot_device_kv"])
    # Arbitrary request-owned addresses, not base+slot arithmetic.
    for req in range(1, batch_size + 1):
        adapter.device_locs[req] = adapter.device_locs[req].flip(0)
    for hot, dense in ((hot_first, v["packed_rows"]), (hot_second, packed)):
        hot.zero_()
        hot.view(-1, 1, 656)[adapter.device_locs[1:, cache_rows + 2 :]] = dense.view(
            -1, 1, 656
        )[-4:].unsqueeze(0)
    references = (
        (v["base_out"], v["base_lse"]),
        call(k_cache=packed, indices=v["physical_indices"], **common),
    )

    def anchor():
        return call(k_cache=hot_first, indices=logical, hisparse_demand=first, **common)

    def follower():
        return call(
            k_cache=hot_second, indices=logical, hisparse_demand=second, **common
        )

    def check(outputs):
        for actual, ref in zip(outputs, references):
            for a, b in zip(actual, ref):
                equal(a, b)

    check((anchor(), follower()))
    validate_plan(plan, first, logical)
    # Every assigned slot now contains THIS layer's row, despite young old tags.
    validate_plan(plan, second, logical)
    assert not torch.equal(references[0][0], references[1][0])
    before = local_tags.clone()
    adapter.decode_calls[1:] = 3
    for a, b in zip(follower(), references[1]):
        equal(a, b)
    assert torch.equal(before, local_tags), "follower hits must not promote generations"

    # Corrupt only Host rows for which every selected occurrence is planned.
    assigned = set(adapter.host_locs[plan >= 0].cpu().tolist())
    host_only = set(
        adapter.host_locs[(plan < 0) & (adapter.host_locs >= 0)].cpu().tolist()
    )
    protected = sorted(assigned - host_only)
    assert len(protected) > 100
    host.view(-1, host_stride).view(torch.uint8)[torch.tensor(protected)] = 0
    for a, b in zip(follower(), references[1]):
        equal(a, b)
    host[..., :656].copy_(packed)

    query, pos = (plan >= 0).nonzero()[0].tolist()
    req, s = int(adapter.req_pool_indices[query]), int(plan[query, pos])
    saved = int(local_tags[req, s])
    for filling in ((saved & ~3) | 1, ((v["HOST_ROWS"] - 1) << 26) | 9):
        local_tags[req, s] = filling
        for a, b in zip(follower(), references[1]):
            equal(a, b)
        assert int(local_tags[req, s]) == filling, "never overwrite an in-flight fill"
    local_tags[req, s] = saved
    old_plan = plan.clone()
    before = local_tags.clone()
    for invalid in (-1, cache_rows):
        plan.fill_(invalid)
        for a, b in zip(follower(), references[1]):
            equal(a, b)
        assert torch.equal(before, local_tags), "Host fallback must not allocate"
    plan.copy_(old_plan)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = (anchor(), follower())
    for generation in (4, (1 << 24) + 1, (1 << 24) + 2):
        adapter.decode_calls[1:] = generation
        graph.replay()
        check(outputs)
        validate_plan(plan, first, logical)
        validate_plan(plan, second, logical)
    adapter.num_real_reqs.zero_()
    before_a, before_b = first.cache_tags.clone(), local_tags.clone()
    graph.replay()
    assert torch.all(plan == -1)
    assert torch.equal(first.cache_tags, before_a) and torch.equal(local_tags, before_b)
    adapter.num_real_reqs.fill_(logical.shape[0])
    first.cache_tags.zero_()
    local_tags.zero_()
    adapter.decode_calls[1:] = 2
    graph.replay()
    check(outputs)
    validate_plan(plan, first, logical)
    validate_plan(plan, second, logical)

    # Move to disjoint historical working sets, retaining each token's fixed
    # Host identity. Stamp follower READY ages young to exercise policy separation.
    count = v["TOPK"] - 4
    old_positions = logical[0, 0, :count].clone()
    old_hosts = adapter.host_locs[0, :count].clone()
    unused = torch.tensor(
        sorted(set(range(v["HOST_ROWS"] - 4)) - set(old_hosts.cpu().tolist())),
        device="cuda",
        dtype=torch.int32,
    )
    windows = [
        (old_positions, old_hosts),
        (torch.arange(4096, 4096 + count, device="cuda"), unused[:count]),
        (torch.arange(8192, 8192 + count, device="cuda"), unused[count : 2 * count]),
        (old_positions, old_hosts),
    ]
    changed_assignments = 0
    for step, (positions, rows) in enumerate(windows, 100):
        logical[:, 0, :count] = positions
        adapter.host_locs[:, :count] = rows
        v["physical_indices"][:, 0, :count] = rows
        adapter.decode_calls[1:] = step + 1
        ready = (local_tags & 3) == 2
        local_tags[ready] = (local_tags[ready] & ~(((1 << 24) - 1) << 2)) | (step << 2)
        before_rows = local_tags >> 26
        references = (
            call(k_cache=v["packed_rows"], indices=v["physical_indices"], **common),
            call(k_cache=packed, indices=v["physical_indices"], **common),
        )
        out_a = anchor()
        valid = plan >= 0
        current_rows = before_rows[
            adapter.req_pool_indices[:, None].expand_as(plan)[valid], plan[valid].long()
        ]
        changed_assignments += int((current_rows != adapter.host_locs[valid]).sum())
        check((out_a, follower()))
        validate_plan(plan, first, logical)
        validate_plan(plan, second, logical)
    assert changed_assignments > 0

    # Reuse identical Host row IDs with NEW bytes after request reset. Stale
    # row tags/data would now fail, unlike resetting with identical contents.
    packed_new = v["_pack_v32_rows"](v["logical_rows"] * 0.5)
    host[..., :656].copy_(packed_new)
    hot_second.view(-1, 1, 656)[adapter.device_locs[1:, cache_rows + 2 :]] = (
        packed_new.view(-1, 1, 656)[-4:].unsqueeze(0)
    )
    new_ref = call(k_cache=packed_new, indices=v["physical_indices"], **common)
    assert not torch.equal(references[1][0], new_ref[0])
    references = (references[0], new_ref)
    first.cache_tags.zero_()
    local_tags.zero_()
    adapter.decode_calls[1:] = 2
    graph.replay()
    check(outputs)
    validate_plan(plan, first, logical)
    validate_plan(plan, second, logical)


@torch.inference_mode()
def test_two_groups_graph_reuse():
    v = _make_demand_fixture(1, 4096, 768)
    call, equal = v["flash_mla_with_kvcache"], v["_assert_bytes_equal"]
    base = v["demand_kwargs"]["hisparse_demand"]
    logical = v["logical_indices"]
    pattern0 = (logical.clone(), base.host_locs.clone())
    pattern1 = (logical.clone(), base.host_locs.clone())
    count = v["TOPK"] - 4
    used = set(base.host_locs[0, :count].cpu().tolist())
    other = torch.tensor(
        sorted(set(range(v["HOST_ROWS"] - 4)) - used)[:count],
        dtype=torch.int32,
        device="cuda",
    )
    pattern1[0][:, 0, :count] = torch.arange(4096, 4096 + count, device="cuda")
    pattern1[1][:, :count] = other
    patterns = [pattern0, pattern1]
    plan = torch.full_like(base.host_locs, -1)
    layers = []
    for i, scale in enumerate((1.0, -1.0, 0.5, -0.5, 2.0, -2.0)):
        dense = v["_pack_v32_rows"](v["logical_rows"] * scale)
        host = torch.empty_like(base.host_kv, device="cpu", pin_memory=True)
        host.view(torch.uint8).fill_(0xA5)
        host[..., :656].copy_(dense)
        hot = torch.zeros_like(v["hot_device_kv"])
        hot.view(-1, 1, 656)[base.device_locs[1:, 4098:]] = dense.view(-1, 1, 656)[
            -4:
        ].unsqueeze(0)
        adapter = replace(
            base,
            host_kv=host,
            cache_tags=torch.zeros_like(base.cache_tags),
            group_slots=plan,
            group_role=1 if i % 3 == 0 else 2,
        )
        layers.append((adapter, hot, dense))

    def execute():
        outputs = []
        for group in range(2):
            logical.copy_(patterns[group][0])
            base.host_locs.copy_(patterns[group][1])
            for adapter, hot, _ in layers[group * 3 : group * 3 + 3]:
                outputs.append(
                    call(
                        k_cache=hot,
                        indices=logical,
                        hisparse_demand=adapter,
                        **v["common"],
                    )
                )
        return outputs

    execute()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = execute()
    for step in range(3):
        # Copy new choices into the existing tensors, preserving graph addresses.
        if step:
            a, b = patterns[0][0].clone(), patterns[0][1].clone()
            patterns[0][0].copy_(patterns[1][0])
            patterns[0][1].copy_(patterns[1][1])
            patterns[1][0].copy_(a)
            patterns[1][1].copy_(b)
        base.decode_calls[1:] = step + 4
        graph.replay()
        for i, (_, _, dense) in enumerate(layers):
            ids, hosts = patterns[i // 3]
            raw = ids[:, 0]
            hist = (raw >= 0) & (raw < base.mtp_committed_lens[:, None])
            offset = raw - base.mtp_committed_lens[:, None]
            physical = torch.where(
                hist,
                hosts,
                torch.where(
                    (offset >= 0) & (offset < 4), v["HOST_ROWS"] - 4 + offset, -1
                ),
            ).unsqueeze(1)
            ref = call(k_cache=dense, indices=physical, **v["common"])
            for actual, expected in zip(outputs[i], ref):
                equal(actual, expected)
