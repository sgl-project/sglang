"""Ordered planner and real-copy regressions for the opt-in gfx95 scan."""

import pytest
import torch

from sglang.kernels.ops.kvcache.hisparse import (
    copy_cache_planned_mla,
    load_cache_to_device_buffer_mla,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is None
    or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx95"),
    reason="Batched prefix requires gfx95 wave64.",
)


@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("req_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("pattern", ["Empty", "Full", "Mixed", "Newest", "Short"])
def test_ordered_plan_and_real_kv(seq_dtype, req_dtype, pattern):
    hot, top, length, rows, width = 4096, 2048, 8193, 4, 144
    generator = torch.Generator().manual_seed(4552)
    order = torch.randperm(hot, generator=generator).tolist()
    selected = torch.randperm(hot, generator=generator)[:top].tolist()
    tags = [-1] * hot
    if pattern == "Full":
        tags = list(range(hot))
    elif pattern in ("Mixed", "Newest"):
        tags = [i if i % 3 else -1 for i in range(hot)]
    if pattern == "Newest":
        selected[-1] = length - 1
    if pattern == "Short":
        length = 127
        selected = list(reversed(range(length))) + [-1] * (top - length)
        tags = list(range(hot))
    host = torch.arange(8194 * width, dtype=torch.int64).reshape(8194, width)
    host = host.pin_memory()
    locations = torch.arange(rows * (hot + 1), dtype=torch.int32).reshape(rows, hot + 1)
    # Physical slots differ from logical hot-slot and host-token indices.
    locations = locations.flip(1).contiguous()
    before_tags = torch.tensor([tags + [-1]] * rows, dtype=torch.int32)
    before_lru = torch.tensor([order] * rows, dtype=torch.int16)
    before_bytes = torch.full((rows * (hot + 1) + 1, width), -99, dtype=torch.int64)
    for rid in range(rows):
        valid = [i for i, token in enumerate(tags) if token >= 0]
        before_bytes[locations[rid, valid].long()] = host[[tags[i] for i in valid]]
        before_bytes[int(locations[rid, hot])] = host[length - 1]
    common = dict(
        top_k_tokens=torch.tensor(
            [selected, selected], dtype=torch.int32, device="cuda"
        ),
        device_buffer_locs=locations.cuda(),
        host_cache_locs=torch.arange(8194, dtype=torch.int64, device="cuda").repeat(
            rows, 1
        ),
        host_cache=host,
        req_pool_indices=torch.tensor([1, 2], dtype=req_dtype, device="cuda"),
        seq_lens=torch.tensor([length, length], dtype=seq_dtype, device="cuda"),
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device="cuda"),
        item_size_bytes=width * 8,
        num_top_k=top,
        hot_buffer_size=hot,
        block_size=1024,
        skip_io=True,
    )
    states, graphs, runners = {}, {}, {}
    for enabled in (False, True):
        state = dict(
            device_buffer_tokens=before_tags.cuda(),
            lru_slots=before_lru.cuda(),
            device_buffer=before_bytes.cuda(),
            top_k_device_locs=torch.full(
                (2, top), -77, dtype=torch.int32, device="cuda"
            ),
            miss_src=torch.full((2, top), -77, dtype=torch.int64, device="cuda"),
            miss_dst=torch.full((2, top), -77, dtype=torch.int32, device="cuda"),
            miss_count=torch.full((2,), -77, dtype=torch.int32, device="cuda"),
        )
        states[enabled] = state

        def run(enabled=enabled, state=state):
            load_cache_to_device_buffer_mla(**common, **state, batched_prefix=enabled)
            copy_cache_planned_mla(
                miss_src=state["miss_src"],
                miss_dst=state["miss_dst"],
                miss_count=state["miss_count"],
                num_real_reqs=common["num_real_reqs"],
                host_cache=host,
                device_buffer=state["device_buffer"],
                item_size_bytes=width * 8,
            )

        run()  # Compile outside capture.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graphs[enabled] = graph
        runners[enabled] = run

    for execution, rid in [("Eager", 1), ("Graph", 1), ("Changed", 3)]:
        common["req_pool_indices"][0] = rid
        expected_tags, expected_lru = before_tags.clone(), before_lru.clone()
        expected_bytes = before_bytes.clone()
        expected_src = torch.full((2, top), -77, dtype=torch.int64)
        expected_dst = torch.full((2, top), -77, dtype=torch.int32)
        expected_out = torch.full((2, top), -1, dtype=torch.int32)
        n = 0
        if pattern == "Short":
            expected_out[0, :length] = locations[rid, selected[:length]]
        else:
            chosen = set(selected) - {length - 1}
            hits = [slot for slot in order if tags[slot] in chosen]
            stale = [slot for slot in order if tags[slot] not in chosen]
            lookup = {tags[slot]: slot for slot in hits}
            missing = [
                token
                for token in selected
                if token != length - 1 and token not in lookup
            ]
            n = len(missing)
            for i, token in enumerate(missing):
                slot = stale[i]
                lookup[token] = slot
                expected_tags[rid, slot] = token
                expected_src[0, i] = token
                expected_dst[0, i] = locations[rid, slot]
                expected_bytes[int(locations[rid, slot])] = host[token]
            expected_lru[rid] = torch.tensor(
                stale[n:] + stale[:n] + hits, dtype=torch.int16
            )
            expected_out[0] = torch.tensor(
                [
                    int(locations[rid, hot if token == length - 1 else lookup[token]])
                    for token in selected
                ],
                dtype=torch.int32,
            )
        expected = dict(
            device_buffer_tokens=expected_tags,
            lru_slots=expected_lru,
            device_buffer=expected_bytes,
            miss_src=expected_src,
            miss_dst=expected_dst,
            miss_count=torch.tensor([n, -77], dtype=torch.int32),
            top_k_device_locs=expected_out,
        )
        for enabled, state in states.items():
            state["device_buffer_tokens"].copy_(before_tags)
            state["lru_slots"].copy_(before_lru)
            state["device_buffer"].copy_(before_bytes)
            for key in ("miss_src", "miss_dst", "miss_count", "top_k_device_locs"):
                state[key].fill_(-77)
            runners[enabled]() if execution == "Eager" else graphs[enabled].replay()
            torch.cuda.synchronize()
            for key, value in expected.items():
                assert torch.equal(state[key].cpu(), value), (execution, enabled, key)
