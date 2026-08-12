"""Extended tests for the HiSparse shared-index (plan-then-IO) prefetch.

Local-only (not registered to CI): these cover the CUDA-graph capture/replay
pattern, the DSv4 page-padded layout, and the SGLANG_DEBUG_HISPARSE_SKIP_IO
probe, each of which JIT-compiles extra kernel instantiations. The cheap
plan-replay correctness guards run in CI via
test/registered/kernels/ops/kvcache/test_hisparse.py, which this file imports
its fixtures from.

Run: python3 test/manual/kernels/test_hisparse_prefetch.py
"""

import sys
from pathlib import Path

import pytest
import torch

from sglang.kernels.ops.kvcache.hisparse import (
    copy_cache_planned_mla,
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
)
from sglang.srt.utils import is_hip

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
        / "registered"
        / "kernels"
        / "ops"
        / "kvcache"
    ),
)
from test_hisparse import (  # noqa: F401  (GPU/platform guard applies here too); noqa: E402
    DEVICE,
    DEVICE_CACHE_SIZE,
    DSV4_ITEM_BYTES,
    DSV4_PAGE_BYTES,
    DSV4_PAGE_SIZE,
    DTYPE,
    HOST_CACHE_SIZE,
    HOT_BUFFER_SIZE,
    ITEM_SIZE_BYTES,
    KV_DIM,
    _host_cache,
    _long_case,
    _make_plan,
    _make_state,
    _run_kernel,
    _write_dsv4_token,
    pytestmark,
)


def test_plan_then_io_dsv4_matches_sync_swap_in() -> None:
    """DSv4 layout: replaying the recorded plan lands the page-padded value+scale
    bytes exactly where the fused swap-in copy puts them."""
    num_pages = 2
    state = _long_case()
    plan_state = _long_case()

    def _dsv4_caches():
        host = torch.zeros(
            (num_pages, DSV4_PAGE_BYTES),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        for token in range(HOST_CACHE_SIZE):
            _write_dsv4_token(host, token, seed=token + 1)
        dev = torch.full(
            (num_pages, DSV4_PAGE_BYTES), 0xFF, dtype=torch.uint8, device=DEVICE
        )
        return host, dev

    common = dict(
        top_k_tokens=torch.tensor([[6]], dtype=torch.int32, device=DEVICE),
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=DEVICE),
        item_size_bytes=DSV4_ITEM_BYTES,
        num_top_k=1,
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=DSV4_PAGE_SIZE,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )

    # Reference: fused swap-in copies host token 6 into evict slot 0 (loc 9).
    ref_host, ref_dev = _dsv4_caches()
    out = torch.full((1, 1), -1, dtype=torch.int32, device=DEVICE)
    load_cache_to_device_buffer_dsv4_mla(
        device_buffer_tokens=state["device_buffer_tokens"],
        host_cache_locs=state["host_cache_locs"],
        device_buffer_locs=state["device_buffer_locs"],
        host_cache=ref_host,
        device_buffer=ref_dev,
        top_k_device_locs=out,
        lru_slots=state["lru_slots"],
        **common,
    )

    # Anchor: same swap-in on a twin state, recording the plan.
    miss_src, miss_dst, miss_count = _make_plan(1, 1)
    anchor_host, anchor_dev = _dsv4_caches()
    anchor_out = torch.full((1, 1), -1, dtype=torch.int32, device=DEVICE)
    load_cache_to_device_buffer_dsv4_mla(
        device_buffer_tokens=plan_state["device_buffer_tokens"],
        host_cache_locs=plan_state["host_cache_locs"],
        device_buffer_locs=plan_state["device_buffer_locs"],
        host_cache=anchor_host,
        device_buffer=anchor_dev,
        top_k_device_locs=anchor_out,
        lru_slots=plan_state["lru_slots"],
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        **common,
    )

    # Skip layer: replay the plan into a fresh buffer; must match the reference.
    replay_host, replay_dev = _dsv4_caches()
    copy_cache_planned_mla(
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        num_real_reqs=common["num_real_reqs"],
        host_cache=replay_host,
        device_buffer=replay_dev,
        item_size_bytes=DSV4_ITEM_BYTES,
        num_blocks=4,
        is_dsv4_layout=True,
    )
    torch.cuda.synchronize()

    assert torch.equal(miss_count.cpu(), torch.tensor([1], dtype=torch.int32))
    assert torch.equal(anchor_out.cpu(), out.cpu())
    assert torch.equal(replay_dev.cpu(), ref_dev.cpu())


def test_skip_io_probe_plans_without_moving_bytes() -> None:
    """skip_io still runs all planning (slot table, LRU, miss plan) but must
    leave the device buffer untouched; replaying the plan then repairs it."""
    locs = [[9, 7, 3, 5, 11]]
    toks = [[1, 4, 2, 5, -1]]
    top_k = torch.tensor([[6, 4]], dtype=torch.int32, device=DEVICE)
    nr, K = top_k.shape

    ref = _make_state(locs, toks, [7])
    ref_out = _run_kernel(top_k_tokens=top_k, seq_len=8, **ref)

    probe = _make_state(locs, toks, [7])
    probe_buffer_before = probe["device_buffer"].clone()
    miss_src, miss_dst, miss_count = _make_plan(nr, K)
    probe_out = _run_kernel(
        top_k_tokens=top_k,
        seq_len=8,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        skip_io=True,
        **probe,
    )

    # All planning outputs match the real run; only the bytes stayed put.
    assert torch.equal(probe_out.cpu(), ref_out.cpu())
    assert torch.equal(probe["lru_slots"].cpu(), ref["lru_slots"].cpu())
    assert torch.equal(probe["device_buffer"].cpu(), probe_buffer_before.cpu())
    assert not torch.equal(probe["device_buffer"].cpu(), ref["device_buffer"].cpu())

    copy_cache_planned_mla(
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        num_real_reqs=torch.tensor([nr], dtype=torch.int32, device=DEVICE),
        host_cache=probe["host_cache"],
        device_buffer=probe["device_buffer"],
        item_size_bytes=ITEM_SIZE_BYTES,
        num_blocks=4,
    )
    torch.cuda.synchronize()
    assert torch.equal(probe["device_buffer"].cpu(), ref["device_buffer"].cpu())


_PIO_LAYERS = 4  # one anchor (layer 0) + three skip layers (GLM group of freq 4)
_PIO_REQS = 2
_PIO_SEQ = 10  # > HOT_BUFFER_SIZE -> long path; newest token = 9
_PIO_DBL = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]  # per-req [4 LRU slots + newest]
_PIO_STEPS = [
    [[4, 5, 0, 9], [6, 7, 1, 9]],
    [[4, 10, 2, 9], [8, 5, 3, 9]],
    [[11, 4, 5, 9], [6, 12, 7, 9]],
    [[0, 1, 2, 9], [3, 4, 5, 9]],
]


def _pio_fresh(host_cache, dbl):
    """Fresh per-layer buffers/tokens/lru, all layers initialized identically."""
    buffers, dbt, lru = [], [], []
    init_tokens = [[0, 1, 2, 3, -1], [0, 1, 2, 3, -1]]
    for _ in range(_PIO_LAYERS):
        db = torch.full((DEVICE_CACHE_SIZE, 1, KV_DIM), -1, dtype=DTYPE, device=DEVICE)
        for rid in range(_PIO_REQS):
            for slot, tok in enumerate(init_tokens[rid][:HOT_BUFFER_SIZE]):
                db[dbl[rid, slot]].copy_(host_cache[tok].to(DEVICE))
            db[dbl[rid, HOT_BUFFER_SIZE]].copy_(host_cache[_PIO_SEQ - 1].to(DEVICE))
        buffers.append(db)
        dbt.append(torch.tensor(init_tokens, dtype=torch.int32, device=DEVICE))
        lru.append(
            torch.arange(HOT_BUFFER_SIZE, dtype=torch.int16, device=DEVICE)
            .view(1, -1)
            .repeat(_PIO_REQS, 1)
            .contiguous()
        )
    torch.cuda.synchronize()
    return buffers, dbt, lru


def _pio_swap_in(topk, dbt, lru, dbl, hcl, host, buffer, out, seq_lens, nrr, rpi, plan):
    miss_src, miss_dst, miss_count = plan if plan else (None, None, None)
    load_cache_to_device_buffer_mla(
        top_k_tokens=topk,
        device_buffer_tokens=dbt,
        host_cache_locs=hcl,
        device_buffer_locs=dbl,
        host_cache=host,
        device_buffer=buffer,
        top_k_device_locs=out,
        req_pool_indices=rpi,
        seq_lens=seq_lens,
        lru_slots=lru,
        item_size_bytes=ITEM_SIZE_BYTES,
        num_top_k=topk.shape[1],
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=1,
        block_size=256,
        num_real_reqs=nrr,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
    )


def _pio_prefetch_step(
    topk, buffers, dbt, lru, dbl, hcl, host, seq_lens, nrr, rpi, out, plan, side, events
):
    """Anchor records the plan; skip layers replay it copy-only on a side stream."""
    miss_src, miss_dst, miss_count = plan
    _pio_swap_in(
        topk, dbt[0], lru[0], dbl, hcl, host, buffers[0], out, seq_lens, nrr, rpi, plan
    )
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for layer in range(1, _PIO_LAYERS):
            copy_cache_planned_mla(
                miss_src=miss_src,
                miss_dst=miss_dst,
                miss_count=miss_count,
                num_real_reqs=nrr,
                host_cache=host,
                device_buffer=buffers[layer],
                item_size_bytes=ITEM_SIZE_BYTES,
                num_blocks=4,
            )
            events[layer].record(side)
    for layer in range(1, _PIO_LAYERS):
        events[layer].wait(torch.cuda.current_stream())


def _pio_sync_step(topk, buffers, dbt, lru, dbl, hcl, host, seq_lens, nrr, rpi):
    """Reference: run the full swap-in independently on every layer."""
    outs = []
    for layer in range(_PIO_LAYERS):
        out = torch.full_like(topk, -1)
        _pio_swap_in(
            topk,
            dbt[layer],
            lru[layer],
            dbl,
            hcl,
            host,
            buffers[layer],
            out,
            seq_lens,
            nrr,
            rpi,
            None,
        )
        outs.append(out.clone())
    return outs


@pytest.mark.skipif(is_hip(), reason="CUDA graph capture test is CUDA-only.")
def test_plan_then_io_cuda_graph_replay() -> None:
    """The plan-then-IO prefetch pattern captures into a CUDA graph and replays
    bit-identically to the eager synchronous swap-in across multiple steps."""
    host = _host_cache()
    dbl = torch.tensor(_PIO_DBL, dtype=torch.int32, device=DEVICE)
    hcl = (
        torch.arange(HOST_CACHE_SIZE, dtype=torch.int64, device=DEVICE)
        .view(1, -1)
        .repeat(_PIO_REQS, 1)
        .contiguous()
    )
    seq_lens = torch.full((_PIO_REQS,), _PIO_SEQ, dtype=torch.int32, device=DEVICE)
    nrr = torch.tensor([_PIO_REQS], dtype=torch.int32, device=DEVICE)
    rpi = torch.arange(_PIO_REQS, dtype=torch.int64, device=DEVICE)
    K = len(_PIO_STEPS[0][0])
    steps = [torch.tensor(s, dtype=torch.int32, device=DEVICE) for s in _PIO_STEPS]

    # Reference: full synchronous swap-in on every layer, snapshotted per step.
    ref_buf, ref_dbt, ref_lru = _pio_fresh(host, dbl)
    ref_slots, ref_snap = [], []
    for topk in steps:
        ref_slots.append(
            _pio_sync_step(
                topk, ref_buf, ref_dbt, ref_lru, dbl, hcl, host, seq_lens, nrr, rpi
            )
        )
        torch.cuda.synchronize()
        ref_snap.append([b.clone() for b in ref_buf])
    torch.cuda.synchronize()

    # Graph-captured prefetch replayed step by step against a fixed topk buffer.
    buf, dbt, lru = _pio_fresh(host, dbl)
    topk_buf = torch.zeros((_PIO_REQS, K), dtype=torch.int32, device=DEVICE)
    out = torch.full((_PIO_REQS, K), -1, dtype=torch.int32, device=DEVICE)
    plan = _make_plan(_PIO_REQS, K)
    side = torch.cuda.Stream()
    events = [torch.cuda.Event() for _ in range(_PIO_LAYERS)]

    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        topk_buf.copy_(steps[0])
        _pio_prefetch_step(
            topk_buf,
            buf,
            dbt,
            lru,
            dbl,
            hcl,
            host,
            seq_lens,
            nrr,
            rpi,
            out,
            plan,
            side,
            events,
        )
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()

    # Reset state mutated by warmup so capture starts from a clean identical state.
    buf, dbt, lru = _pio_fresh(host, dbl)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _pio_prefetch_step(
            topk_buf,
            buf,
            dbt,
            lru,
            dbl,
            hcl,
            host,
            seq_lens,
            nrr,
            rpi,
            out,
            plan,
            side,
            events,
        )
    torch.cuda.synchronize()

    for s, topk in enumerate(steps):
        topk_buf.copy_(topk)
        graph.replay()
        torch.cuda.synchronize()
        # Anchor slot table matches the synchronous layer-0 result.
        assert torch.equal(
            out.cpu(), ref_slots[s][0].cpu()
        ), f"slots differ at step {s}"
        # Every layer's device buffer stays bit-identical to synchronous swap-in.
        for layer in range(_PIO_LAYERS):
            assert torch.equal(
                buf[layer].cpu(), ref_snap[s][layer].cpu()
            ), f"buffer differs at step {s}, layer {layer}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
