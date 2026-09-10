import pytest
import torch

from sglang.srt.mem_cache.dsv41_request_window import (
    RequestWindow,
    copy_packed_tokens,
    window_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class PackedPool:
    def __init__(self, size, layers, page=4):
        self.size = size
        self.dtype = torch.uint8
        self.kv_buffer = [
            torch.full(((size + page) // page, page * 584), 219, dtype=torch.uint8)
            for _ in range(layers)
        ]


def token_bytes(buf, loc, page=4):
    p, off = divmod(int(loc), page)
    return torch.cat(
        (
            buf[p, off * 576 : (off + 1) * 576],
            buf[p, page * 576 + off * 8 : page * 576 + (off + 1) * 8],
        )
    )


def put(buf, loc, value, page=4):
    p, off = divmod(int(loc), page)
    buf[p, off * 576 : (off + 1) * 576] = value
    buf[p, page * 576 + off * 8 : page * 576 + (off + 1) * 8] = value + 1


def run_chunk(state, req, pos, *, floor=None, layer=0):
    layout = window_layout(
        torch.tensor(req),
        torch.tensor(pos),
        window=3,
        capacity=state.capacity,
        floor=torch.tensor(floor) if floor is not None else None,
    )
    state.activate(layout)
    buf = state.buffer(layer)
    for loc, r, p in zip(layout.write_loc.tolist(), req, pos):
        put(buf, loc, r * 40 + p)
    return layout, buf


def test_copy_preserves_separate_scale_region_across_pages():
    src = torch.randint(0, 256, (7, 4 * 584), dtype=torch.uint8)
    dst = torch.full_like(src, 219)
    source = torch.tensor([0, 3, 4, 17, 22])
    target = torch.tensor([18, 2, 7, 11, 0])
    before = dst.clone()
    copy_packed_tokens(src, dst, source, target, page_size=4)
    for a, b in zip(source.tolist(), target.tolist()):
        assert torch.equal(token_bytes(src, a), token_bytes(dst, b))
    for i in set(range(28)) - set(target.tolist()):
        assert torch.equal(token_bytes(before, i), token_bytes(dst, i))


@pytest.mark.parametrize("length", [1, 3, 4, 17])
def test_large_prefill_reads_all_current_kv_before_ring_commit(length):
    state = RequestWindow(PackedPool, num_slots=3, layers=2, page_size=4, capacity=6)
    layout, buf = run_chunk(state, [1] * length, list(range(length)))
    for p, indices in enumerate(layout.indices.tolist()):
        actual = [int(token_bytes(buf, i)[0]) for i in indices if i >= 0]
        assert actual == [40 + j for j in range(p, max(-1, p - 3), -1)]
    state.commit(0)
    next_layout, next_buf = run_chunk(state, [1], [length])
    actual = [
        int(token_bytes(next_buf, i)[0]) for i in next_layout.indices[0] if i >= 0
    ]
    assert actual == [40 + j for j in range(length, max(-1, length - 3), -1)]


def test_replay_ignores_poisoned_pre_tail_state_and_is_request_private():
    state = RequestWindow(PackedPool, num_slots=3, layers=2, page_size=4, capacity=6)
    layout, buf = run_chunk(
        state, [1, 1, 1, 2, 2, 2], [7, 8, 9, 7, 8, 9], floor=[7] * 6
    )
    assert layout.lengths.tolist() == [1, 2, 3, 1, 2, 3]
    assert int(token_bytes(buf, layout.indices[0, 0])[0]) == 47
    state.commit(0)
    assert not torch.equal(state.state.kv_buffer[0][2:4], state.state.kv_buffer[0][4:6])
    state.reset(torch.tensor([1]))
    with pytest.raises(RuntimeError, match="history is missing"):
        run_chunk(state, [1], [10])
    run_chunk(state, [2], [10])
    with pytest.raises(RuntimeError, match="history is missing"):
        run_chunk(state, [2], [10], layer=1)


def test_layout_shapes_depend_only_on_batch_geometry():
    a = window_layout(
        torch.tensor([1, 1, 2]),
        torch.tensor([5, 6, 0]),
        window=3,
        capacity=8,
        num_groups=4,
    )
    b = window_layout(
        torch.tensor([3, 3, 3]),
        torch.tensor([9, 10, 11]),
        window=3,
        capacity=8,
        num_groups=4,
    )
    for first, second in (
        (a.write_loc, b.write_loc),
        (a.indices, b.indices),
        (a.lengths, b.lengths),
        (a.history_req, b.history_req),
        (a.history_pos, b.history_pos),
        (a.history_loc, b.history_loc),
        (a.history_valid, b.history_valid),
        (a.commit_mask, b.commit_mask),
    ):
        assert first.shape == second.shape
    assert a.size == b.size == 4 * 3 + 3
    assert int(a.history_valid.sum()) == 3


def test_startup_dummy_history_does_not_relax_real_request_validation():
    state = RequestWindow(PackedPool, num_slots=3, layers=2, page_size=4, capacity=8)
    layout = window_layout(torch.tensor([1]), torch.tensor([20]), window=3, capacity=8)
    state.reset(torch.tensor([1]))
    state.activate(layout)
    with pytest.raises(RuntimeError, match="history is missing"):
        state.buffer(0)
    state.initialize_dummy_history()
    for layer in (0, 1):
        buf = state.buffer(layer)
        for loc in layout.history_loc:
            assert torch.count_nonzero(token_bytes(buf, loc)) == 0
    state.reset(torch.tensor([1]))
    with pytest.raises(RuntimeError, match="history is missing"):
        state.buffer(0)


@pytest.mark.parametrize("accepted", range(1, 7))
def test_verify_rejection_preserves_required_history(accepted):
    state = RequestWindow(
        PackedPool, num_slots=3, layers=1, page_size=4, capacity=12, workspace_rows=256
    )
    state.reset(torch.tensor([1]))
    run_chunk(state, [1] * 9, list(range(9)))
    state.commit(0)
    # Verification writes all six candidates, even when only one is accepted.
    run_chunk(state, [1] * 6, list(range(9, 15)))
    state.commit(0)
    next_pos = 9 + accepted
    layout, buf = run_chunk(state, [1], [next_pos])
    actual = [int(token_bytes(buf, i)[0]) for i in layout.indices[0]]
    assert actual == [40 + p for p in range(next_pos, next_pos - 3, -1)]


def test_layout_copy_preserves_captured_tensor_addresses():
    a = window_layout(
        torch.tensor([1] * 5), torch.arange(9, 14), window=3, capacity=12, num_groups=1
    )
    b = window_layout(
        torch.tensor([2] * 5), torch.arange(2, 7), window=3, capacity=12, num_groups=1
    )
    addresses = {
        name: getattr(a, name).data_ptr()
        for name in a.__struct_fields__
        if isinstance(getattr(a, name), torch.Tensor)
    }
    a.copy_(b)
    for name, addr in addresses.items():
        assert getattr(a, name).data_ptr() == addr
        assert torch.equal(getattr(a, name), getattr(b, name))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph regression")
def test_cuda_graph_replay_refreshes_request_history():
    def factory(size, layers):
        pool = PackedPool(size, layers)
        pool.kv_buffer = [buf.cuda() for buf in pool.kv_buffer]
        return pool

    state = RequestWindow(
        factory, num_slots=4, layers=1, page_size=4, capacity=12, workspace_rows=128
    )
    for slot in (1, 2):
        for pos in range(7, 10):
            put(state.state.kv_buffer[0], slot * state.capacity + pos, slot * 40 + pos)
    a = window_layout(
        torch.tensor([1], device="cuda"),
        torch.tensor([10], device="cuda"),
        window=3,
        capacity=state.capacity,
        num_groups=1,
    )
    b = window_layout(
        torch.tensor([2], device="cuda"),
        torch.tensor([10], device="cuda"),
        window=3,
        capacity=state.capacity,
        num_groups=1,
    )
    state.activate(a)
    payload = factory(4, 1).kv_buffer[0]
    put(payload, 0, 50)
    out = factory(4, 1).kv_buffer[0]
    source = torch.tensor([0], device="cuda")
    targets = torch.arange(3, device="cuda")

    def step():
        state.prepared = None
        buf = state.buffer(0)
        copy_packed_tokens(payload, buf, source, a.write_loc, page_size=4)
        copy_packed_tokens(buf, out, a.indices.flatten(), targets, page_size=4)
        state.commit(0)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    a.copy_(b)
    put(payload, 0, 90)
    graph.replay()
    torch.cuda.synchronize()
    assert [int(token_bytes(out, i)[0]) for i in range(3)] == [90, 89, 88]
    assert int(token_bytes(state.state.kv_buffer[0], 2 * state.capacity + 10)[0]) == 90


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
