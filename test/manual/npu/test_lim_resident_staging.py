"""Numerical and graph-replay checks for short resident KV staging."""

import pytest
import torch

pytest.importorskip("torch_npu")

from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
    create_fused_shm_tensor as create_shm_tensor,
)
from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
    free_fused_shm as free_shm,
)

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_kv_residency import (
    stage_resident_kv,
)


@pytest.mark.parametrize("graph", [False, True])
def test_staging_dynamic_rows_and_reuse(graph):
    if not torch.npu.is_available():
        pytest.skip("NPU is required")
    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    capacity, context, features = 128, 256, 576
    host, _, pointer = create_shm_tensor(
        shape=(1, 3 * context, 1, features),
        dtype=torch.bfloat16,
        device_id=device_id,
        name="resident_staging_dynamic_test",
    )
    token = torch.arange(3 * context, dtype=torch.int32)[:, None]
    feature = torch.arange(features, dtype=torch.int32)[None, :]
    # All values are exactly representable in BF16; vary both coordinates.
    host_values = ((token * 17 + feature * 13) % 251).to(torch.bfloat16)[:, None, :]
    host[0].copy_(host_values)
    cache = torch.full(
        (3, capacity, 1, features), -1, dtype=torch.bfloat16, device=device
    )
    mapping = torch.arange(3 * context, dtype=torch.int32, device=device).view(
        3, context
    )
    requests = torch.tensor([1, 2], dtype=torch.int32, device=device)
    lengths = torch.tensor([5, capacity], dtype=torch.int32, device=device)
    first = torch.tensor([3, capacity - 2], dtype=torch.int32, device=device)
    token_requests = torch.tensor([1, 1, 2, 2], dtype=torch.int32, device=device)
    positions = torch.tensor(
        [3, 4, capacity - 2, capacity - 1], dtype=torch.int32, device=device
    )
    valid = torch.ones(4, dtype=torch.bool, device=device)
    current = (
        ((torch.arange(4)[:, None] * 29 + feature * 7) % 251).to(torch.bfloat16)[
            :, None, :
        ]
    ).to(device)
    sources = torch.arange(4, dtype=torch.int64, device=device)
    old_begin = torch.full((3,), -1, dtype=torch.int32, device=device)
    old_end = old_begin.clone()
    extents = tuple(old_begin.clone() for _ in range(3))
    history = (
        torch.empty(2 * capacity, dtype=torch.int64, device=device),
        torch.empty(2 * capacity, dtype=torch.int64, device=device),
        torch.empty(2 * capacity, dtype=torch.bool, device=device),
    )
    current_workspace = (
        torch.empty(4, dtype=torch.int64, device=device),
        torch.empty(4, dtype=torch.bool, device=device),
    )

    def run():
        stage_resident_kv(
            current=current,
            source_indices=sources,
            host=host,
            host_address=pointer,
            cache=cache,
            req_to_token=mapping,
            requests=requests,
            lengths=lengths,
            first_positions=first,
            token_requests=token_requests,
            token_positions=positions,
            token_valid=valid,
            old_begin=old_begin,
            old_end=old_end,
            next_extents=extents,
            history_workspace=history,
            current_workspace=current_workspace,
            capacity=capacity,
            context_len=context,
        )

    def check(slot, first_pos, current_rows):
        torch.npu.synchronize()
        torch.testing.assert_close(
            cache[slot, :first_pos].cpu(),
            host_values[slot * context : slot * context + first_pos],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cache[slot, first_pos : first_pos + 2].cpu(),
            current[current_rows].cpu(),
            rtol=0,
            atol=0,
        )

    try:
        if graph:
            # JIT compilation is independent of pytest parameter order and
            # happens before capture. Reset before and after capture so the
            # assertion below tests replay, regardless of capture semantics.
            run()
            torch.npu.synchronize()
            cache.fill_(-1)
            old_begin.fill_(-1)
            old_end.fill_(-1)
            for extent in extents:
                extent.fill_(-1)
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                run()
            torch.npu.synchronize()
            cache.fill_(-1)
            old_begin.fill_(-1)
            old_end.fill_(-1)
            for extent in extents:
                extent.fill_(-1)
            captured.replay()
        else:
            run()
        check(1, 3, slice(0, 2))
        assert torch.all(cache[2] == -1).item()
        assert old_begin.cpu().tolist() == [-1, 0, -1]
        # Cached history must win over subsequently changed host contents.
        resident = cache[1, :3].cpu().clone()
        host[0, context : context + 3].fill_(99)
        if graph:
            captured.replay()
        else:
            run()
        torch.npu.synchronize()
        torch.testing.assert_close(cache[1, :3].cpu(), resident, rtol=0, atol=0)
        assert torch.all(cache[2] == -1).item()
        # Partial residency preserves the prefix and reloads only its gap.
        old_end[1] = 2
        if graph:
            captured.replay()
        else:
            run()
        torch.npu.synchronize()
        torch.testing.assert_close(cache[1, :2].cpu(), resident[:2], rtol=0, atol=0)
        assert torch.all(cache[1, 2] == 99).item()
        assert torch.all(cache[2] == -1).item()
        host[0, context : context + 3].copy_(host_values[context : context + 3])
        # Reverse short/long and request order without recapturing.
        long_before = cache[1].cpu().clone()
        requests.copy_(torch.tensor([2, 1], dtype=torch.int32, device=device))
        lengths.copy_(torch.tensor([5, capacity], dtype=torch.int32, device=device))
        first.copy_(torch.tensor([3, capacity - 2], dtype=torch.int32, device=device))
        token_requests.copy_(
            torch.tensor([2, 2, 1, 1], dtype=torch.int32, device=device)
        )
        if graph:
            captured.replay()
        else:
            run()
        check(2, 3, slice(0, 2))
        torch.testing.assert_close(cache[1].cpu(), long_before, rtol=0, atol=0)
        assert old_begin.cpu().tolist() == [-1, -1, 0]
        # Reuse slot 2 with changed physical mapping and invalidated extent.
        mapping[2, :3].copy_(
            torch.tensor([9, 10, 11], dtype=torch.int32, device=device)
        )
        old_begin[2] = -1
        old_end[2] = -1
        if graph:
            captured.replay()
        else:
            run()
        torch.npu.synchronize()
        torch.testing.assert_close(
            cache[2, :3].cpu(), host_values[[9, 10, 11]], rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache[2, 3:5].cpu(), current[:2].cpu(), rtol=0, atol=0
        )
        # C - 1 remains dense, while C is sparse; both change on replay.
        lengths.copy_(
            torch.tensor([capacity, capacity - 1], dtype=torch.int32, device=device)
        )
        first.copy_(
            torch.tensor([capacity - 2, capacity - 3], dtype=torch.int32, device=device)
        )
        positions.copy_(
            torch.tensor(
                [capacity - 2, capacity - 1, capacity - 3, capacity - 2],
                dtype=torch.int32,
                device=device,
            )
        )
        if graph:
            captured.replay()
        else:
            run()
        check(1, capacity - 3, slice(2, 4))
        assert old_begin.cpu().tolist() == [-1, 0, -1]
        # C + 1 is also long and must invalidate without touching payload.
        lengths.copy_(
            torch.tensor([capacity + 1, capacity], dtype=torch.int32, device=device)
        )
        before = cache[2].cpu().clone()
        if graph:
            captured.replay()
        else:
            run()
        torch.npu.synchronize()
        torch.testing.assert_close(cache[2].cpu(), before, rtol=0, atol=0)
        assert old_begin[2].item() == old_end[2].item() == -1
        # Invalid source IDs leave their destinations untouched.
        lengths.copy_(torch.tensor([5, capacity], dtype=torch.int32, device=device))
        first.copy_(torch.tensor([3, capacity - 2], dtype=torch.int32, device=device))
        positions.copy_(
            torch.tensor(
                [3, 4, capacity - 2, capacity - 1], dtype=torch.int32, device=device
            )
        )
        mapping[2, :3].copy_(
            torch.tensor([-1, 3 * context + 7, 9], dtype=torch.int32, device=device)
        )
        cache[2].fill_(-7)
        old_begin[2] = -1
        old_end[2] = -1
        long_before = cache[1].cpu().clone()
        if graph:
            captured.replay()
        else:
            run()
        torch.npu.synchronize()
        assert torch.all(cache[2, :2] == -7).item()
        torch.testing.assert_close(cache[2, 2].cpu(), host_values[9], rtol=0, atol=0)
        torch.testing.assert_close(
            cache[2, 3:5].cpu(), current[:2].cpu(), rtol=0, atol=0
        )
        torch.testing.assert_close(cache[1].cpu(), long_before, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        free_shm(device_id)


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 8, 14])
@pytest.mark.parametrize("graph", [False, True])
def test_staging_width_padding_and_source_bounds(width, graph):
    """Guard padded rows and invalid source indices across graph replays."""
    if not torch.npu.is_available():
        pytest.skip("NPU is required")
    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    capacity, context, features = 32, 64, 576
    host, _, pointer = create_shm_tensor(
        shape=(1, 2 * context, 1, features),
        dtype=torch.bfloat16,
        device_id=device_id,
        name="resident_staging_width_test",
    )
    feature = torch.arange(features, dtype=torch.int32)
    host_values = (
        (torch.arange(2 * context, dtype=torch.int32)[:, None] * 17 + feature * 13)
        % 251
    ).to(torch.bfloat16)[:, None, :]
    host[0].copy_(host_values)
    current = (
        ((torch.arange(width, dtype=torch.int32)[:, None] * 29 + feature * 7) % 251)
        .to(torch.bfloat16)[:, None, :]
        .to(device)
    )
    cache = torch.full(
        (2, capacity, 1, features), -7, dtype=torch.bfloat16, device=device
    )
    mapping = torch.arange(2 * context, dtype=torch.int32, device=device).view(
        2, context
    )
    # Use a strided view, as decode and speculative paths do.
    position_matrix = (
        torch.arange(width, dtype=torch.int32, device=device).view(1, width) + 2
    )
    first = position_matrix[:, 0]
    requests = torch.tensor([1], dtype=torch.int32, device=device)
    lengths = torch.tensor([2 + width], dtype=torch.int32, device=device)
    token_requests = torch.ones(width, dtype=torch.int32, device=device)
    positions = position_matrix.reshape(-1)
    valid = torch.ones(width, dtype=torch.bool, device=device)
    sources = torch.arange(width, dtype=torch.int64, device=device)
    old_begin = torch.full((2,), -1, dtype=torch.int32, device=device)
    old_end = old_begin.clone()
    extents = tuple(old_begin.clone() for _ in range(3))
    history = (
        torch.empty(capacity, dtype=torch.int64, device=device),
        torch.empty(capacity, dtype=torch.int64, device=device),
        torch.empty(capacity, dtype=torch.bool, device=device),
    )
    workspace = (
        torch.empty(width, dtype=torch.int64, device=device),
        torch.empty(width, dtype=torch.bool, device=device),
    )

    def run():
        stage_resident_kv(
            current=current,
            source_indices=sources,
            host=host,
            host_address=pointer,
            cache=cache,
            req_to_token=mapping,
            requests=requests,
            lengths=lengths,
            first_positions=first,
            token_requests=token_requests,
            token_positions=positions,
            token_valid=valid,
            old_begin=old_begin,
            old_end=old_end,
            next_extents=extents,
            history_workspace=history,
            current_workspace=workspace,
            capacity=capacity,
            context_len=context,
        )

    def replay():
        if graph:
            captured.replay()
        else:
            run()
        torch.npu.synchronize()

    try:
        if graph:
            run()
            torch.npu.synchronize()
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                run()
        cache.fill_(-7)
        old_begin.fill_(-1)
        old_end.fill_(-1)
        replay()
        torch.testing.assert_close(
            cache[1, :2].cpu(), host_values[context : context + 2], rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache[1, 2 : 2 + width].cpu(), current.cpu(), rtol=0, atol=0
        )
        # Reuse the captured graph with negative, out-of-range and last-valid
        # source rows, and padded/invalid request IDs.
        cache.fill_(-7)
        sources.fill_(width - 1)
        sources[0] = -1
        if width > 1:
            sources[1] = width
        if width > 3:
            valid[3] = False
        if width > 4:
            token_requests[4] = 2
        replay()
        assert torch.all(cache[1, 2] == -7).item()
        if width > 1:
            assert torch.all(cache[1, 3] == -7).item()
        if width > 2:
            torch.testing.assert_close(
                cache[1, 4].cpu(), current[-1].cpu(), rtol=0, atol=0
            )
        if width > 3:
            assert torch.all(cache[1, 5] == -7).item()
        if width > 4:
            assert torch.all(cache[1, 6] == -7).item()
        if width == 1:
            # These rows were previously valid; stale extents must not make an
            # invalid length or first position publish current KV on replay.
            sources.zero_()
            for invalid_length in (0, context + 1, 1):
                cache.fill_(-7)
                old_begin.fill_(-1)
                old_end.fill_(-1)
                extents[0][1] = 0
                extents[1][1] = 3
                extents[2][1] = 2
                lengths.fill_(invalid_length)
                replay()
                assert torch.all(cache == -7).item()
                assert old_begin[1].item() == old_end[1].item() == -1
                assert [extent[1].item() for extent in extents] == [-1, -1, -1]
            # A real cache row outside the mapping table must not use stale next
            # metadata, even if a token is marked valid.
            mapping = mapping[:1]
            lengths.fill_(3)
            extents[0][1] = 0
            extents[1][1] = 3
            extents[2][1] = 2
            cache.fill_(-7)
            if graph:
                run()
                torch.npu.synchronize()
                captured = torch.npu.NPUGraph()
                with torch.npu.graph(captured):
                    run()
            replay()
            assert torch.all(cache == -7).item()
            assert old_begin[1].item() == old_end[1].item() == -1
    finally:
        torch.npu.synchronize()
        free_shm(device_id)


@pytest.mark.parametrize("graph", [False, True])
def test_long_rows_leave_cache_unchanged(graph):
    """A 64K long row must invalidate residency without writing cache data."""
    if not torch.npu.is_available():
        pytest.skip("NPU is required")
    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    capacity, context, features = 12288, 65536, 576
    host, _, pointer = create_shm_tensor(
        shape=(1, 4, 1, features),
        dtype=torch.bfloat16,
        device_id=device_id,
        name="resident_staging_long_test",
    )
    cache = torch.full(
        (2, capacity, 1, features), -7, dtype=torch.bfloat16, device=device
    )
    mapping = torch.zeros((2, context), dtype=torch.int32, device=device)
    requests = torch.tensor([1], dtype=torch.int32, device=device)
    lengths = torch.tensor([context], dtype=torch.int32, device=device)
    first = torch.tensor([context - 1], dtype=torch.int32, device=device)
    valid = torch.ones(1, dtype=torch.bool, device=device)
    source = torch.zeros(1, dtype=torch.int64, device=device)
    current = torch.ones((1, 1, features), dtype=torch.bfloat16, device=device)
    old_begin = torch.zeros(2, dtype=torch.int32, device=device)
    old_end = old_begin.clone()
    extents = tuple(old_begin.clone() for _ in range(3))
    history = (source, source, valid)
    workspace = (source, valid)

    def run():
        stage_resident_kv(
            current=current,
            source_indices=source,
            host=host,
            host_address=pointer,
            cache=cache,
            req_to_token=mapping,
            requests=requests,
            lengths=lengths,
            first_positions=first,
            token_requests=requests,
            token_positions=first,
            token_valid=valid,
            old_begin=old_begin,
            old_end=old_end,
            next_extents=extents,
            history_workspace=history,
            current_workspace=workspace,
            capacity=capacity,
            context_len=context,
        )

    try:
        if graph:
            run()
            torch.npu.synchronize()
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                run()
            old_begin[1] = 0
            old_end[1] = 10
            captured.replay()
        else:
            run()
        torch.npu.synchronize()
        assert torch.all(cache == -7).item()
        assert old_begin[1].item() == old_end[1].item() == -1
    finally:
        torch.npu.synchronize()
        free_shm(device_id)
