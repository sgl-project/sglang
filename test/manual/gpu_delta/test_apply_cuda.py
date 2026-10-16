"""Native grouped XOR: full byte storage, metadata lifetime and runtime geometry.

Run on CUDA; the CPU suite covers validation and planning independently.
"""

import gc
import math
import weakref
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

from sglang.srt.weight_sync.gpu_delta.apply import (
    BLOCK,
    ByteApplyGroup,
    _compiled,
    _packed_row_start,
    plan_apply,
)
from sglang.srt.weight_sync.gpu_delta.bindings import _direct_binding
from sglang.srt.weight_sync.gpu_delta.layout import PreparedDelta

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_runtime_geometry_word_units_tails_and_byte_fallback():
    cases = []
    # Packed rows with runtime divisors, including non-power-of-two widths.
    for tiles in (1, 3, 7, 33):
        inner = tiles * BLOCK
        cases.append(
            ((3, inner), (inner + 16, 1), (3 * tiles, BLOCK), (BLOCK + 4, 1), 0)
        )
    # Generic word-coordinate records, inactive strides, and >2-axis fallback.
    for shape, stride in (
        ((3, 1028), (1040, 1)),
        ((3, 2052), (2068, 1)),
        ((3, 5, 7, 4), (140, 4, 20, 1)),
        ((2, 3, 2048), (16384, 2052, 1)),
        ((2, 3, 1, 5, 4), (60, 4, 123, 12, 1)),
        ((1, 1, 4), ((1 << 63) - 1, 17, 1)),
    ):
        size = math.prod(shape)
        cases.extend(
            [
                ((size,), (1,), shape, stride, 0),
                ((size,), (1,), shape, stride, 1),
                (shape, stride, (size,), (1,), 0),
            ]
        )
    cases.extend(
        [
            ((3, 7, 103), (1024, 128, 1), (3, 7, 103), (2048, 256, 2), 1),
            ((2, 4099), (8198, 2), (2, 4099), (12288, 2), 1),
            ((4100,), (1,), (4100,), (1,), 4),
        ]
    )
    cache = None
    for ss, st, ts, tt, shift in cases:
        size = math.prod(ss)
        spans = [
            1 + sum((n - 1) * s for n, s in zip(sh, steps))
            for sh, steps in ((ss, st), (ts, tt))
        ]
        pitch = (max(spans) + 15) // 16 * 16 + 32
        source = torch.randint(256, (pitch * 2,), dtype=torch.uint8, device="cuda")
        target = torch.randint(256, (pitch * 2,), dtype=torch.uint8, device="cuda")
        before_source, before_target = source.clone(), target.clone()
        expected = target.clone()
        for item in range(2):
            expected.as_strided(ts, tt, shift + item * pitch).bitwise_xor_(
                source.as_strided(ss, st, shift + item * pitch).reshape(ts)
            )
        group = ByteApplyGroup(
            [shift, shift + pitch],
            [target.data_ptr() + shift, target.data_ptr() + shift + pitch],
            (((size, ss, st, ts, tt), 2),),
        )
        group.prepare_metadata(source.device.index)
        group.bind(source)
        current = _compiled.cache_info()
        if cache is not None:
            assert (current.misses, current.currsize) == (cache.misses, cache.currsize)
        cache = current
        pointers = torch.tensor(
            list(group.pointer_rows(source.data_ptr())),
            dtype=torch.int64,
            device="cuda",
        )
        metadata = torch.tensor(group.metadata_rows, dtype=torch.int64, device="cuda")
        error = torch.ones(1, dtype=torch.int32, device="cuda")
        for grid in (1, group.tiles):
            target.copy_(before_target)
            launch = group.bind_launch(
                pointers, metadata, error, tuple((grid, 1, 1) for _ in group.programs)
            )
            error.fill_(1)
            launch()
            torch.testing.assert_close(target, before_target, rtol=0, atol=0)
            error.zero_()
            launch()
            torch.testing.assert_close(target, expected, rtol=0, atol=0)
            torch.testing.assert_close(source, before_source, rtol=0, atol=0)


def case(kind, index):
    if kind in ("permuted", "permuted2"):
        shape, stride = (
            ((3, 5, 7, 4), (140, 4, 20, 1))
            if kind == "permuted"
            else ((2, 3, 11, 4), (132, 4, 12, 1))
        )
        size = math.prod(shape)
        canonical_shape, slices = (1, size), [[0, 1], [0, size]]
        extent = 1 + sum((n - 1) * step for n, step in zip(shape, stride))
        storage = torch.full((extent + 32,), 0x5A, dtype=torch.uint8, device="cuda")
        target = storage.as_strided(shape, stride, 16)
    elif kind == "row":
        canonical_shape = (3, 8192)
        slices = [[0, 3], [2048, 6144]]
        storage = torch.full((3, 8192), 0x5A, dtype=torch.uint8, device="cuda")
        target = storage[:, 2048:6144]
    elif kind == "affine":
        canonical_shape = (3, 10, 16)
        slices = [[0, 3], [0, 5], [2, 9]]
        storage = torch.full((4, 7, 11), 0x5A, dtype=torch.uint8, device="cuda")
        target = storage[:3, 1:6, 1:8]
    else:
        size, padding = kind
        canonical_shape = (1, size)
        slices = [[0, 1], [0, size]]
        storage = torch.full(
            (size + 2 * padding,), 0x5A, dtype=torch.uint8, device="cuda"
        )
        target = storage[padding:-padding].view(1, size)
    binding = _direct_binding(
        f"matrix{index}",
        {"dtype": "U8", "shape": list(canonical_shape)},
        target,
        slices,
    )
    mask = torch.randint(
        256, (math.prod(canonical_shape),), dtype=torch.uint8, device="cuda"
    )
    before = storage.clone()
    # Use normal Torch views as the independent complete-storage oracle.
    target.bitwise_xor_(binding.selected_bytes(mask).reshape(target.shape))
    expected = storage.clone()
    storage.copy_(before)
    return binding, mask, storage, before, expected


def test_changed_counts_layouts_and_alignment_reuse_three_fixed_modules_and_exact_bytes():
    kernels = tuple(_compiled(torch.cuda.current_device(), p) for p in range(3))
    misses = _compiled.cache_info().misses
    variants = [
        [(17, 1)],
        [(4100, 4)] * 3 + ["row", "permuted", "permuted2", "affine"],
        ["affine", (4097, 1), (8192, 16), "row", (1, 1)],
        [(17, 1)] * 5,
    ]
    populated_families = set()
    for kinds in variants:
        items = [case(kind, index) for index, kind in enumerate(kinds)]
        outputs, length = [], 0
        for binding, mask, *_ in items:
            offset = (length + 15) // 16 * 16
            outputs.append((binding, offset, mask.numel()))
            length = offset + mask.numel()
        scratch = torch.zeros(length, dtype=torch.uint8, device="cuda")
        for (_, offset, size), (_, mask, *_) in zip(outputs, items):
            scratch[offset : offset + size].copy_(mask)
        group = plan_apply(outputs)
        group.prepare_metadata(scratch.device.index)
        assert group.kernels == kernels
        group.bind(scratch)
        populated_families.update(p.family for p in group.programs)
        pointers = torch.tensor(
            list(group.pointer_rows(scratch.data_ptr())),
            dtype=torch.int64,
            device="cuda",
        )
        metadata = torch.tensor(group.metadata_rows, dtype=torch.int64, device="cuda")
        error = torch.ones(1, dtype=torch.int32, device="cuda")
        launch = group.bind_launch(pointers, metadata, error)
        launch()
        for _, _, storage, before, _ in items:
            torch.testing.assert_close(storage, before, rtol=0, atol=0)
        error.zero_()
        launch()
        for _, _, storage, _, expected in items:
            torch.testing.assert_close(storage, expected, rtol=0, atol=0)
        assert group.tiles == sum(
            count * math.ceil(c[0] / BLOCK) for c, count in group.contracts
        )
        assert _compiled.cache_info().misses == misses
    assert populated_families == {"packed", "word_affine", "general"}


def test_grid_stride_covers_group_boundaries_empty_groups_and_partial_tiles():
    def linear(size):
        return size, (size,), (1,), (size,), (1,)

    source = torch.randint(256, (49152,), dtype=torch.uint8, device="cuda")
    targets = [
        torch.full((size,), 0x5A, dtype=torch.uint8, device="cuda")
        for size in (8193, 8193, 4100)
    ]
    group = ByteApplyGroup(
        [0, 16384, 32768],
        [target.data_ptr() for target in targets],
        (
            (linear((1 << 63) - 1), 0),
            (linear(8193), 2),
            (linear(1), 0),
            (linear(4100), 1),
        ),
    )
    group.prepare_metadata(source.device.index)
    group.bind(source)
    misses = _compiled.cache_info().misses
    pointers = torch.tensor(
        list(group.pointer_rows(source.data_ptr())), dtype=torch.int64, device="cuda"
    )
    metadata = torch.tensor(group.metadata_rows, dtype=torch.int64, device="cuda")
    error = torch.zeros(1, dtype=torch.int32, device="cuda")
    assert group.grid_ctas == sum(
        min(p.tiles, 4 * p.resident_ctas) for p in group.programs
    )
    for grid in (1, 2, 4, group.tiles - 1, group.tiles, group.tiles + 1):
        launch = group.bind_launch(
            pointers, metadata, error, tuple((grid, 1, 1) for p in group.programs)
        )
        for target in targets:
            target.fill_(0x5A)
        error.fill_(1)
        launch()
        for target in targets:
            assert bool(torch.all(target == 0x5A))
        error.zero_()
        launch()
        for offset, target in zip(group.sources, targets):
            torch.testing.assert_close(
                target, source[offset : offset + target.numel()] ^ 0x5A, rtol=0, atol=0
            )
        assert _compiled.cache_info().misses == misses


def prepared(groups):
    owner = PreparedDelta.__new__(PreparedDelta)
    owner.static_plans = [(None, None, group) for group in groups]
    owner.apply_bindings = []
    owner.apply_source_patches = []
    owner.apply_fallback_groups = set()
    size = sum(2 * len(g.sources) + g.metadata_size for g in groups if g is not None)
    owner.apply_host_metadata = torch.empty(size, dtype=torch.int64, pin_memory=True)
    owner.apply_metadata = torch.empty(size, dtype=torch.int64, device="cuda")
    owner.error = torch.ones(1, dtype=torch.int32, device="cuda")
    owner.batches, owner.status_checks, owner.raw_copies = [], [], {}
    owner._prepare_apply_bindings()
    return owner


def test_preplanned_slab_aligned_fallback_restoration_all_families_status_and_leases():
    geometries = [
        ((8192,), (1,)),
        ((3, 5, 7, 4), (140, 4, 20, 1)),
        ((2, 3, 11, 4), (132, 4, 12, 1)),
        ((3, 5, 7), (77, 11, 1)),
    ]
    contracts, target_rows, storage_rows = [], [], []
    for shape, stride in geometries:
        size = math.prod(shape)
        extent = 1 + sum((n - 1) * s for n, s in zip(shape, stride))
        storage = torch.full((extent + 32,), 0x5A, dtype=torch.uint8, device="cuda")
        target = storage.as_strided(shape, stride, 16)
        contracts.append(((size, (size,), (1,), shape, stride), 1))
        target_rows.append(target)
        storage_rows.append(storage)
    offsets = [index * 131072 for index in range(len(contracts))]
    group = ByteApplyGroup(
        offsets, [t.data_ptr() for t in target_rows], tuple(contracts)
    )
    group.prepare_metadata(torch.cuda.current_device())
    assert [p.family for p in group.programs] == ["packed", "word_affine", "general"]
    cache = _compiled.cache_info()
    owner = prepared([group])
    prepared_descriptors = tuple(group.metadata_rows)
    buffers = [
        torch.randint(256, (524320,), dtype=torch.uint8, device="cuda")
        for _ in range(2)
    ]
    buffer_before = [b.clone() for b in buffers]
    # Same template, actual misaligned caller, then recovery to aligned state.
    for skew in (0, 1, 16):
        owner.decoded = [b[skew : skew + 524288] for b in buffers]
        for storage in storage_rows:
            storage.fill_(0x5A)
        before = [storage.clone() for storage in storage_rows]
        expected = []
        for offset, target, storage, (contract, _) in zip(
            offsets, target_rows, storage_rows, contracts
        ):
            target.bitwise_xor_(
                owner.decoded[0][offset : offset + contract[0]].reshape(target.shape)
            )
            expected.append(storage.clone())
            storage.fill_(0x5A)
        _, calls = owner._bind_apply_metadata()
        owner.apply_metadata.copy_(owner.apply_host_metadata, non_blocking=True)
        owner.error.fill_(1)
        calls[0]()
        for storage, original in zip(storage_rows, before):
            torch.testing.assert_close(storage, original, rtol=0, atol=0)
        owner.error.zero_()
        calls[0]()
        for storage, wanted in zip(storage_rows, expected):
            torch.testing.assert_close(storage, wanted, rtol=0, atol=0)
        for buffer, original in zip(buffers, buffer_before):
            torch.testing.assert_close(buffer, original, rtol=0, atol=0)
        if skew % 16 == 0:
            assert tuple(group.metadata_rows) == prepared_descriptors
            assert calls[0] is owner.apply_bindings[0][-1]
        assert _compiled.cache_info().misses == cache.misses
    torch.cuda.synchronize()
    owner.batches = [SimpleNamespace(apply=calls[0])]
    metadata_ref = weakref.ref(owner.apply_metadata)
    del calls
    owner._release_gpu()
    gc.collect()
    assert metadata_ref() is None
    assert owner.apply_host_view is None and owner.apply_bindings == []


def test_prebound_launch_graph_replay_keeps_status_gate_and_prepared_leases():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.randint(256, (8192,), dtype=torch.uint8, device="cuda")
        target = torch.randint(256, (8192,), dtype=torch.uint8, device="cuda")
        before = target.clone()
        contract = (8192, (8192,), (1,), (8192,), (1,))
        group = ByteApplyGroup([0], [target.data_ptr()], ((contract, 1),))
        group.prepare_metadata(source.device.index)
        owner = prepared([group])
        owner.decoded = [source, source]
        _, calls = owner._bind_apply_metadata()
        owner.apply_metadata.copy_(owner.apply_host_metadata, non_blocking=True)
        owner.error.zero_()
        calls[0]()
        target.copy_(before)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        calls[0]()
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(target, before ^ source, rtol=0, atol=0)
    with torch.cuda.stream(stream):
        owner.error.fill_(1)
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(target, before ^ source, rtol=0, atol=0)
    del graph, calls
    owner._release_gpu()


@triton.jit
def _row_probe(bases, metadata, result):
    index = tl.program_id(0)
    base = tl.load(bases + index)
    offset = _packed_row_start(base, metadata, index * 4, 2048)
    tl.store(result + index, offset)


def test_packed_row2d_actual_lowering_wide_tile_coordinates():
    limit = (1 << 31) - 1
    bases, descriptors, expected = [], [], []
    for row_tiles in (1, 3, 5, 7, 24, 33, 768, 1 << 20, limit):
        rows, inner = max(1, limit // row_tiles), row_tiles * 2048
        stride = inner + 4
        for tile in {0, min(row_tiles, rows * row_tiles - 1), rows * row_tiles - 1}:
            base = tile * 2048
            bases.append(base)
            descriptors.extend((rows, inner, stride, 1))
            expected.append(base // inner * stride + base % inner)
    source = torch.tensor(bases, dtype=torch.int64, device="cuda")
    metadata = torch.tensor(descriptors, dtype=torch.int64, device="cuda")
    result = torch.empty_like(source)
    _row_probe[(len(bases),)](source, metadata, result)
    torch.testing.assert_close(
        result, torch.tensor(expected, device="cuda"), rtol=0, atol=0
    )
