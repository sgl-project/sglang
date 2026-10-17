"""Runtime descriptor, pre-pause planning and lease contracts without CUDA.

Import the complete production module with only the Triton compiler mocked.
The CUDA suite separately checks native lowering and complete backing storage.
"""

import gc
import importlib.util
import math
import sys
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.weight_sync.gpu_delta.layout import PreparedDelta
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def planner(monkeypatch):
    warmed, launched = [], []

    class Kernel:
        run = None

        def __getitem__(self, grid):
            return lambda pointers, metadata, error: launched.append(
                (grid, pointers, metadata, error)
            )

    def warmup(*args, **kwargs):
        warmed.append((args, kwargs))
        return Kernel()

    # No module slicing or assertions about function/source shape. Every host
    # method below is the real imported implementation.
    language = SimpleNamespace(constexpr=object())
    triton = SimpleNamespace(
        language=language, jit=lambda f: f, cdiv=lambda n, d: (n + d - 1) // d
    )
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "triton", triton)
        imports.setitem(sys.modules, "triton.language", language)
        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/weight_sync/gpu_delta/apply.py"
        )
        spec = importlib.util.spec_from_file_location("apply_host_under_test", path)
        module = importlib.util.module_from_spec(spec)
        imports.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_xor_batch", SimpleNamespace(warmup=warmup))
    monkeypatch.setattr(module, "_resident_ctas", lambda *args: 2)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    module.warmed, module.launched = warmed, launched
    return module


def linear(size):
    return size, (size,), (1,), (size,), (1,)


def scratch(size=1 << 24, address=4096):
    return SimpleNamespace(numel=lambda: size, data_ptr=lambda: address)


def byte_offset(index, shape, strides):
    offset = 0
    for length, stride in zip(reversed(shape), reversed(strides)):
        index, coordinate = divmod(index, length)
        offset += coordinate * stride
    return offset


def records(group):
    for program in group.programs:
        family = ("packed", "word_affine", "general").index(program.family)
        start, stop = program.metadata_start, program.metadata_stop
        table = group.metadata_rows[start:stop]
        assert start % 2 == stop % 2 == 0
        for index in range(table[0]):
            record = table[1 + table[0] + index]
            yield family, table[record:]


def test_changed_membership_reuses_modules_and_preserves_pointer_order(planner):
    m = planner
    affine = (420, (420,), (1,), (3, 5, 7, 4), (140, 4, 20, 1))
    reverse = (420, affine[3], affine[4], affine[1], affine[2])
    empty = (linear((1 << 63) - 1), 0)
    for contracts in (
        ((linear(17), 2),),
        ((affine, 2), (linear(8192), 1), (reverse, 1), empty),
        ((linear(4100), 3), (affine, 0)),
        (empty,),
    ):
        count = sum(n for _, n in contracts)
        group = m.ByteApplyGroup(
            [i * 16384 for i in range(count)],
            [0x100000 + i * 16384 for i in range(count)],
            contracts,
        )
        group.prepare_metadata(0)
        group.bind(scratch())
        expected, position = [], 0
        for _, n in contracts:
            if n:
                expected.append((position, n))
            position += 2 * n
        assert sorted((r[7], r[6]) for _, r in records(group) if r[6]) == sorted(
            expected
        )
        assert group.tiles == sum(n * math.ceil(c[0] / m.BLOCK) for c, n in contracts)
        assert len(group.metadata_rows) == group.metadata_size
        assert group.metadata_size % 2 == 0
        for program in group.programs:
            assert program.grid[0] == (
                program.tiles
                if program.family == "word_affine"
                else min(program.tiles, 4 * program.resident_ctas)
            )
    assert len(m.warmed) == 3
    # Working on other devices must not evict modules needed by existing ranks.
    original = tuple(m._compiled(0, f) for f in range(3))
    for device in range(1, 8):
        for family in range(3):
            m._compiled(device, family)
    assert tuple(m._compiled(0, f) for f in range(3)) == original
    assert len(m.warmed) == 24


@pytest.mark.parametrize(
    "shape,strides",
    [
        ((3, 1028), (1040, 1)),
        ((3, 5, 7, 4), (140, 4, 20, 1)),
        ((2, 3, 2048), (16384, 2052, 1)),
        ((2, 3, 1, 5, 4), (60, 4, 123, 12, 1)),
        ((1, 1, 4), ((1 << 63) - 1, 17, 1)),
    ],
)
def test_word_coordinates_match_original_bytes_and_alignment_fallback(
    planner, shape, strides
):
    m, size = planner, math.prod(shape)
    contract = (size, (size,), (1,), shape, strides)
    group = m.ByteApplyGroup([0], [0x100000], ((contract, 1),))
    group.prepare_metadata(0)
    initial = list(group.metadata_rows)
    capacity = group.metadata_size
    for address in (4096, 4097, 8192):
        aligned = group.bind(scratch(address=address))
        [(family, row)] = records(group)
        axes = row[10]
        begin = 11 + 2 * axes
        stored_shape = row[begin : begin + axes]
        stored_stride = row[begin + axes : begin + 2 * axes]
        unit = 4 if family == 1 else 1
        assert family == (1 if aligned else 2)
        assert math.prod(stored_shape) * unit == size
        for index in range(size // unit):
            assert unit * byte_offset(
                index, stored_shape, stored_stride
            ) == byte_offset(index * unit, shape, strides)
        assert len(group.metadata_rows) == capacity
    assert group.metadata_rows == initial


@pytest.mark.parametrize(
    "contract,count",
    [
        (linear(0), 1),
        (linear(1), -1),
        (linear(1), True),
        ((3, (3,), (-1,), (3,), (1,)), 1),
        ((3, (4,), (1,), (3,), (1,)), 1),
        ((1, (1,) * 65, (0,) * 65, (1,), (1,)), 1),
        ((1, (1,), (1 << 63,), (1,), (1,)), 1),
        (linear(2048), 1 << 31),
    ],
)
def test_invalid_byte_contract_rejected_before_kernel_warmup(planner, contract, count):
    with pytest.raises(ValueError):
        planner.ByteApplyGroup([0], [4096], ((contract, count),)).prepare_metadata(0)
    assert planner.warmed == []


def test_source_bounds_empty_plans_and_missing_destinations(planner):
    assert planner.plan_apply([]) is None
    with pytest.raises(ValueError, match="affine byte destinations"):
        planner.plan_apply([(SimpleNamespace(destinations=()), 0, 16)])
    assert not planner.warmed and not planner.launched

    m = planner
    group = m.ByteApplyGroup([16], [4096], ((linear(17), 1),))
    group.prepare_metadata(0)
    assert group.prepare_metadata(0)
    with pytest.raises(ValueError, match="exceeds decoded"):
        group.bind(scratch(32))
    group.bind(scratch(33))
    for address in (0, -1, True, (1 << 63) - 16):
        with pytest.raises(ValueError, match="source pointer"):
            group.bind(scratch(33, address))
    for sources, targets in (([], [4096]), ([-1], [4096]), ([0], [0])):
        with pytest.raises(ValueError):
            m.ByteApplyGroup(sources, targets, ((linear(17), 1),)).prepare_metadata(0)
    empty = m.ByteApplyGroup([], [], ((linear((1 << 63) - 1), 0),))
    assert not empty.prepare_metadata(0)
    assert empty.prepare_metadata(0)
    assert empty.bind(scratch(0, 0))
    assert empty.bind_launch(None, None, None) is None


def test_aligned_bind_uses_immutable_template_without_replanning(planner, monkeypatch):
    m = planner
    contract = [4100, [4100], [1], [4100], [1]]
    sources, targets = [0], [4096]
    group = m.ByteApplyGroup(sources, targets, [(contract, 1)])
    assert not group.prepare_metadata(0)
    initial = list(group.metadata_rows)
    contract[1][0], sources[0], targets[0] = 1, 99, 101
    assert not group.bind(scratch(address=4097))
    assert group.prepare_metadata(0)
    assert group.bind(scratch(address=8192))
    assert group.metadata_rows == initial
    assert list(group.pointer_rows(8192)) == [8192, 4096]
    state = group.metadata_rows, group.programs

    def forbidden(*args):
        pytest.fail("aligned paused bind rebuilt a prepared plan")

    monkeypatch.setattr(group, "_plan_binding", forbidden)
    monkeypatch.setattr(group, "pointer_rows", forbidden)
    monkeypatch.setattr(m, "_validate_contract", forbidden)
    monkeypatch.setattr(m, "_compiled", forbidden)
    monkeypatch.setattr(m, "_resident_ctas", forbidden)
    for address in (16384, 32768):
        assert group.prepare_metadata(0)
        assert group.bind(scratch(address=address))
        assert all(a is b for a, b in zip(state, (group.metadata_rows, group.programs)))


def test_metadata_cache_rebuilds_for_changed_device_and_rejects_invalid_devices(
    planner, monkeypatch
):
    monkeypatch.setattr(
        planner, "_resident_ctas", lambda kernel, device: 2 * (device + 1)
    )
    group = planner.ByteApplyGroup([0], [4096], ((linear(81920), 1),))
    group.prepare_metadata(0)
    original_kernels = group.kernels
    assert not group.bind(scratch(address=4097))
    assert not group.prepare_metadata(1)
    assert group.kernels != original_kernels
    assert group.programs[0].grid == (16, 1, 1)
    assert group.prepare_metadata(1)
    assert not group.prepare_metadata(0)
    assert group.kernels == original_kernels
    assert group.programs[0].grid == (8, 1, 1)
    assert group.bind(scratch())
    assert len(planner.warmed) == 6

    for device in (-1, False, 0.0, None):
        group = planner.ByteApplyGroup([0], [4096], ((linear(4100), 1),))
        group.prepare_metadata(0)
        with pytest.raises(ValueError, match="invalid GPU delta apply device"):
            group.prepare_metadata(device)
        with pytest.raises(RuntimeError, match="must be prepared"):
            group.bind(scratch())
        assert not group.prepare_metadata(0)


def test_failed_metadata_preparation_cannot_bind_and_is_retryable(planner, monkeypatch):
    group = planner.ByteApplyGroup([0], [4096], ((linear(4100), 1),))
    group.prepare_metadata(0)

    def failed_plan(*args):
        raise RuntimeError("injected planning failure")

    with monkeypatch.context() as patch:
        patch.setattr(group, "_plan_binding", failed_plan)
        with pytest.raises(RuntimeError, match="injected planning failure"):
            group.prepare_metadata(1)
        with pytest.raises(RuntimeError, match="must be prepared"):
            group.bind(scratch())
    assert not group.prepare_metadata(0)
    assert group.prepare_metadata(0)
    assert group.bind(scratch())


def owner(groups):
    value = PreparedDelta.__new__(PreparedDelta)
    value.static_plans = [(None, None, g) for g in groups]
    value.apply_bindings, value.apply_source_patches = [], []
    value.apply_fallback_groups = set()
    size = sum(2 * len(g.sources) + g.metadata_size for g in groups if g is not None)
    # Real CPU tensors exercise the exact NumPy slab patch and tensor-view leases.
    value.apply_host_metadata = torch.empty(size, dtype=torch.int64)
    value.apply_metadata = torch.empty(size, dtype=torch.int64)
    value.error = torch.zeros(1, dtype=torch.int32)
    value.batches, value.status_checks, value.raw_copies = [], [], {}
    value._prepare_apply_bindings()
    return value


def test_cached_preparation_restores_fallback_before_fresh_owner(planner):
    contract = (420, (420,), (1,), (3, 5, 7, 4), (140, 4, 20, 1))
    group = planner.ByteApplyGroup([0], [4096], ((contract, 1),))
    group.prepare_metadata(0)
    aligned = tuple(group.metadata_rows)
    first = owner([group])
    first.decoded = [scratch(address=4097), scratch()]
    first._bind_apply_metadata()
    assert group.programs[0].family == "general"

    assert group.prepare_metadata(0)
    assert tuple(group.metadata_rows) == aligned
    assert group.programs[0].family == "word_affine"
    second = owner([group])
    assert second.apply_metadata is not first.apply_metadata
    assert second.error is not first.error
    first_refs = weakref.ref(first.apply_metadata), weakref.ref(first.error)
    first._release_gpu()
    gc.collect()
    assert all(ref() is None for ref in first_refs)

    second.decoded = [scratch(address=8192), scratch()]
    _, calls = second._bind_apply_metadata()
    assert calls[0] is second.apply_bindings[0][-1]
    start, stop, *_ = second.apply_bindings[0]
    assert second.apply_host_view[start:stop].tolist() == list(aligned)
    del calls, _
    second_refs = weakref.ref(second.apply_metadata), weakref.ref(second.error)
    second._release_gpu()
    gc.collect()
    assert all(ref() is None for ref in second_refs)


def test_slab_patch_restores_fallback_and_releases_prebound_views(planner):
    m = planner
    affine = (420, (420,), (1,), (3, 5, 7, 4), (140, 4, 20, 1))
    g = m.ByteApplyGroup([0, 8192], [4096, 16384], ((linear(4100), 1), (affine, 1)))
    h = m.ByteApplyGroup([0], [65537], ((linear(17), 1),))
    for group in (g, h):
        group.prepare_metadata(0)
    prepared = owner([g, None, h])
    original = prepared.apply_host_view.copy()
    original_calls = [row[-1] if row else None for row in prepared.apply_bindings]
    for origin in (4096, 4097, 8192):
        prepared.decoded = [scratch(address=origin), scratch(address=origin + 4096)]
        _, calls = prepared._bind_apply_metadata()
        position = 0
        for index, group in enumerate((g, None, h)):
            if group is None:
                assert calls[index] is None
                continue
            middle = position + 2 * len(group.sources)
            stop = middle + group.metadata_size
            assert prepared.apply_host_view[position:middle].tolist() == list(
                group.pointer_rows(prepared.decoded[index % 2].data_ptr())
            )
            assert prepared.apply_host_view[middle:stop].tolist() == group.metadata_rows
            assert middle % 2 == position % 2 == 0
            position = stop
        if origin % 16 == 0:
            assert calls == original_calls
    assert not prepared.apply_fallback_groups
    for start, stop, *_ in (row for row in prepared.apply_bindings if row):
        assert (
            prepared.apply_host_view[start:stop].tolist()
            == original[start:stop].tolist()
        )
    del _  # the loop temporary above held tensor views
    before = prepared.apply_host_view.copy()
    prepared.decoded = [scratch(1), scratch()]
    with pytest.raises(ValueError, match="exceeds decoded"):
        prepared._bind_apply_metadata()
    assert prepared.apply_host_view.tolist() == before.tolist()
    # Bound closures own tensor views until explicit release; the cached group
    # must not retain those per-update tensors afterward.
    host_ref = weakref.ref(prepared.apply_host_metadata)
    gpu_ref = weakref.ref(prepared.apply_metadata)
    del calls, original_calls
    prepared._release_gpu()
    gc.collect()
    assert host_ref() is None and gpu_ref() is None


def test_prebound_launch_default_custom_grids_and_error_gate_identity(planner):
    group = planner.ByteApplyGroup([0], [4096], ((linear(4100), 1),))
    group.prepare_metadata(0)
    pointers, metadata, error = object(), list(group.metadata_rows), object()
    default = group.bind_launch(pointers, metadata, error)
    custom = group.bind_launch(pointers, metadata, error, ((1, 1, 1),))
    assert planner.launched == []
    default()
    custom()
    assert [c[0] for c in planner.launched] == [group.programs[0].grid, (1, 1, 1)]
    assert all(c[1] is pointers and c[3] is error for c in planner.launched)
    for grids in (
        (),
        ((0, 1, 1),),
        ((True, 1, 1),),
        ((1, 2, 1),),
        ((1, 1),),
        ((1 << 31, 1, 1),),
    ):
        with pytest.raises(ValueError, match="family launch grids"):
            group.bind_launch(pointers, metadata, error, grids)
