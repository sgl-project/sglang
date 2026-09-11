"""CPU unit tests for ``graph_serialization.regions`` and ``FixedArena``.

Everything under test is pure Python: bisect classification over live ranges,
relocation arithmetic, the shared ``RegionPlacementPolicy.build`` rules, the
fixed-VA policy's collision degradation, and ``FixedArena``'s reservation and
commit bookkeeping. The CUDA driver is replaced by fakes (a recording
``VmmReservation`` stand-in, a fake ``BumpArenaStub`` and ``torch.cuda.MemPool``),
so no GPU is touched.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.graph_serialization import regions
from sglang.srt.model_executor.graph_serialization.format import (
    RegionKind,
    RegionRef,
    RegionSpec,
)
from sglang.srt.model_executor.graph_serialization.regions import (
    AttnWorkspaceProvider,
    CommTableProvider,
    CublasWorkspaceProvider,
    Disposition,
    FixedVaArenaPolicy,
    GraphPoolProvider,
    KVPoolProvider,
    RegionRegistry,
    RegionRejected,
    RelocatePolicy,
    RelocationMap,
    StaticBufferProvider,
    WeightsProvider,
)
from sglang.srt.utils import cuda_vmm_utils
from sglang.srt.utils.cuda_vmm_utils import FixedArena, FixedArenaCollision
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MiB = 1 << 20


class FakeProvider:
    """A ``RegionProvider`` whose regions are given up front and can be mutated
    between ``refresh`` calls."""

    def __init__(self, kind, regions_and_bases):
        self.kind = kind
        self.regions_and_bases = list(regions_and_bases)
        self.rebound_with = []

    def enumerate(self):
        return list(self.regions_and_bases)

    def rebind_data(self, reloc):
        self.rebound_with.append(reloc)


def _spec(region_id, kind, nbytes, base_at_save=0):
    return RegionSpec(
        region_id=region_id, kind=kind, nbytes=nbytes, base_at_save=base_at_save
    )


def _registry(*entries):
    """``entries`` are ``(region_id, kind, nbytes, live_base)``."""
    by_kind = {}
    for region_id, kind, nbytes, base in entries:
        by_kind.setdefault(kind, []).append((_spec(region_id, kind, nbytes), base))
    providers = [FakeProvider(kind, rb) for kind, rb in by_kind.items()]
    return RegionRegistry(providers), providers


# ---------------------------------------------------------------------------
# RegionRegistry.classify
# ---------------------------------------------------------------------------


def test_classify_hits_at_base_and_interior_offsets():
    registry, _ = _registry(
        ("weight:storage:0", "weight", 0x100, 0x1000),
        ("kv:k_buffer:0", "kv", 0x200, 0x2000),
    )
    assert registry.classify(0x1000) == RegionRef("weight:storage:0", 0)
    assert registry.classify(0x1080) == RegionRef("weight:storage:0", 0x80)
    assert registry.classify(0x10FF) == RegionRef("weight:storage:0", 0xFF)
    assert registry.classify(0x2000) == RegionRef("kv:k_buffer:0", 0)
    assert registry.classify(0x21FF) == RegionRef("kv:k_buffer:0", 0x1FF)


def test_classify_misses_return_none():
    registry, _ = _registry(
        ("weight:storage:0", "weight", 0x100, 0x1000),
        ("kv:k_buffer:0", "kv", 0x200, 0x2000),
    )
    assert registry.classify(0) is None
    assert registry.classify(0x0FFF) is None  # one below the first base
    assert registry.classify(0x1100) is None  # end is exclusive
    assert registry.classify(0x1FFF) is None  # gap between regions
    assert registry.classify(0x2200) is None  # end of the last region
    assert registry.classify(0x400000) is None  # slot_bytes scalar (fact 17)
    assert registry.classify(-8) is None


def test_classify_on_empty_registry_is_none():
    registry = RegionRegistry([])
    assert registry.classify(0x1000) is None
    assert registry.snapshot() == ()
    assert len(registry) == 0


def test_overlapping_registrations_raise_at_refresh():
    with pytest.raises(ValueError, match="overlap"):
        _registry(
            ("a", "weight", 0x100, 0x1000),
            ("b", "kv", 0x100, 0x10F0),
        )


def test_overlap_error_names_both_regions_with_their_ranges():
    # Sorted by base: ``big`` then ``small`` (inside ``big``) then ``late``. The
    # first offending pair is reported, with both ids and both ranges.
    with pytest.raises(ValueError, match="'big'.*0x1000, 0x2000.*'small'.*0x1100"):
        _registry(
            ("big", "weight", 0x1000, 0x1000),
            ("small", "kv", 0x10, 0x1100),
            ("late", "static", 0x10, 0x1800),
        )


def test_adjacent_regions_do_not_overlap():
    registry, _ = _registry(
        ("a", "weight", 0x100, 0x1000),
        ("b", "kv", 0x100, 0x1100),
    )
    assert registry.classify(0x10FF) == RegionRef("a", 0xFF)
    assert registry.classify(0x1100) == RegionRef("b", 0)


def test_duplicate_region_id_and_empty_region_raise():
    with pytest.raises(ValueError, match="registered twice"):
        RegionRegistry(
            [
                FakeProvider("weight", [(_spec("dup", "weight", 8), 0x1000)]),
                FakeProvider("kv", [(_spec("dup", "kv", 8), 0x2000)]),
            ]
        )
    with pytest.raises(ValueError, match="non-empty"):
        RegionRegistry([FakeProvider("kv", [(_spec("z", "kv", 0), 0x1000)])])


def test_refresh_reenumerates_providers():
    registry, (provider,) = _registry(("a", "weight", 0x100, 0x1000))
    assert registry.live_base("a") == 0x1000
    provider.regions_and_bases = [(_spec("a", "weight", 0x100), 0x9000)]
    assert registry.classify(0x9000) is None  # stale until refresh
    registry.refresh()
    assert registry.live_base("a") == 0x9000
    assert registry.classify(0x9010) == RegionRef("a", 0x10)
    assert registry.classify(0x1000) is None


def test_snapshot_keeps_enumeration_order_and_fills_base_at_save():
    registry, _ = _registry(
        ("z", "weight", 0x100, 0x5000),
        ("a", "weight", 0x100, 0x1000),
    )
    snap = registry.snapshot()
    assert [s.region_id for s in snap] == ["z", "a"]
    assert [s.base_at_save for s in snap] == [0x5000, 0x1000]
    assert "a" in registry and "missing" not in registry
    with pytest.raises(KeyError):
        registry.live_base("missing")


def test_rebind_data_fans_out_to_providers_and_tolerates_missing_hook():
    registry, providers = _registry(
        ("w", "weight", 0x100, 0x1000),
        ("k", "kv", 0x100, 0x2000),
    )
    bare = SimpleNamespace(kind="misc", enumerate=lambda: [])
    registry = RegionRegistry([*providers, bare])
    reloc = RelocationMap()
    registry.rebind_data(reloc)
    for provider in providers:
        assert provider.rebound_with == [reloc]


# ---------------------------------------------------------------------------
# RelocationMap
# ---------------------------------------------------------------------------


def test_relocation_map_rebase_and_errors():
    reloc = RelocationMap(
        deltas={"a": 0x100, "p": 0},
        live_bases={"a": 0x1100, "p": 0x7000},
        rejected={"r": "kind 'misc' is rejected"},
    )
    assert reloc.rebase(RegionRef("a", 0x20)) == 0x1120
    assert reloc.rebase(RegionRef("p", 0)) == 0x7000
    assert reloc.is_rejected("r") and not reloc.is_rejected("a")
    assert reloc.accepted == frozenset({"a", "p"})
    with pytest.raises(RegionRejected, match="'r' was rejected"):
        reloc.rebase(RegionRef("r", 0))
    with pytest.raises(KeyError):
        reloc.rebase(RegionRef("unknown", 0))


def test_relocation_map_defaults_are_independent_instances():
    a, b = RelocationMap(), RelocationMap()
    a.deltas["x"] = 1
    assert b.deltas == {}


# ---------------------------------------------------------------------------
# RelocatePolicy
# ---------------------------------------------------------------------------


def test_relocate_policy_surface():
    policy = RelocatePolicy()
    assert policy.name == "relocate"
    assert policy.reserve(0) is None
    assert policy.mem_pool_for("static") is None
    assert policy.graph_pool_id() is None
    for kind in RegionKind:
        assert policy.disposition(kind.value) is Disposition.RELOCATE
    assert policy.disposition("not-a-kind") is Disposition.REJECT


def test_relocate_policy_build_computes_deltas():
    saved = [
        _spec("w", "weight", 0x100, base_at_save=0x1000),
        _spec("k", "kv", 0x200, base_at_save=0x9000),
    ]
    live = {
        "w": (_spec("w", "weight", 0x100), 0x1400),
        "k": (_spec("k", "kv", 0x200), 0x8000),
    }
    reloc = RelocatePolicy().build(saved, live)
    assert reloc.deltas == {"w": 0x400, "k": -0x1000}
    assert reloc.live_bases == {"w": 0x1400, "k": 0x8000}
    assert reloc.rejected == {}
    assert reloc.rebase(RegionRef("k", 0x10)) == 0x8010


def test_relocate_policy_build_reject_cases():
    saved = [
        _spec("unknown-kind", "bogus", 0x100, base_at_save=0x1000),
        _spec("missing", "weight", 0x100, base_at_save=0x2000),
        _spec("resized", "kv", 0x100, base_at_save=0x3000),
        _spec("rekinded", "static", 0x100, base_at_save=0x4000),
        _spec("ok", "pool", 0x100, base_at_save=0x5000),
    ]
    live = {
        "unknown-kind": (_spec("unknown-kind", "bogus", 0x100), 0x1000),
        "resized": (_spec("resized", "kv", 0x180), 0x3000),
        "rekinded": (_spec("rekinded", "attn_ws", 0x100), 0x4000),
        "ok": (_spec("ok", "pool", 0x100), 0x5100),
    }
    reloc = RelocatePolicy().build(saved, live)
    assert set(reloc.rejected) == {"unknown-kind", "missing", "resized", "rekinded"}
    assert "rejected by placement policy 'relocate'" in reloc.rejected["unknown-kind"]
    assert "no live region" in reloc.rejected["missing"]
    assert "saved 256 bytes, live 384 bytes" in reloc.rejected["resized"]
    assert "saved 'static', live 'attn_ws'" in reloc.rejected["rekinded"]
    assert reloc.deltas == {"ok": 0x100}
    for rid in reloc.rejected:
        with pytest.raises(RegionRejected):
            reloc.rebase(RegionRef(rid, 0))


def test_registry_relocation_for_uses_live_table():
    registry, _ = _registry(("w", "weight", 0x100, 0x1400))
    saved = [_spec("w", "weight", 0x100, base_at_save=0x1000)]
    reloc = registry.relocation_for(saved, RelocatePolicy())
    assert reloc.deltas == {"w": 0x400}
    # A snapshot of the live registry relocates onto itself with zero deltas.
    identity = registry.relocation_for(registry.snapshot(), RelocatePolicy())
    assert identity.deltas == {"w": 0}


# ---------------------------------------------------------------------------
# FixedVaArenaPolicy
# ---------------------------------------------------------------------------

_STATIC_BASE = 0x6000_0000_0000
_POOL_BASE = 0x6100_0000_0000


def _fixed_policy(**kwargs):
    return FixedVaArenaPolicy(
        arena_bases={"static": _STATIC_BASE, "pool": _POOL_BASE},
        arena_sizes={"static": 64 * MiB, "pool": 256 * MiB},
        **kwargs,
    )


def test_fixed_va_policy_constructor_validation():
    with pytest.raises(ValueError, match="same kinds"):
        FixedVaArenaPolicy(arena_bases={"static": 1}, arena_sizes={"pool": 1})
    with pytest.raises(ValueError, match="unknown region kinds"):
        FixedVaArenaPolicy(arena_bases={"bogus": 1}, arena_sizes={"bogus": 1})
    with pytest.raises(ValueError, match="positive base and size"):
        FixedVaArenaPolicy(arena_bases={"static": 0}, arena_sizes={"static": 1})


def test_fixed_va_policy_dispositions_before_reserve():
    policy = _fixed_policy()
    assert policy.name == "fixed_va"
    assert policy.effective_placement == {
        "static": Disposition.PIN,
        "pool": Disposition.PIN,
    }
    assert policy.pinned_kinds == ("pool", "static")
    assert policy.disposition("static") is Disposition.PIN
    assert policy.disposition("pool") is Disposition.PIN
    assert policy.disposition("weight") is Disposition.RELOCATE
    assert policy.disposition("bogus") is Disposition.REJECT
    assert policy.mem_pool_for("static") is None
    assert policy.graph_pool_id() is None


def test_fixed_va_policy_build_pin_match_and_mismatch():
    policy = _fixed_policy()
    saved = [
        _spec("static:input_ids", "static", 0x1000, base_at_save=_STATIC_BASE),
        _spec("pool:seg:0", "pool", 0x2000, base_at_save=_POOL_BASE),
        _spec("weight:storage:0", "weight", 0x100, base_at_save=0x1000),
    ]
    live = {
        "static:input_ids": (
            _spec("static:input_ids", "static", 0x1000),
            _STATIC_BASE,
        ),
        "pool:seg:0": (_spec("pool:seg:0", "pool", 0x2000), _POOL_BASE + 0x200000),
        "weight:storage:0": (_spec("weight:storage:0", "weight", 0x100), 0x1800),
    }
    reloc = policy.build(saved, live)
    # PIN match: delta 0, live base recorded.
    assert reloc.deltas["static:input_ids"] == 0
    assert reloc.live_bases["static:input_ids"] == _STATIC_BASE
    # PIN mismatch: rejected with both addresses in the reason.
    reason = reloc.rejected["pool:seg:0"]
    assert f"{_POOL_BASE + 0x200000:#x}" in reason
    assert f"{_POOL_BASE:#x}" in reason
    assert "pool:seg:0" not in reloc.deltas
    # Non-pinned kinds relocate as usual.
    assert reloc.deltas["weight:storage:0"] == 0x800


def test_fixed_va_policy_pin_is_asserted_by_name_not_by_address():
    # A live region at the saved address but under a different name is a
    # miss for the saved name, not a match.
    policy = _fixed_policy()
    saved = [_spec("static:a", "static", 0x100, base_at_save=_STATIC_BASE)]
    live = {"static:b": (_spec("static:b", "static", 0x100), _STATIC_BASE)}
    reloc = policy.build(saved, live)
    assert "no live region" in reloc.rejected["static:a"]


class _FakeArena:
    instances = []

    def __init__(self, *, device_id, size, requested_address, name="", **_):
        if "pool" in name:
            raise FixedArenaCollision(
                f"requested {requested_address:#x} but got {requested_address + 1:#x}"
            )
        self.device_id = device_id
        self.size = size
        self.requested_address = requested_address
        self.name = name
        self.closed = False
        self.pool = SimpleNamespace(id=(7, 3))
        _FakeArena.instances.append(self)

    def mem_pool(self):
        return self.pool

    def close(self):
        self.closed = True


def test_fixed_va_policy_reserve_degrades_colliding_kind(monkeypatch, caplog):
    _FakeArena.instances.clear()
    monkeypatch.setattr(regions, "FixedArena", _FakeArena)
    policy = _fixed_policy()
    with caplog.at_level(logging.WARNING, logger=regions.__name__):
        policy.reserve(device_id=1)

    assert policy.effective_placement == {
        "static": Disposition.PIN,
        "pool": Disposition.RELOCATE,
    }
    assert policy.pinned_kinds == ("static",)
    assert policy.disposition("static") is Disposition.PIN
    assert policy.disposition("pool") is Disposition.RELOCATE
    assert any("degrading that kind to RELOCATE" in r.message for r in caplog.records)
    assert any("'pool'" in r.message for r in caplog.records)

    # The surviving arena answers mem_pool_for; the degraded kind does not, and
    # graph_pool_id is None because the pool arena is the one that collided.
    (static_arena,) = _FakeArena.instances
    assert static_arena.requested_address == _STATIC_BASE
    assert static_arena.size == 64 * MiB
    assert static_arena.device_id == 1
    assert policy.mem_pool_for("static") is static_arena.pool
    assert policy.mem_pool_for("pool") is None
    assert policy.mem_pool_for("weight") is None
    assert policy.graph_pool_id() is None

    # A PIN on the degraded kind now relocates instead of asserting the base.
    saved = [_spec("pool:seg:0", "pool", 0x1000, base_at_save=_POOL_BASE)]
    live = {"pool:seg:0": (_spec("pool:seg:0", "pool", 0x1000), _POOL_BASE + 0x10)}
    assert policy.build(saved, live).deltas == {"pool:seg:0": 0x10}

    with pytest.raises(RuntimeError, match="called twice"):
        policy.reserve(device_id=1)
    policy.close()
    assert static_arena.closed


def test_fixed_va_policy_graph_pool_id_comes_from_pool_arena(monkeypatch):
    class _PoolOnlyArena(_FakeArena):
        def __init__(self, **kwargs):
            kwargs["name"] = kwargs["name"].replace("pool", "P")
            super().__init__(**kwargs)

    _FakeArena.instances.clear()
    monkeypatch.setattr(regions, "FixedArena", _PoolOnlyArena)
    policy = FixedVaArenaPolicy(
        arena_bases={"pool": _POOL_BASE}, arena_sizes={"pool": 8 * MiB}, device_id=2
    )
    policy.reserve(2)
    assert policy.graph_pool_id() == (7, 3)
    with pytest.raises(ValueError, match="device 2"):
        FixedVaArenaPolicy(
            arena_bases={"pool": _POOL_BASE}, arena_sizes={"pool": 8 * MiB}, device_id=2
        ).reserve(3)


# ---------------------------------------------------------------------------
# v1 provider skeletons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider, kind",
    [
        (WeightsProvider(model=object()), RegionKind.WEIGHT),
        (
            KVPoolProvider(token_to_kv_pool=object(), req_to_token_pool=object()),
            RegionKind.KV,
        ),
        (StaticBufferProvider(buffer_registry=object()), RegionKind.STATIC),
        (AttnWorkspaceProvider(attn_backends=[object()]), RegionKind.ATTN_WS),
        (GraphPoolProvider(pool_id=(0, 1)), RegionKind.POOL),
        (CublasWorkspaceProvider(), RegionKind.CUBLAS_WS),
        (CommTableProvider(states=[]), RegionKind.COMM_TABLE),
    ],
)
def test_provider_skeletons_declare_kind_and_stub_enumerate(provider, kind):
    assert provider.kind == kind.value
    assert type(provider).kind == kind.value
    with pytest.raises(NotImplementedError) as info:
        provider.enumerate()
    message = str(info.value)
    assert message.startswith(f"{type(provider).__name__}.enumerate:")
    assert "not implemented in this draft" in message
    assert "DESIGN_cuda_graph_serialization.md section 7.1" in message


def test_provider_rebind_data_defaults():
    reloc = RelocationMap()
    for provider in (
        WeightsProvider(model=object()),
        StaticBufferProvider(),
        AttnWorkspaceProvider(attn_backends=()),
        GraphPoolProvider(pool_id=(0, 0)),
        CublasWorkspaceProvider(),
        CommTableProvider(states=()),
    ):
        assert provider.rebind_data(reloc) is None
    with pytest.raises(NotImplementedError, match="KVPoolProvider.rebind_data"):
        KVPoolProvider(token_to_kv_pool=None, req_to_token_pool=None).rebind_data(reloc)


# ---------------------------------------------------------------------------
# FixedArena (cuda_vmm_utils) with a fake VmmReservation
# ---------------------------------------------------------------------------

_GRANULARITY = 2 * MiB
_ARENA_BASE = 0x6200_0000_0000


class _FakeReservation:
    """Records the ``VmmReservation`` calls ``FixedArena`` makes."""

    instances = []
    # Set by a test to make the fake report a different base (collision).
    base_override = None

    def __init__(self, size, prop, device_id, *, alignment=0, requested_address=0):
        self.size = size
        self.prop = prop
        self.device_id = device_id
        self.alignment = alignment
        self.requested_address = requested_address
        self.base = (
            requested_address
            if _FakeReservation.base_override is None
            else _FakeReservation.base_override
        )
        self.maps = []
        self.existing = []
        self.closed = False
        self._next_handle = 100
        _FakeReservation.instances.append(self)

    def map(self, offset, size, *, retain_handle):
        assert retain_handle is True
        handle = self._next_handle
        self._next_handle += 1
        self.maps.append((offset, size, handle))
        return handle

    def map_existing(self, offset, size, handle):
        self.existing.append((offset, size, handle))

    def close(self, *, release_handles=True):
        self.closed = True


@pytest.fixture
def fake_vmm(monkeypatch):
    _FakeReservation.instances.clear()
    _FakeReservation.base_override = None
    prop = object()
    monkeypatch.setattr(cuda_vmm_utils, "VmmReservation", _FakeReservation)
    monkeypatch.setattr(
        cuda_vmm_utils, "make_device_allocation_prop", lambda device_id: prop
    )
    monkeypatch.setattr(
        cuda_vmm_utils, "get_device_granularity", lambda device_id: _GRANULARITY
    )
    return prop


def test_fixed_arena_collision_closes_reservation_and_raises(fake_vmm):
    _FakeReservation.base_override = _ARENA_BASE + _GRANULARITY
    with pytest.raises(FixedArenaCollision) as info:
        FixedArena(
            device_id=0, size=8 * MiB, requested_address=_ARENA_BASE, name="static"
        )
    (reservation,) = _FakeReservation.instances
    assert reservation.closed is True
    message = str(info.value)
    assert f"requested {_ARENA_BASE:#x}" in message
    assert f"returned {_ARENA_BASE + _GRANULARITY:#x}" in message
    assert "FixedArena[static]" in message


def test_fixed_arena_reserves_with_granularity_alignment_and_requested_base(
    fake_vmm,
):
    arena = FixedArena(
        device_id=3, size=5 * MiB, requested_address=_ARENA_BASE, name="pool"
    )
    (reservation,) = _FakeReservation.instances
    assert reservation.prop is fake_vmm
    assert reservation.device_id == 3
    assert reservation.alignment == _GRANULARITY
    assert reservation.requested_address == _ARENA_BASE
    # Size is rounded up to whole chunks (chunk = max(2 MiB, granularity)).
    assert reservation.size == 6 * MiB
    assert arena.size == 6 * MiB
    assert arena.base == _ARENA_BASE
    assert arena.chunk_bytes == 2 * MiB
    assert arena.granularity == _GRANULARITY
    assert arena.name == "pool"
    # Never eagerly committed.
    assert reservation.maps == []
    assert arena.committed_bytes == 0
    assert arena.committed_extents() == []


def test_fixed_arena_commit_is_chunk_aligned_lazy_and_idempotent(fake_vmm):
    arena = FixedArena(device_id=0, size=16 * MiB, requested_address=_ARENA_BASE)
    (reservation,) = _FakeReservation.instances

    # A sub-chunk request commits exactly the chunk that contains it.
    arena.commit(0x1000, 0x100)
    assert reservation.maps == [(0, 2 * MiB, 100)]
    assert arena.committed_bytes == 2 * MiB

    # Overlapping the committed chunk again maps nothing new.
    arena.commit(0, 2 * MiB)
    arena.commit(0x2000, 0x10)
    assert len(reservation.maps) == 1

    # A range straddling chunks 0..2 commits only the missing chunks 1 and 2.
    arena.commit(2 * MiB - 8, 2 * MiB + 16)
    assert [(o, s) for o, s, _ in reservation.maps] == [
        (0, 2 * MiB),
        (2 * MiB, 2 * MiB),
        (4 * MiB, 2 * MiB),
    ]
    assert arena.committed_bytes == 6 * MiB
    assert arena.committed_extents() == [(_ARENA_BASE, 6 * MiB)]

    # A disjoint chunk yields a second extent.
    arena.commit(10 * MiB, 1)
    assert arena.committed_extents() == [
        (_ARENA_BASE, 6 * MiB),
        (_ARENA_BASE + 10 * MiB, 2 * MiB),
    ]
    assert arena.commit_log[0] == (0x1000, 0x100)
    assert len(arena.commit_log) == 5


def test_fixed_arena_commit_range_validation(fake_vmm):
    arena = FixedArena(device_id=0, size=4 * MiB, requested_address=_ARENA_BASE)
    with pytest.raises(ValueError, match="outside the arena"):
        arena.commit(-1, 8)
    with pytest.raises(ValueError, match="outside the arena"):
        arena.commit(0, 0)
    with pytest.raises(ValueError, match="outside the arena"):
        arena.commit(4 * MiB - 8, 16)
    assert arena.committed_bytes == 0


def test_fixed_arena_constructor_validation(fake_vmm):
    with pytest.raises(ValueError, match="size must be positive"):
        FixedArena(device_id=0, size=0, requested_address=_ARENA_BASE)
    with pytest.raises(ValueError, match="requested_address must be a positive"):
        FixedArena(device_id=0, size=MiB, requested_address=0)
    with pytest.raises(ValueError, match="not aligned to the allocation granularity"):
        FixedArena(device_id=0, size=MiB, requested_address=_ARENA_BASE + 0x1000)
    with pytest.raises(ValueError, match="chunk_bytes must be positive"):
        FixedArena(device_id=0, size=MiB, requested_address=_ARENA_BASE, chunk_bytes=0)
    # No reservation is made when validation fails before it.
    assert _FakeReservation.instances == []
    # A chunk smaller than the granularity is rounded up to it.
    arena = FixedArena(
        device_id=0, size=MiB, requested_address=_ARENA_BASE, chunk_bytes=4096
    )
    assert arena.chunk_bytes == _GRANULARITY


def test_fixed_arena_mem_pool_is_built_once_over_committed_extents(
    fake_vmm, monkeypatch
):
    import torch

    class _FakeStub:
        instances = []

        def __init__(self):
            self.extents = None
            self.align = None
            self.allocator = object()
            _FakeStub.instances.append(self)

        def set_extents(self, extents):
            self.extents = list(extents)

        def set_align(self, nbytes):
            self.align = nbytes

    pools = []

    class _FakeMemPool:
        def __init__(self, allocator, *, no_split=False):
            self.allocator = allocator
            self.no_split = no_split
            self.id = (1, len(pools))
            pools.append(self)

    monkeypatch.setattr(cuda_vmm_utils, "BumpArenaStub", _FakeStub)
    monkeypatch.setattr(torch.cuda, "MemPool", _FakeMemPool)

    arena = FixedArena(device_id=0, size=16 * MiB, requested_address=_ARENA_BASE)
    with pytest.raises(RuntimeError, match="commit\\(\\) the extents"):
        arena.mem_pool()

    arena.commit(0, 4 * MiB)
    arena.commit(8 * MiB, 2 * MiB)
    pool = arena.mem_pool()
    (stub,) = _FakeStub.instances
    assert stub.extents == [
        (_ARENA_BASE, 4 * MiB),
        (_ARENA_BASE + 8 * MiB, 2 * MiB),
    ]
    assert stub.align == _GRANULARITY
    assert pool.allocator is stub.allocator
    assert pool.no_split is True  # fact 13
    assert arena.mem_pool() is pool
    assert len(pools) == 1

    # Growing the arena after the pool exists is refused: set_extents would
    # reset the bump cursors.
    arena.commit(12 * MiB, 1)
    with pytest.raises(RuntimeError, match="committed extents changed"):
        arena.mem_pool()


def test_fixed_arena_map_existing_chunks_and_export_stub(fake_vmm):
    arena = FixedArena(device_id=0, size=8 * MiB, requested_address=_ARENA_BASE)
    (reservation,) = _FakeReservation.instances
    handle = object()
    arena.map_existing_chunks([(2 * MiB, 4 * MiB, handle)])
    assert reservation.existing == [(2 * MiB, 4 * MiB, handle)]
    assert reservation.maps == []
    assert arena.committed_bytes == 4 * MiB
    assert arena.committed_extents() == [(_ARENA_BASE + 2 * MiB, 4 * MiB)]
    # commit() over an imported chunk is a no-op for that chunk.
    arena.commit(2 * MiB, 4 * MiB)
    assert reservation.maps == []
    with pytest.raises(ValueError, match="already backed"):
        arena.map_existing_chunks([(4 * MiB, 2 * MiB, object())])
    with pytest.raises(ValueError, match="not aligned to chunk_bytes"):
        arena.map_existing_chunks([(MiB, 2 * MiB, object())])
    with pytest.raises(ValueError, match="outside the arena"):
        arena.map_existing_chunks([(6 * MiB, 4 * MiB, object())])
    with pytest.raises(NotImplementedError, match="FixedArena.export_chunks"):
        arena.export_chunks()


def test_fixed_arena_close_releases_reservation_and_refuses_further_use(fake_vmm):
    arena = FixedArena(device_id=0, size=4 * MiB, requested_address=_ARENA_BASE)
    (reservation,) = _FakeReservation.instances
    arena.commit(0, 1)
    arena.close()
    assert reservation.closed is True
    assert arena.closed is True
    assert arena.committed_bytes == 0
    arena.close()  # idempotent
    with pytest.raises(RuntimeError, match="commit after close"):
        arena.commit(0, 1)
    with pytest.raises(RuntimeError, match="mem_pool after close"):
        arena.mem_pool()
    with pytest.raises(RuntimeError, match="map_existing_chunks after close"):
        arena.map_existing_chunks([])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
