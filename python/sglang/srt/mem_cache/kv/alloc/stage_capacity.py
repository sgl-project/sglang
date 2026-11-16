"""Capacity under pipeline parallelism: every stage asks every stage.

Each stage's arena and unit bytes follow its layers, but every stage runs the same
logical operations, so all hold the same units. Each stage asks a capacity question
of every stage's snapshot, built from that stage's geometry and the shared units
(`simulated_snapshot`); a demand fits only where it fits on every stage, so the
stages decide alike, as tensor-parallel ranks do.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional, Sequence, Tuple

import msgspec

from sglang.srt.mem_cache.kv.alloc.pricing import BandSnapshot, ChainSnapshot


class BandGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """One band of a stage's chain as its allocator built it: what pricing
    reads that is not a count."""

    name: str
    grow: str
    page_bytes: int
    entry_bytes: int
    page_size: int
    num_pages: int
    min_page: int
    lazy: bool
    num_virtual_pages: int
    # Mints its own virtual ids (the token owner, the state slots); a window
    # binds the owner's.
    id_owner: bool


class StageGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """A stage's arena bytes and its bands in chain order. ``window_tokens``
    is the stage's nominal window share, the sizing's window tokens."""

    total_bytes: int
    page_size: int
    bands: Tuple[BandGeometry, ...]
    window_tokens: int = 0

    def sub_pools(self) -> Tuple[str, ...]:
        return tuple(sorted(band.name for band in self.bands))


def check_stage_pools(pools: Sequence[Tuple[Tuple[str, ...], int]]) -> None:
    """Given every stage's sorted sub-pool names and page size in stage order, raise
    the same error on every stage unless all are alike: each question asks them all."""
    if len(set(pools)) != 1:
        raise ValueError(
            "pipeline stages must plan the same sub-pools on the same page size "
            "under unified memory, got (sub-pools, page size) per stage "
            f"{list(pools)}; choose a --pp-size or SGLANG_PP_LAYER_PARTITION that "
            "gives every stage every kind of layer"
        )


def stage_geometry(
    snapshot: ChainSnapshot, *, id_owners: Tuple[str, ...], window_tokens: int
) -> StageGeometry:
    """The geometry of the stage whose pricing snapshot is ``snapshot``."""
    return StageGeometry(
        total_bytes=snapshot.total_bytes,
        page_size=snapshot.page_size,
        bands=tuple(
            BandGeometry(
                name=b.name,
                grow=b.grow,
                page_bytes=b.page_bytes,
                entry_bytes=b.entry_bytes,
                page_size=b.page_size,
                num_pages=b.num_pages,
                min_page=b.min_page,
                lazy=b.lazy,
                num_virtual_pages=b.num_virtual_pages,
                id_owner=b.name in id_owners,
            )
            for b in snapshot.bands
        ),
        window_tokens=window_tokens,
    )


def simulated_snapshot(
    geometry: StageGeometry,
    live: Mapping[str, int],
    used_ids: Mapping[str, int],
) -> ChainSnapshot:
    """A stage's snapshot of ``live`` pages per band and ``used_ids`` virtual pages
    per id owner, ends compacted and nothing pending or fixed: the stage's own
    snapshot of these counts serves the same demands once its local holds end."""
    bands = []
    for g in geometry.bands:
        pages = live[g.name]
        e = g.page_bytes
        if g.grow == "down":
            low, high = g.num_pages - pages, g.num_pages
        else:
            low, high = g.min_page, g.min_page + pages
        free_ids = 0
        if g.id_owner:
            free_ids = g.num_virtual_pages - g.min_page - used_ids[g.name]
        floats = g.grow == "float"
        bands.append(
            BandSnapshot(
                name=g.name,
                grow=g.grow,
                page_bytes=e,
                entry_bytes=g.entry_bytes,
                page_size=g.page_size,
                num_pages=g.num_pages,
                min_page=g.min_page,
                lazy=g.lazy,
                blocked=False,
                holes=0,
                live=pages,
                span=pages,
                low_byte=low * e,
                high_byte=high * e,
                free_virtual_pages=free_ids,
                num_virtual_pages=g.num_virtual_pages,
                low_page=low if floats else 0,
                high_page=high if floats else 0,
            )
        )
    return ChainSnapshot(
        tuple(bands),
        geometry.page_size,
        geometry.total_bytes,
        {b.name: i for i, b in enumerate(bands)},
    )


def _canonical(value: Any) -> Any:
    """``value`` with every mapping as its sorted items: stages may hold the same
    bands in a different arena order."""
    if isinstance(value, Mapping):
        return tuple((k, _canonical(v)) for k, v in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_canonical(v) for v in value)
    return value


class StageCapacity:
    """This stage's place among the pipeline stages and all their geometries; with
    ``checks``, a digest of its capacity questions and answers (`take_digest`)."""

    def __init__(
        self, stage: int, geometries: Sequence[StageGeometry], *, checks: bool
    ):
        self.stage = stage
        self.geometries: Tuple[StageGeometry, ...] = tuple(geometries)
        self.checks = checks
        self._digest = hashlib.blake2b(digest_size=16) if checks else None

    def note(self, op: str, key: Any, answer: Any) -> None:
        """Fold a question and its answer into the digest."""
        if self._digest is not None:
            self._digest.update(repr(_canonical((op, key, answer))).encode())

    def take_digest(self, counts: Any) -> Optional[bytes]:
        """The digest of the questions asked since the last call, and of
        ``counts``, the units held now; None without checks."""
        if self._digest is None:
            return None
        self._digest.update(repr(_canonical(counts)).encode())
        digest = self._digest.digest()
        self._digest = hashlib.blake2b(digest_size=16)
        return digest
