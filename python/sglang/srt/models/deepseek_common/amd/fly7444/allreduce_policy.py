# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every tuned decision of the FlyDSL all-reduce: which kernel runs a payload,
and in which configuration.

**Kernels**

* **one-shot** (``OneShotAllReduce``) -- one round, no grid-wide barrier, wire
  volume ``(N-1)*S``. Exact: fp32 accumulate, one bf16 rounding.
* **mesh** (``QuickAllReduceInt4(algorithm="mesh")``) -- two-shot, fanout to all
  ``N-1`` peers twice, wire volume ``2(N-1)/N*S``, INT4 on the wire.
* **ring** (``QuickAllReduceInt4(algorithm="ring")``) -- two-shot, ``2(N-1)``
  hops, same wire volume as the mesh, traded for per-destination locality.

The decisions come in two levels.

**Level 1: which algorithm**

``FAMILY_POLICIES`` has one ``FamilyPolicy`` per ``(arch, link,
world_size)``. A policy has one ``Window`` of payload sizes per algorithm
(``AllReduceAlgorithm``): ``min`` inclusive, ``max`` exclusive, ``max=None``
for no upper limit. A missing or empty window disables the algorithm, and a
payload in no window goes to other Aiter all-reduce implementations.
On the payload axis, a policy reads::

    quantized (AITER_QUICK_REDUCE_QUANTIZATION=INT4):
      0 --- oneshot ---)    (gap)    [--- mesh ---)[--- ring --->

    exact (quick reduce off):
      0 --- oneshot ---)    cross_device_reduce / RCCL --->

The one-shot window may overlap the mesh and the ring windows:
``QuickAllReduce`` runs before ``CustomAllreduce``, and with quantization on,
the mesh and the ring take their payloads first.

``resolve_family_policy`` gives the policy for a given rank: the table row for the
host's arch and link, with the ``AITER_FLY_AR_<ALG>_{MIN,MAX}_BYTES``
overrides applied.

**Level 2: which configuration**

Each algorithm has a tuning ladder per ``(arch, link, world_size)``:
``ONESHOT_LADDER``, ``MESH_ST_LADDER`` and ``RING_ST_LADDER``. A ladder is a
tuple of rungs in ascending ``min_bytes`` order, and a payload runs on the last
rung whose ``min_bytes`` is at most its size (``select_rung``). The host builds
one engine per rung. The mesh algorithm's fused add+RMSNorm path has its own ladder,
``MESH_NORM_LADDER``, and runs on the plain mesh ladder where that has no row.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os

from .kernels.comm.allreduce_shared import (
    BLOCK,
    LINKS,
    MAX_PAYLOAD_BYTES,  # noqa: F401  -- re-exported for the dispatchers
    SUPPORTED_ARCHS,  # noqa: F401  -- re-exported for the dispatchers
    SUPPORTED_WORLDS,
    AllReduceAlgorithm,
    FamilyPolicy,
    OneShotRung,
    TwoShotRung,
    Window,
)

logger = logging.getLogger("aiter")


# Level 1: algorithm boundaries
# ==============================

KiB = 1 << 10
MiB = 1 << 20

FAMILY_POLICIES: tuple[FamilyPolicy, ...] = (
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
    FamilyPolicy(
        arch="gfx942",
        link="xgmi",
        world_size=2,
        oneshot=Window(0, 1536 * KiB),
        mesh=Window(384 * KiB, 128 * MiB),
        ring=Window(128 * MiB, None),
    ),
    FamilyPolicy(
        arch="gfx942",
        link="xgmi",
        world_size=4,
        oneshot=Window(0, 384 * KiB),
        mesh=Window(384 * KiB, 128 * MiB),
        ring=Window(128 * MiB, None),
    ),
    FamilyPolicy(
        arch="gfx942",
        link="xgmi",
        world_size=8,
        oneshot=Window(0, 168 * KiB),
        mesh=Window(256 * KiB, 128 * MiB),
        ring=Window(128 * MiB, None),
    ),
    # --- gfx950 (MI355X) / xGMI ----------------------------------------------
    FamilyPolicy(
        arch="gfx950",
        link="xgmi",
        world_size=2,
        oneshot=Window(0, 4 * MiB),
        mesh=Window(512 * KiB, None),
        ring=None,
    ),
    FamilyPolicy(
        arch="gfx950",
        link="xgmi",
        world_size=4,
        oneshot=Window(0, 160 * KiB),
        mesh=Window(384 * KiB, None),
        ring=None,
    ),
    FamilyPolicy(
        arch="gfx950",
        link="xgmi",
        world_size=8,
        oneshot=Window(0, 96 * KiB),
        mesh=Window(384 * KiB, None),
        ring=None,
    ),
    # --- gfx950 (MI350P) / PCIe ----------------------------------------------
    FamilyPolicy(
        arch="gfx950",
        link="pcie",
        world_size=2,
        oneshot=Window(0, 64 * MiB),
        mesh=Window(384 * KiB, None),
        ring=None,
    ),
    FamilyPolicy(
        arch="gfx950",
        link="pcie",
        world_size=4,
        oneshot=Window(0, 896 * KiB),
        mesh=Window(128 * KiB, 18 * MiB),
        ring=Window(18 * MiB, None),
    ),
    FamilyPolicy(
        arch="gfx950",
        link="pcie",
        world_size=8,
        oneshot=Window(0, 1216 * KiB),
        mesh=Window(96 * KiB, 24 * MiB),
        ring=Window(24 * MiB, None),
    ),
)


class NoPolicyError(ValueError):
    """There is no FlyDSL all-reduce policy for the platform."""


# Platforms that are not supported, ``(arch, link)``. A lookup raises for them.
NOT_SUPPORTED_PLATFORMS: dict[tuple[str, str], str] = {
    ("gfx942", "pcie"): "gfx942 has no PCIe-only device",
}


def _index(rows):
    index = {}
    for row in rows:
        key = (row.arch, row.link, row.world_size)
        if key in index:
            raise ValueError(f"duplicate table entry for {key}")
        index[key] = row
    return index


_POLICY_BY_KEY = _index(FAMILY_POLICIES)


# Level 2: configuration ladders
# ==============================
#
# The ladders are keyed by ``(arch, link, world_size)``.


# One-shot, ``OneShotRung``.
#
#   PCIe
#
#     TP2  b256 a1 cap128 -- the 4 KiB tile to 384 KiB, then b512 a4 cap128,
#                            the 32 KiB tile, over the rest of the exact-mode
#                            window, which runs to 64 MiB.
#     TP4  b128 a1 cap128 -- the 2 KiB tile to 96 KiB, then b512 a2 cap64, the
#                            16 KiB tile, to the 896 KiB exact-mode ceiling.
#     TP8  b64 a1 cap64   -- the 1 KiB tile, except b256 a1 cap64, the 4 KiB
#                            tile, from 48 to 192 KiB.
#
#   xGMI
#
#     TP2  atoms=1 cap128 b128/b256 -- the 2 KiB tile to 256 KiB, the most it
#                               covers in one round, then the 4 KiB tile.
#     TP4  atoms=1 cap128 b256 -- one rung over the whole window
#     TP8  atoms=1 cap128 b128 -- one rung over the whole window
#
#     The numbers above are gfx942. On gfx950 the rung is atoms=1 cap128 and
#     one rung over the whole window: b256 for TP2, b128 for TP4 and TP8.
ONESHOT_LADDER: dict[tuple[str, str, int], tuple[OneShotRung, ...]] = {
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
    ("gfx942", "xgmi", 2): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
        OneShotRung(
            min_bytes=256 * KiB, atoms=1, grid_cap=128, fanout="peer", block=256
        ),
    ),
    ("gfx942", "xgmi", 4): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=256),
    ),
    ("gfx942", "xgmi", 8): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
    ),
    # --- gfx950 (MI355X) / xGMI ----------------------------------------------
    ("gfx950", "xgmi", 2): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=256),
    ),
    ("gfx950", "xgmi", 4): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
    ),
    ("gfx950", "xgmi", 8): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
    ),
    # --- gfx950 (MI350P) / PCIe ----------------------------------------------
    ("gfx950", "pcie", 2): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=256),
        OneShotRung(
            min_bytes=384 * KiB, atoms=4, grid_cap=128, fanout="peer", block=512
        ),
    ),
    ("gfx950", "pcie", 4): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
        OneShotRung(min_bytes=96 * KiB, atoms=2, grid_cap=64, fanout="peer", block=512),
    ),
    ("gfx950", "pcie", 8): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=64, fanout="peer", block=64),
        OneShotRung(min_bytes=48 * KiB, atoms=1, grid_cap=64, fanout="peer", block=256),
        OneShotRung(min_bytes=192 * KiB, atoms=1, grid_cap=64, fanout="peer", block=64),
    ),
}

# Mesh, ``TwoShotRung``.
#
#   xGMI  gfx942: ST=1 everywhere, except TP4 below 4 MiB and TP8 from 6 MiB (ST=8).
#           Block 512 from 4 MiB.
#         gfx950: one b256 rung per TP.
#           - TP2 ST=1 cap512
#           - TP4 ST=1 cap1024
#           - TP8 ST=8 cap384:
#   PCIe  TP2 ST=1, then ST=8 from 4 MiB. TP4 ST=1, then ST=8 from 768 KiB.
#         TP8 ST=8 everywhere.
MESH_ST_LADDER: dict[tuple[str, str, int], tuple[TwoShotRung, ...]] = {
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
    ("gfx942", "xgmi", 2): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 * MiB, super_tile=1, grid_cap=128, block=512),
    ),
    ("gfx942", "xgmi", 4): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 * MiB, super_tile=1, grid_cap=128, block=512),
    ),
    ("gfx942", "xgmi", 8): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 * MiB, super_tile=1, grid_cap=128, block=512),
        TwoShotRung(min_bytes=6 * MiB, super_tile=8, grid_cap=128, block=512),
    ),
    # --- gfx950 (MI355X) / xGMI ----------------------------------------------
    ("gfx950", "xgmi", 2): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=512, block=256),
    ),
    ("gfx950", "xgmi", 4): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=1024, block=256),
    ),
    ("gfx950", "xgmi", 8): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=384, block=256),
    ),
    # --- gfx950 (MI350P) / PCIe ----------------------------------------------
    ("gfx950", "pcie", 2): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 * MiB, super_tile=8, grid_cap=128, block=512),
    ),
    ("gfx950", "pcie", 4): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=768 * KiB, super_tile=8, grid_cap=128, block=512),
    ),
    ("gfx950", "pcie", 8): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=512),
    ),
}

# Mesh with the fused add+RMSNorm epilogue, ``TwoShotRung``. Every rank
# normalises the whole payload in the all-gather lap.
# A platform with no row runs on its mesh ladder.
#
#   xGMI  gfx950
#           TP8: ST=1 b256 cap1024 over the whole window, at hidden 4096
#           and 8192, with or without MXFP4. TP2 and TP4 have no row: their mesh
#           rung is also the best fused one.
MESH_NORM_LADDER: dict[tuple[str, str, int], tuple[TwoShotRung, ...]] = {
    # --- gfx950 (MI355X) / xGMI ----------------------------------------------
    ("gfx950", "xgmi", 8): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=1024, block=256),
    ),
}

# Ring, ``TwoShotRung``. The ring publishes ``2(N-1)`` times per super-tile, so
# a larger payload wants a larger super-tile. Only the platforms whose policy
# rings have a row: gfx950/xGMI and gfx950/PCIe at TP2 never use ring algorithm.
RING_ST_LADDER: dict[tuple[str, str, int], tuple[TwoShotRung, ...]] = {
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
    ("gfx942", "xgmi", 2): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=BLOCK),
        TwoShotRung(min_bytes=24 * MiB, super_tile=16, grid_cap=128, block=BLOCK),
    ),
    ("gfx942", "xgmi", 4): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=BLOCK),
        TwoShotRung(min_bytes=24 * MiB, super_tile=16, grid_cap=128, block=BLOCK),
        TwoShotRung(min_bytes=48 * MiB, super_tile=32, grid_cap=128, block=BLOCK),
    ),
    ("gfx942", "xgmi", 8): (
        TwoShotRung(min_bytes=0, super_tile=16, grid_cap=128, block=BLOCK),
        TwoShotRung(min_bytes=48 * MiB, super_tile=32, grid_cap=128, block=BLOCK),
    ),
    # --- gfx950 (MI350P) / PCIe ----------------------------------------------
    ("gfx950", "pcie", 4): (
        TwoShotRung(min_bytes=0, super_tile=32, grid_cap=128, block=512),
    ),
    ("gfx950", "pcie", 8): (
        TwoShotRung(min_bytes=0, super_tile=32, grid_cap=128, block=512),
    ),
}

# The ladder table of each algorithm. A table has a row exactly where the
# policy table gives its algorithm a window.
LADDERS: dict[AllReduceAlgorithm, dict] = {
    AllReduceAlgorithm.OneShot: ONESHOT_LADDER,
    AllReduceAlgorithm.TwoShotMesh: MESH_ST_LADDER,
    AllReduceAlgorithm.TwoShotRing: RING_ST_LADDER,
}


def _ladder(table: dict, name: str, arch, link: str, world_size: int):
    arch = detect_arch() if arch is None else arch
    _check_platform(arch, link, int(world_size))
    rungs = table.get((arch, link, int(world_size)))
    if rungs is None:
        raise NoPolicyError(f"no {name} ladder for {arch}/{link}/TP{world_size}")
    return rungs


def oneshot_ladder(
    *, world_size: int, link: str, arch: str | None = None
) -> tuple[OneShotRung, ...]:
    """The one-shot rungs for *(arch, link, world_size)*. *arch* defaults to
    this host's."""
    return _ladder(ONESHOT_LADDER, "one-shot", arch, link, world_size)


def mesh_st_ladder(
    *, world_size: int, link: str, arch: str | None = None
) -> tuple[TwoShotRung, ...]:
    """The mesh rungs for *(arch, link, world_size)*. *arch* defaults to this
    host's."""
    return _ladder(MESH_ST_LADDER, "mesh", arch, link, world_size)


def mesh_norm_ladder(
    *,
    world_size: int,
    link: str,
    arch: str | None = None,
    mesh: tuple[TwoShotRung, ...] | None = None,
) -> tuple[TwoShotRung, ...]:
    """The rungs of the mesh's fused add+RMSNorm for *(arch, link,
    world_size)*: its row in ``MESH_NORM_LADDER``, or where there is none the
    mesh ladder, *mesh* if given. *arch* defaults to this host's."""
    arch = detect_arch() if arch is None else arch
    rungs = MESH_NORM_LADDER.get((arch, link, int(world_size)))
    if rungs is not None:
        return rungs
    if mesh is not None:
        return mesh
    return mesh_st_ladder(world_size=world_size, link=link, arch=arch)


def ring_st_ladder(
    *, world_size: int, link: str, arch: str | None = None
) -> tuple[TwoShotRung, ...]:
    """The ring rungs for *(arch, link, world_size)*. *arch* defaults to this
    host's."""
    return _ladder(RING_ST_LADDER, "ring", arch, link, world_size)


# Floor on the block count of the mesh and the ring when they batch publishes
# into super-tiles. Shrinking the grid trades parallelism for fewer release fences,
# which is only a good trade once there are enough fences to matter.
MIN_BATCH_BLOCKS = 32

# World size from which the ring's reduce-scatter lap defaults to INT6, to
# clear the 18 dB SQNR floor the schedules are held to. The ring requantizes
# the running partial at every hop, and the partial's extremum grows with the
# contributions folded in, so unlike the mesh its error grows with N. With the
# E5M2 group scale an all-INT4 ring reaches 17.9 dB at TP4 (E4M3: 18.3), under
# the floor, so INT6 starts at TP4. On PCIe that is up to 26/18 of the
# reduce-scatter bytes at TP4; a precision-preserving scale (per-slice
# E4M3/E5M2) would let it move back to TP8.
RS_INT6_MIN_WORLD = 4


# Host and environment
# ====================


_UNSET = object()


def _env_bound(name: str, *, is_max: bool):
    """The window bound that the variable *name* sets, or ``_UNSET``.

    Unset, empty or ``-1``: ``_UNSET``. ``none`` or ``inf``: None, no upper
    limit, for a maximum only. Anything else that is not a non-negative
    integer: a warning, and ``_UNSET``.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return _UNSET
    text = raw.strip().lower()
    if text in ("none", "inf"):
        if is_max:
            return None
        logger.warning(
            "FlyDSL all-reduce: ignoring %s=%r, a minimum is a number", name, raw
        )
        return _UNSET
    try:
        val = int(text)
    except ValueError:
        val = -2
    if val == -1:
        return _UNSET
    if val < 0:
        logger.warning(
            "FlyDSL all-reduce: ignoring %s=%r, expected bytes, -1, none or inf",
            name,
            raw,
        )
        return _UNSET
    return val


# The environment variables that override each window, as their infix.
_WINDOW_ENV = {
    AllReduceAlgorithm.OneShot: "ONESHOT",
    AllReduceAlgorithm.TwoShotMesh: "MESH",
    AllReduceAlgorithm.TwoShotRing: "RING",
}


def _override(window: Window | None, lo, hi) -> Window | None:
    if lo is _UNSET and hi is _UNSET:
        return window
    if hi == 0:
        # A maximum of 0 disables the algorithm, whatever its minimum.
        return Window(0, 0)
    base = Window(0, None) if window is None else window
    return Window(base.min if lo is _UNSET else lo, base.max if hi is _UNSET else hi)


def apply_env_overrides(policy: FamilyPolicy) -> FamilyPolicy:
    """*policy* with the ``AITER_FLY_AR_<ALG>_{MIN,MAX}_BYTES`` overrides.

    An override can move a window and enable an algorithm, but only where
    the algorithm has a ladder for the platform. Elsewhere the override is
    ignored.
    """

    bounds = {
        alg: (
            _env_bound(f"AITER_FLY_AR_{infix}_MIN_BYTES", is_max=False),
            _env_bound(f"AITER_FLY_AR_{infix}_MAX_BYTES", is_max=True),
        )
        for alg, infix in _WINDOW_ENV.items()
    }
    key = (policy.arch, policy.link, policy.world_size)
    windows = {}
    for alg, field in (
        (AllReduceAlgorithm.OneShot, "oneshot"),
        (AllReduceAlgorithm.TwoShotMesh, "mesh"),
        (AllReduceAlgorithm.TwoShotRing, "ring"),
    ):
        window = _override(getattr(policy, field), *bounds[alg])
        if window is not None and not window.is_empty and key not in LADDERS[alg]:
            logger.warning(
                "AITER_FLY_AR_%s_{MIN,MAX}_BYTES ignored: the %s has no tuned "
                "configuration on %s/%s/TP%d.",
                _WINDOW_ENV[alg],
                alg.value,
                *key,
            )
            window = getattr(policy, field)
        windows[field] = window
    return dataclasses.replace(policy, **windows)


def enabled() -> bool:
    """Whether ``AITER_FLY_AR`` opts in to the FlyDSL all-reduce path.

    Opt-in only, and only ``"1"`` opts in -- unset, ``"0"`` and anything else
    means disabled.
    """
    return os.environ.get("AITER_FLY_AR", "").strip() == "1"


@functools.lru_cache(maxsize=1)
def detect_arch() -> str | None:
    """This host's ``gfx`` arch, or None if it cannot be read."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    try:
        return get_gfx_runtime()
    except (KeyError, RuntimeError):
        return None


# The environment variable that sets the interconnect type.
LINK_ENV = "AITER_FLY_AR_LINK"


def resolve_link(link: str | None) -> str:
    """*link*, or this host's when it is None. Raises on an unknown link."""
    if link is None:
        link = detect_link()
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")
    return link


@functools.lru_cache(maxsize=1)
def _probe_link() -> str | None:
    from .allreduce_host_shared import probe_xgmi_peer_links

    peer_link_type = probe_xgmi_peer_links()
    return None if peer_link_type is None else "xgmi" if peer_link_type else "pcie"


def detect_link() -> str:
    """``"xgmi"`` or ``"pcie"`` for this host.

    Try first reading from the KFD topology, once per process
    (``probe_xgmi_peer_links``). Only when that is unreadable, fall back to the
    env variable ``AITER_FLY_AR_LINK``.

    Host-wide, not per group; see ``probe_xgmi_peer_links`` for the
    uniform-node assumption that makes that safe.
    """
    link = _probe_link()

    if link is None:
        raw = os.environ.get(LINK_ENV, "").strip().lower()
        if raw:
            if raw not in LINKS:
                raise ValueError(f"{LINK_ENV} must be one of {LINKS}, got {raw!r}")
            return raw

    if link is None:
        raise ValueError(
            f"cannot tell if this host has xGMI or PCIe links: the KFD topology "
            f"is not readable. Set {LINK_ENV} to one of {LINKS}."
        )
    return link


# Level 1: resolution
# ===================


def _check_platform(arch: str, link: str, world_size: int) -> None:
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    reason = NOT_SUPPORTED_PLATFORMS.get((arch, link))
    if reason is not None:
        raise NoPolicyError(f"no FlyDSL all-reduce for {arch}/{link}: {reason}")


def family_policy(*, arch: str, link: str, world_size: int) -> FamilyPolicy:
    """The table row for *(arch, link, world_size)*, without overrides."""
    _check_platform(arch, link, int(world_size))
    row = _POLICY_BY_KEY.get((arch, link, int(world_size)))
    if row is None:
        raise NoPolicyError(
            f"no FlyDSL all-reduce policy for {arch}/{link}/TP{world_size}"
        )
    return row


def resolve_family_policy(
    *, world_size: int, arch: str | None = None, link: str | None = None
) -> FamilyPolicy:
    """The policy of this rank, with the environment overrides applied."""
    row = family_policy(
        arch=detect_arch() if arch is None else arch,
        link=resolve_link(link),
        world_size=world_size,
    )
    return apply_env_overrides(row)
