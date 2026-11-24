# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Device-free core of the EGM storage tier: control block, handle files,
region planning, the page index with LRU eviction and pins, and checksums.

A segment is one contiguous virtual range: the provider's allocations mapped
back to back. Its first ``CTRL_BYTES`` hold the control block; the data area
follows and is split into one region per host pool. Inside a region every page
owns one slot, stored component-major (all component-0 objects, then all
component-1 objects, ...), so the objects of consecutive slots are adjacent and
a run of them moves as one copy.

One rank owns the index of the segment it maps. A page becomes visible only
after its copy completed; a page being read is pinned, so it is neither evicted
nor handed out again until the read finished.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import struct
import threading
import time
import zlib
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import msgspec

OBJ_ALIGN = 4096

# Control-block wire format, shared with the provider process: a crc32-tailed
# provider block at offset 0 and one rank block per rank from RANK_AREA_OFF.
CTRL_MAGIC = 0x4C3345474D434231
RANK_MAGIC = 0x4C3345474D504231
CTRL_BYTES = 4096
# magic, version, state, epoch, heartbeat, wall_ns, data_off, data_size, deadline_ns
CTRL_FMT = "<QIIQQQQQQ"
PROV_BLOCK = 96
RANK_AREA_OFF = 1024
RANK_SLOT_BYTES = 64
# magic, rank, state, epoch_echo, counter, wall_ns
RANK_FMT = "<QIIQQQ"
CTRL_VERSION = 1

PROV_RUN, PROV_STOPPING, PROV_STOPPED = 1, 2, 3
RANK_ATTACHED, RANK_DETACHED = 1, 2

HANDLE_KIND = "l3egm-handle"


class ProviderBlock(msgspec.Struct, frozen=True):
    state: int
    epoch: int
    heartbeat: int
    wall_ns: int
    data_off: int
    data_size: int
    deadline_ns: int


class RankBlock(msgspec.Struct, frozen=True):
    rank: int
    state: int
    epoch_echo: int
    counter: int
    wall_ns: int


def _crc_tail(body: bytes) -> bytes:
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def _crc_body(raw: bytes, fmt: str) -> Optional[bytes]:
    n = struct.calcsize(fmt)
    if len(raw) < n + 4:
        return None
    body, crc = raw[:n], struct.unpack("<I", raw[n : n + 4])[0]
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        return None
    return body


def pack_provider_block(
    state: int,
    epoch: int,
    heartbeat: int,
    data_off: int,
    data_size: int,
    deadline_ns: int,
    wall_ns: Optional[int] = None,
) -> bytes:
    body = struct.pack(
        CTRL_FMT,
        CTRL_MAGIC,
        CTRL_VERSION,
        state,
        epoch,
        heartbeat,
        time.time_ns() if wall_ns is None else wall_ns,
        data_off,
        data_size,
        deadline_ns,
    )
    out = _crc_tail(body)
    return out + bytes(PROV_BLOCK - len(out))


def unpack_provider_block(raw: bytes) -> Optional[ProviderBlock]:
    """None for a torn or foreign block; the caller retries on its next poll."""
    body = _crc_body(raw, CTRL_FMT)
    if body is None:
        return None
    magic, ver, state, epoch, hb, wall, doff, dsize, dl = struct.unpack(CTRL_FMT, body)
    if magic != CTRL_MAGIC or ver != CTRL_VERSION:
        return None
    return ProviderBlock(
        state=state,
        epoch=epoch,
        heartbeat=hb,
        wall_ns=wall,
        data_off=doff,
        data_size=dsize,
        deadline_ns=dl,
    )


def pack_rank_block(
    rank: int,
    state: int,
    epoch_echo: int,
    counter: int,
    wall_ns: Optional[int] = None,
) -> bytes:
    body = struct.pack(
        RANK_FMT,
        RANK_MAGIC,
        rank,
        state,
        epoch_echo,
        counter,
        time.time_ns() if wall_ns is None else wall_ns,
    )
    out = _crc_tail(body)
    return out + bytes(RANK_SLOT_BYTES - len(out))


def unpack_rank_block(raw: bytes) -> Optional[RankBlock]:
    body = _crc_body(raw, RANK_FMT)
    if body is None:
        return None
    magic, rank, state, echo, counter, wall = struct.unpack(RANK_FMT, body)
    if magic != RANK_MAGIC:
        return None
    return RankBlock(
        rank=rank, state=state, epoch_echo=echo, counter=counter, wall_ns=wall
    )


def rank_slot_offset(rank: int) -> int:
    off = RANK_AREA_OFF + rank * RANK_SLOT_BYTES
    if off + RANK_SLOT_BYTES > CTRL_BYTES:
        raise ValueError("rank %d outside the control block" % rank)
    return off


def handle_body_sha(h: dict) -> str:
    body = {k: v for k, v in h.items() if k != "body_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def validate_handle(
    h: dict, node_index: Optional[int] = None, now: Optional[float] = None
) -> List[str]:
    """Problems with a provider handle file; an empty list means usable."""
    bad = []
    if h.get("kind") != HANDLE_KIND or h.get("version") != 1:
        bad.append("kind/version")
    if h.get("body_sha256") != handle_body_sha(h):
        bad.append("body sha256")
    if node_index is not None and h.get("node_index") != node_index:
        bad.append("node_index %r != %r" % (h.get("node_index"), node_index))
    try:
        if int(str(h.get("epoch")), 16) == 0:
            bad.append("epoch")
    except ValueError:
        bad.append("epoch")
    allocs = h.get("allocs") or []
    if not allocs:
        bad.append("no allocations")
    off = 0
    for a in allocs:
        # A fabric handle is 64 bytes, stored as hex.
        fabric = a.get("fabric", "")
        if a.get("off") != off or not a.get("size") or len(fabric) != 128:
            bad.append("allocation table")
            break
        off += a["size"]
    if allocs and off != h.get("total_size"):
        bad.append("total_size")
    doff, dsize = h.get("data_off", 0), h.get("data_size", 0)
    if doff < CTRL_BYTES or dsize <= 0 or doff + dsize > h.get("total_size", 0):
        bad.append("data area")
    dl = h.get("deadline")
    if dl is not None and (now if now is not None else time.time()) > dl:
        bad.append("provider deadline passed")
    return bad


def host_view(ptr: int, size: int) -> memoryview:
    """A zero-copy byte view of host memory, for checksums."""
    return memoryview((ctypes.c_char * size).from_address(ptr)).cast("B")


def crc_of(view) -> int:
    return zlib.crc32(view) & 0xFFFFFFFF


def coalesce(
    copies: Sequence[Tuple[int, int, int]],
) -> List[Tuple[int, int, int]]:
    """Merge (dst, src, size) copies whose dst and src continue the previous one."""
    out: List[List[int]] = []
    for d, s, n in copies:
        if out and out[-1][0] + out[-1][2] == d and out[-1][1] + out[-1][2] == s:
            out[-1][2] += n
        else:
            out.append([d, s, n])
    return [tuple(x) for x in out]


class DirCopier:
    """Separate copiers for reads and writes, so a write never queues behind a read."""

    def __init__(self, get_copier, set_copier):
        self.get_copier = get_copier
        self.set_copier = set_copier

    def run(self, copies, to_host: bool) -> dict:
        copier = self.get_copier if to_host else self.set_copier
        return copier.run(copies, to_host)


class PoolSpec(msgspec.Struct, frozen=True):
    """One host pool as the tier sees it: object sizes of one page, in page order,
    and the weight its region share defaults to."""

    name: str
    comp_sizes: Tuple[int, ...]
    weight: float

    def __post_init__(self):
        if not self.comp_sizes or any(int(c) <= 0 for c in self.comp_sizes):
            raise ValueError(
                "pool %s: bad component sizes %r" % (self.name, self.comp_sizes)
            )

    @property
    def page_bytes(self) -> int:
        return sum(self.comp_sizes)


class RegionPlan(msgspec.Struct, frozen=True):
    name: str
    comp_sizes: Tuple[int, ...]
    comp_base: Tuple[int, ...]
    slots: int
    start: int
    end: int
    page_bytes: int


def _align(n: int, a: int = OBJ_ALIGN) -> int:
    return ((n + a - 1) // a) * a


def plan_regions(
    specs: Sequence[PoolSpec],
    data_off: int,
    data_size: int,
    shares: Optional[Dict[str, float]] = None,
) -> List[RegionPlan]:
    """Split [data_off, data_off + data_size) into one region per pool.

    Shares default to each pool's weight; an explicit share map must name every
    pool and sum to at most 1.
    """
    if not specs:
        raise ValueError("no pools")
    if shares:
        missing = [s.name for s in specs if s.name not in shares]
        if missing:
            raise ValueError("region shares miss pools %s" % missing)
        frac = [float(shares[s.name]) for s in specs]
        if any(f <= 0 for f in frac) or sum(frac) > 1.0 + 1e-9:
            raise ValueError("region shares must be > 0 and sum to <= 1")
    else:
        tot = sum(s.weight for s in specs)
        if tot <= 0:
            raise ValueError("pool weights sum to 0")
        frac = [s.weight / tot for s in specs]
    regions = []
    base = _align(data_off)
    end = data_off + data_size
    for s, f in zip(specs, frac):
        budget = int(data_size * f) - OBJ_ALIGN * (len(s.comp_sizes) + 1)
        n = max(0, budget // s.page_bytes)
        slack = OBJ_ALIGN * len(s.comp_sizes)
        while n > 0 and sum(_align(c * n) for c in s.comp_sizes) > budget + slack:
            n -= 1
        comp_base = []
        b = base
        for c in s.comp_sizes:
            comp_base.append(b)
            b = _align(b + c * n)
        if b > end:
            raise ValueError("region for %s overflows the data area" % s.name)
        regions.append(
            RegionPlan(
                name=s.name,
                comp_sizes=tuple(s.comp_sizes),
                comp_base=tuple(comp_base),
                slots=n,
                start=base,
                end=b,
                page_bytes=s.page_bytes,
            )
        )
        base = b
    return regions


class Entry:
    __slots__ = ("slot", "crc", "pins")

    def __init__(self, slot: int, crc: List[int]):
        self.slot = slot
        self.crc = crc
        self.pins = 0


class Region:
    """Slots, index and LRU of one pool; every method runs under the Store's lock."""

    def __init__(self, plan: RegionPlan):
        self.name = plan.name
        self.comp_sizes = plan.comp_sizes
        self.comp_base = plan.comp_base
        self.slots = plan.slots
        self.page_bytes = plan.page_bytes
        self.index: OrderedDict[str, Entry] = OrderedDict()
        self.free: List[int] = []
        self.bump = 0
        self.reserved = set()
        self.evictions = 0
        self.cnt = {
            "exists_hit_pages": 0,
            "get_pages_req": 0,
            "get_pages_ok": 0,
            "set_pages_new": 0,
            "set_pages_skip": 0,
            "mismatch_pages": 0,
        }

    def addr(self, slot: int, comp: int) -> int:
        return self.comp_base[comp] + slot * self.comp_sizes[comp]

    def used(self) -> int:
        return len(self.index) + len(self.reserved)

    def _evict_one(self) -> Optional[int]:
        for key, e in self.index.items():
            if e.pins == 0:
                del self.index[key]
                self.evictions += 1
                return e.slot
        return None

    def take_slot(self) -> Optional[int]:
        if self.free:
            s = self.free.pop()
        elif self.bump < self.slots:
            s = self.bump
            self.bump += 1
        else:
            s = self._evict_one()
            if s is None:
                return None
        self.reserved.add(s)
        return s

    def give_back(self, slot: int) -> None:
        self.reserved.discard(slot)
        self.free.append(slot)

    def commit(self, key: str, slot: int, crc: List[int]) -> None:
        self.reserved.discard(slot)
        old = self.index.pop(key, None)
        # A racing writer of the same key committed first: keep the newer entry.
        if old is not None and old.pins == 0:
            self.free.append(old.slot)
        self.index[key] = Entry(slot, crc)


_COUNTERS = (
    "exists_calls",
    "exists_hit_pages",
    "get_calls",
    "get_pages_req",
    "get_pages_ok",
    "get_bytes",
    "get_ms",
    "set_calls",
    "set_pages_req",
    "set_pages_new",
    "set_pages_skip",
    "set_pages_fail",
    "set_bytes",
    "set_ms",
    "mismatch_pages",
    "copy_errors",
    "copies_issued",
    "verify_ms",
    "crc_write_ms",
    "no_slot",
)


class Store:
    """Per-rank tier state: regions, the refusal latch and counters.

    Copies run outside the lock through ``copier.run(copies, to_host)``, which
    returns only after every copy completed and raises on any copy error. Host
    bytes are read through ``host_view(ptr, size)``, a buffer over that memory.
    """

    def __init__(
        self,
        regions: Sequence[RegionPlan],
        seg_va: int,
        copier,
        host_view: Callable[[int, int], object],
        max_mismatch: int = 0,
        crc_map: Optional[Callable] = None,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.regions: Dict[str, Region] = {r.name: Region(r) for r in regions}
        self.seg_va = seg_va
        self.copier = copier
        self.host_view = host_view
        self.max_mismatch = int(max_mismatch)
        self.crc_map = crc_map or (lambda fn, items: [fn(x) for x in items])
        self.log = log or (lambda m: None)
        self.lock = threading.Lock()
        self.idle = threading.Condition(self.lock)
        self.inflight = 0
        self.refused: Optional[str] = None
        self.refused_at: Optional[float] = None
        self.c = {k: 0 for k in _COUNTERS}

    def refuse(self, why: str) -> None:
        """Latch the tier off for the rest of the run; every query then misses."""
        with self.lock:
            if self.refused is None:
                self.refused = why
                self.refused_at = time.time()
                self.log("REFUSED: %s" % why)

    def _enter(self) -> bool:
        with self.lock:
            if self.refused is not None:
                return False
            self.inflight += 1
            return True

    def _leave(self) -> None:
        with self.lock:
            self.inflight -= 1
            if self.inflight == 0:
                self.idle.notify_all()

    def drain(self, timeout_s: float) -> bool:
        """Wait until no operation is inside a copy; True if drained."""
        end = time.time() + timeout_s
        with self.lock:
            while self.inflight > 0:
                left = end - time.time()
                if left <= 0:
                    return False
                self.idle.wait(left)
        return True

    def region(self, name: str) -> Region:
        r = self.regions.get(name)
        if r is None:
            raise KeyError("pool %s has no region" % (name,))
        return r

    def present_prefix(self, name: str, keys: Sequence[str]) -> int:
        if self.refused is not None:
            return 0
        with self.lock:
            if self.refused is not None:
                return 0
            r = self.regions.get(name)
            if r is None:
                return 0
            n = 0
            for k in keys:
                if k not in r.index:
                    break
                r.index.move_to_end(k)
                n += 1
            self.c["exists_calls"] += 1
            self.c["exists_hit_pages"] += n
            r.cnt["exists_hit_pages"] += n
            return n

    def present_flags(self, name: str, keys: Sequence[str]) -> List[bool]:
        if self.refused is not None:
            return [False] * len(keys)
        with self.lock:
            r = self.regions.get(name)
            if r is None or self.refused is not None:
                return [False] * len(keys)
            out = []
            for k in keys:
                hit = k in r.index
                if hit:
                    r.index.move_to_end(k)
                out.append(hit)
            return out

    def put(
        self, name: str, keys: Sequence[str], host_ptrs: Sequence[Sequence[int]]
    ) -> List[bool]:
        """Store pages from pinned host memory; host_ptrs[i][j] is page i, component j.

        Per page True if stored now or before. Once one page gets no slot, it and
        every later page report False.
        """
        n = len(keys)
        if n == 0:
            return []
        if not self._enter():
            return [False] * n
        try:
            r = self.region(name)
            res = [False] * n
            todo = self._reserve_slots(r, keys, res)
            if not todo:
                self._fail_tail(res)
                return res
            sizes = r.comp_sizes
            nc = len(sizes)
            # The controller holds these host pages for the whole storage call,
            # so the bytes checksummed here are the bytes the copy reads.
            t0 = time.perf_counter()
            items = [(host_ptrs[i][j], sizes[j]) for i, _ in todo for j in range(nc)]
            crcs = self.crc_map(lambda it: crc_of(self.host_view(it[0], it[1])), items)
            t1 = time.perf_counter()
            copies = [
                (self.seg_va + r.addr(s, j), host_ptrs[i][j], sizes[j])
                for i, s in todo
                for j in range(nc)
            ]
            try:
                info = self.copier.run(copies, to_host=False)
            except Exception as e:
                with self.lock:
                    for _, s in todo:
                        r.give_back(s)
                    self.c["copy_errors"] += 1
                self.refuse("copy error on write: %r" % (e,))
                return [False] * n
            with self.lock:
                for q, (i, s) in enumerate(todo):
                    r.commit(keys[i], s, crcs[q * nc : (q + 1) * nc])
                    res[i] = True
                self.c["set_pages_new"] += len(todo)
                r.cnt["set_pages_new"] += len(todo)
                self.c["set_bytes"] += len(todo) * r.page_bytes
                self.c["set_ms"] += info.get("ms", 0.0)
                self.c["copies_issued"] += info.get("issued", 0)
                self.c["crc_write_ms"] += (t1 - t0) * 1e3
            self._fail_tail(res)
            return res
        finally:
            self._leave()

    def _reserve_slots(
        self, r: Region, keys: Sequence[str], res: List[bool]
    ) -> List[Tuple[int, int]]:
        """(page index, slot) of each page to copy; stored pages are marked in res."""
        todo: List[Tuple[int, int]] = []
        with self.lock:
            self.c["set_calls"] += 1
            self.c["set_pages_req"] += len(keys)
            for i, k in enumerate(keys):
                if k in r.index:
                    r.index.move_to_end(k)
                    res[i] = True
                    self.c["set_pages_skip"] += 1
                    r.cnt["set_pages_skip"] += 1
                    continue
                s = r.take_slot()
                if s is None:
                    self.c["no_slot"] += 1
                    break
                todo.append((i, s))
        return todo

    def _fail_tail(self, res: List[bool]) -> None:
        # Readers use the leading run only, so a page after a failed one is failed too.
        bad = False
        for i in range(len(res)):
            if not res[i]:
                bad = True
            if bad:
                res[i] = False
        with self.lock:
            self.c["set_pages_fail"] += sum(1 for x in res if not x)

    def get(
        self, name: str, keys: Sequence[str], host_ptrs: Sequence[Sequence[int]]
    ) -> List[bool]:
        """Copy the leading present pages into host memory and verify them.

        A page reports True only after its copy completed and its checksums
        matched; False from the first missing, failed or mismatched page on.
        """
        n = len(keys)
        if n == 0:
            return []
        if not self._enter():
            return [False] * n
        pinned: List[Tuple[int, Entry]] = []
        try:
            r = self.region(name)
            with self.lock:
                self.c["get_calls"] += 1
                self.c["get_pages_req"] += n
                r.cnt["get_pages_req"] += n
                for i, k in enumerate(keys):
                    e = r.index.get(k)
                    if e is None:
                        break
                    e.pins += 1
                    r.index.move_to_end(k)
                    pinned.append((i, e))
            res = [False] * n
            if not pinned:
                return res
            sizes = r.comp_sizes
            copies = [
                (host_ptrs[i][j], self.seg_va + r.addr(e.slot, j), sizes[j])
                for i, e in pinned
                for j in range(len(sizes))
            ]
            try:
                info = self.copier.run(copies, to_host=True)
            except Exception as e:
                with self.lock:
                    self.c["copy_errors"] += 1
                self.refuse("copy error on read: %r" % (e,))
                return [False] * n
            t0 = time.perf_counter()
            ok_upto = self._verify(r, keys, host_ptrs, pinned)
            t1 = time.perf_counter()
            for q in range(ok_upto):
                res[pinned[q][0]] = True
            with self.lock:
                self.c["get_pages_ok"] += ok_upto
                r.cnt["get_pages_ok"] += ok_upto
                self.c["get_bytes"] += len(pinned) * r.page_bytes
                self.c["get_ms"] += info.get("ms", 0.0)
                self.c["copies_issued"] += info.get("issued", 0)
                self.c["verify_ms"] += (t1 - t0) * 1e3
            return res
        finally:
            with self.lock:
                for _, e in pinned:
                    e.pins -= 1
            self._leave()

    def _verify(self, r: Region, keys, host_ptrs, pinned) -> int:
        """Leading pinned pages that pass; the first failing page leaves the index."""
        sizes = r.comp_sizes
        jobs = [
            (q, host_ptrs[i][j], sizes[j], e.crc[j])
            for q, (i, e) in enumerate(pinned)
            for j in range(len(sizes))
        ]

        def check(job):
            _, ptr, size, want = job
            return crc_of(self.host_view(ptr, size)) == want

        ok = self.crc_map(check, jobs)
        bad_q = None
        for job, good in zip(jobs, ok):
            if not good:
                bad_q = job[0] if bad_q is None else min(bad_q, job[0])
        if bad_q is None:
            return len(pinned)
        i_bad = pinned[bad_q][0]
        with self.lock:
            self.c["mismatch_pages"] += 1
            r.cnt["mismatch_pages"] += 1
            # The slot leaves service for the rest of the run.
            r.index.pop(keys[i_bad], None)
            mism = self.c["mismatch_pages"]
        self.log(
            "CHECKSUM MISMATCH pool %s page %s (slot %d); read as a miss"
            % (r.name, keys[i_bad], pinned[bad_q][1].slot)
        )
        if mism > self.max_mismatch:
            self.refuse("checksum mismatches %d > %d" % (mism, self.max_mismatch))
        return bad_q

    def clear(self) -> bool:
        """Forget every page; False while an operation is in flight."""
        with self.lock:
            if self.inflight:
                return False
            for r in self.regions.values():
                r.index.clear()
                r.free = []
                r.bump = 0
                r.reserved = set()
            return True

    def snapshot(self) -> dict:
        with self.lock:
            regs = {
                name: {
                    "slots": r.slots,
                    "used": r.used(),
                    "indexed": len(r.index),
                    "evictions": r.evictions,
                    "page_bytes": r.page_bytes,
                    **r.cnt,
                }
                for name, r in self.regions.items()
            }
            return {
                "counters": dict(self.c),
                "regions": regs,
                "refused": self.refused,
                "refused_at": self.refused_at,
                "inflight": self.inflight,
            }


def exists_v2_core(
    kv_pages: int, pools: Iterable[Tuple[str, str, int, List[bool]]]
) -> Tuple[int, Dict[str, int], List[int]]:
    """Hit rules of batch_exists_v2, same as the Mooncake backend's.

    pools holds (name, hit policy, trailing page count, presence flags of the
    first kv_pages pages). Returns (final pages, {name: boundary}, restorable
    prefix lengths).
    """
    hit_count: Dict[str, int] = {}
    restorable = list(range(1, kv_pages + 1))
    if not restorable:
        return 0, hit_count, []
    for name, policy, trailing_n, page_exists in pools:
        if not restorable:
            break
        boundary = 0
        pool_restorable = []
        if policy == "all_pages":
            try:
                boundary = page_exists.index(False)
            except ValueError:
                boundary = kv_pages
            pool_restorable = list(range(1, boundary + 1))
        elif policy == "trailing_pages":
            trailing = max(1, trailing_n)
            for prefix_len in range(kv_pages, 0, -1):
                if all(
                    page_exists[i]
                    for i in range(max(0, prefix_len - trailing), prefix_len)
                ):
                    pool_restorable.append(prefix_len)
                    if boundary == 0:
                        boundary = prefix_len
        else:
            raise ValueError("unsupported pool hit policy %r" % (policy,))
        if boundary:
            hit_count[name] = boundary
        keep = set(pool_restorable)
        restorable = [p for p in restorable if p in keep]
    return (restorable[-1] if restorable else 0), hit_count, restorable
