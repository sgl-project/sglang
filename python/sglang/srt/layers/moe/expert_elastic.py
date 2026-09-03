"""Elastic expert residency: a resizable, rank-ordered GPU cache of MoE expert rows.

Every (layer, tensor kind) keeps its expert rows in a RowArena on the GPU, in routing-mass
order (rank 0 = hottest). The int64 address table addr[e] points either into the arena
(rank(e) < S) or at the row's home slot in pinned host memory. Growing S gathers the next
ranks from their host slots straight into the arena (table-driven gather kernel, no staging),
rewrites their table entries and returns the host slots to a pool. Shrinking S copies the
tail ranks back into pool slots, rewrites the table, then unmaps the arena tail so the VRAM
really returns to the driver. Values never change, only where they live; the decode GEMV and
the prefill gather read through the tables and never notice. CUDA graphs captured against
the tables stay valid because arena addresses are fixed for the process lifetime.

Host memory is conserved: rows off the GPU always own a host slot. Slots come from the
offloaded layers' pinned tensors (a host layer's hot rows vacate their slots). With 31 host
layers of 512 rows and 48 layers, the pool is empty at S = 181, so S_floor is 184 unless a
pinned fallback budget is granted (SGLANG_MOE_ELASTIC_PIN_MB).

Control (poll() is called from the MoE apply path, prefill/eager only):
    SGLANG_MOE_ELASTIC_CTL=/path/ctl      one command per write:
        S <n>            set every layer to S = n (clamped to [S_floor, 512])
        fill <MB>        grow all layers, keeping <MB> of VRAM free
        free <MB>        shrink all layers until at least <MB> of VRAM is free
        status           write the status file only
    <ctl>.status is rewritten after every command (per-layer S, arena bytes, free VRAM).
"""

import logging
import os
import time

import torch

from sglang.srt.layers.moe.row_arena import ArenaOOM, RowArena

logger = logging.getLogger(__name__)
NAMES = ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales")


def _gather(tab, ids, out_rows_u8, row_bytes):
    """out_rows_u8[i] <- row at tab[ids[i]] (GPU or pinned host address)."""
    import triton

    from sglang.srt.layers.moe.expert_stream import _gather_rows_tab_kernel

    BLOCK = 1024
    _gather_rows_tab_kernel[(int(ids.numel()), triton.cdiv(row_bytes, BLOCK))](
        tab, ids, out_rows_u8, row_bytes, BLOCK=BLOCK
    )


class _Kind:
    __slots__ = ("name", "arena", "tab", "home", "tail", "dtype", "rb", "keep")

    def __init__(self, name, arena, tab, home, tail, dtype, rb, keep):
        self.name, self.arena, self.tab, self.home = name, arena, tab, home
        self.tail, self.dtype, self.rb, self.keep = tail, dtype, rb, keep


class _Layer:
    __slots__ = ("layer", "lid", "E", "S", "kinds")

    def __init__(self, layer, lid, E, S):
        self.layer, self.lid, self.E, self.S, self.kinds = layer, lid, E, S, {}


class ExpertElastic:
    inst = None

    def __init__(self, mass: torch.Tensor, S0: int, device="cuda"):
        self.mass = mass.float()  # [L, E] routing mass
        self.L, self.E = mass.shape
        self.order = torch.argsort(
            self.mass, dim=1, descending=True
        )  # [L, E] rank -> expert
        self.rank = torch.empty_like(self.order)
        self.rank.scatter_(1, self.order, torch.arange(self.E).expand(self.L, -1))
        self.S0 = int(S0)
        self.device = device
        self.layers = {}  # lid -> _Layer
        self.pool = {n: [] for n in NAMES}  # free host slots (tensor, row)
        self.pin_budget = int(os.environ.get("SGLANG_MOE_ELASTIC_PIN_MB", "0")) << 20
        self.pin_used = 0
        self._pinned_keep = []
        self._serving = False  # flips after placement
        self.reserve_rows = int(os.environ.get("SGLANG_MOE_ELASTIC_RESERVE_ROWS", "0"))
        self.ctl = os.environ.get("SGLANG_MOE_ELASTIC_CTL")
        self._ctl_mtime = (
            os.stat(self.ctl).st_mtime if self.ctl and os.path.exists(self.ctl) else 0.0
        )  # ignore stale commands
        _af = os.environ.get(
            "SGLANG_MOE_ELASTIC_FILL_MB"
        )  # reserve to keep free; unset = no autofill
        self._autofill = int(_af) if _af else None
        self.regrow_reserve = (
            int(_af) if _af else 768
        )  # after a forced shrink, grow back to this
        self.pending_regrow = False
        self._calls = 0
        ExpertElastic.inst = self

    # ------------------------------------------------------------------ slots
    def _pin_rows(self, name, need, tail, dtype, why):
        """Allocate pinned host slots. STARTUP ONLY: on a box with ~26 GB pinned and ~2 GB free a
        runtime cudaHostAlloc stalls the kernel for tens of seconds (reclaim + compaction).
        """
        nbytes = (
            need
            * int(torch.empty(0, dtype=dtype).element_size())
            * int(torch.Size(tail).numel())
        )
        if self.pin_used + nbytes > self.pin_budget:
            raise RuntimeError(
                f"elastic: pinned budget exhausted for {name} ({why}); "
                f"SGLANG_MOE_ELASTIC_PIN_MB={self.pin_budget >> 20}, used {self.pin_used >> 20} MB"
            )
        extra = torch.empty((need,) + tuple(tail), dtype=dtype, pin_memory=True)
        self.pin_used += nbytes
        self.pool[name].extend((extra, i) for i in range(need))
        self._pinned_keep.append(extra)
        logger.info(
            "elastic: pinned %d rows of %s (%.0f MB) [%s]",
            need,
            name,
            nbytes / 1e6,
            why,
        )

    def _take_slots(self, name, n, tail, dtype):
        pool = self.pool[name]
        if len(pool) < n:
            if self._serving:
                raise RuntimeError(
                    f"elastic: host slot pool for {name} exhausted ({len(pool)} < {n}) "
                    f"and runtime pinning is disabled (raise SGLANG_MOE_ELASTIC_RESERVE_ROWS)"
                )
            self._pin_rows(name, n - len(pool), tail, dtype, "startup shortfall")
        return [pool.pop() for _ in range(n)]

    @staticmethod
    def _slot_addr(slot, rb):
        t, r = slot
        return t.data_ptr() + r * rb

    # ------------------------------------------------------------------ init
    def add_layer(self, layer):
        """Move one FusedMoE layer into arenas + tables. Host layers donate slots, GPU layers
        consume them: call in an interleaved order (see place_all)."""
        lid = int(layer.layer_id)
        E = int(layer.w13_qweight.data.shape[0])
        S = min(self.S0, E)
        st = _Layer(layer, lid, E, S)
        order = self.order[lid]
        hot = order[:S]
        cold = order[S:]
        addr, proto = {}, {}
        for name in NAMES:
            p = getattr(layer, name, None)
            if p is None:
                continue
            t = p.data
            tail, dtype = tuple(t.shape[1:]), t.dtype
            rb = t[0].numel() * t.element_size()
            arena = RowArena(
                rb, E, torch.device(self.device).index or 0, name=f"L{lid}.{name}"
            )
            arena.ensure_rows(S)
            view = arena.view(S, tail, dtype)
            tab = torch.empty(E, dtype=torch.int64)
            home = [None] * E
            keep = []
            if t.is_cuda:
                view.copy_(t.index_select(0, hot.to(t.device)))
                slots = self._take_slots(name, int(cold.numel()), tail, dtype)
                cold_cpu = t.index_select(0, cold.to(t.device)).cpu()
                for i, e in enumerate(cold.tolist()):
                    dst, r = slots[i]
                    dst[r].copy_(cold_cpu[i])
                    home[e] = slots[i]
                    tab[e] = self._slot_addr(slots[i], rb)
                    keep.append(dst)
                del cold_cpu
            else:
                src_tab = (t.data_ptr() + torch.arange(E, dtype=torch.int64) * rb).to(
                    self.device
                )
                _gather(
                    src_tab, hot.to(self.device), view.view(torch.uint8).view(S, -1), rb
                )
                for e in cold.tolist():
                    home[e] = (t, e)
                    tab[e] = t.data_ptr() + e * rb
                self.pool[name].extend((t, e) for e in hot.tolist())
                keep.append(t)
            tab[hot] = arena.base + self.rank[lid][hot] * rb
            tab = tab.to(self.device)
            st.kinds[name] = _Kind(name, arena, tab, home, tail, dtype, rb, keep)
            p.data = view
            addr[name] = tab
            proto[name] = view
            del t
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        layer._placed = {"addr": addr, "proto": proto, "E": E, "S": S, "keep": []}
        layer._gemv_tabs = None
        self.layers[lid] = st

    def place_all(self, layers):
        """Deferred pass over all layers. Host kinds donate slots, GPU kinds consume 512-S each;
        a layer is placed once every pool it will draw from can serve it (the offloader may
        split a layer: some kinds on the host, some on the GPU)."""

        def need(l):
            return {
                n: (self.E - self.S0 if getattr(l, n).data.is_cuda else 0)
                for n in NAMES
                if getattr(l, n, None) is not None
            }

        donors = [l for l in layers if not any(need(l).values())]
        takers = [l for l in layers if any(need(l).values())]
        placed = 0
        while donors or takers:
            ready = next(
                (
                    l
                    for l in takers
                    if all(len(self.pool[n]) >= k for n, k in need(l).items())
                ),
                None,
            )
            if ready is not None:
                takers.remove(ready)
                self.add_layer(ready)
            elif donors:
                self.add_layer(donors.pop(0))
            else:  # pool short: pinned fallback (or a clear error)
                self.add_layer(takers.pop(0))
            placed += 1
        if (
            self.reserve_rows
        ):  # one-time reserve so shrinking below S0 never pins at runtime
            for name in NAMES:
                k = next(
                    (
                        k
                        for st in self.layers.values()
                        for k in st.kinds.values()
                        if k.name == name
                    ),
                    None,
                )
                if k is not None:
                    self._pin_rows(
                        name, self.reserve_rows, k.tail, k.dtype, "startup reserve"
                    )
        self._serving = True
        logger.info(
            "elastic placement: %d layers at S=%d, free slots %s, VRAM free %.2f GB",
            placed,
            self.S0,
            {n: len(v) for n, v in self.pool.items()},
            torch.cuda.mem_get_info()[0] / 2**30,
        )
        self.write_status()

    # ------------------------------------------------------------------ resize
    def resize_layer(self, lid, S_new):
        st = self.layers[lid]
        S = st.S
        S_new = max(0, min(int(S_new), st.E))
        if S_new == S:
            return
        order = self.order[lid]
        torch.cuda.synchronize()
        if S_new > S:
            ids = order[S:S_new]
            for k in st.kinds.values():
                try:
                    k.arena.ensure_rows(S_new)
                except ArenaOOM:
                    for k2 in st.kinds.values():
                        k2.arena.shrink_rows(S)
                    raise
            for k in st.kinds.values():
                view = k.arena.view(S_new, k.tail, k.dtype)
                dst = view.view(torch.uint8).view(S_new, -1)[S:S_new]
                _gather(k.tab, ids.to(self.device), dst, k.rb)
            torch.cuda.synchronize()
            for k in st.kinds.values():
                k.tab[ids.to(self.device)] = (
                    k.arena.base + torch.arange(S, S_new, dtype=torch.int64) * k.rb
                ).to(self.device)
                for e in ids.tolist():
                    self.pool[k.name].append(k.home[e])
                    k.home[e] = None
        else:
            ids = order[S_new:S]
            n = int(ids.numel())
            taken = {}
            try:  # acquire-then-commit: nothing is touched until every kind has its slots
                for k in st.kinds.values():
                    taken[k.name] = self._take_slots(k.name, n, k.tail, k.dtype)
            except Exception:
                for name, slots in taken.items():
                    self.pool[name].extend(slots)
                raise
            for k in st.kinds.values():
                view = k.arena.view(S, k.tail, k.dtype)
                slots = taken[k.name]
                addrs = torch.empty(n, dtype=torch.int64)
                for i, e in enumerate(ids.tolist()):
                    dst, r = slots[i]
                    dst[r].copy_(view[S_new + i], non_blocking=True)
                    k.home[e] = slots[i]
                    addrs[i] = self._slot_addr(slots[i], k.rb)
                torch.cuda.synchronize()
                k.tab[ids.to(self.device)] = addrs.to(self.device)
            torch.cuda.synchronize()
            for k in st.kinds.values():
                k.arena.shrink_rows(S_new)
        for k in st.kinds.values():
            view = k.arena.view(S_new, k.tail, k.dtype)
            getattr(st.layer, k.name).data = view
            st.layer._placed["proto"][k.name] = view
        st.layer._placed["S"] = S_new
        st.S = S_new
        torch.cuda.synchronize()

    def s_floor(self):
        """Smallest uniform S the host side can absorb: pool slots first, then the pinned
        fallback budget (shared across kinds)."""
        if not self.layers:
            return 0
        L = len(self.layers)
        cur = sum(st.S for st in self.layers.values())
        avail = 0 if self._serving else self.pin_budget - self.pin_used
        rb = {
            n: next(
                k.rb
                for st in self.layers.values()
                for k in st.kinds.values()
                if k.name == n
            )
            for n in NAMES
            if any(n in st.kinds for st in self.layers.values())
        }
        best = -(-cur // L)  # ceil(mean S): always feasible
        for S_ in range(best, -1, -1):
            D = cur - L * S_  # rows per kind that must move to the host
            cost = sum(max(0, D - len(self.pool[n])) * rb[n] for n in rb)
            if cost > avail:
                break
            best = S_
        return best

    def set_S(self, S):
        S = max(self.s_floor(), min(int(S), self.E))
        t0 = time.time()
        torch.cuda.empty_cache()  # cuMemCreate needs driver-free memory, not torch-cached
        for lid in sorted(self.layers):
            self.resize_layer(lid, S)
        logger.info(
            "elastic: S=%d for all layers in %.2fs, VRAM free %.2f GB",
            S,
            time.time() - t0,
            torch.cuda.mem_get_info()[0] / 2**30,
        )

    def fill(self, reserve_mb, step=8):
        """Grow all layers uniformly while more than reserve_mb stays free."""
        reserve = int(reserve_mb) << 20
        torch.cuda.empty_cache()
        while True:
            cur = min(st.S for st in self.layers.values())
            if cur >= self.E:
                break
            need = sum(
                k.arena.bytes_to_reach(min(self.E, st.S + step))
                for st in self.layers.values()
                for k in st.kinds.values()
            )
            if torch.cuda.mem_get_info()[0] - need < reserve:
                break
            try:
                for lid in sorted(self.layers):
                    self.resize_layer(lid, min(self.E, self.layers[lid].S + step))
            except ArenaOOM:
                break
        logger.info(
            "elastic fill: S=%s, VRAM free %.2f GB",
            sorted({st.S for st in self.layers.values()}),
            torch.cuda.mem_get_info()[0] / 2**30,
        )

    def free(self, want_mb, step=8):
        """Shrink all layers uniformly until at least want_mb of VRAM is free."""
        want = int(want_mb) << 20
        while torch.cuda.mem_get_info()[0] < want:
            cur = max(st.S for st in self.layers.values())
            target = max(self.s_floor(), cur - step)
            if target >= cur:
                break
            try:
                for lid in sorted(self.layers):
                    if self.layers[lid].S > target:
                        self.resize_layer(lid, target)
            except (
                ArenaOOM,
                RuntimeError,
            ) as ex:  # partial shrink is still a shrink; report and stop
                logger.warning("elastic free stopped: %s", ex)
                break
        self.pending_regrow = True
        logger.info(
            "elastic free: S=%s, VRAM free %.2f GB",
            sorted({st.S for st in self.layers.values()}),
            torch.cuda.mem_get_info()[0] / 2**30,
        )

    def regrow(self):
        """Grow back into free VRAM after a forced shrink (called at a safe point: KV pool idle)."""
        self.pending_regrow = False
        self.fill(self.regrow_reserve)
        self.write_status()

    # ------------------------------------------------------------------ control
    def write_status(self):
        if not self.ctl:
            return
        free, total = torch.cuda.mem_get_info()
        arena = sum(
            k.arena.backed_bytes
            for st in self.layers.values()
            for k in st.kinds.values()
        )
        S = [self.layers[l].S for l in sorted(self.layers)]
        covered = sum(
            float(self.mass[l][self.order[l][: self.layers[l].S]].sum())
            for l in self.layers
        ) / float(self.mass.sum())
        with open(self.ctl + ".status", "w") as f:
            f.write(
                f"time {time.strftime('%H:%M:%S')}\nS_min {min(S)} S_max {max(S)} floor {self.s_floor()}\n"
                f"arena_GB {arena / 2**30:.3f}\nvram_free_GB {free / 2**30:.3f}\nmass_covered {covered:.4f}\n"
                f"pool {[len(self.pool[n]) for n in NAMES]} pin_used_MB {self.pin_used >> 20}\nS {S}\n"
            )

    def poll(self, m: int = 0, lid: int = 0):
        """Called from the MoE apply path. Acts only at layer 0 of a small eager forward (m <= 16
        rows, i.e. a 1-token prefill or eager decode): a resize inside a real prefill would strip
        that forward's working memory."""
        self._calls += 1
        if torch.cuda.is_current_stream_capturing() or lid != 0 or m > 16:
            return
        if (
            self._autofill is not None and self._calls > 48 * 4
        ):  # after warmup: grow into free VRAM
            reserve, self._autofill = self._autofill, None
            try:
                self.fill(reserve)
            except Exception as ex:
                logger.exception("elastic autofill failed: %s", ex)
            self.write_status()
        if not self.ctl:
            return
        try:
            m = os.stat(self.ctl).st_mtime
        except FileNotFoundError:
            return
        if m <= self._ctl_mtime:
            return
        self._ctl_mtime = m
        try:
            cmd = open(self.ctl).read().split()
            if not cmd:
                return
            t0 = time.time()
            if cmd[0] == "S":
                self.set_S(int(cmd[1]))
            elif cmd[0] == "fill":
                self.fill(int(cmd[1]) if len(cmd) > 1 else 512)
            elif cmd[0] == "free":
                self.free(int(cmd[1]))
            logger.info(
                "elastic ctl '%s' done in %.2fs", " ".join(cmd), time.time() - t0
            )
        except Exception as ex:  # never take the server down over a control command
            logger.exception("elastic ctl failed: %s", ex)
        self.write_status()
