"""Checkpoints for the SM70 CSA2 image.

The compressed rows are prefix-stable: a token writes ``row = pos // ratio``
and does not rewrite earlier rows. What a later token does destroy is the
128-slot SWA ring and the ratio-2 pending pair. Those are a few MiB. Saving
them at a quiescent stop (prefill chunk, prefill end, request end) lets a
later request resume at that stop instead of from position 0.

This is not the radix tree. ``--disable-radix-cache`` stays. A radix hit would
skip the forward while these buffers still held some other tail (turn-3 crash
at position 511, layer 2). The scheduler only returns a prefix the worker has
snapshotted, and the worker copies the ring and pending back before the
forward.

A DSpark request can stop on an accepted draft, and the overlap loop runs
one more verify step after the stop, so the image may run a few tokens past
the scheduler pin. The last few verify images are kept with the per-step
ratio-2 projections, so the next turn cuts the image back to the pin
(``trim_tip``) instead of reading tokens the conversation never had.

Host snaps are capped (see ``evict_recent``). A position-0 prefill still
clears this store. When ``SGLANG_DSV41_CSA2_SESSION_DIR`` is set, that clear
first spills the image and may load a different saved conversation; see
``sm70_csa2_session``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# Newest stops kept. Scheduler and worker both call ``evict_recent`` so a
# match cannot ask for a length the worker has already dropped.
BOUNDARY_KEEP = 32
# Device images kept for a trim: the stop step, the overlap loop's extra
# step, and the image before the stop step.
TRAIL_KEEP = 3


def evict_recent(lengths: Sequence[int], keep: int = BOUNDARY_KEEP) -> List[int]:
    """Keep the newest ``keep`` lengths, preserving order."""
    if keep <= 0 or len(lengths) <= keep:
        return list(lengths)
    return list(lengths[-keep:])


def cap_verify_commit(prefix: int, commit: int, pin_cap: Optional[int]) -> int:
    """How many verify tokens may land in the image.

    The scheduler pins ``len(origin) + max_new_tokens`` and drops the rest of
    a DSpark accept. Snapshotting the uncapped ``new_seq_lens`` then makes the
    next exact continuation ask for a length the worker never saved (live
    failure: pin 73, image 78, restore missed 73).
    """
    if pin_cap is None or commit <= 0:
        return max(commit, 0)
    room = int(pin_cap) - int(prefix)
    if room <= 0:
        return 0
    if commit <= room:
        return int(commit)
    return room


def _capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _clone_map(src: Dict[int, torch.Tensor], *, to_cpu: bool) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    with torch.inference_mode(False):
        for key, value in src.items():
            tensor = value.detach()
            if to_cpu:
                tensor = tensor.to(device="cpu")
            out[int(key)] = tensor.clone()
    return out


def _copy_map_into(live: Dict[int, torch.Tensor], snap: Dict[int, torch.Tensor]) -> None:
    with torch.inference_mode(False):
        for key, dest in live.items():
            src = snap.get(int(key))
            if src is None:
                continue
            dest.copy_(src.to(device=dest.device, dtype=dest.dtype))


class _StateSnap:
    __slots__ = ("swa_ring", "pending_kv", "pending_score")

    def __init__(self, state, *, to_cpu: bool) -> None:
        self.swa_ring = _clone_map(getattr(state, "swa_ring", {}), to_cpu=to_cpu)
        self.pending_kv = _clone_map(getattr(state, "pending_kv", {}), to_cpu=to_cpu)
        self.pending_score = _clone_map(
            getattr(state, "pending_score", {}), to_cpu=to_cpu
        )

    def cpu_copy(self) -> "_StateSnap":
        copied = _StateSnap.__new__(_StateSnap)
        copied.swa_ring = _clone_map(self.swa_ring, to_cpu=True)
        copied.pending_kv = _clone_map(self.pending_kv, to_cpu=True)
        copied.pending_score = _clone_map(self.pending_score, to_cpu=True)
        return copied

    def restore_into(self, state) -> None:
        _copy_map_into(state.swa_ring, self.swa_ring)
        _copy_map_into(state.pending_kv, self.pending_kv)
        _copy_map_into(state.pending_score, self.pending_score)


class _Image:
    """A device image of ``length`` tokens, as ``backup_tip`` took it.

    A verify image also keeps where its block started and the target's
    per-step ratio-2 projections, so it can be cut back to any length inside
    its accepted run.
    """

    __slots__ = ("length", "snaps", "start", "pending_kv", "pending_score")

    def __init__(
        self,
        length: int,
        snaps: Dict[str, _StateSnap],
        start: Optional[int] = None,
        pending_kv: Optional[Dict[int, torch.Tensor]] = None,
        pending_score: Optional[Dict[int, torch.Tensor]] = None,
    ) -> None:
        self.length = int(length)
        self.snaps = snaps
        self.start = start
        self.pending_kv = pending_kv or {}
        self.pending_score = pending_score or {}


def _verify_projections(paired: Sequence[tuple]):
    """Copy the target's per-step ratio-2 projections of the last verify."""
    for label, state in paired:
        if label != "target":
            continue
        kv = _clone_map(getattr(state, "verify_pending_kv_traj", {}), to_cpu=False)
        score = _clone_map(
            getattr(state, "verify_pending_score_traj", {}), to_cpu=False
        )
        return kv, score
    return {}, {}


def _cut_verify_image(
    before: Dict[str, _StateSnap], image: _Image, length: int
) -> Optional[Dict[str, _StateSnap]]:
    """The image at ``length``, inside ``image``'s accepted run.

    ``before`` is the image at ``image.start``. Ring slots of positions
    ``[start, length)`` come from ``image``; the rest of the window is still
    in ``before``, because a verify block is shorter than the ring. Pending
    is the projection of token ``length - 1``.
    """
    step = length - 1 - int(image.start)
    out: Dict[str, _StateSnap] = {}
    with torch.inference_mode(False):
        for label, snap in image.snaps.items():
            base = before.get(label)
            if base is None:
                if label == "target":
                    return None
                continue
            cut = _StateSnap.__new__(_StateSnap)
            cut.swa_ring = {}
            for lid, ring in snap.swa_ring.items():
                old = base.swa_ring.get(lid)
                if old is None:
                    return None
                slots = torch.remainder(
                    torch.arange(int(image.start), length, device=ring.device),
                    int(ring.shape[0]),
                )
                merged = old.clone()
                merged.index_copy_(0, slots, ring.index_select(0, slots))
                cut.swa_ring[lid] = merged
            cut.pending_kv = {}
            cut.pending_score = {}
            for lid in snap.pending_kv:
                kv = image.pending_kv.get(lid)
                score = image.pending_score.get(lid)
                if kv is None or score is None or step >= int(kv.shape[0]):
                    return None
                cut.pending_kv[lid] = kv[step].clone()
                cut.pending_score[lid] = score[step].clone()
            out[label] = cut
    return out


def _paired(backends: Sequence) -> List[tuple]:
    """``(label, csa2_state)`` for backends that already have SM70 state.

    Backends are target, then draft. A missing draft state is skipped so a
    snap taken before the draft ring existed can still restore the target.
    """
    labels = ("target", "draft")
    paired = []
    for index, backend in enumerate(backends):
        if backend is None:
            continue
        state = getattr(backend, "_sm70_csa2", None)
        if state is None:
            continue
        label = labels[index] if index < len(labels) else str(index)
        paired.append((label, state))
    return paired


class Csa2BoundaryStore:
    """One resident sequence. Snaps are keyed ``target`` / ``draft``."""

    def __init__(self) -> None:
        self.resident_end = 0
        self.tip_len = 0
        self.tip_from_extend = False
        self.tip: Dict[str, _StateSnap] = {}
        self.history: Dict[int, Dict[str, _StateSnap]] = {}
        self.order: List[int] = []
        # Newest device images, oldest first; the last one is ``tip``.
        self.trail: List[_Image] = []

    def clear(self) -> None:
        self.resident_end = 0
        self.tip_len = 0
        self.tip_from_extend = False
        self.tip = {}
        self.history = {}
        self.order = []
        self.trail = []

    def backup_tip(
        self,
        paired: Sequence[tuple],
        seq_len: int,
        *,
        from_extend: bool,
        start: Optional[int] = None,
    ) -> None:
        """Snapshot the live image. ``start`` is a verify block's prefix."""
        if _capturing() or seq_len < 0 or not paired:
            return
        self.tip = {
            label: _StateSnap(state, to_cpu=False) for label, state in paired
        }
        self.tip_len = int(seq_len)
        self.resident_end = int(seq_len)
        self.tip_from_extend = bool(from_extend)
        if from_extend or start is None:
            self.trail = [_Image(seq_len, self.tip)]
            return
        kv, score = _verify_projections(paired)
        self.trail.append(_Image(seq_len, self.tip, int(start), kv, score))
        del self.trail[:-TRAIL_KEEP]

    def trim_tip(self, paired: Sequence[tuple], length: int) -> bool:
        """Cut the verify tip back to ``length`` and install that image.

        The tokens past ``length`` were committed after the request stopped.
        False when the trail does not reach back to ``length``.
        """
        length = int(length)
        image = None
        for index in range(len(self.trail) - 1, -1, -1):
            entry = self.trail[index]
            if entry.length == length:
                image = entry.snaps
                break
            if entry.start is None or not entry.start < length < entry.length:
                continue
            if index == 0 or self.trail[index - 1].length != entry.start:
                return False
            image = _cut_verify_image(self.trail[index - 1].snaps, entry, length)
            break
        if not image or "target" not in image:
            return False
        for label, state in paired:
            snap = image.get(label)
            if snap is not None:
                snap.restore_into(state)
        self.tip = image
        self.tip_len = length
        self.resident_end = length
        self.tip_from_extend = False
        self.trail = [_Image(length, image)]
        return True

    def relabel_tip(self, length: int) -> None:
        """Name the current image with a shorter scheduler length.

        Last resort when ``trim_tip`` cannot rebuild the image at the pin.
        The ring slots and ratio-2 pending then still hold the tokens past
        ``length``, which the next turn never had.
        """
        self.tip_len = int(length)
        self.resident_end = int(length)
        self.trail = []

    def _verify_tip_is_pin_slack(self, start: int) -> bool:
        if (
            not self.tip
            or self.tip_from_extend
            or self.tip_len <= start
            or start in self.history
            or self.tip_len in self.history
        ):
            return False
        return not any(start < length < self.tip_len for length in self.history)

    def freeze_tip(self) -> None:
        """Copy the on-device tip into the host history once."""
        if self.tip_len <= 0 or not self.tip or self.tip_len in self.history:
            self.tip_from_extend = False
            return
        self.history[self.tip_len] = {
            label: snap.cpu_copy() for label, snap in self.tip.items()
        }
        self.order = [n for n in self.order if n != self.tip_len]
        self.order.append(self.tip_len)
        self.order = evict_recent(self.order)
        for length in [n for n in self.history if n not in self.order]:
            del self.history[length]
        self.tip_from_extend = False
        logger.info(
            "csa2 boundary freeze len=%d history=%s", self.tip_len, self.order
        )

    def restore(self, paired: Sequence[tuple], length: int) -> None:
        if length == self.tip_len and self.tip and length not in self.history:
            snaps = self.tip
        else:
            snaps = self.history.get(int(length))
        if not snaps or "target" not in snaps:
            raise RuntimeError(
                "CSA2 boundary restore missed length "
                f"{length}; tip={self.tip_len} history={self.order}"
            )
        for label, state in paired:
            snap = snaps.get(label)
            if snap is None:
                if label == "target":
                    raise RuntimeError(
                        f"CSA2 boundary snap {length} has no target ring"
                    )
                continue
            snap.restore_into(state)
        self.resident_end = int(length)
        self.trail = []
        for length_i in [n for n in self.order if n > length]:
            self.history.pop(length_i, None)
        self.order = [n for n in self.order if n <= length]
        if self.tip_len > length:
            self.tip = {}
            self.tip_len = int(length)
        logger.info("csa2 boundary restore len=%d", length)


# This rank's target store. The scheduler runs in the same process and asks
# it how far the image reaches when it pins a finished request.
_RESIDENT_STORE: Optional[Csa2BoundaryStore] = None


def _store(backends: Sequence) -> Optional[Csa2BoundaryStore]:
    global _RESIDENT_STORE
    if not backends or backends[0] is None:
        return None
    backend = backends[0]
    store = getattr(backend, "_csa2_boundary", None)
    if store is None:
        store = Csa2BoundaryStore()
        backend._csa2_boundary = store
    _RESIDENT_STORE = store
    return store


def csa2_image_len() -> Optional[int]:
    """Tokens in this rank's newest CSA2 image, or None without one."""
    store = _RESIDENT_STORE
    if store is None or not store.tip:
        return None
    return int(store.tip_len)


def _cut_tip_to_pin(
    store: Csa2BoundaryStore, paired: Sequence[tuple], pin: int, when: str
) -> None:
    tip = store.tip_len
    if store.trim_tip(paired, pin):
        logger.info("csa2 boundary trim verify snap %d to pin %d%s", tip, pin, when)
        return
    logger.warning(
        "csa2 boundary relabel verify snap %d as pin %d%s: no image to cut "
        "back to, the ring and pending keep %d tokens past the pin "
        "(trail length/start %s)",
        tip,
        pin,
        when,
        tip - pin,
        [(image.length, image.start) for image in store.trail],
    )
    store.relabel_tip(pin)


def _file_spill_under_pin(
    store: Csa2BoundaryStore, paired: Sequence[tuple], key: str
) -> None:
    """Cut the verify image back to the scheduler pin before it is written out.

    A DSpark accept can leave the worker tip a few tokens past the pin. The
    next resident extend cuts it back. A session switch spills first, so the
    file would otherwise omit the pin length and the resume would raise
    ``restore missed length``.
    """
    from sglang.srt.layers.attention.dsv4.sm70_csa2_session import load_meta

    meta = load_meta(key)
    pin = len(meta.get("ids") or []) if meta else 0
    if pin > 0 and store._verify_tip_is_pin_slack(pin):
        _cut_tip_to_pin(store, paired, pin, " before spill")
    if store.tip_len > 0:
        store.freeze_tip()
    if pin <= 0 or pin in store.history or pin == store.resident_end:
        return
    longer = [n for n in store.order if n > pin]
    if len(longer) != 1:
        return
    worker_len = longer[0]
    if any(pin < n < worker_len for n in store.order):
        return
    snap = store.history.get(worker_len)
    if not snap:
        return
    store.history[pin] = store.history.pop(worker_len)
    store.order = [pin if n == worker_len else n for n in store.order]
    if store.resident_end == worker_len:
        store.resident_end = pin
    if store.tip_len == worker_len:
        store.tip_len = pin
    logger.info("csa2 boundary move snap %d onto pin %d", worker_len, pin)


def csa2_prepare_extend(backends: Sequence, start: int) -> None:
    """Call before an extend forward. ``start`` is the cached prefix length."""
    if _capturing():
        return
    store = _store(backends)
    if store is None:
        return
    start = int(start)
    paired = _paired(backends)
    from sglang.srt.layers.attention.dsv4.sm70_csa2_session import (
        load_resident,
        save_resident,
        take_handoff,
    )

    spill, load = take_handoff()
    if spill and paired:
        _file_spill_under_pin(store, paired, spill)
        save_resident(paired, store, spill)
    if start <= 0:
        if store.resident_end or store.history or store.tip_len:
            logger.info(
                "csa2 boundary clear (prefix 0, had resident=%d)",
                store.resident_end,
            )
        store.clear()
        return
    if not paired:
        return
    if load:
        if not load_resident(paired, store, load):
            raise RuntimeError(
                f"CSA2 session load missed key {load[:12]} at prefix {start}"
            )
    # A verify image ends at prefix+commit. When the request stopped on an
    # accepted draft, or the overlap loop verified once more after the stop,
    # that is past the scheduler pin (pin 23696, tip 23698). Cut it back to
    # the pin when nothing else was snapshotted in between. An extend snap
    # stays under its own length: those positions are real tokens.
    if store._verify_tip_is_pin_slack(start):
        _cut_tip_to_pin(store, paired, start, "")
    if store.tip_len > 0:
        store.freeze_tip()
    if start == store.resident_end:
        return
    if start > store.resident_end:
        raise RuntimeError(
            "CSA2 boundary extend starts past the resident image "
            f"(start={start} resident={store.resident_end})"
        )
    store.restore(paired, start)


def csa2_prepare_decode(backends: Sequence) -> None:
    """Freeze a just-finished prefill before decode overwrites the tip."""
    if _capturing():
        return
    backend = backends[0] if backends else None
    store = getattr(backend, "_csa2_boundary", None) if backend is not None else None
    if store is None or not store.tip_from_extend:
        return
    store.freeze_tip()


def csa2_finish_forward(
    backends: Sequence,
    seq_len: int,
    *,
    from_extend: bool,
    start: Optional[int] = None,
) -> None:
    """Snapshot the image after a forward. ``start`` is a verify's prefix."""
    if _capturing():
        return
    paired = _paired(backends)
    if not paired:
        return
    store = _store(backends)
    if store is None:
        return
    store.backup_tip(paired, int(seq_len), from_extend=from_extend, start=start)


def csa2_boundary_clear(backend) -> None:
    store = getattr(backend, "_csa2_boundary", None)
    if store is not None:
        store.clear()
