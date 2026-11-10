"""N on-disk CSA2 conversations. Radix stays off.

One image stays on the GPU. A miss spills that image before prefix 0
overwrites ``row = pos // ratio``, and a later request that shares a recorded
stop loads it back. The catalog is the token ids and cut lengths. LRU drops
whole conversations. This does not share rows across branches.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
from array import array
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

SCHEMA = 1
_SPILL: Optional[str] = None
_LOAD: Optional[str] = None


def sessions_enabled() -> bool:
    from sglang.srt.environ import envs

    directory = envs.SGLANG_DSV41_CSA2_SESSION_DIR.get()
    return bool(directory)


def session_keep() -> int:
    from sglang.srt.environ import envs

    return max(1, int(envs.SGLANG_DSV41_CSA2_SESSION_KEEP.get()))


def rank_id() -> int:
    """TP rank of this process. Scheduler workers do not set ``RANK``."""
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().tp_rank)
    except Exception:
        pass
    for name in ("RANK", "LOCAL_RANK"):
        raw = os.environ.get(name)
        if raw:
            return int(raw)
    return 0


def session_key(
    ids: Sequence[int], extra_key: Optional[str], cache_salt: Optional[str]
) -> str:
    digest = hashlib.blake2s()
    tokens = array("q", (int(token) for token in ids))
    digest.update(len(tokens).to_bytes(8, "little"))
    digest.update(tokens.tobytes())
    digest.update(b"\0")
    digest.update((extra_key or "").encode())
    digest.update(b"\0")
    digest.update((cache_salt or "").encode())
    return digest.hexdigest()


def reset_handoff() -> None:
    global _SPILL, _LOAD
    _SPILL = None
    _LOAD = None


def request_spill(key: str) -> None:
    global _SPILL
    _SPILL = str(key)


def request_load(key: str) -> None:
    global _LOAD
    _LOAD = str(key)


def take_handoff() -> Tuple[Optional[str], Optional[str]]:
    global _SPILL, _LOAD
    spill, load = _SPILL, _LOAD
    _SPILL = None
    _LOAD = None
    return spill, load


def _root() -> Path:
    from sglang.srt.environ import envs

    directory = envs.SGLANG_DSV41_CSA2_SESSION_DIR.get()
    if not directory:
        raise RuntimeError("CSA2 session directory is unset")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stems(key: str) -> Tuple[Path, Path]:
    # Do not use Path.with_suffix: ``.r0`` is itself a suffix, so it would
    # replace the rank and collide every conversation onto one file.
    base = _root() / f"{key}.r{rank_id()}"
    return Path(str(base) + ".json"), Path(str(base) + ".pt")


def image_ready(key: str) -> bool:
    if not sessions_enabled():
        return False
    _meta, blob = _stems(key)
    return blob.is_file()


def _touch(path: Path) -> None:
    path.touch()


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def remember(
    key: str,
    ids: Sequence[int],
    cuts: Sequence[int],
    extra_key: Optional[str],
    cache_salt: Optional[str],
) -> None:
    """Record the token catalog. The GPU bytes are written later by the worker."""
    if not sessions_enabled():
        return
    meta, _blob = _stems(key)
    kept = [int(cut) for cut in cuts if int(cut) > 0]
    if ids and (not kept or kept[-1] != len(ids)):
        if len(ids) not in kept:
            kept.append(len(ids))
    _write_json(
        meta,
        {
            "schema": SCHEMA,
            "key": key,
            "ids": [int(token) for token in ids],
            "cuts": kept,
            "extra_key": extra_key,
            "cache_salt": cache_salt,
        },
    )
    _touch(meta)
    evict_lru(keep_key=key)


def load_meta(key: str) -> Optional[dict]:
    if not sessions_enabled():
        return None
    meta, _blob = _stems(key)
    if not meta.is_file():
        return None
    payload = json.loads(meta.read_text(encoding="utf-8"))
    if int(payload.get("schema", -1)) != SCHEMA:
        logger.info("csa2 session ignore schema %s", payload.get("schema"))
        return None
    return payload


def list_sessions() -> List[dict]:
    if not sessions_enabled():
        return []
    root = _root()
    suffix = f".r{rank_id()}.json"
    found = []
    for path in root.glob(f"*{suffix}"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(payload.get("schema", -1)) != SCHEMA:
            continue
        payload["_path"] = str(path)
        payload["_mtime"] = path.stat().st_mtime
        found.append(payload)
    found.sort(key=lambda item: item["_mtime"])
    return found


def evict_lru(keep_key: Optional[str] = None) -> None:
    if not sessions_enabled():
        return
    records = list_sessions()
    overflow = len(records) - session_keep()
    if overflow <= 0:
        return
    dropped = 0
    for record in records:
        if dropped >= overflow:
            break
        key = record.get("key")
        if key == keep_key:
            continue
        _delete(str(key))
        dropped += 1


def _delete(key: str) -> None:
    meta, blob = _stems(key)
    for path in (meta, blob):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    logger.info("csa2 session evict key=%s", key[:12])


def _cpu_map(src: Dict[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in src.items():
        out[str(int(key))] = value.detach().to(device="cpu").contiguous().clone()
    return out


def _snap_dict(snap) -> dict:
    return {
        "swa_ring": _cpu_map(snap.swa_ring),
        "pending_kv": _cpu_map(snap.pending_kv),
        "pending_score": _cpu_map(snap.pending_score),
    }


def save_resident(paired: Sequence[tuple], store, key: str) -> None:
    if not sessions_enabled() or store is None or not paired:
        return
    if int(store.resident_end) <= 0 and not store.history and int(store.tip_len) <= 0:
        return
    labels = {}
    for label, state in paired:
        labels[label] = {
            "kv_rows": _cpu_map(getattr(state, "kv_rows", {})),
            "index_rows": _cpu_map(getattr(state, "index_rows", {})),
            "swa_ring": _cpu_map(getattr(state, "swa_ring", {})),
            "pending_kv": _cpu_map(getattr(state, "pending_kv", {})),
            "pending_score": _cpu_map(getattr(state, "pending_score", {})),
        }
    history = {
        str(int(length)): {
            label: _snap_dict(snap) for label, snap in snaps.items()
        }
        for length, snaps in store.history.items()
    }
    payload = {
        "schema": SCHEMA,
        "resident_end": int(store.resident_end),
        "order": [int(length) for length in store.order],
        "labels": labels,
        "history": history,
    }
    _meta, blob = _stems(key)
    tmp = Path(str(blob) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, blob)
    # Workaround: torch.save (2.9.1) leaves its pickler class in a reference
    # cycle holding every saved storage; drop once torch.save frees them itself.
    gc.collect()
    logger.info(
        "csa2 session spill key=%s len=%d cuts=%s",
        key[:12],
        store.resident_end,
        store.order,
    )


def _copy_map(live: Dict[int, torch.Tensor], saved: Dict[str, torch.Tensor]) -> None:
    saved_ids = {int(key) for key in saved}
    for key, dest in live.items():
        src = saved.get(str(int(key)))
        dest.zero_()
        if src is None:
            continue
        rows = min(int(dest.shape[0]), int(src.shape[0]))
        if rows <= 0:
            continue
        dest[:rows].copy_(src[:rows].to(device=dest.device, dtype=dest.dtype))
    for key in saved_ids:
        if key not in live:
            logger.info("csa2 session load skipped missing layer %s", key)


def _install_snap(state, snap: dict) -> None:
    _copy_map(getattr(state, "swa_ring", {}), snap.get("swa_ring", {}))
    _copy_map(getattr(state, "pending_kv", {}), snap.get("pending_kv", {}))
    _copy_map(getattr(state, "pending_score", {}), snap.get("pending_score", {}))


def load_resident(paired: Sequence[tuple], store, key: str) -> bool:
    if not sessions_enabled() or store is None:
        return False
    _meta, blob = _stems(key)
    if not blob.is_file():
        logger.info("csa2 session load missed key=%s", key[:12])
        return False
    try:
        payload = torch.load(blob, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(blob, map_location="cpu")
    if int(payload.get("schema", -1)) != SCHEMA:
        raise RuntimeError(
            f"CSA2 session schema {payload.get('schema')} != {SCHEMA}"
        )
    by_label = {label: state for label, state in paired}
    for label, saved in payload.get("labels", {}).items():
        state = by_label.get(label)
        if state is None:
            if label == "target":
                raise RuntimeError(f"CSA2 session {key[:12]} has no target backend")
            continue
        _copy_map(getattr(state, "kv_rows", {}), saved.get("kv_rows", {}))
        _copy_map(getattr(state, "index_rows", {}), saved.get("index_rows", {}))
        _install_snap(state, saved)
    from sglang.srt.layers.attention.dsv4.sm70_csa2_boundary import _StateSnap

    history = {}
    order = [int(length) for length in payload.get("order", [])]
    raw_history = payload.get("history", {})
    for length in order:
        snaps = raw_history.get(str(int(length)), {})
        history[int(length)] = {}
        for label, snap in snaps.items():
            built = _StateSnap.__new__(_StateSnap)
            built.swa_ring = {
                int(k): v.clone() for k, v in snap.get("swa_ring", {}).items()
            }
            built.pending_kv = {
                int(k): v.clone() for k, v in snap.get("pending_kv", {}).items()
            }
            built.pending_score = {
                int(k): v.clone() for k, v in snap.get("pending_score", {}).items()
            }
            history[int(length)][label] = built
    store.history = history
    store.order = order
    store.resident_end = int(payload.get("resident_end", 0))
    store.tip = {}
    store.tip_len = 0
    store.tip_from_extend = False
    store.trail = []
    _touch(_stems(key)[0])
    logger.info(
        "csa2 session load key=%s len=%d cuts=%s",
        key[:12],
        store.resident_end,
        store.order,
    )
    return True
