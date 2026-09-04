"""Diagnostic: snapshot every live CUDA storage before release_memory_occupation and
diff after the last resume_memory_occupation. Reports storages whose content changed,
labelled as param/buffer/pool/other, plus caching-allocator segment provenance.
Never raises into the caller."""
import gc
import json
import logging
import os
import sys
import time
import traceback

import torch

logger = logging.getLogger(__name__)
OUT_DIR = os.environ.get("TMSDIAG_OUT", os.environ.get("SGL_TMSDIAG_OUT", "/root/diag/out"))
_STATE = {"snap": None, "labels": None, "segments": None, "cycle": 0}
_MAX_SAMPLE = 8 << 20  # bytes sampled per storage

try:
    if os.environ.get("TMSDIAG_HISTORY", os.environ.get("SGL_TMSDIAG_HISTORY", "0")) == "1":
        torch.cuda.memory._record_memory_history(max_entries=200000)
except Exception as e:  # pragma: no cover
    print(f"[TMSDIAG] record_memory_history failed: {e}", file=sys.stderr, flush=True)


def _log(msg):
    try:
        dev = torch.cuda.current_device()
    except Exception:
        dev = -1
    print(f"[TMSDIAG dev{dev}] {msg}", file=sys.stderr, flush=True)


def _flat(storage, device):
    return torch.empty(0, dtype=torch.uint8, device=device).set_(storage)


def _sig(storage, device):
    n = storage.nbytes()
    f = _flat(storage, device)
    step = max(1, n // _MAX_SAMPLE)
    s = f[::step]
    nz = int((s != 0).sum().item())
    tot = int(s.to(torch.int64).sum().item())
    return {"nbytes": n, "sampled": int(s.numel()), "nonzero": nz, "sum": tot}


def _iter_storages():
    seen = set()
    frozen = gc.get_freeze_count() > 0
    if frozen:
        gc.unfreeze()
    try:
        objs = gc.get_objects()
    finally:
        if frozen:
            gc.freeze()
    for obj in objs:
        try:
            if type(obj) is not torch.Tensor and type(obj) is not torch.nn.Parameter:
                continue  # skip FakeTensor / wrapper subclasses (device=cuda, storage=meta)
            if obj.device.type != "cuda" or obj.numel() == 0:
                continue
            st = obj.untyped_storage()
            if st.device.type != "cuda":
                continue
            key = (int(st.data_ptr()), int(st.nbytes()))
            if key in seen or key[1] == 0:
                continue
            seen.add(key)
            yield key, obj, st
        except Exception:
            continue


def _label_map(mgr):
    labels = {}

    def add_named(prefix, named):
        for name, t in named:
            try:
                if t.device.type != "cuda":
                    continue
                st = t.untyped_storage()
                labels.setdefault((int(st.data_ptr()), int(st.nbytes())), f"{prefix}:{name}")
            except Exception:
                pass

    def add_obj(prefix, obj, depth=0, seen=None):
        if seen is None:
            seen = set()
        if obj is None or id(obj) in seen or depth > 6:
            return
        seen.add(id(obj))
        if torch.is_tensor(obj):
            try:
                if obj.device.type == "cuda":
                    st = obj.untyped_storage()
                    labels.setdefault((int(st.data_ptr()), int(st.nbytes())), prefix)
            except Exception:
                pass
            return
        if isinstance(obj, (list, tuple)):
            for i, x in enumerate(obj[:512]):
                add_obj(f"{prefix}[{i}]", x, depth + 1, seen)
            return
        if isinstance(obj, dict):
            for k, x in list(obj.items())[:512]:
                add_obj(f"{prefix}[{k!r}]", x, depth + 1, seen)
            return
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for k, x in list(d.items()):
                if k.startswith("__"):
                    continue
                add_obj(f"{prefix}.{k}", x, depth + 1, seen)

    try:
        model = mgr.tp_worker.model_runner.model
        add_named("target.param", model.named_parameters())
        add_named("target.buffer", model.named_buffers())
    except Exception as e:
        _log(f"target model labels failed: {e}")
    try:
        dw = getattr(mgr, "draft_worker", None)
        if dw is not None:
            dr = None
            for path in (("draft_runner",), ("model_runner",), ("draft_worker", "model_runner"), ("_draft_worker", "model_runner"), ("draft_model_runner",)):
                cur = dw
                for a in path:
                    cur = getattr(cur, a, None)
                    if cur is None:
                        break
                if cur is not None:
                    dr = cur
                    break
            if dr is None:
                raise RuntimeError(f"no draft runner on {type(dw).__name__}: {sorted(k for k in vars(dw).keys())[:40]}")
            dm = dr.model
            add_named("draft.param", dm.named_parameters())
            add_named("draft.buffer", dm.named_buffers())
    except Exception as e:
        _log(f"draft model labels failed: {e}")
    sched = getattr(mgr, "scheduler", None)
    try:
        if sched is not None:
            add_obj("pool.req_to_token_pool", sched.req_to_token_pool)
            add_obj("pool.token_to_kv_pool_allocator", sched.token_to_kv_pool_allocator)
            add_obj("pool.tree_cache", sched.tree_cache)
    except Exception as e:
        _log(f"pool labels failed: {e}")
    try:
        mr = mgr.tp_worker.model_runner
        for attr in ("attn_backend", "decode_cuda_graph_runner", "cuda_graph_runner", "sampler", "graph_runner"):
            add_obj(f"target.model_runner.{attr}", getattr(mr, attr, None))
        dw = getattr(mgr, "draft_worker", None)
        if dw is not None:
            add_obj("draft_worker", dw, depth=1)
    except Exception as e:
        _log(f"runner labels failed: {e}")
    return labels


def _segments():
    try:
        segs = []
        for s in torch.cuda.memory_snapshot():
            frames = s.get("frames") or []
            segs.append({
                "address": int(s["address"]), "size": int(s["total_size"]),
                "type": s.get("segment_type"), "frames": [f"{f.get('filename','?')}:{f.get('line','?')}" for f in frames[:12]],
                "n_blocks": len(s.get("blocks", [])),
            })
        return segs
    except Exception as e:
        _log(f"memory_snapshot failed: {e}")
        return []


def _find_segment(segs, ptr):
    for s in segs:
        if s["address"] <= ptr < s["address"] + s["size"]:
            return s
    return None


def _owner_desc(t):
    out = []
    try:
        for r in gc.get_referrers(t)[:6]:
            if isinstance(r, dict):
                keys = [k for k, v in r.items() if v is t][:3]
                owners = [type(o).__name__ for o in gc.get_referrers(r)[:6] if getattr(o, "__dict__", None) is r]
                out.append(f"dict{keys}<-{owners}")
            elif isinstance(r, (list, tuple)):
                owners = [type(o).__name__ for o in gc.get_referrers(r)[:4] if not isinstance(o, (list, tuple, dict))]
                out.append(f"{type(r).__name__}(len={len(r)})<-{owners}")
            else:
                out.append(type(r).__name__)
    except Exception as e:
        out.append(f"referrers-failed:{e}")
    return out


def _poison_free_memory():
    """Fill every byte the caching allocator can hand out with 0xFF (NaN in bf16/fp32),
    then free it back to the allocator WITHOUT empty_cache, so the next forward's
    torch.empty scratch is carved from NaN-filled blocks. Any read-before-write in the
    first post-resume forward then surfaces deterministically."""
    mode = os.environ.get("TMSDIAG_POISON", "0")
    if mode != "1":
        return
    t0 = time.time()
    torch.cuda.synchronize()
    dev = torch.cuda.current_device()
    big, small = [], []
    total_big = 0
    try:
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free < (3 << 30):
                break
            chunk = min(free - (2 << 30), 1 << 30)
            try:
                t = torch.empty(chunk, dtype=torch.uint8, device="cuda")
            except Exception:
                break
            t.fill_(255)
            big.append(t)
            total_big += chunk
    except Exception as e:
        _log(f"poison big failed: {e}")
    try:
        for _ in range(4096):  # small pool: <=1 MiB allocations, 2 MiB segments
            t = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
            t.fill_(255)
            small.append(t)
    except Exception as e:
        _log(f"poison small stopped: {e}")
    torch.cuda.synchronize()
    n_small = len(small)
    del big, small
    reserved = torch.cuda.memory_reserved() >> 20
    _log(f"POISON done: big={total_big >> 20} MiB small={n_small} MiB reserved_now={reserved} MiB took={time.time()-t0:.1f}s")


def on_release(mgr):
    try:
        t0 = time.time()
        torch.cuda.synchronize()
        dev = torch.cuda.current_device()
        snap = {}
        reps = {}
        n_fail = 0
        for key, t, st in _iter_storages():
            try:
                snap[key] = _sig(st, t.device)
                reps[key] = (tuple(t.shape), str(t.dtype), type(t).__name__)
            except Exception:
                n_fail += 1
        _log(f"snapshot signature failures: {n_fail}")
        _STATE["snap"] = snap
        _STATE["reps"] = reps
        _STATE["labels"] = _label_map(mgr)
        _STATE["segments"] = _segments()
        _STATE["cycle"] += 1
        torch.cuda.synchronize()
        _log(f"release snapshot cycle={_STATE['cycle']} storages={len(snap)} labelled={len(_STATE['labels'])} segments={len(_STATE['segments'])} took={time.time()-t0:.1f}s")
    except Exception:
        _log("on_release failed:\n" + traceback.format_exc())


def on_resume(mgr):
    try:
        if _STATE["snap"] is None:
            _log("on_resume without snapshot")
            return
        if len(getattr(mgr, "offload_tags", set())) > 0:
            _log(f"on_resume: tags still offloaded {sorted(mgr.offload_tags)}; deferring diff")
            return
        t0 = time.time()
        torch.cuda.synchronize()
        dev = torch.cuda.current_device()
        snap, reps, labels, segs = _STATE["snap"], _STATE["reps"], _STATE["labels"], _STATE["segments"]
        changed, gone, n_same = [], 0, 0
        live = {}
        for key, t, st in _iter_storages():
            live[key] = (t, st)
        for key, before in snap.items():
            if key not in live:
                gone += 1
                continue
            t, st = live[key]
            try:
                after = _sig(st, t.device)
            except Exception:
                continue
            if after["nonzero"] == before["nonzero"] and after["sum"] == before["sum"]:
                n_same += 1
                continue
            label = labels.get(key, "other")
            shape, dtype, tname = reps.get(key, ("?", "?", "?"))
            seg = _find_segment(segs, key[0])
            rec = {
                "label": label, "shape": list(shape) if shape != "?" else "?", "dtype": dtype, "nbytes": key[1],
                "before": before, "after": after,
                "now_all_zero": after["nonzero"] == 0,
                "segment": None if seg is None else {"address": hex(seg["address"]), "size": seg["size"], "type": seg["type"], "frames": seg["frames"][:8]},
            }
            if label == "other":
                rec["owner"] = _owner_desc(t)
            changed.append(rec)
        # --- non-finite scan: every live float storage, right before the first post-resume forward
        nonfinite = []
        for key, (t, st) in live.items():
            try:
                if not t.dtype.is_floating_point:
                    continue
                esz = t.element_size()
                if key[1] % esz != 0:
                    continue
                v = torch.empty(0, dtype=t.dtype, device=t.device).set_(st)
                bad = int((~torch.isfinite(v)).sum().item())
                if bad > 0:
                    lbl = labels.get(key, "other")
                    shape, dtype, _ = reps.get(key, ("?", "?", "?"))
                    rec = {"label": lbl, "shape": list(shape) if shape != "?" else list(t.shape), "dtype": str(t.dtype), "nbytes": key[1], "nonfinite": bad}
                    if lbl == "other":
                        rec["owner"] = _owner_desc(t)
                    nonfinite.append(rec)
            except Exception:
                continue
        nonfinite.sort(key=lambda r: -r["nonfinite"])
        _log(f"NONFINITE storages before first forward: {len(nonfinite)}")
        for r in nonfinite[:60]:
            _log(f"NONFINITE {r['label']} shape={r['shape']} dtype={r['dtype']} count={r['nonfinite']} owner={r.get('owner')}")
        changed.sort(key=lambda r: (0 if r["label"] == "other" else 1, -r["nbytes"]))
        counts = {}
        for r in changed:
            k = r["label"].split(":")[0].split(".")[0]
            counts[k] = counts.get(k, 0) + 1
        _log(f"resume diff cycle={_STATE['cycle']} changed={len(changed)} same={n_same} gone={gone} by_class={counts} took={time.time()-t0:.1f}s")
        still_zero = [r for r in changed if r["now_all_zero"] and r["before"]["nonzero"] > 0 and not r["label"].startswith("pool.")]
        by_prefix = {}
        for r in still_zero:
            k = r["label"].split(":")[0] + ":" + (r["label"].split(":")[1].split(".")[0] if ":" in r["label"] else "?")
            by_prefix[k] = by_prefix.get(k, 0) + 1
        _log(f"STILL-ZERO (zeroed by resume, not rewritten before first forward): {len(still_zero)} storages by_prefix={by_prefix}")
        for r in still_zero[:80]:
            _log(f"STILL-ZERO {r['label']} shape={r['shape']} dtype={r['dtype']} nbytes={r['nbytes']} owner={r.get('owner')}")
        for r in changed:
            if r["label"] == "other" or not r["label"].startswith(("target.param", "draft.param", "pool.")):
                _log(f"CHANGED {r['label']} shape={r['shape']} dtype={r['dtype']} nbytes={r['nbytes']} before_nz={r['before']['nonzero']}/{r['before']['sampled']} after_nz={r['after']['nonzero']} all_zero_now={r['now_all_zero']} seg={r['segment'] and r['segment']['frames'][:3]} owner={r.get('owner')}")
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"dev{dev}_cycle{_STATE['cycle']}.json")
        with open(path, "w") as f:
            json.dump({"cycle": _STATE["cycle"], "changed": changed, "nonfinite": nonfinite, "n_same": n_same, "gone": gone, "n_segments": len(segs)}, f, indent=1)
        _log(f"wrote {path}")
        _poison_free_memory()
    except Exception:
        _log("on_resume failed:\n" + traceback.format_exc())
