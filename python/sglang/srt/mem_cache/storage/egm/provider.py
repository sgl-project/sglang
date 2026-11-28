# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Provider process of the EGM storage tier.

Allocates host memory on this machine's host NUMA nodes with a fabric handle,
publishes the handles in a file, heartbeats in the segment's control block, and
frees the memory at stop::

    python -m sglang.srt.mem_cache.storage.egm.provider run <outdir> \\
        --node-index N --gib-per-numa G --numas 0,1 --dev D --ttl SECONDS \\
        [--hb-s 0.5] [--grace-s 60] [--detach]

Stop paths: the file ``<outdir>/STOP``, or the deadline (start + ttl - stop
reserve). On stop the control block goes to STOPPING; the provider waits up to
``--grace-s`` for every attached rank to acknowledge its detach (the rank
unmaps first), then unmaps, releases and exits. A watchdog timer and a
faulthandler timer end the process at the deadline even if the main thread hangs.

Files written, only under ``<outdir>``: ``handle_<node>.json`` (fabric handles,
sizes, data area, epoch, deadline, body sha256), ``provider_status.log``, and
``provider_result.json`` (written at READY with status RUNNING, rewritten at
the end with memory snapshots, timings and rank acknowledgements).
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import socket
import sys
import threading
import time

from sglang.srt.mem_cache.storage.egm import store

GiB = 1 << 30
STOP_RESERVE_S = 20.0
_RANK_AREA_BYTES = store.CTRL_BYTES - store.RANK_AREA_OFF
_ACK_SCAN_PERIOD_S = 10.0


def utc(t=None) -> str:
    t = time.time() if t is None else t
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def utc_ms(t=None) -> str:
    t = time.time() if t is None else t
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".%03dZ" % int(
        (t % 1.0) * 1000
    )


def _read(path: str, limit: int = 200000) -> str:
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read(limit)
    except OSError as e:
        return "ERR %s" % e


def memsnap() -> dict:
    """Machine and cgroup memory: meminfo, per host NUMA node, cgroup v2."""
    out = {"t": utc(), "meminfo": {}, "nodes": {}, "cgroup": {}}
    keep = (
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Cached",
        "KReclaimable",
        "SUnreclaim",
        "Unevictable",
        "Mlocked",
        "SwapTotal",
        "SwapFree",
    )
    for line in _read("/proc/meminfo").splitlines():
        k = line.split(":")[0]
        if k in keep:
            out["meminfo"][k] = line.split(":")[1].strip()
    base = "/sys/devices/system/node"
    try:
        nodes = sorted(
            d for d in os.listdir(base) if d.startswith("node") and d[4:].isdigit()
        )
    except OSError:
        nodes = []
    for d in nodes:
        info = {}
        for line in _read(os.path.join(base, d, "meminfo")).splitlines():
            parts = line.split()
            field = parts[2].rstrip(":") if len(parts) >= 4 else ""
            if field in ("MemTotal", "MemFree", "MemUsed", "KReclaimable"):
                info[field] = parts[3] + " kB"
        out["nodes"][d] = info
    for f in ("memory.current", "memory.max", "memory.swap.current"):
        out["cgroup"][f] = _read("/sys/fs/cgroup/" + f, 200).strip()
    return out


class Status:
    def __init__(self, path: str):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = "%s %s\n" % (utc_ms(), msg)
        with open(self.path, "a") as fh:
            fh.write(line)
        sys.stdout.write(line)
        sys.stdout.flush()


def _detach_stdio(path: str) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    nul = os.open(os.devnull, os.O_RDONLY)
    os.dup2(nul, 0)
    os.dup2(fd, 1)
    os.dup2(fd, 2)


def _wjson(path: str, obj) -> None:
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)


def _watchdog(deadline: float, code: int = 97) -> None:
    def fire():
        try:
            sys.stdout.write("%s WATCHDOG os._exit(%d)\n" % (utc(), code))
            sys.stdout.flush()
        finally:
            os._exit(code)

    t = threading.Timer(max(0.0, deadline - time.time()), fire)
    t.daemon = True
    t.start()


def rank_blocks(raw: bytes, epoch: int) -> dict:
    """{rank: state} of the rank blocks in the control area that echo this epoch."""
    out = {}
    for r in range(_RANK_AREA_BYTES // store.RANK_SLOT_BYTES):
        off = r * store.RANK_SLOT_BYTES
        b = store.unpack_rank_block(raw[off : off + store.RANK_SLOT_BYTES])
        if b is not None and b.epoch_echo == epoch and b.rank == r:
            out[r] = b.state
    return out


class _Segment:
    """The provider's view of its segment: control-block writes and rank scans."""

    def __init__(self, small, va: int, epoch: int, data_off: int, data_size: int):
        self.small = small
        self.va = va
        self.epoch = epoch
        self.data_off = data_off
        self.data_size = data_size
        self.hb = 0

    def beat(self, state: int, stop_at: float) -> None:
        self.small.write(
            self.va,
            store.pack_provider_block(
                state,
                self.epoch,
                self.hb,
                self.data_off,
                self.data_size,
                int(stop_at * 1e9),
            ),
        )

    def ranks(self) -> dict:
        raw = self.small.read(self.va + store.RANK_AREA_OFF, _RANK_AREA_BYTES)
        return rank_blocks(raw, self.epoch)


def _publish_handle(
    args, seg, R: dict, epoch: int, data_off: int, stop_at: float
) -> str:
    H = {
        "kind": store.HANDLE_KIND,
        "version": 1,
        "node_index": args.node_index,
        "epoch": "%x" % epoch,
        "gran": seg.gran,
        "total_size": seg.total,
        "data_off": data_off,
        "data_size": seg.total - data_off,
        "allocs": [
            {k: a[k] for k in ("numa", "size", "off", "fabric")} for a in seg.allocs
        ],
        "host": R["host"],
        "deadline": stop_at,
        "written": utc(),
    }
    H["body_sha256"] = store.handle_body_sha(H)
    hpath = os.path.join(args.outdir, "handle_%d.json" % args.node_index)
    _wjson(hpath, H)
    with open(hpath, "rb") as fh:
        hsha = hashlib.sha256(fh.read()).hexdigest()
    R.update(
        {
            "handle": hpath,
            "handle_sha256": hsha,
            "handle_body_sha256": H["body_sha256"],
            "epoch": "%x" % epoch,
            "data_off": data_off,
            "data_size": seg.total - data_off,
            "total_size": seg.total,
            "gran": seg.gran,
            "status": "OK",
            "ready": utc_ms(),
        }
    )
    return hpath


def _serve(args, cs: _Segment, R: dict, st: Status, stop_at: float) -> None:
    """Heartbeat until STOP or the deadline, then wait for every rank's detach."""
    stop = os.path.join(args.outdir, "STOP")
    last_scan = 0.0
    while time.time() < stop_at and not os.path.exists(stop):
        cs.hb += 1
        cs.beat(store.PROV_RUN, stop_at)
        if time.time() - last_scan >= _ACK_SCAN_PERIOD_S:
            last_scan = time.time()
            seen = cs.ranks()
            if seen != R["acks"]:
                st("PROVIDER ranks %s" % seen)
                R["acks"] = seen
        time.sleep(args.hb_s)
    R["stop_reason"] = "STOP" if os.path.exists(stop) else "deadline"
    st("PROVIDER STOPPING (%s)" % R["stop_reason"])
    t_wait = time.time()
    deadline = R["deadline"]
    while True:
        cs.hb += 1
        cs.beat(store.PROV_STOPPING, stop_at)
        seen = cs.ranks()
        attached = [r for r, s in seen.items() if s == store.RANK_ATTACHED]
        if not attached:
            break
        if (
            time.time() - t_wait > args.grace_s
            or time.time() > deadline - STOP_RESERVE_S
        ):
            R["ack_missing"] = attached
            st(
                "PROVIDER ranks %s did not acknowledge within %.0f s; "
                "releasing anyway" % (attached, args.grace_s)
            )
            break
        time.sleep(args.hb_s)
    R["acks_final"] = seen
    R["ack_wait_s"] = time.time() - t_wait
    cs.beat(store.PROV_STOPPED, stop_at)


def _release(R: dict, small, seg, ctx) -> None:
    errs = []
    if small is not None:
        try:
            small.close()
        except Exception as e:
            errs.append("small: %r" % (e,))
    if seg is not None:
        errs += seg.close()
    if ctx is not None:
        try:
            ctx.release()
        except Exception as e:
            errs.append("ctx: %r" % (e,))
    R["free_errors"] = errs


def run(args) -> int:
    from sglang.srt.mem_cache.storage.egm import cuda_vmm

    t_start = time.time()
    deadline = t_start + args.ttl
    _watchdog(deadline + 8)
    faulthandler.dump_traceback_later(args.ttl + 15, exit=True)
    os.makedirs(args.outdir, exist_ok=True)
    if args.detach:
        _detach_stdio(os.path.join(args.outdir, "provider.stdout"))
    st = Status(os.path.join(args.outdir, "provider_status.log"))
    numas = [int(x) for x in args.numas.split(",") if x != ""]
    R = {
        "host": socket.gethostname(),
        "start": utc(t_start),
        "deadline": deadline,
        "ttl": args.ttl,
        "node_index": args.node_index,
        "numas": numas,
        "gib_per_numa": args.gib_per_numa,
        "dev": args.dev,
        "mem": {"before": memsnap()},
        "status": "RUNNING",
        "acks": {},
    }
    st(
        "PROVIDER START node %d numas %s gib/numa %s dev %d ttl %s deadline %s"
        % (
            args.node_index,
            numas,
            args.gib_per_numa,
            args.dev,
            args.ttl,
            utc(deadline),
        )
    )
    seg = ctx = small = None
    epoch = int.from_bytes(os.urandom(8), "little") | 1
    try:
        ctx = cuda_vmm.Context(args.dev)
        t = time.perf_counter()
        seg = cuda_vmm.ExportedSegment(ctx, numas, int(args.gib_per_numa * GiB))
        R["alloc_s"] = time.perf_counter() - t
        R["allocs"] = [
            {k: a[k] for k in ("numa", "size", "off", "create_s")} for a in seg.allocs
        ]
        R["mem"]["after_alloc"] = memsnap()
        small = cuda_vmm.SmallIO(ctx)
        data_off = max(store.CTRL_BYTES, seg.gran)
        cs = _Segment(small, seg.va, epoch, data_off, seg.total - data_off)
        small.write(seg.va + store.RANK_AREA_OFF, bytes(_RANK_AREA_BYTES))
        stop_at = deadline - STOP_RESERVE_S - args.grace_s
        cs.beat(store.PROV_RUN, stop_at)
        hpath = _publish_handle(args, seg, R, epoch, data_off, stop_at)
        # The record exists from READY on, so a provider that dies before its
        # finally block still leaves its allocation and memory snapshots.
        result_path = os.path.join(args.outdir, "provider_result.json")
        _wjson(result_path, dict(R, status="RUNNING"))
        st(
            "PROVIDER READY %s sha256 %s epoch %x total %.1f GiB data %.1f GiB "
            "(alloc %.1f s); serving until STOP or %s"
            % (
                hpath,
                R["handle_sha256"],
                epoch,
                seg.total / GiB,
                cs.data_size / GiB,
                R["alloc_s"],
                utc(stop_at),
            )
        )
        _serve(args, cs, R, st, stop_at)
    except Exception as e:
        R["status"] = "ERROR"
        R["error"] = repr(e)
        st("PROVIDER ERROR %r" % (e,))
    finally:
        tf = time.perf_counter()
        _release(R, small, seg, ctx)
        R["free_s"] = time.perf_counter() - tf
        R["mem"]["after_free"] = memsnap()
        w = min(10.0, deadline + 5.0 - time.time())
        if w > 0:
            time.sleep(w)
            R["mem"]["after_free_wait"] = memsnap()
        R["end"] = utc()
        _wjson(os.path.join(args.outdir, "provider_result.json"), R)
        st(
            "PROVIDER END status %s free %.1f s errors %s"
            % (R["status"], R["free_s"], R["free_errors"])
        )
        faulthandler.cancel_dump_traceback_later()
    return 0 if R["status"] == "OK" else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sglang.srt.mem_cache.storage.egm.provider"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("outdir")
    r.add_argument("--node-index", type=int, required=True)
    r.add_argument("--gib-per-numa", type=float, required=True)
    r.add_argument(
        "--numas",
        required=True,
        help="host NUMA ids in placement order; the first holds the control block",
    )
    r.add_argument("--dev", type=int, default=0)
    r.add_argument("--ttl", type=float, required=True)
    r.add_argument("--hb-s", type=float, default=0.5)
    r.add_argument("--grace-s", type=float, default=60.0)
    r.add_argument("--detach", action="store_true")
    a = ap.parse_args(argv)
    if a.ttl <= STOP_RESERVE_S + a.grace_s + 5:
        ap.error("--ttl must exceed the stop reserve + grace")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
