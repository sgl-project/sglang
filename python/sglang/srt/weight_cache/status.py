# SPDX-License-Identifier: Apache-2.0
"""``python -m sglang.srt.weight_cache.status``: inspect running weight cache daemons.

Read-only. For every daemon discovered on this host (or the ones named with
``--device-uuid`` / ``--socket``), connect to its Unix socket, ask for a status
snapshot, and print it. When a daemon's ``.ready`` file exists but its socket
does not answer (wedged, or died without cleanup), the row falls back to the
on-disk ``.ready`` contents and is marked ``UNREACHABLE``. A daemon that answers
but predates the ``status`` request is marked ``UNSUPPORTED``.

Examples::

    # Every daemon on this host, human-readable
    python -m sglang.srt.weight_cache.status

    # One GPU's daemon, machine-readable
    python -m sglang.srt.weight_cache.status --device-uuid GPU-abcd... --json

    # An explicit socket path (skips discovery)
    python -m sglang.srt.weight_cache.status --socket /tmp/sglang_weight_cache_GPU-abcd....sock

Exit codes: ``0`` at least one daemon answered (even if ``UNSUPPORTED``);
``3`` no daemon found; ``4`` daemons were found but none answered.
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

from sglang.srt.weight_cache.protocol import (
    get_ready_path,
    get_socket_path,
    iter_daemon_device_uuids,
    query_daemon_status,
    read_ready_file,
)

EXIT_OK = 0
EXIT_NO_DAEMONS = 3
EXIT_ALL_UNREACHABLE = 4


def collect(
    targets: List[Dict[str, Optional[str]]], *, timeout: float
) -> List[Dict[str, Any]]:
    # Each target is {"label", "socket_path", "ready_path"}; ready_path optional.
    # "reachable" means the socket answered with a well-formed reply, even an
    # error one; the daemon's own "status" field says whether it was "ok".
    rows: List[Dict[str, Any]] = []
    for target in targets:
        socket_path = target["socket_path"]
        ready_path = target.get("ready_path")
        row: Dict[str, Any] = {
            "label": target["label"],
            "socket_path": socket_path,
            "ready_path": ready_path,
        }
        try:
            row.update(query_daemon_status(socket_path, timeout=timeout))
            row["reachable"] = True
            if row.get("status") != "ok":
                row["error"] = (
                    row.get("message") or f"reply status {row.get('status')!r}"
                )
        except (OSError, ValueError) as exc:
            row["reachable"] = False
            row["error"] = str(exc) or exc.__class__.__name__
            ready_info = read_ready_file(ready_path) if ready_path else None
            if ready_info:
                row["ready_pid"] = ready_info.get("pid")
                row["ready_config"] = ready_info.get("config")
        rows.append(row)
    return rows


def _resolve_targets(args: argparse.Namespace) -> List[Dict[str, Optional[str]]]:
    targets: List[Dict[str, Optional[str]]] = []
    for sock in args.sockets or []:
        targets.append({"label": sock, "socket_path": sock, "ready_path": None})
    # Explicit --socket targets are always included. --device-uuid targets are
    # appended when named, or, if nothing was named, every discovered daemon.
    uuids = args.device_uuids or ([] if args.sockets else iter_daemon_device_uuids())
    for uuid in uuids:
        targets.append(
            {
                "label": uuid,
                "socket_path": get_socket_path(uuid),
                "ready_path": get_ready_path(uuid),
            }
        )
    return targets


def render_human(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No weight cache daemons found."
    out: List[str] = []
    for row in rows:
        label = row["label"]
        if not row.get("reachable"):
            ready_pid = row.get("ready_pid")
            pid_note = f"ready pid {ready_pid}, " if ready_pid else ""
            out.append(
                f"{label}  UNREACHABLE  ({pid_note}socket not answering: {row.get('error')})"
            )
            out.append(_field("socket", row["socket_path"]))
            out.append("")
            continue
        if row.get("status") != "ok":
            out.append(
                f"{label}  UNSUPPORTED  (daemon answered but rejected the status "
                f"request, upgrade it: {row.get('error')})"
            )
            out.append(_field("socket", row["socket_path"]))
            out.append("")
            continue

        cfg = row.get("config") or {}
        out.append(
            f"{label}  pid {row.get('pid')}  "
            f"up {_format_duration(row.get('uptime_seconds'))}"
        )
        fields = [
            (
                "model",
                f"{cfg.get('model_path', '?')} (arch={cfg.get('model_arch', '?')})",
            ),
            (
                "parallel",
                f"tp {cfg.get('tp_size', '?')}/{cfg.get('tp_rank', '?')}  "
                f"pp {cfg.get('pp_size', '?')}/{cfg.get('pp_rank', '?')}  "
                f"dp {cfg.get('dp_size', '?')}  ep {cfg.get('ep_size', '?')}",
            ),
            (
                "quant",
                f"{cfg.get('quant_method') or 'none'}  dtype {cfg.get('dtype', '?')}",
            ),
            (
                "transport",
                f"{row.get('transport_backend') or 'n/a'}  "
                f"tensors {row.get('num_tensors', 0)}",
            ),
            ("preloaded", _format_bytes(row.get("preloaded_weights_bytes"))),
            (
                "load",
                f"{_format_duration(row.get('load_seconds'))}  "
                f"(loaded {_format_age(row.get('loaded_at'))})",
            ),
            (
                "serves",
                f"hit {row.get('serve_count', 0)}  "
                f"mismatch {row.get('mismatch_count', 0)}  "
                f"(last {_format_age(row.get('last_served_at'))})",
            ),
            (
                "clients",
                f"live {row.get('live_client_count', 0)} "
                f"{row.get('live_client_pids') or []}",
            ),
            ("socket", row["socket_path"]),
        ]
        out.extend(_field(name, value) for name, value in fields)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sglang.srt.weight_cache.status",
        description="Inspect running SGLang weight cache daemons.",
    )
    parser.add_argument(
        "--device-uuid",
        action="append",
        dest="device_uuids",
        metavar="UUID",
        help="Inspect only this GPU's daemon (repeatable). Default: every daemon on the host.",
    )
    parser.add_argument(
        "--socket",
        action="append",
        dest="sockets",
        metavar="PATH",
        help="Inspect a daemon at an explicit socket path (repeatable); skips discovery.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON array instead of the human-readable table.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-daemon socket timeout in seconds (default: 5).",
    )
    args = parser.parse_args(argv)

    targets = _resolve_targets(args)
    if not targets:
        if args.json:
            print("[]")
        else:
            print("No weight cache daemons found.", file=sys.stderr)
        return EXIT_NO_DAEMONS

    rows = collect(targets, timeout=args.timeout)

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True, default=str))
    else:
        print(render_human(rows), end="")

    return EXIT_OK if any(r.get("reachable") for r in rows) else EXIT_ALL_UNREACHABLE


def _field(label: str, value: str) -> str:
    """One aligned ``label  value`` line of the human-readable table."""
    return f"    {label:<11}{value}"


def _format_bytes(n: Optional[int]) -> str:
    if not n:
        return "0 B"
    val = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if val < 1024.0:
            return f"{val:.0f} B" if unit == "B" else f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} PiB"


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


def _format_age(epoch: Optional[float]) -> str:
    if not epoch:
        return "never"
    delta = time.time() - epoch
    if delta < 0:
        return "just now"
    return f"{_format_duration(delta)} ago"


if __name__ == "__main__":
    sys.exit(main())
