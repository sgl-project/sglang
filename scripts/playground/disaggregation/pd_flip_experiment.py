#!/usr/bin/env python3
"""Drive a drain-to-idle PD flip experiment.

This helper is intentionally orchestration-light: it talks to each SGLang
worker's HTTP endpoint, observes ``/server_info`` -> ``internal_states[*].pd_flip``,
and acks ``/set_internal_state`` once the worker has drained.

Example:
  python scripts/playground/disaggregation/pd_flip_experiment.py trigger \
    --worker-url http://127.0.0.1:30001 --direction d_to_p

  python scripts/playground/disaggregation/pd_flip_experiment.py run-once \
    --worker-url http://127.0.0.1:30001 \
    --restart-command 'docker compose restart decode0'
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib import error, request
from urllib.parse import urljoin


JsonDict = Dict[str, Any]
LogFn = Callable[[str], None]
SleepFn = Callable[[float], None]


class HttpClient:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: float = 10.0):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def get_json(self, base_url: str, path: str) -> JsonDict:
        req = self._request(base_url, path, method="GET")
        return self._open_json(req)

    def post_json(self, base_url: str, path: str, payload: JsonDict) -> JsonDict:
        body = json.dumps(payload).encode("utf-8")
        req = self._request(base_url, path, method="POST", data=body)
        req.add_header("Content-Type", "application/json")
        return self._open_json(req)

    def _request(
        self,
        base_url: str,
        path: str,
        method: str,
        data: Optional[bytes] = None,
    ) -> request.Request:
        url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
        req = request.Request(url, data=data, method=method)
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        return req

    def _open_json(self, req: request.Request) -> JsonDict:
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{req.full_url} returned HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"failed to connect to {req.full_url}: {exc}") from exc


@dataclass
class WorkerSnapshot:
    url: str
    server_info: JsonDict
    pd_flip: JsonDict


def extract_pd_flip(server_info: JsonDict) -> JsonDict:
    for state in server_info.get("internal_states", []) or []:
        if isinstance(state, dict) and isinstance(state.get("pd_flip"), dict):
            return state["pd_flip"]
    pd_flip = server_info.get("pd_flip")
    return pd_flip if isinstance(pd_flip, dict) else {}


def summarize_pd_flip(pd_flip: JsonDict) -> str:
    fields = [
        ("state", pd_flip.get("state")),
        ("direction", pd_flip.get("direction")),
        ("role", pd_flip.get("current_role")),
        ("target", pd_flip.get("target_role") or pd_flip.get("requested_role")),
        ("idle", pd_flip.get("is_idle_for_flip")),
        ("paused", pd_flip.get("admission_paused")),
        ("router", pd_flip.get("router_action")),
    ]
    return " ".join(f"{name}={value}" for name, value in fields)


def is_pending_flip(pd_flip: JsonDict) -> bool:
    return pd_flip.get("state") in ("preparing", "flipping")


def is_idle_for_flip(pd_flip: JsonDict) -> bool:
    return bool(pd_flip.get("is_idle_for_flip"))


def fetch_snapshots(client: HttpClient, worker_urls: Iterable[str]) -> List[WorkerSnapshot]:
    snapshots = []
    for url in worker_urls:
        server_info = client.get_json(url, "/server_info")
        snapshots.append(
            WorkerSnapshot(
                url=url,
                server_info=server_info,
                pd_flip=extract_pd_flip(server_info),
            )
        )
    return snapshots


def set_internal_state(client: HttpClient, worker_url: str, server_args: JsonDict) -> JsonDict:
    return client.post_json(
        worker_url,
        "/set_internal_state",
        {"server_args": server_args},
    )


def ack_prepare(client: HttpClient, worker_url: str) -> JsonDict:
    return set_internal_state(client, worker_url, {"pd_flip_prepare_ack": True})


def ack_commit(client: HttpClient, worker_url: str) -> JsonDict:
    return set_internal_state(client, worker_url, {"pd_flip_commit_ack": True})


def trigger_flip(
    client: HttpClient,
    worker_url: str,
    direction: str,
    prefill_nodes: int,
    decode_nodes: int,
    threshold: float,
    window_seconds: float,
) -> JsonDict:
    if direction == "d_to_p":
        prefill_slo = 0.0
        decode_slo = 1.0
    elif direction == "p_to_d":
        prefill_slo = 1.0
        decode_slo = 0.0
    else:
        raise ValueError(f"unknown direction: {direction}")

    return set_internal_state(
        client,
        worker_url,
        {
            "pd_flip_prefill_nodes": prefill_nodes,
            "pd_flip_decode_nodes": decode_nodes,
            "pd_flip_slo_threshold": threshold,
            "pd_flip_window_seconds": window_seconds,
            "pd_flip_prefill_slo_attainment": prefill_slo,
            "pd_flip_decode_slo_attainment": decode_slo,
        },
    )


def run_once(
    client: HttpClient,
    worker_urls: List[str],
    timeout_seconds: float,
    poll_interval_seconds: float,
    restart_command: Optional[str],
    sleep_fn: SleepFn = time.sleep,
    log_fn: LogFn = print,
) -> JsonDict:
    deadline = time.monotonic() + timeout_seconds
    target_url = None
    target_role = None
    prepare_acked = False
    restart_done = False
    commit_acked = False

    while time.monotonic() < deadline:
        snapshots = fetch_snapshots(client, worker_urls)
        for snapshot in snapshots:
            if snapshot.pd_flip:
                log_fn(f"{snapshot.url}: {summarize_pd_flip(snapshot.pd_flip)}")

        pending = [s for s in snapshots if is_pending_flip(s.pd_flip)]
        if not pending:
            if prepare_acked and commit_acked:
                return {"status": "completed", "worker_url": target_url}
            if prepare_acked and restart_done and target_role:
                for snapshot in snapshots:
                    if snapshot.pd_flip.get("current_role") == target_role:
                        return {
                            "status": "completed_by_restart",
                            "worker_url": snapshot.url,
                        }
            sleep_fn(poll_interval_seconds)
            continue

        current = pending[0]
        target_url = current.url
        target_role = target_role or current.pd_flip.get("target_role")
        state = current.pd_flip.get("state")

        if not is_idle_for_flip(current.pd_flip):
            sleep_fn(poll_interval_seconds)
            continue

        if state == "preparing" and not prepare_acked:
            ack_prepare(client, current.url)
            prepare_acked = True
            log_fn(f"{current.url}: sent pd_flip_prepare_ack")
            sleep_fn(poll_interval_seconds)
            continue

        if state == "flipping" and restart_command and not restart_done:
            run_restart_command(restart_command, log_fn=log_fn)
            restart_done = True
            sleep_fn(poll_interval_seconds)
            continue

        if state == "flipping" and not commit_acked:
            ack_commit(client, current.url)
            commit_acked = True
            log_fn(f"{current.url}: sent pd_flip_commit_ack")
            sleep_fn(poll_interval_seconds)
            continue

        if target_role and current.pd_flip.get("current_role") == target_role:
            return {"status": "completed_by_restart", "worker_url": current.url}

        sleep_fn(poll_interval_seconds)

    return {"status": "timeout", "worker_url": target_url}


def run_restart_command(command: str, log_fn: LogFn = print) -> None:
    log_fn(f"running restart command: {command}")
    subprocess.run(shlex.split(command), check=True)


def observe(
    client: HttpClient,
    worker_urls: List[str],
    poll_interval_seconds: float,
    watch: bool,
    log_fn: LogFn = print,
) -> None:
    while True:
        for snapshot in fetch_snapshots(client, worker_urls):
            if snapshot.pd_flip:
                log_fn(f"{snapshot.url}: {summarize_pd_flip(snapshot.pd_flip)}")
            else:
                log_fn(f"{snapshot.url}: pd_flip=<missing>")
        if not watch:
            return
        time.sleep(poll_interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--http-timeout", type=float, default=10.0)

    subparsers = parser.add_subparsers(dest="command", required=True)

    observe_parser = subparsers.add_parser("observe")
    add_worker_url_arg(observe_parser)
    observe_parser.add_argument("--poll-interval", type=float, default=1.0)
    observe_parser.add_argument("--watch", action="store_true")

    trigger_parser = subparsers.add_parser("trigger")
    trigger_parser.add_argument("--worker-url", required=True)
    trigger_parser.add_argument("--direction", choices=("d_to_p", "p_to_d"), required=True)
    trigger_parser.add_argument("--prefill-nodes", type=int, default=1)
    trigger_parser.add_argument("--decode-nodes", type=int, default=2)
    trigger_parser.add_argument("--threshold", type=float, default=0.9)
    trigger_parser.add_argument("--window-seconds", type=float, default=0.0)

    run_parser = subparsers.add_parser("run-once")
    add_worker_url_arg(run_parser)
    run_parser.add_argument("--timeout", type=float, default=300.0)
    run_parser.add_argument("--poll-interval", type=float, default=1.0)
    run_parser.add_argument("--restart-command", default=None)

    return parser


def add_worker_url_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--worker-url",
        action="append",
        required=True,
        help="SGLang worker base URL. Repeat for multiple workers.",
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    client = HttpClient(api_key=args.api_key, timeout_seconds=args.http_timeout)

    if args.command == "observe":
        observe(
            client=client,
            worker_urls=args.worker_url,
            poll_interval_seconds=args.poll_interval,
            watch=args.watch,
        )
        return 0

    if args.command == "trigger":
        result = trigger_flip(
            client=client,
            worker_url=args.worker_url,
            direction=args.direction,
            prefill_nodes=args.prefill_nodes,
            decode_nodes=args.decode_nodes,
            threshold=args.threshold,
            window_seconds=args.window_seconds,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "run-once":
        result = run_once(
            client=client,
            worker_urls=args.worker_url,
            timeout_seconds=args.timeout,
            poll_interval_seconds=args.poll_interval,
            restart_command=args.restart_command,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] in ("completed", "completed_by_restart") else 1

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
