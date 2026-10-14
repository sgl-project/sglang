#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""Bring up a real Tair KVCache Manager (KVCM) for `tests/kvcm_live.rs`.

The KVCM backend of the KV Indexer has no in-repo server to talk to; this
script starts one from a source build and provisions the event-report resources
the backend needs (event-report storages, an instance group listing them, a
registered instance). It is the reproducible half of the live integration:
`tests/kvcm_live.rs` asserts behavior, this script only allocates processes and
configuration.

    # once, in the KVCM checkout
    bazelisk build //kv_cache_manager:kv_cache_manager_bin

    python3 tests/kvcm_live/setup_kvcm.py start --workdir /tmp/kvcm-live
    source /tmp/kvcm-live/kvcm_live.env
    cargo test -p sgl-kv-indexer --test kvcm_live -- --ignored
    python3 tests/kvcm_live/setup_kvcm.py stop --workdir /tmp/kvcm-live

`start` is idempotent: an already running manager is left alone and the
provisioning is re-applied. `config` re-applies provisioning only. `start`
verifies the state file's pid is still the manager it started before reusing or
signalling it, so a stale workdir (they live in `/tmp`, which survives reboots)
cannot make `stop` take down an unrelated process; `status` reports that check
too. If the binary moved, point at it with `--binary`.

The liveness knobs below are deliberately tight (2s timeout, 2s grace, 100ms
check interval) so the reclamation tests finish in seconds; a production
deployment sets them to values an order of magnitude larger and the backend
heartbeats once every 30s. Tests therefore configure their own heartbeat
cadence explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import requests

# --- Provisioning constants ------------------------------------------------

INSTANCE_GROUP = "sgl_kv_indexer_live_group"
INSTANCE_ID = "sgl_kv_indexer_live_instance"
STORAGE_L1P5 = "sgl_kv_indexer_live_event_report_l1p5"
STORAGE_L2 = "sgl_kv_indexer_live_event_report_l2"
# The backend reports one spec per resident component under these names; an
# instance that does not declare them fails every BLOCK_ADD.
SPEC_NAMES = ("full", "swa", "mamba")
BLOCK_SIZE = 128
# KVCM requires a model deployment on every RegisterInstance. Every field but
# the name and dtype keeps its KVCM default, which is what `KvcmAutoRegister`
# sends, so a backend that auto-registers this same instance is idempotent.
MODEL_NAME = "sgl-kv-indexer-live"
MODEL_DTYPE = "FP16"
# Must match what `KvcmAutoRegister` sends, otherwise re-registration of this
# instance fails with DUPLICATE_ENTITY (KVCM compares every field).
DEFAULT_QUERY_TYPE = "QT_PREFIX_MATCH"

LIVENESS = {
    "heartbeat_timeout_ms": 2000,
    "cleanup_grace_ms": 2000,
    "liveness_check_interval_ms": 100,
}

DEFAULT_BINARY = "bazel-bin/kv_cache_manager/kv_cache_manager_bin"
DEFAULT_KVCM_REPO = "/root/ws/kv/tair-kvcache"
STATE_FILE = "kvcm_live.json"
ENV_FILE = "kvcm_live.env"
REGISTRY_FILE = "registry.json"


class SetupError(RuntimeError):
    pass


# --- Process management ----------------------------------------------------


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def free_ports(count: int) -> list:
    """Distinct free ports: four independent bind-then-close calls can hand back
    the same number, and a colliding set only shows up as a startup timeout."""
    ports = set()
    while len(ports) < count:
        ports.add(free_port())
    return sorted(ports)


def process_identity(pid: int) -> Dict[str, str]:
    """What `/proc` says the pid is right now, for the checks below."""
    identity = {"exe": "", "cmdline": ""}
    try:
        identity["exe"] = os.path.realpath(f"/proc/{pid}/exe")
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            identity["cmdline"] = handle.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        pass
    return identity


def registry_uri(state: Dict[str, Any]) -> str:
    """The registry URI every manager of this workdir is started with."""
    return f"local://{os.path.join(state.get('workdir', ''), REGISTRY_FILE)}"


def is_our_manager(state: Dict[str, Any], pid: int) -> bool:
    """Whether `pid` is the manager this state file started.

    State files live in workdirs under /tmp, which survive reboots, and pids are
    reused: signalling on the strength of a stale file would take down an
    unrelated process. The discriminator is the workdir-scoped registry URI in
    the cmdline, because `/proc/<pid>/exe` is neither exact nor tolerant: another
    manager from the same binary (routinely running on this host) reads as ours,
    while a wrapper script or a renamed copy reads as a stranger we then refuse
    to stop.
    """
    return bool(state.get("workdir")) and registry_uri(state) in process_identity(pid)["cmdline"]


def terminate(pid: int, timeout: float) -> bool:
    """SIGTERM, then SIGKILL after `timeout` seconds. True when it is gone.

    A manager is started in its own session, so the signal goes to the process
    group: a wrapper script that execs the manager (or spawns it) leaves children
    in that group, and signalling only the recorded pid would orphan them.
    """
    try:
        group = os.getpgid(pid)
    except ProcessLookupError:
        return True

    def send(signum: int) -> bool:
        try:
            # Only ever a group the manager leads; signalling our own would be
            # catastrophic, and a pid without its own session cannot be ours.
            if group == pid:
                os.killpg(group, signum)
            else:
                os.kill(pid, signum)
            return True
        except ProcessLookupError:
            return False

    if not send(signal.SIGTERM):
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and pid_alive(pid):
        time.sleep(0.2)
    if pid_alive(pid):
        print(f"[setup] pid {pid} ignored SIGTERM; sending SIGKILL")
        if not send(signal.SIGKILL):
            return True
        time.sleep(0.5)
    return not pid_alive(pid)


def state_path(workdir: str) -> str:
    return os.path.join(workdir, STATE_FILE)


def read_state(workdir: str) -> Optional[Dict[str, Any]]:
    try:
        with open(state_path(workdir)) as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def write_state(workdir: str, state: Dict[str, Any]) -> None:
    with open(state_path(workdir), "w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def pid_alive(pid: int) -> bool:
    """False for a pid that is gone, and for a zombie: `kill(pid, 0)` succeeds on
    a zombie, which would read as "still running" until its parent reaps it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            # "<pid> (<comm>) <state> ..."; comm may itself contain spaces and
            # parentheses, so split on the last one.
            state = handle.read().rsplit(b")", 1)[-1].split()
        return not state or state[0] != b"Z"
    except OSError:
        return True


def resolve_binary(args: argparse.Namespace) -> str:
    binary = args.binary
    if not os.path.isabs(binary):
        binary = os.path.join(args.kvcm_repo, binary)
    if not os.access(binary, os.X_OK):
        raise SetupError(
            f"KVCM binary not found at {binary}; build it with "
            f"`cd {args.kvcm_repo} && bazelisk build //kv_cache_manager:kv_cache_manager_bin`"
        )
    return binary


def http_post(url: str, payload: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as error:
        raise SetupError(f"POST {url} failed: {error}") from error
    if response.status_code != 200:
        raise SetupError(f"POST {url} returned HTTP {response.status_code}: {response.text[:400]}")
    try:
        return response.json()
    except ValueError as error:
        raise SetupError(f"POST {url} returned non-JSON: {response.text[:400]}") from error


def status_code(body: Dict[str, Any]) -> str:
    return str(body.get("header", {}).get("status", {}).get("code"))


def status_message(body: Dict[str, Any]) -> str:
    return str(body.get("header", {}).get("status", {}).get("message", ""))


def api_post(url: str, payload: Dict[str, Any], accept=("OK",)) -> Dict[str, Any]:
    body = http_post(url, payload)
    code = status_code(body)
    if code not in accept:
        raise SetupError(
            f"POST {url} failed: code={code} message={status_message(body)} body={json.dumps(body)[:400]}"
        )
    return body


def wait_ready(http_port: int, timeout: float) -> None:
    """Polls the meta HTTP API; the gRPC service starts in the same process."""
    url = f"http://127.0.0.1:{http_port}/api/getClusterInfo"
    deadline = time.monotonic() + timeout
    last_error = "no attempt made"
    while time.monotonic() < deadline:
        try:
            body = http_post(url, {"trace_id": "sgl_kv_indexer_live_readiness"}, timeout=2.0)
            code = status_code(body)
            if code in ("OK", "SERVER_NOT_LEADER"):
                return
            last_error = f"code={code} message={status_message(body)}"
        except SetupError as error:
            last_error = str(error)
        time.sleep(0.2)
    raise SetupError(f"KVCM did not become ready within {timeout:.0f}s: {last_error}")


def start_once(args: argparse.Namespace) -> Dict[str, Any]:
    """One attempt: spawn a manager on a fresh port set and wait for readiness."""
    binary = resolve_binary(args)
    rpc_port, http_port, admin_rpc_port, admin_http_port = free_ports(4)
    registry = os.path.join(args.workdir, REGISTRY_FILE)
    command = [
        binary,
        f"--env=kvcm.service.rpc_port={rpc_port}",
        f"--env=kvcm.service.http_port={http_port}",
        f"--env=kvcm.service.admin_rpc_port={admin_rpc_port}",
        f"--env=kvcm.service.admin_http_port={admin_http_port}",
        f"--env=kvcm.registry_storage.uri=local://{registry}",
        "--env=kvcm.service.enable_debug_service=false",
        "--env=kvcm.logger.log_level=4",
    ]
    print(f"[setup] starting {shlex.join(command)}")
    with open(os.path.join(args.workdir, "stdout.log"), "ab") as stdout, open(
        os.path.join(args.workdir, "stderr.log"), "ab"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=args.workdir,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    state = {
        "binary": binary,
        "pid": process.pid,
        "rpc_port": rpc_port,
        "http_port": http_port,
        "admin_rpc_port": admin_rpc_port,
        "admin_http_port": admin_http_port,
        "workdir": args.workdir,
        "instance_id": INSTANCE_ID,
        "instance_group": INSTANCE_GROUP,
    }
    write_state(args.workdir, state)
    try:
        wait_ready(http_port, args.startup_timeout)
    except SetupError as error:
        # A manager that never became ready is still holding its ports and
        # writing logs; leaving it behind turns one failed start into a leak.
        tail(os.path.join(args.workdir, "stderr.log"))
        reaped = terminate(process.pid, args.stop_timeout)
        if reaped:
            try:
                # Reap our own child, so it cannot linger as a zombie.
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                reaped = False
        if reaped:
            os.remove(state_path(args.workdir))
            raise SetupError(
                f"{error} (the manager never became ready; pid {process.pid} shut down)"
            ) from error
        raise SetupError(
            f"{error} (pid {process.pid} survived shutdown and still holds "
            f"ports {rpc_port}/{http_port}; `stop` will retry it)"
        ) from error
    print(f"[setup] KVCM ready: grpc=127.0.0.1:{rpc_port} http=127.0.0.1:{http_port}")
    return state


def start_manager(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.workdir, exist_ok=True)
    existing = read_state(args.workdir)
    if existing and pid_alive(int(existing["pid"])):
        pid = int(existing["pid"])
        if is_our_manager(existing, pid):
            print(f"[setup] KVCM already running (pid={pid}, grpc=127.0.0.1:{existing['rpc_port']})")
            return existing
        # A reused pid would otherwise make `start` report "already running"
        # and provision against ports this state file no longer owns.
        print(
            f"[setup] state file is stale: pid {pid} is "
            f"{process_identity(pid)['exe'] or 'unknown'}, not the recorded manager"
        )
        os.remove(state_path(args.workdir))

    last_error = None
    for attempt in range(1, args.start_attempts + 1):
        try:
            return start_once(args)
        except SetupError as error:
            last_error = error
            print(f"[setup] start attempt {attempt}/{args.start_attempts} failed: {error}")
    raise last_error


def stop_manager(args: argparse.Namespace) -> int:
    state = read_state(args.workdir)
    if not state:
        print(f"[setup] no state file in {args.workdir}; nothing to stop")
        return 0
    pid = int(state["pid"])
    if not pid_alive(pid):
        print(f"[setup] KVCM (pid={pid}) is not running")
        os.remove(state_path(args.workdir))
        return 0
    if not is_our_manager(state, pid):
        identity = process_identity(pid)
        raise SetupError(
            f"pid {pid} is {identity['exe'] or 'unknown'} "
            f"({identity['cmdline'] or 'cmdline unreadable'}), not the manager this state file "
            f"started ({state.get('binary')}); refusing to signal it. "
            f"The manager is gone or the pid was reused: remove {state_path(args.workdir)} and run `start`."
        )
    if not terminate(pid, args.stop_timeout):
        raise SetupError(f"pid {pid} survived SIGKILL; check it manually")
    print(f"[setup] KVCM (pid={pid}) stopped")
    os.remove(state_path(args.workdir))
    return 0


def tail(path: str, lines: int = 20) -> None:
    try:
        with open(path) as handle:
            content = handle.readlines()[-lines:]
    except OSError:
        return
    if content:
        print(f"[setup] last lines of {path}:")
        sys.stdout.writelines(content)


# --- Provisioning ----------------------------------------------------------


def event_report_storage(name: str, storage_type: str) -> Dict[str, Any]:
    return {
        "global_unique_name": name,
        "storage_type": storage_type,
        "event_report": LIVENESS,
        # Event-report storages hold metadata only; there is no medium to open.
        "check_storage_available_when_open": False,
    }


def instance_group() -> Dict[str, Any]:
    return {
        "name": INSTANCE_GROUP,
        "storage_candidates": ["nfs_01"],
        "global_quota_group_name": "default_quota_group",
        "max_instance_count": 10000,
        "quota": {
            "capacity": 10737418240,
            "quota_config": [{"storage_type": 4, "capacity": 10737418240}],
        },
        "cache_config": {
            "reclaim_strategy": {
                "storage_unique_name": "nfs_01",
                "reclaim_policy": 1,
                "trigger_strategy": {"used_size": 1073741824, "used_percentage": 0.8},
                "trigger_period_seconds": 60,
                "reclaim_step_size": 1073741824,
                "reclaim_step_percentage": 10,
            },
            "data_storage_strategy": 2,
            "meta_indexer_config": {
                "max_key_count": 1000000,
                "mutex_shard_num": 16,
                "batch_key_size": 16,
                "meta_storage_backend_config": {"storage_type": "local", "storage_uri": ""},
                "meta_cache_policy_config": {"type": "LRU", "capacity": 10000},
            },
        },
        "event_report_storage_candidates": [STORAGE_L1P5, STORAGE_L2],
        "version": 1,
    }


def instance_registration() -> Dict[str, Any]:
    return {
        "trace_id": "sgl_kv_indexer_live_register",
        "instance_group": INSTANCE_GROUP,
        "instance_id": INSTANCE_ID,
        "block_size": BLOCK_SIZE,
        "location_spec_infos": [{"name": name, "size": 0} for name in SPEC_NAMES],
        "model_deployment": {"model_name": MODEL_NAME, "dtype": MODEL_DTYPE},
        "default_query_type": DEFAULT_QUERY_TYPE,
    }


def provision(state: Dict[str, Any]) -> None:
    admin = f"http://127.0.0.1:{state['admin_http_port']}"
    meta = f"http://127.0.0.1:{state['http_port']}"

    existing_storages = {
        storage["global_unique_name"]
        for storage in api_post(f"{admin}/api/listStorage", {"trace_id": "sgl_kv_indexer_live_list_storage"}).get(
            "storage", []
        )
    }
    for name, storage_type in ((STORAGE_L1P5, "ST_EVENT_REPORT_L1P5"), (STORAGE_L2, "ST_EVENT_REPORT_L2")):
        storage = event_report_storage(name, storage_type)
        if name not in existing_storages:
            api_post(f"{admin}/api/addStorage", {"trace_id": f"sgl_kv_indexer_live_add_{name}", "storage": storage})
        # updateStorage is remove + add, so it makes the liveness knobs
        # authoritative on every run whether or not the storage already existed.
        api_post(
            f"{admin}/api/updateStorage",
            {"trace_id": f"sgl_kv_indexer_live_update_{name}", "storage": storage, "force_update": True},
        )
        print(f"[setup] event-report storage {name} ({storage_type}) configured, liveness={LIVENESS}")

    existing_groups = {
        group["name"]
        for group in api_post(
            f"{admin}/api/listInstanceGroup", {"trace_id": "sgl_kv_indexer_live_list_group"}
        ).get("instance_group", [])
    }
    if INSTANCE_GROUP not in existing_groups:
        api_post(
            f"{admin}/api/createInstanceGroup",
            {"trace_id": "sgl_kv_indexer_live_create_group", "instance_group": instance_group()},
        )
    print(f"[setup] instance group {INSTANCE_GROUP} configured")

    api_post(
        f"{meta}/api/registerInstance",
        instance_registration(),
        accept=("OK", "DUPLICATE_ENTITY"),
    )
    print(f"[setup] instance {INSTANCE_ID} registered (specs={','.join(SPEC_NAMES)}, block_size={BLOCK_SIZE})")


def write_env_file(workdir: str, state: Dict[str, Any]) -> str:
    lines = [
        "# Generated by sgl-kv-indexer tests/kvcm_live/setup_kvcm.py",
        f"export KVCM_LIVE_WORKDIR={workdir}",
        f"export KVCM_LIVE_ENDPOINT=http://127.0.0.1:{state['rpc_port']}",
        f"export KVCM_LIVE_INSTANCE_ID={state['instance_id']}",
        f"export KVCM_LIVE_INSTANCE_GROUP={state['instance_group']}",
        f"export KVCM_LIVE_MODEL_NAME={MODEL_NAME}",
        f"export KVCM_LIVE_MODEL_DTYPE={MODEL_DTYPE}",
        f"export KVCM_LIVE_HTTP_ENDPOINT=http://127.0.0.1:{state['http_port']}",
        f"export KVCM_LIVE_ADMIN_ENDPOINT=http://127.0.0.1:{state['admin_http_port']}",
    ]
    path = os.path.join(workdir, ENV_FILE)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def print_instructions(state: Dict[str, Any]) -> None:
    path = write_env_file(state["workdir"], state)
    print(f"\n[setup] environment written to {path}\n")
    print(f"  source {path}")
    print("  cd experimental/sgl-router")
    print("  cargo test -p sgl-kv-indexer --test kvcm_live -- --ignored")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("start", "config", "stop", "status"))
    parser.add_argument("--workdir", default="/tmp/kvcm-live", help="state, registry and logs live here")
    parser.add_argument("--kvcm-repo", default=DEFAULT_KVCM_REPO, help="Tair KVCache Manager checkout")
    parser.add_argument("--binary", default=DEFAULT_BINARY, help="manager binary (relative to --kvcm-repo)")
    parser.add_argument("--startup-timeout", type=float, default=60.0, help="seconds to wait for readiness")
    parser.add_argument("--stop-timeout", type=float, default=15.0, help="seconds to wait for SIGTERM")
    parser.add_argument(
        "--start-attempts",
        type=int,
        default=2,
        help="start attempts with a fresh port set; a stolen port is otherwise just a timeout",
    )
    args = parser.parse_args()

    try:
        if args.command == "start":
            state = start_manager(args)
            provision(state)
            print_instructions(state)
            return 0
        if args.command == "config":
            state = read_state(args.workdir)
            if not state or not pid_alive(int(state["pid"])):
                raise SetupError(f"KVCM is not running in {args.workdir}; run `start` first")
            provision(state)
            print_instructions(state)
            return 0
        if args.command == "stop":
            return stop_manager(args)
        state = read_state(args.workdir)
        if not state:
            print(f"[setup] no KVCM state in {args.workdir}")
            return 1
        pid = int(state["pid"])
        alive = pid_alive(pid)
        ours = alive and is_our_manager(state, pid)
        print(
            f"[setup] pid={pid} alive={alive} ours={ours} grpc=127.0.0.1:{state['rpc_port']} "
            f"http=127.0.0.1:{state['http_port']} workdir={state['workdir']}"
        )
        if alive and not ours:
            print(f"[setup] pid {pid} is {process_identity(pid)['exe'] or 'unknown'}, not this state file's manager")
        return 0 if ours else 1
    except SetupError as error:
        print(f"[setup] error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
