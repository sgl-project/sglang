"""Peer agent for the multi-node KDA prefill parity test.

Runs on every node except rank 0 and watches the shared control directory
(``SGLANG_TEST_KDA_PARITY_PEER_CONTROL_DIR``): each ``launch-<seq>-r<rank>.json``
written by the kit starts the described ``sglang serve --node-rank <rank>``
process (stdout/stderr go to ``peer-<seq>-r<rank>.log`` in the same directory),
``stop-<seq>-r<rank>`` kills it and is acknowledged with ``stopped-<seq>-r<rank>``,
and a ``done`` marker ends the agent.

Usage: ``python -m sglang.test.kits.kda_cake_parity_peer --rank 1``
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from sglang.srt.utils import kill_process_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument(
        "--control-dir",
        default=os.environ.get("SGLANG_TEST_KDA_PARITY_PEER_CONTROL_DIR"),
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    control = args.control_dir
    os.makedirs(control, exist_ok=True)
    suffix = f"-r{args.rank}"
    running: dict[str, subprocess.Popen] = {}
    handled: set[str] = set()
    print(f"[kda-parity-peer] rank {args.rank} watching {control}", flush=True)
    while True:
        names = sorted(os.listdir(control))
        for name in names:
            if name.startswith("launch-") and name.endswith(f"{suffix}.json"):
                seq = name[len("launch-") : -len(f"{suffix}.json")]
                if seq in handled:
                    continue
                handled.add(seq)
                with open(os.path.join(control, name)) as f:
                    request = json.load(f)
                env = os.environ.copy()
                env.update(request.get("env", {}))
                log = open(os.path.join(control, f"peer-{seq}{suffix}.log"), "w")
                proc = subprocess.Popen(
                    request["argv"], env=env, stdout=log, stderr=subprocess.STDOUT
                )
                running[seq] = proc
                print(f"[kda-parity-peer] started {seq} pid={proc.pid}", flush=True)
        for name in names:
            if name.startswith("stop-") and name.endswith(suffix):
                seq = name[len("stop-") : -len(suffix)]
                ack = os.path.join(control, f"stopped-{seq}{suffix}")
                if os.path.exists(ack):
                    continue
                proc = running.pop(seq, None)
                if proc is not None and proc.poll() is None:
                    try:
                        kill_process_tree(proc.pid)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[kda-parity-peer] kill {seq}: {exc}", flush=True)
                    proc.wait(timeout=120)
                open(ack, "w").close()
                print(f"[kda-parity-peer] stopped {seq}", flush=True)
        if os.path.exists(os.path.join(control, "done")):
            for seq, proc in running.items():
                if proc.poll() is None:
                    kill_process_tree(proc.pid)
            print("[kda-parity-peer] done", flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
