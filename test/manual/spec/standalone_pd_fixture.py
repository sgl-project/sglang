"""Single-GPU STANDALONE PD test support; no inference POST is retried.

This is deliberately separate from the registered two-GPU/RDMA/LB fixture.
Subclasses can supply per-role forward-hook arguments and environment values.
``post_pair`` preserves HTTP errors and terminal aborts for negative tests.
"""

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    terminate_and_kill_process_tree,
)
from sglang.utils import wait_for_http_ready

QUEUE_METRICS = {
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:num_grammar_queue_reqs",
    "sglang:num_prefill_bootstrap_queue_reqs",
    "sglang:num_prefill_inflight_queue_reqs",
    "sglang:num_decode_prealloc_queue_reqs",
    "sglang:num_decode_transfer_queue_reqs",
    "sglang:num_decode_host_receive_queue_reqs",
}


@dataclass
class NativeResponse:
    status_code: int
    body: object
    raw: bytes


class StandalonePDServerBase(CustomTestCase):
    # The method owns server shutdown and must never replay a timed-out POST.
    expect_server_exit = True
    target_model = "Qwen/Qwen3-4B"
    draft_model = "Qwen/Qwen3-0.6B"
    extra_prefill_args = ()
    extra_decode_args = ()
    extra_common_args = ()
    extra_common_env = {}
    extra_prefill_env = {}
    extra_decode_env = {}
    max_new_tokens = 24
    request_timeout = 180

    @classmethod
    def setUpClass(cls):
        cls.processes = {}
        cls.owned = {}
        cls.artifact_lock = threading.Lock()
        cls.get_count = 0
        cls.temporary_output = None
        output = os.environ.get("SGLANG_TEST_STANDALONE_PD_OUTPUT_DIR")
        if output:
            cls.output_dir = Path(output).expanduser().resolve()
        else:
            cls.temporary_output = tempfile.TemporaryDirectory(prefix="standalone-pd-")
            cls.output_dir = Path(cls.temporary_output.name) / "run"
        cls.output_dir.mkdir(parents=True, exist_ok=False)
        (cls.output_dir / "fault-events").mkdir()
        (cls.output_dir / "http").mkdir()
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Requires one Hopper GPU and Mooncake CUDA transport"
            )
        if torch.cuda.get_device_capability(0)[0] != 9:
            raise unittest.SkipTest("This manual regression uses Hopper FA3")
        if torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3:
            raise unittest.SkipTest(
                "The same-GPU target/draft pair needs an 80GB-class GPU"
            )
        # Require NVML for the strict ordinary -> PD GPU-release gate.
        import pynvml

        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        cls.target_model = os.environ.get(
            "SGLANG_TEST_STANDALONE_PD_TARGET", cls.target_model
        )
        cls.draft_model = os.environ.get(
            "SGLANG_TEST_STANDALONE_PD_DRAFT", cls.draft_model
        )
        # Hold all six sockets during selection so the plan has no duplicates.
        sockets = []
        try:
            for _ in range(6):
                sock = socket.socket()
                sock.bind(("127.0.0.1", 0))
                sockets.append(sock)
            ports = [sock.getsockname()[1] for sock in sockets]
        finally:
            for sock in sockets:
                sock.close()
        cls.ports = dict(zip(("ordinary", "prefill", "decode"), ports[:3]))
        cls.bootstrap_port = ports[3]
        cls.nccl_ports = {"prefill": ports[4], "decode": ports[5]}
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        commands = {}
        for role in ("ordinary", "prefill", "decode"):
            command = cls.server_command(role)
            parser.parse_args(command[3:])
            commands[role] = command
        cls.record("cli-grammar.json", {"parsed_commands": commands})

    @classmethod
    def record(cls, name, body):
        with (cls.output_dir / name).open("x") as file:
            json.dump(body, file, indent=2, allow_nan=False)
            file.write("\n")

    @classmethod
    def server_args(cls, role):
        args = [
            "--tokenizer-path",
            cls.target_model,
            "--tp-size",
            "1",
            "--pp-size",
            "1",
            "--base-gpu-id",
            "0",
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "fa3",
            "--page-size",
            "1",
            "--context-length",
            "2048",
            "--max-total-tokens",
            "4096",
            "--max-running-requests",
            "1",
            "--mem-fraction-static",
            "0.25",
            "--chunked-prefill-size",
            "512",
            "--disable-overlap-schedule",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--skip-server-warmup",
            "--enable-metrics",
        ]
        if role != "ordinary":
            args += [
                "--speculative-algorithm",
                "STANDALONE",
                "--speculative-draft-model-path",
                cls.draft_model,
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--disaggregation-mode",
                role,
                "--disaggregation-transfer-backend",
                "mooncake_tcp",
                "--disaggregation-bootstrap-port",
                str(cls.bootstrap_port),
                "--nccl-port",
                str(cls.nccl_ports[role]),
            ]
        args += list(cls.extra_common_args)
        if role != "ordinary":
            args += list(getattr(cls, f"extra_{role}_args"))
        return args

    @classmethod
    def server_command(cls, role):
        return [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.target_model,
            "--host",
            "127.0.0.1",
            "--port",
            str(cls.ports[role]),
            *cls.server_args(role),
        ]

    @classmethod
    def capture_owned(cls, role):
        """Find the session even after its leader exits, plus observed descendants."""
        owner = cls.owned[role]
        with owner["lock"]:
            session_is_ours = True
            try:
                leader = psutil.Process(owner["session"])
                if leader.create_time() != owner["leader_create_time"]:
                    session_is_ours = False
                    owner["error"] = RuntimeError(
                        "Original session leader PID was reused"
                    )
            except psutil.NoSuchProcess:
                pass
            snapshot = []
            for proc in psutil.process_iter(
                ["pid", "ppid", "uids", "create_time", "status"]
            ):
                try:
                    row = proc.info
                    if row["uids"].real != owner["uid"]:
                        continue
                    row = {
                        **row,
                        "session": os.getsid(proc.pid),
                        "pgid": os.getpgid(proc.pid),
                    }
                    snapshot.append(row)
                except (psutil.NoSuchProcess, ProcessLookupError):
                    pass
            # Session membership survives reparenting; the ancestry closure also
            # retains descendants that create another process group/session.
            owned_pids = {
                row["pid"]
                for row in snapshot
                if (session_is_ours and row["session"] == owner["session"])
                or (
                    row["pid"] in owner["members"]
                    and owner["members"][row["pid"]]["create_time"]
                    == row["create_time"]
                )
            }
            changed = True
            while changed:
                changed = False
                for row in snapshot:
                    if row["pid"] in owned_pids or row["ppid"] in owned_pids:
                        if row["pid"] not in owned_pids:
                            changed = True
                            owned_pids.add(row["pid"])
                        owner["members"][row["pid"]] = row
            return [
                row
                for row in snapshot
                if row["pid"] in owner["members"]
                and owner["members"][row["pid"]]["create_time"] == row["create_time"]
                and row["status"] != psutil.STATUS_ZOMBIE
            ]

    @classmethod
    def track_owned(cls, role):
        owner = cls.owned[role]
        while not owner["stop"].is_set():
            try:
                cls.capture_owned(role)
            except Exception as exc:
                owner["error"] = exc
                return
            owner["stop"].wait(0.1)

    @classmethod
    def start_server(cls, role):
        if role in cls.processes:
            raise RuntimeError(f"Already started {role}")
        if role != "ordinary" and "ordinary" in cls.processes:
            raise RuntimeError(
                "Stop and release ordinary before starting either PD peer"
            )
        env = {
            **os.environ,
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
            "SGLANG_SIMULATE_ACC_LEN": "-1",
            "MC_FORCE_TCP": "1",
        }
        env.update(cls.extra_common_env)
        if role != "ordinary":
            env.update(getattr(cls, f"extra_{role}_env"))
        sibling = str(Path(__file__).resolve().parent)
        env["PYTHONPATH"] = sibling + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        url = f"http://127.0.0.1:{cls.ports[role]}"
        # Use this interpreter, not a potentially different python3 on PATH.
        # No startup retry. Each server owns a distinct session for teardown.
        command = cls.server_command(role)
        with (
            (cls.output_dir / f"{role}.stdout.log").open("xb") as stdout,
            (cls.output_dir / f"{role}.stderr.log").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                command, env=env, start_new_session=True, stdout=stdout, stderr=stderr
            )
        cls.processes[role] = process
        cls.owned[role] = {
            "uid": os.getuid(),
            "session": process.pid,
            "pgid": process.pid,
            "leader_create_time": None,
            "members": {},
            "lock": threading.RLock(),
            "stop": threading.Event(),
            "error": None,
        }
        try:
            cls.owned[role]["leader_create_time"] = psutil.Process(
                process.pid
            ).create_time()
        except psutil.Error as exc:
            cls.owned[role]["error"] = exc
        cls.capture_owned(role)
        cls.record(
            f"{role}-launch.json",
            {
                "command": command,
                "pid": process.pid,
                "session": process.pid,
                "pgid": process.pid,
                "uid": os.getuid(),
                "observed_leader": cls.owned[role]["members"].get(process.pid),
            },
        )
        if cls.owned[role]["error"] is not None:
            raise RuntimeError("Could not pin launcher identity") from cls.owned[role][
                "error"
            ]
        thread = threading.Thread(target=cls.track_owned, args=(role,), daemon=True)
        cls.owned[role]["thread"] = thread
        thread.start()
        wait_for_http_ready(
            url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=process,
        )

    @classmethod
    def stop_server(cls, role):
        error = None
        try:
            cls._stop_server(role)
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            owner = cls.owned.get(role)
            if owner is not None:
                owner["stop"].set()
                if "thread" in owner:
                    owner["thread"].join(timeout=5)
                with (cls.output_dir / f"{role}-cleanup.jsonl").open("a") as file:
                    json.dump(
                        {
                            "time": time.time(),
                            "error": error,
                            "pid": cls.processes.get(role).pid
                            if role in cls.processes
                            else None,
                            "live_snapshot": owner.get("cleanup", {}),
                            "observed_members": list(owner["members"].values()),
                        },
                        file,
                    )
                    file.write("\n")

    @classmethod
    def _stop_server(cls, role):
        process = cls.processes.get(role)
        if process is None:
            return
        owner = cls.owned[role]

        def signal_owned(kill=False):
            try:
                rows = cls.capture_owned(role)
            except Exception as exc:
                owner["error"] = exc
                with owner["lock"]:
                    rows = list(owner["members"].values())
            for row in rows:
                try:
                    child = psutil.Process(row["pid"])
                    if (
                        child.create_time() == row["create_time"]
                        and child.uids().real == owner["uid"]
                    ):
                        child.kill() if kill else child.terminate()
                except psutil.NoSuchProcess:
                    pass

        parent_errors = []
        try:
            if process.poll() is None:
                leader = psutil.Process(process.pid)
                if (
                    leader.create_time() != owner["leader_create_time"]
                    or leader.uids().real != owner["uid"]
                ):
                    raise RuntimeError("Refusing reused launcher PID")
                terminate_and_kill_process_tree(
                    process, terminate_timeout=30, wait_timeout=30
                )
        except Exception as exc:
            parent_errors.append(f"parent cleanup: {exc}")
        try:
            process.wait(timeout=30)
        except Exception as exc:
            parent_errors.append(f"parent wait: {exc}")
        signal_owned()
        grace = time.monotonic() + 10
        while time.monotonic() < grace:
            try:
                if not cls.capture_owned(role):
                    break
            except Exception as exc:
                owner["error"] = exc
                break
            time.sleep(0.1)
        signal_owned(kill=True)
        deadline = time.monotonic() + 60
        import pynvml

        pynvml.nvmlInit()
        pending_observations = []
        try:
            while True:
                live = cls.capture_owned(role)
                holders = []
                gpu_pids = []
                unknown_pids = []
                for index in range(pynvml.nvmlDeviceGetCount()):
                    device = pynvml.nvmlDeviceGetHandleByIndex(index)
                    compute = pynvml.nvmlDeviceGetComputeRunningProcesses(device)
                    gpu_pids += [row.pid for row in compute]
                    unknown_pids += [
                        row.pid for row in compute if not psutil.pid_exists(row.pid)
                    ]
                    holders += [
                        row.pid
                        for row in compute
                        if row.pid in owner["members"] or row.pid == process.pid
                    ]
                gpu_pids = sorted(set(gpu_pids))
                unknown_pids = sorted(set(unknown_pids))
                if unknown_pids and (
                    not pending_observations
                    or pending_observations[-1]["unknown_gpu_pids"] != unknown_pids
                ):
                    pending_observations.append(
                        {
                            "time": time.time(),
                            "gpu_pids": gpu_pids,
                            "unknown_gpu_pids": unknown_pids,
                        }
                    )
                ports = [cls.ports[role]]
                if role == "prefill":
                    ports.append(cls.bootstrap_port)
                port_open = False
                for port in ports:
                    with socket.socket() as sock:
                        sock.settimeout(1)
                        port_open |= sock.connect_ex(("127.0.0.1", port)) == 0
                owner["cleanup"] = {
                    "members_remaining": live,
                    "owned_gpu_pids_remaining": holders,
                    "gpu_pids_remaining": gpu_pids,
                    "unknown_gpu_pids_remaining": unknown_pids,
                    "nvml_pending_observations": pending_observations,
                    "port_open": port_open,
                    "leader_exit_code": process.returncode,
                    "parent_cleanup_errors": parent_errors,
                }
                if (
                    not live
                    and not holders
                    and not unknown_pids
                    and not gpu_pids
                    and not port_open
                ):
                    if owner["error"] is not None:
                        raise RuntimeError("Owned-process tracking failed") from owner[
                            "error"
                        ]
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"{role} did not release: {live=}, {holders=}, "
                        f"{gpu_pids=}, {unknown_pids=}, {port_open=}"
                    )
                time.sleep(0.25)
        finally:
            pynvml.nvmlShutdown()
        if parent_errors:
            raise RuntimeError("; ".join(parent_errors))
        del cls.processes[role]

    @classmethod
    def tearDownClass(cls):
        errors = []
        # Ask both peers to exit before waiting on either one's GPU release.
        # NVML can report host PIDs that are not visible to container psutil.
        for role, process in list(getattr(cls, "processes", {}).items()):
            try:
                if process.poll() is None:
                    owner = cls.owned[role]
                    leader = psutil.Process(process.pid)
                    if (
                        leader.create_time() != owner["leader_create_time"]
                        or leader.uids().real != owner["uid"]
                    ):
                        raise RuntimeError("Refusing reused launcher PID")
                    leader.terminate()
            except psutil.NoSuchProcess:
                # A launcher may exit naturally between poll and lookup.
                pass
            except Exception as exc:
                errors.append(f"{role} initial TERM: {exc}")
        for role in list(getattr(cls, "processes", {})):
            try:
                cls.stop_server(role)
            except Exception as exc:
                errors.append(f"{role}: {exc}")
        try:
            if errors:
                raise RuntimeError("; ".join(errors))
        finally:
            if getattr(cls, "temporary_output", None) is not None:
                cls.temporary_output.cleanup()

    def http(self, role, method, path, payload=None):
        process = self.processes[role]
        self.assertIsNone(process.poll(), f"{role} process exited")
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            self.ports[role],
            timeout=self.request_timeout if method == "POST" else 10,
        )
        encoded = (
            None
            if payload is None
            else json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        )
        if method == "POST":
            rid = payload["rid"]
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", rid):
                raise ValueError("RID must be safe for the exclusive POST ledger")
            name = f"http/{role}-{rid}"
            with (self.output_dir / (name + ".request.raw")).open("xb") as file:
                file.write(encoded)
        else:
            with self.artifact_lock:
                type(self).get_count += 1
                number = self.get_count
            name = f"http/get-{number:04d}-{role}-{path.strip('/')}"
        self.record(
            name + ".attempt.json",
            {
                "role": role,
                "method": method,
                "path": path,
                "request": payload,
                "started_unix": time.time(),
                "request_sha256": hashlib.sha256(encoded).hexdigest()
                if encoded is not None
                else None,
                "automatic_retries": 0,
            },
        )
        try:
            connection.request(
                method, path, body=encoded, headers={"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            raw = response.read(4 * 1024 * 1024 + 1)
            self.assertLessEqual(len(raw), 4 * 1024 * 1024)
            with (self.output_dir / (name + ".response.raw")).open("xb") as file:
                file.write(raw)
            try:
                body = json.loads(raw)
            except ValueError:
                body = raw.decode(errors="replace")
            meta = body.get("meta_info", {}) if isinstance(body, dict) else {}
            self.record(
                name + ".response.json",
                {
                    "status_code": response.status,
                    "json_body": body if isinstance(body, (dict, list)) else None,
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                    "response_bytes": len(raw),
                    "finished_unix": time.time(),
                    "response_rid": meta.get("id"),
                    "finish_reason": meta.get("finish_reason"),
                },
            )
            return NativeResponse(response.status, body, raw)
        except BaseException as exc:
            self.record(
                name + ".error.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "transport_outcome_may_be_unknown": method == "POST",
                    "automatic_retries": 0,
                },
            )
            raise
        finally:
            connection.close()

    def assert_completion(self, response, request, *, prefill=False):
        self.assertEqual(response.status_code, 200, response.body)
        self.assertIsInstance(response.body, dict)
        meta = response.body["meta_info"]
        self.assertEqual(meta["id"], request["rid"])
        self.assertEqual(meta["prompt_tokens"], len(request["input_ids"]))
        self.assertEqual(meta["finish_reason"]["type"], "length")
        ids = response.body["output_ids"]
        self.assertTrue(all(type(token) is int and token >= 0 for token in ids))
        self.assertEqual(len(ids), 1 if prefill else self.max_new_tokens)
        self.assertEqual(meta["completion_tokens"], len(ids))
        return ids, meta

    def new_payload(self, input_ids, *, pd=True):
        body = {
            "rid": "standalone-pd-" + secrets.token_hex(12),
            "input_ids": input_ids,
            "stream": False,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": self.max_new_tokens,
                "ignore_eos": True,
            },
        }
        if pd:
            body.update(
                bootstrap_host="127.0.0.1",
                bootstrap_port=self.bootstrap_port,
                bootstrap_room=secrets.randbelow(2**62 - 1) + 1,
            )
        return body

    def post_pair(self, body):
        # Send the same RID/room/payload to both peers exactly once, as mini_lb does.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                role: pool.submit(self.http, role, "POST", "/generate", body)
                for role in ("prefill", "decode")
            }
            return {role: future.result() for role, future in futures.items()}

    def assert_idle(self, role):
        required = {"sglang:num_running_reqs", "sglang:num_queue_reqs"}
        if role == "prefill":
            required |= {
                "sglang:num_prefill_bootstrap_queue_reqs",
                "sglang:num_prefill_inflight_queue_reqs",
            }
        if role == "decode":
            required |= {
                "sglang:num_decode_prealloc_queue_reqs",
                "sglang:num_decode_transfer_queue_reqs",
            }
        deadline = time.monotonic() + 45
        while True:
            metrics = self.http(role, "GET", "/metrics")
            self.assertEqual(metrics.status_code, 200, metrics.body)
            counts = {}
            for line in metrics.raw.decode().splitlines():
                if not line or line.startswith("#"):
                    continue
                name = re.split(r"[\s{]", line, maxsplit=1)[0]
                if name in QUEUE_METRICS:
                    value = float(line.rsplit(None, 1)[1])
                    self.assertTrue(math.isfinite(value) and value >= 0, line)
                    counts.setdefault(name, []).append(value)
            if required <= counts.keys() and all(
                v == 0 for values in counts.values() for v in values
            ):
                health = self.http(role, "GET", "/health")
                self.assertEqual(health.status_code, 200, health.body)
                return counts
            if time.monotonic() >= deadline:
                self.fail(f"{role} queues did not become idle: {counts}")
            time.sleep(0.5)
