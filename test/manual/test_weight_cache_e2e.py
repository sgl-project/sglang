"""Manual GPU smoke test for registry-discovered weight-cache daemons.

Usage:
    python test/manual/test_weight_cache_e2e.py \
        --model-path Qwen/Qwen3-0.6B --tp-size 1

The test launches the real standalone daemon supervisor, waits for valid file
registry records, performs the config/fetch protocol against every daemon, and
imports sample tensor handles. It uses a private temporary runtime directory so
it cannot collide with another deployment on the host.
"""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time


def _wait_for_registrations(process, registry, expected, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        registrations = registry.list_registrations()
        if len(registrations) == expected:
            return registrations
        if process.poll() is not None:
            raise RuntimeError(
                f"daemon launcher exited prematurely with code {process.returncode}"
            )
        time.sleep(2)
    raise TimeoutError(
        f"only {len(registry.list_registrations())}/{expected} daemons registered "
        f"within {timeout}s"
    )


def _request(socket_path, request, timeout):
    from sglang.srt.weight_cache.protocol import recv_msg, send_msg

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(socket_path)
        send_msg(sock, request)
        return recv_msg(sock)
    finally:
        sock.close()


def _fetch(socket_path, config, timeout):
    from sglang.srt.weight_cache.protocol import recv_msg, send_msg

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(socket_path)
        send_msg(sock, {"type": "fetch_state", "config": config})
        response = recv_msg(sock)
        return response
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    from sglang.srt.utils import kill_process_tree
    from sglang.srt.weight_cache.registry import FileWeightCacheRegistry

    namespace = f"manual-e2e-{os.getpid()}"
    expected = args.tp_size * args.pp_size
    with tempfile.TemporaryDirectory(
        prefix="sglang-weight-cache-e2e-", dir="/tmp"
    ) as runtime_dir:
        registry = FileWeightCacheRegistry(runtime_dir, namespace=namespace)
        command = [
            sys.executable,
            "-m",
            "sglang.srt.weight_cache.daemon",
            "--model-path",
            args.model_path,
            "--tp-size",
            str(args.tp_size),
            "--pp-size",
            str(args.pp_size),
            "--load-format",
            args.load_format,
            "--dtype",
            args.dtype,
            "--weight-cache-runtime-dir",
            runtime_dir,
            "--weight-cache-namespace",
            namespace,
        ]
        if args.quantization:
            command += ["--quantization", args.quantization]
        if args.trust_remote_code:
            command += ["--trust-remote-code"]

        process = subprocess.Popen(command)
        try:
            started = time.monotonic()
            registrations = _wait_for_registrations(
                process, registry, expected, args.timeout
            )
            print(
                f"Registered {len(registrations)} daemon(s) in "
                f"{time.monotonic() - started:.1f}s"
            )

            for registration in registrations:
                query = _request(registration.socket_path, {"type": "query_config"}, 30)
                if query.get("status") != "ok":
                    raise RuntimeError(f"query_config failed: {query}")
                response = _fetch(
                    registration.socket_path,
                    query["config"],
                    300,
                )
                if response.get("status") != "ok":
                    raise RuntimeError(f"fetch_state failed: {response}")
                daemon = response["daemon"]
                if daemon["daemon_id"] != registration.daemon_id:
                    raise RuntimeError("response daemon_id differs from registry")
                if daemon["device_uuid"] != registration.identity.device_uuid:
                    raise RuntimeError("response GPU UUID differs from registry")

                entries = response["entries"]
                from sglang.srt.utils import MultiprocessingSerializer

                for name in list(entries)[:5]:
                    tensor = MultiprocessingSerializer.deserialize(
                        entries[name]["handle"]
                    )
                    print(
                        f"{registration.identity.device_uuid} {name}: "
                        f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                    )
                    del tensor

            print("Weight-cache registry + IPC E2E passed")
        finally:
            if process.poll() is None:
                kill_process_tree(process.pid)
            process.wait(timeout=30)


if __name__ == "__main__":
    main()
