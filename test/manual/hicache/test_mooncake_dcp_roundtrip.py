"""Real TCP Mooncake: writer exits before a fresh process restores DCP shards.

Run from the repository root with PYTHONPATH=python:. and a CPU-visible Torch
environment. Requires mooncake_master on PATH and the Mooncake Python binding.
No existing service is used: the driver owns a master and a storage donor, and
both worker processes have zero-byte storage segments. No model weights needed.

Add --controller to exercise four Gloo ranks and real controller workers against
the same isolated native TCP service, including eviction and in-flight aborts.
"""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

import torch

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

# Reuse the independent allocation/byte oracle; the native client is never
# replaced here. Avoid the standard library's unrelated `test` package.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "registered/unit/mem_cache")
)
try:
    from test_mooncake_dcp_storage import (
        KEYS,
        SOURCE_PAGES,
        TARGET_PAGES,
        _config,
        _io,
        _page_segments,
        _pool,
    )
finally:
    sys.path.pop(0)


def worker(mode, address):
    for layout, api, rank in product(
        ("layer_first", "page_first", "page_first_direct"), (1, 2), range(2)
    ):
        config = _config(rank=rank, layout=layout, page=64)
        config.extra_config.update(
            master_server_address=address,
            metadata_server="P2PHANDSHAKE",
            local_hostname="127.0.0.1",
            global_segment_size=0,
            extra_backend_tag=f"native-dcp-api{api}",
        )
        pool = _pool(config)
        source = _pool(config)
        words = source.kv_buffer.view(torch.int32)
        words.copy_(
            (
                torch.arange(words.numel(), dtype=torch.int32) + (rank + 1) * 100003
            ).reshape(words.shape)
        )
        store = MooncakeStore(config)
        try:
            store.register_mem_pool_host(pool)
            store.registered_pools["kv"] = pool
            if mode == "write":
                pool.kv_buffer.copy_(source.kv_buffer)
                assert _io(store, config, SOURCE_PAGES, api, True) == [True] * 3
            else:
                pool.kv_buffer.fill_(165)
                expected = pool.kv_buffer.clone()
                for src, dst in zip(SOURCE_PAGES, TARGET_PAGES):
                    for a, b in zip(
                        _page_segments(source, source.kv_buffer, src),
                        _page_segments(pool, expected, dst),
                    ):
                        b.copy_(a)
                assert store.batch_exists(KEYS) == 3
                assert _io(store, config, TARGET_PAGES, api, False) == [True] * 3
                torch.testing.assert_close(pool.kv_buffer, expected, rtol=0, atol=0)
            print(f"PASS {mode} layout={layout} api={api} shard={rank}", flush=True)
        finally:
            store.store.close()


def controller_worker(address):
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "registered/unit/mem_cache")
    )
    try:
        from test_mooncake_dcp_storage_controller import run_workers

        with tempfile.TemporaryDirectory(prefix="mooncake-controller-") as directory:
            reports = run_workers(directory, None, address=address)
            assert all(len(rows) == 14 for rows in reports)
    finally:
        sys.path.pop(0)


def run(controller=False):
    from mooncake.store import MooncakeDistributedStore

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    address = f"127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="mooncake-dcp-") as directory:
        with open(os.path.join(directory, "master.log"), "w+") as log:
            master = subprocess.Popen(
                [
                    "mooncake_master",
                    f"--rpc_port={port}",
                    "--enable_metric_reporting=false",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            donor = None
            try:
                deadline = time.monotonic() + 30
                while True:
                    if master.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("Mooncake master did not become ready")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=1):
                            break
                    except OSError:
                        time.sleep(0.1)
                donor = MooncakeDistributedStore()
                assert (
                    donor.setup(
                        "127.0.0.1",
                        "P2PHANDSHAKE",
                        128 << 20,
                        16 << 20,
                        "tcp",
                        "",
                        address,
                    )
                    == 0
                )
                for mode in ("controller",) if controller else ("write", "read"):
                    subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--worker",
                            mode,
                            "--address",
                            address,
                        ],
                        check=True,
                        timeout=240,
                    )
                print(
                    "PASS native TCP controller: 14 scenarios on four Gloo ranks"
                    if controller
                    else "PASS fresh-process TCP restore: 12 cases, writer exited before reader"
                )
            finally:
                if donor is not None:
                    donor.close()
                master.terminate()
                try:
                    master.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    master.kill()
                    master.wait(timeout=10)
                log.seek(0)
                print(log.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("write", "read", "controller"))
    parser.add_argument("--address")
    parser.add_argument("--controller", action="store_true")
    args = parser.parse_args()
    if args.worker == "controller":
        controller_worker(args.address)
    elif args.worker:
        worker(args.worker, args.address)
    else:
        run(args.controller)
