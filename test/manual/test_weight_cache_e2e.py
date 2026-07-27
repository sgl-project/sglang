"""E2E test for WeightCacheDaemon with real model loading.

Launches TP daemons that load a real model, export IPC handles,
and verifies the client can fetch and import them.

Usage:
    # With a small model (single GPU):
    python test/manual/test_weight_cache_e2e.py --model-path /path/to/model --tp-size 1

    # With a large model (multi-GPU):
    python test/manual/test_weight_cache_e2e.py --model-path /path/to/model --tp-size 4

Requires GPUs and the model to be available on disk.
"""

import argparse
import multiprocessing as mp
import os
import socket
import sys
import time


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _global_rank(tp_size, pp_rank, tp_rank):
    return tp_size * pp_rank + tp_rank


def _temp_path(tp_size, pp_rank, tp_rank, suffix):
    return f"/tmp/sglang_weight_cache_rank{_global_rank(tp_size, pp_rank, tp_rank)}{suffix}"


def run_single_daemon(
    model_path,
    gpu_id,
    tp_size,
    tp_rank,
    pp_size,
    pp_rank,
    dist_init_method,
    socket_path,
    ready_path,
    done_path,
    load_format,
    dtype,
    quantization,
    trust_remote_code,
):
    """Run a single daemon process for one (pp_rank, tp_rank)."""
    import traceback

    from sglang.srt.weight_cache.daemon import WeightCacheDaemon

    daemon = WeightCacheDaemon(
        model_path=model_path,
        gpu_id=gpu_id,
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_size=pp_size,
        pp_rank=pp_rank,
        dp_size=1,
        load_format=load_format,
        dtype=dtype,
        quantization=quantization,
        trust_remote_code=trust_remote_code,
        dist_init_method=dist_init_method,
    )
    daemon.socket_path = socket_path
    daemon.ready_path = ready_path

    try:
        daemon.load()
    except Exception as e:
        tb = traceback.format_exc()
        print(
            f"[Daemon gpu={gpu_id} pp_rank={pp_rank} tp_rank={tp_rank}] "
            f"LOAD ERROR: {e}\n{tb}",
            flush=True,
        )
        with open(ready_path, "w") as f:
            f.write(f"ERROR: {e}\n")
        return

    with open(ready_path, "w") as f:
        f.write(f"pid={os.getpid()}\n")
        f.write(f"num_entries={len(daemon.state_entries)}\n")
    print(
        f"[Daemon gpu={gpu_id} pp_rank={pp_rank} tp_rank={tp_rank}] Ready with "
        f"{len(daemon.state_entries)} tensors",
        flush=True,
    )

    from sglang.srt.weight_cache.protocol import CacheConfig, recv_msg, send_msg

    if os.path.exists(socket_path):
        os.unlink(socket_path)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(socket_path)
    s.listen(5)
    s.settimeout(60)

    try:
        while not os.path.exists(done_path):
            try:
                conn, _ = s.accept()
                try:
                    req = recv_msg(conn)
                    if req.get("type") == "query_config":
                        send_msg(
                            conn, {"status": "ok", "config": daemon.config.to_dict()}
                        )
                    elif req.get("type") == "fetch_state":
                        engine_config = CacheConfig.from_dict(req["config"])
                        if daemon.config.matches(engine_config):
                            send_msg(
                                conn,
                                {
                                    "status": "ok",
                                    "config": daemon.config.to_dict(),
                                    "entries": daemon.state_entries,
                                },
                            )
                        else:
                            send_msg(
                                conn,
                                {
                                    "status": "mismatch",
                                    "daemon_config": daemon.config.to_dict(),
                                },
                            )
                    elif req.get("type") == "ping":
                        send_msg(conn, {"status": "ok"})
                except Exception:
                    pass
                finally:
                    conn.close()
            except socket.timeout:
                continue
    finally:
        s.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)

    print(
        f"[Daemon gpu={gpu_id} pp_rank={pp_rank} tp_rank={tp_rank}] Exiting",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="E2E test for WeightCacheDaemon")
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--load-format", default="auto", help="Weight load format")
    parser.add_argument("--dtype", default="auto", help="Model dtype")
    parser.add_argument("--quantization", default=None, help="Quantization method")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds")
    args = parser.parse_args()

    tp_size = args.tp_size
    pp_size = args.pp_size
    total_ranks = tp_size * pp_size
    print(f"=== E2E Test: WeightCacheDaemon TP={tp_size} PP={pp_size} ===")
    print(f"Model: {args.model_path}")

    dist_port = find_free_port()
    dist_init_method = f"tcp://127.0.0.1:{dist_port}"
    print(f"dist_init_method = {dist_init_method}")

    # Clean up old files
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            for suffix in [".sock", ".ready", ".done"]:
                p = _temp_path(tp_size, pp_rank, tp_rank, suffix)
                if os.path.exists(p):
                    os.unlink(p)

    mp.set_start_method("spawn", force=True)

    # Launch all daemon processes
    procs = []
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            gpu_id = pp_rank * tp_size + tp_rank
            sock_path = _temp_path(tp_size, pp_rank, tp_rank, ".sock")
            rdy_path = _temp_path(tp_size, pp_rank, tp_rank, ".ready")
            done_path = _temp_path(tp_size, pp_rank, tp_rank, ".done")

            p = mp.Process(
                target=run_single_daemon,
                args=(
                    args.model_path,
                    gpu_id,
                    tp_size,
                    tp_rank,
                    pp_size,
                    pp_rank,
                    dist_init_method,
                    sock_path,
                    rdy_path,
                    done_path,
                    args.load_format,
                    args.dtype,
                    args.quantization,
                    args.trust_remote_code,
                ),
                name=f"daemon_gpu{gpu_id}",
            )
            p.start()
            procs.append(p)
            print(
                f"Launched daemon gpu={gpu_id} pp_rank={pp_rank} "
                f"tp_rank={tp_rank} pid={p.pid}"
            )

    # Wait for all daemons to be ready
    print("Waiting for all daemons to load model...")
    start = time.time()
    error_found = False
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            ready_path = _temp_path(tp_size, pp_rank, tp_rank, ".ready")
            while not os.path.exists(ready_path):
                elapsed = time.time() - start
                if elapsed > args.timeout:
                    print(
                        f"ERROR: Daemon pp_rank={pp_rank} tp_rank={tp_rank} "
                        f"timeout after {args.timeout}s"
                    )
                    error_found = True
                    break
                for p in procs:
                    if not p.is_alive() and not os.path.exists(ready_path):
                        print(
                            f"ERROR: Daemon pid={p.pid} exited with code {p.exitcode}"
                        )
                        error_found = True
                        break
                if error_found:
                    break
                time.sleep(2)

            if error_found:
                break

            with open(ready_path) as f:
                content = f.read()
                if content.startswith("ERROR:"):
                    print(
                        f"ERROR: Daemon pp_rank={pp_rank} tp_rank={tp_rank} "
                        f"failed: {content.strip()}"
                    )
                    error_found = True
                    break

            print(
                f"Daemon pp_rank={pp_rank} tp_rank={tp_rank} ready "
                f"({time.time()-start:.0f}s)"
            )
        if error_found:
            break

    if error_found:
        for p in procs:
            if p.is_alive():
                p.terminate()
        sys.exit(1)

    print(
        f"\nAll {total_ranks} daemons ready! Total load time: {time.time()-start:.1f}s"
    )

    # Query config from daemon (pp_rank=0, tp_rank=0)
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_cache.protocol import recv_msg, send_msg

    socket_path_0 = _temp_path(tp_size, 0, 0, ".sock")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(socket_path_0)
    send_msg(s, {"type": "query_config"})
    result = recv_msg(s)
    s.close()
    daemon_config = result.get("config", {})
    print(
        f"\nDaemon (0,0) config: model={daemon_config.get('model_path')}, "
        f"arch={daemon_config.get('model_arch')}, "
        f"tp_size={daemon_config.get('tp_size')}, "
        f"dtype={daemon_config.get('dtype')}"
    )

    # Fetch state from daemon (pp_rank=0, tp_rank=0)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(300)
    s.connect(socket_path_0)
    send_msg(s, {"type": "fetch_state", "config": daemon_config})
    tic = time.perf_counter()
    result = recv_msg(s)
    fetch_time = time.perf_counter() - tic
    s.close()

    print(
        f"\nFetch from daemon (0,0): status={result['status']}, time={fetch_time:.2f}s"
    )

    if result["status"] == "ok":
        entries = result["entries"]
        print(f"Received {len(entries)} IPC handles from daemon (0,0)")

        sample_names = list(entries.keys())[:5]
        for name in sample_names:
            entry = entries[name]
            imported = MultiprocessingSerializer.deserialize(entry["handle"])
            print(
                f"  {name}: shape={tuple(imported.shape)}, "
                f"dtype={imported.dtype}, device={imported.device}"
            )
            del imported

        print("\nIPC import OK!")
    else:
        print(f"ERROR: fetch_state returned {result}")
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                done = _temp_path(tp_size, pp_rank, tp_rank, ".done")
                with open(done, "w") as f:
                    f.write("done\n")
        for p in procs:
            if p.is_alive():
                p.terminate()
        sys.exit(1)

    # Test config mismatch
    mismatch_config = dict(daemon_config)
    mismatch_config["tp_size"] = daemon_config.get("tp_size", 1) + 1
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(socket_path_0)
    send_msg(s, {"type": "fetch_state", "config": mismatch_config})
    result = recv_msg(s)
    s.close()
    if result["status"] == "mismatch":
        print("Config mismatch detection: OK")
    else:
        print(f"WARNING: Expected mismatch, got {result['status']}")

    # Signal all daemons done
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            done = _temp_path(tp_size, pp_rank, tp_rank, ".done")
            with open(done, "w") as f:
                f.write("done\n")
    for p in procs:
        p.join(timeout=15)
        if p.is_alive():
            p.terminate()

    # Clean up
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            for suffix in [".sock", ".ready", ".done"]:
                path = _temp_path(tp_size, pp_rank, tp_rank, suffix)
                if os.path.exists(path):
                    os.unlink(path)

    print("\n=== E2E Test Passed! ===")


if __name__ == "__main__":
    main()
