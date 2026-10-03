"""Run native detokenization, IPC, fanout and decoded-consumer validation."""

import argparse
import copy
import hashlib
import json
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import zmq
from fanout_fixtures import assert_equal, canonical, make_stream, make_tokenizer

from sglang.srt.managers import io_struct as io
from sglang.srt.managers import multi_tokenizer_mixin as mixin
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import get_zmq_socket


def make_manager(tokenizer, endpoint):
    """Initialize an actual detokenizer using the published serving config."""
    args = ServerArgs(
        model_path="dummy",
        tokenizer_path=str(tokenizer),
        tokenizer_worker_num=4,
        soft_watchdog_timeout=None,
    )
    publish(args, role="detokenizer")
    return DetokenizerManager(args, SimpleNamespace(detokenizer_ipc_name=endpoint))


def receipt():
    """Record native source hashes so parent and worker versions can be checked."""
    return {
        "ready": True,
        "transport": "pickle" if io._USE_PICKLE_IPC else "msgpack",
        "sources": {
            Path(module.__file__).name: hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (io, mixin)
        },
    }


def read_reply(process, timeout=60):
    """Read one bounded JSON response from a child process."""
    if not select.select([process.stdout], [], [], timeout)[0]:
        raise TimeoutError("Native worker did not reply")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"Worker exited: {process.poll()}")
    return json.loads(line)


def consume(sender, receivers, poller, batches):
    """Time through native consumer decode, including opaque metadata reads."""
    outputs, first = [], None
    started = time.perf_counter_ns()
    for batch in batches:
        routes = dict(zip(batch.rids, batch.http_worker_ipcs))
        io.sock_send(sender, batch)
        received = {}
        deadline = time.monotonic() + 15
        while len(received) < len(batch.rids):
            if time.monotonic() > deadline:
                raise TimeoutError("Fanout did not deliver all requests")
            for socket, _event in poller.poll(100):
                row = io.sock_recv(socket)
                row.time_stats = io.unwrap_from_pickle(row.time_stats)
                row.customized_info = io.unwrap_from_pickle(row.customized_info)
                if first is None:
                    first = (time.perf_counter_ns() - started) / 1e6
                assert len(row.rids) == 1
                rid = row.rids[0]
                assert rid not in received and routes[rid] == receivers[socket]
                received[rid] = row
        outputs.append(received)
    return {
        "complete_ms": (time.perf_counter_ns() - started) / 1e6,
        "first_ms": first,
    }, outputs


def run_controller(output):
    """Keep one native worker alive across sequential benchmark commands."""
    context = zmq.Context()
    sockets, worker = [], None
    with tempfile.TemporaryDirectory(prefix="sgfanout-") as temp:
        root = Path(temp)
        tokenizer = make_tokenizer(root / "tokenizer")
        endpoints = [f"ipc://{root}/http-{i}" for i in range(4)]
        incoming = f"ipc://{root}/input"
        receivers = {
            get_zmq_socket(context, zmq.PULL, endpoint, True): endpoint
            for endpoint in endpoints
        }
        sockets.extend(receivers)
        poller = zmq.Poller()
        for socket in receivers:
            poller.register(socket, zmq.POLLIN)
        with output.open("w") as log:
            try:
                worker = subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        "--tokenizer",
                        str(root / "tokenizer"),
                        "--endpoint",
                        incoming,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                )
                worker_receipt = read_reply(worker)
                own = receipt()
                assert own == worker_receipt, "Parent and worker source mismatch"
                sender = get_zmq_socket(context, zmq.PUSH, incoming, False)
                sender.setsockopt(zmq.SNDTIMEO, 10000)
                sockets.append(sender)
                oracle = make_manager(root / "tokenizer", f"ipc://{root}/oracle")
                oracle.handle_freeze_gc_req(io.FreezeGCReq())
                print(json.dumps(own), flush=True)
                for line in sys.stdin:
                    command = json.loads(line)
                    if command.get("stop"):
                        break
                    trials, rows, fences = [], 0, 0
                    for repetition in range(command["repetitions"]):
                        seed = command["seed"] + repetition
                        for is_fence in (False, True):
                            batches = make_stream(
                                4 if is_fence else command["count"],
                                False if is_fence else command["rich"],
                                seed + (1000000 if is_fence else 0),
                                endpoints,
                            )
                            expected = []
                            for batch in copy.deepcopy(batches):
                                result = oracle.handle_batch_token_id_out(batch)
                                expected.append(
                                    {
                                        rid: canonical(
                                            mixin._handle_output_by_index(result, i)
                                        )
                                        for i, rid in enumerate(result.rids)
                                    }
                                )
                            timing, actual = consume(sender, receivers, poller, batches)
                            text = dict.fromkeys(batches[0].rids, "")
                            for batch, received, wanted in zip(
                                batches, actual, expected
                            ):
                                assert_equal(
                                    {
                                        rid: canonical(row)
                                        for rid, row in received.items()
                                    },
                                    wanted,
                                )
                                for endpoint in endpoints:
                                    assert [
                                        rid
                                        for rid in received
                                        if batch.http_worker_ipcs[batch.rids.index(rid)]
                                        == endpoint
                                    ] == [
                                        rid
                                        for rid, owner in zip(
                                            batch.rids, batch.http_worker_ipcs
                                        )
                                        if owner == endpoint
                                    ]
                                for rid, row in received.items():
                                    text[rid] += row.output_strs[0]
                            assert set(text.values()) == {
                                tokenizer.decode([1, 2, 3], skip_special_tokens=True)
                            }
                            if is_fence:
                                fences += sum(len(step) for step in actual)
                            else:
                                trials.append(timing)
                                rows += sum(len(step) for step in actual)
                        assert not poller.poll(0), "Duplicate output after FIFO fence"
                    print(
                        json.dumps(
                            {
                                **command,
                                "trials": trials,
                                "verified_rows": rows,
                                "fence_rows": fences,
                            }
                        ),
                        flush=True,
                    )
            finally:
                if worker is not None:
                    worker.terminate()
                    try:
                        worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait(timeout=10)
                for socket in sockets:
                    socket.close(linger=0)
                context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.worker:
        manager = make_manager(args.tokenizer, args.endpoint)
        manager.handle_freeze_gc_req(io.FreezeGCReq())
        print(json.dumps(receipt()), flush=True)
        manager.multi_http_worker_event_loop()
    else:
        run_controller(args.log)
