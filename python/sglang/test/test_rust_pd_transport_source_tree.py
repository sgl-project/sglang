"""Source-tree smoke for the independent Rust PD transport handles.

Run with the interpreter into which the current checkout's maturin extension
was installed. The test intentionally uses only the standard library.
"""

from __future__ import annotations

import http.client
import importlib
import json
import os
import signal
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path


def _free_control_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _config(
    role: str,
    control_port: int,
    psk_file: Path,
    *,
    process_epoch: str | None = None,
    registration_epoch: str | None = None,
) -> str:
    process_epoch = (
        process_epoch
        or {
            "prefill": "11111111-1111-4111-8111-111111111111",
            "decode": "22222222-2222-4222-8222-222222222222",
        }[role]
    )
    registration_epoch = (
        registration_epoch
        or {
            "prefill": "33333333-3333-4333-8333-333333333333",
            "decode": "44444444-4444-4444-8444-444444444444",
        }[role]
    )
    return json.dumps(
        {
            "role": role,
            "process_epoch": process_epoch,
            "registration_epoch": registration_epoch,
            "model_manifest_digest": "11" * 32,
            "tokenizer_manifest_digest": "22" * 32,
            "layout_fingerprint": "33" * 32,
            "native_abi_digest": "44" * 32,
            "mooncake_host": "127.0.0.1",
            "mooncake_ports": [19000],
            "control_host": "127.0.0.1",
            "control_port": control_port,
            "pd_control_psk_file": str(psk_file),
            "mock_data_plane": True,
        },
        separators=(",", ":"),
    )


def _serialize_result(value):
    if isinstance(value, list):
        return [
            {
                name: getattr(item, name)
                for name in (
                    "handle",
                    "ok",
                    "status",
                    "pd_reason",
                    "terminal_generation",
                    "first_token_id",
                    "first_token_consumed",
                )
                if hasattr(item, name)
            }
            for item in value
        ]
    if hasattr(value, "lifecycle"):
        return {
            name: getattr(value, name)
            for name in (
                "lifecycle",
                "pair_ready",
                "accepting_rooms",
                "session_count",
                "reconnect_generation",
                "process_epoch",
                "peer_process_epoch",
                "active_handles",
                "fatal_generation",
                "shutdown_outcome",
            )
        }
    if hasattr(value, "native_leases"):
        return {
            name: getattr(value, name)
            for name in (
                "active_rooms",
                "active_handles",
                "result_slots",
                "pending_prepares",
                "wire_plans",
                "native_leases",
                "source_kv_pages",
                "destination_kv_pages",
                "aux_slots",
                "completion_slots",
                "request_slots",
                "in_flight_transfers",
                "native_batches",
                "pending_bytes",
                "quarantined_rooms",
            )
        }
    if hasattr(value, "active_handles"):
        return {"active_handles": value.active_handles}
    return value


def _peer_main(role, config, commands, results):
    module = importlib.import_module(os.environ.get("SGLANG_PD_CORE_MODULE", "_core"))
    transport = module.PdTransport(config)
    try:
        while True:
            operation, arguments = commands.get()
            try:
                if operation == "snapshot":
                    value = transport.readiness().snapshot()
                else:
                    value = getattr(transport, operation)(*arguments)
                results.put((operation, True, _serialize_result(value)))
            except Exception as error:  # pragma: no cover - surfaced in parent
                results.put((operation, False, repr(error)))
            if operation == "shutdown":
                return
    finally:
        del transport


def _send(queue, operation, *arguments):
    queue.put((operation, arguments))


def _receive(queue, expected, timeout=10):
    operation, ok, value = queue.get(timeout=timeout)
    if operation != expected:
        raise AssertionError(f"expected {expected}, received {operation}")
    if not ok:
        raise AssertionError(f"{operation} failed: {value}")
    return value


def _receive_outcome(queue, expected, timeout=10):
    operation, ok, value = queue.get(timeout=timeout)
    if operation != expected:
        raise AssertionError(f"expected {expected}, received {operation}")
    return ok, value


def _wait_snapshot(commands, results, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        _send(commands, "snapshot")
        last = _receive(results, "snapshot", timeout=2)
        if predicate(last):
            return last
        time.sleep(0.05)
    raise AssertionError(f"snapshot condition timed out; last={last!r}")


class _ControlFrameProxy:
    """Transparent framed proxy with a one-shot pre-forward barrier."""

    _HEADER_BYTES = 32
    _TAG_BYTES = 32

    def __init__(self, listen_port, upstream_port, pause_kind=None):
        self.listen_port = listen_port
        self.upstream_port = upstream_port
        self.pause_kind = pause_kind
        self.reached = threading.Event()
        self.release = threading.Event()
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._first_session = True

    def start(self):
        self._thread.start()
        if not self.ready.wait(5):
            raise AssertionError("control proxy did not start")

    def close(self):
        self._stop.set()
        self.release.set()
        try:
            with socket.create_connection(("127.0.0.1", self.listen_port), timeout=0.2):
                pass
        except OSError:
            pass
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise AssertionError("control proxy did not stop")

    def _run(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", self.listen_port))
            listener.listen()
            listener.settimeout(0.1)
            self.ready.set()
            while not self._stop.is_set():
                try:
                    downstream, _ = listener.accept()
                except TimeoutError:
                    continue
                upstream = self._connect_upstream(wait=self._first_session)
                if upstream is None:
                    downstream.close()
                    continue
                pause_enabled = self._first_session
                self._first_session = False
                self._relay_session(downstream, upstream, pause_enabled)

    def _connect_upstream(self, wait):
        deadline = time.monotonic() + (5 if wait else 0.2)
        while not self._stop.is_set() and time.monotonic() < deadline:
            try:
                return socket.create_connection(
                    ("127.0.0.1", self.upstream_port), timeout=0.2
                )
            except OSError:
                time.sleep(0.02)
        return None

    def _relay_session(self, downstream, upstream, pause_enabled):
        closed = threading.Event()
        peers = ((downstream, upstream), (upstream, downstream))
        threads = [
            threading.Thread(
                target=self._relay,
                args=(*peer, closed, pause_enabled),
                daemon=True,
            )
            for peer in peers
        ]
        for thread in threads:
            thread.start()
        while not self._stop.is_set() and not closed.wait(0.05):
            pass
        closed.set()
        for peer in peers:
            try:
                peer[0].shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        downstream.close()
        upstream.close()
        for thread in threads:
            thread.join(timeout=1)

    def _relay(self, source, destination, closed, pause_enabled):
        try:
            source.settimeout(0.1)
            while not self._stop.is_set() and not closed.is_set():
                header = self._receive_exact(source, self._HEADER_BYTES, closed)
                if header is None:
                    return
                payload_length = int.from_bytes(header[12:16], "big")
                body = self._receive_exact(
                    source, payload_length + self._TAG_BYTES, closed
                )
                if body is None:
                    return
                kind = int.from_bytes(header[8:10], "big")
                if (
                    pause_enabled
                    and self.pause_kind == kind
                    and not self.reached.is_set()
                ):
                    self.reached.set()
                    while (
                        not self.release.wait(0.05)
                        and not self._stop.is_set()
                        and not closed.is_set()
                    ):
                        pass
                if self._stop.is_set() or closed.is_set():
                    return
                destination.sendall(header + body)
        except OSError:
            pass
        finally:
            closed.set()

    def _receive_exact(self, connection, length, closed):
        chunks = bytearray()
        while len(chunks) < length and not self._stop.is_set() and not closed.is_set():
            try:
                chunk = connection.recv(length - len(chunks))
            except TimeoutError:
                continue
            if not chunk:
                closed.set()
                return None
            chunks.extend(chunk)
        return bytes(chunks) if len(chunks) == length else None


class RustPdTransportSourceTreeTest(unittest.TestCase):
    def test_plain_server_starts_without_transport_or_psk(self):
        module = importlib.import_module(
            os.environ.get("SGLANG_PD_CORE_MODULE", "_core")
        )
        http_port = _free_control_port()
        model_path = "/mnt/models/Qwen/Qwen3-0.6B"
        server = module.Server(
            http_addr=f"127.0.0.1:{http_port}",
            server_args_json=json.dumps(
                {
                    "model_path": model_path,
                    "tokenizer_path": model_path,
                    "served_model_name": "qwen3-0.6b",
                    "host": "127.0.0.1",
                    "port": http_port,
                    "tokenizer_worker_num": 1,
                    "detokenizer_worker_num": 1,
                    "model_config": {"context_len": 4096, "vocab_size": 151936},
                },
                separators=(",", ":"),
            ),
        )
        try:
            connection = http.client.HTTPConnection("127.0.0.1", http_port, timeout=5)
            connection.request("GET", "/readiness")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 503)
            connection.close()
        finally:
            server.shutdown()

    def test_symmetric_abort_ack_is_terminal_once_and_clears_handles(self):
        module = importlib.import_module(
            os.environ.get("SGLANG_PD_CORE_MODULE", "_core")
        )
        with tempfile.TemporaryDirectory() as directory:
            psk_file = Path(directory) / "control.psk"
            psk_file.write_bytes(bytes(range(32)))
            psk_file.chmod(0o400)
            control_port = _free_control_port()
            prefill = module.PdTransport(_config("prefill", control_port, psk_file))
            decode = module.PdTransport(_config("decode", control_port, psk_file))
            with ThreadPoolExecutor(max_workers=4) as executor:
                prefill_start = executor.submit(prefill.start)
                decode_start = executor.submit(decode.start)
                decode_start.result(timeout=10)
                prefill_start.result(timeout=10)

                attempt = "77777777-7777-4777-8777-777777777777"
                digest = "77" * 32
                decode_handle = decode.receiver_create_many([0], [attempt], [digest])[
                    0
                ].handle
                prefill_handle = prefill.sender_create_many(
                    ["22222222-2222-4222-8222-222222222222"],
                    [0],
                    [attempt],
                    [digest],
                )[0].handle
                prefill_abort = executor.submit(
                    prefill.abort_many, [prefill_handle], "PD_ABORTED"
                )
                decode_abort = executor.submit(
                    decode.abort_many, [decode_handle], "PD_ABORTED"
                )
                self.assertTrue(prefill_abort.result(timeout=10)[0].ok)
                self.assertTrue(decode_abort.result(timeout=10)[0].ok)

            for transport, handle in (
                (prefill, prefill_handle),
                (decode, decode_handle),
            ):
                result = transport.poll_many([handle])[0]
                self.assertEqual(result.status, 0)
                self.assertEqual(result.pd_reason, "PD_ABORTED")
                self.assertTrue(transport.clear_many([handle])[0].ok)
                self.assertEqual(transport.readiness().snapshot().active_handles, 0)
                transport.shutdown()

    def test_authenticated_batch_transfers_completion_and_first_token_once(self):
        module = importlib.import_module(
            os.environ.get("SGLANG_PD_CORE_MODULE", "_core")
        )
        with tempfile.TemporaryDirectory() as directory:
            psk_file = Path(directory) / "control.psk"
            psk_file.write_bytes(bytes(range(32)))
            psk_file.chmod(0o400)
            control_port = _free_control_port()
            prefill = module.PdTransport(_config("prefill", control_port, psk_file))
            decode = module.PdTransport(_config("decode", control_port, psk_file))

            with ThreadPoolExecutor(max_workers=4) as executor:
                prefill_start = executor.submit(prefill.start)
                time.sleep(0.05)
                decode_start = executor.submit(decode.start)
                decode_start.result(timeout=10)
                prefill_start.result(timeout=10)

                attempts = [
                    "55555555-5555-4555-8555-555555555555",
                    "66666666-6666-4666-8666-666666666666",
                ]
                rooms = [0, 2**63 - 1]
                digests = ["55" * 32, "66" * 32]
                decode_items = decode.receiver_create_many(rooms, attempts, digests)
                prefill_items = prefill.sender_create_many(
                    [
                        "22222222-2222-4222-8222-222222222222",
                        "22222222-2222-4222-8222-222222222222",
                    ],
                    rooms,
                    attempts,
                    digests,
                )
                decode_handles = [item.handle for item in decode_items]
                prefill_handles = [item.handle for item in prefill_items]
                self.assertEqual(
                    [item.terminal_generation for item in decode_items],
                    [1, 2],
                )
                self.assertEqual(
                    [item.terminal_generation for item in prefill_items],
                    [1, 2],
                )

                prepare = executor.submit(
                    decode.receiver_prepare_many,
                    decode_handles,
                    [[0], [1]],
                    [1, 64],
                )
                initialize = executor.submit(prefill.sender_init_many, prefill_handles)
                self.assertTrue(all(item.ok for item in initialize.result(timeout=10)))
                send = executor.submit(
                    prefill.sender_send_chunks,
                    prefill_handles,
                    [56 * 2_048, 64 * 56 * 2_048],
                    [[0], [1]],
                    [42, None],
                    [1, 64],
                )
                prepared_items = prepare.result(timeout=10)
                self.assertTrue(all(item.ok for item in prepared_items))
                destination_poll = executor.submit(decode.poll_many, decode_handles)
                self.assertTrue(all(item.ok for item in send.result(timeout=10)))
                first = destination_poll.result(timeout=10)

            self.assertEqual([item.status for item in first], [4, 4])
            self.assertEqual([item.terminal_generation for item in first], [1, 2])
            self.assertEqual([item.first_token_id for item in first], [42, None])
            second = decode.poll_many(decode_handles)
            self.assertEqual([item.first_token_id for item in second], [None, None])
            self.assertTrue(second[0].first_token_consumed)
            self.assertTrue(
                all(item.status == 4 for item in prefill.poll_many(prefill_handles))
            )

            self.assertTrue(all(item.ok for item in decode.clear_many(decode_handles)))
            self.assertTrue(
                all(item.ok for item in prefill.clear_many(prefill_handles))
            )
            self.assertEqual(decode.readiness().snapshot().active_handles, 0)
            self.assertEqual(prefill.readiness().snapshot().active_handles, 0)
            resource_fields = (
                "active_rooms",
                "active_handles",
                "result_slots",
                "pending_prepares",
                "wire_plans",
                "native_leases",
                "source_kv_pages",
                "destination_kv_pages",
                "aux_slots",
                "completion_slots",
                "request_slots",
                "in_flight_transfers",
                "native_batches",
                "pending_bytes",
                "quarantined_rooms",
            )
            for transport in (prefill, decode):
                snapshot = transport.resource_snapshot()
                self.assertEqual(
                    [getattr(snapshot, field) for field in resource_fields],
                    [0] * len(resource_fields),
                )
            prefill.shutdown()
            decode.shutdown()

    def test_two_independent_processes_use_the_authenticated_mock_data_plane(self):
        context = get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            psk_file = Path(directory) / "control.psk"
            psk_file.write_bytes(bytes(reversed(range(32))))
            psk_file.chmod(0o400)
            control_port = _free_control_port()
            p_commands, p_results = context.Queue(), context.Queue()
            d_commands, d_results = context.Queue(), context.Queue()
            prefill = context.Process(
                target=_peer_main,
                args=(
                    "prefill",
                    _config("prefill", control_port, psk_file),
                    p_commands,
                    p_results,
                ),
            )
            decode = context.Process(
                target=_peer_main,
                args=(
                    "decode",
                    _config("decode", control_port, psk_file),
                    d_commands,
                    d_results,
                ),
            )
            prefill.start()
            decode.start()
            try:
                _send(p_commands, "start")
                time.sleep(0.05)
                _send(d_commands, "start")
                _receive(d_results, "start")
                _receive(p_results, "start")

                attempt = ["77777777-7777-4777-8777-777777777777"]
                room = [0]
                digest = ["77" * 32]
                _send(d_commands, "receiver_create_many", room, attempt, digest)
                d_handle = _receive(d_results, "receiver_create_many")[0]["handle"]
                _send(
                    p_commands,
                    "sender_create_many",
                    ["22222222-2222-4222-8222-222222222222"],
                    room,
                    attempt,
                    digest,
                )
                p_handle = _receive(p_results, "sender_create_many")[0]["handle"]

                _send(d_commands, "receiver_prepare_many", [d_handle], [[0]], [1])
                _send(p_commands, "sender_init_many", [p_handle])
                self.assertTrue(_receive(p_results, "sender_init_many")[0]["ok"])
                _send(
                    p_commands,
                    "sender_send_chunks",
                    [p_handle],
                    [56 * 2_048],
                    [[0]],
                    [43],
                    [1],
                )
                self.assertTrue(_receive(d_results, "receiver_prepare_many")[0]["ok"])
                _send(d_commands, "poll_many", [d_handle])
                self.assertTrue(_receive(p_results, "sender_send_chunks")[0]["ok"])
                result = _receive(d_results, "poll_many")[0]
                self.assertEqual((result["status"], result["first_token_id"]), (4, 43))

                _send(d_commands, "poll_many", [d_handle])
                self.assertIsNone(_receive(d_results, "poll_many")[0]["first_token_id"])
                _send(d_commands, "clear_many", [d_handle])
                _send(p_commands, "clear_many", [p_handle])
                self.assertTrue(_receive(d_results, "clear_many")[0]["ok"])
                self.assertTrue(_receive(p_results, "clear_many")[0]["ok"])
                _send(d_commands, "snapshot")
                _send(p_commands, "snapshot")
                self.assertEqual(_receive(d_results, "snapshot")["active_handles"], 0)
                self.assertEqual(_receive(p_results, "snapshot")["active_handles"], 0)
                _send(p_commands, "shutdown")
                _send(d_commands, "shutdown")
                _receive(p_results, "shutdown")
                _receive(d_results, "shutdown")
            finally:
                prefill.join(timeout=5)
                decode.join(timeout=5)
                if prefill.is_alive():
                    prefill.terminate()
                    prefill.join(timeout=5)
                if decode.is_alive():
                    decode.terminate()
                    decode.join(timeout=5)
            self.assertEqual(prefill.exitcode, 0)
            self.assertEqual(decode.exitcode, 0)

    def test_surviving_prefill_rejects_old_epoch_and_recovers_with_new_decode(self):
        module = importlib.import_module(
            os.environ.get("SGLANG_PD_CORE_MODULE", "_core")
        )
        with tempfile.TemporaryDirectory() as directory:
            psk_file = Path(directory) / "control.psk"
            psk_file.write_bytes(bytes(range(32)))
            psk_file.chmod(0o400)
            control_port = _free_control_port()
            prefill = module.PdTransport(_config("prefill", control_port, psk_file))
            first_decode = module.PdTransport(_config("decode", control_port, psk_file))
            with ThreadPoolExecutor(max_workers=4) as executor:
                prefill_start = executor.submit(prefill.start)
                first_start = executor.submit(first_decode.start)
                first_start.result(timeout=10)
                prefill_start.result(timeout=10)
            first_epoch = first_decode.readiness().snapshot().process_epoch
            self.assertTrue(prefill.readiness().snapshot().pair_ready)

            first_decode.shutdown()
            deadline = time.monotonic() + 10
            while (
                prefill.readiness().snapshot().lifecycle != "LocalReady"
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            disconnected = prefill.readiness().snapshot()
            self.assertFalse(disconnected.pair_ready)
            self.assertFalse(disconnected.accepting_rooms)
            self.assertEqual(disconnected.reconnect_generation, 1)

            duplicate = module.PdTransport(_config("decode", control_port, psk_file))
            try:
                duplicate.start()
            except RuntimeError:
                pass
            # A same-epoch candidate may complete its half of the final canary
            # before the surviving endpoint rejects ownership. It must still
            # converge no later than the frozen two-miss heartbeat boundary.
            deadline = time.monotonic() + 12
            while (
                duplicate.readiness().snapshot().pair_ready
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            self.assertFalse(duplicate.readiness().snapshot().pair_ready)
            duplicate.shutdown()
            self.assertNotEqual(prefill.readiness().snapshot().lifecycle, "PairReady")

            second_epoch = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
            second_decode = module.PdTransport(
                _config(
                    "decode",
                    control_port,
                    psk_file,
                    process_epoch=second_epoch,
                    registration_epoch="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                )
            )
            second_decode.start()
            deadline = time.monotonic() + 10
            while (
                not prefill.readiness().snapshot().pair_ready
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            recovered = prefill.readiness().snapshot()
            self.assertTrue(recovered.pair_ready)
            self.assertTrue(second_decode.readiness().snapshot().pair_ready)
            self.assertEqual(recovered.peer_process_epoch, second_epoch)
            self.assertNotEqual(second_epoch, first_epoch)
            self.assertEqual(recovered.session_count, 2)

            attempt = "99999999-9999-4999-8999-999999999999"
            digest = "99" * 32
            decode_handle = second_decode.receiver_create_many(
                [1], [attempt], [digest]
            )[0].handle
            prefill_handle = prefill.sender_create_many(
                [second_epoch], [1], [attempt], [digest]
            )[0].handle
            with ThreadPoolExecutor(max_workers=4) as executor:
                prepare = executor.submit(
                    second_decode.receiver_prepare_many,
                    [decode_handle],
                    [[0]],
                    [1],
                )
                initialize = executor.submit(prefill.sender_init_many, [prefill_handle])
                self.assertTrue(initialize.result(timeout=10)[0].ok)
                send = executor.submit(
                    prefill.sender_send_chunks,
                    [prefill_handle],
                    [56 * 2_048],
                    [[0]],
                    [17],
                    [1],
                )
                self.assertTrue(prepare.result(timeout=10)[0].ok)
                poll = executor.submit(second_decode.poll_many, [decode_handle])
                self.assertTrue(send.result(timeout=10)[0].ok)
                result = poll.result(timeout=10)[0]
            self.assertEqual((result.status, result.first_token_id), (4, 17))

            self.assertTrue(second_decode.clear_many([decode_handle])[0].ok)
            self.assertTrue(prefill.clear_many([prefill_handle])[0].ok)
            prefill.shutdown()
            second_decode.shutdown()

    def _create_process_room(self, queues, decode_epoch, attempt):
        room = [0]
        digest = ["77" * 32]
        _send(
            queues["decode"][0],
            "receiver_create_many",
            room,
            [attempt],
            digest,
        )
        decode_handle = _receive(queues["decode"][1], "receiver_create_many")[0][
            "handle"
        ]
        _send(
            queues["prefill"][0],
            "sender_create_many",
            [decode_epoch],
            room,
            [attempt],
            digest,
        )
        prefill_handle = _receive(queues["prefill"][1], "sender_create_many")[0][
            "handle"
        ]
        return {"prefill": prefill_handle, "decode": decode_handle}

    def _prepare_process_room(self, queues, handles):
        _send(
            queues["decode"][0],
            "receiver_prepare_many",
            [handles["decode"]],
            [[0]],
            [1],
        )
        _send(
            queues["prefill"][0],
            "sender_init_many",
            [handles["prefill"]],
        )
        self.assertTrue(_receive(queues["prefill"][1], "sender_init_many")[0]["ok"])

    def _send_process_room(self, queues, handles, first_token_id):
        _send(
            queues["prefill"][0],
            "sender_send_chunks",
            [handles["prefill"]],
            [56 * 2_048],
            [[0]],
            [first_token_id],
            [1],
        )
        self.assertTrue(_receive(queues["decode"][1], "receiver_prepare_many")[0]["ok"])

    def _complete_process_room(self, queues, handles, first_token_id):
        self._send_process_room(queues, handles, first_token_id)
        _send(queues["decode"][0], "poll_many", [handles["decode"]])
        self.assertTrue(_receive(queues["prefill"][1], "sender_send_chunks")[0]["ok"])
        result = _receive(queues["decode"][1], "poll_many")[0]
        self.assertEqual(
            (result["status"], result["pd_reason"], result["first_token_id"]),
            (4, "PD_SUCCESS", first_token_id),
        )
        return result

    def _clear_process_room(self, queues, handles):
        for role in ("prefill", "decode"):
            _send(queues[role][0], "clear_many", [handles[role]])
        for role in ("prefill", "decode"):
            self.assertTrue(_receive(queues[role][1], "clear_many")[0]["ok"])

    def _assert_process_resources_clean(self, queues, roles=("prefill", "decode")):
        for role in roles:
            _send(queues[role][0], "resource_snapshot")
        for role in roles:
            snapshot = _receive(queues[role][1], "resource_snapshot")
            self.assertEqual(
                snapshot,
                {
                    "active_rooms": 0,
                    "active_handles": 0,
                    "result_slots": 0,
                    "pending_prepares": 0,
                    "wire_plans": 0,
                    "native_leases": 0,
                    "source_kv_pages": 0,
                    "destination_kv_pages": 0,
                    "aux_slots": 0,
                    "completion_slots": 0,
                    "request_slots": 0,
                    "in_flight_transfers": 0,
                    "native_batches": 0,
                    "pending_bytes": 0,
                    "quarantined_rooms": 0,
                },
            )

    def _assert_stage_signal_recovery(self, killed_role, process_signal, stage):
        context = get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            psk_file = Path(directory) / "control.psk"
            psk_file.write_bytes(bytes(range(32)))
            psk_file.chmod(0o400)
            upstream_port = _free_control_port()
            proxy_port = _free_control_port()
            while proxy_port == upstream_port:
                proxy_port = _free_control_port()
            pause_kind = {
                "post_submit_pre_data_ready": 12,
                "data_ready_pre_ack": 13,
            }.get(stage)
            proxy = _ControlFrameProxy(proxy_port, upstream_port, pause_kind)
            proxy.start()
            queues = {
                role: (context.Queue(), context.Queue())
                for role in ("prefill", "decode")
            }
            control_ports = {"prefill": upstream_port, "decode": proxy_port}
            processes = {
                role: context.Process(
                    target=_peer_main,
                    args=(
                        role,
                        _config(role, control_ports[role], psk_file),
                        *queues[role],
                    ),
                )
                for role in ("prefill", "decode")
            }
            replacement = None
            replacement_queues = None
            active_queues = None
            clean_shutdown = False
            for process in processes.values():
                process.start()
            try:
                _send(queues["prefill"][0], "start")
                time.sleep(0.05)
                _send(queues["decode"][0], "start")
                _receive(queues["decode"][1], "start")
                _receive(queues["prefill"][1], "start")

                survivor_role = "decode" if killed_role == "prefill" else "prefill"
                handles = None
                pending_operations = {}
                if stage != "startup_pair_ready":
                    handles = self._create_process_room(
                        queues,
                        "22222222-2222-4222-8222-222222222222",
                        "77777777-7777-4777-8777-777777777777",
                    )
                if stage == "rendezvous_pre_submit":
                    self._prepare_process_room(queues, handles)
                    pending_operations["decode"] = "receiver_prepare_many"
                elif stage in (
                    "post_submit_pre_data_ready",
                    "data_ready_pre_ack",
                    "terminal_pre_clear",
                ):
                    self._prepare_process_room(queues, handles)
                if stage in (
                    "post_submit_pre_data_ready",
                    "data_ready_pre_ack",
                ):
                    self._send_process_room(queues, handles, 43)
                    pending_operations["prefill"] = "sender_send_chunks"
                    if stage == "data_ready_pre_ack":
                        _send(
                            queues["decode"][0],
                            "poll_many",
                            [handles["decode"]],
                        )
                        pending_operations["decode"] = "poll_many"
                    self.assertTrue(
                        proxy.reached.wait(10),
                        f"control proxy did not reach stage {stage}",
                    )
                elif stage == "terminal_pre_clear":
                    self._complete_process_room(queues, handles, 43)

                os.kill(processes[killed_role].pid, process_signal)
                processes[killed_role].join(timeout=5)
                self.assertEqual(
                    processes[killed_role].exitcode,
                    -int(process_signal),
                )
                proxy.release.set()

                survivor_operation = pending_operations.get(survivor_role)
                if survivor_operation is not None:
                    _receive_outcome(
                        queues[survivor_role][1],
                        survivor_operation,
                        timeout=35,
                    )
                disconnected = _wait_snapshot(
                    *queues[survivor_role],
                    lambda snapshot: snapshot["lifecycle"] == "LocalReady",
                    timeout=15,
                )
                self.assertFalse(disconnected["pair_ready"])
                self.assertFalse(disconnected["accepting_rooms"])
                if handles is None:
                    self.assertEqual(disconnected["active_handles"], 0)
                else:
                    _send(
                        queues[survivor_role][0],
                        "poll_many",
                        [handles[survivor_role]],
                    )
                    first = _receive(queues[survivor_role][1], "poll_many")[0]
                    _send(
                        queues[survivor_role][0],
                        "poll_many",
                        [handles[survivor_role]],
                    )
                    second = _receive(queues[survivor_role][1], "poll_many")[0]
                    self.assertEqual(first["pd_reason"], second["pd_reason"])
                    if stage == "terminal_pre_clear":
                        self.assertEqual(first["pd_reason"], "PD_SUCCESS")
                    else:
                        self.assertIn(
                            first["pd_reason"],
                            ("PD_PEER_UNAVAILABLE", "PD_SUCCESS"),
                        )
                    self.assertIsNone(second["first_token_id"])
                    _send(
                        queues[survivor_role][0],
                        "clear_many",
                        [handles[survivor_role]],
                    )
                    self.assertTrue(
                        _receive(queues[survivor_role][1], "clear_many")[0]["ok"]
                    )
                    self._assert_process_resources_clean(queues, roles=(survivor_role,))

                replacement_epoch = {
                    "prefill": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "decode": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                }[killed_role]
                replacement_registration = {
                    "prefill": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                    "decode": "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
                }[killed_role]
                replacement_queues = (context.Queue(), context.Queue())
                replacement = context.Process(
                    target=_peer_main,
                    args=(
                        killed_role,
                        _config(
                            killed_role,
                            control_ports[killed_role],
                            psk_file,
                            process_epoch=replacement_epoch,
                            registration_epoch=replacement_registration,
                        ),
                        *replacement_queues,
                    ),
                )
                replacement.start()
                _send(replacement_queues[0], "start")
                _receive(replacement_queues[1], "start", timeout=15)
                recovered = _wait_snapshot(
                    *queues[survivor_role],
                    lambda snapshot: snapshot["pair_ready"],
                )
                self.assertEqual(recovered["peer_process_epoch"], replacement_epoch)
                self.assertEqual(recovered["session_count"], 2)

                active_queues = dict(queues)
                active_queues[killed_role] = replacement_queues
                decode_epoch = (
                    replacement_epoch
                    if killed_role == "decode"
                    else "22222222-2222-4222-8222-222222222222"
                )
                recovered_handles = self._create_process_room(
                    active_queues,
                    decode_epoch,
                    "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                )
                self._prepare_process_room(active_queues, recovered_handles)
                self._complete_process_room(active_queues, recovered_handles, 29)
                self._clear_process_room(active_queues, recovered_handles)
                self._assert_process_resources_clean(active_queues)
                for role in ("prefill", "decode"):
                    _send(active_queues[role][0], "shutdown")
                for role in ("prefill", "decode"):
                    self.assertEqual(
                        _receive(active_queues[role][1], "shutdown"), "SafeTerminal"
                    )
                clean_shutdown = True
            finally:
                for process in (*processes.values(), replacement):
                    if process is None:
                        continue
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                proxy.close()
            if clean_shutdown:
                self.assertEqual(processes[survivor_role].exitcode, 0)
                self.assertEqual(replacement.exitcode, 0)

    def test_prefill_decode_stage_signal_restart_matrix(self):
        stages = (
            "post_submit_pre_data_ready",
            "rendezvous_pre_submit",
            "startup_pair_ready",
            "data_ready_pre_ack",
            "terminal_pre_clear",
        )
        for process_signal in (signal.SIGTERM, signal.SIGKILL):
            for killed_role in ("prefill", "decode"):
                for stage in stages:
                    with self.subTest(
                        signal=signal.Signals(process_signal).name,
                        killed_role=killed_role,
                        stage=stage,
                    ):
                        self._assert_stage_signal_recovery(
                            killed_role, process_signal, stage
                        )


if __name__ == "__main__":
    unittest.main()
