"""Source-tree smoke for the independent Rust PD transport handles.

Run with the interpreter into which the current checkout's maturin extension
was installed. The test intentionally uses only the standard library.
"""

from __future__ import annotations

import http.client
import importlib
import json
import os
import socket
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path


def _free_control_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _config(role: str, control_port: int, psk_file: Path) -> str:
    process_epoch = {
        "prefill": "11111111-1111-4111-8111-111111111111",
        "decode": "22222222-2222-4222-8222-222222222222",
    }[role]
    registration_epoch = {
        "prefill": "33333333-3333-4333-8333-333333333333",
        "decode": "44444444-4444-4444-8444-444444444444",
    }[role]
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


if __name__ == "__main__":
    unittest.main()
