from __future__ import annotations

import http.client
import importlib
import json
import multiprocessing
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sglang.srt.disaggregation.rust_pd import RustPdSchedulerAdapter
from sglang.srt.managers.rust_server import RustServer
from sglang.srt.managers.utils import is_health_check_generate_req
from sglang.test.rust_pd_http_e2e_support import (
    API_KEY,
    CLEAN_RESOURCES,
    MODEL_PATH,
    normal_batch,
    output_payload,
    prepare_request,
    region_table,
    resource_counts,
    scheduler_request,
    server_args,
    transport_config,
    typed_failure,
)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _worker_main(
    role: str,
    http_port: int,
    control_port: int,
    data_port: int,
    psk_file: str,
    commands: multiprocessing.Queue,
    events: multiprocessing.Queue,
) -> None:
    module = importlib.import_module(os.environ.get("SGLANG_PD_CORE_MODULE", "_core"))
    sys.modules["sglang.srt.server._core"] = module
    transport = module.PdTransport(
        transport_config(role, control_port, data_port, psk_file)
    )
    server = None
    try:
        transport.start()
        server = module.Server(
            http_addr=f"127.0.0.1:{http_port}",
            server_args_json=server_args(role, http_port, control_port, psk_file),
            pd_readiness=transport.readiness(),
        )
        rust_server = RustServer(server)
        pool, table = region_table(role)
        adapter = RustPdSchedulerAdapter(transport, role, table, pool)
        held: dict[str, tuple[Any, bool]] = {}
        events.put((role, "ready", http_port))

        while True:
            try:
                command = commands.get_nowait()
                if command == "stop":
                    break
                if command == "snapshot":
                    events.put((role, "snapshot", resource_counts(adapter, transport)))
                    continue
            except Exception:
                pass
            rust_server.wait_ingress(50)
            incoming = rust_server.drain(8)
            generated: list[Any] = []
            for item in incoming:
                if type(item).__name__ == "AbortReq":
                    held_item = held.pop(item.rid, None)
                    if held_item is not None:
                        request, owns_room = held_item
                        if owns_room:
                            _, active = adapter.abort_matching(request.rid)
                            adapter.clear_terminal(active)
                        events.put(
                            (
                                role,
                                "aborted",
                                item.rid,
                                resource_counts(adapter, transport),
                            )
                        )
                    continue
                generated.append(item)

            if generated:
                time.sleep(0.02)
                for item in rust_server.drain(8):
                    if type(item).__name__ != "AbortReq":
                        generated.append(item)

            normal: list[Any] = []
            post_submit: list[Any] = []
            for request in generated:
                if type(request).__name__ == "GetInternalStateReq":
                    rust_server.push_control_output(request, {"internal_state": {}})
                    continue
                if is_health_check_generate_req(request):
                    health_request = SimpleNamespace(
                        rid=request.rid,
                        origin_input_ids=list(request.input_ids),
                        output_ids=[0],
                    )
                    rust_server.push_generation(
                        output_payload(
                            [health_request],
                            [{"type": "length", "length": 1}],
                        )
                    )
                    events.put((role, "health", request.rid))
                    continue
                request = scheduler_request(request)
                ids = list(request.input_ids)
                sentinel = ids[0] if ids else None
                if sentinel == 9001:
                    if role == "prefill":
                        prepare_request(request)
                        typed_failure(rust_server, request, "PD_PROTOCOL_MISMATCH", 500)
                    else:
                        held[request.rid] = (request, False)
                    continue
                if sentinel == 9002:
                    if role == "decode":
                        prepare_request(request)
                        typed_failure(rust_server, request, "PD_TRANSFER_TIMEOUT", 500)
                    else:
                        held[request.rid] = (request, False)
                    continue
                if sentinel == 9003:
                    prepare_request(request)
                    adapter.enqueue(request)
                    adapter.create_many([request])
                    held[request.rid] = (request, True)
                    events.put((role, "held", request.rid))
                    continue
                if sentinel == 9004:
                    post_submit.append(request)
                    continue
                normal.append(request)

            if normal:
                rooms = normal_batch(role, adapter, rust_server, normal)
                events.put(
                    (
                        role,
                        "complete",
                        len(normal),
                        resource_counts(adapter, transport),
                        rooms,
                    )
                )
            if post_submit:
                normal_batch(
                    role,
                    adapter,
                    rust_server,
                    post_submit,
                    publish=False,
                )
                for request in post_submit:
                    held[request.rid] = (request, True)
                    events.put((role, "held_terminal", request.rid))
    except BaseException as error:
        events.put((role, "fatal", repr(error)))
        raise
    finally:
        if server is not None:
            server.shutdown()
        transport.shutdown()


class _EventInbox:
    def __init__(self, queue: multiprocessing.Queue) -> None:
        self.queue = queue
        self.pending: list[tuple[Any, ...]] = []

    def take(self, role: str, name: str, timeout: float = 10) -> tuple[Any, ...]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for index, event in enumerate(self.pending):
                if event[:2] == (role, name):
                    return self.pending.pop(index)
                if len(event) > 1 and event[1] == "fatal":
                    raise AssertionError(f"worker failed: {event}")
            try:
                event = self.queue.get(timeout=min(0.2, deadline - time.monotonic()))
            except Exception:
                continue
            if event[:2] == (role, name):
                return event
            if len(event) > 1 and event[1] == "fatal":
                raise AssertionError(f"worker failed: {event}")
            self.pending.append(event)
        raise AssertionError(f"timed out waiting for {(role, name)}: {self.pending}")


def _http_request(
    port: int, body: dict[str, Any], token: str | None = API_KEY
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    headers = {"content-type": "application/json"}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    connection.request(
        "POST",
        "/generate",
        body=json.dumps(body, separators=(",", ":")),
        headers=headers,
    )
    response = connection.getresponse()
    payload = response.read()
    result = response.status, dict(response.getheaders()), payload
    connection.close()
    return result


def _wait_ready(port: int, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            connection.request("GET", "/readiness")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise AssertionError("Gateway never became ready")


def _get(port: int, path: str) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.request("GET", path)
    response = connection.getresponse()
    body = response.read()
    status = response.status
    connection.close()
    return status, body


class RustPdHttpE2ETest(unittest.TestCase):
    def test_gateway_frontend_scheduler_transport_end_to_end(self):
        gateway = Path(
            os.environ.get(
                "SGLANG_PD_GATEWAY_BIN",
                Path(__file__).parents[3]
                / "sgl-model-gateway"
                / "target"
                / "debug"
                / "smg",
            )
        )
        self.assertTrue(gateway.is_file(), f"build Gateway first: {gateway}")
        self.assertTrue((MODEL_PATH / "tokenizer.json").is_file())

        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            psk_file = Path(directory) / "control.psk"
            psk_file.write_bytes(bytes(range(32)))
            psk_file.chmod(0o400)
            control_port = _free_port()
            data_port = 19000
            prefill_port = _free_port()
            decode_port = _free_port()
            gateway_port = _free_port()
            commands = {role: context.Queue() for role in ("prefill", "decode")}
            event_queue = context.Queue()
            inbox = _EventInbox(event_queue)
            workers = [
                context.Process(
                    target=_worker_main,
                    args=(
                        role,
                        prefill_port if role == "prefill" else decode_port,
                        control_port,
                        data_port,
                        str(psk_file),
                        commands[role],
                        event_queue,
                    ),
                )
                for role in ("prefill", "decode")
            ]
            for worker in workers:
                worker.start()

            router = None
            gateway_log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
            try:
                inbox.take("prefill", "ready", timeout=30)
                inbox.take("decode", "ready", timeout=30)
                for role, port in (
                    ("prefill", prefill_port),
                    ("decode", decode_port),
                ):
                    self.assertEqual(_get(port, "/readiness")[0], 200)
                    health_status, health_body = _get(port, "/health")
                    inbox.take(role, "health", timeout=1)
                    self.assertEqual(
                        health_status,
                        200,
                        f"Frontend /health failed: {health_body!r}",
                    )
                router = subprocess.Popen(
                    [
                        str(gateway),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(gateway_port),
                        "--api-key",
                        API_KEY,
                        "--pd-disaggregation",
                        "--prefill",
                        f"http://127.0.0.1:{prefill_port}",
                        str(control_port),
                        "--decode",
                        f"http://127.0.0.1:{decode_port}",
                        "--prefill-policy",
                        "round_robin",
                        "--decode-policy",
                        "round_robin",
                        "--health-check-endpoint",
                        "/readiness",
                        "--health-success-threshold",
                        "1",
                        "--health-failure-threshold",
                        "1",
                        "--health-check-interval-secs",
                        "1",
                        "--worker-startup-timeout-secs",
                        "30",
                        "--worker-startup-check-interval",
                        "1",
                        "--disable-retries",
                    ],
                    stdout=gateway_log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    _wait_ready(gateway_port)
                except AssertionError as error:
                    gateway_log.flush()
                    gateway_log.seek(0)
                    diagnostics = [
                        line
                        for line in gateway_log.read().splitlines()
                        if "uri=/readiness" not in line
                    ][-200:]
                    raise AssertionError(
                        f"{error}; Gateway exit={router.poll()}\n{'\n'.join(diagnostics)}"
                    ) from error

                request = {
                    "input_ids": [1, 2],
                    "sampling_params": {
                        "temperature": 0,
                        "top_p": 1,
                        "min_p": 0,
                        "max_new_tokens": 2,
                    },
                    "bootstrap_host": "attacker.invalid",
                    "bootstrap_port": 1,
                    "bootstrap_room": 2**63 - 1,
                    "bootstrap_attempt_id": "client-controlled",
                }
                self.assertEqual(_http_request(gateway_port, request, None)[0], 401)
                self.assertEqual(_http_request(gateway_port, request, "wrong")[0], 401)

                status, _, body = _http_request(gateway_port, request)
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["output_ids"], [42, 43])
                observed_rooms = []
                for role in ("prefill", "decode"):
                    event = inbox.take(role, "complete")
                    self.assertEqual(event[3], CLEAN_RESOURCES)
                    observed_rooms.append(event[4])
                self.assertEqual(observed_rooms, [[0], [0]])

                text_request = {
                    "text": "hello",
                    "sampling_params": request["sampling_params"],
                }
                status, _, body = _http_request(gateway_port, text_request)
                self.assertEqual(status, 200)
                self.assertIn("meta_info", json.loads(body))
                for role in ("prefill", "decode"):
                    self.assertEqual(inbox.take(role, "complete")[3], CLEAN_RESOURCES)

                batch_request = {
                    "input_ids": [[3], [4, 5]],
                    "sampling_params": request["sampling_params"],
                }
                status, _, body = _http_request(gateway_port, batch_request)
                self.assertEqual(status, 200)
                batch = json.loads(body)
                self.assertEqual([item["output_ids"] for item in batch], [[42, 43]] * 2)
                for role in ("prefill", "decode"):
                    event = inbox.take(role, "complete")
                    self.assertEqual(event[2], 2)
                    self.assertEqual(event[3], CLEAN_RESOURCES)

                stream_request = {
                    **request,
                    "stream": True,
                }
                status, headers, body = _http_request(gateway_port, stream_request)
                self.assertEqual(status, 200)
                self.assertTrue(headers["content-type"].startswith("text/event-stream"))
                self.assertEqual(body.count(b"data: [DONE]"), 1)
                for role in ("prefill", "decode"):
                    self.assertEqual(inbox.take(role, "complete")[3], CLEAN_RESOURCES)

                edge_cases = [
                    (0, [9012], [], "length", None),
                    (1, [9013], [42], "length", None),
                    (2, [9010], [], "stop", 151645),
                    (2, [9011], [], "stop", 77),
                ]
                for limit, input_ids, expected_ids, finish_type, matched in edge_cases:
                    sampling = {
                        **request["sampling_params"],
                        "max_new_tokens": limit,
                    }
                    if input_ids == [9011]:
                        sampling["stop_token_ids"] = [77]
                    status, _, body = _http_request(
                        gateway_port,
                        {
                            "input_ids": input_ids,
                            "sampling_params": sampling,
                        },
                    )
                    self.assertEqual(status, 200)
                    result = json.loads(body)
                    self.assertEqual(result.get("output_ids", []), expected_ids)
                    finish = result["meta_info"]["finish_reason"]
                    self.assertEqual(finish["type"], finish_type)
                    if matched is not None:
                        self.assertEqual(finish["matched"], matched)
                    for role in ("prefill", "decode"):
                        self.assertEqual(
                            inbox.take(role, "complete")[3], CLEAN_RESOURCES
                        )

                unsupported = {
                    **request,
                    "sampling_params": {
                        **request["sampling_params"],
                        "temperature": 0.5,
                    },
                }
                status, headers, body = _http_request(gateway_port, unsupported)
                self.assertEqual(status, 422)
                self.assertEqual(headers["x-sglang-pd-reason"], "PD_UNSUPPORTED")
                self.assertEqual(
                    json.loads(body)["error"]["pd_reason"], "PD_UNSUPPORTED"
                )
                for role in ("prefill", "decode"):
                    commands[role].put("snapshot")
                    self.assertEqual(inbox.take(role, "snapshot")[2], CLEAN_RESOURCES)

                p_first = {
                    **request,
                    "input_ids": [9001],
                    "stream": True,
                }
                status, headers, body = _http_request(gateway_port, p_first)
                self.assertEqual(status, 503)
                self.assertEqual(headers["x-sglang-pd-reason"], "PD_PROTOCOL_MISMATCH")
                self.assertTrue(
                    headers["content-type"].startswith("text/event-stream"),
                    headers,
                )
                self.assertEqual(body.count(b"data: [DONE]"), 1, body)
                self.assertEqual(
                    inbox.take("decode", "aborted")[3],
                    CLEAN_RESOURCES,
                )

                d_first = {**request, "input_ids": [9002]}
                status, headers, body = _http_request(gateway_port, d_first)
                self.assertEqual(status, 504)
                self.assertEqual(headers["x-sglang-pd-reason"], "PD_TRANSFER_TIMEOUT")
                self.assertEqual(
                    json.loads(body)["error"]["pd_reason"],
                    "PD_TRANSFER_TIMEOUT",
                )
                self.assertEqual(
                    inbox.take("prefill", "aborted")[3],
                    CLEAN_RESOURCES,
                )

                for sentinel, held_event in (
                    (9003, "held"),
                    (9004, "held_terminal"),
                ):
                    cancel_body = json.dumps(
                        {**request, "input_ids": [sentinel]},
                        separators=(",", ":"),
                    ).encode()
                    client = socket.create_connection(("127.0.0.1", gateway_port))
                    client.sendall(
                        (
                            "POST /generate HTTP/1.1\r\n"
                            f"Host: 127.0.0.1:{gateway_port}\r\n"
                            "Content-Type: application/json\r\n"
                            f"Authorization: Bearer {API_KEY}\r\n"
                            f"Content-Length: {len(cancel_body)}\r\n\r\n"
                        ).encode()
                        + cancel_body
                    )
                    inbox.take("prefill", held_event)
                    inbox.take("decode", held_event)
                    client.close()
                    for role in ("prefill", "decode"):
                        self.assertEqual(
                            inbox.take(role, "aborted", timeout=15)[3],
                            CLEAN_RESOURCES,
                        )
            finally:
                if router is not None:
                    if router.poll() is None:
                        router.terminate()
                    try:
                        router.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        router.kill()
                        router.wait(timeout=5)
                gateway_log.close()
                for role in ("prefill", "decode"):
                    commands[role].put("stop")
                for worker in workers:
                    worker.join(timeout=15)
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                self.assertEqual([worker.exitcode for worker in workers], [0, 0])
