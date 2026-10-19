"""CPU-only protocol tests: no model, torch, or SRT imports."""

import asyncio
from contextlib import contextmanager

import pytest
import zmq
import zmq.asyncio
from sglang_simulator.simulation.client import (
    AsyncCoordinatorClient,
    validate_endpoint,
)
from sglang_simulator.simulation.coordinator import (
    Cancel,
    Coordinator,
    Grant,
    ProtocolError,
    Ready,
    Submit,
    Terminal,
    decode_grant,
    decode_ready,
    encode_message,
)


def test_payload_barrier_and_manifest_order():
    coordinator = Coordinator()
    coordinator.accept_ready(Ready("run", 0, [Submit("a"), Submit("b")]))
    with pytest.raises(ProtocolError, match="missing payloads"):
        coordinator.advance(10)
    with pytest.raises(ProtocolError, match="not next in manifest"):
        coordinator.mark_admitted("b")
    coordinator.mark_admitted("a")
    assert coordinator.pending_rids == ("b",)
    with pytest.raises(ProtocolError, match="missing payloads"):
        coordinator.grant()
    coordinator.mark_admitted("b")
    coordinator.advance(10)
    assert coordinator.now_ns == 10


def test_completion_requires_driver_acknowledgement_before_next_batch():
    coordinator = Coordinator()
    coordinator.accept_ready(Ready("run", 0, [Submit("a")], next_wakeup_ns=100))
    coordinator.mark_admitted("a")
    coordinator.advance(10)
    coordinator.terminal("a", "COMPLETED")
    with pytest.raises(ProtocolError, match="deliver terminal events"):
        coordinator.advance(20)
    grant = coordinator.grant()
    assert grant.terminals == [Terminal("a", "COMPLETED", 10)]
    with pytest.raises(ProtocolError, match="before Ready"):
        coordinator.advance(20)
    coordinator.accept_ready(Ready("run", 1, [Submit("a2")], next_wakeup_ns=100))
    coordinator.mark_admitted("a2")
    coordinator.advance(20)
    coordinator.terminal("a2", "COMPLETED")
    assert coordinator.grant().now_ns == 20


def test_timer_inside_batch_preserves_arrival_and_completion_time():
    coordinator = Coordinator()
    coordinator.accept_ready(Ready("run", 0, [Submit("a")], next_wakeup_ns=12))
    coordinator.mark_admitted("a")
    with pytest.raises(ProtocolError, match="past timer"):
        coordinator.advance(20)
    coordinator.advance(12)
    assert coordinator.grant().terminals == []
    coordinator.accept_ready(Ready("run", 1, [Submit("b")]))
    coordinator.mark_admitted("b")
    coordinator.advance(20)
    coordinator.terminal("a", "COMPLETED")
    assert coordinator.grant().terminals == [Terminal("a", "COMPLETED", 20)]
    assert coordinator.open_rids == ("b",)


def test_submit_cancel_does_not_wait_for_payload_and_late_done_is_ignored():
    coordinator = Coordinator()
    coordinator.accept_ready(Ready("run", 0, [Submit("a"), Cancel("a")]))
    assert coordinator.pending_rids == ()
    assert coordinator.is_cancelled("a")
    assert coordinator.is_terminal("a")
    assert coordinator.has_terminals
    assert not coordinator.terminal("a", "COMPLETED")
    assert coordinator.grant().terminals == [
        Terminal("a", "CANCELLED", 0, "cancelled by driver")
    ]
    coordinator.accept_ready(Ready("run", 1, [Cancel("a")], finished=True))
    grant = coordinator.grant(finished=True)
    assert grant.status == "FINISHED"
    assert not grant.terminals


def test_running_cancel_keeps_virtual_batch_time_and_has_one_terminal():
    coordinator = Coordinator()
    coordinator.accept_ready(Ready("run", 0, [Submit("a")], next_wakeup_ns=12))
    coordinator.mark_admitted("a")
    coordinator.advance(12)
    coordinator.grant()
    coordinator.accept_ready(Ready("run", 1, [Cancel("a")]))
    assert coordinator.grant().terminals[0].at_ns == 12
    coordinator.accept_ready(Ready("run", 2, [], finished=True))
    coordinator.advance(20)
    assert not coordinator.terminal("a", "COMPLETED")
    assert coordinator.grant(finished=True).now_ns == 20


def test_terminals_follow_submit_order_not_native_callback_order():
    coordinator = Coordinator()
    coordinator.accept_ready(Ready("run", 0, [Submit("a"), Submit("b")]))
    coordinator.mark_admitted("a")
    coordinator.mark_admitted("b")
    coordinator.advance(20)
    coordinator.terminal("b", "COMPLETED")
    coordinator.terminal("a", "COMPLETED")
    assert [event.rid for event in coordinator.grant().terminals] == ["a", "b"]


@pytest.mark.parametrize(
    "bad_ready, pattern",
    [
        (Ready("wrong", 1, []), "unexpected run_id"),
        (Ready("run", 0, []), "expected round"),
        (Ready("run", 2, []), "expected round"),
        (Ready("run", 1, [], protocol_version=2), "protocol version"),
        (Ready("run", 1, [], next_wakeup_ns=9), "precedes now"),
        (Ready("run", 1, [Submit("a")]), "duplicate Submit"),
        (Ready("run", 1, [Cancel("unknown")]), "unknown rid"),
        (Ready("run", 1, [Submit("b")], finished=True), "finished driver"),
    ],
)
def test_invalid_ready_does_not_mutate_run(bad_ready, pattern):
    coordinator = Coordinator()
    coordinator.accept_ready(Ready("run", 0, [Submit("a")]))
    coordinator.mark_admitted("a")
    coordinator.advance(10)
    coordinator.terminal("a", "COMPLETED")
    coordinator.grant()
    with pytest.raises(ProtocolError, match=pattern):
        coordinator.accept_ready(bad_ready)
    assert coordinator.awaiting_ready
    assert coordinator.now_ns == 10
    assert coordinator.open_rids == ()
    coordinator.accept_ready(Ready("run", 1, [], finished=True))
    assert coordinator.grant(finished=True).status == "FINISHED"


def test_manifest_validation_is_atomic():
    coordinator = Coordinator()
    with pytest.raises(ProtocolError, match="duplicate Submit"):
        coordinator.accept_ready(Ready("run", 0, [Submit("a"), Submit("a")]))
    assert coordinator.run_id is None
    assert coordinator.open_rids == ()
    coordinator.accept_ready(Ready("run", 0, [Submit("a")]))
    with pytest.raises(ProtocolError, match="previous Ready"):
        coordinator.accept_ready(Ready("run", 0, []))


def test_finish_requires_no_open_requests_and_cannot_resume():
    coordinator = Coordinator()
    coordinator.accept_ready(Ready("run", 0, [Submit("a")]))
    coordinator.mark_admitted("a")
    with pytest.raises(ProtocolError, match="active driver or open requests"):
        coordinator.grant(finished=True)
    coordinator.grant()
    coordinator.accept_ready(Ready("run", 1, [], finished=True))
    with pytest.raises(ProtocolError, match="active driver or open requests"):
        coordinator.grant(finished=True)
    coordinator.terminal("a", "COMPLETED")
    coordinator.grant(finished=True)
    with pytest.raises(ProtocolError, match="run already finished"):
        coordinator.accept_ready(Ready("run", 2, [], finished=True))


def test_json_contract_rejects_unknown_fields_tags_and_noninteger_time():
    ready = Ready("run", 0, [Submit("a"), Cancel("a")])
    assert decode_ready(encode_message(ready)) == ready
    grant = Grant("run", 0, 12, [Terminal("a", "CANCELLED", 12)])
    assert decode_grant(encode_message(grant)) == grant
    for payload in (
        b'{"run_id":"run","round":0,"operations":[],"unexpected":1}',
        b'{"run_id":"run","round":0,"operations":[{"type":"exec"}]}',
        b'{"run_id":"run","round":0,"operations":[],"next_wakeup_ns":1.5}',
        b'{"run_id":"run","round":false,"operations":[]}',
        b"not JSON",
    ):
        with pytest.raises(ProtocolError, match="Invalid Ready"):
            decode_ready(payload)


@pytest.mark.parametrize(
    "endpoint",
    [
        "tcp://0.0.0.0:12345",
        "tcp://192.0.2.1:12345",
        "tcp://localhost:12345",
        "tcp://127.0.0.1:0",
        "tcp://127.0.0.1:65536",
        "tcp://user@127.0.0.1:12345",
        "tcp://127.0.0.1:12345/path",
        "ipc:///tmp/coordinator",
    ],
)
def test_control_endpoint_requires_explicit_loopback(endpoint):
    with pytest.raises(ValueError, match="loopback"):
        validate_endpoint(endpoint)


@contextmanager
def control_server():
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    try:
        yield socket, f"tcp://127.0.0.1:{port}"
    finally:
        socket.close(linger=0)
        context.term()


def test_loopback_client_exchanges_typed_protocol():
    async def scenario():
        with control_server() as (server, endpoint):
            async with AsyncCoordinatorClient(endpoint) as client:
                task = asyncio.create_task(client.request(Ready("run", 0, [])))
                ready = decode_ready(await server.recv())
                assert ready == Ready("run", 0, [])
                await server.send(encode_message(Grant("run", 0, 10, [])))
                assert (await task).now_ns == 10

    asyncio.run(scenario())


def test_control_timeout_fails_run_instead_of_retrying_round():
    async def scenario():
        with control_server() as (_, endpoint):
            async with AsyncCoordinatorClient(endpoint, timeout_s=0.02) as client:
                with pytest.raises(ProtocolError, match="run='run' round=0"):
                    await client.request(Ready("run", 0, []))
                with pytest.raises(ProtocolError, match="client is closed"):
                    await client.request(Ready("run", 0, []))

    asyncio.run(scenario())


def test_client_rejects_concurrent_round_and_mismatched_grant():
    async def scenario():
        with control_server() as (server, endpoint):
            async with AsyncCoordinatorClient(endpoint) as client:
                task = asyncio.create_task(client.request(Ready("run", 0, [])))
                await server.recv()
                with pytest.raises(ProtocolError, match="one Ready"):
                    await client.request(Ready("run", 1, []))
                await server.send(encode_message(Grant("other", 0, 0, [])))
                with pytest.raises(ProtocolError, match="does not match"):
                    await task

    asyncio.run(scenario())
