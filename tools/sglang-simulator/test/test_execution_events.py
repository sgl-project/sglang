"""Retirement exports bind actual records and never seal an interrupted run."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_simulation_coordinated_bridge import harness  # noqa: F401
from sglang_simulator.simulation.sglang.execution_events import ExecutionEventRecorder


def test_original_prefix_and_request_identity(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = ExecutionEventRecorder(path)
    writer.record(10, ["r1", "r2"])
    writer.record(12, [])
    writer.record(20, ["r2"])
    prefix = path.read_bytes()
    writer.finish("run", 30)
    writer.finish("run", 30)
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert records[1] == {
        "event": "retired_worker",
        "end_ns": 10,
        "requests": [{"rid": "r1"}, {"rid": "r2"}],
    }
    assert records[-1]["prefix_bytes"] == len(prefix)
    assert records[-1]["prefix_sha256"] == hashlib.sha256(prefix).hexdigest()
    assert records[-1]["retired_forwards"] == 2
    assert records[-1]["status"] == "FINISHED"
    with pytest.raises(RuntimeError, match="closed"):
        writer.record(31, ["r3"])


def test_interruption_and_invalid_result_have_no_success_footer(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = ExecutionEventRecorder(path)
    writer.record(10, ["r1"])
    original = path.read_bytes()
    with pytest.raises(ValueError, match="monotonic"):
        writer.record(9, ["r2"])
    with pytest.raises(ValueError, match="unique"):
        writer.record(10, ["r2", "r2"])
    writer.close()
    assert path.read_bytes() == original and b"FINISHED" not in original


def test_failed_write_closes_and_cannot_publish_success(tmp_path):
    writer = ExecutionEventRecorder(tmp_path / "events.jsonl")
    writer._stream.close()

    class BrokenStream:
        closed = False

        def write(self, _):
            raise OSError("disk full")

        def close(self):
            self.closed = True

    writer._stream = BrokenStream()
    with pytest.raises(OSError, match="disk full"):
        writer.record(1, ["r1"])
    assert writer._stream.closed
    with pytest.raises(RuntimeError, match="closed"):
        writer.finish("run", 2)


@pytest.fixture
def execution_path(monkeypatch, tmp_path):
    path = tmp_path / "execution.jsonl"
    monkeypatch.setenv("SGLANG_SIMULATOR_EXECUTION_EVENTS", str(path))
    return path


@pytest.fixture
def recorded_harness(execution_path, request):
    # Reuse the existing coordinator fixture's two native message-type seams;
    # its real clock/socket/result-processing implementation stays active.
    return request.getfixturevalue("harness"), execution_path


def test_real_coordinator_records_processed_result_before_finished_grant(
    recorded_harness,
):
    from test_simulation_coordinated_bridge import NativeRequest, rid
    from sglang_simulator.simulation.coordinator import (
        Ready,
        Submit,
        decode_grant,
        encode_message,
    )

    instance, path = recorded_harness

    def driver(control):
        control.send(
            encode_message(Ready("run", 0, [Submit(rid(1))], next_wakeup_ns=20))
        )
        instance.send_payload(rid(1))
        completion = decode_grant(control.recv())
        assert completion.now_ns == 20
        assert [(row.rid, row.at_ns) for row in completion.terminals] == [(rid(1), 20)]
        control.send(encode_message(Ready("run", 1, [], finished=True)))
        return decode_grant(control.recv())

    try:
        with ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(instance.run_driver, driver)
            instance.receive()
            assert b"retired_worker" not in path.read_bytes()
            instance.process([NativeRequest(rid(1))], 20, [])
            assert b"retired_worker" in path.read_bytes()
            assert b"FINISHED" not in path.read_bytes()
            instance.receive()
            assert future.result(timeout=3).status == "FINISHED"
    finally:
        instance.close()
    raw = path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(row) for row in raw]
    assert rows[1] == {
        "event": "retired_worker",
        "end_ns": 20,
        "requests": [{"rid": rid(1)}],
    }
    assert rows[-1]["retired_forwards"] == 1
    assert rows[-1]["prefix_sha256"] == hashlib.sha256(b"".join(raw[:-1])).hexdigest()
