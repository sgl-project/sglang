"""CPU-only validation of the independent native token-commit ledger."""

import json

import pytest
from sglang_simulator.simulation.sglang.token_events import TokenEventRecorder


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_streamed_batches_preserve_exact_times_and_incremental_counts(tmp_path):
    path = tmp_path / "tokens.jsonl"
    recorder = TokenEventRecorder(path)
    assert read_events(path) == [
        {"event": "start", "version": 1, "clock": "virtual_ns"}
    ]
    at_ns = 2**53 + 1
    recorder.record(at_ns, iter([("left", 0, 2), ("right", 0, 3)]))
    recorder.record(at_ns, [("left", 2, 1), ("left", 3, 2)])
    recorder.record(at_ns + 1, [])
    events = read_events(path)
    assert len(events) == 3  # Batches are flushed; empty batches are omitted.
    assert events[1] == {
        "event": "output_commit",
        "at_ns": at_ns,
        "commits": [
            {"rid": "left", "output_index": 0, "token_count": 2},
            {"rid": "right", "output_index": 0, "token_count": 3},
        ],
    }
    recorder.finish("run", at_ns + 2)
    recorder.finish("run", at_ns + 2)
    assert read_events(path)[-1] == {
        "event": "finish",
        "run_id": "run",
        "at_ns": at_ns + 2,
        "status": "FINISHED",
        "total_output_tokens": 8,
    }
    assert len(read_events(path)) == 4
    with pytest.raises(RuntimeError, match="closed"):
        recorder.record(at_ns + 2, [("left", 5, 1)])
    with pytest.raises(RuntimeError, match="closed"):
        recorder.finish("different-run", at_ns + 2)


def test_partial_outputs_survive_close_without_success_footer(tmp_path):
    path = tmp_path / "interrupted.jsonl"
    recorder = TokenEventRecorder(path)
    recorder.record(10, [("eventually-cancelled", 0, 4)])
    recorder.close()
    recorder.close()
    events = read_events(path)
    assert [event["event"] for event in events] == ["start", "output_commit"]
    assert events[1]["commits"][0]["token_count"] == 4
    with pytest.raises(RuntimeError, match="closed"):
        recorder.finish("interrupted", 10)


def test_exclusive_creation_preserves_prior_ledger(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("historical evidence\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        TokenEventRecorder(path)
    assert path.read_text(encoding="utf-8") == "historical evidence\n"


@pytest.mark.parametrize("at_ns", [-1, 1.5, "1", True, None])
def test_invalid_times_do_not_write_or_consume_indices(tmp_path, at_ns):
    path = tmp_path / "invalid-time.jsonl"
    recorder = TokenEventRecorder(path)
    with pytest.raises(ValueError, match="at_ns"):
        recorder.record(at_ns, [("request", 0, 1)])
    recorder.record(0, [("request", 0, 1)])
    recorder.finish("run", 0)
    assert read_events(path)[-1]["total_output_tokens"] == 1


@pytest.mark.parametrize(
    "commit",
    [
        ("", 0, 1),
        (None, 0, 1),
        ("request", -1, 1),
        ("request", False, 1),
        ("request", 0.0, 1),
        ("request", 0, 0),
        ("request", 0, -1),
        ("request", 0, True),
        ("request", 0, 1.5),
    ],
)
def test_invalid_commit_rejected_before_any_batch_write(tmp_path, commit):
    path = tmp_path / "invalid-commit.jsonl"
    recorder = TokenEventRecorder(path)
    with pytest.raises(ValueError):
        recorder.record(10, [("valid", 0, 2), commit])
    assert len(read_events(path)) == 1
    recorder.record(9, [("valid", 0, 2)])
    recorder.finish("run", 9)
    assert read_events(path)[-1]["total_output_tokens"] == 2


@pytest.mark.parametrize("output_index", [0, 1, 3])
def test_duplicate_overlapping_and_skipped_indices_rejected(tmp_path, output_index):
    path = tmp_path / "noncontiguous.jsonl"
    recorder = TokenEventRecorder(path)
    recorder.record(10, [("request", 0, 2)])
    with pytest.raises(ValueError, match="non-contiguous"):
        recorder.record(20, [("other", 0, 1), ("request", output_index, 1)])
    recorder.record(20, [("other", 0, 1), ("request", 2, 1)])
    recorder.finish("run", 20)
    assert read_events(path)[-1]["total_output_tokens"] == 4


def test_new_requests_cannot_start_after_zero(tmp_path):
    recorder = TokenEventRecorder(tmp_path / "gap.jsonl")
    with pytest.raises(ValueError, match="non-contiguous"):
        recorder.record(0, [("request", 1, 1)])
    recorder.close()


def test_empty_batches_still_enforce_clock_and_finish_cannot_go_back(tmp_path):
    path = tmp_path / "clock.jsonl"
    recorder = TokenEventRecorder(path)
    recorder.record(20, [])
    with pytest.raises(ValueError, match="monotonic"):
        recorder.record(19, [("request", 0, 1)])
    with pytest.raises(ValueError, match="monotonic"):
        recorder.finish("run", 19)
    assert len(read_events(path)) == 1
    recorder.finish("run", 20)
    assert read_events(path)[-1]["total_output_tokens"] == 0


def test_failed_generator_does_not_partially_commit_a_batch(tmp_path):
    def interrupted_batch():
        yield "request", 0, 1
        raise RuntimeError("interrupted input")

    path = tmp_path / "generator.jsonl"
    recorder = TokenEventRecorder(path)
    with pytest.raises(RuntimeError, match="interrupted input"):
        recorder.record(10, interrupted_batch())
    assert len(read_events(path)) == 1
    recorder.record(0, [("request", 0, 1)])
    recorder.finish("run", 0)
    assert read_events(path)[-1]["total_output_tokens"] == 1


def test_io_failure_closes_without_success_footer(tmp_path):
    class FailingStream:
        closed = False

        def write(self, _line):
            raise OSError("disk write failed")

        def close(self):
            self.closed = True

    path = tmp_path / "io-failure.jsonl"
    recorder = TokenEventRecorder(path)
    recorder._stream.close()
    recorder._stream = FailingStream()
    with pytest.raises(OSError, match="disk write failed"):
        recorder.record(0, [("request", 0, 1)])
    with pytest.raises(RuntimeError, match="closed"):
        recorder.finish("run", 0)
    assert [event["event"] for event in read_events(path)] == ["start"]
