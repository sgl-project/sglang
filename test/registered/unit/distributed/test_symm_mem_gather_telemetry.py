import json

from sglang.srt.distributed.device_communicators.symm_mem_gather_telemetry import (
    SymmMemGatherTelemetry,
    common_generation_ids,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _finish_record(recorder, generation, *, start_ns=100, ready=None):
    ready = ready or [1] * recorder.world_size
    record = recorder.begin(
        generation=generation,
        slot=generation % 2,
        gather_start_ns=start_ns,
        local_row=range(7),
    )
    assert record is not None
    recorder.note_poll(
        record,
        ready=ready,
        poll_begin_ns=start_ns + 10,
        sync_begin_ns=start_ns + 20,
        sync_done_ns=start_ns + 30,
    )
    recorder.finish(
        record,
        gather_done_ns=start_ns + 40,
        peer_rows=[range(7) for _ in range(recorder.world_size)],
    )


def test_ready_transitions_ties_and_host_timings(tmp_path):
    recorder = SymmMemGatherTelemetry(world_size=4, group_rank=1, max_records=8)
    recorder.start(output_dir=str(tmp_path), profile_id="unit", dp_rank=5)
    record = recorder.begin(
        generation=7,
        slot=1,
        gather_start_ns=1_000,
        local_row=[10, 11, 12, 13, 14, 15, 16],
    )
    assert record is not None

    assert (
        recorder.note_poll(
            record,
            ready=[1, 0, 0, 0],
            poll_begin_ns=1_010,
            sync_begin_ns=1_020,
            sync_done_ns=1_040,
        )
        == 0b0001
    )
    recorder.note_poll(
        record,
        ready=[1, 1, 1, 0],
        poll_begin_ns=1_050,
        sync_begin_ns=1_060,
        sync_done_ns=1_090,
    )
    recorder.note_poll(
        record,
        ready=[1, 1, 1, 0],
        poll_begin_ns=1_100,
        sync_begin_ns=1_110,
        sync_done_ns=1_140,
    )
    recorder.note_poll(
        record,
        ready=[1, 1, 1, 1],
        poll_begin_ns=1_150,
        sync_begin_ns=1_160,
        sync_done_ns=1_200,
    )
    recorder.finish(
        record,
        gather_done_ns=1_210,
        peer_rows=[[peer] * 7 for peer in range(4)],
    )
    path = recorder.stop()
    assert path is not None
    payload = json.loads(path.read_text())
    saved = payload["records"][0]
    assert [entry["mask"] for entry in saved["ready_mask_rle"]] == [1, 7, 15]
    assert [entry["polls"] for entry in saved["ready_mask_rle"]] == [1, 2, 1]
    assert saved["first_ready_poll"] == [0, 1, 1, 3]
    assert saved["slowest_peers"] == [3]
    assert saved["d2h_sync_wall_ns"] == 120
    assert saved["d2h_sync_wall_max_ns"] == 40
    assert saved["host_retry_gap_ns"] == 30
    assert saved["host_retry_gap_max_ns"] == 10
    assert saved["peer_rows"] == [[peer] * 7 for peer in range(4)]


def test_generation_wrap_regression_and_bounded_records(tmp_path):
    recorder = SymmMemGatherTelemetry(world_size=2, group_rank=0, max_records=2)
    recorder.start(output_dir=str(tmp_path), profile_id="wrap", dp_rank=0)
    for generation in (0xFFFFFFFF, 1, 1):
        _finish_record(recorder, generation)
    path = recorder.stop()
    assert path is not None
    payload = json.loads(path.read_text())
    assert payload["generation_regressions"] == 1
    assert [record["generation"] for record in payload["records"]] == [1, 1]


def test_tied_slowest_peers_and_common_generation_intersection(tmp_path):
    recorder = SymmMemGatherTelemetry(world_size=4, group_rank=0, max_records=4)
    recorder.start(output_dir=str(tmp_path), profile_id="tie", dp_rank=0)
    record = recorder.begin(
        generation=3,
        slot=0,
        gather_start_ns=0,
        local_row=range(7),
    )
    assert record is not None
    recorder.note_poll(
        record,
        ready=[1, 1, 0, 0],
        poll_begin_ns=1,
        sync_begin_ns=2,
        sync_done_ns=3,
    )
    recorder.note_poll(
        record,
        ready=[1, 1, 1, 1],
        poll_begin_ns=4,
        sync_begin_ns=5,
        sync_done_ns=6,
    )
    recorder.finish(record, gather_done_ns=7, peer_rows=[range(7)] * 4)
    path = recorder.stop()
    assert path is not None
    payload = json.loads(path.read_text())
    assert payload["records"][0]["slowest_peers"] == [2, 3]

    payloads = [
        {"records": [{"generation": value} for value in (3, 4, 5)]},
        {"records": [{"generation": value} for value in (2, 3, 4)]},
        {"records": [{"generation": value} for value in (3, 4, 6)]},
    ]
    assert common_generation_ids(payloads) == [3, 4]
    assert common_generation_ids([]) == []


def test_default_environment_does_not_enable_telemetry(monkeypatch):
    monkeypatch.delenv("SGLANG_SYMM_MEM_DP_SYNC_TELEMETRY", raising=False)
    monkeypatch.delenv("SGLANG_SYMM_MEM_DP_SYNC_TELEMETRY_MAX_RECORDS", raising=False)
    assert envs.SGLANG_SYMM_MEM_DP_SYNC_TELEMETRY.get() is False
    assert envs.SGLANG_SYMM_MEM_DP_SYNC_TELEMETRY_MAX_RECORDS.get() == 256
