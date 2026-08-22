import json

import pytest

from sglang.srt.disaggregation import decode as disagg_decode
from sglang.srt.distributed.device_communicators.symm_mem_gather_telemetry import (
    SymmMemGatherTelemetry,
    common_generation_ids,
)
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import dp_attn
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


def test_same_process_entry_timing_is_attached_to_next_generation(tmp_path):
    recorder = SymmMemGatherTelemetry(world_size=2, group_rank=0, max_records=4)
    recorder.start(output_dir=str(tmp_path), profile_id="entry", dp_rank=0)
    entry_timing = {
        "scheduler_loop_entry_ns": 900,
        "adapter_entry_ns": 1_000,
        "prepare_raw_entry_ns": 1_020,
        "all_gather_call_entry_ns": 1_080,
    }
    recorder.set_pending_entry_timing(entry_timing)
    _finish_record(recorder, 1, start_ns=1_100)
    _finish_record(recorder, 2, start_ns=2_000)
    path = recorder.stop()
    assert path is not None
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 2
    assert payload["records"][0]["entry_timing"] == entry_timing
    assert "entry_timing" not in payload["records"][1]
    assert (
        entry_timing["scheduler_loop_entry_ns"]
        <= entry_timing["adapter_entry_ns"]
        <= entry_timing["prepare_raw_entry_ns"]
        <= entry_timing["all_gather_call_entry_ns"]
        <= payload["records"][0]["gather_start_ns"]
    )


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


def test_default_off_does_not_read_host_clock(monkeypatch):
    monkeypatch.delenv("SGLANG_SYMM_MEM_DP_SYNC_TELEMETRY", raising=False)

    def fail_if_called():
        raise AssertionError("default-off path sampled perf_counter_ns")

    monkeypatch.setattr(dp_attn.time, "perf_counter_ns", fail_if_called)
    assert dp_attn._symm_dp_adapter_entry_ns() is None


def test_enabled_entry_reads_host_clock(monkeypatch):
    monkeypatch.setenv("SGLANG_SYMM_MEM_DP_SYNC_TELEMETRY", "true")
    monkeypatch.setattr(dp_attn.time, "perf_counter_ns", lambda: 123_456)
    assert dp_attn._symm_dp_adapter_entry_ns() == 123_456


def test_scheduler_loop_entry_clock_is_default_off(monkeypatch):
    monkeypatch.delenv("SGLANG_SYMM_MEM_DP_SYNC_TELEMETRY", raising=False)

    def fail_if_called():
        raise AssertionError("default-off scheduler boundary sampled perf_counter_ns")

    monkeypatch.setattr(dp_attn.time, "perf_counter_ns", fail_if_called)
    assert dp_attn.symm_dp_scheduler_loop_entry_ns() is None


def test_scheduler_loop_entry_clock_is_enabled(monkeypatch):
    monkeypatch.setenv("SGLANG_SYMM_MEM_DP_SYNC_TELEMETRY", "true")
    monkeypatch.setattr(dp_attn.time, "perf_counter_ns", lambda: 654_321)
    assert dp_attn.symm_dp_scheduler_loop_entry_ns() == 654_321


@pytest.mark.parametrize(
    "event_loop_name",
    ("event_loop_normal_disagg_decode", "event_loop_overlap_disagg_decode"),
)
def test_disagg_decode_production_loop_samples_scheduler_entry(
    monkeypatch, event_loop_name
):
    class StopLoop(Exception):
        pass

    class RequestReceiver:
        def recv_requests(self):
            raise StopLoop

    class Queue:
        queue = []

        def prefetch_prefill_dp_rank_queries(self):
            pass

    class RunningBatch:
        def batch_size(self):
            return 0

    class FakeScheduler(disagg_decode.SchedulerDisaggregationDecodeMixin):
        request_receiver = RequestReceiver()
        disagg_decode_prealloc_queue = Queue()
        disagg_decode_transfer_queue = Queue()
        waiting_queue = []
        running_batch = RunningBatch()
        _engine_paused = False
        _symm_dp_scheduler_loop_entry_ns = None
        _symm_dp_scheduler_stage_timing = None

    monkeypatch.setattr(
        disagg_decode, "symm_dp_scheduler_loop_entry_ns", lambda: 765_432
    )
    scheduler = FakeScheduler()
    with pytest.raises(StopLoop):
        getattr(scheduler, event_loop_name)()
    assert scheduler._symm_dp_scheduler_loop_entry_ns == 765_432
    timing = scheduler._symm_dp_scheduler_stage_timing
    assert timing is not None
    assert timing["scheduler_loop_entry_ns"] == 765_432
    assert timing["after_top_prefetch_ns"] >= timing["scheduler_loop_entry_ns"]
    assert {
        key: timing[key]
        for key in (
            "prealloc_before",
            "transfer_before",
            "waiting_before",
            "running_before",
        )
    } == {
        key: 0
        for key in (
            "prealloc_before",
            "transfer_before",
            "waiting_before",
            "running_before",
        )
    }
