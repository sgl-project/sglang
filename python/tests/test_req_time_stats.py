from sglang.srt.tracing.req_time_stats import ReqTimePoint, ReqTimeStatsBase


def test_req_time_stats_mark_and_duration_non_negative():
    s = ReqTimeStatsBase.create()
    s.mark(ReqTimePoint.received)
    s.mark(ReqTimePoint.request_sent_to_scheduler)
    assert (
        s.duration_s(ReqTimePoint.received, ReqTimePoint.request_sent_to_scheduler)
        >= 0.0
    )


def test_req_time_stats_mark_is_idempotent():
    s = ReqTimeStatsBase.create()
    s.mark(ReqTimePoint.received)
    t1 = s.ts_mono_s[int(ReqTimePoint.received)]
    s.mark(ReqTimePoint.received)
    t2 = s.ts_mono_s[int(ReqTimePoint.received)]
    assert t1 == t2


def test_req_time_stats_mark_at_is_idempotent():
    s = ReqTimeStatsBase.create()
    s.mark_at(ReqTimePoint.received, 123.0)
    s.mark_at(ReqTimePoint.received, 456.0)
    assert s.ts_mono_s[int(ReqTimePoint.received)] == 123.0


def test_req_time_stats_to_log_record_is_copy():
    s = ReqTimeStatsBase.create()
    s.mark_at(ReqTimePoint.received, 1.0)
    rec = s.to_log_record()
    rec["req_time_ts_mono_s"][int(ReqTimePoint.received)] = 999.0
    assert s.ts_mono_s[int(ReqTimePoint.received)] == 1.0
