from sglang.srt.tracing.req_time_stats import ReqTimePoint, ReqTimeStatsBase

def test_req_time_stats_mark_and_duration_non_negative():
    s = ReqTimeStatsBase.create()
    s.mark(ReqTimePoint.received)
    s.mark(ReqTimePoint.request_sent_to_scheduler)
    assert s.duration_s(ReqTimePoint.received, ReqTimePoint.request_sent_to_scheduler) >= 0.0

def test_req_time_stats_mark_is_idempotent():
    s = ReqTimeStatsBase.create()
    s.mark(ReqTimePoint.received)
    t1 = s.ts_mono_s[int(ReqTimePoint.received)]
    s.mark(ReqTimePoint.received)
    t2 = s.ts_mono_s[int(ReqTimePoint.received)]
    assert t1 == t2
