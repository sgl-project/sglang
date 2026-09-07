import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler import Scheduler, _make_abort_req
from sglang.srt.utils.weight_versions import (
    WeightVersionEvent,
    WeightVersionSpan,
    add_weight_versions_to_meta_info,
    build_endpoint_weight_version_metadata,
    compute_weight_version_spans,
    record_weight_version_events,
    truncate_weight_version_events,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _ReqStub:
    def __init__(self, output_len: int):
        self.output_ids = [0] * output_len
        self.weight_version_events = []

    def record_weight_version_change(self, old_version):
        record_weight_version_events([self], old_version=old_version)

    def compute_weight_version_spans(self, current_version, num_output_tokens):
        return compute_weight_version_spans(
            self.weight_version_events,
            current_version=current_version,
            num_output_tokens=num_output_tokens,
        )


def _expected_spans(events, current_version, num_output_tokens):
    """Reference model: name the version that sampled each token, then run-length encode."""
    per_token = []
    for index in range(num_output_tokens):
        owner = next(
            (event.old_version for event in events if event.num_output_tokens > index),
            current_version,
        )
        per_token.append(owner)

    if not per_token:
        first_event_end_at_zero = next(
            (event for event in events if event.num_output_tokens >= 0), None
        )
        version = (
            first_event_end_at_zero.old_version
            if first_event_end_at_zero is not None
            else current_version
        )
        return [WeightVersionSpan(version=version, start=0, end=0)]

    spans = []
    for index, version in enumerate(per_token):
        if spans and spans[-1].version == version:
            spans[-1].end = index + 1
        else:
            spans.append(WeightVersionSpan(version=version, start=index, end=index + 1))
    return spans


class TestComputeWeightVersionSpans(CustomTestCase):
    def test_no_events_returns_single_span(self):
        """A request untouched by updates gets one span covering all output tokens."""
        self.assertEqual(
            _ReqStub(5).compute_weight_version_spans(
                current_version="v1", num_output_tokens=5
            ),
            [WeightVersionSpan(version="v1", start=0, end=5)],
        )

    def test_zero_output_tokens_returns_empty_span_span(self):
        """A request finishing with no output still reports the current version."""
        self.assertEqual(
            _ReqStub(0).compute_weight_version_spans(
                current_version="v1", num_output_tokens=0
            ),
            [WeightVersionSpan(version="v1", start=0, end=0)],
        )

    def test_one_update_splits_into_two_spans(self):
        """An update at 3 output tokens attributes [0,3) to the old version and the rest to the new."""
        req = _ReqStub(3)
        req.record_weight_version_change(old_version="v1")
        req.output_ids.extend([0] * 4)
        self.assertEqual(
            req.compute_weight_version_spans(current_version="v2", num_output_tokens=7),
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=7),
            ],
        )

    def test_two_updates_split_into_three_spans(self):
        """Each update the request lives through adds one span."""
        req = _ReqStub(2)
        req.record_weight_version_change(old_version="v1")
        req.output_ids.extend([0] * 3)
        req.record_weight_version_change(old_version="v2")
        req.output_ids.extend([0] * 1)
        self.assertEqual(
            req.compute_weight_version_spans(current_version="v3", num_output_tokens=6),
            [
                WeightVersionSpan(version="v1", start=0, end=2),
                WeightVersionSpan(version="v2", start=2, end=5),
                WeightVersionSpan(version="v3", start=5, end=6),
            ],
        )

    def test_update_before_first_token_records_no_event(self):
        """An update while the request has no output leaves the whole output on the new version."""
        req = _ReqStub(0)
        req.record_weight_version_change(old_version="v1")
        self.assertEqual(req.weight_version_events, [])
        req.output_ids.extend([0] * 4)
        self.assertEqual(
            req.compute_weight_version_spans(current_version="v2", num_output_tokens=4),
            [WeightVersionSpan(version="v2", start=0, end=4)],
        )

    def test_back_to_back_updates_skip_empty_span(self):
        """Two updates with no tokens in between produce no empty span."""
        req = _ReqStub(2)
        req.record_weight_version_change(old_version="v1")
        req.record_weight_version_change(old_version="v2")
        req.output_ids.extend([0] * 2)
        self.assertEqual(
            req.compute_weight_version_spans(current_version="v3", num_output_tokens=4),
            [
                WeightVersionSpan(version="v1", start=0, end=2),
                WeightVersionSpan(version="v3", start=2, end=4),
            ],
        )

    def test_update_at_final_length_yields_no_trailing_span(self):
        """An event recorded exactly at the final output length adds no empty trailing span."""
        req = _ReqStub(4)
        req.record_weight_version_change(old_version="v1")
        self.assertEqual(
            req.compute_weight_version_spans(current_version="v2", num_output_tokens=4),
            [WeightVersionSpan(version="v1", start=0, end=4)],
        )

    def test_event_beyond_reported_length_is_clamped(self):
        """A spec-decode overshoot event is clamped to the reported output length."""
        req = _ReqStub(6)
        req.record_weight_version_change(old_version="v1")
        self.assertEqual(
            req.compute_weight_version_spans(current_version="v2", num_output_tokens=4),
            [WeightVersionSpan(version="v1", start=0, end=4)],
        )

    def test_clamp_to_zero_reports_the_pre_update_version(self):
        """With no visible tokens the single span keeps the version that sampled them."""
        req = _ReqStub(3)
        req.record_weight_version_change(old_version="v1")
        self.assertEqual(
            req.compute_weight_version_spans(current_version="v2", num_output_tokens=0),
            [WeightVersionSpan(version="v1", start=0, end=0)],
        )

    def test_clamped_output_can_end_before_the_current_version(self):
        """When the visible tokens stop early the newest version never appears."""
        req = _ReqStub(3)
        req.record_weight_version_change(old_version="v1")
        req.output_ids.extend([0] * 2)
        req.record_weight_version_change(old_version="v2")
        self.assertEqual(
            req.compute_weight_version_spans(current_version="v3", num_output_tokens=4),
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=4),
            ],
        )

    def test_clamping_collapses_events_and_merges_into_one_span(self):
        """Events beyond the visible range collapse instead of leaving empty spans."""
        req = _ReqStub(3)
        req.record_weight_version_change(old_version="v1")
        req.output_ids.extend([0] * 2)
        req.record_weight_version_change(old_version="v2")
        self.assertEqual(
            req.compute_weight_version_spans(current_version="v1", num_output_tokens=2),
            [WeightVersionSpan(version="v1", start=0, end=2)],
        )

    def test_duplicate_event_at_the_same_length_is_idempotent(self):
        """Recording twice at the same token count matches recording once."""
        req = _ReqStub(3)
        req.record_weight_version_change(old_version="v1")
        req.record_weight_version_change(old_version="v1")
        req.output_ids.extend([0] * 2)
        self.assertEqual(
            req.compute_weight_version_spans(current_version="v2", num_output_tokens=5),
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=5),
            ],
        )

    def test_version_returning_after_tokens_keeps_separate_spans(self):
        """A version that comes back after another one sampled tokens gets its own span."""
        req = _ReqStub(2)
        req.record_weight_version_change(old_version="v1")
        req.output_ids.extend([0] * 2)
        req.record_weight_version_change(old_version="v2")
        req.output_ids.extend([0] * 2)
        self.assertEqual(
            req.compute_weight_version_spans(current_version="v1", num_output_tokens=6),
            [
                WeightVersionSpan(version="v1", start=0, end=2),
                WeightVersionSpan(version="v2", start=2, end=4),
                WeightVersionSpan(version="v1", start=4, end=6),
            ],
        )

    def test_spans_are_never_empty_even_when_everything_is_clamped_away(self):
        """add_weight_versions_to_meta_info reads the newest span by index, so an empty list would crash it."""
        req = _ReqStub(3)
        req.record_weight_version_change(old_version="v1")
        req.record_weight_version_change(old_version="v2")

        for current_version in ("v1", "v3"):
            self.assertTrue(
                req.compute_weight_version_spans(
                    current_version=current_version, num_output_tokens=0
                )
            )
        self.assertTrue(
            _ReqStub(0).compute_weight_version_spans(
                current_version="v1", num_output_tokens=0
            )
        )

    def test_spans_satisfy_the_contract_for_random_event_sequences(self):
        """Randomized event sequences always yield ordered, contiguous, non-duplicated spans."""
        rng = random.Random(0)
        versions = ["v0", "v1", "v2"]
        for _ in range(300):
            counts = sorted(rng.randint(1, 8) for _ in range(rng.randint(0, 5)))
            events = [
                WeightVersionEvent(
                    old_version=rng.choice(versions), num_output_tokens=count
                )
                for count in counts
            ]
            current_version = rng.choice(versions)
            num_output_tokens = rng.randint(0, 10)
            spans = compute_weight_version_spans(
                events,
                current_version=current_version,
                num_output_tokens=num_output_tokens,
            )

            self.assertEqual(
                spans,
                _expected_spans(events, current_version, num_output_tokens),
            )
            self.assertTrue(spans)
            self.assertEqual(spans[0].start, 0)
            self.assertEqual(spans[-1].end, num_output_tokens)
            for previous, current in zip(spans, spans[1:]):
                self.assertEqual(previous.end, current.start)
                self.assertNotEqual(previous.version, current.version)
            if num_output_tokens == 0:
                self.assertEqual(len(spans), 1)
                self.assertEqual(spans[0].end, 0)
            else:
                for span in spans:
                    self.assertLess(span.start, span.end)


class _ServingStub:
    def __init__(self, weight_version):
        self.weight_version = weight_version


class _ContextStub:
    def __init__(self, serving: _ServingStub):
        self.serving = serving

    def override(self, source, **fields):
        self.serving.weight_version = fields["weight_version"]


class _SchedulerStub:
    collect_inflight_reqs = Scheduler.collect_inflight_reqs

    def __init__(
        self,
        version,
        running,
        waiting,
        chunked=None,
        last_batch=None,
        pp_size=1,
        hisparse=None,
    ):
        self.serving = _ServingStub(version)
        self.ps = SimpleNamespace(pp_size=pp_size)
        self.running_batch = SimpleNamespace(reqs=running)
        self.last_batch = last_batch
        self.waiting_queue = waiting
        self.chunked_req = chunked
        self.hisparse_coordinator = hisparse


class TestSchedulerRecordWeightVersionChange(CustomTestCase):
    def _scheduler(self, *args, **kwargs):
        scheduler = _SchedulerStub(*args, **kwargs)
        for name, value in (
            ("get_serving", scheduler.serving),
            ("get_context", _ContextStub(scheduler.serving)),
        ):
            patcher = patch(f"sglang.srt.managers.scheduler.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return scheduler

    def test_records_on_running_waiting_and_chunked_requests(self):
        """A version change records an event on every live request holding output tokens."""
        running_req = _ReqStub(3)
        waiting_req = _ReqStub(5)
        chunked_req = _ReqStub(2)
        scheduler = self._scheduler(
            "v1", [running_req], [waiting_req], chunked=chunked_req
        )

        Scheduler.record_weight_version_change(scheduler, new_version="v2")

        self.assertEqual(scheduler.serving.weight_version, "v2")
        for req, output_len in (
            (running_req, 3),
            (waiting_req, 5),
            (chunked_req, 2),
        ):
            self.assertEqual(len(req.weight_version_events), 1)
            self.assertEqual(req.weight_version_events[0].old_version, "v1")
            self.assertEqual(req.weight_version_events[0].num_output_tokens, output_len)

    def test_records_on_last_batch_requests_exactly_once(self):
        """A request in both the running and last batch is recorded once, and last-batch-only requests are recorded too."""
        shared_req = _ReqStub(3)
        prefill_req = _ReqStub(1)
        scheduler = self._scheduler(
            "v1",
            [shared_req],
            [],
            last_batch=SimpleNamespace(reqs=[shared_req, prefill_req]),
        )

        Scheduler.record_weight_version_change(scheduler, new_version="v2")

        self.assertEqual(len(shared_req.weight_version_events), 1)
        self.assertEqual(len(prefill_req.weight_version_events), 1)

    def test_records_on_all_pp_microbatches(self):
        """With pipeline parallelism every microbatch is visited, not just the selected one."""
        mb0_req = _ReqStub(2)
        mb1_req = _ReqStub(4)
        pending_req = _ReqStub(6)
        scheduler = self._scheduler("v1", [], [], pp_size=2)
        scheduler.running_mbs = [
            SimpleNamespace(reqs=[mb0_req]),
            SimpleNamespace(reqs=[mb1_req]),
        ]
        scheduler.mbs = [None, SimpleNamespace(reqs=[pending_req])]

        Scheduler.record_weight_version_change(scheduler, new_version="v2")

        for req in (mb0_req, mb1_req, pending_req):
            self.assertEqual(len(req.weight_version_events), 1)

    def test_records_on_hisparse_staging_requests(self):
        """Requests parked in the HiSparse staging queue are swept like any live request."""
        staging_req = _ReqStub(2)
        coordinator = SimpleNamespace(
            ack_staging_queue=[SimpleNamespace(req=staging_req)]
        )
        scheduler = self._scheduler("v1", [], [], hisparse=coordinator)

        Scheduler.record_weight_version_change(scheduler, new_version="v2")

        self.assertEqual(len(staging_req.weight_version_events), 1)
        self.assertEqual(staging_req.weight_version_events[0].old_version, "v1")
        self.assertEqual(staging_req.weight_version_events[0].num_output_tokens, 2)

    def test_same_version_is_a_noop(self):
        """Re-announcing the current version must not record events."""
        req = _ReqStub(3)
        scheduler = self._scheduler("v1", [req], [])

        Scheduler.record_weight_version_change(scheduler, new_version="v1")

        self.assertEqual(scheduler.serving.weight_version, "v1")
        self.assertEqual(req.weight_version_events, [])

    def test_none_version_is_a_noop(self):
        """An update without a weight version must not disturb attribution."""
        req = _ReqStub(3)
        scheduler = self._scheduler("v1", [req], [])

        Scheduler.record_weight_version_change(scheduler, new_version=None)

        self.assertEqual(scheduler.serving.weight_version, "v1")
        self.assertEqual(req.weight_version_events, [])


class TestRecordWeightVersionEvents(CustomTestCase):
    def test_records_only_for_requests_that_have_output(self):
        """Requests without output tokens are skipped without stopping the sweep."""
        empty_req = _ReqStub(0)
        started_req = _ReqStub(4)

        num_recorded = record_weight_version_events(
            [empty_req, started_req, _ReqStub(2)], old_version="v1"
        )

        self.assertEqual(num_recorded, 2)
        self.assertEqual(empty_req.weight_version_events, [])
        self.assertEqual(len(started_req.weight_version_events), 1)
        self.assertEqual(started_req.weight_version_events[0].old_version, "v1")
        self.assertEqual(started_req.weight_version_events[0].num_output_tokens, 4)


class TestTruncateWeightVersionEvents(CustomTestCase):
    def test_events_within_the_kept_prefix_are_preserved(self):
        """Events entirely inside the streamed prefix survive a retract untouched."""
        events = [WeightVersionEvent(old_version="v1", num_output_tokens=3)]
        self.assertEqual(
            truncate_weight_version_events(events, num_kept_tokens=5), events
        )

    def test_events_beyond_the_kept_prefix_are_clamped(self):
        """An event past the streamed prefix is clamped so re-generated tokens forget it."""
        events = [
            WeightVersionEvent(old_version="v1", num_output_tokens=3),
            WeightVersionEvent(old_version="v2", num_output_tokens=9),
        ]
        self.assertEqual(
            truncate_weight_version_events(events, num_kept_tokens=5),
            [
                WeightVersionEvent(old_version="v1", num_output_tokens=3),
                WeightVersionEvent(old_version="v2", num_output_tokens=5),
            ],
        )

    def test_zero_kept_tokens_drops_all_events(self):
        """With nothing streamed yet the whole history is discarded, as before the fix."""
        events = [WeightVersionEvent(old_version="v1", num_output_tokens=3)]
        self.assertEqual(truncate_weight_version_events(events, num_kept_tokens=0), [])


class TestSpanlessAbortPaths(CustomTestCase):
    def test_priority_disabled_rejection_attaches_spans(self):
        """The pre-scheduler priority rejection reports spans like every other abort."""
        req = _ReqStub(0)
        req.rid = "r0"
        req.priority = 5
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda abort_info: None)
        )
        sent = []
        scheduler = SimpleNamespace(
            enable_priority_scheduling=False,
            abort_on_priority_when_disabled=True,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(
                    send_output=lambda obj, req_arg: sent.append(obj)
                )
            ),
        )

        with patch(
            "sglang.srt.managers.scheduler.get_serving",
            return_value=SimpleNamespace(weight_version="v1"),
        ):
            accepted = Scheduler._set_or_validate_priority(scheduler, req)

        self.assertFalse(accepted)
        self.assertEqual(
            sent[0].weight_versions, [WeightVersionSpan(version="v1", start=0, end=0)]
        )


class TestMakeAbortReq(CustomTestCase):
    def test_abort_req_carries_spans_clamped_to_sampled_tokens(self):
        """An aborted request reports the versions that sampled the tokens it produced."""
        req = _ReqStub(4)
        req.rid = "r0"
        req.weight_version_events.append(
            WeightVersionEvent(old_version="v1", num_output_tokens=3)
        )
        with patch(
            "sglang.srt.managers.scheduler.get_serving",
            return_value=SimpleNamespace(weight_version="v2"),
        ):
            abort_req = _make_abort_req(req)

        self.assertIsInstance(abort_req, AbortReq)
        self.assertEqual(abort_req.rid, "r0")
        self.assertEqual(
            abort_req.weight_versions,
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=4),
            ],
        )


class TestBuildEndpointWeightVersionMetadata(CustomTestCase):
    def test_metadata_projects_only_the_weight_fields(self):
        """Endpoint metadata exposes the version fields and nothing else from meta_info."""
        spans = [
            {"version": "v1", "start": 0, "end": 3},
            {"version": "v2", "start": 3, "end": 7},
        ]
        metadata = build_endpoint_weight_version_metadata(
            {
                "weight_version": "v2",
                "weight_versions": spans,
                "id": "r0",
                "completion_tokens": 7,
            }
        )
        self.assertEqual(metadata, {"weight_version": "v2", "weight_versions": spans})

    def test_metadata_omits_spans_when_absent(self):
        """Responses without spans still report the legacy version alone."""
        metadata = build_endpoint_weight_version_metadata({"weight_version": "v2"})
        self.assertEqual(metadata, {"weight_version": "v2"})


class TestAddWeightVersionsToMetaInfo(CustomTestCase):
    def test_meta_info_gets_dict_spans_and_legacy_field(self):
        """Spans become dicts and the legacy weight_version reports the newest span."""
        meta_info = {"weight_version": "stale"}
        add_weight_versions_to_meta_info(
            meta_info,
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=7),
            ],
            num_output_tokens=7,
        )
        self.assertEqual(
            meta_info["weight_versions"],
            [
                {"version": "v1", "start": 0, "end": 3},
                {"version": "v2", "start": 3, "end": 7},
            ],
        )
        self.assertEqual(meta_info["weight_version"], "v2")

    def test_spans_are_clamped_to_returned_tokens(self):
        """Aborts returning fewer tokens than sampled clamp spans to the visible range."""
        meta_info = {}
        add_weight_versions_to_meta_info(
            meta_info,
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=7),
            ],
            num_output_tokens=5,
        )
        self.assertEqual(
            meta_info["weight_versions"],
            [
                {"version": "v1", "start": 0, "end": 3},
                {"version": "v2", "start": 3, "end": 5},
            ],
        )
        self.assertEqual(meta_info["weight_version"], "v2")

    def test_clamp_never_extends_spans(self):
        """A response longer than the sampled range leaves the spans untouched."""
        meta_info = {}
        add_weight_versions_to_meta_info(
            meta_info,
            [WeightVersionSpan(version="v1", start=0, end=3)],
            num_output_tokens=10,
        )
        self.assertEqual(
            meta_info["weight_versions"], [{"version": "v1", "start": 0, "end": 3}]
        )
        self.assertEqual(meta_info["weight_version"], "v1")

    def test_clamp_drops_trailing_spans_and_rewrites_the_legacy_field(self):
        """Dropping invisible spans also moves the legacy version back."""
        meta_info = {}
        add_weight_versions_to_meta_info(
            meta_info,
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=7),
            ],
            num_output_tokens=3,
        )
        self.assertEqual(
            meta_info["weight_versions"], [{"version": "v1", "start": 0, "end": 3}]
        )
        self.assertEqual(meta_info["weight_version"], "v1")

    def test_clamp_to_zero_tokens_keeps_a_degenerate_span(self):
        """A response with no visible tokens still reports a well-formed empty span."""
        meta_info = {}
        add_weight_versions_to_meta_info(
            meta_info,
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=7),
            ],
            num_output_tokens=0,
        )
        self.assertEqual(
            meta_info["weight_versions"], [{"version": "v1", "start": 0, "end": 0}]
        )
        self.assertEqual(meta_info["weight_version"], "v1")


if __name__ == "__main__":
    unittest.main()
