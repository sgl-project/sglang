"""Unit tests for ``sglang.srt.distributed.nccl_ras`` — no server, no GPU.

Covers the parser (golden + degraded JSON), the socket client's ``OK\\n`` ack
handling, and the detector state machine with the priority ordering and
edge cases verified on a real NCCL 2.30.7 job (see
``.omc/specs/deep-dive-nccl-ras-gpu-verification.md``): all-zero sample drop,
per-rank error states, poll-failure, the ``missing_ranks[]`` dead path with
``considered_dead`` False→True transitions, symmetric vs subgroup hang, the
``is_initializing`` grace window, staleness fail-closed, and RECOVERED reset.
"""

import copy
import json
import os
import socket
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.distributed.nccl_ras import (
    KNOWN_COLLECTIVES,
    RasDetector,
    RasSocketClient,
    RasState,
    _RAS_TIMESTAMP_FMT,
    parse_status,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
_HEALTHY = os.path.join(_FIXTURES, "nccl_ras_status.json")
_DEAD = os.path.join(_FIXTURES, "nccl_ras_status_dead.json")

# A wall-clock ``now`` whose local time matches the healthy fixture timestamp
# ("2026-07-19 11:01:47") so collection_age is ~0 for fresh samples. We pin
# via strptime/mktime round-trip rather than a hard epoch to stay TZ-agnostic.
import time as _time

_FRESH_NOW = _time.mktime(_time.strptime("2026-07-19 11:01:47", _RAS_TIMESTAMP_FMT))


def _load(path):
    with open(path) as f:
        return json.load(f)


def _healthy():
    return copy.deepcopy(_load(_HEALTHY))


def _dead():
    return copy.deepcopy(_load(_DEAD))


def _set_counts(raw, rank_idx, **counts):
    raw["communicators"][0]["ranks"][rank_idx]["collective_counts"].update(counts)


class TestParseStatus(CustomTestCase):
    """Parser robustness: golden fixture, missing fields, half-payload, ack."""

    def test_parse_healthy_golden_fixture(self):
        st = parse_status(_healthy())
        self.assertEqual(st.communicators_count, 1)
        comm = st.communicators[0]
        self.assertEqual(comm.size, 2)
        self.assertEqual(comm.ranks_count, 2)
        self.assertEqual(comm.missing_ranks_count, 0)
        self.assertEqual(comm.missing_ranks, [])
        # comm identity key is hash:secondary_hash
        self.assertEqual(
            comm.comm_key, f"{comm.hash}:{comm.secondary_hash}"
        )
        # healthy live ranks OMIT unresponsive/considered_dead (absent, not False)
        for r in comm.ranks:
            self.assertFalse(r.unresponsive)
            self.assertFalse(r.considered_dead)
            self.assertEqual(r.async_error, 0)
            self.assertFalse(r.abort_flag)
        # All 5 collective keys present (non-sparse), unused=0.
        self.assertEqual(
            set(comm.ranks[0].collective_counts.keys()), set(KNOWN_COLLECTIVES)
        )
        self.assertIsNotNone(st.timestamp_epoch)

    def test_parse_dead_fixture_missing_ranks_schema(self):
        """Dead rank lives in missing_ranks[], status in missing_ranks[].status."""
        st = parse_status(_dead())
        comm = st.communicators[0]
        self.assertEqual(comm.ranks_count, 1)
        self.assertEqual(comm.missing_ranks_count, 1)
        self.assertEqual(len(comm.missing_ranks), 1)
        miss = comm.missing_ranks[0]
        self.assertEqual(miss.rank, 1)
        self.assertTrue(miss.unresponsive)
        self.assertFalse(miss.considered_dead)
        # live ranks[].status has NO unresponsive/considered_dead fields
        for r in comm.ranks:
            self.assertFalse(r.unresponsive)
            self.assertFalse(r.considered_dead)

    def test_parse_tolerates_missing_ras_and_empty_ranks(self):
        raw = {
            "nccl_version": "2.30.7",
            "timestamp": "2026-07-19 11:01:47",
            "communicators": [
                {
                    "hash": "0x1",
                    "secondary_hash": "0x2:0x3",
                    "size": 2,
                    "ranks": [],
                    "missing_ranks": [],
                }
            ],
        }
        st = parse_status(raw)
        self.assertEqual(st.communicators[0].ranks_count, 0)
        # no ras{} block → no crash; defaults stand.
        self.assertEqual(st.communicators[0].ranks, [])

    def test_parse_unparseable_timestamp_sets_flag(self):
        raw = _healthy()
        raw["timestamp"] = "not-a-date"
        st = parse_status(raw)
        self.assertIsNone(st.timestamp_epoch)

    def test_parse_missing_collective_counts(self):
        raw = _healthy()
        del raw["communicators"][0]["ranks"][0]["collective_counts"]
        st = parse_status(raw)
        self.assertEqual(st.communicators[0].ranks[0].collective_counts, {})

    def test_socket_client_skips_ok_ack_prefix(self):
        """``SET FORMAT json`` returns ``OK\\n`` before the JSON — must skip to {."""
        payload = b"OK\n" + json.dumps(_healthy()).encode()
        st = RasSocketClient._parse_payload(payload)
        self.assertIsNotNone(st)
        self.assertEqual(st.communicators_count, 1)

    def test_socket_client_text_only_returns_none(self):
        # < 2.28.7: STATUS returns plain text, no JSON.
        self.assertIsNone(RasSocketClient._parse_payload(b"OK\nsome text log\n"))

    def test_socket_client_empty_returns_none(self):
        self.assertIsNone(RasSocketClient._parse_payload(b""))

    def test_socket_client_malformed_json_returns_none(self):
        self.assertIsNone(RasSocketClient._parse_payload(b"OK\n{not json"))

    def test_socket_client_connection_refused_returns_none(self):
        client = RasSocketClient(addr="127.0.0.1", port=1, connect_timeout=0.1)
        self.assertIsNone(client.poll_status())

    def test_socket_client_poll_round_trip_via_mock(self):
        client = RasSocketClient(addr="127.0.0.1", port=28028)
        fake_sock = MagicMock()
        payload = b"OK\n" + json.dumps(_healthy()).encode()
        fake_sock.recv.side_effect = [payload, b""]
        fake_sock.__enter__ = lambda self: fake_sock
        fake_sock.__exit__ = lambda self, *a: None
        with patch("socket.create_connection", return_value=fake_sock):
            st = client.poll_status()
        self.assertIsNotNone(st)
        self.assertEqual(st.communicators_count, 1)

    def test_from_endpoint_parsing(self):
        c = RasSocketClient.from_endpoint("10.0.0.1:28029")
        self.assertEqual(c.addr, "10.0.0.1")
        self.assertEqual(c.port, 28029)
        c2 = RasSocketClient.from_endpoint("bad-no-port")
        self.assertEqual(c2.addr, "bad-no-port")
        self.assertEqual(c2.port, 28028)


class TestDetectorPriority(CustomTestCase):
    """One-shot detector.evaluate() per priority signal."""

    def _det(self):
        return RasDetector(stuck_polls=3)

    def test_healthy_fixture_is_healthy(self):
        f = self._det().evaluate(parse_status(_healthy()), _FRESH_NOW)
        self.assertEqual(f.worst_state, RasState.HEALTHY)
        self.assertTrue(f.poll_success)
        self.assertEqual(f.total_ranks_in_error, 0)
        self.assertEqual(f.total_missing, 0)
        # fresh timestamp → non-negative, bounded age
        self.assertIsNotNone(f.last_collection_age_sec)
        self.assertGreaterEqual(f.last_collection_age_sec, -1.0)

    def test_poll_none_poll_success_zero(self):
        f = self._det().evaluate(None, _FRESH_NOW)
        self.assertFalse(f.poll_success)
        self.assertEqual(f.worst_state, RasState.UNKNOWN)
        self.assertIsNone(f.last_collection_age_sec)

    def test_per_rank_abort_flag_immediate_error(self):
        raw = _healthy()
        raw["communicators"][0]["ranks"][0]["status"]["abort_flag"] = True
        f = self._det().evaluate(parse_status(raw), _FRESH_NOW)
        self.assertEqual(f.worst_state, RasState.ERROR)
        self.assertEqual(f.total_ranks_in_error, 1)

    def test_per_rank_async_error_immediate_error(self):
        raw = _healthy()
        raw["communicators"][0]["ranks"][1]["status"]["async_error"] = 7
        f = self._det().evaluate(parse_status(raw), _FRESH_NOW)
        self.assertEqual(f.worst_state, RasState.ERROR)
        self.assertEqual(f.total_ranks_in_error, 1)

    def test_per_rank_finalize_and_destroy_flags(self):
        for flag in ("finalize_called", "destroy_flag"):
            with self.subTest(flag=flag):
                raw = _healthy()
                raw["communicators"][0]["ranks"][0]["status"][flag] = True
                f = self._det().evaluate(parse_status(raw), _FRESH_NOW)
                self.assertEqual(f.worst_state, RasState.ERROR)

    def test_missing_ranks_immediate_unresponsive_suspect(self):
        # dead fixture: missing_ranks_count>0, unresponsive=true, considered_dead=false
        d = self._det()
        f = d.evaluate(parse_status(_dead()), _FRESH_NOW + 60)
        self.assertEqual(f.worst_state, RasState.SUSPECT)
        self.assertEqual(f.total_missing, 1)
        self.assertEqual(f.total_unresponsive, 1)
        self.assertEqual(f.total_dead, 0)  # not yet confirmed dead
        # No transition counted yet (considered_dead still False).
        ck = next(iter(f.per_comm))
        self.assertEqual(f.per_comm[ck].peer_dead_transitions, 0)

    def test_considered_dead_false_to_true_counts_transition(self):
        d = self._det()
        # Stage 1: missing + unresponsive, considered_dead=False (from fixture)
        f1 = d.evaluate(parse_status(_dead()), _FRESH_NOW + 1)
        self.assertEqual(f1.total_dead, 0)
        ck = next(iter(f1.per_comm))
        self.assertEqual(f1.per_comm[ck].peer_dead_transitions, 0)
        # Stage 2: RAS confirms death ~57s later.
        raw = _dead()
        raw["communicators"][0]["missing_ranks"][0]["status"]["considered_dead"] = True
        f2 = d.evaluate(parse_status(raw), _FRESH_NOW + 60)
        self.assertEqual(f2.worst_state, RasState.DEAD)
        self.assertEqual(f2.total_dead, 1)
        self.assertEqual(f2.per_comm[ck].peer_dead_transitions, 1)
        # Stage 3: re-eval same dead snapshot → no new transition.
        f3 = d.evaluate(parse_status(raw), _FRESH_NOW + 70)
        self.assertEqual(f3.per_comm[ck].peer_dead_transitions, 0)


class TestDetectorStuckAdvisory(CustomTestCase):
    """collective_counts divergence is advisory: 0-dropout + symmetric-vs-subgroup."""

    @staticmethod
    def _frozen_pair(leader, lagger):
        raw = _healthy()
        _set_counts(raw, 0, AllReduce=leader)
        _set_counts(raw, 1, AllReduce=lagger)
        return parse_status(raw)

    def test_all_zero_sample_does_not_false_stuck(self):
        """GPU head-line pitfall: ~1/3–1/2 of polls return all-zero counts."""
        d = RasDetector(stuck_polls=3)
        # seed a non-zero sample first
        raw = _healthy()
        _set_counts(raw, 0, AllReduce=100)
        _set_counts(raw, 1, AllReduce=100)
        d.evaluate(parse_status(raw), _FRESH_NOW)
        # now an all-zero sample (cached "no fresh counts")
        raw0 = _healthy()
        for r in raw0["communicators"][0]["ranks"]:
            for k in r["collective_counts"]:
                r["collective_counts"][k] = 0
        f = d.evaluate(parse_status(raw0), _FRESH_NOW + 1)
        self.assertEqual(f.worst_state, RasState.HEALTHY)
        key = next(iter(f.per_comm))
        self.assertFalse(f.per_comm[key].stuck)
        self.assertEqual(f.per_comm[key].divergence, {})

    def test_symmetric_hang_not_stuck(self):
        """Both ranks frozen at ~equal counts, nobody advances → not stuck.

        This is the watchdog's job; RAS collective_counts divergence does not
        diverge for a whole-collective hang.
        """
        d = RasDetector(stuck_polls=3)
        states = []
        for i in range(6):
            f = d.evaluate(self._frozen_pair(696, 695), _FRESH_NOW + i)
            states.append(f.worst_state)
        # SUSPECT (spread=1 present) but never STUCK
        self.assertNotIn(RasState.STUCK, states)
        self.assertEqual(f.worst_state, RasState.SUSPECT)

    def test_subgroup_lag_reports_stuck(self):
        """Leader advances each poll while lagger stalls → STUCK after stuck_polls."""
        d = RasDetector(stuck_polls=3)
        leaders = [100, 200, 300, 400, 500]
        stuck_flags = []
        for i, leader in enumerate(leaders):
            f = d.evaluate(self._frozen_pair(leader, 10), _FRESH_NOW + i)
            stuck_flags.append(f.per_comm[next(iter(f.per_comm))].stuck)
        # first two polls: freezing, not yet stuck; from poll index 2 (3rd
        # consecutive frozen non-zero sample with a moving leader): stuck.
        self.assertFalse(stuck_flags[1])
        self.assertTrue(stuck_flags[-1])
        self.assertEqual(f.worst_state, RasState.STUCK)

    def test_subgroup_lag_recovers_when_lagger_catches_up(self):
        d = RasDetector(stuck_polls=3)
        for leader in [100, 200, 300]:
            f = d.evaluate(self._frozen_pair(leader, 10), _FRESH_NOW)
        self.assertEqual(f.worst_state, RasState.STUCK)
        # lagger catches up → spread collapses, freeze pops on advance
        f = d.evaluate(self._frozen_pair(400, 400), _FRESH_NOW + 1)
        self.assertEqual(f.worst_state, RasState.HEALTHY)
        key = next(iter(f.per_comm))
        self.assertFalse(f.per_comm[key].stuck)
        self.assertEqual(f.per_comm[key].divergence, {})

    def test_is_initializing_grace_skips_stuck(self):
        d = RasDetector(stuck_polls=3)
        for i, leader in enumerate([100, 200, 300, 400, 500]):
            f = d.evaluate(
                self._frozen_pair(leader, 10),
                _FRESH_NOW + i,
                is_initializing=True,
            )
        self.assertEqual(f.worst_state, RasState.HEALTHY)
        key = next(iter(f.per_comm))
        self.assertFalse(f.per_comm[key].stuck)


class TestDetectorStaleness(CustomTestCase):
    """BLOCKER 1: a well-formed but stale/unparseable snapshot fails closed."""

    def test_unparseable_timestamp_unknown(self):
        raw = _healthy()
        raw["timestamp"] = "garbage"
        f = RasDetector(stuck_polls=3).evaluate(parse_status(raw), _FRESH_NOW)
        self.assertEqual(f.worst_state, RasState.UNKNOWN)
        self.assertIsNone(f.last_collection_age_sec)

    def test_missing_timestamp_unknown(self):
        raw = _healthy()
        raw.pop("timestamp")
        f = RasDetector(stuck_polls=3).evaluate(parse_status(raw), _FRESH_NOW)
        self.assertEqual(f.worst_state, RasState.UNKNOWN)
        self.assertIsNone(f.last_collection_age_sec)

    def test_stale_timestamp_unknown(self):
        raw = _healthy()
        # 1 hour old → beyond the staleness threshold
        raw["timestamp"] = "2026-07-19 10:01:47"
        f = RasDetector(stuck_polls=3).evaluate(parse_status(raw), _FRESH_NOW)
        self.assertEqual(f.worst_state, RasState.UNKNOWN)
        self.assertIsNone(f.last_collection_age_sec)


class TestRasMetrics(CustomTestCase):
    """Guard the RAS Prometheus series + PR1 label contract (no rank/comm_name)."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        cls._tmp = tempfile.mkdtemp()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = cls._tmp

    def test_log_nccl_ras_emits_series_with_bounded_labels(self):
        """A findings snapshot surfaces as the expected gauges/counters and the
        labels stay bounded to ``comm_hash`` (+collective/state) — PR1 must not
        emit ``rank`` (high cardinality) or ``comm_name`` (PR2, to avoid series
        churn here)."""
        from prometheus_client import Counter, Gauge, generate_latest

        from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
        from sglang.srt.distributed.nccl_ras import (
            RasCommFinding,
            RasFindings,
            RasState,
        )

        labels = {"model_name": "m", "engine_type": "e"}
        # Bypass the heavy __init__ (server_args/torch); exercise only the RAS
        # metric wiring with REAL prometheus objects so labelnames are validated.
        c = SchedulerMetricsCollector.__new__(SchedulerMetricsCollector)
        c.labels = labels
        comm = list(labels.keys()) + ["comm_hash"]
        ctype = comm + ["collective"]
        cstate = comm + ["state"]
        c.nccl_ras_missing_ranks = Gauge("u_missing", "", comm, multiprocess_mode="mostrecent")
        c.nccl_ras_unresponsive_ranks = Gauge("u_unresp", "", comm, multiprocess_mode="mostrecent")
        c.nccl_ras_dead_ranks = Gauge("u_dead", "", comm, multiprocess_mode="mostrecent")
        c.nccl_ras_ranks_in_error = Gauge("u_err", "", comm, multiprocess_mode="mostrecent")
        c.nccl_ras_collective_divergence = Gauge("u_div", "", ctype, multiprocess_mode="mostrecent")
        c.nccl_ras_stuck = Gauge("u_stuck", "", comm, multiprocess_mode="mostrecent")
        c.nccl_ras_communicator_state = Gauge("u_state", "", cstate, multiprocess_mode="mostrecent")
        c.nccl_ras_peer_dead_transitions_total = Counter("u_pdt", "", comm)
        c.nccl_ras_poll_success = Gauge("u_ps", "", list(labels.keys()), multiprocess_mode="mostrecent")
        c.nccl_ras_poll_errors_total = Counter("u_pe", "", list(labels.keys()))
        c.nccl_ras_last_collection_age_sec = Gauge("u_age", "", list(labels.keys()), multiprocess_mode="mostrecent")
        c._ras_state_values = ["HEALTHY", "UNKNOWN", "SUSPECT", "STUCK", "ERROR", "DEAD"]

        findings = RasFindings(
            per_comm={
                "0x1:0x2": RasCommFinding(
                    comm_key="0x1:0x2",
                    state=RasState.ERROR,
                    missing=0,
                    unresponsive=0,
                    dead=0,
                    ranks_in_error=2,
                    divergence={"AllReduce": 5},
                    stuck=False,
                    peer_dead_transitions=2,
                )
            },
            total_missing=0,
            total_unresponsive=0,
            total_dead=0,
            total_ranks_in_error=2,
            divergence_comms=["0x1:0x2"],
            worst_state=RasState.ERROR,
            poll_success=True,
            last_collection_age_sec=3.5,
        )
        c.log_nccl_ras(findings)

        out = generate_latest().decode()
        # Expected series all present.
        for name in (
            "u_ps",
            "u_err",
            "u_state",
            "u_div",
            "u_age",
            "u_pdt",
        ):
            self.assertIn(name, out, f"missing metric {name}")
        # Bounded label contract.
        self.assertIn('comm_hash="0x1:0x2"', out)
        self.assertIn('collective="AllReduce"', out)
        self.assertIn('state="ERROR"', out)
        self.assertNotIn("rank=", out, "PR1 must not emit a rank label")
        self.assertNotIn("comm_name=", out, "PR1 defers comm_name to PR2")

    def test_poll_failure_increments_errors_counter(self):
        from prometheus_client import Counter, Gauge, generate_latest

        from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
        from sglang.srt.distributed.nccl_ras import RasFindings, RasState

        labels = {"model_name": "m", "engine_type": "e"}
        c = SchedulerMetricsCollector.__new__(SchedulerMetricsCollector)
        c.labels = labels
        c.nccl_ras_poll_success = Gauge(
            "v_ps", "", list(labels.keys()), multiprocess_mode="mostrecent"
        )
        c.nccl_ras_poll_errors_total = Counter("v_pe", "", list(labels.keys()))
        # No per_comm / other series referenced by the poll-failure branch.
        findings = RasFindings(
            worst_state=RasState.UNKNOWN,
            poll_success=False,
            last_collection_age_sec=None,
        )
        c.log_nccl_ras(findings)
        out = generate_latest().decode()
        self.assertIn("v_ps", out)
        self.assertIn("v_pe", out)


class TestRasCollectorElection(CustomTestCase):
    """Guard the headline correctness bug: election is by WORLD rank 0, not a
    dp/tp/pp all-zero predicate (which silently elects zero collectors under
    plain TP/PP where dp_rank is None, and may elect several under
    DP-attention). The gate must start a monitor only on the world-rank-0
    process and must not crash when --enable-metrics is off (log-only
    fallback)."""

    @classmethod
    def setUpClass(cls):
        # Importing Scheduler pulls the full GPU-adjacent stack (torch,
        # sglang.kernels, ...). Skip cleanly where that stack is absent so the
        # pure-Python parser/detector/metrics cases above still run on a
        # stripped CPU host; CI's base-a-test-cpu runner has the full repo.
        try:
            from sglang.srt.managers.scheduler import Scheduler

            cls._Scheduler = Scheduler
        except Exception as e:  # pragma: no cover - env-dependent
            raise unittest.SkipTest(f"Scheduler import unavailable: {e}")
        if not hasattr(Scheduler, "init_nccl_ras_collector"):
            raise unittest.SkipTest(
                "Scheduler.init_nccl_ras_collector absent (older sglang); "
                "RAS election tests require the current repo."
            )

    @classmethod
    def _bare_scheduler(cls):
        # init_nccl_ras_collector reads only self.metrics_collector and
        # self.is_initializing (plus envs + get_world_group), so a bare __new__
        # instance with those set is enough to exercise the gate without
        # spinning the full Scheduler.__init__.
        s = cls._Scheduler.__new__(cls._Scheduler)
        s.is_initializing = True
        s.metrics_collector = None
        return s

    def test_disabled_env_no_monitor(self):
        s = self._bare_scheduler()
        with patch("sglang.srt.environ.envs.SGLANG_NCCL_RAS_ENABLE") as enabled:
            enabled.get.return_value = False
            s.init_nccl_ras_collector()
        self.assertIsNone(s.ras_monitor)

    def test_elected_world_rank_zero_starts_monitor_with_metrics(self):
        s = self._bare_scheduler()
        s.metrics_collector = object()  # non-None → metrics path
        with patch("sglang.srt.environ.envs.SGLANG_NCCL_RAS_ENABLE") as enabled, \
             patch("sglang.srt.managers.scheduler.get_world_group") as gw, \
             patch("sglang.srt.distributed.nccl_ras.RasMonitor") as Monitor:
            enabled.get.return_value = True
            gw.return_value.rank_in_group = 0
            inst = MagicMock()
            Monitor.return_value = inst
            s.init_nccl_ras_collector()
        self.assertIs(s.ras_monitor, inst)
        inst.start.assert_called_once()

    def test_non_elected_no_monitor(self):
        s = self._bare_scheduler()
        s.metrics_collector = object()
        with patch("sglang.srt.environ.envs.SGLANG_NCCL_RAS_ENABLE") as enabled, \
             patch("sglang.srt.managers.scheduler.get_world_group") as gw, \
             patch("sglang.srt.distributed.nccl_ras.RasMonitor") as Monitor:
            enabled.get.return_value = True
            # A non-zero world rank must NOT start a collector.
            gw.return_value.rank_in_group = 3
            s.init_nccl_ras_collector()
        self.assertIsNone(s.ras_monitor)
        Monitor.assert_not_called()

    def test_metrics_off_elected_emits_warning_and_runs_log_only(self):
        s = self._bare_scheduler()
        s.metrics_collector = None
        with patch("sglang.srt.environ.envs.SGLANG_NCCL_RAS_ENABLE") as enabled, \
             patch("sglang.srt.managers.scheduler.get_world_group") as gw, \
             patch("sglang.srt.distributed.nccl_ras.RasMonitor") as Monitor:
            enabled.get.return_value = True
            gw.return_value.rank_in_group = 0
            inst = MagicMock()
            Monitor.return_value = inst
            with self.assertLogs("sglang", level="WARNING") as cm:
                s.init_nccl_ras_collector()
        # Monitor still started (log-only fallback), but constructed with
        # metrics_collector=None.
        self.assertIs(s.ras_monitor, inst)
        _, kwargs = Monitor.call_args
        self.assertIsNone(kwargs["metrics_collector"])
        inst.start.assert_called_once()
        self.assertTrue(
            any("--enable-metrics is off" in m for m in cm.output),
            cm.output,
        )

    def test_bootstrap_setdefaults_nccl_ras_enable(self):
        s = self._bare_scheduler()
        s.metrics_collector = object()
        os.environ.pop("NCCL_RAS_ENABLE", None)
        with patch("sglang.srt.environ.envs.SGLANG_NCCL_RAS_ENABLE") as enabled, \
             patch("sglang.srt.managers.scheduler.get_world_group") as gw, \
             patch("sglang.srt.distributed.nccl_ras.RasMonitor"):
            enabled.get.return_value = True
            gw.return_value.rank_in_group = 0
            s.init_nccl_ras_collector()
        self.assertEqual(os.environ.get("NCCL_RAS_ENABLE"), "1")
        os.environ.pop("NCCL_RAS_ENABLE", None)


if __name__ == "__main__":
    unittest.main()
