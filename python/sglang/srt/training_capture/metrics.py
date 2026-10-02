"""Bounded Prometheus series for one training-capture producer."""

from collections import Counter
from time import time

from sglang.srt.training_capture.timings import CaptureTimings


class CaptureMetrics:
    EVENTS = (
        "considered",
        "admitted",
        "sampled_out",
        "adaptive_sampled_out",
        "excluded_disabled",
        "excluded_paused",
        "control_pause",
        "control_resume",
        "control_abort",
        "excluded_health_check",
        "excluded_length",
        "excluded_unsupported",
        "admission_backpressure",
        "admission_invalid_request",
        "admission_catalog_error",
        "lease_rejected",
        "lease_renew_error",
        "capture_failed",
        "writer_started",
        "writer_failed",
        "copies_completed",
        "snapshot_built",
        "sealed",
        "ready",
        "failure_report_error",
        "eager_forwards",
        "cuda_graph_forwards",
        "overlap_forwards",
        "speculative_verify_forwards",
        "speculative_commits_copied",
        "other",
    )
    STATES = ("available", "active", "queued", "writing", "pending_publication")
    ACTIONS = ("decreases", "recoveries", "failures", "pauses")

    def __init__(self, labels, *, registry=None):
        # Import after the server configures PROMETHEUS_MULTIPROC_DIR.
        from prometheus_client import REGISTRY, Counter, Gauge

        registry = REGISTRY if registry is None else registry
        self.labels = dict(labels)
        self.previous = {}

        def gauge(name, documentation, extra=()):
            return Gauge(
                "sglang:training_capture_" + name,
                documentation,
                labelnames=[*labels, *extra],
                multiprocess_mode="mostrecent",
                registry=registry,
            )

        def counter(name, documentation, extra):
            return Counter(
                "sglang:training_capture_" + name,
                documentation,
                labelnames=[*labels, *extra],
                registry=registry,
            )

        self.events = counter(
            "events",
            "Capture lifecycle events; event categories may overlap.",
            ("event",),
        )
        self.adjustments = counter(
            "admission_adjustments", "Adaptive capture control actions.", ("action",)
        )
        self.stage_calls = counter(
            "stage_calls", "Completed background capture stage attempts.", ("stage",)
        )
        self.stage_failures = counter(
            "stage_failures",
            "Background capture stage attempts that raised.",
            ("stage",),
        )
        self.stage_seconds = counter(
            "stage_seconds",
            "Wall time in completed capture stage attempts.",
            ("stage",),
        )
        self.stage_max = gauge(
            "stage_max_seconds", "Lifetime maximum capture stage wall time.", ("stage",)
        )
        self.ratio = gauge("sample_ratio", "Capture admission probability.", ("kind",))
        self.reservations = gauge(
            "reservations", "Capture reservations by lifecycle state.", ("state",)
        )
        self.host_slots = gauge(
            "host_slots", "Registered Host slots by ownership state.", ("state",)
        )
        self.disabled = gauge(
            "disabled", "One when capture is disabled by a failure or shutdown."
        )
        self.paused = gauge(
            "admission_paused", "One when an operator paused capture admission."
        )
        self.adaptive = gauge(
            "adaptive_enabled", "One when adaptive admission is configured."
        )
        self.queued = gauge(
            "queue_depth", "Reservations awaiting background processing."
        )
        self.host_bytes = gauge(
            "host_allocated_bytes", "Allocated registered Host arena bytes."
        )
        self.device_bytes = gauge(
            "kv_staging_allocated_bytes", "Allocated KV staging tensor bytes."
        )
        self.device_limit = gauge(
            "kv_staging_limit_bytes", "Configured KV staging tensor byte budget."
        )
        self.occupancy = gauge(
            "occupied_fraction", "Busy or quarantined fraction of capture capacity."
        )
        self.writer_age = gauge(
            "writer_age_seconds",
            "Age of the oldest queued, writing or pending publication.",
        )
        self.cooldown = gauge(
            "cooldown_seconds", "Remaining adaptive admission cooldown."
        )
        self.latency_enabled = gauge(
            "latency_control_enabled", "Scheduler latency protection configured."
        )
        self.latency_blocked = gauge(
            "latency_blocked", "Capture paused by scheduler latency protection."
        )
        self.latency_recovery_ready = gauge(
            "latency_recovery_ready", "Fresh observations permit latency recovery."
        )
        self.latency_state = gauge(
            "latency_state", "Scheduler latency control state.", ("state",)
        )
        self.latency_seconds = gauge(
            "scheduler_latency_seconds",
            "Scheduler-side latency quantile/budget, not client latency.",
            ("metric", "kind"),
        )
        self.latency_count = gauge(
            "latency_window_observations",
            "Observations in the bounded latency window.",
            ("metric",),
        )
        self.latency_observations = counter(
            "latency_observations", "Valid scheduler latency observations.", ("metric",)
        )
        self.latency_percentile = gauge(
            "latency_percentile", "Quantile used for scheduler latency protection."
        )
        self.updated = gauge(
            "metrics_update_timestamp_seconds",
            "Unix time of the last successful capture metrics update.",
        )

    def _increment(self, collector, key, value, **labels):
        previous = self.previous.get(key, 0)
        # Retrying a snapshot must not count the same event again.
        collector.labels(**self.labels, **labels).inc(max(0, value - previous))
        self.previous[key] = max(value, previous)

    def update(self, stats):
        events = Counter()
        for name, value in stats["counters"].items():
            if name in self.EVENTS:
                event = name
            elif name.startswith("failed_"):
                event = "capture_failed"
            elif name.startswith("writer_failed_"):
                event = "writer_failed"
            elif name.startswith("excluded_"):
                event = "excluded_unsupported"
            else:
                event = "other"
            events[event] += value
        for event in self.EVENTS:
            self._increment(self.events, ("event", event), events[event], event=event)
        for stage in CaptureTimings.STAGES:
            value = stats.get("stage_timings", {}).get(stage, {})
            for field, collector in (
                ("calls", self.stage_calls),
                ("errors", self.stage_failures),
                ("seconds", self.stage_seconds),
            ):
                self._increment(
                    collector,
                    ("stage", stage, field),
                    value.get(field, 0),
                    stage=stage,
                )
            self.stage_max.labels(**self.labels, stage=stage).set(
                value.get("max_seconds", 0)
            )
        admission = stats["admission"]
        latency = admission.get("latency")
        self.latency_enabled.labels(**self.labels).set(int(latency is not None))
        self.latency_blocked.labels(**self.labels).set(
            int(bool(latency and latency["blocked"]))
        )
        self.latency_recovery_ready.labels(**self.labels).set(
            int(bool(latency and latency["recovery_ready"]))
        )
        self.latency_percentile.labels(**self.labels).set(
            latency["percentile"] if latency else float("nan")
        )
        for state in ("disabled", "warming", "healthy", "breached", "stale"):
            current = latency["state"] if latency else "disabled"
            self.latency_state.labels(**self.labels, state=state).set(
                int(current == state)
            )
        for name in ("ttft", "tpot"):
            observation = latency["metrics"].get(name, {}) if latency else {}
            for kind, field in (
                ("observed", "percentile_seconds"),
                ("budget", "budget_seconds"),
            ):
                value = observation.get(field)
                self.latency_seconds.labels(**self.labels, metric=name, kind=kind).set(
                    value if value is not None else float("nan")
                )
            self.latency_count.labels(**self.labels, metric=name).set(
                observation.get("window_observations", 0)
            )
            self._increment(
                self.latency_observations,
                ("latency", name),
                observation.get("observations", 0),
                metric=name,
            )
        for action in self.ACTIONS:
            self._increment(
                self.adjustments, ("action", action), admission[action], action=action
            )
        for kind in ("configured", "target", "effective"):
            self.ratio.labels(**self.labels, kind=kind).set(admission[kind + "_ratio"])
        for state in self.STATES:
            self.reservations.labels(**self.labels, state=state).set(
                stats["states"].get(state, 0)
            )
        for state in ("free", "filling", "quarantined"):
            self.host_slots.labels(**self.labels, state=state).set(
                stats["host_pool"][state]
            )
        for metric, value in (
            (self.disabled, int(stats["disabled_reason"] is not None)),
            (self.paused, int(stats.get("admission_paused", False))),
            (self.adaptive, int(admission["adaptive"])),
            (self.queued, stats["queued"]),
            (self.host_bytes, stats["host_pool"]["allocated_bytes"]),
            (self.device_bytes, stats["host_pool"]["device_allocated_bytes"]),
            (self.device_limit, stats["host_pool"]["device_limit_bytes"]),
            (self.occupancy, stats["occupied_fraction"]),
            (self.writer_age, stats["writer_age_seconds"]),
            (self.cooldown, admission["cooldown_remaining_seconds"]),
        ):
            metric.labels(**self.labels).set(value)
        self.updated.labels(**self.labels).set(time())
