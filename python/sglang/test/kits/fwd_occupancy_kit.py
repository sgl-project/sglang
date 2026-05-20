"""GPU forward-pass occupancy sanity kit (single-batch decode).

Probes the ``sglang:fwd_occupancy`` Prometheus gauge while a single
long ``/generate`` request is in flight (batch_size = 1 decode) and
asserts the steady-state value clears a threshold.

Single-batch decode is the worst case for CPU overhead: each decode
step produces just one token, so per-step host-side work (kernel
dispatch, sampling, metadata bookkeeping) competes with a single tiny
forward pass. If the overlap scheduler or cuda graph capture stalls,
GPU idle ratio shoots up here before anything shows in batched
throughput numbers.

The gauge is a percentage in [0, 100] -- ``gpu_busy / wall_clock * 100``
over the most recent ``decode_log_interval`` batches. It cycles back to
NaN on every window reset, so the probe filters NaN samples.

Prerequisites on the server launched by the consuming test class:
    env:          SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
    server flag:  --enable-metrics

Without both, ``sglang:fwd_occupancy`` is either not exposed or
permanently NaN, and the probe fails fast with a clear message.

Mix into any ``CustomTestCase`` subclass that exposes ``self.base_url``.
"""

import re
import statistics
import threading
import time

import requests

_FWD_OCCUPANCY_RE = re.compile(
    r"^sglang:fwd_occupancy(?:\{[^}]*\})?\s+(\S+)", re.MULTILINE
)
_GENERATE_REQUEST_TIMEOUT = 600
_METRICS_REQUEST_TIMEOUT = 10


class FwdOccupancyMixin:
    """Assert steady-state ``sglang:fwd_occupancy`` for single-batch decode.

    Threshold is a percentage in [0, 100]. Default 95.0 -- single-batch
    decode with overlap scheduler + cuda graph on a healthy GPU should
    hold steady-state 95+; falling below this points at CPU-side
    regressions even when batched throughput still looks fine.
    """

    # Threshold + detection
    fwd_occupancy_threshold: float = 95.0
    fwd_occupancy_min_samples: int = 5
    fwd_occupancy_scrape_interval: float = 0.5

    # Warmup phase -- one short request to fill cuda graphs and let the
    # device-timer window emit its first non-NaN sample.
    fwd_occupancy_warmup_max_new_tokens: int = 64
    fwd_occupancy_warmup_settle_seconds: float = 1.0

    # Measurement load -- one long single-batch request. max_new_tokens
    # must cover many decode_log_interval windows worth of decode steps
    # so the scraper collects enough samples.
    fwd_occupancy_max_new_tokens: int = 2048
    fwd_occupancy_prompt: str = "Write a long, detailed, multi-paragraph story about "

    def _scrape_fwd_occupancy(self):
        """Return the max non-NaN ``sglang:fwd_occupancy`` value across
        exposed labels (e.g. dp ranks); None on transient scrape failure
        or no non-NaN sample."""
        try:
            resp = requests.get(
                self.base_url + "/metrics", timeout=_METRICS_REQUEST_TIMEOUT
            )
        except requests.RequestException:
            return None
        if resp.status_code != 200:
            return None
        vals = []
        for raw in _FWD_OCCUPANCY_RE.findall(resp.text):
            try:
                v = float(raw)
            except ValueError:
                continue
            if v == v:  # NaN filter (gauge resets to NaN on window boundary)
                vals.append(v)
        return max(vals) if vals else None

    def _assert_metrics_device_timer_enabled(self):
        """Fail loudly if --enable-metrics or
        SGLANG_ENABLE_METRICS_DEVICE_TIMER aren't in effect; otherwise
        a missing gauge would look like a real occupancy regression."""
        resp = requests.get(
            self.base_url + "/metrics", timeout=_METRICS_REQUEST_TIMEOUT
        )
        assert resp.status_code == 200, (
            f"/metrics returned {resp.status_code}; the test class's server "
            "must be launched with --enable-metrics"
        )
        assert "sglang:fwd_occupancy" in resp.text, (
            "sglang:fwd_occupancy gauge not exposed; set "
            "SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 in the server's env and "
            "pass --enable-metrics"
        )

    def _fwd_occupancy_fire(self, prompt: str, max_new_tokens: int):
        """Synchronously fire one /generate request. Single batch_size=1
        decode -- this method must never be called concurrently if the
        probe's single-batch invariant is to hold."""
        try:
            requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": max_new_tokens,
                    },
                },
                timeout=_GENERATE_REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            # Individual fire failure is not the probe's concern; the
            # final stats-vs-threshold check is the signal.
            pass

    def _fwd_occupancy_warmup(self):
        """Fire one short request synchronously to fill cuda graphs and
        let the device-timer window emit its first non-NaN sample before
        measurement starts."""
        self._fwd_occupancy_fire(
            "warmup " + self.fwd_occupancy_prompt,
            self.fwd_occupancy_warmup_max_new_tokens,
        )
        time.sleep(self.fwd_occupancy_warmup_settle_seconds)

    def _fwd_occupancy_measure(self):
        """Fire one long single-batch request in a background thread and
        scrape /metrics from the main thread while it's in flight;
        return the list of non-NaN occupancy samples observed."""
        samples = []
        samples_lock = threading.Lock()
        request_done = threading.Event()

        def fire_one():
            try:
                self._fwd_occupancy_fire(
                    self.fwd_occupancy_prompt,
                    self.fwd_occupancy_max_new_tokens,
                )
            finally:
                request_done.set()

        firer = threading.Thread(target=fire_one, daemon=True)
        firer.start()

        while not request_done.is_set():
            v = self._scrape_fwd_occupancy()
            if v is not None:
                with samples_lock:
                    samples.append(v)
            time.sleep(self.fwd_occupancy_scrape_interval)

        firer.join(timeout=_GENERATE_REQUEST_TIMEOUT)
        return samples

    def test_fwd_occupancy(self):
        self._assert_metrics_device_timer_enabled()
        self._fwd_occupancy_warmup()
        samples = self._fwd_occupancy_measure()

        self.assertGreaterEqual(
            len(samples),
            self.fwd_occupancy_min_samples,
            f"only {len(samples)} non-NaN occupancy samples collected "
            f"(need >= {self.fwd_occupancy_min_samples}); the measurement "
            "window may be too short or the gauge stuck at NaN",
        )

        # Median is the steady-state signal; peak / p10 included in the
        # assertion message for triage.
        samples_sorted = sorted(samples)
        median = statistics.median(samples_sorted)
        peak = samples_sorted[-1]
        p10 = samples_sorted[max(0, len(samples_sorted) // 10 - 1)]
        print(
            f"fwd_occupancy samples (n={len(samples)}): "
            f"median={median:.2f} peak={peak:.2f} p10={p10:.2f}"
        )

        self.assertGreater(
            median,
            self.fwd_occupancy_threshold,
            f"sglang:fwd_occupancy median={median:.2f} did not exceed "
            f"threshold {self.fwd_occupancy_threshold} "
            f"(peak={peak:.2f}, p10={p10:.2f}, n={len(samples)})",
        )
