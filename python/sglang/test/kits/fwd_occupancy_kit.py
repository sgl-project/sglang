"""GPU forward-pass occupancy sanity kit.

Probes the ``sglang:fwd_occupancy`` Prometheus gauge under sustained
``/generate`` load and asserts the steady-state value clears a
threshold. Catches CPU-side regressions (overlap scheduler hiccups,
metadata copy bloat, host-side syncs) that don't break correctness but
starve the GPU.

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
from concurrent.futures import ThreadPoolExecutor

import requests

_FWD_OCCUPANCY_RE = re.compile(
    r"^sglang:fwd_occupancy(?:\{[^}]*\})?\s+(\S+)", re.MULTILINE
)
_GENERATE_REQUEST_TIMEOUT = 300
_METRICS_REQUEST_TIMEOUT = 10


class FwdOccupancyMixin:
    """Assert steady-state ``sglang:fwd_occupancy`` under sustained load.

    Threshold is a percentage in [0, 100]. Default is conservative for
    single-batch decode with overlap scheduler + cuda graph on a
    mid-size model; saturated multi-batch decode on H100/H200 should
    reach 95+.
    """

    # Threshold + detection
    fwd_occupancy_threshold: float = 85.0
    fwd_occupancy_min_samples: int = 5
    fwd_occupancy_scrape_interval: float = 0.5

    # Warmup phase -- fires synchronously before measurement starts
    fwd_occupancy_warmup_requests: int = 8
    fwd_occupancy_warmup_max_new_tokens: int = 64
    fwd_occupancy_warmup_settle_seconds: float = 1.0

    # Measurement load
    fwd_occupancy_workers: int = 32
    fwd_occupancy_total_requests: int = 128
    fwd_occupancy_max_new_tokens: int = 256
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
        """Fire a few requests synchronously to drive the server into
        steady-state before sampling. The device-timer window also needs
        a few batches of activity before the gauge produces non-NaN
        values."""
        prompts = [
            f"warmup#{i} {self.fwd_occupancy_prompt}"
            for i in range(self.fwd_occupancy_warmup_requests)
        ]
        with ThreadPoolExecutor(
            max_workers=self.fwd_occupancy_warmup_requests
        ) as executor:
            futures = [
                executor.submit(
                    self._fwd_occupancy_fire,
                    p,
                    self.fwd_occupancy_warmup_max_new_tokens,
                )
                for p in prompts
            ]
            for f in futures:
                f.result()
        # Give the scheduler a beat to settle before measurement starts.
        time.sleep(self.fwd_occupancy_warmup_settle_seconds)

    def _fwd_occupancy_measure(self):
        """Fire concurrent load + scrape /metrics in a background
        thread; return the list of non-NaN occupancy samples observed
        while the load was in-flight."""
        samples = []
        samples_lock = threading.Lock()
        scraping_done = threading.Event()

        def scrape_loop():
            while not scraping_done.is_set():
                v = self._scrape_fwd_occupancy()
                if v is not None:
                    with samples_lock:
                        samples.append(v)
                time.sleep(self.fwd_occupancy_scrape_interval)

        scraper = threading.Thread(target=scrape_loop, daemon=True)
        scraper.start()

        prompts = [
            f"#{i} {self.fwd_occupancy_prompt}"
            for i in range(self.fwd_occupancy_total_requests)
        ]
        try:
            with ThreadPoolExecutor(max_workers=self.fwd_occupancy_workers) as executor:
                futures = [
                    executor.submit(
                        self._fwd_occupancy_fire,
                        p,
                        self.fwd_occupancy_max_new_tokens,
                    )
                    for p in prompts
                ]
                for f in futures:
                    f.result()
        finally:
            scraping_done.set()
            scraper.join(timeout=5.0)

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
