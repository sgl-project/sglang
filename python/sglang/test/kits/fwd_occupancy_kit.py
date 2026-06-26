"""Single-batch decode GPU occupancy sanity kit.

Probes ``sglang:fwd_occupancy`` (a 0-100 percentage averaged over the
last ``decode_log_interval`` batches; resets to NaN at window
boundaries) under one long single-batch ``/generate`` request, and
asserts median above a threshold. Single-batch is where CPU overhead
dominates -- overlap scheduler / cuda graph regressions surface here
before batched throughput moves.

Prerequisites on the consuming server:
    env:          SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
    server flag:  --enable-metrics

Mix into a ``CustomTestCase`` subclass exposing ``self.base_url``.
"""

import re
import statistics
import threading
import time

import requests
import tabulate

_FWD_OCCUPANCY_RE = re.compile(
    r"^sglang:fwd_occupancy(?:\{[^}]*\})?\s+(\S+)", re.MULTILINE
)
_GENERATE_REQUEST_TIMEOUT = 600
_METRICS_REQUEST_TIMEOUT = 10


class FwdOccupancyMixin:
    """Assert single-batch ``sglang:fwd_occupancy`` median > threshold."""

    fwd_occupancy_threshold: float = 95.0
    fwd_occupancy_min_samples: int = 5
    fwd_occupancy_scrape_interval: float = 0.5

    # Spec-decoding accept-length floor. Only enforced when the server
    # is running with a spec algorithm (avg_spec_accept_length present
    # in /server_info); silently skipped otherwise. EAGLE3 3/1/4 on
    # 5090 + Llama-3.1-8B measured ~2.0 in CI; 1.8 leaves a small
    # buffer while still catching silent fallback to vanilla (~1.0).
    fwd_occupancy_acc_length_threshold: float = 1.8

    # Warmup: one short request to fill cuda graphs + get the
    # device-timer past its first NaN window.
    fwd_occupancy_warmup_max_new_tokens: int = 64
    fwd_occupancy_warmup_settle_seconds: float = 1.0

    # Measurement: one long single-batch request -- max_new_tokens must
    # span several decode_log_interval windows for enough samples.
    fwd_occupancy_max_new_tokens: int = 2048
    fwd_occupancy_prompt: str = (
        "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:"
    )

    def _scrape_fwd_occupancy(self):
        """Max non-NaN gauge value across exposed labels (e.g. dp ranks);
        None on transient scrape failure."""
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
        """Fail loudly on missing flag/env -- otherwise a NaN-only gauge
        looks like a real occupancy regression."""
        resp = requests.get(
            self.base_url + "/metrics", timeout=_METRICS_REQUEST_TIMEOUT
        )
        if resp.status_code != 200:
            raise AssertionError(
                f"/metrics returned {resp.status_code}; the test class's "
                "server must be launched with --enable-metrics"
            )
        if "sglang:fwd_occupancy" not in resp.text:
            raise AssertionError(
                "sglang:fwd_occupancy gauge not exposed; set "
                "SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 in the server's env "
                "and pass --enable-metrics"
            )

    def _fwd_occupancy_fire(self, prompt: str, max_new_tokens: int):
        """Fire one /generate, return (meta_info, wall_time). Not
        concurrent -- breaks the single-batch invariant. ``meta_info``
        is None on failure."""
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                },
                timeout=_GENERATE_REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            # Individual fire failure isn't the signal; stats-vs-threshold is.
            return None, 0.0
        elapsed = time.perf_counter() - t0
        try:
            meta_info = resp.json().get("meta_info", {})
        except ValueError:  # non-JSON body
            meta_info = {}
        return meta_info, elapsed

    def _fwd_occupancy_warmup(self):
        """Fill cuda graphs + step the device-timer past its first NaN
        window before measurement starts."""
        self._fwd_occupancy_fire(
            "warmup " + self.fwd_occupancy_prompt,
            self.fwd_occupancy_warmup_max_new_tokens,
        )
        time.sleep(self.fwd_occupancy_warmup_settle_seconds)

    def _fwd_occupancy_measure(self):
        """Background-fire one long single-batch request, scrape
        /metrics on the foreground; return (non-NaN samples, perf)."""
        samples = []
        request_done = threading.Event()
        result = {"meta_info": None, "elapsed": 0.0}

        def fire_one():
            try:
                result["meta_info"], result["elapsed"] = self._fwd_occupancy_fire(
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
                samples.append(v)
            time.sleep(self.fwd_occupancy_scrape_interval)

        firer.join(timeout=_GENERATE_REQUEST_TIMEOUT)

        meta = result["meta_info"] or {}
        elapsed = result["elapsed"]
        completion_tokens = meta.get("completion_tokens", 0) or 0
        decode_throughput = meta.get("decode_throughput", 0.0) or 0.0
        perf = {
            "input_tokens": meta.get("prompt_tokens", 0) or 0,
            "output_tokens": completion_tokens,
            "decode_throughput": decode_throughput,
            "mean_itl_ms": (
                (1000.0 / decode_throughput) if decode_throughput > 0 else 0.0
            ),
            "wall_throughput": (completion_tokens / elapsed if elapsed > 0 else 0.0),
        }
        return samples, perf

    def test_fwd_occupancy(self):
        self._assert_metrics_device_timer_enabled()
        self._fwd_occupancy_warmup()
        samples, perf = self._fwd_occupancy_measure()

        # avg_spec_accept_length is only present under a spec algorithm;
        # absent otherwise (vanilla decode skips the accept-length check).
        try:
            info = requests.get(
                self.base_url + "/server_info", timeout=_METRICS_REQUEST_TIMEOUT
            ).json()
            avg_accept = info["internal_states"][0].get("avg_spec_accept_length")
        except (requests.RequestException, KeyError, IndexError):
            avg_accept = None

        # Compute stats up front so both tables print before any
        # assertion -- failing assertions must still surface the numbers.
        samples_sorted = sorted(samples)
        if samples_sorted:
            median = statistics.median(samples_sorted)
            peak = samples_sorted[-1]
            p10_idx = min(len(samples_sorted) - 1, max(0, len(samples_sorted) // 10))
            p10 = samples_sorted[p10_idx]
        else:
            median = peak = p10 = float("nan")

        # Perf table: request-level throughput (not the occupancy gauge).
        perf_rows = [
            ["input tokens", perf["input_tokens"]],
            ["output tokens", perf["output_tokens"]],
            ["decode tps", f"{perf['decode_throughput']:.2f}"],
            ["mean itl (ms)", f"{perf['mean_itl_ms']:.2f}"],
            ["wall tps", f"{perf['wall_throughput']:.2f}"],
        ]
        if avg_accept is not None:
            perf_rows.append(["avg spec accept", f"{avg_accept:.3f}"])
        print(
            "\n"
            + tabulate.tabulate(
                perf_rows,
                headers=["perf metric", "value"],
                tablefmt="github",
            )
        )

        # Occupancy table.
        print(
            "\n"
            + tabulate.tabulate(
                [
                    ["samples (n)", len(samples)],
                    ["median", f"{median:.2f}"],
                    ["peak", f"{peak:.2f}"],
                    ["p10", f"{p10:.2f}"],
                    ["threshold", f"{self.fwd_occupancy_threshold:.2f}"],
                ],
                headers=["fwd_occupancy", "value"],
                tablefmt="github",
            )
        )

        # All reporting done above.
        self.assertGreaterEqual(
            len(samples),
            self.fwd_occupancy_min_samples,
            f"only {len(samples)} non-NaN occupancy samples collected "
            f"(need >= {self.fwd_occupancy_min_samples}); the measurement "
            "window may be too short or the gauge stuck at NaN",
        )

        self.assertGreater(
            median,
            self.fwd_occupancy_threshold,
            f"sglang:fwd_occupancy median={median:.2f} did not exceed "
            f"threshold {self.fwd_occupancy_threshold} "
            f"(peak={peak:.2f}, p10={p10:.2f}, n={len(samples)})",
        )

        if avg_accept is not None:
            self.assertGreater(
                avg_accept,
                self.fwd_occupancy_acc_length_threshold,
                f"avg_spec_accept_length={avg_accept:.3f} did not exceed "
                f"threshold {self.fwd_occupancy_acc_length_threshold} -- spec "
                "barely accepted, possibly degraded to vanilla decode",
            )
