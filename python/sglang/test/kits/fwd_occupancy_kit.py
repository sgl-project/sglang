"""Single-batch decode GPU occupancy sanity kit.

Probes ``sglang:fwd_occupancy`` (a 0-100 percentage averaged over the
last ``decode_log_interval`` batches; resets to NaN at window
boundaries) under one long single-batch ``/generate`` request, and
asserts median above a threshold. Single-batch is where CPU overhead
dominates -- overlap scheduler / cuda graph regressions surface here
before batched throughput moves.

The gauge is meaningless if any other request shares the batch, so the
kit waits for an idle server before measuring and aborts if concurrent
traffic appears mid-window.

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
_NUM_RUNNING_REQS_RE = re.compile(
    r"^sglang:num_running_reqs(?:\{[^}]*\})?\s+(\S+)", re.MULTILINE
)
_GENERATE_REQUEST_TIMEOUT = 600
_METRICS_REQUEST_TIMEOUT = 10
_IDLE_WAIT_TIMEOUT = 60
_IDLE_POLL_INTERVAL = 0.5


class FwdOccupancyMixin:
    """Assert single-batch ``sglang:fwd_occupancy`` median > threshold."""

    fwd_occupancy_threshold: float = 95.0
    fwd_occupancy_min_samples: int = 5
    fwd_occupancy_scrape_interval: float = 0.5

    # Spec accept-length floor, only enforced under a spec algorithm.
    fwd_occupancy_acc_length_threshold: float = 1.8

    fwd_occupancy_warmup_max_new_tokens: int = 64
    fwd_occupancy_warmup_settle_seconds: float = 1.0
    fwd_occupancy_max_new_tokens: int = 2048
    fwd_occupancy_prompt: str = (
        "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:"
    )

    def _scrape(self):
        """Return (max non-NaN occupancy across labels, total running reqs),
        (None, None) on scrape failure."""
        try:
            resp = requests.get(
                self.base_url + "/metrics", timeout=_METRICS_REQUEST_TIMEOUT
            )
        except requests.RequestException:
            return None, None
        if resp.status_code != 200:
            return None, None
        vals = [
            float(v)
            for v in _FWD_OCCUPANCY_RE.findall(resp.text)
            if float(v) == float(v)  # NaN filter (gauge resets at window boundary)
        ]
        occ = max(vals) if vals else None
        try:
            running = sum(
                int(float(v)) for v in _NUM_RUNNING_REQS_RE.findall(resp.text)
            )
        except ValueError:
            running = 0
        return occ, running

    def _assert_metrics_device_timer_enabled(self):
        """Fail loudly on missing flag/env -- a NaN-only gauge looks like a
        real occupancy regression."""
        resp = requests.get(
            self.base_url + "/metrics", timeout=_METRICS_REQUEST_TIMEOUT
        )
        assert (
            resp.status_code == 200
        ), f"/metrics returned {resp.status_code}; launch with --enable-metrics"
        assert "sglang:fwd_occupancy" in resp.text, (
            "sglang:fwd_occupancy not exposed; set "
            "SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 and pass --enable-metrics"
        )

    def _fire(self, prompt: str, max_new_tokens: int):
        """Fire one /generate, return (meta_info, wall_time); (None, 0.0) on
        failure. Not concurrent -- breaks the single-batch invariant."""
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
            return None, 0.0
        elapsed = time.perf_counter() - t0
        try:
            return resp.json().get("meta_info", {}), elapsed
        except ValueError:  # non-JSON body
            return {}, elapsed

    def _warmup(self):
        """Fill cuda graphs + step the device-timer past its first NaN
        window."""
        self._fire(
            "warmup " + self.fwd_occupancy_prompt,
            self.fwd_occupancy_warmup_max_new_tokens,
        )
        time.sleep(self.fwd_occupancy_warmup_settle_seconds)

    def _wait_for_idle(self):
        """Block until the server holds no running/waiting reqs. The gauge is
        a single-batch signal; concurrent traffic corrupts it."""
        deadline = time.perf_counter() + _IDLE_WAIT_TIMEOUT
        while time.perf_counter() < deadline:
            try:
                resp = requests.get(
                    self.base_url + "/get_load", timeout=_METRICS_REQUEST_TIMEOUT
                )
                resp.raise_for_status()
                if sum(dp.get("num_reqs", 0) for dp in resp.json()) == 0:
                    return
            except (requests.RequestException, ValueError):
                pass  # transient scrape failure -- retry, don't assume idle
            time.sleep(_IDLE_POLL_INTERVAL)
        raise AssertionError(
            f"server not idle after {_IDLE_WAIT_TIMEOUT}s; fwd_occupancy is a "
            "single-batch metric -- concurrent traffic must drain first"
        )

    def _measure(self):
        """Background-fire one long single-batch request, scrape /metrics on
        the foreground; return (non-NaN samples, perf). Aborts immediately if
        any foreign request shares the batch (single-batch is a hard
        invariant)."""
        samples = []
        request_done = threading.Event()
        result = {"meta_info": None, "elapsed": 0.0}

        def fire_one():
            try:
                result["meta_info"], result["elapsed"] = self._fire(
                    self.fwd_occupancy_prompt, self.fwd_occupancy_max_new_tokens
                )
            finally:
                request_done.set()

        firer = threading.Thread(target=fire_one, daemon=True)
        firer.start()

        while not request_done.is_set():
            occ, running = self._scrape()
            if occ is not None and running is not None:
                if running > 1:
                    request_done.set()
                    raise AssertionError(
                        f"single-batch invariant violated: {running} reqs running "
                        f"(expected 1). fwd_occupancy is a bs=1 metric -- a sibling "
                        "process/test is hitting this shared server."
                    )
                samples.append(occ)
            time.sleep(self.fwd_occupancy_scrape_interval)

        firer.join(timeout=_GENERATE_REQUEST_TIMEOUT)
        return samples, self._perf(result)

    def _perf(self, result):
        meta = result["meta_info"] or {}
        elapsed = result["elapsed"]
        out = meta.get("completion_tokens", 0) or 0
        decode_tps = meta.get("decode_throughput", 0.0) or 0.0
        return {
            "input_tokens": meta.get("prompt_tokens", 0) or 0,
            "output_tokens": out,
            "decode_tps": decode_tps,
            "mean_itl_ms": (1000.0 / decode_tps) if decode_tps > 0 else 0.0,
            "wall_tps": (out / elapsed) if elapsed > 0 else 0.0,
        }

    def test_fwd_occupancy(self):
        self._assert_metrics_device_timer_enabled()
        self._warmup()
        self._wait_for_idle()
        samples, perf = self._measure()

        # avg_spec_accept_length is only present under a spec algorithm.
        try:
            info = requests.get(
                self.base_url + "/server_info", timeout=_METRICS_REQUEST_TIMEOUT
            ).json()
            avg_accept = info["internal_states"][0].get("avg_spec_accept_length")
        except (requests.RequestException, KeyError, IndexError):
            avg_accept = None

        # Stats + tables printed before assertions so failures still surface them.
        samples_sorted = sorted(samples)
        if samples_sorted:
            median = statistics.median(samples_sorted)
            peak = samples_sorted[-1]
            p10 = samples_sorted[
                min(len(samples_sorted) - 1, len(samples_sorted) // 10)
            ]
        else:
            median = peak = p10 = float("nan")

        perf_rows = [
            ["input tokens", perf["input_tokens"]],
            ["output tokens", perf["output_tokens"]],
            ["decode tps", f"{perf['decode_tps']:.2f}"],
            ["mean itl (ms)", f"{perf['mean_itl_ms']:.2f}"],
            ["wall tps", f"{perf['wall_tps']:.2f}"],
        ]
        if avg_accept is not None:
            perf_rows.append(["avg spec accept", f"{avg_accept:.3f}"])
        print(
            "\n"
            + tabulate.tabulate(
                perf_rows, headers=["perf metric", "value"], tablefmt="github"
            )
        )

        print(
            "\n\n"
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

        self.assertGreaterEqual(
            len(samples),
            self.fwd_occupancy_min_samples,
            f"only {len(samples)} non-NaN samples (need >= "
            f"{self.fwd_occupancy_min_samples}); window too short or gauge at NaN",
        )
        self.assertGreater(
            median,
            self.fwd_occupancy_threshold,
            f"median={median:.2f} <= threshold {self.fwd_occupancy_threshold} "
            f"(peak={peak:.2f}, p10={p10:.2f}, n={len(samples)})",
        )
        if avg_accept is not None:
            self.assertGreater(
                avg_accept,
                self.fwd_occupancy_acc_length_threshold,
                f"avg_spec_accept_length={avg_accept:.3f} <= "
                f"{self.fwd_occupancy_acc_length_threshold} -- spec barely accepted",
            )
