"""NPU test for the Prefill Delayer queue-based trigger: above/below threshold.

Two test classes (both with assertions):
    - TestNpuPrefillDelayerAboveThreshold: sends requests exceeding
      queue_min (= min(running*ratio, max_prefill_bs)) so "waiting >= queue_min"
      holds and prefill is released promptly. Asserts short requests complete
      below max_delay_ms.
    - TestNpuPrefillDelayerBelowThreshold: sends a request count far below
      queue_min while keeping running high. The delay can only be released on
      the max_delay_ms timeout. Asserts each short request waits > max_delay_ms.
"""

import os
import re
import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


# Server config
MODEL_PATH = QWEN3_0_6B_WEIGHTS_PATH
MAX_DELAY_MS = 5000
QUEUE_MIN_RATIO = 0.8

# Long request params (used to fill up running)
LONG_PROMPT = (
    "Please describe in detail the development history of artificial intelligence "
    "from the Turing test to deep learning, including key figures and milestone "
    "events. Start from the early neural network research in the 1940s, through the "
    "proposal of the Turing test in the 1950s, the birth of the perceptron in the "
    "1960s, the development of expert systems in the 1970s, the breakthrough of the "
    "backpropagation algorithm in the 1980s, the rise of statistical learning in the "
    "1990s, the revival of deep learning in the 2000s, and AlexNet's breakthrough "
    "result in the ImageNet competition in 2012. In recent years, the development of "
    "large language models such as GPT and BERT has pushed artificial intelligence to "
    "new heights."
)
LONG_MAX_TOKENS = 256
# Large max_tokens keeps running high across the whole max_delay window.
LONG_SUSTAINED_MAX_TOKENS = 2048
LONG_CONCURRENT = 20

# Short request params (used to trigger queueing)
SHORT_PROMPT = "What is artificial intelligence?"
SHORT_MAX_TOKENS = 50
# Above threshold: waiting >= queue_min -> no delay, immediate release.
SHORT_CONCURRENT_ABOVE = 25
# Below threshold: waiting < queue_min -> released only on max_delay_ms timeout.
SHORT_CONCURRENT_BELOW = 3

# Module-level server process shared by both test classes.
GLOBAL_SERVER_PROCESS = None


def setUpModule():
    global GLOBAL_SERVER_PROCESS
    env = os.environ.copy()
    env["SGLANG_PREFILL_DELAYER_DEBUG_LOG"] = "1"

    other_args = [
        "--attention-backend",
        "ascend",
        "--enable-metrics",
        "--enable-prefill-delayer",
        "--prefill-delayer-queue-min-ratio",
        str(QUEUE_MIN_RATIO),
        "--prefill-delayer-max-delay-ms",
        str(MAX_DELAY_MS),
    ]
    GLOBAL_SERVER_PROCESS = popen_launch_server(
        MODEL_PATH,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        env=env,
    )


def tearDownModule():
    if GLOBAL_SERVER_PROCESS is not None:
        kill_process_tree(GLOBAL_SERVER_PROCESS.pid)


def _parse_gauge(metrics_text: str, name: str):
    """Return the last value of a prometheus gauge line, or None."""
    matches = re.findall(
        rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)",
        metrics_text,
        re.MULTILINE,
    )
    if not matches:
        return None
    return int(float(matches[-1]))


def _query_status(base_url: str, tag: str, verbose: bool = True):
    try:
        stats = requests.get(f"{base_url}/metrics", timeout=10).text
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[{tag}] query /metrics failed: {e}")
        return None, None
    running = _parse_gauge(stats, "sglang:num_running_reqs")
    waiting = _parse_gauge(stats, "sglang:num_queue_reqs")
    if verbose:
        print(f"[{tag}] num_running_reqs = {running}")
        print(f"[{tag}] num_queue_reqs   = {waiting}")
    return running, waiting


def _wait_until_stable_running(
    base_url: str,
    expected_running: int,
    tolerance: int = 2,
    stable_secs: float = 4.0,
    interval: float = 0.5,
    timeout: float = 40.0,
):
    """Poll /metrics until running stays >= (expected - tolerance) for stable_secs.

    A single sample is not enough: the metric counts before actual prefill/decode.
    Requiring continuous stability skips the long requests' own prefill/flush
    phase so short requests land in a steady decode window. Returns the running
    value once stable, or None on timeout.
    """
    target = max(1, expected_running - tolerance)
    deadline = time.perf_counter() + timeout
    stable_since = None
    while time.perf_counter() < deadline:
        running, _ = _query_status(base_url, "wait", verbose=False)
        now = time.perf_counter()
        if running is not None and running >= target:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= stable_secs:
                return running
        else:
            stable_since = None
        time.sleep(interval)
    return None


def _send_long_request(base_url: str, max_tokens: int = LONG_MAX_TOKENS):
    try:
        requests.post(
            f"{base_url}/generate",
            json={
                "text": LONG_PROMPT,
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                },
                # ignore_eos keeps running high; otherwise a request may finish
                # early on EOS and running drops below the stable target.
                "ignore_eos": True,
            },
            timeout=180,
        )
    except Exception as e:  # noqa: BLE001
        print(f"long request error: {e}")


def _send_short_request(base_url: str, idx: int, results: dict):
    """Send a short request; store (elapsed_ms, status_code) in results[idx]."""
    start = time.perf_counter()
    status = None
    try:
        resp = requests.post(
            f"{base_url}/generate",
            json={
                "text": SHORT_PROMPT,
                "sampling_params": {
                    "max_new_tokens": SHORT_MAX_TOKENS,
                    "temperature": 0.7,
                },
            },
            timeout=180,
        )
        status = resp.status_code
    except Exception as e:  # noqa: BLE001
        print(f"short request {idx} error: {e}")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    results[idx] = (elapsed_ms, status)


def _print_short_results(results: dict):
    """Print short request latencies and check all status codes are 200."""
    all_ok = True
    for idx in sorted(results):
        elapsed_ms, status = results[idx]
        status_str = "OK" if status == 200 else str(status)
        print(f"    {idx}: {elapsed_ms:.0f}ms  (HTTP {status_str})")
        if status != 200:
            all_ok = False
    return all_ok


def _start_long_requests(base_url: str):
    """Start LONG_CONCURRENT background long requests, return the thread list."""
    threads = [
        threading.Thread(
            target=_send_long_request,
            args=(base_url, LONG_SUSTAINED_MAX_TOKENS),
        )
        for _ in range(LONG_CONCURRENT)
    ]
    for t in threads:
        t.start()
    return threads


def _run_short_requests(base_url: str, count: int):
    """Send `count` short requests concurrently, return the results dict."""
    results = {}
    threads = [
        threading.Thread(target=_send_short_request, args=(base_url, i, results))
        for i in range(1, count + 1)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


class TestNpuPrefillDelayerAboveThreshold(CustomTestCase):
    """Testcase: Send short requests exceeding queue_min so "waiting >= queue_min"
    holds and prefill is released promptly (no delay).

    [Test Category] Scheduling
    [Test Target] --enable-prefill-delayer; --prefill-delayer-queue-min-ratio;
                  --prefill-delayer-max-delay-ms
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_above_threshold_released_immediately(self):
        # 1. Fill up running with long requests.
        long_threads = _start_long_requests(self.base_url)

        # 2. Wait until long requests reach a stable decode window.
        stable_running = _wait_until_stable_running(
            self.base_url, expected_running=LONG_CONCURRENT
        )
        self.assertIsNotNone(
            stable_running,
            f"long requests failed to keep running stable (>= {LONG_CONCURRENT})",
        )
        _query_status(self.base_url, "before")

        # 3. Send short requests above the threshold.
        results = _run_short_requests(self.base_url, SHORT_CONCURRENT_ABOVE)
        all_ok = _print_short_results(results)

        # 4. Query status again.
        _query_status(self.base_url, "after")

        for t in long_threads:
            t.join()

        self.assertTrue(all_ok, "some short requests did not return HTTP 200")
        elapsed_values = [ms for ms, _ in results.values()]
        avg_ms = sum(elapsed_values) / len(elapsed_values)
        self.assertLess(
            avg_ms,
            MAX_DELAY_MS,
            f"avg latency {avg_ms:.0f}ms not below max_delay_ms({MAX_DELAY_MS}ms); "
            f"requests above threshold were still delayed",
        )


class TestNpuPrefillDelayerBelowThreshold(CustomTestCase):
    """Testcase: With waiting below the threshold and running kept high for the
    whole delay window, prefill is released only on the max_delay_ms timeout, so
    each short request must wait longer than max_delay_ms.

    [Test Category] Scheduling
    [Test Target] --prefill-delayer-max-delay-ms
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_below_threshold_hits_max_delay(self):
        # 1. Fill up running with long requests.
        long_threads = _start_long_requests(self.base_url)

        # 2. Wait until long requests reach a stable decode window.
        stable_running = _wait_until_stable_running(
            self.base_url, expected_running=LONG_CONCURRENT
        )
        self.assertIsNotNone(
            stable_running,
            f"long requests failed to keep running stable (>= {LONG_CONCURRENT})",
        )
        _query_status(self.base_url, "before")

        # 3. Send short requests far below the threshold.
        results = _run_short_requests(self.base_url, SHORT_CONCURRENT_BELOW)
        all_ok = _print_short_results(results)

        # 4. Query status again.
        _query_status(self.base_url, "after")

        for t in long_threads:
            t.join()

        self.assertTrue(all_ok, "some short requests did not return HTTP 200")
        for idx in sorted(results):
            elapsed_ms, _ = results[idx]
            self.assertGreater(
                elapsed_ms,
                MAX_DELAY_MS,
                f"request {idx} latency {elapsed_ms:.0f}ms not above "
                f"max_delay_ms({MAX_DELAY_MS}ms); delay released too early",
            )


if __name__ == "__main__":
    unittest.main()
