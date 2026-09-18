"""Verify --mamba-max-states-per-path evicts cached mamba states on Ascend NPU."""

import os
import re
import time
import unittest

import requests

from sglang.test.ascend.test_ascend_utils import QWEN3_5_27B_MODEL_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=3600, suite="full-2-npu-a3", nightly=True)

MODEL = QWEN3_5_27B_MODEL_WEIGHTS_PATH
BASE_URL = DEFAULT_URL_FOR_TEST
METRIC = "sglang:mamba_evictable_tokens"
# NPU runtime environment variables consistent with the local script.
NPU_ENV = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "SGLANG_DEEPEP_BF16_DISPATCH": "1",
    "ENABLE_ASCEND_MOE_NZ": "1",
}
# Optional: specify the starting NPU card for local manual runs (not set in CI,
# delegated to the scheduler).
BASE_GPU_ID = os.environ.get("BASE_GPU_ID")

# Words "added" per turn. mamba state is only donated when crossing a track
# boundary (seq_len % 256 == 0) (see
# batch_result_processor._mamba_prefix_cache_update). Adding ~500 words (~600
# tokens) per turn crosses several 256 boundaries, guaranteeing multiple
# evictable states accumulate on the path. If each turn is shorter than 256 and
# never crosses a boundary, no state is ever persisted and evictable stays 0.
PER_TURN_ADDED_WORDS = 500
NUM_TURNS = 6
# Output tokens generated per turn: slightly longer so the decode stage also
# crosses boundaries and assists donation.
MAX_NEW_TOKENS = 32
# cap values to sweep: 1 (strictest, keeps only the tail) -> 3 -> 64 (almost no
# eviction, used as the upper-bound reference).
CAPS = [1, 3, 64]

_BASE_SEGMENT = (
    "The quick brown fox jumps over the lazy dog and then runs across the "
    "sunny meadow while the birds sing in the tall green trees. "
)


def _build_shared_prefix_turns(
    num_turns=NUM_TURNS, per_turn_added_words=PER_TURN_ADDED_WORDS
):
    """Build multi-round shared prefixes, each round appending to the previous.

    Returns [T1, T2, ..., Tn] where Ti is a strict superset of T(i-1): Ti's word
    count = i * per_turn_added_words. This makes each round append a new node
    after the previously cached node (rather than repeating the same text), and
    keeps accumulating mamba state on the same radix path.
    """
    turns = []
    prefix = ""
    for i in range(1, num_turns + 1):
        while len(prefix.split()) < i * per_turn_added_words:
            prefix += _BASE_SEGMENT
        turns.append(prefix)
    return turns


def _wait_metrics_ready(base_url, timeout=120):
    """Confirm /metrics is truly accessible before entering the load phase.

    If 200 is never returned, --enable-metrics is not in effect (or an old server
    without metrics was hit); raise an actionable error instead of masking the
    issue until after the load.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(base_url + "/metrics", timeout=5)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass	
        time.sleep(2)
    raise AssertionError(
        f"/metrics kept returning non-200 (port {base_url}). Please check: "
        "(1) whether --enable-metrics was passed; (2) whether the port is "
        "occupied by a leftover server without metrics enabled (try pkill sglang "
        "first or retry with a different port)."
    )


def _get_metric_lines(base_url, name, timeout=10):
    """Pull all raw lines for a metric from /metrics.

    Under TP2 the same gauge is split across tp_rank/pp_rank/moe_ep_rank into
    multiple lines (e.g. sglang:mamba_evictable_tokens{tp_rank="0",...} 0.0 and
    a tp_rank="1" line). Returns (list of values, full text).
    """
    resp = requests.get(base_url + "/metrics", timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    values = []
    for line in text.splitlines():
        if line.startswith(name):
            m = re.match(rf"{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.e+-]+)", line)
            if m:
                values.append(float(m.group(1)))
    return values, text


def _get_metric_value(base_url, name, timeout=10):
    """Pull the cross-rank summed value of a gauge from /metrics.

    Under TP2 the mamba state may only land on some ranks, reading only the first
    line would wrongly give 0; sum all same-name rank lines to keep the reading
    stable (proportional consistency across cap comparisons).
    """
    values, _ = _get_metric_lines(base_url, name, timeout=timeout)
    if not values:
        raise AssertionError(f"metric '{name}' not found in /metrics")
    return sum(values)


def _dump_mamba_metrics(base_url, cap):
    """Print all mamba-related raw metrics to help locate why some rank is 0."""
    for name in (
        "sglang:mamba_evictable_tokens",
        "sglang:mamba_available_tokens",
        "sglang:mamba_used_tokens",
    ):
        values, text = _get_metric_lines(base_url, name)
        lines = [l for l in text.splitlines() if l.startswith(name)]
        print(f"[cap={cap}] {name}  (sum={sum(values)})")
        for l in lines:
            print(f"        {l}")


def _wait_metric_stable(base_url, name, stable_reads=2, poll_interval=2.0, timeout=30):
    """Poll until two consecutive readings are equal, returning that stable
    value (waits for the periodic pool stats refresh)."""
    last, stable = None, 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            cur = _get_metric_value(base_url, name)
        except AssertionError:
            cur = None
        if cur == last and cur is not None:
            stable += 1
            if stable >= stable_reads:
                return cur
        else:
            stable = 0
        last = cur
        time.sleep(poll_interval)
    return last


class TestNPUMambaPathStateCapEvictable(CustomTestCase):
    """Testcase: Verify --mamba-max-states-per-path evicts cached mamba states
    (the evictable metric is non-decreasing in cap) on Ascend NPU.

    [Test Category] HiCache
    [Test Target] --mamba-max-states-per-path
    """

    def _send_workload(self, base_url):
        """Send multiple rounds of shared-prefix requests, returning after all complete."""
        for turn in _build_shared_prefix_turns():
            requests.post(
                base_url + "/generate",
                json={
                    "text": turn,
                    "sampling_params": {
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "temperature": 0,
                    },
                },
                timeout=180,
            ).raise_for_status()

    def _measure_evictable_tokens(self, cap):
        """Launch a server with the given cap on a fixed port, run the same
        workload, and return the idle evictable tokens.

        In-container ports are isolated by the network namespace, so a fixed port
        is fine; _wait_metrics_ready still fails fast when a leftover server
        occupies the port (/generate ok but /metrics 404).
        """
        other_args = [
            "--trust-remote-code",
            "--device",
            "npu",
            "--tp-size",
            "2",
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            "0.8",
            "--enable-metrics",
            "--mamba-max-states-per-path",
            str(cap),
        ]
        if BASE_GPU_ID is not None:
            other_args += ["--base-gpu-id", BASE_GPU_ID]

        process = popen_launch_server(
            MODEL,
            BASE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=NPU_ENV,
            device="npu",
        )
        try:
            # Confirm /metrics is truly available first (excluding the case where
            # the port is occupied by an old server), then run the workload so
            # the issue is not masked until after the load.
            _wait_metrics_ready(BASE_URL)
            self._send_workload(BASE_URL)
            # After all requests complete, pool stats refresh asynchronously; wait
            # for the metric to stabilize.
            stable = _wait_metric_stable(BASE_URL, METRIC)
            _dump_mamba_metrics(BASE_URL, cap)  # print per-rank raw values for diagnosis
            return stable
        finally:
            terminate_and_kill_process_tree(process, terminate_timeout=60)
            time.sleep(2)

    def test_evictable_tokens_monotonic_in_cap(self):
        """The smaller the cap, the fewer cached mamba states remain on the
        path, so the evictable metric gets smaller.

        Under the same workload the metric should be non-decreasing in cap
        (strictly increasing is ideal). If any cap makes evictable larger
        instead, the eviction path is not working as expected.
        """
        values = {}
        for cap in CAPS:
            values[cap] = self._measure_evictable_tokens(cap)
            print(f"cap={cap} -> {METRIC}={values[cap]}")

        # Monotonicity: as cap increases, evictable must not decrease.
        for a, b in zip(CAPS, CAPS[1:]):
            self.assertGreaterEqual(
                values[b],
                values[a],
                f"cap={b} should retain >= cap={a} cached states "
                f"({values[a]} -> {values[b]}), eviction did not take effect as expected",
            )

        # Strongest assertion: cap=1 must be strictly smaller than cap=64
        # (eviction really happened).
        self.assertLess(
            values[CAPS[0]],
            values[CAPS[-1]],
            f"cap=1({values[CAPS[0]]}) should be significantly smaller than "
            f"cap={CAPS[-1]}({values[CAPS[-1]]}), proving path-cap eviction works",
        )


if __name__ == "__main__":
    unittest.main()