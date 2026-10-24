"""Verify --mamba-max-states-per-path evicts cached mamba states on Ascend NPU."""

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

# A disjoint base segment used to form a second, independent radix path for the
# multi-path independence check.
_BASE_SEGMENT_B = (
    "Quantum mechanics governs the behavior of matter and energy at atomic "
    "scales, where classical physics no longer gives accurate predictions. "
)


def _build_shared_prefix_turns(
    num_turns=NUM_TURNS,
    per_turn_added_words=PER_TURN_ADDED_WORDS,
    base_segment=_BASE_SEGMENT,
):
    """Build multi-round shared prefixes, each round appending to the previous.

    Returns [T1, T2, ..., Tn] where Ti is a strict superset of T(i-1): Ti's word
    count = i * per_turn_added_words. This makes each round append a new node
    after the previously cached node (rather than repeating the same text), and
    keeps accumulating mamba state on the same radix path. A different
    ``base_segment`` yields a disjoint radix path for independence checks.
    """
    turns = []
    prefix = ""
    for i in range(1, num_turns + 1):
        while len(prefix.split()) < i * per_turn_added_words:
            prefix += base_segment
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
    if last is None:
        raise AssertionError(
            f"metric '{name}' never appeared in /metrics within {timeout}s"
        )
    return last


class TestNPUMambaPathStateCapEvictable(CustomTestCase):
    """Testcase: Verify --mamba-max-states-per-path evicts cached mamba states
    (the evictable metric is non-decreasing in cap) on Ascend NPU.

    [Test Category] Parameter
    [Test Target] --mamba-max-states-per-path
    """

    def _launch_server(self, cap):
        """Launch a server with the given per-path mamba cap on a fixed port."""
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
            "--mamba-ssm-dtype",
            "bfloat16",
        ]
        return popen_launch_server(
            MODEL,
            BASE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=NPU_ENV,
            device="npu",
        )

    def _send_workload(self, base_url, turns=None):
        """Send multi-round shared-prefix requests, returning per-round results.

        Each entry keeps the generated ``text`` plus the KV cache hit
        (``cached_tokens``/``prompt_tokens``) so callers can verify output
        stability and that full KV survives mamba eviction.
        """
        if turns is None:
            turns = _build_shared_prefix_turns()
        results = []
        for turn in turns:
            resp = requests.post(
                base_url + "/generate",
                json={
                    "text": turn,
                    "sampling_params": {
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "temperature": 0,
                    },
                },
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            meta = data.get("meta_info", {})
            results.append(
                {
                    "text": data.get("text", ""),
                    "cached_tokens": int(meta.get("cached_tokens", 0)),
                    "prompt_tokens": int(meta.get("prompt_tokens", 0)),
                }
            )
        return results

    def _measure_evictable_tokens(self, cap):
        """Launch a server with the given cap, run the workload, and snapshot the
        stable mamba pool metrics plus per-round results.

        In-container ports are isolated by the network namespace, so a fixed port
        is fine; _wait_metrics_ready still fails fast when a leftover server
        occupies the port (/generate ok but /metrics 404).
        """
        process = self._launch_server(cap)
        try:
            # Confirm /metrics is truly available first (excluding the case where
            # the port is occupied by an old server), then run the workload so
            # the issue is not masked until after the load.
            _wait_metrics_ready(BASE_URL)
            results = self._send_workload(BASE_URL)
            # After all requests complete, pool stats refresh asynchronously; wait
            # for the metrics to stabilize.
            evictable = _wait_metric_stable(BASE_URL, METRIC)
            available = _wait_metric_stable(BASE_URL, "sglang:mamba_available_tokens")
            _dump_mamba_metrics(
                BASE_URL, cap
            )  # print per-rank raw values for diagnosis
            return {"evictable": evictable, "available": available, "results": results}
        finally:
            terminate_and_kill_process_tree(process, terminate_timeout=60)
            time.sleep(2)

    def test_evictable_tokens_monotonic_in_cap(self):
        """The smaller the cap, the fewer cached mamba states remain on the
        path, so the evictable metric gets smaller.

        Beyond monotonicity this also checks functional correctness: greedy
        outputs must be identical across caps (eviction only affects the cache
        hit path), full KV must survive mamba eviction, and the available gauge
        must shrink as more states are retained.
        """
        snapshots = {}
        for cap in CAPS:
            snapshots[cap] = self._measure_evictable_tokens(cap)
            print(
                f"cap={cap} -> evictable={snapshots[cap]['evictable']}, "
                f"available={snapshots[cap]['available']}"
            )

        evictable = {cap: snapshots[cap]["evictable"] for cap in CAPS}
        available = {cap: snapshots[cap]["available"] for cap in CAPS}

        # Functional correctness: temperature=0 means greedy decoding, so every
        # cap must emit exactly the same tokens. Eviction only changes hit-path
        # lookups, never the sampled output.
        outputs = {cap: [r["text"] for r in snapshots[cap]["results"]] for cap in CAPS}
        for a, b in zip(CAPS, CAPS[1:]):
            self.assertEqual(
                outputs[a],
                outputs[b],
                f"greedy outputs diverge between cap={a} and cap={b}",
            )

        # Monotonicity: as cap increases, evictable must not decrease.
        for a, b in zip(CAPS, CAPS[1:]):
            self.assertGreaterEqual(
                evictable[b],
                evictable[a],
                f"cap={b} should retain >= cap={a} cached states "
                f"({evictable[a]} -> {evictable[b]}), eviction did not take effect as expected",
            )

        # Strongest assertion: cap=1 must be strictly smaller than cap=64
        # (eviction really happened).
        self.assertLess(
            evictable[CAPS[0]],
            evictable[CAPS[-1]],
            f"cap=1({evictable[CAPS[0]]}) should be significantly smaller than "
            f"cap={CAPS[-1]}({evictable[CAPS[-1]]}), proving path-cap eviction works",
        )

        # Available must be non-increasing in cap: retaining more mamba states
        # consumes more free slots. Equal readings are tolerated for small pools.
        for a, b in zip(CAPS, CAPS[1:]):
            self.assertLessEqual(
                available[b],
                available[a],
                f"cap={b} should not free more slots than cap={a} "
                f"({available[a]} -> {available[b]})",
            )

        # Full KV survives mamba eviction: at the strictest cap, rounds 2..6
        # still hit (nearly) the whole previous prompt in KV. If full KV were
        # evicted alongside mamba state, cached_tokens would collapse.
        cap1_results = snapshots[CAPS[0]]["results"]
        for i in range(1, len(cap1_results)):
            prev_prompt = cap1_results[i - 1]["prompt_tokens"]
            cached = cap1_results[i]["cached_tokens"]
            self.assertGreaterEqual(
                cached,
                prev_prompt * 0.9,
                f"round {i + 1}: full KV was evicted along with mamba state "
                f"(cached={cached}, previous prompt={prev_prompt})",
            )

    def test_multipath_independence(self):
        """Evicting one radix path must not evict a disjoint path's mamba states.

        cap is per-path: driving path A over its cap should only trim A, leaving
        path B's accumulated states intact. Cross-path leakage is detected by
        measuring evictable after A alone and after A+B; B's join must add fresh
        evictable slots rather than being silently absorbed by A's cap.
        """
        cap = 1  # strictest cap, so any cross-path leakage is most visible
        turns_a = _build_shared_prefix_turns()
        turns_b = _build_shared_prefix_turns(base_segment=_BASE_SEGMENT_B)

        process = self._launch_server(cap)
        try:
            _wait_metrics_ready(BASE_URL)

            self._send_workload(BASE_URL, turns=turns_a)
            evictable_a = _wait_metric_stable(BASE_URL, METRIC)

            self._send_workload(BASE_URL, turns=turns_b)
            evictable_ab = _wait_metric_stable(BASE_URL, METRIC)

            _dump_mamba_metrics(BASE_URL, cap)
            self.assertGreater(
                evictable_ab,
                evictable_a,
                "path B's mamba states were not retained after path A hit its cap "
                f"(A={evictable_a}, A+B={evictable_ab}); per-path eviction is leaking across paths",
            )
        finally:
            terminate_and_kill_process_tree(process, terminate_timeout=60)
            time.sleep(2)


if __name__ == "__main__":
    unittest.main()
