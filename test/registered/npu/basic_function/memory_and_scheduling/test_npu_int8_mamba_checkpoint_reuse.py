"""Verify --enable-int8-mamba-checkpoint doubles cacheable prefix capacity on NPU."""

import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

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

# Prefixes must cross the mamba chunk granularity (~512), otherwise they cannot
# be cached and reuse stays 0 (note from PR 28185).
PREFIX_TOKENS = 768
SUFFIX_TOKENS = 16
# Number of active bf16 pool slots; the int8 checkpoint pool defaults to 2x this.
MAX_MAMBA_CACHE_SIZE = 128
# Distinct prefix counts to sweep; the reuse collapse point should fall in this
# range (off ~128, on ~256).
K_SCAN = [64, 128, 256, 512]
PARALLEL = 8
# reuse below this threshold is considered "collapsed" (cache overflowed at K).
COLLAPSE_THRESHOLD = 0.5
# The maximum reuse gain of int8 over bf16 must exceed this value.
REUSE_GAP = 0.3


def _random_ids(length, seed):
    rng = random.Random(seed)
    return [rng.randint(1, 30000) for _ in range(length)]


def _make_requests():
    """Pre-generate K groups of WARM/PROBE requests (deterministic, independent
    of the int8 switch).

    Both measurements use the same requests so the off/on comparison is fair.
    WARM: each prefix plus a dedicated suffix, occupying its own cached state
    slot. PROBE: the same prefix plus a different suffix, hitting only the prefix
    itself.
    """
    plan = {}
    for K in K_SCAN:
        # Each prefix carries a unique tag head + a long random segment so they
        # are mutually distinct.
        prefixes = [
            _random_ids(1, seed=1000 + i) + _random_ids(PREFIX_TOKENS, seed=2000 + i)
            for i in range(K)
        ]
        warm = [
            p + _random_ids(SUFFIX_TOKENS, seed=3000 + i)
            for i, p in enumerate(prefixes)
        ]
        probe = [
            p + _random_ids(SUFFIX_TOKENS, seed=4000 + i)
            for i, p in enumerate(prefixes)
        ]
        plan[K] = (warm, probe)
    return plan


def _generate(input_ids):
    resp = requests.post(
        BASE_URL + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def _flush():
    try:
        requests.post(BASE_URL + "/flush_cache", timeout=60)
    except requests.RequestException:
        pass
    time.sleep(1.5)


def _measure_config(int8_enabled, plan):
    """Launch a server, sweep K, and return {K: reuse_frac}."""
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
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--max-mamba-cache-size",
        str(MAX_MAMBA_CACHE_SIZE),
        "--mamba-ssm-dtype",
        "bfloat16",
    ]
    if int8_enabled:
        other_args.append("--enable-int8-mamba-checkpoint")

    proc = popen_launch_server(
        MODEL,
        BASE_URL,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        env=NPU_ENV,
        device="npu",
    )
    try:
        results = {}
        for K, (warm, probe) in plan.items():
            _flush()
            with ThreadPoolExecutor(PARALLEL) as ex:
                list(ex.map(_generate, warm))  # WARM
            with ThreadPoolExecutor(PARALLEL) as ex:
                metas = [r["meta_info"] for r in ex.map(_generate, probe)]

            sum_prompt = sum(m["prompt_tokens"] for m in metas)
            sum_cached = sum(m["cached_tokens"] for m in metas)
            reuse = sum_cached / max(1, sum_prompt)
            results[K] = reuse
            print(
                f"[int8={'on' if int8_enabled else 'off'}] K={K}: reuse_frac={reuse:.3f}"
            )
        return results
    finally:
        terminate_and_kill_process_tree(proc, terminate_timeout=60)
        time.sleep(2)


def _collapse_point(results):
    """The first K where reuse drops below the threshold; if none collapsed,
    return a sentinel larger than the largest K."""
    for K in K_SCAN:
        if results[K] < COLLAPSE_THRESHOLD:
            return K
    return K_SCAN[-1] * 2


class TestNPUInt8MambaCheckpointReuse(CustomTestCase):
    """Testcase: Verify the int8 checkpoint pool roughly doubles cacheable prefix
    capacity and pushes the reuse collapse point farther out on Ascend NPU.

    [Test Category] Memory and Scheduling
    [Test Target] --enable-int8-mamba-checkpoint
    """

    def test_int8_delays_prefix_reuse_collapse(self):
        plan = _make_requests()
        off = _measure_config(int8_enabled=False, plan=plan)
        on = _measure_config(int8_enabled=True, plan=plan)

        print(f"off={off}")
        print(f"on ={on}")
        print(f"collapse point: off={_collapse_point(off)}, on={_collapse_point(on)}")

        # 1) Within each config reuse is non-increasing in K (capacity cap).
        for results, tag in ((off, "off"), (on, "on")):
            for a, b in zip(K_SCAN, K_SCAN[1:]):
                self.assertGreaterEqual(
                    results[a] + 1e-6,
                    results[b],
                    f"[{tag}] reuse should be non-increasing in K "
                    f"({results[a]:.3f}@{a} -> {results[b]:.3f}@{b})",
                )

        # 2) The int8 collapse point is not earlier than bf16 (collapses later).
        self.assertGreaterEqual(
            _collapse_point(on),
            _collapse_point(off),
            "int8 checkpoint pool should not collapse earlier than bf16",
        )

        # 3) Strong assertion: take the maximum reuse gain of int8 over bf16
        #    across the K sweep; it must be significant. This avoids pinning a
        #    fixed K (collapse location drifts with per-prefix slot usage).
        diffs = {K: on[K] - off[K] for K in K_SCAN}
        max_k = max(diffs, key=lambda k: diffs[k])
        max_diff = diffs[max_k]
        print(f"max reuse gain: {max_diff:.3f} at K={max_k}")
        self.assertGreater(
            max_diff,
            REUSE_GAP,
            f"expected int8 to keep reuse higher than off at some K by > {REUSE_GAP}, "
            f"max gain {max_diff:.3f} @K={max_k}",
        )


if __name__ == "__main__":
    unittest.main()
