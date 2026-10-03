"""PDMux + HiCache end-to-end verification. Needs one CUDA GPU with green
context support (compute capability >= 8.0).

    python -m pytest test/manual/pdmux/test_pdmux_hicache.py -v

Covers both PDMux SM layouts, since they share `event_loop_pdmux` and differ
only in how the prefill and decode streams are masked:

  - exclusive green-context partitions (the default divide)
  - overlapped masks (`overlap_decode_full_sm`: decode runs on the full device)

The server is deliberately configured to make the HiCache paths run rather than
sit idle: a small device KV pool forces eviction to host, `write_through` backs
every finished request up, and a small `split_forward_token_budget` makes each
prefill split per layer so it spans many scheduler iterations.

The scheduler-loop property HiCache depends on -- that transfer acks are
drained exactly once per iteration, including the iterations a split prefill
occupies -- is pinned deterministically in
test/registered/unit/multiplex/test_pdmux_hicache_events.py. These tests check
the end-to-end consequences: correct output after a host reload, and acks that
keep draining under long overlapped prefills.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

LOAD_BACK_TOKENS = "sglang:load_back_tokens_total"
BACKUP_TOKENS = "sglang:hicache_backup_tokens_total"

# Long enough that a prefill splits per layer at the budget below, so a single
# request occupies many scheduler iterations alongside decode.
PROMPT_LEN = 2048
SPLIT_FORWARD_TOKEN_BUDGET = 512

# Device KV pool, in tokens. Small on purpose: several of these prompts do not
# fit at once, which is what pushes evicted pages to host.
MAX_TOTAL_TOKENS = 8192

# Bounds the eviction phase's runtime when the pool ends up larger than asked.
MAX_EVICTION_ROUNDS = 32

# Green-context partitions must be a multiple of 8 SMs to satisfy both the
# Ampere (min 4, multiple 2) and Hopper (min 8, multiple 8) constraints.
SM_GRANULARITY = 8


def sm_multiple(value: int) -> int:
    return max(SM_GRANULARITY, (value // SM_GRANULARITY) * SM_GRANULARITY)


def _write_pdmux_config(body: str) -> str:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", prefix="pdmux_hicache_", delete=False
    )
    with handle:
        handle.write(body)
    return handle.name


class PDMuxHiCacheMixin:
    """Server lifecycle plus the checks both SM layouts must pass.

    Plain mixin rather than a `CustomTestCase` base so the shared cases are not
    collected on their own -- they need a concrete layout's config to run.
    Subclasses override the class attributes to retarget a different model;
    `test_dsv4_pdmux_hicache_tp8.py` reuses the cases against DeepSeek V4.
    """

    sm_group_num = 4
    model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    max_total_tokens = MAX_TOTAL_TOKENS
    prompt_len = PROMPT_LEN
    # How PDMux submits the prefill. The default keeps this test on the
    # layer_split path it was written against; the *StandardPrefill subclasses
    # run the same checks against the standard-EXTEND lane.
    pdmux_prefill_mode = "layer_split"
    # Extra launch args and env, appended after the shared ones.
    extra_server_args: list = []
    server_env: dict = {}

    @classmethod
    def pdmux_mode_args(cls) -> list:
        if cls.pdmux_prefill_mode == "layer_split":
            return []
        # The standard lane rejects a prefill CUDA graph: that graph carries no
        # stream-group key, so its nodes keep the context they were captured in.
        return [
            "--pdmux-prefill-mode",
            cls.pdmux_prefill_mode,
            "--cuda-graph-backend-prefill",
            "disabled",
        ]

    @classmethod
    def pdmux_config_body(cls) -> str:
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.config_path = _write_pdmux_config(cls.pdmux_config_body())
        popen_kwargs = (
            {"env": {**os.environ, **cls.server_env}} if cls.server_env else {}
        )
        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-pdmux",
                "--disable-overlap-schedule",
                "--sm-group-num",
                str(cls.sm_group_num),
                "--pdmux-config-path",
                cls.config_path,
                "--enable-hierarchical-cache",
                "--hicache-write-policy",
                "write_through",
                # DSV4 rejects --hicache-size outright (hybrid_pool_assembler
                # raises); --hicache-ratio is the portable knob.
                "--hicache-ratio",
                "2",
                "--max-total-tokens",
                str(cls.max_total_tokens),
                "--mem-fraction-static",
                "0.7",
                # Both checks read Prometheus counters, which the server only
                # exports when metrics are on (default off).
                "--enable-metrics",
                *cls.pdmux_mode_args(),
                *cls.extra_server_args,
            ],
            **popen_kwargs,
        )
        # Size the eviction workload from the pool the server actually built:
        # page rounding and hybrid-SWA splits move it away from the requested
        # --max-total-tokens, and a hardcoded round count silently stops
        # evicting when it drifts.
        info = requests.get(cls.base_url + "/get_server_info", timeout=120).json()
        # A server that resolved to the other lane would validate nothing about
        # the mode this class is about, so check what actually took effect.
        if info["pdmux_prefill_mode"] != cls.pdmux_prefill_mode:
            raise AssertionError(
                f"server launched with pdmux_prefill_mode="
                f"{info['pdmux_prefill_mode']!r}, expected {cls.pdmux_prefill_mode!r}"
            )
        if not info["enable_metrics"]:
            raise AssertionError("server launched without metrics; counters unreadable")
        cls.pool_tokens = info["max_total_num_tokens"]
        cls.eviction_rounds = min(
            MAX_EVICTION_ROUNDS, cls.pool_tokens // cls.prompt_len + 2
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        os.unlink(cls.config_path)

    # --- helpers -----------------------------------------------------------

    def _generate(self, input_ids, max_new_tokens=8):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def _counter(self, name) -> float:
        """Sum a Prometheus counter across its per-pool label sets.

        An unmounted or failing /metrics must surface as an error: parsed as
        text it matches nothing and reads as zero, i.e. as "no progress".
        """
        response = requests.get(self.base_url + "/metrics", timeout=60)
        response.raise_for_status()
        metrics = response.text
        return sum(
            float(match.group(1))
            for match in re.finditer(
                rf"^{re.escape(name)}\{{[^}}]*\}}\s+([0-9.eE+-]+)$",
                metrics,
                re.MULTILINE,
            )
        )

    def _settled_counter(self, name, *, interval=1.0, attempts=30) -> float:
        """A quiet baseline: the counter after two equal reads a second apart.

        Acks from the previous case's traffic can retire after that case has
        returned; a baseline taken while they land would make "advanced while
        this case's requests ran" trivially true. Two equal reads show the
        counter is not moving at that moment, not that every earlier ack has
        retired -- a late one can still land after this returns.
        """
        value = self._counter(name)
        for _ in range(attempts):
            time.sleep(interval)
            current = self._counter(name)
            if current == value:
                return value
            value = current
        raise AssertionError(f"{name} kept moving with no requests in flight")

    def _flush(self):
        requests.post(self.base_url + "/flush_cache", timeout=120).raise_for_status()

    def _prompt(self, seed) -> list:
        base = 1000 + seed * self.prompt_len
        return list(range(base, base + self.prompt_len))

    # --- cases -------------------------------------------------------------

    def test_evicted_prefix_reloads_from_host_with_identical_output(self):
        """Greedy output must survive a round trip through the host pool.

        The prompt is prefilled, evicted to host by unrelated traffic, then
        requested again -- which loads its KV back layer by layer while the
        model is already reading those layers. A layer read before its
        host-to-device transfer landed would show up as output diverging from
        the cold reference, not as a loud failure.
        """
        self._flush()
        prompt = self._prompt(0)
        cold = self._generate(prompt)

        # Push distinct traffic through the device pool until the prompt's
        # pages are evicted; write_through already backed them up.
        for seed in range(1, self.eviction_rounds + 1):
            self._generate(self._prompt(seed), max_new_tokens=1)

        before = self._counter(LOAD_BACK_TOKENS)
        warm = self._generate(prompt)
        after = self._counter(LOAD_BACK_TOKENS)

        self.assertGreater(after, before, "prompt was not served from the host cache")
        self.assertEqual(warm["output_ids"], cold["output_ids"])

    def test_backups_advance_while_requests_are_running(self):
        """Backup acks retire while this case's requests are still running.

        A smoke check, not a timing proof. Request completion includes the
        decode tail, and PDMux can admit several of these prompts into one
        prefill batch, so an advance before the last completion does not show
        that acks retired while a prefill was in flight. That property is the
        TP8 timing item in standard_prefill_runbook.md, which correlates the
        counter with individual prefill work items. What this case does rule
        out is a counter that only moves once the server is idle again, which
        is what a scheduler that never pumps HiCache events between prefills
        would show.
        """
        self._flush()
        before = self._settled_counter(BACKUP_TOKENS)

        # Seeds well clear of the eviction phase's range, so these prompts are
        # genuinely cold rather than prefix hits from the other case.
        prompts = [self._prompt(seed) for seed in range(200, 206)]

        samples = []
        sampler_errors = []
        stop_sampling = threading.Event()

        def sample_counter():
            try:
                while not stop_sampling.is_set():
                    value = self._counter(BACKUP_TOKENS)
                    # Stamp after the read: the value is known to hold at this
                    # instant, so "before the last completion" is conservative.
                    samples.append((time.monotonic(), value))
                    stop_sampling.wait(0.2)
            except Exception as exc:
                # join() does not propagate this; re-raised on the test thread.
                sampler_errors.append(exc)

        sampler = threading.Thread(target=sample_counter, daemon=True)
        sampler.start()
        completion_times = []
        try:
            with ThreadPoolExecutor(max_workers=len(prompts)) as pool:

                def run(prompt):
                    result = self._generate(prompt, max_new_tokens=32)
                    completion_times.append(time.monotonic())
                    return result

                results = list(pool.map(run, prompts))
        finally:
            stop_sampling.set()
            sampler.join(timeout=60)
        self.assertFalse(sampler.is_alive(), "counter sampler did not exit")
        if sampler_errors:
            raise sampler_errors[0]

        after = self._counter(BACKUP_TOKENS)

        for result in results:
            self.assertEqual(result["meta_info"]["completion_tokens"], 32)
        self.assertGreater(
            after, before, "no device-to-host backup acks retired during the run"
        )
        first_progress = next((t for t, value in samples if value > before), None)
        self.assertIsNotNone(
            first_progress, "backup counter never advanced while requests ran"
        )
        self.assertLess(
            first_progress,
            max(completion_times),
            "backups only retired after every request had finished",
        )
        requests.get(self.base_url + "/health", timeout=60).raise_for_status()


class TestPDMuxExclusivePartitionsHiCache(PDMuxHiCacheMixin, CustomTestCase):
    """Default layout: prefill and decode own disjoint green-context SM sets."""

    @classmethod
    def pdmux_config_body(cls) -> str:
        return (
            f"sm_group_num: {cls.sm_group_num}\n"
            f"split_forward_token_budget: {SPLIT_FORWARD_TOKEN_BUDGET}\n"
        )


class TestPDMuxOverlappedMasksHiCache(PDMuxHiCacheMixin, CustomTestCase):
    """Overlapped layout: prefill stays capped, decode reaches every SM."""

    @classmethod
    def pdmux_config_body(cls) -> str:
        import torch

        total_sm = torch.cuda.get_device_properties(0).multi_processor_count
        # overlap_decode_full_sm requires manual_divisions, and each prefill cap
        # must sit strictly inside (0, total_sm). The decode column is ignored:
        # the loader rewrites it to the full device. The first threshold is 1 so
        # every non-empty decode batch selects a group.
        divisions = [
            (sm_multiple(total_sm // 4), 1),
            (sm_multiple(total_sm // 2), 8),
        ]
        entries = "".join(
            f"  - [{prefill_sm}, 0, {threshold}]\n"
            for prefill_sm, threshold in divisions
        )
        return (
            f"sm_group_num: {cls.sm_group_num}\n"
            f"split_forward_token_budget: {SPLIT_FORWARD_TOKEN_BUDGET}\n"
            "overlap_decode_full_sm: true\n"
            f"manual_divisions:\n{entries}"
        )


class TestPDMuxExclusivePartitionsHiCacheStandardPrefill(
    TestPDMuxExclusivePartitionsHiCache
):
    """Exclusive layout, prefill submitted as one standard EXTEND."""

    pdmux_prefill_mode = "standard"


class TestPDMuxOverlappedMasksHiCacheStandardPrefill(TestPDMuxOverlappedMasksHiCache):
    """Overlapped layout, prefill submitted as one standard EXTEND."""

    pdmux_prefill_mode = "standard"


if __name__ == "__main__":
    unittest.main()
