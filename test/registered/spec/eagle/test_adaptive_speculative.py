import json
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=700, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=1000, suite="stage-b-test-1-gpu-large-amd")

HIGH_ACCEPT_PROMPT = (
    "Output exactly 128 new lines. "
    "Every line must be READY. "
    "Do not add numbering, punctuation, or commentary."
)

LOW_ACCEPT_PROMPT = (
    "Compose a poem in the style of Emily Dickinson about quantum entanglement. "
    "Make it emotionally resonant and at least 100 words."
)

MAX_UPSHIFT_ATTEMPTS = 4
MAX_DOWNSHIFT_ATTEMPTS = 6


class _AdaptiveEagleServerBase(CustomTestCase):
    """Shared scaffold for the adaptive-EAGLE test servers: launch on the triton
    backend with adaptive speculative decoding + a temp config, and drive/observe
    tier switches. Subclasses set ``adaptive_config`` (defaults to the [1, 3] fast
    switcher) plus optional ``extra_launch_args`` / ``launch_env``."""

    model = DEFAULT_TARGET_MODEL_EAGLE
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE
    base_url = DEFAULT_URL_FOR_TEST

    adaptive_config = {
        "1": {
            "candidate_steps": [1, 3],
            "ema_alpha": 1.0,
            "warmup_batches": 1,
            "update_interval": 1,
            "up_hysteresis": 0.0,
        },
    }
    extra_launch_args = ()
    launch_env = None

    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cls.adaptive_config, f)
            cls.adaptive_config_path = f.name

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                env=cls.launch_env,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "triton",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-draft-model-path",
                    cls.draft_model,
                    "--speculative-adaptive",
                    "--speculative-adaptive-config",
                    cls.adaptive_config_path,
                    "--skip-server-warmup",
                    "--mem-fraction-static",
                    "0.7",
                    *cls.extra_launch_args,
                ],
            )
        except Exception:
            os.unlink(cls.adaptive_config_path)
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid, wait_timeout=60)
        if getattr(cls, "adaptive_config_path", None) and os.path.exists(
            cls.adaptive_config_path
        ):
            os.unlink(cls.adaptive_config_path)

    def _get_internal_state(self) -> dict:
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["internal_states"][0]

    def _steps(self) -> int:
        return self._get_internal_state()["speculative_num_steps"]

    def _scrape_metric(self, name: str, **label_filter) -> float | None:
        """Return the value of a Prometheus sample line, or None if absent.

        Matches a line whose metric name is exactly *name* (next char is '{'
        or whitespace) and whose labels include every key=value in
        *label_filter*.
        """
        text = requests.get(self.base_url + "/metrics", timeout=30).text
        for line in text.splitlines():
            if line.startswith("#") or not line.startswith(name):
                continue
            rest = line[len(name) :]
            if rest and rest[0] not in "{ ":
                continue
            if all(f'{k}="{v}"' in line for k, v in label_filter.items()):
                return float(line.rsplit(" ", 1)[1])
        return None

    def _generate(self, prompt: str, max_new_tokens: int = 64) -> dict:
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _drive_upshift(self) -> dict:
        """Send high-acceptance prompts until steps upshift to 3."""
        state = self._get_internal_state()
        for _ in range(MAX_UPSHIFT_ATTEMPTS):
            self._generate(HIGH_ACCEPT_PROMPT)
            state = self._get_internal_state()
            if state["speculative_num_steps"] == 3:
                return state
        return state

    def _drive_downshift(self) -> dict:
        """Send low-acceptance prompts until steps downshift to 1."""
        state = self._get_internal_state()
        for _ in range(MAX_DOWNSHIFT_ATTEMPTS):
            self._generate(LOW_ACCEPT_PROMPT)
            state = self._get_internal_state()
            if state["speculative_num_steps"] == 1:
                return state
        return state

    def _assert_adaptive_switches(self):
        """Drive the EMA up (high-accept) then down (low-accept); the active step
        count must move to 3 and back to 1."""
        self.assertEqual(
            self._drive_upshift()["speculative_num_steps"],
            3,
            "adaptive never upshifted to steps=3",
        )
        self.assertEqual(
            self._drive_downshift()["speculative_num_steps"],
            1,
            "adaptive never downshifted to steps=1",
        )


class TestAdaptiveSpeculativeServer(_AdaptiveEagleServerBase):
    """State switching + GSM8K accuracy, metric gauges, streaming losslessness,
    and abort under an active adaptive tier switch (config [1, 3])."""

    extra_launch_args = ("--enable-metrics",)

    def test_gsm8k_after_adaptive_switches(self):
        """Exercise up/down/up adaptive switches, then verify GSM8K accuracy."""
        state = self._drive_upshift()
        self.assertEqual(state["speculative_num_steps"], 3, f"Never upshifted: {state}")

        state = self._drive_downshift()
        self.assertEqual(
            state["speculative_num_steps"], 1, f"Never downshifted: {state}"
        )

        self._drive_upshift()

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=64,
        )
        metrics = run_eval(args)
        print(f"GSM8K after adaptive switches: {metrics}")
        self.assertGreater(metrics["score"], 0.20)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_accept_len = server_info["internal_states"][0]["avg_spec_accept_length"]
        print(f"avg_spec_accept_length={avg_accept_len:.4f}")

    def test_adaptive_metrics_exposed(self):
        """After an upshift, the adaptive current-state gauges are scrapeable."""
        state = self._drive_upshift()
        self.assertEqual(state["speculative_num_steps"], 3, f"Never upshifted: {state}")
        # One more decode so the reporter emits a fresh logging interval.
        self._generate(HIGH_ACCEPT_PROMPT)

        steps = self._scrape_metric("sglang:spec_num_steps")
        draft_tokens = self._scrape_metric("sglang:spec_num_draft_tokens")

        self.assertIn(steps, {1.0, 3.0}, "spec_num_steps gauge has unexpected value")
        self.assertIn(
            draft_tokens,
            {2.0, 4.0},
            "spec_num_draft_tokens gauge has unexpected value",
        )

    def test_streaming_lossless_across_switch(self):
        """Streamed greedy output is identical before vs after an adaptive tier
        switch (spec decoding is lossless, so num_steps must not change the
        result) and arrives in multiple SSE chunks. Guards streaming-path or
        runtime-state-swap corruption that a non-streaming test would miss."""
        prompt = "List the numbers from 1 to 40 separated by commas:"

        def stream_generate():
            chunks, last_text = 0, ""
            with requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 48,
                        "ignore_eos": True,
                    },
                    "stream": True,
                },
                stream=True,
                timeout=180,
            ) as resp:
                if resp.status_code != 200:
                    self.fail(
                        f"stream request failed ({resp.status_code}): {resp.text}"
                    )
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data:"):
                        continue
                    payload = raw[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    last_text = json.loads(payload).get("text", last_text)
                    chunks += 1
            return chunks, last_text

        self._drive_downshift()
        self.assertEqual(self._get_internal_state()["speculative_num_steps"], 1)
        n_low, text_low = stream_generate()
        self.assertGreater(n_low, 1, "response did not stream incrementally")

        self._drive_upshift()
        self.assertEqual(self._get_internal_state()["speculative_num_steps"], 3)
        n_high, text_high = stream_generate()
        self.assertGreater(n_high, 1, "response did not stream incrementally")

        self.assertEqual(
            text_low,
            text_high,
            "streamed greedy output changed across an adaptive tier switch "
            "(spec decoding must be lossless)",
        )

    def test_abort_around_switch(self):
        """abort_all with requests in flight under adaptive aborts them cleanly
        and leaves the server alive and serving. Guards a hang or crash when the
        abort path races the tier machinery."""
        self._drive_upshift()  # land at steps=3 so the tier machinery is active

        def long_decode():
            return requests.post(
                self.base_url + "/generate",
                json={
                    "text": "Write a very long story about a magical kingdom.",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4000,
                        "ignore_eos": True,
                    },
                },
                timeout=120,
            ).json()

        with ThreadPoolExecutor(8) as executor:
            futures = [executor.submit(long_decode) for _ in range(8)]
            time.sleep(2)
            requests.post(
                self.base_url + "/abort_request",
                json={"abort_all": True},
                timeout=10,
            ).raise_for_status()
            for future in as_completed(futures):
                meta = future.result()["meta_info"]
                self.assertEqual(meta["finish_reason"]["type"], "abort")

        self.assertIsNone(self.process.poll(), "server died after abort under adaptive")
        # The server still serves correctly after the abort storm.
        self.assertIn("text", self._generate("The capital of France is"))


class TestAdaptiveZeroStepBatchSizeServer(_AdaptiveEagleServerBase):
    """steps=0 (nospec) fallback triggered by batch size, plus logprob and grammar
    correctness across the steps=3 <-> steps=0 boundary.

    Config routes BS>=8 -> steps=0 (drafting disabled) and BS<8 -> steps=3, so the
    server cycles steps=3 -> steps=0 -> steps=3 as load rises and falls.
    """

    adaptive_config = {
        "1": {"candidate_steps": [3], "warmup_batches": 0},
        "8": {"candidate_steps": [0], "warmup_batches": 0},
    }
    extra_launch_args = (
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--max-running-requests",
        "32",
    )

    COUNT_PROMPT = "Count from 1 to 400, separated by commas. Output only the numbers."

    def test_batch_size_step_cycle(self):
        """The server cycles steps=3 -> steps=0 -> steps=3 as load rises and falls:
        a BS=1 request drafts at steps=3; a 14-way batch (BS>=8) routes the worker
        to nospec steps=0; a following BS=1 request returns to steps=3 with drafting
        restored (high accept rate again)."""
        one = {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True}

        def generate_single() -> dict:
            r = requests.post(
                self.base_url + "/generate",
                json={"text": self.COUNT_PROMPT, "sampling_params": one},
                timeout=600,
            )
            self.assertEqual(r.status_code, 200, r.text)
            return r.json()["meta_info"]

        # Phase 1: BS=1 -> steps=3, drafting active.
        m1 = generate_single()
        self.assertEqual(self._steps(), 3, "expected steps=3 at BS=1")
        self.assertGreater(
            m1["spec_accept_rate"], 0.8, f"not drafting at steps=3: {m1}"
        )

        # Phase 2: BS=14 -> the worker switches to nospec steps=0. Equal-length
        # requests finish together, so the last decode batch (and thus the state)
        # is at BS=14 -> steps=0.
        full = {"temperature": 0, "max_new_tokens": 128, "ignore_eos": True}
        r = requests.post(
            self.base_url + "/generate",
            json={"text": [self.COUNT_PROMPT] * 14, "sampling_params": [full] * 14},
            timeout=600,
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self._steps(), 0, "BS>=8 did not switch to steps=0")

        # Phase 3: BS=1 -> steps=3 again, drafting restored.
        m3 = generate_single()
        self.assertEqual(self._steps(), 3, "did not reopen to steps=3")
        self.assertGreater(
            m3["spec_accept_rate"], 0.8, f"drafting not restored after steps=0: {m3}"
        )

    def test_logprob_count_across_step_boundary(self):
        """output_token_logprobs count must equal completion_tokens on BOTH the
        spec-verify path (steps=3) and the trivial no-spec path (steps=0). A
        wrong-count regression on either verify path is silent without this."""

        def gen(texts, params):
            r = requests.post(
                self.base_url + "/generate",
                json={
                    "text": texts,
                    "sampling_params": params,
                    "return_logprob": True,
                    "logprob_start_len": 0,
                },
                timeout=600,
            )
            self.assertEqual(r.status_code, 200, r.text)
            return r.json()

        sp = {"temperature": 0, "max_new_tokens": 32, "ignore_eos": True}

        # steps=3: BS=1, drafting active.
        single = gen(self.COUNT_PROMPT, sp)
        self.assertEqual(self._steps(), 3, "expected steps=3 at BS=1")
        meta = single["meta_info"]
        self.assertGreater(meta.get("spec_verify_ct", 0), 0)
        self.assertEqual(len(meta["output_token_logprobs"]), meta["completion_tokens"])

        # steps=0: a BS>=8 batch routes the worker to nospec.
        batch = gen([self.COUNT_PROMPT] * 12, [sp] * 12)
        self.assertEqual(self._steps(), 0, "BS>=8 did not route to steps=0")
        for res in batch:
            m = res["meta_info"]
            self.assertEqual(len(m["output_token_logprobs"]), m["completion_tokens"])

    def test_grammar_json_across_step_boundary(self):
        """json_schema output stays valid JSON with no trailing tokens on BOTH
        steps=3 (spec) and steps=0 (no-spec) -- grammar termination must hold
        whichever verify path the adaptive tier selects."""
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "pattern": "^[\\w ]+$"},
                    "population": {"type": "integer"},
                },
                "required": ["city", "population"],
            }
        )
        prompt = "Return the capital of France and its population as JSON.\n"
        sp = {"temperature": 0, "max_new_tokens": 64, "json_schema": schema}

        def check(res):
            text = res["text"]
            parsed = json.loads(text)
            self.assertIn("city", parsed)
            self.assertIn("population", parsed)
            self.assertTrue(
                text.strip().endswith("}"), f"trailing tokens past grammar: {text!r}"
            )

        # steps=3: BS=1.
        single = requests.post(
            self.base_url + "/generate",
            json={"text": prompt, "sampling_params": sp},
            timeout=600,
        )
        self.assertEqual(single.status_code, 200, single.text)
        self.assertEqual(self._steps(), 3, "expected steps=3 at BS=1")
        check(single.json())

        # steps=0: BS>=8 batch.
        batch = requests.post(
            self.base_url + "/generate",
            json={"text": [prompt] * 12, "sampling_params": [sp] * 12},
            timeout=600,
        )
        self.assertEqual(batch.status_code, 200, batch.text)
        self.assertEqual(self._steps(), 0, "BS>=8 did not route to steps=0")
        for res in batch.json():
            check(res)


class TestAdaptiveDisableOverlapSwitch(_AdaptiveEagleServerBase):
    """Spec V1 (overlap scheduler off). The accepted-draft counts feed the EMA
    synchronously instead of through the deferred overlap path, so adaptive
    switching must still work end-to-end with --disable-overlap-schedule."""

    extra_launch_args = ("--disable-overlap-schedule",)

    def test_switch_under_disable_overlap(self):
        """Adaptive still up/down-shifts with the overlap scheduler off, where
        accept counts feed the EMA synchronously rather than via the deferred
        overlap path."""
        self._assert_adaptive_switches()


class TestAdaptiveEagerSwitch(_AdaptiveEagleServerBase):
    """Eager mode (cuda graphs disabled). The tier swap has no per-tier cuda graph
    runners to re-point, so adaptive switching must still work with
    --disable-cuda-graph."""

    extra_launch_args = ("--disable-cuda-graph",)

    def test_switch_eager(self):
        """Adaptive still up/down-shifts in eager mode, where the tier swap has
        no per-tier cuda-graph runners to re-point."""
        self._assert_adaptive_switches()


class TestAdaptiveRetract(_AdaptiveEagleServerBase):
    """Retract under a tiny KV budget while adaptive is active. Retract and the
    adaptive tier machinery both mutate KV / runtime state; under load the server
    must retract and recover without leaking KV or crashing (the strict-mem check
    asserts no pool leak), and every request must still finish cleanly."""

    extra_launch_args = (
        "--max-running-requests",
        "48",
        "--max-total-tokens",
        "4500",
    )
    launch_env = {
        "SGLANG_TEST_RETRACT": "1",
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY": "1",
    }

    def test_retract_under_adaptive(self):
        def long_decode():
            return requests.post(
                self.base_url + "/generate",
                json={
                    "text": "Count slowly from 1 to 300, one number per line.",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 256,
                        "ignore_eos": True,
                    },
                },
                timeout=600,
            ).json()

        with ThreadPoolExecutor(24) as executor:
            results = [
                future.result()
                for future in as_completed(
                    [executor.submit(long_decode) for _ in range(24)]
                )
            ]

        for res in results:
            self.assertIn("text", res, f"server error under retract: {res}")
            self.assertEqual(
                res["meta_info"]["finish_reason"]["type"],
                "length",
                f"request did not finish cleanly under retract: {res['meta_info']}",
            )
        self.assertIsNone(self.process.poll(), "server died under retract + adaptive")


if __name__ == "__main__":
    unittest.main()
