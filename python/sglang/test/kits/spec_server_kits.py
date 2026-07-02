"""Reusable test-method mixins (kits) for EAGLE/EAGLE3 spec-decoding servers.

Pair these with ``SpecEagleServerBase`` (sglang.test.server_fixtures.spec_eagle_fixture).
Each kit is a cohesive group of ``test_*`` methods with no launch logic; concrete
test classes mix in the fixture (which owns launch knobs) + whichever kits apply.

Thresholds are read off ``self`` so a config can tune them as class attributes.
"""

import concurrent.futures
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

import numpy as np
import requests

from sglang.srt.utils.common import kill_process_tree
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    run_logprob_check,
)


class SpecCorrectnessKit:
    """Acceptance-quality + EOS checks (single server, cheap)."""

    # Tunable thresholds (override per config class).
    acc_length_thres = 3.1
    batch_accept_len_thres = 1.75

    def test_acc_length(self):
        prompt = [
            "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:",
        ] * 5
        sampling_params = {"temperature": 0, "max_new_tokens": 512}
        output = requests.post(
            self.base_url + "/generate",
            json={"text": prompt, "sampling_params": sampling_params},
        ).json()[0]

        meta = output["meta_info"]
        if "spec_verify_ct" in meta and meta["spec_verify_ct"] > 0:
            acc_length = meta["completion_tokens"] / meta["spec_verify_ct"]
        else:
            acc_length = 1.0
        print(f"{acc_length=:.4f}")
        self.assertGreater(acc_length, self.acc_length_thres)

    def test_batch_generation(self):
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        results = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {"temperature": 0, "max_new_tokens": 50},
            },
        ).json()
        # Accept length from per-request meta_info (self-contained). The
        # internal_states `avg_spec_accept_length` isn't populated on the v1 /
        # disable-overlap path after a small batch, so don't read server_info.
        total_completion, total_verify = 0, 0
        for r in results:
            self.assertIn("text", r, f"Server error: {r}")
            meta = r["meta_info"]
            total_completion += meta["completion_tokens"]
            total_verify += meta.get("spec_verify_ct", 0)
        if total_verify > 0:
            acc_length = total_completion / total_verify
            print(f"batch {acc_length=:.4f}")
            self.assertGreater(acc_length, self.batch_accept_len_thres)

    def test_eos_token(self):
        prompt = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\nToday is a sunny day and I like [/INST]"
        res = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.1,
                    "max_new_tokens": 1024,
                    "skip_special_tokens": False,
                },
            },
        ).json()
        output = res["text"]
        tokens = self.tokenizer.encode(output, truncation=False)
        self.assertNotIn(self.tokenizer.eos_token_id, tokens)

    def test_first_token_finish(self):
        # Very short max_new_tokens (1-3): exercise the immediate-finish path,
        # where a request stops within the first draft window. Just must not crash.
        prompts = [
            f"There are {i} apples on the table. How to divide them equally?"
            for i in range(8)
        ]
        sampling_params = [
            {"temperature": 0, "max_new_tokens": random.randint(1, 3)} for _ in range(8)
        ]
        results = requests.post(
            self.base_url + "/generate",
            json={"text": prompts, "sampling_params": sampling_params},
        ).json()
        for r in results:
            self.assertIn("text", r, f"Server error: {r}")


def _greedy(url, text, max_new_tokens=48):
    return requests.post(
        url + "/generate",
        json={
            "text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        },
    ).json()["text"]


class SpecParityKit:
    """Lossless output parity vs a non-spec reference.

    Sequential (NOT concurrent): launch a non-spec reference server on the
    standard port, capture greedy outputs, tear it down, THEN let the fixture
    launch the spec server. Only one model is resident at a time -- two 8B
    servers don't fit on one GPU. Mix this kit FIRST in the bases so its
    setUpClass runs before the fixture's:  ``class T(SpecParityKit, Eagle3Base)``.
    """

    parity_prompts = [
        "The capital of France is",
        "Once upon a time, there was a",
        "The three primary colors are",
        "def fibonacci(n):",
    ]

    @classmethod
    def setUpClass(cls):
        ref_url = DEFAULT_URL_FOR_TEST
        ref_proc = popen_launch_server(
            cls.model,
            ref_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--mem-fraction-static",
                "0.8",  # ref alone -> full GPU available
                "--attention-backend",
                cls.attention_backend,
                "--page-size",
                "1",
                "--dtype",
                cls.dtype,
                *(["--trust-remote-code"] if cls.trust_remote_code else []),
            ],
        )
        try:
            cls.parity_ref_outputs = {
                p: _greedy(ref_url, p) for p in cls.parity_prompts
            }
        finally:
            kill_process_tree(ref_proc.pid, wait_timeout=60)
        # Now the spec server (same port; ref is gone).
        super().setUpClass()

    def test_parity_vs_reference(self):
        """Spec decode greedy output must equal the non-spec reference."""
        for prompt in self.parity_prompts:
            spec_out = _greedy(self.base_url, prompt)
            self.assertEqual(
                spec_out,
                self.parity_ref_outputs[prompt],
                f"spec != ref for prompt {prompt!r}",
            )


class SpecAccuracyKit:
    """gsm8k accuracy + acceptance length, and throughput at max_tokens=1."""

    gsm8k_num_examples = 200
    gsm8k_score_thres = 0.20
    gsm8k_check_accept_len = True
    # If set, use this; else fall back to topk-based default (2.5 / 3.47).
    gsm8k_accept_len_thres = None

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=self.gsm8k_num_examples,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], self.gsm8k_score_thres)

        if self.gsm8k_check_accept_len:
            server_info = requests.get(self.base_url + "/server_info").json()
            avg_spec_accept_length = server_info["internal_states"][0].get(
                "avg_spec_accept_length"
            )
            print(f"{avg_spec_accept_length=}")
            # The metric isn't always populated (e.g. v1 / disable-overlap).
            # Only enforce the threshold when it's reported.
            if avg_spec_accept_length is not None:
                topk = server_info["speculative_eagle_topk"]
                thres = self.gsm8k_accept_len_thres
                if thres is None:
                    thres = 2.5 if topk == 1 else 3.47
                self.assertGreater(avg_spec_accept_length, thres)


class SpecPerfKit:
    """Throughput perf check (GPU-specific -> run on the reference/Hopper runner)."""

    perf_output_throughput_thres = 50

    def test_max_token_one(self):
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=1,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        self.assertGreater(
            metrics["output_throughput"], self.perf_output_throughput_thres
        )


class SpecLogprobKit:
    """Logprob correctness: start_len, prefill-rescore match, mixed sweep,
    spec-v2 decode-vs-prefill match, and ragged token_ids_logprob."""

    # Max |decode-path - prefill-rescore| logprob gap. The two paths run
    # different kernels / batch shapes, so the gap is accumulated rounding
    # noise of the fixture dtype: ~0.25 observed for bf16 (up to 0.36 on
    # some CI runners), ~8x smaller for fp16 (3 extra mantissa bits).
    logprob_match_delta = 0.5

    def test_logprob_start_len(self):
        logprob_start_len = 4
        new_tokens = 4
        prompts = [
            "I have a very good idea on",
            "Today is a sunndy day and",
        ]

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": new_tokens,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
                "logprob_start_len": logprob_start_len,
            },
        )
        response_json = response.json()
        for res in response_json:
            self.assertEqual(
                res["meta_info"]["prompt_tokens"],
                logprob_start_len + len(res["meta_info"]["input_token_logprobs"]),
            )
            self.assertEqual(res["meta_info"]["completion_tokens"], new_tokens)
            self.assertEqual(len(res["meta_info"]["output_token_logprobs"]), new_tokens)

    def test_logprob_match(self):
        """Output logprobs should match a fresh prefill of the same sequence."""

        def run_generate(
            prompt,
            return_logprob=False,
            max_new_tokens=512,
            logprob_start_len=-1,
            temperature=1.0,
        ):
            if isinstance(prompt, str):
                prompt_kwargs = {"text": prompt}
            else:
                prompt_kwargs = {"input_ids": prompt}

            response = requests.post(
                self.base_url + "/generate",
                json={
                    **prompt_kwargs,
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                    "return_logprob": return_logprob,
                    "return_text_in_logprobs": True,
                    "logprob_start_len": logprob_start_len,
                },
            )
            return response.json()

        prompt = "I have a very good idea on how to"

        for temperature in [1.0]:
            gen = run_generate(
                prompt,
                return_logprob=True,
                logprob_start_len=0,
                temperature=temperature,
            )
            output_logprobs = np.array(
                [x[0] for x in gen["meta_info"]["output_token_logprobs"]]
            )
            num_prompts_tokens = gen["meta_info"]["prompt_tokens"]

            input_tokens = [x[1] for x in gen["meta_info"]["input_token_logprobs"]]
            output_tokens = [x[1] for x in gen["meta_info"]["output_token_logprobs"]]

            new_prompt = input_tokens + output_tokens
            score = run_generate(
                new_prompt,
                return_logprob=True,
                logprob_start_len=0,
                max_new_tokens=0,
                temperature=temperature,
            )
            output_logprobs_score = np.array(
                [
                    x[0]
                    for x in score["meta_info"]["input_token_logprobs"][
                        num_prompts_tokens:
                    ]
                ]
            )

            diff = np.abs(output_logprobs - output_logprobs_score)
            max_diff = np.max(diff)
            self.assertLess(max_diff, self.logprob_match_delta)

    def test_logprob_mixed(self):
        args = []
        temperature = 0
        # input_len, output_len, temperature, logprob_start_len, return_logprob, top_logprobs_num
        for input_len in [200, 500, 1000, 2000]:
            for output_len in [4, 8]:
                for logprob_start_len in [0, 100, 300, 800, 1998]:
                    for return_logprob in [True, False]:
                        for top_logprobs_num in [0, 5]:
                            if logprob_start_len >= input_len:
                                continue
                            args.append(
                                (
                                    input_len,
                                    output_len,
                                    temperature,
                                    logprob_start_len,
                                    return_logprob,
                                    top_logprobs_num,
                                )
                            )

        random.shuffle(args)
        func = partial(run_logprob_check, self)
        with ThreadPoolExecutor(8) as executor:
            list(executor.map(func, args))

    def test_logprob_spec_v2_match(self):
        """Verify spec v2 decode logprobs match prefill scoring logprobs."""
        top_k = 5
        probe_token_ids = [1, 2, 10, 100, 1000]
        prompts = [
            "The capital of France is",
            "Explain quantum computing in simple terms:",
        ]

        for round_idx, prompt in enumerate(prompts):
            with self.subTest(round=round_idx, prompt=prompt):
                gen_res = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 32,
                            "ignore_eos": True,
                        },
                        "return_logprob": True,
                        "top_logprobs_num": top_k,
                        "token_ids_logprob": probe_token_ids,
                        "logprob_start_len": 0,
                    },
                ).json()

                decode_logprobs = gen_res["meta_info"]["output_token_logprobs"]
                decode_top_logprobs = gen_res["meta_info"]["output_top_logprobs"]
                decode_tid_logprobs = gen_res["meta_info"]["output_token_ids_logprobs"]
                input_token_ids = [
                    t[1] for t in gen_res["meta_info"]["input_token_logprobs"]
                ]
                output_token_ids = [t[1] for t in decode_logprobs]
                num_prompt_tokens = gen_res["meta_info"]["prompt_tokens"]

                score_res = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": input_token_ids + output_token_ids,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 0,
                        },
                        "return_logprob": True,
                        "top_logprobs_num": top_k,
                        "token_ids_logprob": probe_token_ids,
                        "logprob_start_len": 0,
                    },
                ).json()

                score_logprobs = score_res["meta_info"]["input_token_logprobs"][
                    num_prompt_tokens:
                ]
                score_top_logprobs = score_res["meta_info"]["input_top_logprobs"][
                    num_prompt_tokens:
                ]
                score_tid_logprobs = score_res["meta_info"]["input_token_ids_logprobs"][
                    num_prompt_tokens:
                ]

                self.assertEqual(len(decode_logprobs), len(score_logprobs))

                decode_vals = np.array([t[0] for t in decode_logprobs])
                score_vals = np.array([t[0] for t in score_logprobs])
                max_diff = np.max(np.abs(decode_vals - score_vals))
                print(f"[round {round_idx}] logprob max_diff={max_diff:.6f}")
                self.assertLess(max_diff, self.logprob_match_delta)

                for pos in range(len(decode_logprobs)):
                    dec_top = {t[1]: t[0] for t in decode_top_logprobs[pos]}
                    scr_top = {t[1]: t[0] for t in score_top_logprobs[pos]}
                    common_ids = set(dec_top.keys()) & set(scr_top.keys())
                    self.assertGreater(len(common_ids), 0)
                    for tid in common_ids:
                        self.assertAlmostEqual(
                            dec_top[tid], scr_top[tid], delta=self.logprob_match_delta
                        )

                self.assertEqual(len(decode_tid_logprobs), len(score_tid_logprobs))
                for pos in range(len(decode_tid_logprobs)):
                    dec_tid = {t[1]: t[0] for t in decode_tid_logprobs[pos]}
                    scr_tid = {t[1]: t[0] for t in score_tid_logprobs[pos]}
                    self.assertEqual(set(dec_tid.keys()), set(scr_tid.keys()))
                    for tid in dec_tid:
                        self.assertAlmostEqual(
                            dec_tid[tid], scr_tid[tid], delta=self.logprob_match_delta
                        )

    def test_token_ids_logprob_ragged(self):
        """Regression: ragged token_ids_logprob lists in one batch must not crash."""

        def send(probe_ids):
            return requests.post(
                self.base_url + "/generate",
                json={
                    "text": "Hello world",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    "return_logprob": True,
                    "top_logprobs_num": 3,
                    "token_ids_logprob": probe_ids,
                },
            ).json()

        ragged_probes = [
            [1, 2],
            [3, 4, 5],
            [6],
            [10, 20, 30, 40],
            [1, 2],
            [3, 4, 5],
            [6],
            [10, 20, 30, 40],
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(send, ids) for ids in ragged_probes]
            for f in concurrent.futures.as_completed(futs):
                res = f.result()
                self.assertIn("text", res, f"Server error: {res}")


class SpecPenaltyKit:
    """Penalty parameters under concurrency must not crash / corrupt output."""

    def test_penalty_mixed(self):
        args = [
            {},
            {},
            {},
            {"frequency_penalty": 2},
            {"presence_penalty": 1},
            {"min_new_tokens": 16},
            {"frequency_penalty": 0.2},
            {"presence_penalty": 0.4},
            {"min_new_tokens": 8},
            {"frequency_penalty": 0.4, "presence_penalty": 0.8},
            {"frequency_penalty": 0.4, "min_new_tokens": 12},
            {"presence_penalty": 0.8, "min_new_tokens": 12},
            {"presence_penalty": -0.3, "frequency_penalty": 1.3, "min_new_tokens": 32},
            {"presence_penalty": 0.3, "frequency_penalty": -1.3, "min_new_tokens": 32},
        ]
        random.shuffle(args * 5)
        with ThreadPoolExecutor(8) as executor:
            list(executor.map(self.run_decode, args))


class SpecFeatureKit:
    """Radix attention, constrained decoding, concurrent abort."""

    def test_radix_attention(self):
        run_radix_attention_test(self.base_url)
        self.assertIsNone(self.process.poll())

    def test_request_abort(self):
        concurrency = 4
        threads = [
            threading.Thread(target=self.send_request) for _ in range(concurrency)
        ] + [
            threading.Thread(target=self.send_requests_abort)
            for _ in range(concurrency)
        ]
        for worker in threads:
            worker.start()
        for p in threads:
            p.join()

    def test_constrained_decoding(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me a json"},
        ]
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertIn("choices", res)
        self.assertEqual(len(res["choices"]), 1)
        self.assertIn("message", res["choices"][0])
        self.assertIn("content", res["choices"][0]["message"])

        content_json = res["choices"][0]["message"]["content"]
        try:
            content = json.loads(content_json)
            self.assertIsInstance(content, dict)
        except Exception:
            self.fail(f"parse JSON failed: {content_json}")


class SpecHiddenStatesKit:
    """return_hidden_states under spec V2 (regression for issue #26163).

    Requires the server launched with --enable-return-hidden-states
    (set ``enable_return_hidden_states = True`` on the fixture class).
    """

    def test_return_hidden_states(self):
        # Two prompts of different lengths to exercise the per-req stride
        # window: under spec V2 hidden_states is [bs * num_draft_tokens, dim],
        # so a wrong index aliases a neighbor request's accepted rows.
        prompts = [
            "Repeat: the quick brown fox the quick brown fox the quick brown fox",
            "Count down from ten: ten nine eight",
        ]
        res = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                "return_hidden_states": True,
            },
        )
        self.assertEqual(res.status_code, 200)
        outputs = res.json()

        for out in outputs:
            meta = out["meta_info"]
            hs = meta["hidden_states"]
            ct = meta["completion_tokens"]
            # One hidden-state entry per completion token: hs[0] is the prefill
            # block (List[List[float]]), hs[1:] are per-decode-token rows.
            self.assertEqual(
                len(hs),
                ct,
                f"len(hidden_states)={len(hs)} but completion_tokens={ct}",
            )
            decode_rows = hs[1:]
            self.assertGreater(len(decode_rows), 0)
            hidden_dim = len(decode_rows[0])
            self.assertGreater(hidden_dim, 0)
            for row in decode_rows:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), hidden_dim)


class SpecGrammarKit:
    """Grammar-constrained structured output under spec decoding.

    Regression for spec verify accepting tokens past grammar termination: the
    output must be valid JSON with nothing emitted after completion, and the
    logprob count must match the (truncated) completion-token count.
    """

    # Override per config if a different schema is desired.
    grammar_json_schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "pattern": "^[\\w]+$"},
                "population": {"type": "integer"},
                "country": {"type": "string", "pattern": "^[\\w ]+$"},
                "capital": {"type": "string", "pattern": "^[\\w ]+$"},
            },
            "required": ["name", "population", "country", "capital"],
        }
    )

    def _generate_grammar(self, return_logprob: bool):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Here is the information of the capital of France in the JSON format.\n",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 256,
                    "json_schema": self.grammar_json_schema,
                },
                "return_logprob": return_logprob,
                "logprob_start_len": 0,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        out = response.json()
        self.assertGreater(
            out["meta_info"]["spec_verify_ct"],
            0,
            "expected spec decoding to run (spec_verify_ct > 0)",
        )
        return out

    def test_grammar_structured_output_no_trailing_tokens(self):
        """Output is valid JSON with nothing emitted past grammar completion."""
        out = self._generate_grammar(return_logprob=False)
        text = out["text"]
        parsed = json.loads(text)
        for key in ("name", "population", "country", "capital"):
            self.assertIn(key, parsed)
        self.assertTrue(
            text.strip().endswith("}"), f"unexpected trailing tokens: {text!r}"
        )

    def test_grammar_logprob_count_matches_completion_tokens(self):
        """Trimmed spec tokens keep logprob count == completion token count."""
        out = self._generate_grammar(return_logprob=True)
        meta = out["meta_info"]
        completion_tokens = meta["completion_tokens"]
        output_logprobs = meta["output_token_logprobs"]
        self.assertEqual(
            len(output_logprobs),
            completion_tokens,
            "output logprobs must align with retained (trimmed) tokens: "
            f"got {len(output_logprobs)} logprobs vs {completion_tokens} completion tokens",
        )
        json.loads(out["text"])
