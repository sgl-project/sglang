"""Live-server parity and independent CPU teacher-forcing oracle.

RAWSYSTEMONE_URL=http://127.0.0.1:30000 python test/manual/test_rawsystemone.py
See benchmark/rawsystemone/README.md for pinned-model launch commands/matrix.
Unconfigured tests skip, so CPU-only discovery never downloads a model.
"""

import json
import math
import os
import unittest

URL = os.environ.get("RAWSYSTEMONE_URL")
SAME_BACKEND_ATOL = float(os.environ.get("RAWSYSTEMONE_PARITY_ATOL", "0.0002"))
ORACLE_ATOL = float(os.environ.get("RAWSYSTEMONE_ORACLE_ATOL", "0.0005"))


@unittest.skipUnless(URL, "Set RAWSYSTEMONE_URL to an already-running test server")
class TestRawSystemOneLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import requests
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cls.session = requests.Session()
        cls.key = os.environ.get("SGLANG_API_KEY")
        if cls.key:
            cls.session.headers["Authorization"] = "Bearer " + cls.key
        cls.info = cls.session.get(URL + "/server_info", timeout=60).json()
        model = os.environ.get("RAWSYSTEMONE_ORACLE_MODEL", cls.info["model_path"])
        revision = os.environ.get(
            "RAWSYSTEMONE_ORACLE_REVISION", cls.info.get("revision")
        )
        if cls.info.get("dtype") not in ("float32", "float") or cls.info.get(
            "quantization"
        ):
            raise RuntimeError(
                "This tight-tolerance oracle suite requires an unquantized float32 server"
            )
        cls.torch = torch
        cls.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        cls.oracle = (
            AutoModelForCausalLM.from_pretrained(
                model,
                revision=revision,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            .cpu()
            .eval()
        )
        cls.max_token_diff = 0.0
        cls.max_score_diff = 0.0
        print(
            json.dumps(
                {
                    "model": model,
                    "revision": revision,
                    "tokenizer": cls.tokenizer.name_or_path,
                    "tokenizer_init_kwargs": {
                        k: cls.tokenizer.init_kwargs.get(k)
                        for k in ("add_bos_token", "add_eos_token", "_commit_hash")
                    },
                    "oracle": "CPU float32 eager, full-vocabulary log_softmax",
                    "same_backend_atol": SAME_BACKEND_ATOL,
                    "oracle_atol": ORACLE_ATOL,
                    "server": {
                        k: cls.info.get(k)
                        for k in (
                            "dtype",
                            "attention_backend",
                            "chunked_prefill_size",
                            "tp_size",
                            "disable_radix_cache",
                            "rawsystemone_max_candidates_per_batch",
                            "rawsystemone_max_inflight_batches_per_request",
                        )
                    },
                },
                indent=2,
            )
        )

    @classmethod
    def tearDownClass(cls):
        print(
            json.dumps(
                {
                    "maximum_per_token_difference": cls.max_token_diff,
                    "maximum_suffix_mean_difference": cls.max_score_diff,
                }
            )
        )
        cls.session.close()

    def post(self, path, payload):
        response = self.session.post(URL + path, json=payload, timeout=310)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def score(self, prefix, suffixes):
        result = self.post(
            "/v1/rawsystemone",
            dict(prefix=prefix, suffixes=suffixes, return_token_logprobs=True),
        )
        self.assertEqual(result["scoring"], "softmax_mean_suffix_logprob")
        self.assertEqual(result["tokenization"], "prefix_suffix_tokens_v1")
        # Check normalization separately from the native/teacher-forced token
        # likelihood comparisons below. Each original duplicate has weight.
        logits = [c["logprob_sum"] / c["option_token_count"] for c in result["data"]]
        weights = [math.exp(x - max(logits)) for x in logits]
        for candidate, weight in zip(result["data"], weights):
            self.assertAlmostEqual(candidate["score"], weight / math.fsum(weights))
        self.assertAlmostEqual(math.fsum(c["score"] for c in result["data"]), 1.0)
        return result

    def assert_candidate(self, prefix, suffix, candidate, oracle=True):
        # Independent full-sequence reference over the fixed prefix/option IDs.
        prefix_ids = self.tokenizer.encode(prefix)
        option_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        ids = prefix_ids + option_ids
        boundary = len(prefix_ids)
        full = self.post(
            "/generate",
            {
                "input_ids": ids,
                "return_logprob": True,
                "logprob_start_len": 0,
                "return_text_in_logprobs": False,
                "sampling_params": {
                    "max_new_tokens": 0,
                    "temperature": 1,
                    "top_p": 1,
                    "top_k": -1,
                    "min_p": 0,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "repetition_penalty": 1,
                },
            },
        )
        rows = full["meta_info"]["input_token_logprobs"]
        self.assertEqual([r[1] for r in rows], ids)
        self.assertEqual([r["token_id"] for r in candidate["token_logprobs"]], ids)
        self.assertEqual(
            [r["position"] for r in candidate["token_logprobs"]], list(range(len(ids)))
        )
        self.assertIsNone(rows[0][0])
        self.assertIsNone(candidate["token_logprobs"][0]["logprob"])
        native = [r[0] for r in rows[1:]]
        actual = [r["logprob"] for r in candidate["token_logprobs"][1:]]
        actual_suffix = actual[boundary - 1 :]
        native_suffix = native[boundary - 1 :]
        self.assertEqual(candidate["option_token_count"], len(option_ids))
        self.assertEqual(candidate["logprob_sum"], math.fsum(actual_suffix))
        suffix_mean = candidate["logprob_sum"] / len(option_ids)
        for a, b in zip(actual, native):
            self.assertLessEqual(abs(a - b), SAME_BACKEND_ATOL)
        self.assertLessEqual(
            abs(suffix_mean - math.fsum(native_suffix) / len(option_ids)),
            SAME_BACKEND_ATOL,
        )
        if oracle:
            tensor = self.torch.tensor([ids])
            with self.torch.inference_mode():
                logits = self.oracle(tensor).logits[0, :-1].float()
                expected = (
                    logits.log_softmax(-1)
                    .gather(1, tensor[0, 1:, None])
                    .squeeze(1)
                    .tolist()
                )
            error = max(abs(a - b) for a, b in zip(actual, expected))
            expected_suffix = expected[boundary - 1 :]
            score_error = abs(
                suffix_mean - math.fsum(expected_suffix) / len(option_ids)
            )
            type(self).max_token_diff = max(self.max_token_diff, error)
            type(self).max_score_diff = max(self.max_score_diff, score_error)
            self.assertLessEqual(
                error,
                ORACLE_ATOL,
                "Report and investigate; do not silently loosen tolerance",
            )
            self.assertLessEqual(score_error, ORACLE_ATOL)

    def test_boundaries_unequal_lengths_and_oracle(self):
        cases = [
            (
                "The requested operation is",
                [" booking.", " cancellation.", " moving an appointment to Friday."],
            ),
            ("inter", ["national", "pretation", "nal", "national"]),
            ("hello ", ["world", " world", "\nworld", "λ🙂 café"]),
            ("line\n", ["next", "\nnext", "next! "]),
            ("The cat sat", [" down.", " still."]),
        ]
        if self.tokenizer.encode(""):
            cases.append(("", ["The cat sat.", "café🙂"]))
        merge_seen = False
        for prefix, suffixes in cases:
            result = self.score(prefix, suffixes)
            self.assertEqual(result["usage"]["generated_tokens"], 0)
            for suffix, row in zip(suffixes, result["data"]):
                self.assert_candidate(prefix, suffix, row)
                # Check a real lexical boundary with special tokens disabled,
                # so a duplicated BOS cannot be the sole reason for inequality.
                encode = lambda text: self.tokenizer.encode(
                    text, add_special_tokens=False
                )
                merge_seen |= encode(prefix + suffix) != encode(prefix) + encode(suffix)
        self.assertTrue(
            merge_seen, "Test checkpoint needs a verified lexical merge-boundary case"
        )

    def test_permutation_duplicates_distractors_and_split_batches(self):
        prefix = "A customer requests an operation. The operation is"
        suffixes = [f" moving appointment number {i}." for i in range(40)] + [
            " cancellation.",
            " cancellation.",
        ]
        one = self.score(prefix, suffixes)
        reversed_result = self.score(prefix, suffixes[::-1])
        for a, b in zip(one["data"], reversed_result["data"][::-1]):
            self.assertLessEqual(abs(a["score"] - b["score"]), SAME_BACKEND_ATOL)
        self.assertEqual(one["data"][-1]["score"], one["data"][-2]["score"])
        for index in (0, 31, 32, 41):
            alone = self.score(prefix, [suffixes[index]])["data"][0]
            self.assertEqual(alone["score"], 1.0)
            self.assertLessEqual(
                abs(alone["logprob_sum"] - one["data"][index]["logprob_sum"]),
                SAME_BACKEND_ATOL * alone["option_token_count"],
            )
            self.assertEqual(
                alone["option_token_count"], one["data"][index]["option_token_count"]
            )
            self.assert_candidate(
                prefix, suffixes[index], one["data"][index], oracle=False
            )

    def test_cache_flush_and_warm_parity(self):
        prefix = "The same long context is repeated. " * 40
        suffixes = [" one.", " two.", " three."]
        self.session.post(URL + "/flush_cache", timeout=60).raise_for_status()
        cold = self.score(prefix, suffixes)
        warm = self.score(prefix, suffixes)
        self.session.post(URL + "/flush_cache", timeout=60).raise_for_status()
        evicted = self.score(prefix, suffixes)
        for a, b, c in zip(cold["data"], warm["data"], evicted["data"]):
            self.assertLessEqual(abs(a["score"] - b["score"]), SAME_BACKEND_ATOL)
            self.assertLessEqual(abs(a["score"] - c["score"]), SAME_BACKEND_ATOL)

    def test_http_errors_auth_and_legacy_endpoints(self):
        for data in [
            dict(prefix=1, suffixes=["x"]),
            dict(prefix="", suffixes=[]),
            dict(prefix="secret", suffixes=["x"], temperature=1),
        ]:
            response = self.session.post(
                URL + "/v1/rawsystemone", json=data, timeout=30
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["type"], "invalid_request")
            self.assertNotIn("secret", response.text)
        for prefix, suffixes, code in [
            ("A prompt", [""], "no_option_tokens"),
            *(
                []
                if self.tokenizer.encode("")
                else [("", ["text"], "no_prefix_tokens")]
            ),
        ]:
            response = self.session.post(
                URL + "/v1/rawsystemone",
                json=dict(prefix=prefix, suffixes=suffixes),
                timeout=30,
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["type"], code)
        if self.key:
            import requests

            self.assertEqual(
                requests.post(
                    URL + "/v1/rawsystemone",
                    json=dict(prefix="ab", suffixes=["c"]),
                    timeout=30,
                ).status_code,
                401,
            )
        self.post(
            "/generate", dict(text="Hello", sampling_params={"max_new_tokens": 1})
        )
        self.post(
            "/v1/completions",
            dict(
                model=self.info.get("served_model_name") or self.info["model_path"],
                prompt="Hello",
                max_tokens=1,
            ),
        )
        self.post(
            "/v1/chat/completions",
            dict(
                model=self.info.get("served_model_name") or self.info["model_path"],
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
            ),
        )
        legacy = self.post(
            "/v1/score",
            dict(
                query="The operation is",
                items=[" booking", " cancellation"],
                label_token_ids=[1, 2],
            ),
        )
        self.assertEqual(len(legacy["scores"]), 2)


if __name__ == "__main__":
    unittest.main()
