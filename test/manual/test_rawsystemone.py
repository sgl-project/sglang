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
                    "maximum_mean_score_difference": cls.max_score_diff,
                }
            )
        )
        cls.session.close()

    def post(self, path, payload):
        response = self.session.post(URL + path, json=payload, timeout=310)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def score(self, prefix, suffixes):
        return self.post(
            "/v1/rawsystemone",
            dict(prefix=prefix, suffixes=suffixes, return_token_logprobs=True),
        )

    def assert_candidate(self, prefix, suffix, candidate, oracle=True):
        # Independent reference: native raw-text path, no prefix bookkeeping.
        full = self.post(
            "/generate",
            {
                "text": prefix + suffix,
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
        ids = self.tokenizer.encode(prefix + suffix)
        self.assertEqual([r[1] for r in rows], ids)
        self.assertEqual([r["token_id"] for r in candidate["token_logprobs"]], ids)
        self.assertEqual(
            [r["position"] for r in candidate["token_logprobs"]], list(range(len(ids)))
        )
        self.assertIsNone(rows[0][0])
        self.assertIsNone(candidate["token_logprobs"][0]["logprob"])
        native = [r[0] for r in rows[1:]]
        actual = [r["logprob"] for r in candidate["token_logprobs"][1:]]
        self.assertEqual(candidate["scored_token_count"], len(ids) - 1)
        self.assertEqual(candidate["logprob_sum"], math.fsum(actual))
        self.assertEqual(candidate["score"], math.fsum(actual) / len(actual))
        for a, b in zip(actual, native):
            self.assertLessEqual(abs(a - b), SAME_BACKEND_ATOL)
        self.assertLessEqual(
            abs(candidate["score"] - math.fsum(native) / len(native)), SAME_BACKEND_ATOL
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
            score_error = abs(candidate["score"] - math.fsum(expected) / len(expected))
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
            ("The cat sat", ["", " down."]),
            ("", ["The cat sat.", "café🙂"]),
        ]
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
            self.assertLessEqual(
                abs(alone["score"] - one["data"][index]["score"]), SAME_BACKEND_ATOL
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
