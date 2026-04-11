import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TARGET_MODEL_NGRAM,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=200, suite="stage-b-test-1-gpu-large")

GSM_DATASET_PATH = None


# Default server arguments shared across all tests
DEFAULT_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "NGRAM",
    "--speculative-num-draft-tokens",
    "16",
    "--mem-fraction-static",
    0.8,
]


class TestNgramSpeculativeDecodingBase(GSM8KMixin, CustomTestCase):
    model = DEFAULT_TARGET_MODEL_NGRAM
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_accuracy_thres = 0.79  # derived tests need to override this
    gsm8k_accept_length_thres = 1.8  # derived spec decoding tests need to override this

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "fa3"]

    @classmethod
    def setUpClass(cls):
        # disable deep gemm precompile to make launch server faster
        # please don't do this if you want to make your inference workload faster
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        model = cls.model
        cls.process = popen_launch_server(
            model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestNgramSpeculativeDecodingTriton(TestNgramSpeculativeDecodingBase):

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "triton"]


class TestNgramSpeculativeDecodingFlashinfer(TestNgramSpeculativeDecodingBase):
    TARGET_COUNTRIES = [
        "France",
        "Japan",
        "Brazil",
        "Egypt",
        "Canada",
    ]
    DISTRACTOR_COUNTRY_FAMILIES = [
        ["Spain", "South Korea", "Argentina", "Morocco", "Mexico"],
        ["Italy", "Thailand", "Chile", "Algeria", "Australia"],
        ["Germany", "Vietnam", "Peru", "Tunisia", "India"],
        ["Portugal", "Indonesia", "Colombia", "Kenya", "Norway"],
    ]
    MAX_NEW_TOKENS = 96
    NUM_ROUNDS = 2

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "flashinfer"]

    @classmethod
    def _make_country_card_prompts(cls, countries):
        prompts = []
        for country in countries:
            prompts.append(
                "Write a country reference card.\n"
                "Return exactly 8 lines with these keys in this order and no extra text:\n"
                "country:\n"
                "continent:\n"
                "capital:\n"
                "currency:\n"
                "language:\n"
                "landmark:\n"
                "famous_for:\n"
                "summary:\n"
                f"Country: {country}"
            )
        return prompts

    def _generate_batch(self, prompts, max_new_tokens=None, rounds=1):
        if max_new_tokens is None:
            max_new_tokens = self.MAX_NEW_TOKENS

        outputs = []
        for _ in range(rounds):
            outputs = []
            for prompt in prompts:
                resp = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": max_new_tokens,
                        },
                    },
                    timeout=120,
                )
                self.assertEqual(resp.status_code, 200, resp.text)
                outputs.append(resp.json()["text"])
        return outputs

    def _get_accept_length(self):
        resp = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(resp.status_code, 200, resp.text)
        info = resp.json()
        if "decode" in info:
            info = info["decode"][0]

        accept_lengths = []
        for state in info.get("internal_states", []):
            accept_length = state.get("avg_spec_accept_length")
            if accept_length is None:
                accept_length = state.get("spec_accept_length")
            if accept_length is not None:
                accept_lengths.append(accept_length)

        self.assertTrue(accept_lengths, f"No speculative accept length found in {info}")
        return sum(accept_lengths) / len(accept_lengths)

    def _flush_cache(self):
        resp = requests.post(self.base_url + "/flush_cache", timeout=30)
        self.assertEqual(resp.status_code, 200, resp.text)

    def _reset_spec_stats(self):
        # avg_spec_accept_length is cumulative, so reset it between benchmark phases.
        resp = requests.post(
            self.base_url + "/set_internal_state",
            json={"server_args": {}},
            timeout=30,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertTrue(all(payload), payload)

    def _add_corpus(self, corpus_id, documents):
        resp = requests.post(
            self.base_url + "/add_external_corpus",
            json={"corpus_id": corpus_id, "documents": documents},
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertTrue(payload["success"], payload.get("message"))
        return payload

    def _clear_external_corpora(self):
        resp = requests.get(self.base_url + "/list_external_corpora", timeout=30)
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertTrue(payload["success"], payload.get("message"))
        for corpus_id in payload["corpus_ids"]:
            remove_resp = requests.post(
                self.base_url + "/remove_external_corpus",
                json={"corpus_id": corpus_id},
                timeout=30,
            )
            self.assertEqual(remove_resp.status_code, 200, remove_resp.text)
            remove_payload = remove_resp.json()
            self.assertTrue(remove_payload["success"], remove_payload.get("message"))

    def _measure_accept_length(self, prompts, rounds=None):
        if rounds is None:
            rounds = self.NUM_ROUNDS
        self._flush_cache()
        self._reset_spec_stats()
        outputs = self._generate_batch(prompts, rounds=rounds)
        return outputs, self._get_accept_length()

    def test_weighted_sam_retains_accept_length_with_distractors(self):
        """Benchmark ranked trie/SAM matching against the no-SAM and multi-SAM baselines."""
        target_prompts = self._make_country_card_prompts(self.TARGET_COUNTRIES)
        distractor_prompt_families = [
            self._make_country_card_prompts(countries)
            for countries in self.DISTRACTOR_COUNTRY_FAMILIES
        ]

        accept_length_by_num_distractors = {}
        sam_only_accept_len = None

        try:
            self._clear_external_corpora()

            sam_docs, baseline_accept_len = self._measure_accept_length(target_prompts)
            print(f"\n  trieOnly accept length: {baseline_accept_len:.2f}")

            distractor_corpora = [
                self._generate_batch(prompts, rounds=1)
                for prompts in distractor_prompt_families
            ]

            for num_distractors in (0, 2, 4):
                with self.subTest(num_distractors=num_distractors):
                    self._clear_external_corpora()
                    self._add_corpus("sam", sam_docs)
                    for idx in range(num_distractors):
                        self._add_corpus(
                            f"distractor-{idx}",
                            distractor_corpora[idx],
                        )

                    _, accept_length = self._measure_accept_length(target_prompts)
                    accept_length_by_num_distractors[num_distractors] = accept_length

                    if num_distractors == 0:
                        sam_only_accept_len = accept_length
                        print(
                            "  samOnly accept length: "
                            f"{accept_length:.2f} ({accept_length / baseline_accept_len:.2f}x vs trieOnly)"
                        )
                        self.assertGreater(
                            accept_length,
                            baseline_accept_len * 2.0,
                            f"samOnly accept length ({accept_length:.2f}) should be at least 2x "
                            f"trieOnly ({baseline_accept_len:.2f})",
                        )
                    else:
                        print(
                            f"  samPlusDistractors[{num_distractors}] accept length: "
                            f"{accept_length:.2f} "
                            f"({accept_length / sam_only_accept_len:.2f}x vs samOnly)"
                        )
                        self.assertGreaterEqual(
                            accept_length,
                            sam_only_accept_len * 0.85,
                            f"accept length with {num_distractors} distractors ({accept_length:.2f}) "
                            f"should retain at least 85% of samOnly ({sam_only_accept_len:.2f})",
                        )
                        self.assertGreater(
                            accept_length,
                            baseline_accept_len * 1.5,
                            f"accept length with {num_distractors} distractors ({accept_length:.2f}) "
                            f"should still exceed trieOnly ({baseline_accept_len:.2f}) by 1.5x",
                        )
        finally:
            self._clear_external_corpora()
            self._flush_cache()
            self._reset_spec_stats()

        print(f"  acceptLengthByNumDistractors={accept_length_by_num_distractors}")


class TestNgramSpeculativeDecodingPaged(TestNgramSpeculativeDecodingBase):

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + [
            "--attention-backend",
            "flashinfer",
            "--page-size",
            "64",
        ]


if __name__ == "__main__":
    unittest.main()
