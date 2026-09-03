"""End-to-end country-card benchmark for NGRAM allocation modes.

Adapted from Khoa Pham's benchmark in PR #22569:
https://github.com/sgl-project/sglang/commit/f19b0e9ad344d5140e079f071a1ea9b8ef071eb7
"""

from __future__ import annotations

import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.ngram_fixture import NgramServerBase

register_cuda_ci(est_time=600, stage="extra-a", runner_config="1-gpu-large")


class _CountryCardBenchmarkMixin:
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

    attention_backend = "flashinfer"
    global_tree_mode = ""

    # Keep numerical benchmark results report-only for now.
    enforce_benchmark_assertions = False

    @classmethod
    def get_server_args(cls):
        assert cls.global_tree_mode
        return super().get_server_args() + [
            "--speculative-ngram-global-tree-mode",
            cls.global_tree_mode,
            # The disabled control uses legacy fixed allocation. Enabled global
            # modes accept this flag too, but deliberately ignore the value.
            "--speculative-ngram-external-sam-budget",
            "8",
        ]

    @classmethod
    def _make_country_card_prompts(cls, countries):
        prompts = []
        for country in countries:
            prompts.append(
                "Write a country reference card.\n"
                "Return exactly 8 lines with these keys in this order "
                "and no extra text:\n"
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
        # avg_spec_accept_length is cumulative, so reset it between phases.
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
        for corpus_id in payload["corpus_token_counts"]:
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

    def test_country_card_accept_length_with_distractors(self):
        """Compare Trie-only, one matching SAM, and matching plus distractor SAMs."""
        target_prompts = self._make_country_card_prompts(self.TARGET_COUNTRIES)
        distractor_prompt_families = [
            self._make_country_card_prompts(countries)
            for countries in self.DISTRACTOR_COUNTRY_FAMILIES
        ]

        accept_length_by_num_distractors = {}
        sam_only_accept_len = None

        print(f"\n  globalTreeMode={self.global_tree_mode}")
        try:
            self._clear_external_corpora()

            sam_docs, baseline_accept_len = self._measure_accept_length(target_prompts)
            print(f"  trieOnly accept length: {baseline_accept_len:.2f}")

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
                            f"{accept_length:.2f} "
                            f"({accept_length / baseline_accept_len:.2f}x vs trieOnly)"
                        )
                        if self.enforce_benchmark_assertions:
                            self.assertGreater(
                                accept_length,
                                baseline_accept_len * 2.0,
                                f"samOnly accept length ({accept_length:.2f}) should be "
                                "at least "
                                f"2x trieOnly ({baseline_accept_len:.2f})",
                            )
                    else:
                        self.assertIsNotNone(sam_only_accept_len)
                        print(
                            f"  samPlusDistractors[{num_distractors}] accept length: "
                            f"{accept_length:.2f} "
                            f"({accept_length / sam_only_accept_len:.2f}x vs samOnly)"
                        )
                        if self.enforce_benchmark_assertions:
                            self.assertGreater(accept_length, 0.0)
                            if self.global_tree_mode != "disabled":
                                self.assertGreaterEqual(
                                    accept_length,
                                    sam_only_accept_len * 0.85,
                                    f"accept length with {num_distractors} distractors "
                                    f"({accept_length:.2f}) should retain at least 85% of "
                                    f"samOnly ({sam_only_accept_len:.2f})",
                                )
                                self.assertGreater(
                                    accept_length,
                                    baseline_accept_len * 1.5,
                                    f"accept length with {num_distractors} distractors "
                                    f"({accept_length:.2f}) should still exceed trieOnly "
                                    f"({baseline_accept_len:.2f}) by 1.5x",
                                )
        finally:
            self._clear_external_corpora()
            self._flush_cache()
            self._reset_spec_stats()

        print(
            f"  globalTreeMode={self.global_tree_mode}, "
            f"acceptLengthByNumDistractors={accept_length_by_num_distractors}"
        )


class TestNgramCountryCardDisabled(_CountryCardBenchmarkMixin, NgramServerBase):
    global_tree_mode = "disabled"


class TestNgramCountryCardPathProbability(_CountryCardBenchmarkMixin, NgramServerBase):
    global_tree_mode = "path_probability"


class TestNgramCountryCardSpecificityPathProbability(
    _CountryCardBenchmarkMixin, NgramServerBase
):
    global_tree_mode = "specificity_path_probability"


if __name__ == "__main__":
    unittest.main()
