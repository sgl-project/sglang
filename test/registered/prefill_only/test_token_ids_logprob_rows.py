"""token_ids_logprob_rows end to end: each scored input token reports the
log-probabilities of its own requested ids."""

import random
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")


class TestTokenIdsLogprobRows(DefaultServerBase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    # Small chunks, so the scored rows span several extends of one prefill.
    other_args = [
        "--chunked-prefill-size",
        "256",
        "--context-length",
        "1024",
        "--allow-auto-truncate",
    ]

    def _score(self, input_ids, logprob_start_len, **ask):
        payload = {
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
            **ask,
        }
        return requests.post(self.base_url + "/generate", json=payload, timeout=120)

    def _case(self, length, logprob_start_len, seed):
        rng = random.Random(seed)
        input_ids = [rng.randrange(1000, 20000) for _ in range(length)]
        pool = rng.sample(range(1000, 30000), 200)
        rows = []
        for position in range(logprob_start_len + 1, length):
            # The actual token first: its score is also in input_token_logprobs.
            row = [input_ids[position]]
            row += [t for t in rng.sample(pool, 5) if t != input_ids[position]]
            rows.append(row)
        rows[3] = []
        return input_ids, rows

    def _check_rows(self, meta, rows):
        reported = meta["input_token_ids_logprobs"]
        tokens = meta["input_token_logprobs"]
        self.assertEqual(len(reported), len(rows) + 1)
        self.assertIsNone(reported[0])
        for i, row in enumerate(rows):
            entry = reported[i + 1]
            if not row:
                self.assertFalse(entry)
                continue
            self.assertEqual([item[1] for item in entry], row)
            self.assertAlmostEqual(entry[0][0], tokens[i + 1][0], places=5)

    def _check_case(self, length, logprob_start_len, seed):
        input_ids, rows = self._case(length, logprob_start_len, seed)
        self.flush_cache()
        response = self._score(
            input_ids, logprob_start_len, token_ids_logprob_rows=rows
        )
        self.assertEqual(response.status_code, 200, response.text)
        meta = response.json()["meta_info"]
        self._check_rows(meta, rows)

        # Cold again, so both prefills are identical.
        self.flush_cache()
        union = sorted({t for row in rows for t in row})
        flat = self._score(input_ids, logprob_start_len, token_ids_logprob=union)
        self.assertEqual(flat.status_code, 200, flat.text)
        flat_rows = flat.json()["meta_info"]["input_token_ids_logprobs"]
        for i, row in enumerate(rows):
            at = {item[1]: item[0] for item in flat_rows[i + 1]}
            for value, requested, *_ in meta["input_token_ids_logprobs"][i + 1] or []:
                self.assertAlmostEqual(value, at[requested], places=3)

    def test_rows_across_prefill_chunks(self):
        self._check_case(length=700, logprob_start_len=399, seed=0)

    def test_rows_from_the_first_token(self):
        self._check_case(length=48, logprob_start_len=0, seed=1)

    def test_rows_after_a_cached_prefix(self):
        input_ids, rows = self._case(length=600, logprob_start_len=449, seed=2)
        self.flush_cache()
        warm = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids[:450],
                "sampling_params": {"max_new_tokens": 1},
            },
            timeout=120,
        )
        self.assertEqual(warm.status_code, 200, warm.text)
        response = self._score(input_ids, 449, token_ids_logprob_rows=rows)
        self.assertEqual(response.status_code, 200, response.text)
        meta = response.json()["meta_info"]
        self.assertGreater(meta["cached_tokens"], 0)
        self._check_rows(meta, rows)

    def test_truncated_input_is_rejected(self):
        input_ids, rows = self._case(length=1100, logprob_start_len=999, seed=4)
        response = self._score(input_ids, 999, token_ids_logprob_rows=rows)
        self.assertEqual(response.status_code, 400, response.text)


if __name__ == "__main__":
    unittest.main()
