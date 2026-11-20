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
    other_args = ["--chunked-prefill-size", "256"]

    def _score(self, input_ids, logprob_start_len, **ask):
        self.flush_cache()
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 0},
                "return_logprob": True,
                "logprob_start_len": logprob_start_len,
                **ask,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["meta_info"]

    def test_rows_across_prefill_chunks(self):
        rng = random.Random(0)
        length, logprob_start_len = 700, 399
        input_ids = [rng.randrange(1000, 20000) for _ in range(length)]
        pool = rng.sample(range(1000, 30000), 200)
        rows = []
        for position in range(logprob_start_len + 1, length):
            # The actual token first: its score is also in input_token_logprobs.
            row = [input_ids[position]]
            row += [t for t in rng.sample(pool, 5) if t != input_ids[position]]
            rows.append(row)
        rows[3] = []

        meta = self._score(input_ids, logprob_start_len, token_ids_logprob_rows=rows)
        reported = meta["input_token_ids_logprobs"]
        self.assertEqual(len(reported), len(rows) + 1)
        self.assertIsNone(reported[0])
        for i, row in enumerate(rows):
            if not row:
                self.assertFalse(reported[i + 1])
                continue
            self.assertEqual([item[1] for item in reported[i + 1]], row)
            self.assertAlmostEqual(
                reported[i + 1][0][0], meta["input_token_logprobs"][i + 1][0], places=5
            )

        # The flat field over the union of ids, from an identical cold prefill.
        union = sorted({t for row in rows for t in row})
        flat = self._score(input_ids, logprob_start_len, token_ids_logprob=union)
        for i, row in enumerate(rows):
            at = {item[1]: item[0] for item in flat["input_token_ids_logprobs"][i + 1]}
            for value, requested, *_ in reported[i + 1] or []:
                self.assertAlmostEqual(value, at[requested], places=3)


if __name__ == "__main__":
    unittest.main()
