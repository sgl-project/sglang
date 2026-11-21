"""Serving check of min_new_tokens under sampling penalties."""

from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer


class PenalizedEOSMixin:
    """Greedy requests with sampling penalties, min_new_tokens and a +100 logit
    bias on the end-of-sequence token. The bias makes the end-of-sequence token
    the greedy choice as soon as min_new_tokens allows it, so each request must
    stop one token after its minimum, which needs no reference server. A
    verifier that bans the token at every draft position of a block that
    starts below the minimum stops later."""

    def test_penalized_eos_after_minimum(self):
        eos = AutoTokenizer.from_pretrained(self.model).eos_token_id

        def generate(i: int) -> None:
            minimum = (4, 17, 32)[i % 3]
            # Prompts of different lengths keep earlier requests decoding while
            # later ones prefill.
            prompt = "Count from one to ten. " + "one two three four. " * (i * 20)
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 128,
                        "min_new_tokens": minimum,
                        "repetition_penalty": 1.1,
                        "frequency_penalty": 0.25,
                        "presence_penalty": 0.2,
                        "logit_bias": {str(eos): 100.0},
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            meta = response.json()["meta_info"]
            self.assertEqual(meta["finish_reason"]["type"], "stop", meta)
            self.assertEqual(meta["completion_tokens"], minimum + 1, meta)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(generate, range(8)))
