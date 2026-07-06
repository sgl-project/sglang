import random
import unittest

import numpy as np
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _get_input_logprobs,
    get_input_ids,
)
from sglang.test.test_utils import (
    DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")


MODEL = DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN
KL_THRESHOLD = 0.0025
NUM_SAMPLES = 48
MAX_PROMPT_TOKENS = 3000
MAX_NEW_TOKENS = 1024


def _load_input_ids(tokenizer_path, num_samples, max_prompt_tokens):
    try:
        return get_input_ids(
            tokenizer_path,
            max_prompt_tokens=max_prompt_tokens,
            num_samples=num_samples,
        )
    except (ValueError, OSError) as e:
        print(
            f"WARNING: Could not load LongBench inputs with tokenizer "
            f"'{tokenizer_path}': {e}"
        )
        print("Falling back to random token IDs")
        return [
            [
                random.randint(1, 32000 - 1)
                for _ in range(int(max_prompt_tokens * random.uniform(0.5, 1.5)))
            ]
            for _ in range(num_samples)
        ]


def _compute_kl(input_logprobs, output_logprobs):
    kl_divs = []
    for idx, (inp_lp, out_lp) in enumerate(zip(input_logprobs, output_logprobs)):
        inp_none = any(v is None for v in inp_lp)
        out_none = any(v is None for v in out_lp)
        if inp_none or out_none:
            src = "input" if inp_none else "output"
            if inp_none and out_none:
                src = "input and output"
            print(f"  WARNING: sample {idx}: skipping due to None in {src} logprobs")
            continue
        logr = np.array(inp_lp) - np.array(out_lp)
        kl_divs.append(float(np.mean((np.exp(logr) - 1) - logr)))

    assert kl_divs, "No valid KL samples"
    avg = sum(kl_divs) / len(kl_divs)
    print(f"  per-sample KL: {kl_divs}")
    print(f"  avg KL: {avg:.6f}")
    print(f"  KL threshold: {KL_THRESHOLD:.6f}")
    return avg


def _is_prefill_hit(result):
    return result["meta_info"]["cached_tokens"] > 0


def _is_decode_hit(result, first_turn_len):
    return result["meta_info"]["cached_tokens"] > first_turn_len + 1


def _hit_info(result):
    return f"cached_tokens={result['meta_info']['cached_tokens']}"


def test_prefill_cache_hit(base_url, input_ids, max_new_tokens):
    print("--- Prefill cache hit KL test ---")
    _flush_cache(base_url)
    _generate(base_url, input_ids, max_new_tokens=0)

    results = _generate(base_url, input_ids, max_new_tokens, return_logprob=True)
    new_input_ids, output_logprobs = [], []
    for i, r in enumerate(results):
        hit = _is_prefill_hit(r)
        info = _hit_info(r)
        print(f"  [{i}] prefix_len={len(input_ids[i])} {info} hit={hit}")
        if not hit:
            continue
        new_input_ids.append(input_ids[i] + r["output_ids"])
        output_logprobs.append(_extract_output_logprobs(r))

    print(f"  cache hits: {len(new_input_ids)}/{len(input_ids)}")
    assert (
        len(new_input_ids) > len(input_ids) // 2
    ), f"too few cache hits: {len(new_input_ids)}/{len(input_ids)}"

    input_logprobs = _get_input_logprobs(base_url, new_input_ids, output_logprobs)
    return _compute_kl(input_logprobs, output_logprobs)


def test_decode_cache_hit(base_url, input_ids, max_new_tokens):
    print("--- Decode cache hit KL test ---")
    suffix_token = [1]

    _flush_cache(base_url)
    first = _generate(base_url, input_ids, max_new_tokens, return_logprob=True)
    turn2_ids = [
        input_ids[i] + r["output_ids"] + suffix_token for i, r in enumerate(first)
    ]

    results = _generate(base_url, turn2_ids, max_new_tokens, return_logprob=True)
    new_input_ids, output_logprobs = [], []
    for i, r in enumerate(results):
        hit = _is_decode_hit(r, len(input_ids[i]))
        info = _hit_info(r)
        print(f"  [{i}] prefix_len={len(turn2_ids[i])} {info} hit={hit}")
        if not hit:
            continue
        new_input_ids.append(turn2_ids[i] + r["output_ids"])
        output_logprobs.append(_extract_output_logprobs(r))

    print(f"  cache hits: {len(new_input_ids)}/{len(turn2_ids)}")
    assert (
        len(new_input_ids) > len(turn2_ids) // 2
    ), f"too few cache hits: {len(new_input_ids)}/{len(turn2_ids)}"

    input_logprobs = _get_input_logprobs(base_url, new_input_ids, output_logprobs)
    return _compute_kl(input_logprobs, output_logprobs)


class TestDPAttentionBreakableCudaGraphKL(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--cuda-graph-backend-prefill=breakable",
            ],
        )
        random.seed(42)
        cls.server_info = requests.get(f"{cls.base_url}/server_info").json()
        tokenizer_path = (
            cls.server_info.get("tokenizer_path")
            or cls.server_info.get("model_path")
            or cls.model
        )
        cls.input_ids = _load_input_ids(
            tokenizer_path,
            NUM_SAMPLES,
            MAX_PROMPT_TOKENS,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _assert_kl(self, test_name, kl_div):
        self.assertLess(
            kl_div,
            KL_THRESHOLD,
            f"avg_kl_div={kl_div} >= threshold={KL_THRESHOLD} "
            f"for {self.model} {test_name}",
        )

    def _print_test_config(self):
        print(
            "KL test config: "
            f"model={self.model}, samples={len(self.input_ids)}, "
            f"max_prompt_tokens={MAX_PROMPT_TOKENS}, "
            f"max_new_tokens={MAX_NEW_TOKENS}, "
            f"threshold={KL_THRESHOLD}"
        )

    def test_prefill_cache_hit_kl(self):
        if self.server_info.get("disable_radix_cache"):
            self.skipTest("Radix cache is disabled")
        self._print_test_config()
        kl_div = test_prefill_cache_hit(
            self.base_url, self.input_ids, MAX_NEW_TOKENS
        )
        self._assert_kl("test_prefill_cache_hit_kl", kl_div)

    def test_decode_cache_hit_kl(self):
        if self.server_info.get("disable_radix_cache"):
            self.skipTest("Radix cache is disabled")
        self._print_test_config()
        kl_div = test_decode_cache_hit(self.base_url, self.input_ids, MAX_NEW_TOKENS)
        self._assert_kl("test_decode_cache_hit_kl", kl_div)


if __name__ == "__main__":
    unittest.main()
