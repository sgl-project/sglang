"""Regression test for the mamba-radix single-token-tail cache-fidelity bug (the tail<2 retreat fix).

Bug (pre-fix): with `--mamba-scheduler-strategy extra_buffer` on a hybrid GDN+MoE model, a radix cache
HIT on a prompt whose token length is == 1 mod 64 restores the nearest GDN checkpoint and recomputes
exactly one token (extend_input_len == 1). On gfx950 that single-token (M=1) MoE forward selects tile
block_m=16, whose aiter ck_moe_stage1 GEMM returns NaN -> flat logits -> a wrong, silent first token.
The GDN/mamba recurrence is bit-exact correct; the corruption is the M=1 MoE tile.

The corruption is reliable (every tail=1 hit) but the specific garbage text varies run-to-run (the
aiter MoE reduction is non-deterministic, and a flat-logit state flips the argmax), so an
output-equality assertion is an unreliable gate. This test instead gates on the deterministic
structural property the fix guarantees: a tail=1 re-send no longer produces a 1-token GDN extend
(extend_input_len >= 2, or a fresh cached=0). A tail=2 control guards against over-triggering. Standard
accuracy gates (e.g. GSM8K) do NOT catch the bug (a multi-token answer re-grounds).

VERIFIED on Qwen3.6-35B-A3B-FP8 (TP1) on current main (bug reproduced + fix confirmed).
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(est_time=1200, suite="nightly-amd-1-gpu-mi35x", nightly=True)

MODEL_PATH = "Qwen3.6-35B-A3B-FP8"  # validated 35B hybrid GDN+MoE (TP1); adjust to the lane's model path
SERVER_LAUNCH_TIMEOUT = 1800
TP_SIZE = 1
CHUNK = (
    64  # mamba_cache_chunk_size = max(FLA_CHUNK_SIZE=64, page_size); page_size=64 here
)


def _first_token(base_url, prompt):
    """Greedy first generated token + the prompt token count + cached-token count, via /v1/completions."""
    r = requests.post(
        base_url + "/v1/completions",
        json={
            "model": MODEL_PATH,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 1,
        },
        timeout=120,
    )
    r.raise_for_status()
    c = r.json()["choices"][0]
    usage = r.json().get("usage", {}) or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return c["logprobs"]["tokens"][0], usage.get("prompt_tokens"), cached


def _build_prompt_of_len_mod(base_url, target_mod):
    """Pad a fixed base prompt until its token length % CHUNK == target_mod (so a full re-send leaves
    tail == (CHUNK - matched) ... specifically target_mod==1 -> tail==1)."""
    base = (
        "Context: a deterministic policy passage repeated to build a long, stable shared prefix. "
        "The assistant must answer strictly from the passage. " * 4
    )
    filler = (
        "Answer concisely and accurately using only the passage above. " * 80
    ).split()
    for w in range(len(filler)):
        p = f"{base}{' '.join(filler[:w])}\nQuestion: what is the policy?\nAnswer:"
        _, ptok, _ = _first_token(base_url, p)
        if ptok % CHUNK == target_mod:
            return p, ptok
    raise RuntimeError(
        f"could not build a prompt with token_len % {CHUNK} == {target_mod}"
    )


class TestQwen35RadixTail1FidelityMI35x(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
        # Minimal launch; the structural assertion below is deterministic by construction.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--tp",
                str(TP_SIZE),
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--page-size",
                "64",
                "--cuda-graph-max-bs",
                "64",
                "--max-running-requests",
                "64",
                "--max-queued-requests",
                "256",
                "--chunked-prefill-size",
                "32768",
                "--max-prefill-tokens",
                "32768",
                "--mem-fraction-static",
                "0.85",
                "--watchdog-timeout",
                "1800",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_tail1_retreat_fires(self):
        # PRIMARY, DETERMINISTIC gate. The garbage-output symptom comes from the single-token (M=1) MoE
        # returning NaN (aiter ck_moe_stage1, tile block_m=16 on gfx950); the corruption is reliable but the
        # specific garbage text varies per process launch (the MoE reduction is non-deterministic, and a
        # flat-logit state flips the argmax), so an output-equality assertion is an unreliable gate. Instead
        # assert the fix's ROUTING change, which is data-deterministic: a tail=1 re-send (token_len % 64 == 1)
        # must no longer yield a 1-token GDN extend -- the tail<2 retreat makes extend_input_len >= 2 (or
        # fresh, cached=0).
        prompt, ptok = _build_prompt_of_len_mod(self.base_url, target_mod=1)
        requests.get(self.base_url + "/flush_cache", timeout=60)
        _first_token(self.base_url, prompt)  # warm: cache the prefix
        _hit_tok, hit_ptok, hit_cached = _first_token(
            self.base_url, prompt
        )  # re-send (tail=1 without the fix)
        extend = hit_ptok - hit_cached
        self.assertGreaterEqual(
            extend,
            2,
            f"tail=1 re-send produced a {extend}-token extend (cached={hit_cached}/{hit_ptok}); the tail<2 "
            f"retreat did NOT fire -> fix inactive. (Un-fixed code yields extend==1; that single-token MoE "
            f"forward NaNs and corrupts the first token, and the garbage varies per launch, so output equality "
            f"cannot gate this -- the retreat firing is the gate.)",
        )

    def test_tail2_control_still_hits_cache(self):
        # Control: token_len % 64 == 2 -> tail=2 (already correct, never retreated). Assert the fix did NOT
        # over-trigger: the cache is still used (cached > 0) and the extend stays 2. Guards the R1/over-trigger regression.
        prompt, ptok = _build_prompt_of_len_mod(self.base_url, target_mod=2)
        requests.get(self.base_url + "/flush_cache", timeout=60)
        _first_token(self.base_url, prompt)  # warm
        _tok, hit_ptok, hit_cached = _first_token(self.base_url, prompt)  # hit
        self.assertGreater(
            hit_cached,
            0,
            "tail=2 hit should still use the cache (fix must not over-trigger)",
        )
        self.assertEqual(
            hit_ptok - hit_cached,
            2,
            "tail=2 extend should stay 2 (no spurious retreat)",
        )


if __name__ == "__main__":
    unittest.main()
