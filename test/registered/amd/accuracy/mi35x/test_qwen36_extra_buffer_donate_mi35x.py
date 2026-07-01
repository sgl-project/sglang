"""AMD Qwen3.6 extra_buffer shared-prefix DONATE identical-output correctness test (MI35x).

This is the correctness gate that the mamba `extra_buffer` scheduler strategy needs on
ROCm: it validates the *branching-point mamba-state donate* path — the fragile core of
extra_buffer — produces byte-identical output vs recomputing the prefix fresh.

Why this test (vs a plain accuracy eval): extra_buffer caches the SSM recurrent state at
chunk boundaries so radix-cache-hitting requests can FORK ("donate") a shared prefix's
state instead of recomputing it. The dominant path is the UNALIGNED case, which extracts
the state from the chunk kernel's packed intermediate `h`
(`hybrid_linear_attn_backend.py::_init_track_ssm_indices` / `_track_mamba_state_extend`).
If the FLA (Triton) chunk kernel's `h` contract or the donate/track math is wrong on ROCm,
the forked state is wrong and cache-hit requests silently diverge — an accuracy eval that
never re-hits the same prefix would not catch it. This test forces the donate path and
asserts byte-exact equality.

Runs on the FLA-Triton GDN path (`--linear-attn-backend triton`), i.e. the upstream ROCm
path enabled by whitelisting `is_hip()` in the extra_buffer device guard (server_args.py).

Registry: nightly-amd-accuracy-1-gpu-qwen36-extra-buffer-donate suite
"""

import concurrent.futures
import json
import os
import unittest
import urllib.request

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=1200,
    suite="nightly-amd-accuracy-1-gpu-qwen36-extra-buffer-donate",
    nightly=True,
)

MODEL_PATH = "Qwen/Qwen3.6-27B-FP8"

# Long, IDENTICAL shared prefix (a deterministic "secret code" KB), longer than
# mamba_track_interval (256) so the chunk-boundary state-tracking / donate path fires.
_FACTS = [f"Fact {i}: item number {i} has secret code {(i * 7919) % 100000}." for i in range(1, 91)]
_PREFIX = "Knowledge base (memorize exactly):\n" + "\n".join(_FACTS)
_ITEMS = list(range(1, 61))
_QUESTIONS = [
    f"What is the secret code of item number {i}? Answer with only the number." for i in _ITEMS
]


class TestQwen36ExtraBufferDonate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.port = int(cls.base_url.split(":")[-1])
        other_args = [
            "--trust-remote-code",
            "--attention-backend", "aiter",
            "--linear-attn-backend", "triton",  # upstream ROCm FLA path (pending native)
            "--fp8-gemm-backend", "triton",
            "--kv-cache-dtype", "fp8_e4m3",
            "--page-size", "64",
            "--mamba-scheduler-strategy", "extra_buffer",
            "--mem-fraction-static", "0.85",
            "--max-running-requests", "64",
            "--reasoning-parser", "qwen3",
            "--model-loader-extra-config", '{"enable_multithread_load": true}',
            "--skip-server-warmup",
            "--watchdog-timeout", "900",
        ]
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, path, payload, timeout=300):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def _flush(self):
        try:
            self._post("/flush_cache", {}, timeout=30)
        except Exception:
            pass

    def _chat(self, question, max_tokens=24):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": _PREFIX + "\n\nQuestion: " + question}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return self._post("/v1/chat/completions", payload)["choices"][0]["message"]["content"]

    def test_a_donate_identical_output(self):
        """donate (radix fork of shared-prefix mamba state) must == fresh (recomputed), byte-exact."""
        # REFERENCE (fresh): flush before each → mamba state computed fresh, no cross-req fork.
        ref = []
        for q in _QUESTIONS:
            self._flush()
            ref.append(self._chat(q))

        # DONATE: prime the shared prefix once → radix hit → extra_buffer forks/donates the
        # recurrent mamba state to every subsequent (concurrent) request.
        self._flush()
        self._chat(_QUESTIONS[0])  # prime radix with the shared prefix
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(_QUESTIONS)) as ex:
            don = list(ex.map(self._chat, _QUESTIONS))

        exact = sum(1 for r, d in zip(ref, don) if r.strip() == d.strip())
        truth_ok = sum(
            1 for d, it in zip(don, _ITEMS) if str((it * 7919) % 100000) in d
        )
        n = len(_QUESTIONS)
        divergences = [
            (it, r.strip()[:60], d.strip()[:60])
            for it, r, d in zip(_ITEMS, ref, don)
            if r.strip() != d.strip()
        ]
        print(f"extra_buffer donate==fresh: {exact}/{n} ; donate correct-vs-truth: {truth_ok}/{n}")
        for it, r, d in divergences[:8]:
            print(f"  DIVERGE item {it}: fresh={r!r} donate={d!r}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_extra_buffer_donate (Qwen3.6 MI35x)\n"
                f"donate==fresh: {exact}/{n}\n"
                f"donate correct-vs-truth: {truth_ok}/{n}\n"
            )
        # The model must actually use the prefix (else the test is vacuous)...
        self.assertGreaterEqual(truth_ok, int(0.9 * n))
        # ...and the donate/branch path must be byte-identical to fresh recompute.
        self.assertEqual(exact, n, f"donate diverged from fresh on {n - exact}/{n} items")


if __name__ == "__main__":
    unittest.main()
