"""SGLANG_ROCM_BCG_PREFIX_EXTEND: breakable-graph replay of prefix extends matches eager.

Runs an MLA model (MHA companion layers) with aiter prefill and breakable prefill
graphs, once with prefix extends replayed and once with them eager. Both cover
radix-cache hits and chunked-prefill continuations. aiter is not bitwise
deterministic run to run, so later greedy tokens can flip on near ties even
between two eager runs; the check is the first token, which the prefix extend
itself produces.
"""

import os
import re
import tempfile
import unittest

import requests

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# MI35x only: the logprob tolerance was measured on MI355X.
register_amd_ci(est_time=600, suite="nightly-amd-1-gpu-mi35x", nightly=True)

_SERVER_ARGS = [
    "--trust-remote-code",
    "--prefill-attention-backend",
    "aiter",
    "--cuda-graph-backend-prefill",
    "breakable",
    "--chunked-prefill-size",
    "1024",
    "--mem-fraction-static",
    "0.8",
]
_PREFIX = " ".join(
    f"Record {i}: sensor {i % 7} read {(i * 37) % 101} units." for i in range(150)
)
_QUESTIONS = [
    "Which sensor appears in record 12?",
    "What did record 40 read?",
    "Summarize the first three records.",
    "How many distinct sensors are there?",
]
_MAX_NEW_TOKENS = 4
# Two eager runs differ by up to 0.04 in first-token logprob.
_LOGPROB_TOL = 0.1
_CACHED_PREFIX_REPLAY = re.compile(
    r"Prefill batch.*#cached-token: [1-9][0-9]*.*cuda graph: True"
)


def _generate(base_url, text):
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": _MAX_NEW_TOKENS},
            "return_logprob": True,
        },
        timeout=300,
    )
    resp.raise_for_status()
    meta = resp.json()["meta_info"]
    return {
        "ids": [t[1] for t in meta["output_token_logprobs"]],
        "logprobs": [t[0] for t in meta["output_token_logprobs"]],
        "cached": meta["cached_tokens"],
    }


class TestBcgPrefixExtend(CustomTestCase):
    def _run(self, replay_prefix_extend):
        env = os.environ.copy()
        env["SGLANG_ROCM_BCG_PREFIX_EXTEND"] = "1" if replay_prefix_extend else "0"
        log = tempfile.TemporaryFile(mode="w+")
        process = popen_launch_server(
            DEFAULT_MLA_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_SERVER_ARGS,
            env=env,
            return_stdout_stderr=(log, log),
        )
        try:
            requests.post(DEFAULT_URL_FOR_TEST + "/flush_cache", timeout=60)
            # Cold: the prompt exceeds the chunk size, so later chunks extend a prefix.
            outputs = [_generate(DEFAULT_URL_FOR_TEST, _PREFIX + "\n" + _QUESTIONS[0])]
            # Warm: the shared prefix is a radix-cache hit.
            outputs += [
                _generate(DEFAULT_URL_FOR_TEST, _PREFIX + "\n" + q) for q in _QUESTIONS
            ]
        finally:
            terminate_and_kill_process_tree(process)
        log.seek(0)
        replays = len(_CACHED_PREFIX_REPLAY.findall(log.read()))
        log.close()
        return outputs, replays

    def test_replay_matches_eager(self):
        eager, eager_replays = self._run(replay_prefix_extend=False)
        replay, replay_replays = self._run(replay_prefix_extend=True)
        self.assertEqual(eager_replays, 0)
        self.assertGreater(replay_replays, 0)
        for i, (e, r) in enumerate(zip(eager, replay)):
            with self.subTest(request=i):
                if i > 0:
                    self.assertGreater(r["cached"], 1024)
                self.assertEqual(r["ids"][0], e["ids"][0])
                self.assertLess(abs(r["logprobs"][0] - e["logprobs"][0]), _LOGPROB_TOL)


if __name__ == "__main__":
    unittest.main()
