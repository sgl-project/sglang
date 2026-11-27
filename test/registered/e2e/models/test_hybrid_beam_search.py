import io
import math
import random
import unittest

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

MODEL = "Qwen/Qwen3.5-0.8B"
WIDTH = 64


class TestHybridBeamSearch(CustomTestCase):
    def _generate(self, length, n, seed):
        rng = random.Random(seed)
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": [rng.randrange(100, 10000) for _ in range(1024)],
                "sampling_params": {
                    "beam_width": WIDTH,
                    "max_new_tokens": length,
                    "ignore_eos": True,
                    "n": n,
                },
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)
        beams = response.json()["meta_info"]["beam_results"]
        result = [
            (tuple(beam["output_ids"]), beam["meta_info"]["sequence_score"])
            for beam in beams
        ]
        self.assertEqual(len(result), n)
        self.assertEqual(len({ids for ids, _ in result}), n)
        self.assertTrue(
            all(len(ids) == length and math.isfinite(s) for ids, s in result)
        )
        return result

    def _run(self, cases, *, shared, replay):
        log = io.StringIO()
        with (
            envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION.override(shared),
            envs.SGLANG_ENABLE_BEAM_TERMINAL_READ_ONLY_STATE.override(replay),
            envs.SGLANG_ENABLE_BEAM_GDN_RANK_ONE_REPLAY.override(replay),
        ):
            process = popen_launch_server(
                MODEL,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--skip-tokenizer-init",
                    "--context-length=2048",
                    "--max-total-tokens=4096",
                    "--max-running-requests=72",
                    "--max-mamba-cache-size=72",
                    "--dtype=bfloat16",
                    "--mamba-ssm-dtype=bfloat16",
                    "--attention-backend=flashinfer",
                    "--linear-attn-decode-backend=triton",
                    "--page-size=1",
                    "--disable-radix-cache",
                    "--disable-decode-cuda-graph",
                    "--disable-prefill-cuda-graph",
                    "--skip-server-warmup",
                ],
                return_stdout_stderr=(log, log),
            )
        try:
            results = {case: self._generate(*case) for case in cases}
        finally:
            terminate_and_kill_process_tree(process)
        output = log.getvalue()
        self.assertNotIn("shared-context beam attention is unsupported", output)
        self.assertEqual("Enabled rank-one GDN beam replay." in output, replay)
        return results

    def test_shared_attention_and_replay_match_references(self):
        # The 72-slot pool fits only one 64-beam group, so each request also
        # checks that the preceding request returned its state and replay slots.
        cases = [
            (2, WIDTH, 0),
            (3, WIDTH, 0),
            (10, WIDTH, 1),
            (10, 1, 1),
            (10, WIDTH, 0),
        ]
        ordinary = self._run([(3, WIDTH, 0)], shared=False, replay=False)
        shared = self._run(cases, shared=True, replay=False)
        replay = self._run(cases, shared=True, replay=True)
        self.assertEqual(replay, shared)
        self.assertEqual(replay[(10, 1, 1)], replay[(10, WIDTH, 1)][:1])
        ref = ordinary[(3, WIDTH, 0)]
        actual = shared[(3, WIDTH, 0)]
        self.assertEqual(ref[0][0], actual[0][0])
        overlap = {ids for ids, _ in ref} & {ids for ids, _ in actual}
        self.assertGreaterEqual(len(overlap) / WIDTH, 0.9)


if __name__ == "__main__":
    unittest.main()
