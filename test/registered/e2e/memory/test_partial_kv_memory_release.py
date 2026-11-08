"""Keep a live token stream across CPU offload and a staged weight refit."""

import json
import math
import unittest
import uuid

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")


class TestPartialKvMemoryRelease(CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    tp_size = 1

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-memory-saver",
                "--enable-weights-cpu-backup",
                "--enable-kv-cache-cpu-backup",
                "--skip-tokenizer-init",
                "--enable-prefill-weight-versions",
                "--attention-backend",
                "triton",
                "--tp-size",
                str(cls.tp_size),
                "--weight-version",
                "before",
                "--max-total-tokens",
                "4096",
                "--max-running-requests",
                "2",
                "--cuda-graph-max-bs-decode",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def _control(self, endpoint, payload):
        response = requests.post(self.base_url + endpoint, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()

    def test_tokens_and_prefill_survive_offload_refit_restore(self):
        payload = dict(
            input_ids=[1, 2, 3, 4] * 16,
            sampling_params=dict(temperature=0, max_new_tokens=768, ignore_eos=True),
            return_logprob=True,
        )
        baseline = self._control(
            "/generate", payload | dict(cache_salt=uuid.uuid4().hex)
        )
        offloaded = False
        prefix_ids = None
        last = None
        with requests.post(
            self.base_url + "/generate",
            json=payload | dict(stream=True, cache_salt=uuid.uuid4().hex),
            stream=True,
            timeout=180,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                last = json.loads(line[6:])
                if not offloaded and len(last["output_ids"]) >= 16:
                    prefix_ids = last["output_ids"][:]
                    self.assertLess(len(prefix_ids), 768)
                    self._control("/pause_generation", dict(mode="in_place"))
                    self._control("/release_memory_occupation", {})
                    self._control("/resume_memory_occupation", dict(tags=["weights"]))
                    result = self._control(
                        "/update_weights_from_disk",
                        dict(
                            model_path=self.model,
                            weight_version="after",
                            flush_cache=False,
                            abort_all_requests=False,
                        ),
                    )
                    self.assertTrue(result["success"], result)
                    self._control(
                        "/resume_memory_occupation",
                        dict(tags=["cuda_graph", "kv_cache"]),
                    )
                    self._control("/continue_generation", {})
                    offloaded = True

        self.assertTrue(offloaded, "The handoff must interrupt an unfinished request")
        self.assertEqual(last["output_ids"][: len(prefix_ids)], prefix_ids)
        self.assertEqual(last["output_ids"], baseline["output_ids"])
        meta = last["meta_info"]
        self.assertEqual(meta["completion_tokens"], 768)
        self.assertEqual(
            [span["version"] for span in meta["weight_versions"]], ["before", "after"]
        )
        self.assertEqual(
            {span["version"] for span in meta["prefill_weight_versions"]}, {"before"}
        )
        for actual, expected in zip(
            meta["output_token_logprobs"],
            baseline["meta_info"]["output_token_logprobs"],
            strict=True,
        ):
            self.assertTrue(math.isfinite(actual[0]))
            self.assertEqual(actual[1], expected[1])
            self.assertAlmostEqual(actual[0], expected[0], delta=1e-3)


if __name__ == "__main__":
    unittest.main()
