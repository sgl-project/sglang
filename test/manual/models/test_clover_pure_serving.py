"""Check the reference path on a local dedicated NVIDIA BF16 server.

From the repo root, set MODEL_PATH to Diffuspace/Clover-35B-A3B-Preview,
including its block diffusion weights and generation_config.json. Launch with

    SGLANG_FLASHINFER_USE_PAGED=1 python -m sglang.launch_server \
        --model-path "$MODEL_PATH" --trust-remote-code --dtype bfloat16 \
        --tp-size 1 --no-dllm-fdfo --dllm-algorithm CloverPureBlock \
        --context-length 2048 --max-running-requests 1 --port 30000

Then run

    CLOVER_SERVER_URL=http://127.0.0.1:30000 python test/manual/models/test_clover_pure_serving.py

The greedy fixture was generated with the checkpoint's Transformers generate
method on an RTX 6000D. The 65-token request resolves three width-32 blocks.
"""

import os
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

# Input and greedy continuation from Diffuspace/Clover-35B-A3B-Preview.generate.
PROMPT = [
    248045,
    846,
    198,
    96220,
    98682,
    108829,
    110528,
    98277,
    103725,
    3709,
    96172,
    97016,
    128626,
    95845,
    108969,
    98277,
    122826,
    103910,
    96760,
    1710,
    248046,
    198,
    248045,
    74455,
    198,
    248068,
    198,
]
REFERENCE = [
    198,
    97237,
    96094,
    96876,
    110528,
    98277,
    103725,
    9616,
    67,
    3950,
    44,
    3992,
    1558,
    46662,
    98404,
    128626,
    95845,
    108969,
    98277,
    122826,
    103910,
    96760,
    97016,
    1710,
    128595,
    124501,
    98682,
    109514,
    99518,
    3709,
    96172,
    108573,
    111085,
    103910,
    98853,
    1710,
    271,
    101600,
    96809,
    97434,
    100079,
    4960,
    271,
    16,
    13,
    220,
    108829,
    110528,
    98277,
    103725,
    9616,
    67,
    3950,
    44,
    7313,
    198,
    256,
    471,
    220,
    103715,
    110528,
    116578,
    109120,
    103725,
    198,
]


@unittest.skipUnless(os.getenv("CLOVER_SERVER_URL"), "requires a running Clover server")
class TestCloverPureServing(unittest.TestCase):
    def post(self, endpoint, **payload):
        return requests.post(
            os.environ["CLOVER_SERVER_URL"] + endpoint, json=payload, timeout=180
        )

    def generate(self, prompt=PROMPT, **params):
        response = self.post("/generate", input_ids=prompt, sampling_params=params)
        response.raise_for_status()
        return response.json()

    def test_queue_and_cross_block(self):
        short = PROMPT[-8:]
        expected = self.generate(short, max_new_tokens=33)["output_ids"]
        with ThreadPoolExecutor(2) as pool:
            long_request = pool.submit(self.generate, max_new_tokens=65)
            short_request = pool.submit(self.generate, short, max_new_tokens=33)
            self.assertEqual(long_request.result()["output_ids"], REFERENCE)
            self.assertEqual(short_request.result()["output_ids"], expected)

    def test_reject_experimental_policy(self):
        response = self.post(
            "/generate",
            input_ids=PROMPT,
            sampling_params={"custom_params": {"max_denoise_steps": 1}},
        )
        self.assertEqual(response.status_code, 400)

    def test_stop_and_default_budget(self):
        output = self.generate(stop_token_ids=[REFERENCE[3]], no_stop_trim=True)
        self.assertEqual(output["meta_info"]["finish_reason"]["type"], "stop")
        self.assertEqual(output["output_ids"], REFERENCE[:4])
        self.assertEqual(self.generate(max_new_tokens=0)["output_ids"], [])

    def test_abort_then_reuse(self):
        rid = uuid.uuid4().hex
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(
                self.post,
                "/generate",
                rid=rid,
                input_ids=PROMPT,
                sampling_params={"max_new_tokens": 512},
            )
            # Let the request enter generation before cancelling it.
            time.sleep(2)
            self.post("/abort_request", rid=rid).raise_for_status()
            response = pending.result()
            if response.ok:
                self.assertEqual(
                    response.json()["meta_info"]["finish_reason"]["type"], "abort"
                )
            else:
                self.assertIn("abort", response.text.lower())
        self.assertEqual(self.generate(max_new_tokens=33)["output_ids"], REFERENCE[:33])


if __name__ == "__main__":
    unittest.main()
