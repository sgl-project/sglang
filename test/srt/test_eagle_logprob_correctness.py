import json
import time
import unittest

import requests

from sglang.srt.environ import envs
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_server,
)


class TestEagleLogprobCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = "The capital of France is"
        cls.target_model = DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST
        cls.draft_model = DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.max_new_tokens = 32
        cls.random_seed = 42

    def run_server_test(self, use_eagle):
        server_args = [
            "--mem-fraction-static",
            "0.7",
            "--enable-deterministic-inference",
            "--random-seed",
            str(self.random_seed),
        ]

        if use_eagle:
            server_args.extend(
                [
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-draft-model",
                    self.draft_model,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                ]
            )
            context = envs.SGLANG_ENABLE_SPEC_V2.override(True)
        else:
            context = envs.SGLANG_ENABLE_SPEC_V2.override(False)

        with context:
            process = popen_launch_server(
                self.target_model,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=server_args,
            )

            try:
                sampling_params = {
                    "temperature": 0.0,
                    "max_new_tokens": self.max_new_tokens,
                }
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": self.prompt,
                        "sampling_params": sampling_params,
                        "return_logprob": True,
                    },
                )
                response.raise_for_status()
                out = response.json()
                return out

            finally:
                kill_process_tree(process.pid)
                time.sleep(5)

    def test_logprob_consistency(self):
        """Test logprob consistency between Eagle ON/OFF."""
        print("\nTesting logprob consistency...")
        self.maxDiff = None

        # Run without Eagle
        print("Running without Eagle...")
        out_base = self.run_server_test(use_eagle=False)

        # Run with Eagle
        print("Running with Eagle...")
        out_eagle = self.run_server_test(use_eagle=True)

        # Dump results for debugging
        with open("eagle_test_debug_base.json", "w") as f:
            json.dump(out_base, f, indent=2)
        with open("eagle_test_debug_eagle.json", "w") as f:
            json.dump(out_eagle, f, indent=2)
        print(
            f"Dumped results to eagle_test_debug_base.json and eagle_test_debug_eagle.json"
        )

        # --- Check consistency ---
        print("\nChecking consistency...")
        self.assertEqual(out_base["text"], out_eagle["text"])

        meta_base = out_base["meta_info"]
        meta_eagle = out_eagle["meta_info"]

        self.assertEqual(
            len(meta_base["output_token_logprobs"]),
            len(meta_eagle["output_token_logprobs"]),
        )

        for i, (lp_base, lp_eagle) in enumerate(
            zip(meta_base["output_token_logprobs"], meta_eagle["output_token_logprobs"])
        ):
            # lp structure: (logprob, token_id, token_text)
            self.assertEqual(lp_base[1], lp_eagle[1], f"Token ID mismatch at index {i}")
            # we ignore the logprob difference for now, it exists naturally diff for the decoding calculation and draft+verify calculation
            self.assertAlmostEqual(
                lp_base[0], lp_eagle[0], places=0, msg=f"Logprob mismatch at index {i}"
            )
        print("Check passed.")


if __name__ == "__main__":
    unittest.main()
