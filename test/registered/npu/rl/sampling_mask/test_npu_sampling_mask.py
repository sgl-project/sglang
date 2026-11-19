import json
import math
import os
import shlex
import unittest

import requests

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

GLM_5_2_BF16_MODEL_PATH = os.environ.get(
    "GLM_5_2_BF16_MODEL_PATH", "/root/.cache/modelscope/hub/models/ZhipuAI/GLM-5.2"
)

register_npu_ci(est_time=300, suite="extra-b-test-2-npu-a3")
register_npu_ci(est_time=300, suite="nightly-test-2-npu-a3", nightly=True)


@unittest.skipUnless(
    os.path.isfile(os.path.join(GLM_5_2_BF16_MODEL_PATH, "config.json")),
    "GLM-5.2 config missing",
)
class TestGLM52BF16SamplingMask(CustomTestCase):
    model = GLM_5_2_BF16_MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST
    model_overrides = {"num_hidden_layers": 6, "quantization_config": None}

    @classmethod
    def setUpClass(cls):
        env = {
            **os.environ,
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
        }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=[
                *shlex.split(
                    "--device npu --sampling-backend ascend --sampling-mask-max-tokens 2048 "
                    "--dtype bfloat16 --load-format dummy --trust-remote-code "
                    "--tp-size 2 --attn-dp-size 2 --enable-dp-lm-head --disable-radix-cache "
                    "--mem-fraction-static 0.8 --context-length 4096 --max-total-tokens 4096 "
                    "--chunked-prefill-size 512 --cuda-graph-max-bs-decode 4"
                ),
                "--attention-backend",
                "ascend",
                "--json-model-override-args",
                json.dumps(cls.model_overrides),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def _generate(self, params, text="The capital of France is", **kwargs):
        response = requests.post(
            self.base_url + "/generate",
            timeout=60,
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                    **params,
                },
                "return_logprob": True,
                "return_sampling_logprobs": True,
                "return_sampling_mask": True,
                **kwargs,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _check_output(self, body, *, greedy=False, filters=None):
        meta = body["meta_info"]
        self.assertEqual(len(body["output_ids"]), 4)
        for step, (token, mask_row, sampled_logprob) in enumerate(
            zip(
                body["output_ids"],
                meta["output_sampling_mask_logprobs"],
                meta["output_token_sampling_logprobs"],
                strict=True,
            )
        ):
            mask = [token_id for _, token_id in mask_row]
            values = [logprob for logprob, _ in mask_row]
            self.assertIn(token, mask)
            self.assertEqual(len(mask), len(set(mask)))
            self.assertLessEqual(len(mask), 2048)
            self.assertTrue(all(math.isfinite(v) and v <= 0 for v in values))
            self.assertAlmostEqual(sum(math.exp(v) for v in values), 1.0, delta=1e-4)
            self.assertAlmostEqual(
                sampled_logprob, values[mask.index(token)], delta=1e-6
            )
            if greedy:
                self.assertEqual(mask, [token])
                self.assertEqual(values, [0.0])
            if filters is not None:
                k, p = filters
                ranked = sorted(
                    meta["output_top_logprobs"][step],
                    key=lambda item: item[0],
                    reverse=True,
                )
                # Establish a complete, unambiguous top-k cutoff in the prefix.
                self.assertGreater(len(ranked), k)
                self.assertGreater(ranked[k - 1][0] - ranked[k][0], 1e-4)
                expected = {}
                mass = 0.0
                for logprob, token_id, _ in ranked[:k]:
                    if p < 1.0 and mass > p:
                        break
                    weight = math.exp(logprob)
                    expected[int(token_id)] = weight
                    mass += weight
                self.assertEqual(set(mask), set(expected))
                for token_id, value in zip(mask, values, strict=True):
                    self.assertAlmostEqual(
                        value, math.log(expected[token_id] / mass), delta=1e-3
                    )

    def test_single_request_sampling_masks(self):
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        info = response.json()
        expected = {
            "dtype": "bfloat16",
            "load_format": "dummy",
            "tp_size": 2,
            "attn_dp_size": 2,
        }
        self.assertEqual({key: info[key] for key in expected}, expected)
        self.assertEqual(info["cuda_graph_config"]["decode"]["backend"], "full")
        self.assertEqual(
            json.loads(info["json_model_override_args"])["num_hidden_layers"], 6
        )
        cases = [
            ("greedy", 1, 1.0),
            ("topk", 10, 1.0),
            ("topp", 10, 0.8),
            ("fallback", 1025, 1.0),
            ("fallback_topp", 1025, 0.8),
        ]
        for name, k, p in cases:
            with self.subTest(case=name):
                params = {
                    "temperature": 0.0 if name == "greedy" else 1.0,
                    "top_k": k,
                    "top_p": p,
                }
                exact = name != "greedy"
                if exact:
                    # Separate cutoff values after logprob rounding.
                    spacing, count = (0.7, 16) if k <= 1024 else (0.005, 1100)
                    params["logit_bias"] = {
                        str(t): 40.0 - spacing * t for t in range(count)
                    }
                body = self._generate(params, top_logprobs_num=2048)
                self._check_output(
                    body,
                    greedy=name == "greedy",
                    filters=(k, p) if exact else None,
                )

    def test_mixed_batch_flags(self):
        """Check each flag, opt-in and request/token association independently."""
        bodies = self._generate(
            {"top_k": 10},
            text=[
                "The capital of France is",
                "The capital of Germany is",
                "The capital of Italy is",
            ],
            return_sampling_logprobs=[True, False, False],
            return_sampling_mask=[True, False, True],
        )
        self.assertEqual(len(bodies), 3)
        for key in ("output_sampling_mask_logprobs", "output_token_sampling_logprobs"):
            self.assertNotIn(key, bodies[1]["meta_info"])
        self._check_output(bodies[0])
        mask_only = bodies[2]["meta_info"]
        self.assertNotIn("output_token_sampling_logprobs", mask_only)
        for token, mask_row in zip(
            bodies[2]["output_ids"],
            mask_only["output_sampling_mask_logprobs"],
            strict=True,
        ):
            self.assertIn(token, [token_id for _, token_id in mask_row])


if __name__ == "__main__":
    unittest.main()
