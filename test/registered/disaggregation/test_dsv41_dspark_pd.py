"""Set DSV41_MODEL_PATH to a V4.1 checkpoint with its bundled DSpark head."""

import json
import os
import tempfile
import unittest

import numpy as np
import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

register_cuda_ci(est_time=1200, stage="nightly", runner_config="4-gpu-gb300")


class TestDSV41DSparkPD(PDDisaggregationServerBase):
    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2
    extra_prefill_env = {
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        "SGLANG_MOONCAKE_CUSTOM_MEM_POOL": "NVLINK",
        "SGLANG_RETURN_ORIGINAL_LOGPROB": "true",
    }
    extra_decode_env = {
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        "SGLANG_MOONCAKE_CUSTOM_MEM_POOL": "NVLINK",
        "SGLANG_RETURN_ORIGINAL_LOGPROB": "true",
        "SGLANG_TEST_RETRACT": "true",
        "SGLANG_TEST_RETRACT_INTERVAL": "3",
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = os.environ.get("DSV41_MODEL_PATH")
        if not cls.model:
            raise unittest.SkipTest("Set DSV41_MODEL_PATH to the V4.1 checkpoint")
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        cls.common_args = [
            "--attention-backend",
            "dsv4",
            "--moe-runner-backend",
            "flashinfer_mxfp4",
            "--ep-size",
            "2",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            "5",
            "--mem-fraction-static",
            "0.94",
            "--max-total-tokens",
            "4096",
            "--chunked-prefill-size",
            "256",
            "--context-length",
            "2048",
            "--max-running-requests",
            "2",
            "--cuda-graph-max-bs",
            "2",
            "--disable-radix-cache",
            "--random-seed",
            "0",
        ]
        cls.extra_prefill_args = cls.common_args
        cls.extra_decode_args = cls.common_args
        cls.launch_all()

    def _sample(self, url, prompts):
        generated = []
        num_retractions = 0
        for start in range(0, len(prompts), 2):
            response = requests.post(
                url + "/generate",
                json={
                    "input_ids": prompts[start : start + 2],
                    "sampling_params": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_new_tokens": 512,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": -1,
                },
                timeout=300,
            )
            self.assertEqual(response.status_code, 200, response.text)
            for result in response.json():
                meta = result["meta_info"]
                num_retractions += meta["num_retractions"]
                self.assertEqual(meta["completion_tokens"], 512, result)
                self.assertEqual(len(meta["output_token_logprobs"]), 512, result)
                generated.append(meta["output_token_logprobs"])
        return generated, num_retractions

    def _score(self, prompts, generated):
        errors = []
        for prompt, output in zip(prompts, generated):
            output_ids = [entry[1] for entry in output]
            response = requests.post(
                self.prefill_url + "/generate",
                json={
                    "input_ids": prompt + output_ids,
                    "sampling_params": {"temperature": 0.7, "max_new_tokens": 1},
                    "return_logprob": True,
                    "logprob_start_len": len(prompt) - 1,
                },
                timeout=300,
            )
            response.raise_for_status()
            # The API leaves the first requested input position unscored.
            expected = response.json()["meta_info"]["input_token_logprobs"][1:]
            self.assertEqual([entry[1] for entry in expected], output_ids)
            errors.append(
                np.array([entry[0] for entry in output])
                - np.array([entry[0] for entry in expected])
            )
        return np.stack(errors)

    def test_sampled_logprobs_match_colocated_dspark(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        filler = tokenizer.encode("The sky is blue. " * 256, add_special_tokens=False)
        suffix = tokenizer.encode(
            "\nExplain why the sky is blue:", add_special_tokens=False
        )
        lengths = (255, 256, 257, 511, 512, 513)
        prompts = [
            filler[: lengths[i % len(lengths)] - len(suffix)] + suffix
            for i in range(32)
        ]
        generated, num_retractions = self._sample(self.lb_url, prompts)
        self.assertGreater(num_retractions, 0)
        print(f"num_retractions={num_retractions}", flush=True)

        # Teacher-force the sampled trajectories on colocated DSpark. This compares
        # target probabilities without assuming identical RNG streams across PD.
        type(self)._fail_fast_stop.set()
        for name in ("process_lb", "process_decode", "process_prefill"):
            process = getattr(type(self), name)
            kill_process_tree(process.pid)
            setattr(type(self), name, None)
        reference = popen_launch_server(
            self.model,
            self.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code", "--tp", "2"] + self.common_args,
            env={
                **os.environ,
                "SGLANG_RAGGED_VERIFY_MODE": "static",
                "SGLANG_RETURN_ORIGINAL_LOGPROB": "true",
            },
        )
        try:
            # Measure colocated prefill/verify as a control for numerical differences.
            colocated, _ = self._sample(self.prefill_url, prompts)
            pd_errors = self._score(prompts, generated)
            control_errors = self._score(prompts, colocated)
            with tempfile.NamedTemporaryFile(
                mode="w", prefix="dsv41-dspark-pd-", suffix=".json", delete=False
            ) as artifact:
                json.dump(
                    {
                        "prompts": prompts,
                        "prompt_lengths": [len(prompt) for prompt in prompts],
                        "pd_output_logprobs": generated,
                        "colocated_output_logprobs": colocated,
                        "pd_errors": pd_errors.tolist(),
                        "control_errors": control_errors.tolist(),
                        "num_retractions": num_retractions,
                    },
                    artifact,
                )
                print(f"logprob_comparison={artifact.name}", flush=True)
            pd_abs, control_abs = np.abs(pd_errors), np.abs(control_errors)
            for label, errors in (("pd", pd_abs), ("colocated", control_abs)):
                print(
                    f"{label}: mean={errors.mean():.6f}, "
                    f"p99={np.quantile(errors, 0.99):.6f}, max={errors.max():.6f}, "
                    f"fraction_above_0.5={(errors > 0.5).mean():.6f}",
                    flush=True,
                )
            self.assertLess(float(pd_abs.mean()), 0.1)
            self.assertLess(float(pd_abs.mean()), float(control_abs.mean()) + 0.02)
            self.assertLess(
                float(np.quantile(pd_abs, 0.99)),
                float(np.quantile(control_abs, 0.99)) + 0.1,
            )
            self.assertLess(
                float((pd_abs > 0.5).mean()),
                float((control_abs > 0.5).mean()) + 0.005,
            )
        finally:
            kill_process_tree(reference.pid)


if __name__ == "__main__":
    unittest.main()
