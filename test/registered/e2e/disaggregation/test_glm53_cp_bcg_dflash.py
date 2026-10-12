"""GLM CP4/BCG prefill -> TP4 DFlash2 decode, including eager fallbacks."""

import unittest

import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import try_cached_model

register_cuda_ci(est_time=900, stage="nightly", runner_config="8-gpu-b200")

COMMON_ARGS = [
    "--dsa-prefill-backend",
    "trtllm",
    "--dsa-decode-backend",
    "trtllm",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--moe-runner-backend",
    "flashinfer_trtllm",
    "--speculative-algorithm",
    "DFLASH",
    "--speculative-draft-attention-backend",
    "fa4",
    "--reasoning-parser",
    "auto",
    "--tool-call-parser",
    "auto",
    "--max-running-requests",
    "16",
    "--cuda-graph-max-bs-decode",
    "16",
    "--chunked-prefill-size",
    "8192",
    "--mem-fraction-static",
    "0.9",
    "--enable-metrics",
]


class TestGlm53CpBcgDflashPD(PDDisaggregationServerBase):
    prefill_tp_size = 4
    decode_tp_size = 4
    decode_base_gpu_id = 4

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model("zai-org/GLM-5.3-Flash")
        draft = try_cached_model("incoai/GLM-5.3-Flash-DFlash2")
        common = COMMON_ARGS + ["--speculative-draft-model-path", draft]
        cls.extra_prefill_args = common + [
            "--attn-cp-size",
            "4",
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--cuda-graph-backend-prefill",
            "breakable",
            "--cuda-graph-bs-prefill",
            "512",
            "8192",
        ]
        cls.extra_decode_args = common
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)
        cls.launch_all()

    def _prompt(self, length):
        suffix = self.tokenizer.encode(
            "\nThe access code is 739391. Copy the access code: 739391\n"
            "The access code is 739391. Copy the access code:",
            add_special_tokens=False,
        )
        filler = self.tokenizer.encode(
            "These are background notes. ", add_special_tokens=False
        )
        count = length - len(suffix)
        self.assertGreater(count, 0)
        return (filler * ((count + len(filler) - 1) // len(filler)))[:count] + suffix

    def test_exact_bucket_fallback_chunk_and_prefix(self):
        # Repeats cover both creation and reuse of KDA prefix checkpoints.
        before = self._prefill_graph_passes()
        outputs = []
        for length in (512, 512, 512, 513, 16421, 16421):
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "input_ids": self._prompt(length),
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                    "return_logprob": True,
                },
                timeout=600,
            )
            self.assertEqual(response.status_code, 200, response.text)
            output = response.json()
            self.assertIn("739391", output["text"])
            self.assertEqual(output["meta_info"]["prompt_tokens"], length)
            self.assertGreater(output["meta_info"]["spec_verify_ct"], 0)
            outputs.append(output)
        self.assertEqual(outputs[0]["text"], outputs[1]["text"])
        self.assertGreater(self._prefill_graph_passes(), before)
        self.assertTrue(any(out["meta_info"]["cached_tokens"] > 0 for out in outputs))
        for process in (self.process_prefill, self.process_decode):
            self.assertIsNone(process.poll())

    def _prefill_graph_passes(self):
        response = requests.get(self.prefill_url + "/metrics", timeout=30)
        response.raise_for_status()
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in response.text.splitlines()
            if line.startswith("sglang:cuda_graph_passes_total{")
            and 'mode="prefill_cuda_graph"' in line
        )


if __name__ == "__main__":
    unittest.main()
