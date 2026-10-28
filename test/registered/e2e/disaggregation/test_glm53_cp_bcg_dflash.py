"""GLM CP4/BCG prefill -> TP4 DFlash2 decode, with padding and cached prefixes."""

import unittest

import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import popen_with_error_check, try_cached_model

register_cuda_ci(est_time=900, stage="extra-b", runner_config="8-gpu-b300")

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
    "--random-seed",
    "42",
]


class TestGlm53CpBcgDflashPD(PDDisaggregationServerBase, GSM8KMixin):
    gsm8k_score_threshold = 0.96
    gsm8k_num_examples = 1319
    gsm8k_num_threads = 256
    gsm8k_backend = "sgl_eval"
    gsm8k_thinking = True
    gsm8k_max_tokens = 8192
    gsm8k_temperature = 1.0
    gsm8k_top_p = 0.95

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
            str(cls.prefill_tp_size),
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--cuda-graph-backend-prefill",
            "breakable",
            "--cuda-graph-bs-prefill",
            "2",
            "4",
            "8",
            "16",
            "32",
            "64",
            "128",
            "256",
            "512",
            "1024",
            "2048",
            "4096",
            "8192",
        ]
        cls.extra_decode_args = common
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)
        cls.launch_all()

    @classmethod
    def launch_lb(cls):
        cls.process_lb = popen_with_error_check(
            [
                "python3",
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--prefill",
                cls.prefill_url,
                cls.bootstrap_port,
                "--decode",
                cls.decode_url,
                "--host",
                cls.base_host,
                "--port",
                cls.lb_port,
                "--request-timeout-secs",
                "7200",
                "--worker-startup-timeout-secs",
                "7200",
                "--retry-max-retries",
                "1",
                "--health-check-timeout-secs",
                "60",
            ]
        )
        cls.wait_server_ready(cls.lb_url + "/readiness", process=cls.process_lb)

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

    def test_padding_chunk_and_prefix_replay(self):
        outputs = []
        cases = (
            (257, True),
            (257, False),
            (512, True),
            (513, True),
            (513, False),
            (16421, True),
            (16421, False),
            (16423, False),
        )
        for length, cold_cache in cases:
            if cold_cache:
                for url in (self.prefill_url, self.decode_url):
                    requests.post(
                        url + "/flush_cache", params={"timeout": 30}, timeout=60
                    ).raise_for_status()
            before = self._prefill_graph_passes()
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
            graph_passes = self._prefill_graph_passes() - before
            cached_tokens = output["meta_info"]["cached_tokens"]
            print(
                f"PD smoke: length={length}, cached_tokens={cached_tokens}, "
                f"graph_passes={graph_passes}"
            )
            # Fewer tokens than the CP size cannot activate CP. Other text
            # shapes fit a bucket, including prefills with cached prefixes.
            if length - cached_tokens >= self.prefill_tp_size:
                self.assertGreater(graph_passes, 0, f"length={length}")
            else:
                self.assertEqual(graph_passes, 0, f"length={length}")
            if not cold_cache and length >= 16421:
                self.assertGreater(cached_tokens, 0)
            outputs.append(output)
        self.assertEqual(outputs[0]["text"], outputs[1]["text"])
        for process in (self.process_prefill, self.process_decode):
            self.assertIsNone(process.poll())

    def test_uneven_batched_padding_replay(self):
        for url in (self.prefill_url, self.decode_url):
            requests.post(
                url + "/flush_cache", params={"timeout": 30}, timeout=60
            ).raise_for_status()
        before = self._prefill_graph_passes()
        lengths = [257, 513, 1027]
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "input_ids": [self._prompt(length) for length in lengths],
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        for length, output in zip(lengths, response.json(), strict=True):
            self.assertIn("739391", output["text"])
            self.assertEqual(output["meta_info"]["prompt_tokens"], length)
            self.assertGreater(output["meta_info"]["spec_verify_ct"], 0)
        self.assertGreater(self._prefill_graph_passes(), before)

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
