"""Full-vocabulary decode KL after GLM-5.3-Flash HiCache host restoration."""

import json
import os
import random
import unittest

import requests
import torch

from sglang.srt.utils.hf_transformers_utils import get_config, get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import _flush_cache
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(
    est_time=1200,
    stage="extra-b",
    runner_config="4-gpu-b200",
    disabled="Requires upstream Glm5NextForConditionalGeneration support and B200 KL calibration.",
)

MODEL = "RadixArk/GLM-5.3-Flash-NVFP4"
PROMPT_TOKENS = 180000
OUTPUT_TOKENS = 8
KL_THRESHOLD = 0.1
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


class TestGLM53FlashHiCacheKL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _wait_for_gpu_idle_in_ci()
        cls.model = os.environ.get("SGLANG_TEST_GLM53_MODEL") or try_cached_model(MODEL)
        if cls.model == MODEL:
            from modelscope import snapshot_download

            cls.model = snapshot_download(
                MODEL, local_dir="/models/GLM-5.3-Flash-NVFP4"
            )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=[
                "--tp-size",
                "4",
                "--ep-size",
                "4",
                "--attention-backend",
                "dsa",
                "--dsa-prefill-backend",
                "trtllm",
                "--dsa-decode-backend",
                "trtllm",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--max-total-tokens",
                "620000",
                "--max-running-requests",
                "8",
                "--max-mamba-cache-size",
                "128",
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--chunked-prefill-size",
                "8192",
                "--context-length",
                "262144",
                "--page-size",
                "64",
                "--mem-fraction-static",
                "0.7",
                "--enable-hierarchical-cache",
                "--hicache-size",
                "20",
                "--hicache-write-policy",
                "write_through",
                "--hicache-mem-layout",
                "page_first_direct",
                "--hicache-io-backend",
                "direct",
                "--enable-cache-report",
                "--random-seed",
                "919433029",
            ],
        )
        cls.addClassCleanup(
            terminate_and_kill_process_tree, cls.process, wait_timeout=60
        )
        cls.tokenizer = get_tokenizer(cls.model, trust_remote_code=True)
        cls.vocab_ids = list(
            range(
                get_config(cls.model, trust_remote_code=True)
                .get_text_config()
                .vocab_size
            )
        )

    def _prompt(self, seed):
        rng = random.Random(seed)
        words = (
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
            "mike november oscar papa quebec romeo sierra tango uniform victor "
            "whiskey xray yankee zulu"
        ).split()
        text = f"FILLER {seed}\n" + " ".join(rng.choices(words, k=PROMPT_TOKENS))
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:PROMPT_TOKENS]
        self.assertEqual(len(ids), PROMPT_TOKENS)
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.tokenizer.decode(ids)},
                {
                    "role": "user",
                    "content": "Use the bash tool to run: echo hello-repro. Call the tool now.",
                },
            ],
            tools=TOOLS,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )

    def _generate(self, prompt, *, capture=False, count=OUTPUT_TOKENS):
        payload = {
            "input_ids": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": count,
                "ignore_eos": True,
            },
        }
        if capture:
            payload.update(
                return_logprob=True,
                return_text_in_logprobs=False,
                logprob_start_len=-1,
                token_ids_logprob=self.vocab_ids,
            )
        response = requests.post(
            self.base_url + "/generate", json=payload, timeout=1800
        )
        response.raise_for_status()
        result = response.json()
        self.assertNotIn("error", result)
        meta = result["meta_info"]
        self.assertEqual(meta["finish_reason"]["type"], "length")
        self.assertEqual(len(result["output_ids"]), count)
        if capture:
            rows = meta["output_token_ids_logprobs"]
            self.assertEqual(len(rows), count)
            for row in rows:
                self.assertEqual([entry[1] for entry in row], self.vocab_ids)
            logprobs = torch.tensor(
                [[entry[0] for entry in row] for row in rows], dtype=torch.float64
            )
            self.assertTrue(torch.isfinite(logprobs).all().item())
            torch.testing.assert_close(
                logprobs.exp().sum(dim=-1),
                torch.ones(count, dtype=torch.float64),
                atol=1e-5,
                rtol=0,
            )
            result["logprobs"] = logprobs
            del meta["output_token_ids_logprobs"]
        return result

    def _score(self, prompt, continuation, *, host=False):
        steps = []
        for i in range(OUTPUT_TOKENS):
            result = self._generate(prompt + continuation[:i], capture=True, count=1)
            meta = result["meta_info"]
            self.assertGreaterEqual(meta["cached_tokens"], len(prompt) + i - 255)
            details = meta.get("cached_tokens_details") or {}
            if host and i == 0:
                self.assertGreaterEqual(details.get("host", 0), len(prompt) - 256)
            else:
                self.assertEqual(details.get("host", 0), 0)
            steps.append(result)
        return {
            "logprobs": torch.cat([step["logprobs"] for step in steps]),
            "output_ids": [step["output_ids"][0] for step in steps],
            "meta_info": steps[0]["meta_info"],
        }

    def _compare(self, seed, scenario, reference, candidate, continuation):
        p, q = reference["logprobs"], candidate["logprobs"]
        # Each output distribution conditions on the same fixed continuation prefix.
        values = (p.exp() * (p - q)).sum(dim=-1)
        self.assertGreaterEqual(values.min().item(), -1e-6)
        mean_kl = values.mean().item()
        print(
            json.dumps(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "scored_steps": OUTPUT_TOKENS,
                    "kl": values.tolist(),
                    "mean_kl": mean_kl,
                    "cache": candidate["meta_info"].get("cached_tokens_details"),
                    "conditioning_ids": continuation[:-1],
                    "reference_argmax_ids": reference["output_ids"],
                    "candidate_argmax_ids": candidate["output_ids"],
                }
            ),
            flush=True,
        )
        return mean_kl

    def test_host_restore_kl(self):
        observations = {}
        for round_id in range(5):
            seed = 919433029 + 100 * round_id
            with self.subTest(seed=seed):
                prompt = self._prompt(seed)
                _flush_cache(self.base_url)
                cold = self._generate(prompt, capture=True)
                self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
                continuation = cold["output_ids"]
                warm = self._score(prompt, continuation)
                observations[seed, "device_hit"] = self._compare(
                    seed, "device_hit", cold, warm, continuation
                )
                for offset in range(1, 5):
                    self._generate(self._prompt(seed + offset))
                host = self._score(prompt, continuation, host=True)
                observations[seed, "host_restore"] = self._compare(
                    seed, "host_restore", warm, host, continuation
                )
        self.assertEqual(len(observations), 10)
        for (seed, scenario), mean_kl in observations.items():
            with self.subTest(seed=seed, scenario=scenario):
                self.assertLess(mean_kl, KL_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
