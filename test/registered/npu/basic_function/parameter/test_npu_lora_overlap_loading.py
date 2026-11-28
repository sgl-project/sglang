"""Ascend regression test for --enable-lora-overlap-loading.

Port the mixed-adapter/batch-splitting scenario from
test/registered/lora/test_lora_overlap_loading.py and additionally compare
overlap loading against synchronous loading. Only base-model weights are needed:
three deterministic, nonzero PEFT adapters are generated in a temporary directory.

Run on one NPU:
    SGLANG_NPU_LORA_MODEL=/path/to/Qwen3-0.6B \
    ASCEND_RT_VISIBLE_DEVICES=0 python3 test/registered/npu/basic_function/parameter/test_npu_lora_overlap_loading.py -v

SGLANG_NPU_LORA_DTYPE (default: bfloat16), SGLANG_NPU_LORA_RANKS (default:
8,16,8), SGLANG_NPU_LORA_TEST_URL and SGLANG_NPU_LORA_TEST_LOG_DIR optionally
select the dtype, ranks, server address and persistent logs/results directory.
The generated adapters cover q_proj, v_proj and o_proj, including fused QKV.
"""

import json
import os
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import requests
import torch
from safetensors.torch import save_file
from transformers import AutoConfig

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=180, suite="full-1-npu-a3", nightly=True)


@unittest.skipUnless(is_npu(), "Requires an Ascend NPU")
class TestNPULoRAOverlapLoading(CustomTestCase):
    """[Test Category] Parameter
    [Test Target] --enable-lora-overlap-loading

    Exercise cold loads, more adapters than device slots, base-model requests,
    repeated adapters, and eviction/reload with the native Ascend LoRA backend.
    """

    max_new_tokens = 16
    logprob_atol = 1e-2
    prompt = "The capital of France is"

    def _create_adapters(self, root):
        config = AutoConfig.from_pretrained(self.model)
        self.assertIn(config.model_type, ("qwen3", "llama"))
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        attention_width = config.num_attention_heads * head_dim
        shapes = {
            "q_proj": (config.hidden_size, attention_width),
            "v_proj": (config.hidden_size, config.num_key_value_heads * head_dim),
            "o_proj": (attention_width, config.hidden_size),
        }
        paths = {}
        for index, rank in enumerate(self.ranks):
            name = f"adapter_{index}"
            directory = root / name
            directory.mkdir()
            generator = torch.Generator(device="cpu").manual_seed(42 + index)
            weights = {}
            for layer in range(config.num_hidden_layers):
                for target, (input_size, output_size) in shapes.items():
                    prefix = f"base_model.model.model.layers.{layer}.self_attn.{target}"
                    # Both matrices must be nonzero: a normal zero-initialized
                    # LoRA B matrix would make a missing adapter go undetected.
                    weights[f"{prefix}.lora_A.weight"] = (
                        torch.randn(rank, input_size, generator=generator) * 0.02
                    ).to(getattr(torch, self.dtype))
                    weights[f"{prefix}.lora_B.weight"] = (
                        torch.randn(output_size, rank, generator=generator) * 0.02
                    ).to(getattr(torch, self.dtype))
            save_file(weights, str(directory / "adapter_model.safetensors"))
            (directory / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": self.model,
                        "peft_type": "LORA",
                        "task_type": "CAUSAL_LM",
                        "r": rank,
                        "lora_alpha": rank,
                        "lora_dropout": 0.0,
                        "target_modules": list(shapes),
                        "bias": "none",
                        "inference_mode": True,
                    }
                )
            )
            paths[name] = str(directory)
        return paths

    def _generate(self, adapters):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": [self.prompt] * len(adapters),
                "lora_path": adapters,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": self.max_new_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
            timeout=120,
        )
        response.raise_for_status()
        outputs = response.json()
        (
            self.log_dir / f"{self.mode}-request-{self.request_index:02d}.json"
        ).write_text(json.dumps({"adapters": adapters, "outputs": outputs}, indent=2))
        self.request_index += 1
        self.assertEqual(len(outputs), len(adapters))
        for output in outputs:
            self.assertTrue(output["text"])
            self.assertEqual(
                output["meta_info"]["completion_tokens"], self.max_new_tokens
            )
            logprobs = output["meta_info"]["output_token_logprobs"]
            self.assertEqual(len(logprobs), self.max_new_tokens)
            self.assertTrue(all(p[0] is not None for p in logprobs), output)
            self.assertTrue(
                torch.isfinite(torch.tensor([p[0] for p in logprobs])).all()
            )
        return outputs

    def _run_workload(self, paths, overlap):
        mode = "overlap" if overlap else "sync"
        self.mode, self.request_index = mode, 0
        log_path = self.log_dir / f"{mode}.log"
        args = [
            "--device",
            "npu",
            "--attention-backend",
            "ascend",
            "--lora-backend",
            "ascend",
            "--dtype",
            self.dtype,
            "--enable-lora",
            "--lora-paths",
            *[f"{name}={path}" for name, path in paths.items()],
            "--max-loras-per-batch",
            "2",
            "--max-loaded-loras",
            "4",
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.5",
            "--max-total-tokens",
            "4096",
            "--context-length",
            "1024",
            "--chunked-prefill-size",
            "128",
            "--random-seed",
            "42",
            "--log-level",
            "debug",
        ]
        if overlap:
            args.append("--enable-lora-overlap-loading")

        process = None
        with log_path.open("w") as log_file:
            try:
                process = popen_launch_server(
                    self.model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=args,
                    device="npu",
                    return_stdout_stderr=(log_file, log_file),
                )
                response = requests.get(f"{self.base_url}/server_info", timeout=30)
                response.raise_for_status()
                info = response.json()
                for key, expected in {
                    "device": "npu",
                    "lora_backend": "ascend",
                    "attention_backend": "ascend",
                    "enable_lora": True,
                    "enable_lora_overlap_loading": overlap,
                    "max_loras_per_batch": 2,
                    "max_loaded_loras": 4,
                }.items():
                    self.assertEqual(info[key], expected, key)

                a, b, c = paths
                # Single-adapter requests produce an independent reference.
                # Visiting three adapters also guarantees eviction from two slots.
                references = {
                    name: self._generate([name])[0] for name in [None, a, b, c]
                }
                batches = [
                    [None, a, b, c],
                    [c, b, a, None],
                    [a, a, c, b],
                    [None, None, None],
                    [b, c, b, a],
                    [a, None, c, b],
                ]
                results = [self._generate(batch) for batch in batches]
                return references, batches, results
            finally:
                if process is not None:
                    kill_process_tree(process.pid)
                    process.wait(timeout=30)
                print(f"{mode} server log: {log_path}", flush=True)

    def _assert_same_output(self, expected, actual):
        expected_probs = expected["meta_info"]["output_token_logprobs"]
        actual_probs = actual["meta_info"]["output_token_logprobs"]
        self.assertEqual([p[1] for p in expected_probs], [p[1] for p in actual_probs])
        self.assertEqual(expected["text"], actual["text"])
        torch.testing.assert_close(
            torch.tensor([p[0] for p in expected_probs]),
            torch.tensor([p[0] for p in actual_probs]),
            atol=self.logprob_atol,
            rtol=0,
        )

    def test_overlap_loading_matches_sync(self):
        self.model = os.environ.get("SGLANG_NPU_LORA_MODEL", QWEN3_0_6B_WEIGHTS_PATH)
        self.dtype = os.environ.get("SGLANG_NPU_LORA_DTYPE", "bfloat16")
        self.assertIn(self.dtype, ("float16", "bfloat16"))
        self.ranks = tuple(
            int(rank)
            for rank in os.environ.get("SGLANG_NPU_LORA_RANKS", "8,16,8").split(",")
        )
        self.assertEqual(len(self.ranks), 3)
        self.assertTrue(all(rank in (8, 16, 32, 64) for rank in self.ranks))
        self.base_url = os.environ.get("SGLANG_NPU_LORA_TEST_URL", DEFAULT_URL_FOR_TEST)
        log_dir = os.environ.get("SGLANG_NPU_LORA_TEST_LOG_DIR")
        self.log_dir = Path(
            log_dir or tempfile.mkdtemp(prefix="npu-lora-overlap-logs-")
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"LoRA overlap test model: {self.model}; dtype: {self.dtype}; "
            f"ranks: {self.ranks}; logs: {self.log_dir}",
            flush=True,
        )
        (self.log_dir / "configuration.json").write_text(
            json.dumps(
                {
                    "model": self.model,
                    "dtype": self.dtype,
                    "ranks": self.ranks,
                    "target_modules": ["q_proj", "v_proj", "o_proj"],
                    "max_loras_per_batch": 2,
                    "max_loaded_loras": 4,
                    "logprob_atol": self.logprob_atol,
                },
                indent=2,
            )
        )

        with tempfile.TemporaryDirectory(prefix="npu-lora-overlap-adapters-") as root:
            paths = self._create_adapters(Path(root))
            sync_refs, batches, sync_outputs = self._run_workload(paths, overlap=False)
            overlap_refs, overlap_batches, overlap_outputs = self._run_workload(
                paths, overlap=True
            )
            self.assertEqual(batches, overlap_batches)
            (self.log_dir / "outputs.json").write_text(
                json.dumps(
                    {
                        "batches": batches,
                        "sync": sync_outputs,
                        "overlap": overlap_outputs,
                    },
                    indent=2,
                )
            )
            for name in paths:
                # Check the first generated token/probability so this also catches
                # adapters silently falling back to the base model.
                base = sync_refs[None]["meta_info"]["output_token_logprobs"][0]
                adapted = sync_refs[name]["meta_info"]["output_token_logprobs"][0]
                self.assertTrue(
                    base[1] != adapted[1] or abs(base[0] - adapted[0]) > 1e-3,
                    f"{name} has no observable effect",
                )
            for name in sync_refs:
                with self.subTest(adapter=name, comparison="single sync/overlap"):
                    self._assert_same_output(sync_refs[name], overlap_refs[name])
            for batch, sync_batch, overlap_batch in zip(
                batches, sync_outputs, overlap_outputs
            ):
                for name, sync, overlap in zip(batch, sync_batch, overlap_batch):
                    with self.subTest(batch=batch, adapter=name):
                        self._assert_same_output(sync_refs[name], sync)
                        self._assert_same_output(sync, overlap)

            # Require actual asynchronous loads and a reload of every adapter;
            # merely accepting the CLI flag must not make this test pass.
            pattern = r"Loading LoRA adapter (\S+) asynchronously"
            sync_log = (self.log_dir / "sync.log").read_text()
            overlap_log = (self.log_dir / "overlap.log").read_text()
            self.assertFalse(re.findall(pattern, sync_log))
            loads = Counter(
                uid for uid in re.findall(pattern, overlap_log) if uid != "None"
            )
            self.assertEqual(len(loads), len(paths), loads)
            self.assertTrue(all(count >= 2 for count in loads.values()), loads)
            print(f"Verified async load/reload counts: {dict(loads)}", flush=True)


if __name__ == "__main__":
    unittest.main()
