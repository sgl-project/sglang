# SPDX-License-Identifier: Apache-2.0
"""HTTP merge/unmerge correctness with a nonzero adapter and decode CUDA graphs.

SGLANG_LORA_MERGE_MODEL=Qwen/Qwen3-4B SGLANG_LORA_MERGE_TP=2 \
    python -m unittest test.registered.lora.test_lora_merge
"""

import concurrent.futures
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import requests
import torch
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, popen_launch_server

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-large")

MODEL = os.environ.get("SGLANG_LORA_MERGE_MODEL", "Qwen/Qwen3-0.6B")
TP = int(os.environ.get("SGLANG_LORA_MERGE_TP", "1"))
BACKEND = os.environ.get("SGLANG_LORA_MERGE_BACKEND", "triton")
PREFILL_BACKEND = os.environ.get("SGLANG_LORA_MERGE_PREFILL", "disabled")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
URL = "http://127.0.0.1:30000"


def make_adapter(directory, config, seed):
    directory.mkdir()
    rank = 8
    generator = torch.Generator().manual_seed(seed)
    hidden = config.hidden_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    shapes = {
        "q_proj": (hidden, q),
        "k_proj": (hidden, kv),
        "v_proj": (hidden, kv),
        "o_proj": (q, hidden),
        "gate_proj": (hidden, config.intermediate_size),
        "up_proj": (hidden, config.intermediate_size),
        "down_proj": (config.intermediate_size, hidden),
    }
    weights = {}
    for layer in range(config.num_hidden_layers):
        for target, (in_dim, out_dim) in shapes.items():
            block = "self_attn" if target in TARGETS[:4] else "mlp"
            prefix = f"base_model.model.model.layers.{layer}.{block}.{target}"
            weights[f"{prefix}.lora_A.weight"] = (
                torch.randn(rank, in_dim, generator=generator) * 0.02
            ).bfloat16()
            weights[f"{prefix}.lora_B.weight"] = (
                torch.randn(out_dim, rank, generator=generator) * 0.02
            ).bfloat16()
    save_file(weights, directory / "adapter_model.safetensors")
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": MODEL,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": rank,
                "lora_alpha": 16,
                "target_modules": TARGETS,
                "bias": "none",
            }
        )
    )


def post(path, data=None, success=True):
    response = requests.post(URL + path, json=data or {}, timeout=180)
    if success:
        response.raise_for_status()
    else:
        assert response.status_code == 400, (path, response.status_code, response.text)
    return (
        response.json()
        if "application/json" in response.headers.get("content-type", "")
        else response.text
    )


def input_logprobs(result):
    return torch.tensor(
        [row[0] for row in result["meta_info"]["input_token_logprobs"][1:]]
    )


class TestLoRAMerge(CustomTestCase):
    def test_merge_unmerge(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = AutoConfig.from_pretrained(MODEL)
            for name, seed in [("a", 31), ("b", 32)]:
                make_adapter(directory / name, config, seed)
            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            tokens = tokenizer.encode(
                "Explain how a cache improves inference latency. " * 12
            )
            request = {
                "input_ids": tokens,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
            }
            args = [
                "--enable-lora",
                "--enable-lora-merge",
                "--lora-backend",
                BACKEND,
                "--lora-paths",
                f"a={directory / 'a'}",
                f"b={directory / 'b'}",
                "--max-lora-rank",
                "16",
                "--tp-size",
                str(TP),
                "--mem-fraction-static",
                "0.65",
                "--context-length",
                "4096",
                "--cuda-graph-backend-decode",
                "full",
                "--cuda-graph-max-bs-decode",
                "8",
                "--cuda-graph-backend-prefill",
                PREFILL_BACKEND,
                "--attention-backend",
                "fa3",
            ]
            if PREFILL_BACKEND == "full":
                args.extend(
                    [
                        "--cuda-graph-config",
                        json.dumps(
                            {
                                "decode": {"backend": "full", "bs": [1, 2, 4, 8]},
                                "prefill": {
                                    "backend": "full",
                                    "bs": [128, 256, 512, 1024],
                                    "full_prefill_max_req": 8,
                                },
                            }
                        ),
                    ]
                )
            process = popen_launch_server(MODEL, URL, timeout=600, other_args=args)
            metrics = {
                "model": MODEL,
                "tp": TP,
                "backend": BACKEND,
                "prefill_backend": PREFILL_BACKEND,
            }
            try:
                base = post("/generate", request)
                adapted = post("/generate", dict(request, lora_path="a"))
                other = post("/generate", dict(request, lora_path="b"))
                self.assertGreater(
                    (input_logprobs(base) - input_logprobs(adapted))
                    .abs()
                    .mean()
                    .item(),
                    0.005,
                )
                self.assertGreater(
                    (input_logprobs(other) - input_logprobs(adapted))
                    .abs()
                    .mean()
                    .item(),
                    0.005,
                )
                post("/merge_lora_weights", {"lora_name": "missing"}, success=False)
                transitions = []
                for cycle in range(3):
                    started = time.perf_counter()
                    self.assertTrue(
                        post("/merge_lora_weights", {"lora_name": "a"})["success"]
                    )
                    merge_s = time.perf_counter() - started
                    self.assertEqual(
                        requests.get(URL + "/model_info", timeout=10).json()[
                            "merged_lora_name"
                        ],
                        "a",
                    )
                    post("/merge_lora_weights", {"lora_name": "a"})  # Idempotent.
                    post("/merge_lora_weights", {"lora_name": "b"}, success=False)
                    post("/generate", request, success=False)
                    post("/generate", dict(request, lora_path="b"), success=False)
                    post("/unload_lora_adapter", {"lora_name": "a"}, success=False)
                    merged = post("/generate", dict(request, lora_path="a"))
                    difference = (
                        input_logprobs(merged) - input_logprobs(adapted)
                    ).abs()
                    print(
                        "unmerged_vs_merged",
                        difference.mean().item(),
                        difference.max().item(),
                        flush=True,
                    )
                    self.assertLess(difference.mean().item(), 0.03)
                    self.assertLess(difference.max().item(), 0.25)
                    self.assertEqual(
                        requests.get(URL + "/health_generate", timeout=30).status_code,
                        200,
                    )
                    # Restore while generation is active: it must complete under
                    # the merged adapter before unmerge acknowledges the switch.
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            post,
                            "/generate",
                            dict(
                                request,
                                lora_path="a",
                                sampling_params={
                                    "temperature": 0,
                                    "max_new_tokens": 128,
                                    "ignore_eos": True,
                                },
                            ),
                        )
                        deadline = time.monotonic() + 10
                        while time.monotonic() < deadline:
                            loads = requests.get(
                                URL + "/v1/loads?include=core", timeout=5
                            ).json()["loads"]
                            if any(load["num_running_reqs"] for load in loads):
                                break
                            time.sleep(0.01)
                        else:
                            self.fail("Generation did not start before unmerge")
                        started = time.perf_counter()
                        self.assertTrue(post("/unmerge_lora_weights")["success"])
                        drain_and_unmerge_s = time.perf_counter() - started
                        self.assertEqual(
                            future.result()["meta_info"]["completion_tokens"], 128
                        )
                    restored = post("/generate", request)
                    torch.testing.assert_close(
                        input_logprobs(restored), input_logprobs(base), rtol=0, atol=0
                    )
                    self.assertEqual(restored["output_ids"], base["output_ids"])
                    post("/flush_cache")
                    again = post("/generate", dict(request, lora_path="a"))
                    torch.testing.assert_close(
                        input_logprobs(again), input_logprobs(adapted), rtol=0, atol=0
                    )
                    transitions.append(
                        {
                            "merge_s": merge_s,
                            "drain_and_unmerge_s": drain_and_unmerge_s,
                            "mean_logprob_delta": difference.mean().item(),
                            "max_logprob_delta": difference.max().item(),
                        }
                    )
                # A second adapter must also merge correctly after restoration.
                post("/merge_lora_weights", {"lora_name": "b"})
                merged_b = post("/generate", dict(request, lora_path="b"))
                started = time.perf_counter()
                post("/unmerge_lora_weights")
                metrics["idle_unmerge_s"] = time.perf_counter() - started
                post("/unmerge_lora_weights")  # Idempotent.
                post("/unload_lora_adapter", {"lora_name": "a"})
                post(
                    "/load_lora_adapter",
                    {"lora_name": "a", "lora_path": str(directory / "a")},
                )
                metrics["transitions"] = transitions
                # Same work and cold prefix cache for both execution modes.
                metrics["generation_seconds"] = {"unmerged": [], "merged": []}
                for mode in ("unmerged", "merged"):
                    if mode == "merged":
                        post("/merge_lora_weights", {"lora_name": "a"})
                    for trial in range(4):
                        post("/flush_cache")
                        started = time.perf_counter()
                        outputs = post(
                            "/generate",
                            {
                                "input_ids": [tokens] * 8,
                                "lora_path": "a",
                                "sampling_params": {
                                    "temperature": 0,
                                    "max_new_tokens": 128,
                                    "ignore_eos": True,
                                },
                            },
                        )
                        self.assertEqual(
                            sum(o["meta_info"]["completion_tokens"] for o in outputs),
                            1024,
                        )
                        elapsed = time.perf_counter() - started
                        if trial:  # Warm kernels for this batch size before timing.
                            metrics["generation_seconds"][mode].append(elapsed)
                post("/unmerge_lora_weights")
            finally:
                kill_process_tree(process.pid)

            # Build an independent FP32 PEFT merge, then load that checkpoint in
            # a fresh SGLang server. This keeps attention and normalization
            # kernels the same on both sides of the correctness comparison.
            metrics["peft_reference"] = {}
            for name, actual in [("a", merged), ("b", merged_b)]:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL, torch_dtype=torch.bfloat16
                )
                model = (
                    PeftModel.from_pretrained(model, directory / name)
                    .float()
                    .merge_and_unload(safe_merge=True)
                    .to(torch.bfloat16)
                )
                checkpoint = directory / "merged_checkpoint"
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
                del model
                reference_process = popen_launch_server(
                    str(checkpoint),
                    URL,
                    timeout=600,
                    other_args=[
                        "--tp-size",
                        str(TP),
                        "--mem-fraction-static",
                        "0.65",
                        "--context-length",
                        "4096",
                        "--cuda-graph-backend-decode",
                        "full",
                        "--cuda-graph-max-bs-decode",
                        "8",
                        "--cuda-graph-backend-prefill",
                        "disabled",
                        "--attention-backend",
                        "fa3",
                    ],
                )
                try:
                    reference = input_logprobs(post("/generate", request))
                finally:
                    kill_process_tree(reference_process.pid)
                difference = (input_logprobs(actual) - reference).abs()
                metrics["peft_reference"][name] = {
                    "mean_logprob_delta": difference.mean().item(),
                    "max_logprob_delta": difference.max().item(),
                }
                print(
                    "PEFT_SGLANG_REFERENCE="
                    + json.dumps(metrics["peft_reference"][name]),
                    flush=True,
                )
                self.assertLess(difference.mean().item(), 0.03)
                self.assertLess(difference.max().item(), 0.25)
            print("LORA_MERGE_RESULTS=" + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    unittest.main()
