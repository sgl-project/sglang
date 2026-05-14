# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Regression test for Qwen3-8B LoRA logprob accuracy.

Compares SGLang LoRA logprobs against reference training logprobs from a
pre-computed dataset. The LoRA adapter and reference data are downloaded from:
https://huggingface.co/datasets/yushengsu/lora-diff-Qwen3-8B

Usage:
    python -m unittest test_lora_qwen3_8b_logprob_diff
"""

import multiprocessing as mp
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

import sglang as sgl
from sglang.srt.lora.utils import auto_detect_lora_target_modules
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=40,
    stage="stage-b",
    runner_config="1-gpu-large",
)

BASE_MODEL = "Qwen/Qwen3-8B"
LORA_HF_REPO = "yushengsu/lora-diff-Qwen3-8B"
LORA_BACKEND = "triton"
MAX_LORA_RANK = 32
TP_SIZE = 1
PREFILL_ATTENTION_BACKEND = "fa4"
DECODE_ATTENTION_BACKEND = "fa4"

KL_THRESHOLD = 5e-3


def kl_v2(a, b):
    a = torch.tensor(a) if not torch.is_tensor(a) else a
    b = torch.tensor(b) if not torch.is_tensor(b) else b
    return (((a - b) ** 2) * 0.5).mean().item()


def get_prompt_logprobs(engine, input_ids, lora_path):
    out = engine.generate(
        input_ids=input_ids,
        sampling_params={"max_new_tokens": 0, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,
        lora_path=lora_path,
    )
    return [logprob for logprob, _, _ in out["meta_info"]["input_token_logprobs"]][1:]


class _MockLinearBase(nn.Module):
    pass


class _MockFusedMoE(nn.Module):
    pass


class _MockParallelLMHead(nn.Module):
    pass


def _build_qwen3_mock():
    """Build a lightweight nn.Module tree that mirrors Qwen3-8B's named modules."""
    model = nn.Module()
    inner = nn.Module()
    layer = nn.Module()

    attn = nn.Module()
    attn.qkv_proj = _MockLinearBase()
    attn.o_proj = _MockLinearBase()
    layer.self_attn = attn

    mlp = nn.Module()
    mlp.gate_up_proj = _MockLinearBase()
    mlp.down_proj = _MockLinearBase()
    layer.mlp = mlp

    inner.layers = nn.ModuleList([layer])
    inner.embed_tokens = nn.Embedding(10, 8)  # not a LinearBase — should be excluded
    model.model = inner
    model.lm_head = _MockParallelLMHead()
    return model


class TestLoRAQwen3_8BLogprobDiff(CustomTestCase):

    def test_auto_detect_lora_target_modules(self):
        """Verify auto_detect_lora_target_modules returns the expected module
        set for a Qwen3-8B-like (dense) architecture.  Catches silent renames
        of internal param names that would break LoRA auto-detection."""
        model = _build_qwen3_mock()

        with (
            patch("sglang.srt.layers.linear.LinearBase", _MockLinearBase),
            patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE", _MockFusedMoE
            ),
            patch(
                "sglang.srt.layers.vocab_parallel_embedding.ParallelLMHead",
                _MockParallelLMHead,
            ),
        ):
            detected = auto_detect_lora_target_modules(model)

        expected = {"qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"}
        self.assertEqual(detected, expected)

    def test_lora_qwen3_8b_logprob_accuracy(self):
        adapter_path = snapshot_download(
            LORA_HF_REPO,
            repo_type="dataset",
        )

        engine = sgl.Engine(
            model_path=BASE_MODEL,
            tp_size=TP_SIZE,
            enable_lora=True,
            max_lora_rank=MAX_LORA_RANK,
            lora_paths={"my_lora": adapter_path},
            lora_backend=LORA_BACKEND,
            attention_backend="flashinfer",
            prefill_attention_backend=PREFILL_ATTENTION_BACKEND,
            decode_attention_backend=DECODE_ATTENTION_BACKEND,
        )

        try:
            cdata = torch.load(
                os.path.join(adapter_path, "compare_sample_train_data.pt"),
                weights_only=False,
            )

            base_logprobs = get_prompt_logprobs(engine, cdata["tokens"], lora_path=None)
            logprobs = get_prompt_logprobs(engine, cdata["tokens"], lora_path="my_lora")

            base_t = torch.tensor(base_logprobs)
            lora_t = torch.tensor(logprobs)
            diff = (base_t - lora_t).abs()
            print(
                f"[VERIFY] base vs lora: mean_diff={diff.mean().item():.6f}, "
                f"max_diff={diff.max().item():.6f}, "
                f"identical={torch.equal(base_t, lora_t)}"
            )

            self.assertFalse(
                torch.equal(base_t, lora_t),
                "LoRA logprobs should differ from base model logprobs",
            )

            kl_sglang_trainer = kl_v2(cdata["training_logprobs"], logprobs)
            kl_orig_trainer = kl_v2(
                cdata["training_logprobs"], cdata["sampling_logprobs"]
            )
            kl_sglang_orig = kl_v2(logprobs, cdata["sampling_logprobs"])

            print(f"KL(orig_sampler, trainer) = {kl_orig_trainer:.6e}")
            print(f"KL(sglang, trainer)       = {kl_sglang_trainer:.6e}")
            print(f"KL(sglang, orig_sampler)  = {kl_sglang_orig:.6e}")

            self.assertLessEqual(
                kl_sglang_trainer,
                KL_THRESHOLD,
                f"KL(sglang, trainer) = {kl_sglang_trainer:.6e} exceeds "
                f"threshold {KL_THRESHOLD}",
            )

        finally:
            engine.shutdown()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
