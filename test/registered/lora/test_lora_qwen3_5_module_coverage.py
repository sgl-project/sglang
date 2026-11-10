# Copyright 2023-2026 SGLang Team
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

"""Every LoRA target module type on Qwen3.5 must be applied, and match HF+PEFT.

One random-init adapter per module type (full attention, GatedDeltaNet, MLP).
For each, the LoRA-minus-base shift in teacher-forced prompt logprobs from
SGLang must match the shift from HF+PEFT. A module type that SGLang loads but
never applies has a zero shift; one applied to the wrong output slice has a
shift in the wrong direction.
"""

import gc
import multiprocessing as mp
import tempfile
import unittest

import torch

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import get_prompt_logprobs
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=150, stage="nightly", runner_config="1-gpu-small")

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
# HF+PEFT target names in Qwen3.5's language model. Each becomes one adapter.
MODULE_TYPES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LORA_RANK = 8
# Picked so each adapter shifts logprobs well above the bf16 SGLang-vs-HF floor.
LORA_B_STD = 0.02
PROMPTS = (
    "The capital of France is Paris. The capital of Japan is Tokyo. The capital "
    "of Italy is Rome, and the capital of Spain is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1)"
    " + fibonacci(n - 2)\n\nprint([fibonacci(i) for i in range(10)])",
)
MIN_COSINE = 0.9
NORM_RATIO_RANGE = (0.8, 1.25)


def _load_hf_model():
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16
    ).cuda()


def _save_random_adapter(model, module_type: str, output_dir: str, seed: int):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=2 * LORA_RANK,
        target_modules=[module_type],
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(model, config)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A" in name:
                std = param.shape[1] ** -0.5
            elif "lora_B" in name:
                std = LORA_B_STD
            else:
                continue
            noise = torch.randn(param.shape, generator=generator) * std
            param.copy_(noise.to(param.dtype))
    peft_model.save_pretrained(output_dir)
    return peft_model


def _hf_prompt_logprobs(model, input_ids):
    ids = torch.tensor([input_ids], device="cuda")
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    return logprobs.gather(-1, ids[0, 1:, None]).squeeze(-1).cpu()


def _hf_reference(prompt_ids, adapter_dirs):
    """Build one adapter per module type; return HF base and LoRA logprobs."""
    model = _load_hf_model()
    model.eval()
    base = [_hf_prompt_logprobs(model, ids) for ids in prompt_ids]
    lora = {}
    for seed, module_type in enumerate(MODULE_TYPES):
        peft_model = _save_random_adapter(
            model, module_type, adapter_dirs[module_type], seed
        )
        if not any(module_type in n for n, _ in peft_model.named_parameters()):
            raise AssertionError(f"PEFT wrapped no {module_type} module")
        lora[module_type] = [_hf_prompt_logprobs(peft_model, ids) for ids in prompt_ids]
        model = peft_model.unload()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return base, lora


def _srt_logprobs(prompt_ids, adapter_dirs):
    engine = sgl.Engine(
        model_path=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        max_loras_per_batch=len(MODULE_TYPES),
        lora_paths=adapter_dirs,
        disable_radix_cache=True,
        mem_fraction_static=0.6,
    )
    try:
        base = [
            torch.tensor(get_prompt_logprobs(engine, ids, None)) for ids in prompt_ids
        ]
        lora = {
            module_type: [
                torch.tensor(get_prompt_logprobs(engine, ids, module_type))
                for ids in prompt_ids
            ]
            for module_type in MODULE_TYPES
        }
    finally:
        engine.shutdown()
    return base, lora


class TestQwen3_5LoRAModuleCoverage(CustomTestCase):
    def test_each_module_type_matches_hf(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        prompt_ids = [tokenizer(p)["input_ids"] for p in PROMPTS]

        with tempfile.TemporaryDirectory() as tmp:
            adapter_dirs = {m: f"{tmp}/{m}" for m in MODULE_TYPES}
            hf_base, hf_lora = _hf_reference(prompt_ids, adapter_dirs)
            srt_base, srt_lora = _srt_logprobs(prompt_ids, adapter_dirs)

        hf_base = torch.cat(hf_base).double()
        srt_base = torch.cat(srt_base).double()
        print(f"base |srt - hf| mean: {(srt_base - hf_base).abs().mean():.4f}")
        for module_type in MODULE_TYPES:
            with self.subTest(module_type=module_type):
                hf_shift = torch.cat(hf_lora[module_type]).double() - hf_base
                srt_shift = torch.cat(srt_lora[module_type]).double() - srt_base
                cosine = torch.nn.functional.cosine_similarity(
                    srt_shift, hf_shift, dim=0
                ).item()
                ratio = (srt_shift.norm() / hf_shift.norm()).item()
                print(
                    f"{module_type:>10}: hf shift {hf_shift.abs().mean():.4f}, "
                    f"srt shift {srt_shift.abs().mean():.4f}, "
                    f"cosine {cosine:.4f}, norm ratio {ratio:.4f}"
                )
                self.assertGreater(cosine, MIN_COSINE)
                self.assertGreater(ratio, NORM_RATIO_RANGE[0])
                self.assertLess(ratio, NORM_RATIO_RANGE[1])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    unittest.main()
