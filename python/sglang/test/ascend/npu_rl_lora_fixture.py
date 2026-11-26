"""Shared Qwen3 LoRA adapter fixture for the Ascend RL tests."""

import hashlib
import json
import os

import torch

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH

MODEL = QWEN3_0_6B_WEIGHTS_PATH
LORA_RANK = 8
LORA_ALPHA = 16
# Adapter weight scale: large enough that the adapted model must answer
# differently, small enough to stay in a sane numeric range.
LORA_A_STD = 0.1
LORA_B_STD = 0.1
ADAPTER_SEED = 20261021
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
)
SERVER_ARGS = (
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    "--disable-overlap-schedule",
    "--disable-radix-cache",
    "--context-length",
    "2048",
    "--max-running-requests",
    "8",
    "--mem-fraction-static",
    "0.5",
    "--enable-lora",
    "--max-lora-rank",
    str(LORA_RANK),
    "--lora-target-modules",
    "all",
    # The default csgmv backend is Triton-only; Ascend needs its own kernels.
    "--lora-backend",
    "ascend",
)


def _adapter_seed(name: str) -> int:
    digest = hashlib.sha256(f"{ADAPTER_SEED}:{name}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def _target_linears(model_path: str):
    """``(module_name, in_features, out_features)`` for the adapter targets.

    Building the model on the meta device keeps this cheap: only the config is
    needed to know which projections exist and how wide they are.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in TARGET_MODULES:
            continue
        targets.append((name, module.in_features, module.out_features))
    if not targets:
        raise AssertionError(f"{model_path}: no LoRA target projections found")
    return targets


def adapter_tensors(model_path: str):
    """Deterministic PEFT adapter for ``model_path``, in the same key layout a
    ``save_pretrained`` adapter directory uses."""
    tensors = {}
    for name, in_features, out_features in _target_linears(model_path):
        key = f"base_model.model.{name}"
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_adapter_seed(key + ".lora_A.weight"))
        a = (
            torch.empty(LORA_RANK, in_features, dtype=torch.float32)
            .normal_(0.0, LORA_A_STD, generator=generator)
            .to(torch.bfloat16)
        )
        generator.manual_seed(_adapter_seed(key + ".lora_B.weight"))
        b = (
            torch.empty(out_features, LORA_RANK, dtype=torch.float32)
            .normal_(0.0, LORA_B_STD, generator=generator)
            .to(torch.bfloat16)
        )
        tensors[key + ".lora_A.weight"] = a
        tensors[key + ".lora_B.weight"] = b
    return tensors


def adapter_config() -> dict:
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": MODEL,
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "target_modules": list(TARGET_MODULES),
        "use_dora": False,
    }


def write_adapter(destination: str, tensors, config: dict) -> str:
    from safetensors.torch import save_file

    os.makedirs(destination, exist_ok=True)
    with open(
        os.path.join(destination, "adapter_config.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        os.path.join(destination, "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    return destination
