"""
Create a minimal embedding LoRA adapter for testing SGLang's embedding LoRA implementation.

Usage:
    python create_embedding_lora_adapter.py
    huggingface-cli upload YOUR_USERNAME/sglang_embedding_lora_test_adapter ./sglang_embedding_lora_test_adapter
"""

import argparse
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_MODULES = [
    "embed_tokens",
    "lm_head",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def create_embedding_lora_adapter(
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir: str = "./sglang_embedding_lora_test_adapter",
    rank: int = 8,
    lora_alpha: int = 16,
):
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # embed_tokens must be in target_modules (not modules_to_save) for SGLang's
    # run_lora_a_embedding() to work - it expects LoRA A/B matrices, not full weights
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"Applying LoRA with target_modules: {lora_config.target_modules}")
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    print(f"\nSaving adapter to: {output_dir}")
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Generate README
    readme = generate_readme(base_model, rank, lora_alpha, peft_model)
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)

    print("Done!")


def generate_readme(base_model: str, rank: int, lora_alpha: int, peft_model) -> str:
    # Collect weight shapes
    weight_shapes = []
    for name, param in peft_model.named_parameters():
        if "lora_" in name and param.requires_grad:
            # Clean up name: model.base_model.model.model.embed_tokens.lora_A.default.weight -> embed_tokens.lora_A
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part in TARGET_MODULES:
                    layer_name = part
                    lora_type = parts[i + 1]  # lora_A or lora_B
                    weight_shapes.append(
                        f"{layer_name}.{lora_type}: {tuple(param.shape)}"
                    )
                    break

    # Dedupe and sort
    weight_shapes = sorted(set(weight_shapes))

    readme = f"""# Test LoRA Adapter for SGLang Embedding LoRA

This is a test LoRA adapter (randomly initialized without tuning) for testing SGLang's embedding LoRA implementation.

## Configuration

- **Base model:** `{base_model}`
- **LoRA rank (r):** {rank}
- **LoRA alpha:** {lora_alpha}
- **Target modules:** {", ".join(TARGET_MODULES)}

## Weight Shapes

```
{chr(10).join(weight_shapes)}
```

## Purpose

This adapter tests that SGLang's `ChunkedSgmvLoRABackend.run_lora_a_embedding()` correctly handles embedding LoRA layers (`embed_tokens`, `lm_head`).

**Key:** `embed_tokens` is in `target_modules` (LoRA decomposition), NOT `modules_to_save` (full weights).

## Usage with SGLang

```python
# This adapter contains randomly initialized weights for testing purposes only.
# Used by: test/srt/lora/test_lora_hf_sgl_logprob_diff.py
```

## Created with

```bash
python scripts/playground/lora/create_embedding_lora_adapter.py
```
"""
    return readme


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./sglang_embedding_lora_test_adapter"
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    args = parser.parse_args()
    create_embedding_lora_adapter(
        base_model=args.base_model,
        output_dir=args.output_dir,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
    )
