import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

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


def format_alpaca_prompt(example):
    """Format alpaca dataset example into a prompt."""
    if example.get("input") and example["input"].strip():
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"


def train_embedding_lora_adapter(
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir: str = "./sglang_embedding_lora_test_adapter",
    num_train_steps: int = 500,
    rank: int = 8,
    lora_alpha: int = 16,
):
    print(f"Training embedding LoRA adapter")
    print(f"  Base model: {base_model}")
    print(f"  Output dir: {output_dir}")
    print(f"  Training steps: {num_train_steps}")
    print(f"  Rank: {rank}, Alpha: {lora_alpha}")
    print(f"  Target modules: {TARGET_MODULES}")
    print()

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # embed_tokens must be in target_modules (not modules_to_save) for SGLang's
    # run_lora_a_embedding() to work correctly
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nLoading alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:2000]")
    dataset = dataset.map(
        lambda x: {"text": format_alpaca_prompt(x)},
        remove_columns=dataset.column_names,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Example:\n{dataset[0]['text'][:200]}...")

    sft_config = SFTConfig(
        output_dir=os.path.join(output_dir, "checkpoints"),
        max_steps=num_train_steps,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=25,
        save_steps=num_train_steps,
        save_total_limit=1,
        fp16=True,
        report_to="none",
        max_length=512,
        dataset_text_field="text",
    )

    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
    )
    trainer.train()

    print(f"\nSaving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    readme = generate_readme(base_model, rank, lora_alpha, num_train_steps, model)
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)

    print(f"\nDone! Adapter saved to: {output_dir}")
    print("\nTo upload to HuggingFace:")
    print(
        f"  huggingface-cli upload YOUR_USERNAME/sglang_embedding_lora_test_adapter {output_dir}"
    )

    print("\nTesting generation...")
    test_generation(model, tokenizer)


def test_generation(model, tokenizer):
    """Test that the model produces coherent outputs."""
    prompts = [
        "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
        "### Instruction:\nWrite a short greeting.\n\n### Response:\n",
    ]

    model.eval()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"Response: {response[len(prompt):][:100]}...")


def generate_readme(base_model, rank, lora_alpha, num_train_steps, peft_model):
    """Generate README for the adapter."""
    weight_shapes = []
    for name, param in peft_model.named_parameters():
        if "lora_" in name and param.requires_grad:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part in TARGET_MODULES:
                    layer_name = part
                    lora_type = parts[i + 1]
                    weight_shapes.append(
                        f"{layer_name}.{lora_type}: {tuple(param.shape)}"
                    )
                    break

    weight_shapes = sorted(set(weight_shapes))

    return f"""# Trained LoRA Adapter for SGLang Embedding LoRA Testing

This is a fine-tuned LoRA adapter for testing SGLang's embedding LoRA implementation.

## Configuration

- **Base model:** `{base_model}`
- **LoRA rank (r):** {rank}
- **LoRA alpha:** {lora_alpha}
- **Target modules:** {", ".join(TARGET_MODULES)}
- **Training steps:** {num_train_steps}
- **Training data:** alpaca dataset

## Weight Shapes

```
{chr(10).join(weight_shapes)}
```

## Purpose

This adapter tests that SGLang's `ChunkedSgmvLoRABackend.run_lora_a_embedding()` correctly
handles embedding LoRA layers (`embed_tokens`, `lm_head`).

**Key:** `embed_tokens` is in `target_modules` (LoRA decomposition), NOT `modules_to_save` (full weights).

## Usage with SGLang

```python
# Used by: test/srt/lora/test_lora_hf_sgl_logprob_diff.py
```

## Created with

```bash
python scripts/playground/lora/train_embedding_lora_adapter.py --num_train_steps {num_train_steps}
```
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./sglang_embedding_lora_test_adapter"
    )
    parser.add_argument("--num_train_steps", type=int, default=500)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    args = parser.parse_args()
    train_embedding_lora_adapter(
        base_model=args.base_model,
        output_dir=args.output_dir,
        num_train_steps=args.num_train_steps,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
    )
