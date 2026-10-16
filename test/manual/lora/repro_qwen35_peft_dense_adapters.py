"""Generate a synthetic PEFT adapter for public Qwen3.5-family hybrid MoE models.

No training data or model weights are needed to generate the adapter. It covers
attention, GDN projections and the dense shared expert, with no routed experts.
Start a server with this adapter in --lora-paths, then use --check-url and
--already-loaded to compare next-token logprobs with the base model.
"""

import argparse
import json
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig

TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def post(url, route, payload):
    request = urllib.request.Request(
        url + route, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def generate(args):
    config = AutoConfig.from_pretrained(args.model_path, revision=args.revision)
    config = config.get_text_config()
    hidden = config.hidden_size
    head = config.head_dim
    key = config.linear_num_key_heads * config.linear_key_head_dim
    value = config.linear_num_value_heads * config.linear_value_head_dim
    shared = config.shared_expert_intermediate_size
    generator = torch.Generator().manual_seed(42)
    tensors = {}
    for layer, kind in enumerate(config.layer_types):
        if kind == "linear_attention":
            dimensions = {
                "linear_attn.in_proj_qkv": (hidden, 2 * key + value),
                "linear_attn.in_proj_z": (hidden, value),
                "linear_attn.in_proj_b": (hidden, config.linear_num_value_heads),
                "linear_attn.in_proj_a": (hidden, config.linear_num_value_heads),
                "linear_attn.out_proj": (value, hidden),
            }
        else:
            dimensions = {
                "self_attn.q_proj": (
                    hidden,
                    config.num_attention_heads
                    * head
                    * (2 if config.attn_output_gate else 1),
                ),
                "self_attn.k_proj": (hidden, config.num_key_value_heads * head),
                "self_attn.v_proj": (hidden, config.num_key_value_heads * head),
                "self_attn.o_proj": (config.num_attention_heads * head, hidden),
            }
        dimensions.update(
            {
                "mlp.shared_expert.gate_proj": (hidden, shared),
                "mlp.shared_expert.up_proj": (hidden, shared),
                "mlp.shared_expert.down_proj": (shared, hidden),
            }
        )
        for module, (inputs, outputs) in dimensions.items():
            prefix = f"base_model.model.model.layers.{layer}.{module}"
            tensors[prefix + ".lora_A.weight"] = (
                torch.randn(args.rank, inputs, generator=generator) * 0.02
            )
            tensors[prefix + ".lora_B.weight"] = (
                torch.randn(outputs, args.rank, generator=generator) * 0.002
            )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": args.base_model,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": args.rank,
                "lora_alpha": 8,
                "use_rslora": True,
                "lora_dropout": 0.0,
                "bias": "none",
                "inference_mode": True,
                "target_modules": TARGETS,
            },
            indent=2,
        )
    )
    return len(tensors)


def check(args):
    loaded = False
    if not args.already_loaded:
        result = post(
            args.check_url,
            "/load_lora_adapter",
            {
                "lora_name": args.adapter_name,
                "lora_path": args.server_adapter_path or args.output_dir,
            },
        )
        if not result.get("success", False):
            raise RuntimeError(result)
        loaded = True
    try:
        rows = []
        for prompt in [
            "What is 2 + 2?",
            "Name a primary color.",
            "Complete: The quick brown fox",
        ]:
            outputs = []
            for model in [
                args.served_model,
                args.served_model + ":" + args.adapter_name,
            ]:
                result = post(
                    args.check_url,
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1,
                        "temperature": 0,
                        "logprobs": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                outputs.append(result["choices"][0])
            base, adapted = outputs
            base_lp = base["logprobs"]["content"][0]
            adapted_lp = adapted["logprobs"]["content"][0]
            changed = (
                base_lp["token"] != adapted_lp["token"]
                or abs(base_lp["logprob"] - adapted_lp["logprob"]) > 1e-6
            )
            rows.append(
                {
                    "prompt": prompt,
                    "base": base_lp,
                    "adapter": adapted_lp,
                    "changed": changed,
                }
            )
        if not all(row["changed"] for row in rows):
            raise AssertionError(rows)
        return {"passed": True, "prompts": rows}
    finally:
        if loaded:
            result = post(
                args.check_url, "/unload_lora_adapter", {"lora_name": args.adapter_name}
            )
            if not result.get("success", False):
                raise RuntimeError(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--base-model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument(
        "--revision", default="995ad96eacd98c81ed38be0c5b274b04031597b0"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--check-url")
    parser.add_argument("--served-model", default="qwen")
    parser.add_argument("--adapter-name", default="dense-example")
    parser.add_argument("--server-adapter-path")
    parser.add_argument("--already-loaded", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = {}
    if not args.check_only:
        result["generated_factors"] = generate(args)
    if args.check_url:
        result["check"] = check(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
