"""Generate a synthetic PEFT adapter for selected dense modules.

Small random weights expose the LoRA delta without overwhelming the base output.
Validate against a reference forward, not the quality of generated text.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors.torch import save_file

# module name -> (out_features, in_features) from the config, per model family.


def qwen3_5_dims(cfg: dict) -> dict[str, tuple[int, int]]:
    tc = cfg.get("text_config", cfg)
    hidden = tc["hidden_size"]
    head = tc["head_dim"]
    heads = tc["num_attention_heads"]
    kv = tc["num_key_value_heads"]
    gate = tc.get("attn_output_gate", True)
    key_dim = tc["linear_num_key_heads"] * tc["linear_key_head_dim"]
    value_dim = tc["linear_num_value_heads"] * tc["linear_value_head_dim"]
    shared = tc["shared_expert_intermediate_size"]
    return {
        "q_proj": (heads * head * (2 if gate else 1), hidden),
        "k_proj": (kv * head, hidden),
        "v_proj": (kv * head, hidden),
        "o_proj": (hidden, heads * head),
        "in_proj_qkvz": (2 * key_dim + 2 * value_dim, hidden),
        "out_proj": (hidden, value_dim),
        # shared expert MLP (dense wrappers); the routed experts are MoE LoRA
        "gate_proj": (shared, hidden),
        "up_proj": (shared, hidden),
        "down_proj": (hidden, shared),
    }


def qwen3_dense_dims(cfg: dict) -> dict[str, tuple[int, int]]:
    hidden = cfg["hidden_size"]
    head = cfg["head_dim"]
    heads = cfg["num_attention_heads"]
    kv = cfg["num_key_value_heads"]
    inter = cfg["intermediate_size"]
    return {
        "q_proj": (heads * head, hidden),
        "k_proj": (kv * head, hidden),
        "v_proj": (kv * head, hidden),
        "o_proj": (hidden, heads * head),
        "gate_proj": (inter, hidden),
        "up_proj": (inter, hidden),
        "down_proj": (hidden, inter),
    }


def inkling_dims(cfg: dict) -> dict[str, tuple[int, int]]:
    tc = cfg.get("text_config", cfg)
    hidden = tc["hidden_size"]
    head = tc["head_dim"]
    heads = tc["num_attention_heads"]
    kv = tc["num_key_value_heads"]
    d_rel = tc["d_rel"]
    dense_inter = tc["dense_intermediate_size"]
    return {
        "qkvr": (heads * head + 2 * kv * head + heads * d_rel, hidden),
        "wo_ud": (hidden, heads * head),
        "gate_up_proj": (2 * dense_inter, hidden),
        "down_proj": (hidden, dense_inter),
    }


def deepseek_v2_dims(cfg: dict) -> dict[str, tuple[int, int]]:
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv_lora = cfg["kv_lora_rank"]
    shared = cfg["moe_intermediate_size"] * cfg["n_shared_experts"]
    dims = {}
    if cfg.get(
        "q_lora_rank"
    ):  # MLA with a compressed q: the fused replicated q_a/kv_a site
        dims["q_a_proj"] = (cfg["q_lora_rank"], hidden)
        dims["q_b_proj"] = (
            heads * (cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]),
            cfg["q_lora_rank"],
        )
    return dims | {
        "kv_a_proj_with_mqa": (kv_lora + cfg["qk_rope_head_dim"], hidden),
        "kv_b_proj": (heads * (cfg["qk_nope_head_dim"] + cfg["v_head_dim"]), kv_lora),
        "o_proj": (hidden, heads * cfg["v_head_dim"]),
        # shared-expert MLP; layers below first_k_dense_replace use intermediate_size
        "gate_proj": (shared, hidden),
        "up_proj": (shared, hidden),
        "down_proj": (hidden, shared),
    }


def layer_dims(cfg: dict, arch: str, layer_idx: int, module: str, dims: dict):
    if (arch.startswith("DeepseekV") or arch.startswith("Glm4MoeLite")) and module in (
        "gate_proj",
        "up_proj",
        "down_proj",
    ):
        if layer_idx < cfg.get("first_k_dense_replace", 0):
            inter, hidden = cfg["intermediate_size"], cfg["hidden_size"]
            return (hidden, inter) if module == "down_proj" else (inter, hidden)
    return dims[module]


FAMILIES = {
    "DeepseekV2ForCausalLM": deepseek_v2_dims,
    "DeepseekV3ForCausalLM": deepseek_v2_dims,
    "Glm4MoeLiteForCausalLM": deepseek_v2_dims,
    "Qwen3_5MoeForConditionalGeneration": qwen3_5_dims,
    "Qwen3_5MoeForCausalLM": qwen3_5_dims,
    "Qwen3NextForCausalLM": qwen3_5_dims,
    "Qwen3ForCausalLM": qwen3_dense_dims,
    "InklingForConditionalGeneration": inkling_dims,
    "InklingForCausalLM": inkling_dims,
}


def layer_prefixes(cfg: dict, arch: str) -> list[str]:
    tc = cfg.get("text_config", cfg)
    n = tc["num_hidden_layers"]
    base = "model.language_model.layers" if "text_config" in cfg else "model.layers"
    return [f"{base}.{i}" for i in range(n)]


def module_path(prefix: str, module: str, arch: str, cfg: dict | None = None) -> str:
    if arch.startswith("DeepseekV") or arch.startswith("Glm4MoeLite"):
        if module in (
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
        ):
            return f"{prefix}.self_attn.{module}"
        layer_idx = int(prefix.rsplit(".", 1)[1])
        if layer_idx < (cfg or {}).get("first_k_dense_replace", 0):
            return f"{prefix}.mlp.{module}"
        return f"{prefix}.mlp.shared_experts.{module}"
    if arch.startswith("Inkling"):
        if module in ("qkvr", "wo_ud"):
            return f"{prefix}.attn.{module}"
        return f"{prefix}.mlp.{module}"
    if module in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return f"{prefix}.self_attn.{module}"
    if module in ("in_proj_qkvz", "out_proj"):
        return f"{prefix}.linear_attn.{module}"
    if arch == "Qwen3ForCausalLM":
        return f"{prefix}.mlp.{module}"
    return f"{prefix}.mlp.shared_expert.{module}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument(
        "--alpha", type=float, default=None, help="default = rank (scaling 1)"
    )
    parser.add_argument("--targets", required=True, help="comma list of module names")
    parser.add_argument(
        "--layers", default="all", help="'all' or comma list of layer ids"
    )
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(os.path.join(args.model_path, "config.json")) as handle:
        cfg = json.load(handle)
    arch = cfg["architectures"][0]
    dims = FAMILIES[arch](cfg)
    targets = args.targets.split(",")
    unknown = [
        t for t in targets if t not in dims and t not in ("embed_tokens", "lm_head")
    ]
    if unknown:
        raise SystemExit(f"unknown targets {unknown}; known: {sorted(dims)}")

    prefixes = layer_prefixes(cfg, arch)
    if args.layers != "all":
        keep = {int(i) for i in args.layers.split(",")}
        prefixes = [p for i, p in enumerate(prefixes) if i in keep]

    gen = torch.Generator().manual_seed(args.seed)
    tensors = {}
    # Vocabulary sites use PEFT's names: embed_tokens.lora_embedding_{A,B}, lm_head.lora_{A,B}.
    tc = cfg.get("text_config", cfg)
    vocab, hidden = tc["vocab_size"], tc["hidden_size"]
    model_root = "model.language_model" if "text_config" in cfg else "model"
    if "embed_tokens" in targets:
        tensors[
            f"base_model.model.{model_root}.embed_tokens.lora_embedding_A.weight"
        ] = (torch.randn(args.rank, vocab, generator=gen) * args.scale).to(
            torch.bfloat16
        )
        tensors[
            f"base_model.model.{model_root}.embed_tokens.lora_embedding_B.weight"
        ] = (torch.randn(hidden, args.rank, generator=gen) * args.scale).to(
            torch.bfloat16
        )
    if "lm_head" in targets:
        tensors["base_model.model.lm_head.lora_A.weight"] = (
            torch.randn(args.rank, hidden, generator=gen) * args.scale
        ).to(torch.bfloat16)
        tensors["base_model.model.lm_head.lora_B.weight"] = (
            torch.randn(vocab, args.rank, generator=gen) * args.scale
        ).to(torch.bfloat16)
    layer_targets = [t for t in targets if t not in ("embed_tokens", "lm_head")]
    for prefix in prefixes:
        layer_idx = int(prefix.rsplit(".", 1)[1])
        for module in layer_targets:
            out_f, in_f = layer_dims(cfg, arch, layer_idx, module, dims)
            path = module_path(prefix, module, arch, cfg)
            tensors[f"base_model.model.{path}.lora_A.weight"] = (
                torch.randn(args.rank, in_f, generator=gen) * args.scale
            ).to(torch.bfloat16)
            tensors[f"base_model.model.{path}.lora_B.weight"] = (
                torch.randn(out_f, args.rank, generator=gen) * args.scale
            ).to(torch.bfloat16)

    os.makedirs(args.out, exist_ok=True)
    save_file(tensors, os.path.join(args.out, "adapter_model.safetensors"))
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": args.model_path,
        "r": args.rank,
        "lora_alpha": args.alpha if args.alpha is not None else args.rank,
        "lora_dropout": 0.0,
        "target_modules": targets,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(args.out, "adapter_config.json"), "w") as handle:
        json.dump(adapter_config, handle, indent=2)
    print(
        f"wrote {len(tensors)} tensors for {len(prefixes)} layers x {targets} -> {args.out}"
    )


if __name__ == "__main__":
    main()
