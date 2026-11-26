"""Checkpoint-to-runtime names shared by the DeepSeek V4 weight and metadata loaders."""


def remap_v4_weight_name(
    name: str,
    is_nextn: bool = False,
    num_hidden_layers: int | None = None,
) -> str:
    if name.startswith("vision."):
        return name.replace(".attn.wqkv.", ".attn.qkv_proj.").replace(
            ".attn.wo.", ".attn.proj."
        )
    if name.startswith(("aligner.", "image_")):
        return name
    if name.startswith("embed."):
        return "model.embed_tokens." + name.removeprefix("embed.")
    if name.startswith("head."):
        return "lm_head." + name.removeprefix("head.")
    if name == "norm.weight":
        return "model.norm.weight"
    if name.startswith("hc_head_"):
        return "model." + name

    if is_nextn and name.startswith("mtp."):
        parts = name.split(".", 2)
        if len(parts) >= 3:
            rest = parts[2]
            nextn_spec_prefixes = [
                "e_proj",
                "h_proj",
                "emb",
                "enorm",
                "hnorm",
                "norm",
                "head",
                "hc_head",
            ]
            is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
            if is_nextn_spec:
                if rest.startswith("emb.tok_emb"):
                    rest = rest.replace("emb.tok_emb", "embed_tokens")
                elif rest == "norm.weight":
                    rest = "shared_head.norm.weight"
                elif rest.startswith("head."):
                    rest = "shared_head.head.weight"
                elif rest == "e_proj.scale":
                    rest = "e_proj.weight_scale_inv"
                elif rest == "h_proj.scale":
                    rest = "h_proj.weight_scale_inv"
            name = f"model.layers.{num_hidden_layers}." + rest

    if name.startswith("layers."):
        name = "model." + name
    name = name.replace(".attn.", ".self_attn.")
    name = name.replace(".ffn.", ".mlp.")
    name = name.replace(".attn_norm.", ".input_layernorm.")
    name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

    if "self_attn" in name and name.endswith(".scale"):
        name = name.removesuffix(".scale") + ".weight_scale_inv"
    if ".engram.wkv." in name and name.endswith(".scale"):
        name = name.removesuffix(".scale") + ".weight_scale_inv"

    name = name.replace(".gate.tid2eid", ".topk.tid2eid")
    name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
    name = name.replace(".w1.", ".gate_proj.")
    name = name.replace(".w2.", ".down_proj.")
    name = name.replace(".w3.", ".up_proj.")
    if "mlp" in name and name.endswith(".scale"):
        name = name.removesuffix(".scale") + ".weight_scale_inv"

    return name


def remap_dspark_export_name(
    name: str, *, for_quant_config: bool = False
) -> str | None:
    """Map a checkpoint name without consulting model state.

    The loader selects vocabulary stages and filters disabled confidence-head
    weights. ModelSlim uses the same names to add quantization-config aliases.
    Config aliases retain scale suffixes and rotary entries and omit vocabulary
    weights, matching the original ModelSlim configuration semantics.
    """
    if not name.startswith("mtp."):
        return None
    if not for_quant_config and "rotary_emb.inv_freq" in name:
        return None
    parts = name.split(".", 2)
    if len(parts) < 3:
        return None
    stage_id, rest = parts[1], parts[2]
    if not stage_id.isdigit():
        return None
    if not for_quant_config:
        # Preserve the loader's integer-stage validation; isdigit() alone also
        # accepts characters such as superscript digits that int() rejects.
        int(stage_id)

    if rest.startswith(("embed.", "embed_tokens.", "head.", "lm_head.")):
        if not for_quant_config:
            if rest in ("embed.weight", "embed_tokens.weight"):
                return "embed_tokens.weight"
            if rest in ("head.weight", "lm_head.weight"):
                return "lm_head.weight"
        return None
    if rest.startswith("markov_head."):
        rest = rest.replace("markov_head.embed.", "markov_head.markov_w1.", 1)
        return rest.replace("markov_head.head.", "markov_head.markov_w2.", 1)
    if rest.startswith("confidence_head."):
        return rest

    mapped_rest = rest
    if mapped_rest.startswith("attn."):
        mapped_rest = "self_attn." + mapped_rest.removeprefix("attn.")
    elif mapped_rest.startswith("ffn."):
        mapped_rest = "mlp." + mapped_rest.removeprefix("ffn.")
    elif mapped_rest.startswith("attn_norm."):
        mapped_rest = "input_layernorm." + mapped_rest.removeprefix("attn_norm.")
    elif mapped_rest.startswith("ffn_norm."):
        mapped_rest = "post_attention_layernorm." + mapped_rest.removeprefix(
            "ffn_norm."
        )
    mapped_rest = mapped_rest.replace(".w1.", ".gate_proj.")
    mapped_rest = mapped_rest.replace(".w2.", ".down_proj.")
    mapped_rest = mapped_rest.replace(".w3.", ".up_proj.")
    mapped_rest = mapped_rest.replace(".gate.tid2eid", ".topk.tid2eid")
    mapped_rest = mapped_rest.replace(".gate.bias", ".gate.e_score_correction_bias")
    if not for_quant_config and mapped_rest.endswith(".scale"):
        mapped_rest = mapped_rest.removesuffix(".scale") + ".weight_scale_inv"
    return f"stages.{stage_id}.{mapped_rest}"
