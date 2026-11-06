# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""DSpine export normalization and supported serving geometry."""

from copy import deepcopy


def normalize_dspine_config(source: dict) -> dict:
    if source.get("quantization_config"):
        raise ValueError("DSPINE quantized draft exports are not supported")
    if source.get("has_own_lm_head") or source.get("has_lm_head"):
        raise ValueError("DSPINE requires the target output head")
    config = deepcopy(source)
    if source.get("speculators_model_type") == "dspine":
        backbone = source.get("transformer_layer_config")
        if not isinstance(backbone, dict):
            raise ValueError("DSPINE requires transformer_layer_config")
        config = deepcopy(backbone)
        for key in (
            "block_size",
            "dtype",
            "torch_dtype",
            "vocab_size",
            "code_bridge_second_pass",
        ):
            if source.get(key) is not None:
                config[key] = deepcopy(source[key])
        aux = source["aux_hidden_state_layer_ids"]
        if not aux or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in aux
        ):
            raise ValueError(
                "DSPINE auxiliary hidden-state indices must be positive integers"
            )
        cfg = {
            "mask_token_id": source["mask_token_id"],
            "target_layer_ids": [i - 1 for i in aux],
            "sample_from_anchor": source.get("sample_from_anchor", False),
        }
        cfg.update(deepcopy(source.get("dflash_config") or {}))
        for key, default in (
            ("selected_prefix_terminal", False),
            ("selected_prefix_feature_writes", True),
            ("selected_prefix_feature_write_mode", "independent"),
            ("draft_block_causal", True),
        ):
            cfg[key] = source.get(key, cfg.get(key, default))
        for key in (
            "selected_prefix_code_bridge",
            "selected_prefix_last_write_refine",
            "selected_prefix_feature_layers",
            "selected_prefix_feature_reliability",
            "selected_prefix_feature_next_token_rows",
            "dspine_top_k",
            "transition_top_k",
        ):
            if key in source:
                cfg[key] = deepcopy(source[key])
        cfg.setdefault(
            "selected_prefix_feature_layers",
            list(range(1, config["num_hidden_layers"] + 1)),
        )
        bridge = source.get("dspine_config") or {}
        if "selected_prefix_code_bridge" not in cfg:
            code_size = (
                source.get("code_size")
                or source.get("transfer_dim")
                or bridge.get("transfer_rank")
            )
            rank = (
                source.get("rank")
                or source.get("message_dim")
                or bridge.get("message_size")
            )
            if code_size is not None and rank is not None:
                cfg["selected_prefix_code_bridge"] = {
                    "code_size": code_size,
                    "rank": rank,
                    "eps": source.get(
                        "bridge_eps",
                        bridge.get("eps", config.get("rms_norm_eps", 1e-6)),
                    ),
                }
        if "selected_prefix_last_write_refine" not in cfg:
            topk = (
                source.get("head_topk")
                or source.get("transition_top_k")
                or source.get("dspine_top_k")
                or bridge.get("top_k")
            )
            if topk is not None:
                cfg["selected_prefix_last_write_refine"] = {
                    "kind": "exact_sender",
                    "head_topk": topk,
                }
        config["dflash_config"] = cfg

    cfg = config.get("dflash_config") or {}
    if (
        config.get("is_causal") is False
        or cfg.get("causal") is False
        or cfg.get("draft_block_causal") is not True
    ):
        raise ValueError("DSPINE requires causal draft attention")
    if cfg.get("sample_from_anchor", False):
        raise ValueError("DSPINE requires an anchor row")
    config.update(
        architectures=["DSpineDraftModel"], is_causal=True, tie_word_embeddings=False
    )
    config.setdefault("model_type", "qwen3")
    if config["model_type"] != "qwen3":
        raise ValueError("DSPINE currently supports Qwen3 draft backbones only")
    if int(config.get("block_size", 0)) < 2:
        raise ValueError("DSPINE requires its trained block_size >= 2")
    if "vocab_size" not in config:
        if source.get("draft_vocab_size") is None:
            raise ValueError("DSPINE requires an explicit vocabulary size")
        config["vocab_size"] = source["draft_vocab_size"]
    if source.get("draft_vocab_size", config["vocab_size"]) != config["vocab_size"]:
        raise ValueError("DSPINE does not support a reduced draft vocabulary")
    target_hidden = source.get("target_hidden_size")
    if target_hidden is not None and target_hidden != config["hidden_size"]:
        raise ValueError("DSPINE requires matching target and draft hidden widths")
    return config


def validate_dspine_runtime(cfg, draft_config) -> None:
    if not cfg.device.startswith("cuda"):
        raise ValueError("DSPINE currently requires NVIDIA CUDA")
    if cfg.tp_size != 1 or cfg.dp_size != 1 or cfg.pp_size != 1:
        raise ValueError("DSPINE currently requires TP=DP=PP=1")
    if getattr(cfg, "enable_lora", False):
        raise ValueError("DSPINE does not support LoRA-adapted target heads")
    if getattr(cfg, "disaggregation_mode", "null") != "null":
        raise ValueError("DSPINE currently requires colocated serving")
    if cfg.speculative_num_draft_tokens != int(draft_config.block_size):
        raise ValueError(
            "DSPINE speculative_num_draft_tokens must equal the trained block_size"
        )
