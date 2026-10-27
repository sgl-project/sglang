# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""Configuration contract shared by the UBTree exporter and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class UBTreeConfig:
    block_size: int
    candidate_k: int
    tree_budget: int
    supertree_width: int
    rank: int
    depth_bonus: float
    factorizer_hidden_width: int
    target_embedding_dim: int
    depth_count: int
    token_factorizer: str

    @property
    def verify_node_count(self) -> int:
        return self.tree_budget + 1


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return vars(value)


def parse_ubtree_config(config: Any) -> UBTreeConfig:
    """Parse and validate the serving subset of a DeepSpec UBTree config."""

    root = _mapping(config)
    metadata = _mapping(
        root.get("ubtree_config", getattr(config, "ubtree_config", None))
    )
    dflash = _mapping(root.get("dflash_config", getattr(config, "dflash_config", None)))

    block_size = int(
        root.get("block_size", getattr(config, "block_size", 0))
        or dflash.get("block_size", 0)
    )
    parsed = UBTreeConfig(
        block_size=block_size,
        candidate_k=int(metadata.get("candidate_k", 0)),
        tree_budget=int(metadata.get("tree_budget", 0)),
        supertree_width=int(metadata.get("tree_supertree_width", 0)),
        rank=int(metadata.get("rank", 0)),
        depth_bonus=float(metadata.get("tree_depth_bonus", 0.0)),
        factorizer_hidden_width=int(metadata.get("factorizer_hidden_width", 0)),
        target_embedding_dim=int(
            metadata.get(
                "target_embedding_dim",
                root.get("hidden_size", getattr(config, "hidden_size", 0)),
            )
        ),
        depth_count=int(metadata.get("depth_condition_count", 0)),
        token_factorizer=str(metadata.get("token_factorizer", "target_embedding_mlp")),
    )
    positive = {
        "block_size": parsed.block_size,
        "candidate_k": parsed.candidate_k,
        "tree_budget": parsed.tree_budget,
        "tree_supertree_width": parsed.supertree_width,
        "rank": parsed.rank,
    }
    if parsed.token_factorizer == "target_embedding_mlp":
        positive.update(
            factorizer_hidden_width=parsed.factorizer_hidden_width,
            target_embedding_dim=parsed.target_embedding_dim,
        )
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"Invalid UBTree configuration values: {invalid}")
    if parsed.block_size < 2:
        raise ValueError("UBTree block_size must be at least 2")
    if parsed.depth_count != parsed.block_size - 1:
        raise ValueError(
            "UBTree depth_condition_count must equal block_size - 1, got "
            f"{parsed.depth_count} and {parsed.block_size}"
        )
    if parsed.supertree_width > parsed.candidate_k:
        raise ValueError("UBTree tree_supertree_width cannot exceed candidate_k")
    if parsed.tree_budget > parsed.depth_count * parsed.supertree_width:
        raise ValueError("UBTree tree_budget exceeds the generated supertree")
    if parsed.depth_bonus > 0:
        raise ValueError("UBTree tree_depth_bonus must be non-positive")
    if metadata.get("objective") != "kl" or not metadata.get("adds_unary_logits"):
        raise ValueError("UBTree serving requires the trained S1/S3 KL selector")
    if parsed.token_factorizer not in {"lookup", "target_embedding_mlp"}:
        raise ValueError(
            "UBTree serving requires token_factorizer to be lookup or "
            "target_embedding_mlp"
        )
    if metadata.get("hidden_projection_type", "shared") != "shared":
        raise ValueError(
            "The initial SGLang UBTree path supports shared projection only"
        )
    return parsed


def resolve_verify_width(config: UBTreeConfig, requested: int | None) -> int:
    """Select one fixed KDA verification shape for the process lifetime."""
    # Keep the existing default for larger exports without restricting explicit
    # widths to the shapes with tuned kernels.
    width = min(49, config.verify_node_count) if requested is None else requested
    if not 2 <= width <= config.verify_node_count:
        raise ValueError(
            "UBTREE --speculative-num-draft-tokens must be at least 2 and fit "
            f"the checkpoint tree budget ({config.verify_node_count} nodes), got {width}"
        )
    return width


def validate_runtime_config(cfg, target_backends: tuple[str, ...]) -> None:
    """Validate the fixed-width, single-request Ling serving contract."""
    if not cfg.device.startswith("cuda"):
        raise ValueError("UBTREE currently requires a CUDA device")
    if cfg.pp_size != 1 or cfg.dp_size != 1:
        raise ValueError("UBTREE currently requires --pp-size 1 and --dp-size 1")
    if cfg.max_running_requests != 1:
        raise ValueError("UBTREE currently requires --max-running-requests 1")
    if cfg.disaggregation_mode != "null" or cfg.enable_pd_role_switch:
        raise ValueError("UBTREE does not support PD disaggregation or role switching")
    # nv_cutedsl dispatches Ling KDA tree verify to Triton. Helion only supports
    # per-node state snapshots, not UBTree's compact factor caches.
    if cfg.linear_attn_verify_backend not in (None, "triton", "nv_cutedsl"):
        raise ValueError(
            "UBTREE factorized KDA verification requires "
            "--linear-attn-verify-backend triton (nv_cutedsl also falls back to Triton)"
        )
    if cfg.enable_strict_thinking:
        raise ValueError(
            "UBTREE does not support --enable-strict-thinking because it requires "
            "grammar decoding"
        )
    if cfg.speculative_draft_window_size is not None:
        raise ValueError("UBTREE does not support --speculative-draft-window-size")
    if cfg.speculative_draft_sink_size is not None:
        raise ValueError("UBTREE does not support --speculative-draft-sink-size")
    if cfg.speculative_use_rejection_sampling or cfg.speculative_use_block_verification:
        raise ValueError("UBTREE currently supports greedy tree verification only")
    if cfg.speculative_adaptive:
        raise ValueError(
            "UBTREE requires a fixed verify width; adaptive steps are unsupported"
        )
    if not target_backends or set(target_backends) != {"fa3"}:
        raise ValueError("UBTREE target attention requires FA3")
    if cfg.speculative_draft_attention_backend != "fa3":
        raise ValueError("UBTREE draft attention requires FA3")
    if cfg.page_size != 1:
        raise ValueError("UBTREE with FA3 requires --page-size 1")
    decode = str(cfg.cuda_graph_config.decode.backend).lower()
    if not cfg.disable_cuda_graph and "disabled" not in decode:
        if "full" not in decode:
            raise ValueError("UBTREE requires full decode CUDA Graphs")
        if "disabled" not in str(cfg.cuda_graph_config.prefill.backend).lower():
            raise ValueError("UBTREE requires --cuda-graph-backend-prefill disabled")
        if not cfg.disable_cuda_graph_padding:
            raise ValueError("UBTREE requires --disable-cuda-graph-padding")
        if tuple(cfg.cuda_graph_config.decode.bs or ()) != (1,):
            raise ValueError("UBTREE requires --cuda-graph-bs-decode 1")
