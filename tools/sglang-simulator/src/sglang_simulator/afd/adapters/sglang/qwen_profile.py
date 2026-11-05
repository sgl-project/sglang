"""Cost geometry and billing boundaries for the pinned Qwen/none profile.

This module is owned by the source adapter, not by the generic cost provider.
Its restrictions are qualifications of this implementation, not AFD laws.
"""

import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelGeometry:
    hidden: int = 4096
    layers: int = 94
    q_heads: int = 64
    kv_heads: int = 4
    head_dim: int = 128
    experts: int = 128
    topk: int = 8
    vocab: int = 151936


@dataclass(frozen=True)
class DeploymentLayout:
    attention_ranks: int
    stages: int
    ffn_ranks: int
    batch_size: int

    @property
    def stage_rows(self):
        return self.batch_size // (self.attention_ranks * self.stages)

    @property
    def global_stage_rows(self):
        return self.batch_size // self.stages


def layout(config):
    mode = config["mode"]
    # These widths/stages qualify the pinned native implementation only.
    if mode != "afd":
        raise ValueError("AFD_COST_MODE_UNSUPPORTED")
    world = 8
    return DeploymentLayout(world, 2, config["ffn_ranks"], config["batch_size"])


def shapes(config):
    if set(config) != {
        "model",
        "mode",
        "ffn_ranks",
        "batch_size",
        "effective_kv",
        "layers",
    }:
        raise ValueError(
            "AFD_PROFILE_CONFIG_UNSUPPORTED: use a matching execution adapter for other layouts/backends"
        )
    geometry = ModelGeometry()
    topology = layout(config)
    mode, batch, kv = (config[key] for key in ("mode", "batch_size", "effective_kv"))
    if mode != "afd":
        raise ValueError("AFD_COST_MODE_UNSUPPORTED")
    world = topology.attention_ranks
    ffn = config["ffn_ranks"]
    if type(config["ffn_ranks"]) is not int:
        raise ValueError("AFD_COST_FFN_WIDTH_MISMATCH")
    if ffn not in (2, 4, 8):
        raise ValueError("AFD_COST_WIDTH_UNSUPPORTED")
    if type(batch) is not int or batch <= 0 or batch % (world * topology.stages):
        raise ValueError("AFD_COST_BATCH_INVALID")
    if type(kv) is not int or kv <= 0:
        raise ValueError("AFD_COST_KV_INVALID")
    if (
        config["model"] != "qwen3_235b"
        or type(config["layers"]) is not int
        or config["layers"] != geometry.layers
    ):
        raise ValueError("AFD_MODEL_PROFILE_UNSUPPORTED")
    rows = topology.stage_rows
    local = batch // world
    global_rows = topology.global_stage_rows
    result = {
        "input_norm_first": {"rows": rows, "hidden": geometry.hidden},
        "input_norm_residual": {"rows": rows, "hidden": geometry.hidden},
        "post_attention_norm": {"rows": rows, "hidden": geometry.hidden},
        "qkv": {
            "m": rows,
            "n": (geometry.q_heads + 2 * geometry.kv_heads) * geometry.head_dim,
            "k": geometry.hidden,
            "recipe": "ModelOpt_static_FP8_including_activation_quant",
        },
        "o": {
            "m": rows,
            "n": geometry.hidden,
            "k": geometry.q_heads * geometry.head_dim,
            "recipe": "ModelOpt_static_FP8_including_activation_quant",
        },
        "qk_norm_rope": {
            "rows": rows,
            "q_heads": geometry.q_heads,
            "kv_heads": geometry.kv_heads,
            "head_dim": geometry.head_dim,
        },
        "trtllm_decode": {
            "rows": rows,
            "effective_kv": kv,
            "q_heads": geometry.q_heads,
            "kv_heads": geometry.kv_heads,
            "head_dim": geometry.head_dim,
            "page_size": 64,
            "recipe": "BF16_io_FP8_Q_KV_including_Qcast_KVwrite",
        },
        "router": {
            "m": global_rows,
            "n": geometry.experts,
            "k": geometry.hidden,
            "dtype": "BF16",
        },
        "topk": {
            "rows": global_rows,
            "experts": geometry.experts,
            "topk": geometry.topk,
        },
        "routed_experts_local_combine": {
            "rows": global_rows,
            "ep": ffn,
            "local_experts": geometry.experts // ffn,
            "recipe": "common_block128_ue8m0",
            "route": "balanced_diagnostic",
            "boundary": "experts_and_local_combine_only",
        },
        "all_gatherv": {
            "world": ffn,
            "global_bytes": global_rows * geometry.hidden * 2,
            "dtype": "BF16",
            "primitive": "equal_count_AllGather",
        },
        "reduce_scatterv": {
            "world": ffn,
            "global_bytes": global_rows * geometry.hidden * 2,
            "dtype": "BF16",
            "primitive": "ordered_grouped_Reduce_SUM",
        },
        "embedding": {"rows": local, "hidden": geometry.hidden},
        "final_norm": {"rows": local, "hidden": geometry.hidden},
        "lm_hidden_gather": {
            "world": world,
            "global_bytes": batch * geometry.hidden * 2,
            "dtype": "BF16",
        },
        "lm_head": {
            "m": batch,
            "n": geometry.vocab // world,
            "k": geometry.hidden,
            "dtype": "BF16",
        },
        "logits_exchange": {
            "world": world,
            "per_rank_input_bytes": batch * (geometry.vocab // world) * 2,
            "dtype": "BF16",
        },
        "logits_postprocess": {"rows": local, "vocab": geometry.vocab},
        "sampling": {
            "rows": local,
            "vocab": geometry.vocab,
            "algorithm": "greedy_argmax",
        },
        "graph_replay_overhead": {
            "roles": ["A", "F"],
            "scope": "once_per_actor_per_step",
        },
        "trt_metadata": {
            "rows": local,
            "stages": 2,
            "scope": "once_per_A_actor_per_step",
        },
    }
    result.update(
        stage_join={"rows": local, "hidden": geometry.hidden},
        p2p_A2F={"bytes_per_peer": rows * geometry.hidden * 2, "direction": "A2F"},
        p2p_F2A={"bytes_per_peer": rows * geometry.hidden * 2, "direction": "F2A"},
    )
    if ffn < 8:
        result["pack"] = {
            "rows": global_rows // ffn,
            "hidden": geometry.hidden,
            "boundary": "zero_and_copy_lane_inputs",
        }
    return result


class QwenCostMapping:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self._shapes = shapes(self.config)
        self.layout = layout(self.config)

    def expected_shapes(self):
        return copy.deepcopy(self._shapes)

    def event_call(self, event):
        kind, metadata = event["kind"], dict(event["metadata"])
        if kind in ("A2F", "F2A"):
            metadata["direction"], kind = kind, "p2p"
        if event["actor"].startswith(("A", "F")):
            metadata["role"] = event["actor"][0]
        return kind, metadata

    def resolve(self, kind, metadata):
        self._validate_call(kind, metadata)
        if kind == "attention_module":
            keys = [
                "input_norm_first" if metadata["layer"] == 0 else "input_norm_residual",
                "qkv",
                "qk_norm_rope",
                "trtllm_decode",
                "o",
                "post_attention_norm",
            ]
        elif kind == "router_deepgemm_module":
            keys = ["router", "topk", "routed_experts_local_combine"]
        elif kind == "p2p":
            keys = ["p2p_" + metadata["direction"]]
        else:
            keys = [kind]
        if kind == "graph_replay_overhead" and metadata.get("role") != "F":
            keys += ["trt_metadata"]
        return keys

    def _validate_call(self, kind, metadata):
        config = self.config
        global_rows = self.layout.global_stage_rows
        if kind == "attention_module":
            if config["mode"] != "afd" or metadata["rows"] != self.layout.stage_rows:
                raise ValueError("AFD_COST_CALL_SHAPE_MISMATCH: attention_module")
            return
        if kind == "router_deepgemm_module":
            if (
                metadata["global_stage_tokens"] != global_rows
                or metadata["ep"] != config["ffn_ranks"]
                or metadata["local_experts"]
                != self._shapes["routed_experts_local_combine"]["local_experts"]
            ):
                raise ValueError("AFD_COST_CALL_SHAPE_MISMATCH: FFN module")
            if (
                "recipe" in metadata
                and metadata["recipe"]
                != self._shapes["routed_experts_local_combine"]["recipe"]
            ):
                raise ValueError("AFD_COST_CALL_RECIPE_MISMATCH: FFN module")
            return
        key = "p2p_" + metadata["direction"] if kind == "p2p" else kind
        shape = self._shapes[key]
        if "rows" in metadata:
            expected_rows = shape["rows"] if "rows" in shape else shape["m"]
            if metadata["rows"] != expected_rows:
                raise ValueError(f"AFD_COST_CALL_SHAPE_MISMATCH: {key} rows")
        for field in ("world", "ep", "local_experts", "k", "n"):
            if field in metadata and field in shape and metadata[field] != shape[field]:
                raise ValueError(f"AFD_COST_CALL_SHAPE_MISMATCH: {key} {field}")
        if "bytes" in metadata and metadata["bytes"] != shape["bytes_per_peer"]:
            raise ValueError("AFD_COST_CALL_SHAPE_MISMATCH: P2P bytes")
        if (
            "global_stage_tokens" in metadata
            and metadata["global_stage_tokens"] != global_rows
        ):
            raise ValueError("AFD_COST_CALL_SHAPE_MISMATCH: collective rows")
        if (
            "global_rows" in metadata
            and metadata["global_rows"] != config["batch_size"]
        ):
            raise ValueError("AFD_COST_CALL_SHAPE_MISMATCH: global rows")
