"""Native Direct control with symbolic tensors; conditional costs only.

Runs unchanged attention _run_layers, Direct dispatch/_exchange/run_ffn_step/
run_layer and native two-stage executor. GPU initialization is outside scope.
"""

import copy
import math
from pathlib import Path

from ...native_config import NativeAFDConfig
from .qwen_profile import DeploymentLayout, ModelGeometry, decoder_shapes
from .source import verify_source

BASE_FIELDS = ("model", "mode", "ffn_ranks", "batch_size", "effective_kv", "layers")


class DirectMapping:
    """Qwen-specific signatures; no model defaults enter the generic provider."""

    def __init__(self, c):
        required = set(BASE_FIELDS) | {
            "routing",
            "direct_group_alpha_ns",
            "direct_link_bandwidth_Bps",
            "direct_resource_policy",
        }
        optional = {"ffn_hardware", "ffn_recipe", "native_config", "model_config"}
        if not required <= c.keys() or c.keys() - required - optional:
            raise ValueError("DIRECT_PROFILE_CONFIG_UNSUPPORTED")
        if c["routing"] != "direct":
            raise ValueError("DIRECT_ROUTING_MISMATCH")
        if type(c["ffn_ranks"]) is not int:
            raise ValueError("DIRECT_PROFILE_WIDTH")
        for name in ("direct_group_alpha_ns", "direct_link_bandwidth_Bps"):
            value = c[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (not math.isfinite(value))
                or (value <= 0)
            ):
                raise ValueError("DIRECT_PROFILE_PARAMETER_INVALID: " + name)
        if c["direct_resource_policy"] not in (
            "shared_endpoint_nic",
            "independent_links",
        ):
            raise ValueError("DIRECT_RESOURCE_POLICY_UNSUPPORTED")
        for name in {"ffn_hardware", "ffn_recipe"} & c.keys():
            if not isinstance(c[name], str) or not c[name]:
                raise ValueError("DIRECT_PROFILE_PARAMETER_INVALID: " + name)
        self.config = copy.deepcopy(c)
        if c["mode"] != "afd" or c["ffn_ranks"] < 1:
            raise ValueError("DIRECT_PROFILE_WIDTH")
        native = c.get("native_config", {})
        if not isinstance(native, dict):
            raise ValueError("DIRECT_NATIVE_CONFIG_REQUIRED")
        selected = NativeAFDConfig.parse(native) if "native_config" in c else None
        if selected and selected.routing != "direct":
            raise ValueError("DIRECT_ROUTING_MISMATCH")
        A = selected.attention_ranks if selected else 8
        # Old cost documents identify the qualified eight-A profile. New
        # deployments carry the original native configuration explicitly.
        if "native_config" not in c:
            A = DeploymentLayout(8, 2, c["ffn_ranks"], c["batch_size"]).attention_ranks
        stages = selected.stages if selected else 2
        if (
            type(A) is not int
            or A < 1
            or type(stages) is not int
            or stages < 1
            or native.get("lanes", c["ffn_ranks"]) != c["ffn_ranks"]
        ):
            raise ValueError("DIRECT_NATIVE_LAYOUT_MISMATCH")
        self.layout = DeploymentLayout(A, stages, c["ffn_ranks"], c["batch_size"])
        model = c.get("model_config")
        if model is None:
            if c["model"] != "qwen3_235b":
                raise ValueError("DIRECT_MODEL_CONFIG_REQUIRED")
            self.geometry = ModelGeometry()
        else:
            if not isinstance(model, dict):
                raise ValueError("DIRECT_MODEL_CONFIG_REQUIRED")
            if (
                model.get("architectures", ["Qwen3MoeForCausalLM"])
                != ["Qwen3MoeForCausalLM"]
                or model.get("mlp_only_layers", [])
                or model.get("decoder_sparse_step", 1) != 1
                or model.get("num_shared_experts", 0)
            ):
                raise ValueError("DIRECT_MODEL_STRUCTURE_UNSUPPORTED")
            self.geometry = ModelGeometry(
                hidden=model["hidden_size"],
                layers=model["num_hidden_layers"],
                q_heads=model["num_attention_heads"],
                kv_heads=model["num_key_value_heads"],
                head_dim=model["head_dim"],
                experts=model["num_experts"],
                topk=model["num_experts_per_tok"],
                vocab=model["vocab_size"],
            )
        g = self.geometry
        if (
            any(type(v) is not int or v < 1 for v in vars(g).values())
            or c["layers"] != g.layers
            or g.topk > g.experts
            or g.experts % c["ffn_ranks"]
            or g.vocab % A
            or type(c["batch_size"]) is not int
            or c["batch_size"] < 1
            or c["batch_size"] % (A * stages)
            or type(c["effective_kv"]) is not int
            or c["effective_kv"] < 1
        ):
            raise ValueError("DIRECT_MODEL_OR_BATCH_MISMATCH")
        self.rows = self.layout.stage_rows
        self.shapes = decoder_shapes(c, g, self.layout)
        if "pack" in self.shapes:
            del self.shapes["pack"]
        self.shapes["routed_experts_local_combine"].update(
            ep=c["ffn_ranks"],
            local_experts=g.experts // c["ffn_ranks"],
            recipe=c.get("ffn_recipe", "common_block128_ue8m0"),
        )
        self.shapes["ffn_step_submit"] = dict(
            scope="once_per_F_actor_per_step", hardware=c.get("ffn_hardware", "GB300")
        )
        for k in ("all_gatherv", "reduce_scatterv", "p2p_A2F", "p2p_F2A"):
            del self.shapes[k]
        self.shapes["router"]["m"] = self.rows
        self.shapes["topk"]["rows"] = self.rows
        self.shapes.update(
            direct_pack=dict(
                rows=self.rows,
                hidden=g.hidden,
                topk=g.topk,
                boundary="3zero_fill+3copy fixed payload",
            ),
            direct_combine=dict(
                rows=self.rows,
                hidden=g.hidden,
                F=c["ffn_ranks"],
                boundary="FP32 sum of BF16 partials plus BF16 output copy",
            ),
            direct_group_startup=dict(
                alpha_ns=c["direct_group_alpha_ns"],
                scope="once per rank grouped exchange; paired endpoints concurrent",
            ),
            direct_A2F_message=dict(
                bytes=self.rows * (g.hidden * 2 + g.topk * 8 + g.topk * 4),
                bandwidth_Bps=c["direct_link_bandwidth_Bps"],
                scope="one A/F peer hidden+ids+weights",
            ),
            direct_F2A_message=dict(
                bytes=self.rows * g.hidden * 2,
                bandwidth_Bps=c["direct_link_bandwidth_Bps"],
                scope="one F/A peer weighted BF16 partial",
            ),
        )

    def expected_shapes(self):
        return copy.deepcopy(self.shapes)

    def resolve(self, kind, metadata):
        if kind == "attention_module":
            if metadata["rows"] != self.rows:
                raise ValueError("DIRECT_ATTENTION_ROWS")
            return [
                "input_norm_first" if metadata["layer"] == 0 else "input_norm_residual",
                "qkv",
                "qk_norm_rope",
                "trtllm_decode",
                "o",
                "post_attention_norm",
            ]
        if kind == "graph_replay_overhead":
            return (
                [kind, "trt_metadata"]
                if metadata.get("role") != "F"
                else ["ffn_step_submit"]
            )
        if kind not in self.shapes:
            raise ValueError("DIRECT_UNKNOWN_QUERY " + kind)
        shape = self.shapes[kind]
        if "rows" in metadata and metadata["rows"] != shape.get("rows", shape.get("m")):
            raise ValueError("DIRECT_ROWS_MISMATCH " + kind)
        for k in ("bytes", "world", "ep", "local_experts"):
            if k in metadata and k in shape and (metadata[k] != shape[k]):
                raise ValueError("DIRECT_SHAPE_MISMATCH " + kind)
        return [kind]

    def event_call(self, event):
        meta = dict(event["metadata"])
        actor = event["actor"]
        if actor.startswith(("A", "F")):
            meta["role"] = actor[0]
        return (event["kind"], meta)


class DirectAdapter:
    """Replay native Direct control with symbolic devices and paired events."""

    def __init__(self, source_root, mode):
        if mode != "afd":
            raise ValueError("AFD_ADAPTER_MODE_UNSUPPORTED: " + str(mode))
        if source_root is None:
            raise ValueError("AFD_SOURCE_ROOT_REQUIRED")
        self.root = Path(source_root)
        self.mode = mode

    def verify(self):
        return verify_source(self.root, mode=self.mode, routing="direct")

    def template_config(self, *, batch_size, effective_kv, ffn_ranks):
        return dict(
            model="qwen3_235b",
            mode="afd",
            routing="direct",
            ffn_ranks=ffn_ranks,
            batch_size=batch_size,
            effective_kv=effective_kv,
            layers=94,
            direct_group_alpha_ns=10000,
            direct_link_bandwidth_Bps=900000000000.0,
            direct_resource_policy="shared_endpoint_nic",
        )

    def cost_mapping(self, config):
        return DirectMapping(config)

    def execute(self, provider):
        from .direct_execution import execute_native

        return execute_native(self.root, provider, self.verify())
