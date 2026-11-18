"""Loading tests for the transformers-native (>=5.13) ZAYA1 checkpoint format.

Zyphra ships ZAYA1 in two checkpoint layouts. The legacy Megatron-style
export (ZAYA1-base, ZAYA1-8B-legacy) matches SGLang's internal module tree
one-to-one: 2L interleaved decoder layers (even = attention, odd = MoE) and
per-expert ``local_experts.<i>.linear_fc{1,2}`` tensors. The native layout
(ZAYA1-8B, ZAYA1-74B-preview, defined by ``transformers`` >= v5.13) folds
each attention + MoE pair into one hybrid layer ``model.layers.k`` and fuses
expert weights into 3D ``mlp.experts.{gate_up_proj,down_proj}`` tensors, so
native layer ``k`` must be split onto internal layers ``(2k, 2k + 1)`` at
load time.

Bug mechanism guarded here (black box): feeding a native-format config and
checkpoint into SGLang today constructs a model with the wrong geometry and
then skips every native tensor with "no matching parameter", producing an
unusable model instead of a correct load or a fast failure.

The residual-scaling association is the trickiest derived property: native
checkpoints attach scaling blocks exit-style (``post_attention_*`` /
``post_mlp_*`` on the layer they follow) while SGLang attaches them
entry-style (``res_scale`` on the layer they precede), so every block shifts
one position later: the model-input scale lands on internal layer 0, layer
``k``'s post-attention scale on internal layer ``2k + 1``, layer ``k``'s
post-MLP scale on internal layer ``2k + 2``, and the last layer's post-MLP
scale on the model-level ``res_scale``.
"""

import os
import unittest

import torch

from sglang.srt.configs.zaya import ZayaConfig
from sglang.srt.models.zaya import ZayaForCausalLM
from sglang.srt.runtime_context import get_context, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _ensure_dist_initialized() -> None:
    """Minimal single-rank gloo distributed environment plus the SGLang
    model-parallel groups (TP=1, PP=1, EP=1), required before constructing
    any parallel linear layer or ``ZayaForCausalLM`` itself."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29636")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
        )

    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


# ---------------------------------------------------------------------------
# Tiny two-layer geometry, expressed once in each checkpoint schema
# ---------------------------------------------------------------------------

_L_NATIVE = 2  # native hybrid layers -> 2 * _L_NATIVE internal layers
_HIDDEN = 16
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8
_MOE_I = 16  # native moe_intermediate_size -> legacy ffn_hidden_size = 2 * I
_EXPERTS = 2
_ROUTER_HIDDEN = 8
_VOCAB = 64

# Native schema (transformers >= v5.13), field set mirroring the pinned
# Zyphra/ZAYA1-8B config.json shrunk to the tiny geometry above.
_NATIVE_TINY = {
    "model_type": "zaya",
    "vocab_size": _VOCAB,
    "hidden_size": _HIDDEN,
    "num_hidden_layers": _L_NATIVE,
    "num_attention_heads": _HEADS,
    "num_key_value_heads": _KV_HEADS,
    "head_dim": _HEAD_DIM,
    "hidden_act": "silu",
    "moe_intermediate_size": _MOE_I,
    "num_experts": _EXPERTS,
    "num_experts_per_tok": 1,
    "router_hidden_size": _ROUTER_HIDDEN,
    "cca_time0": 2,
    "cca_time1": 2,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 64,
    "layer_types": ["hybrid"] * _L_NATIVE,
    "rope_parameters": {
        "hybrid": {
            "rope_type": "default",
            "rope_theta": 5_000_000.0,
            "partial_rotary_factor": 0.5,
        },
        "hybrid_sliding": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
        },
    },
    "sliding_window": None,
    "attention_bias": False,
    "lm_head_bias": False,
    "tie_word_embeddings": True,
    "pad_token_id": 0,
    "bos_token_id": 2,
    "eos_token_id": 106,
}

# The same tiny model in the legacy schema SGLang already understands. Used
# to build the reference internal module tree (names and shapes).
_LEGACY_TINY = {
    "model_type": "zaya",
    "vocab_size": _VOCAB,
    "hidden_size": _HIDDEN,
    "ffn_hidden_size": 2 * _MOE_I,
    "num_hidden_layers": 2 * _L_NATIVE,
    "num_experts": _EXPERTS,
    "num_attention_heads": _HEADS,
    "num_query_groups": _KV_HEADS,
    "num_key_value_heads": _KV_HEADS,
    "head_dim": _HEAD_DIM,
    "cca_time0": 2,
    "cca_time1": 2,
    "max_position_embeddings": 64,
    "moe_router_topk": 1,
    "zaya_mlp_expansion": _ROUTER_HIDDEN,
    "rope_theta": 5_000_000.0,
    "attention_bias": False,
    "tie_word_embeddings": True,
}

# The first MoE layer has no previous router state, so neither checkpoint
# format ships an EDA ``router_states_scale`` for it (verified against the
# pinned ZAYA1-8B and ZAYA1-8B-legacy safetensors indexes).
_FIRST_MOE_ROUTER_SCALE = "model.layers.1.zaya_block.router.router_states_scale"


def _expected_native_to_internal_map(num_native_layers: int) -> dict:
    """The mapping contract under test: every non-expert native checkpoint
    key -> the internal parameter/buffer it must load into."""
    mapping = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.final_norm.weight",
        "model.input_hidden_states_scale": "model.layers.0.res_scale.hidden_states_scale",
        "model.input_hidden_states_bias": "model.layers.0.res_scale.hidden_states_bias",
    }
    attn = {
        "input_layernorm.weight": "input_norm.weight",
        "self_attn.qkv_proj.q_proj.weight": "self_attn.qkv.linear_q.weight",
        "self_attn.qkv_proj.k_proj.weight": "self_attn.qkv.linear_k.weight",
        "self_attn.qkv_proj.v_proj_current.weight": "self_attn.qkv.val_proj1.weight",
        "self_attn.qkv_proj.v_proj_delayed.weight": "self_attn.qkv.val_proj2.weight",
        "self_attn.qkv_proj.conv_qk_depthwise.weight": "self_attn.qkv.conv_qk.0.weight",
        "self_attn.qkv_proj.conv_qk_depthwise.bias": "self_attn.qkv.conv_qk.0.bias",
        "self_attn.qkv_proj.conv_qk_grouped.weight": "self_attn.qkv.conv_qk.1.weight",
        "self_attn.qkv_proj.conv_qk_grouped.bias": "self_attn.qkv.conv_qk.1.bias",
        "self_attn.qk_norm.temp": "self_attn.qkv.temp",
        "self_attn.o_proj.weight": "self_attn.o_proj.weight",
    }
    moe = {
        "post_attention_layernorm.weight": "input_norm.weight",
        "mlp.gate.down_proj.weight": "zaya_block.router.down_proj.weight",
        "mlp.gate.down_proj.bias": "zaya_block.router.down_proj.bias",
        "mlp.gate.router_mlp.norm.weight": "zaya_block.router.rmsnorm_eda.weight",
        "mlp.gate.router_mlp.fc1.weight": "zaya_block.router.router_mlp.0.weight",
        "mlp.gate.router_mlp.fc1.bias": "zaya_block.router.router_mlp.0.bias",
        "mlp.gate.router_mlp.fc2.weight": "zaya_block.router.router_mlp.2.weight",
        "mlp.gate.router_mlp.fc2.bias": "zaya_block.router.router_mlp.2.bias",
        "mlp.gate.router_mlp.out_proj.weight": "zaya_block.router.router_mlp.4.weight",
        "mlp.gate.balancing_biases": "zaya_block.router.balancing_biases",
    }
    res_fields = (
        "hidden_states_scale",
        "hidden_states_bias",
        "residual_scale",
        "residual_bias",
    )
    for k in range(num_native_layers):
        for suffix, target in attn.items():
            mapping[f"model.layers.{k}.{suffix}"] = f"model.layers.{2 * k}.{target}"
        for suffix, target in moe.items():
            mapping[f"model.layers.{k}.{suffix}"] = f"model.layers.{2 * k + 1}.{target}"
        if k != 0:
            mapping[f"model.layers.{k}.mlp.gate.router_states_scale"] = (
                f"model.layers.{2 * k + 1}.zaya_block.router.router_states_scale"
            )
        for field in res_fields:
            mapping[f"model.layers.{k}.post_attention_residual_scale.{field}"] = (
                f"model.layers.{2 * k + 1}.res_scale.{field}"
            )
            if k < num_native_layers - 1:
                mapping[f"model.layers.{k}.post_mlp_residual_scale.{field}"] = (
                    f"model.layers.{2 * k + 2}.res_scale.{field}"
                )
            else:
                mapping[f"model.layers.{k}.post_mlp_residual_scale.{field}"] = (
                    f"model.res_scale.{field}"
                )
    return mapping


def _expert_marker(layer: int, expert: int, part: str) -> float:
    part_offset = {"gate": 1.0, "up": 2.0, "down": 3.0}[part]
    return 1000.0 + 100.0 * layer + 10.0 * expert + part_offset


def _state_of(model: torch.nn.Module) -> dict:
    state = dict(model.named_parameters())
    state.update(dict(model.named_buffers()))
    return state


def _loadable_targets(ref_state: dict) -> set:
    """Every internal name a full checkpoint load must populate: all params
    and persistent buffers except rotary caches (computed, never loaded) and
    the first MoE layer's EDA scale (absent from both checkpoint formats)."""
    return {
        name
        for name in ref_state
        if "rotary_emb" not in name and name != _FIRST_MOE_ROUTER_SCALE
    }


def _make_native_checkpoint(ref_state: dict, keymap: dict):
    """Synthetic native checkpoint stream. Every non-expert tensor is filled
    with a distinct constant so each test can verify exactly where it landed;
    fused expert tensors carry per-(layer, expert, part) markers."""
    entries = []
    markers = {}
    for i, (native_key, internal_key) in enumerate(sorted(keymap.items())):
        marker = float(i + 1)
        entries.append((native_key, torch.full_like(ref_state[internal_key], marker)))
        markers[native_key] = marker
    for k in range(_L_NATIVE):
        gate_up = torch.empty(_EXPERTS, 2 * _MOE_I, _HIDDEN)
        down = torch.empty(_EXPERTS, _HIDDEN, _MOE_I)
        for e in range(_EXPERTS):
            gate_up[e, :_MOE_I] = _expert_marker(k, e, "gate")
            gate_up[e, _MOE_I:] = _expert_marker(k, e, "up")
            down[e] = _expert_marker(k, e, "down")
        entries.append((f"model.layers.{k}.mlp.experts.gate_up_proj", gate_up))
        entries.append((f"model.layers.{k}.mlp.experts.down_proj", down))
    return entries, markers


def _make_legacy_checkpoint(ref_state: dict):
    """Synthetic legacy checkpoint stream mirroring the pinned
    ZAYA1-8B-legacy key set for the tiny geometry."""
    entries = []
    for name in sorted(_loadable_targets(ref_state)):
        if ".zaya_block.experts." in name:
            continue  # replaced by per-expert linear_fc keys below
        entries.append((name, torch.zeros_like(ref_state[name])))
    for internal_layer in range(1, 2 * _L_NATIVE, 2):
        prefix = f"model.layers.{internal_layer}.zaya_block.experts"
        for e in range(_EXPERTS):
            entries.append(
                (
                    f"{prefix}.local_experts.{e}.linear_fc1.weight",
                    torch.zeros(2 * _MOE_I, _HIDDEN),
                )
            )
            entries.append(
                (
                    f"{prefix}.local_experts.{e}.linear_fc2.weight",
                    torch.zeros(_HIDDEN, _MOE_I),
                )
            )
    return entries


class _ZayaCpuModelTestBase(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()
        cls._saved_server_args = get_context()._server_args
        get_context().set_server_args(ServerArgs(model_path="dummy"))

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_saved_server_args", None) is None:
            reset_context()
        else:
            get_context().set_server_args(cls._saved_server_args)


class TestZayaNativeCheckpointLoading(_ZayaCpuModelTestBase):
    def _reference_state(self) -> dict:
        return _state_of(ZayaForCausalLM(config=ZayaConfig(**_LEGACY_TINY)))

    def _load_native(self):
        """Build the model from the native config and load the synthetic
        native checkpoint into it, returning everything tests inspect."""
        ref_state = self._reference_state()
        keymap = _expected_native_to_internal_map(_L_NATIVE)
        entries, markers = _make_native_checkpoint(ref_state, keymap)
        model = ZayaForCausalLM(config=ZayaConfig(**_NATIVE_TINY))
        loaded = model.load_weights(iter(entries))
        return model, ref_state, keymap, markers, loaded

    def test_native_checkpoint_loads_completely(self):
        """Every tensor of a native checkpoint must load, and together they
        must populate every loadable internal parameter and buffer."""
        model, ref_state, keymap, _, loaded = self._load_native()

        expected = set(keymap.values())
        for internal_layer in range(1, 2 * _L_NATIVE, 2):
            prefix = f"model.layers.{internal_layer}.zaya_block.experts"
            expected.add(f"{prefix}.w13_weight")
            expected.add(f"{prefix}.w2_weight")

        # Fixture self-consistency: the expected key map covers exactly the
        # loadable surface of the internal module tree.
        self.assertEqual(expected, _loadable_targets(ref_state))

        self.assertEqual(loaded, expected)

    def test_native_weights_land_on_mapped_parameters(self):
        """Marker sweep over every non-expert native key: the constant put
        into a native tensor must reappear in exactly the mapped internal
        parameter (this pins the 2:1 layer folding and the one-position
        residual-scale shift described in the module docstring)."""
        model, _, keymap, markers, _ = self._load_native()
        state = _state_of(model)
        for native_key, internal_key in keymap.items():
            param = state[internal_key]
            self.assertTrue(
                bool(torch.all(param == markers[native_key])),
                msg=(
                    f"{native_key} -> {internal_key}: expected constant "
                    f"{markers[native_key]}, got {param.flatten()[:4].tolist()}"
                ),
            )

    def test_native_fused_experts_split_gate_first(self):
        """Native ``gate_up_proj[e]`` rows ``[:I]`` are the gate (w1) and rows
        ``[I:]`` the up projection (w3), matching the legacy ``linear_fc1``
        half-split order; ``down_proj[e]`` is w2."""
        model, _, _, _, _ = self._load_native()
        state = _state_of(model)
        for k in range(_L_NATIVE):
            prefix = f"model.layers.{2 * k + 1}.zaya_block.experts"
            w13 = state[f"{prefix}.w13_weight"]
            w2 = state[f"{prefix}.w2_weight"]
            for e in range(_EXPERTS):
                self.assertTrue(
                    bool(torch.all(w13[e, :_MOE_I] == _expert_marker(k, e, "gate"))),
                    msg=f"layer {k} expert {e}: gate half mismatch",
                )
                self.assertTrue(
                    bool(torch.all(w13[e, _MOE_I:] == _expert_marker(k, e, "up"))),
                    msg=f"layer {k} expert {e}: up half mismatch",
                )
                self.assertTrue(
                    bool(torch.all(w2[e] == _expert_marker(k, e, "down"))),
                    msg=f"layer {k} expert {e}: down projection mismatch",
                )

    def test_legacy_checkpoint_still_loads_completely(self):
        """Regression guard: the legacy per-expert key set keeps loading the
        full parameter surface after the native-format support lands."""
        ref_state = self._reference_state()
        model = ZayaForCausalLM(config=ZayaConfig(**_LEGACY_TINY))
        loaded = model.load_weights(iter(_make_legacy_checkpoint(ref_state)))
        self.assertEqual(loaded, _loadable_targets(ref_state))

    def test_native_hybrid_sliding_fails_fast(self):
        """`hybrid_sliding` layers (ZAYA1-74B-preview) are not wired up yet;
        the config must still translate, but building the model must fail
        fast instead of silently serving without the sliding window."""
        cfg = ZayaConfig(
            **{
                **_NATIVE_TINY,
                "layer_types": ["hybrid", "hybrid_sliding"],
                "sliding_window": 8,
            }
        )
        self.assertEqual(cfg.checkpoint_format, "native")
        # Full-attention layers keep the "hybrid" rope theta.
        self.assertEqual(cfg.rope_theta, 5_000_000.0)
        with self.assertRaises(NotImplementedError):
            ZayaForCausalLM(config=cfg)


if __name__ == "__main__":
    unittest.main()
