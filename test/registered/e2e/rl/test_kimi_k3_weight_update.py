"""Kimi-K3 weight updates (RL weight sync) on a tiny real model.

CUDA graphs captured at startup hold the tensors that post-load derives from
the weights, so an update streamed over many load_weights calls must write
the new values into those same tensors and must not drop any tensor."""

import socket
import unittest
from collections import defaultdict

import torch

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.kimi_k3 import (
    KimiK3DeltaAttention,
    KimiK3LinearForCausalLM,
    KimiK3MoE,
)
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-large")

_NUM_LAYERS = 4

# Runtime module -> checkpoint shards, in shard order.
_STACKED = {
    "fused_qkvg_proj": ("q_proj", "k_proj", "v_proj", "g_proj"),
    "qkv_conv1d": ("q_conv1d", "k_conv1d", "v_conv1d"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "fused_qkv_a_proj_with_mqa": ("q_a_proj", "kv_a_proj_with_mqa"),
}


def _config() -> KimiLinearConfig:
    # K3's head shapes (12 KDA heads of 128 per rank, as at TP8, which the
    # fused KDA decode covers), few experts and layers: KDA, MLA, KDA, MLA;
    # the first MLP dense.
    return KimiLinearConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        rms_norm_eps=1e-5,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        mla_use_output_gate=True,
        num_experts=8,
        num_experts_per_token=2,
        moe_intermediate_size=128,
        num_shared_experts=1,
        first_k_dense_replace=1,
        routed_expert_hidden_size=128,
        latent_moe_use_norm=True,
        attn_res_block_size=2,
        linear_attn_config={
            "full_attn_layers": [2, 4],
            "kda_layers": [1, 3],
            "head_dim": 128,
            "num_heads": 12,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )


def _random_model(seed: int) -> KimiK3LinearForCausalLM:
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        model = KimiK3LinearForCausalLM(_config())
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for param in model.parameters():
        param.data.normal_(std=0.1, generator=gen)
    return model


def _checkpoint(model: KimiK3LinearForCausalLM) -> list[tuple[str, torch.Tensor]]:
    """The model's weights under checkpoint names, as a trainer sends them."""
    tensors = []
    for name, param in model.named_parameters():
        module_name, _, leaf = name.rpartition(".")
        parent, _, module = module_name.rpartition(".")
        if module in _STACKED:
            if module == "fused_qkv_a_proj_with_mqa":
                q_lora_rank = model.config.q_lora_rank
                sizes = [q_lora_rank, param.shape[0] - q_lora_rank]
            else:
                sizes = model.get_submodule(module_name).output_sizes
            for shard, chunk in zip(_STACKED[module], param.data.split(sizes)):
                tensors.append((f"{parent}.{shard}.{leaf}", chunk.clone()))
        elif leaf == "w13_weight":
            for expert, weight in enumerate(param.data):
                w1, w3 = weight.chunk(2)
                tensors.append((f"{module_name}.{expert}.w1.weight", w1.clone()))
                tensors.append((f"{module_name}.{expert}.w3.weight", w3.clone()))
        elif leaf == "w2_weight":
            for expert, weight in enumerate(param.data):
                tensors.append((f"{module_name}.{expert}.w2.weight", weight.clone()))
        else:
            tensors.append((name, param.data.clone()))
    return tensors


def _derived(model: KimiK3LinearForCausalLM) -> dict[str, torch.Tensor]:
    """Every tensor post-load derives from the weights, by a readable name."""
    derived = {}

    def add_cw(name, proj):
        for dtype, cw in proj._attn_res_cw_cache.items():
            derived[f"{name}.cw.{dtype}"] = cw

    for layer_id, layer in enumerate(model.model.layers):
        prefix = f"layers.{layer_id}"
        add_cw(f"{prefix}.self_attention_res", layer.self_attention_res_proj)
        add_cw(f"{prefix}.mlp_res", layer.mlp_res_proj)
        attn = layer.self_attn
        if isinstance(attn, KimiK3DeltaAttention):
            derived[f"{prefix}.bfa_w"] = attn._bfa_w
            derived[f"{prefix}.bfa_f_b_w"] = attn._bfa_f_b_w
            for index, arg in enumerate(attn.attn._k3_fused_decode_args[:-1]):
                derived[f"{prefix}.fused_decode_args.{index}"] = arg
        else:
            derived[f"{prefix}.w_kc"] = attn.w_kc
            derived[f"{prefix}.w_vc"] = attn.w_vc
        if isinstance(layer.mlp, KimiK3MoE):
            derived[f"{prefix}.front_w"] = layer.mlp._front_w
    add_cw("output", model.model.output_attn_res_proj)
    return derived


class TestKimiK3WeightUpdate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        cls.addClassCleanup(reset_context)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="nccl",
        )
        cls.addClassCleanup(destroy_distributed_environment)
        publish_build_topology(tp_size=1)
        initialize_model_parallel()
        cls.addClassCleanup(destroy_model_parallel)

    def test_bucketed_update_refreshes_derived_tensors_in_place(self):
        checkpoint = _checkpoint(_random_model(seed=0))

        # Reference: a first load of the whole checkpoint in one call, which
        # skips a layer the (truncated) config does not have.
        reference = _random_model(seed=1)
        extra_layer = (f"model.layers.{_NUM_LAYERS}.mlp.gate.weight", checkpoint[0][1])
        reference.load_weights(checkpoint + [extra_layer])

        # Engine: dummy weights, then an update in buckets like an RL sync.
        engine = _random_model(seed=2)
        engine.post_load_weights()
        self.assertIn(
            "model.layers.0.self_attn.f_b_proj.weight", dict(engine.named_parameters())
        )
        captured = _derived(engine)
        stale = {name: tensor.clone() for name, tensor in captured.items()}
        expected = _derived(reference)
        self.assertEqual(captured.keys(), expected.keys())
        self.assertEqual(
            [name for name in expected if torch.equal(stale[name], expected[name])],
            # The conv bias: K3 has none, so the fused decode gets zeros.
            ["layers.0.fused_decode_args.3", "layers.2.fused_decode_args.3"],
        )

        # One call per layer, except that the attention-residual norms of layer 1
        # and of the output arrive a call before their projections.
        calls = defaultdict(list)
        for name, tensor in checkpoint:
            layer_id = get_layer_id(name)
            early = "_res_norm." in name and layer_id in (1, None)
            calls[(-1 if layer_id is None else layer_id, not early)].append(
                (name, tensor)
            )
        for key in sorted(calls):
            engine.load_weights(calls[key])

        refreshed = _derived(engine)
        for name, tensor in expected.items():
            self.assertEqual(
                refreshed[name].data_ptr(), captured[name].data_ptr(), name
            )
            self.assertTrue(torch.equal(refreshed[name], tensor), name)
        reference_params = dict(reference.named_parameters())
        for name, param in engine.named_parameters():
            self.assertTrue(torch.equal(param, reference_params[name]), name)

        # A tensor the model has no parameter for is skipped (e.g. the
        # embedding or LM head that a pipeline stage does not build).
        unknown = ("model.layers.0.self_attn.no_such_proj.weight", extra_layer[1])
        self.assertEqual(engine.load_weights([unknown]), set())


if __name__ == "__main__":
    unittest.main()
