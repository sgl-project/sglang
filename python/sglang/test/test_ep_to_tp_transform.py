"""Test EP→TP weight transformation for MoE layers.

Verifies that loading weights via SGLANG_EP_LOAD_FOR_TP=1 (EP-style loading
followed by all_to_all redistribution) produces identical MoE weights to
normal TP loading.

Launch with torchrun:
    torchrun --nproc_per_node=2 -m pytest python/sglang/test/test_ep_to_tp_transform.py -v -s
"""

import os
import unittest
from typing import Dict, Iterator, List, Tuple
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers import PretrainedConfig

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    destroy_model_parallel,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def _make_tiny_config(n_routed_experts: int = 16) -> PretrainedConfig:
    """Create a minimal DeepSeek V3 config for testing."""
    config = PretrainedConfig()
    config.architectures = ["DeepseekV3ForCausalLM"]
    config.model_type = "deepseek_v3"
    config.num_hidden_layers = 4
    config.first_k_dense_replace = 3
    config.moe_layer_freq = 1
    config.n_routed_experts = n_routed_experts
    config.n_shared_experts = 1
    config.num_experts_per_tok = 2
    config.moe_intermediate_size = 256
    config.hidden_size = 512
    config.intermediate_size = 1024
    config.num_attention_heads = 8
    config.num_key_value_heads = 8
    config.q_lora_rank = None
    config.kv_lora_rank = 64
    config.qk_nope_head_dim = 32
    config.qk_rope_head_dim = 16
    config.v_head_dim = 32
    config.vocab_size = 1000
    config.rms_norm_eps = 1e-6
    config.rope_theta = 10000
    config.max_position_embeddings = 4096
    config.hidden_act = "silu"
    config.attention_bias = False
    config.attention_dropout = 0.0
    config.norm_topk_prob = True
    config.scoring_func = "sigmoid"
    config.topk_method = "noaux_tc"
    config.topk_group = 4
    config.n_group = 8
    config.routed_scaling_factor = 2.5
    config.tie_word_embeddings = False
    config.rope_scaling = None
    config.pad_token_id = None
    return config


def _generate_checkpoint_weights(
    config: PretrainedConfig,
) -> List[Tuple[str, torch.Tensor]]:
    """Generate random checkpoint weights matching DeepSeek V3 tensor names."""
    weights = []
    H = config.hidden_size  # 512
    V = config.vocab_size
    I_dense = config.intermediate_size  # 1024
    I_moe = config.moe_intermediate_size  # 256
    kv_lora = config.kv_lora_rank  # 64
    qk_nope = config.qk_nope_head_dim  # 32
    qk_rope = config.qk_rope_head_dim  # 16
    v_head = config.v_head_dim  # 32
    n_heads = config.num_attention_heads  # 8

    # Use a fixed seed so both loading passes get identical weights
    gen = torch.Generator()
    gen.manual_seed(42)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.bfloat16)

    # Embeddings
    weights.append(("model.embed_tokens.weight", rand(V, H)))

    for i in range(config.num_hidden_layers):
        pfx = f"model.layers.{i}"

        # Layer norms
        weights.append((f"{pfx}.input_layernorm.weight", rand(H)))
        weights.append((f"{pfx}.post_attention_layernorm.weight", rand(H)))

        # Attention (no q_lora_rank → uses q_proj directly)
        weights.append((f"{pfx}.self_attn.q_proj.weight", rand(n_heads * (qk_nope + qk_rope), H)))
        weights.append((f"{pfx}.self_attn.kv_a_proj_with_mqa.weight", rand(kv_lora + qk_rope, H)))
        weights.append((f"{pfx}.self_attn.kv_a_layernorm.weight", rand(kv_lora)))
        weights.append((f"{pfx}.self_attn.kv_b_proj.weight", rand(n_heads * (qk_nope + v_head), kv_lora)))
        weights.append((f"{pfx}.self_attn.o_proj.weight", rand(H, n_heads * v_head)))

        is_moe = (
            i >= config.first_k_dense_replace and i % config.moe_layer_freq == 0
        )

        if is_moe:
            # Gate / router
            weights.append((f"{pfx}.mlp.gate.weight", rand(config.n_routed_experts, H)))
            weights.append(
                (f"{pfx}.mlp.gate.e_score_correction_bias", rand(config.n_routed_experts))
            )

            # Routed experts
            for e in range(config.n_routed_experts):
                weights.append((f"{pfx}.mlp.experts.{e}.gate_proj.weight", rand(I_moe, H)))
                weights.append((f"{pfx}.mlp.experts.{e}.up_proj.weight", rand(I_moe, H)))
                weights.append((f"{pfx}.mlp.experts.{e}.down_proj.weight", rand(H, I_moe)))

            # Shared experts
            for s in range(config.n_shared_experts):
                weights.append(
                    (f"{pfx}.mlp.shared_experts.gate_proj.weight", rand(I_moe, H))
                )
                weights.append(
                    (f"{pfx}.mlp.shared_experts.up_proj.weight", rand(I_moe, H))
                )
                weights.append(
                    (f"{pfx}.mlp.shared_experts.down_proj.weight", rand(H, I_moe))
                )
        else:
            # Dense MLP
            weights.append((f"{pfx}.mlp.gate_proj.weight", rand(I_dense, H)))
            weights.append((f"{pfx}.mlp.up_proj.weight", rand(I_dense, H)))
            weights.append((f"{pfx}.mlp.down_proj.weight", rand(H, I_dense)))

    # Final norm
    weights.append(("model.norm.weight", rand(H)))
    # lm_head
    weights.append(("lm_head.weight", rand(V, H)))

    return weights


def _snapshot_moe_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Extract cloned MoE parameter tensors from the model."""
    snapshot = {}
    for layer_id, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, "experts") or not isinstance(
            layer.mlp.experts, FusedMoE
        ):
            continue
        for name, param in layer.mlp.experts.named_parameters():
            key = f"layer{layer_id}.{name}"
            snapshot[key] = param.data.clone()
    return snapshot


def _create_model_and_load(
    config: PretrainedConfig, weights: List[Tuple[str, torch.Tensor]], ep_load: bool
) -> Dict[str, torch.Tensor]:
    """Create a DeepseekV3 model, load weights, and return MoE param snapshot."""
    from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM

    env_patch = {"SGLANG_EP_LOAD_FOR_TP": "1" if ep_load else "0"}
    with patch.dict(os.environ, env_patch):
        with torch.device("cuda"):
            model = DeepseekV3ForCausalLM(config, quant_config=None)

        # load_weights expects an iterator; make a fresh copy each time
        def weight_iter():
            for name, tensor in weights:
                yield name, tensor.clone()

        model.load_weights(weight_iter())

    snapshot = _snapshot_moe_params(model)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return snapshot


class TestEpToTpTransform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        server_args = ServerArgs(model_path="dummy")
        server_args.disable_shared_experts_fusion = True
        set_global_server_args_for_scheduler(server_args)
        initialize_moe_config(server_args)

        init_distributed_environment(
            world_size=-1,
            rank=-1,
            local_rank=-1,
            backend="nccl",
        )
        cls.world_size = dist.get_world_size()
        cls.rank = dist.get_rank()
        device = torch.device(f"cuda:{cls.rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        initialize_model_parallel(tensor_model_parallel_size=cls.world_size)

    @classmethod
    def tearDownClass(cls):
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_ep_to_tp_matches_normal_tp(self):
        """EP-load + transform should produce identical weights to normal TP load."""
        config = _make_tiny_config(n_routed_experts=16)
        weights = _generate_checkpoint_weights(config)

        # Load normally (TP)
        tp_snapshot = _create_model_and_load(config, weights, ep_load=False)

        # Load as EP, then transform to TP
        ep_snapshot = _create_model_and_load(config, weights, ep_load=True)

        # Compare
        self.assertEqual(
            set(tp_snapshot.keys()),
            set(ep_snapshot.keys()),
            "Mismatch in parameter names",
        )
        for key in sorted(tp_snapshot.keys()):
            torch.testing.assert_close(
                ep_snapshot[key],
                tp_snapshot[key],
                msg=f"Mismatch on rank {self.rank} for {key}",
            )

    def test_ep_to_tp_field_restoration(self):
        """After transform, FusedMoE fields should be restored to normal TP values."""
        from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM

        config = _make_tiny_config(n_routed_experts=16)
        weights = _generate_checkpoint_weights(config)

        with patch.dict(os.environ, {"SGLANG_EP_LOAD_FOR_TP": "1"}):
            with torch.device("cuda"):
                model = DeepseekV3ForCausalLM(config, quant_config=None)

            def weight_iter():
                for name, tensor in weights:
                    yield name, tensor.clone()

            model.load_weights(weight_iter())

        # Check fields on MoE layers
        for layer_id, layer in enumerate(model.model.layers):
            if not hasattr(layer.mlp, "experts") or not isinstance(
                layer.mlp.experts, FusedMoE
            ):
                continue
            moe = layer.mlp.experts
            self.assertEqual(moe.moe_tp_size, self.world_size)
            self.assertEqual(moe.moe_tp_rank, self.rank)
            self.assertEqual(moe.moe_ep_size, 1)
            self.assertEqual(moe.moe_ep_rank, 0)
            self.assertEqual(
                moe.intermediate_size_per_partition,
                config.moe_intermediate_size // self.world_size,
            )
            self.assertEqual(
                moe.num_local_experts,
                config.n_routed_experts,  # no shared fusion → all routed experts
            )
            # MoeRunnerConfig should match
            self.assertEqual(
                moe.moe_runner_config.num_local_experts, moe.num_local_experts
            )
            self.assertEqual(
                moe.moe_runner_config.intermediate_size_per_partition,
                moe.intermediate_size_per_partition,
            )

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
