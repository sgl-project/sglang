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
    # Transformers >=5.0 maps rope_scaling → rope_parameters via attribute_map.
    # Provide rope_parameters directly so both old and new transformers work.
    config.rope_parameters = {"rope_type": "default", "rope_theta": 10000}
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

        # Initialize dp attention globals (not using DP attention, just set defaults)
        # Hack to avoid using top-level APIs in unit test.
        import sglang.srt.layers.dp_attention as dp_attn

        dp_attn._ATTN_DP_SIZE = 1
        dp_attn._ATTN_DP_RANK = 0
        dp_attn._LOCAL_ATTN_DP_SIZE = 1
        dp_attn._LOCAL_ATTN_DP_RANK = 0
        dp_attn._ENABLE_DP_ATTENTION_FLAG = False

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

    def test_ep_to_tp_shard_dim_fp4_params(self):
        """_get_ep_to_tp_shard_dim must return correct dims for FP4 block scale params.

        Regression test: w13_weight_scale and w2_weight_scale (without _inv/_1
        suffix) were previously falling through to the per-tensor catch-all
        (returning None instead of a shard dim), causing the EP→TP transform to
        all_gather instead of shard. This left block scales at full size, failing
        the modelopt post-processing assertion:
            AssertionError: Expected w2_weight_scale.dim(2) == 16, got 128
        """
        config = _make_tiny_config(n_routed_experts=16)

        with torch.device("cuda"):
            from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM

            model = DeepseekV3ForCausalLM(config, quant_config=None)

        # Find any MoE layer to test _get_ep_to_tp_shard_dim
        moe = None
        for layer in model.model.layers:
            if hasattr(layer.mlp, "experts") and isinstance(
                layer.mlp.experts, FusedMoE
            ):
                moe = layer.mlp.experts
                break
        self.assertIsNotNone(moe, "No FusedMoE layer found in model")

        # FP4 block scales — these MUST return a shard dim (not None)
        self.assertEqual(moe._get_ep_to_tp_shard_dim("w13_weight_scale"), 1)
        self.assertEqual(moe._get_ep_to_tp_shard_dim("w2_weight_scale"), 2)

        # FP8 block scale variants — also must return a shard dim
        self.assertEqual(moe._get_ep_to_tp_shard_dim("w13_weight_scale_inv"), 1)
        self.assertEqual(moe._get_ep_to_tp_shard_dim("w2_weight_scale_inv"), 2)
        self.assertEqual(moe._get_ep_to_tp_shard_dim("w13_weight_scale1"), 1)
        self.assertEqual(moe._get_ep_to_tp_shard_dim("w2_weight_scale1"), 2)

        # FP4 per-tensor scales — must return None (no intermediate dim to shard)
        self.assertIsNone(moe._get_ep_to_tp_shard_dim("w13_weight_scale_2"))
        self.assertIsNone(moe._get_ep_to_tp_shard_dim("w2_weight_scale_2"))
        self.assertIsNone(moe._get_ep_to_tp_shard_dim("w13_input_scale"))
        self.assertIsNone(moe._get_ep_to_tp_shard_dim("w2_input_scale"))

        # Weights (no scale/bias) — must return a shard dim
        self.assertIsNotNone(moe._get_ep_to_tp_shard_dim("w13_weight"))
        self.assertIsNotNone(moe._get_ep_to_tp_shard_dim("w2_weight"))

        del model
        torch.cuda.empty_cache()

    def test_ep_to_tp_block_scale_shapes(self):
        """EP→TP transform must correctly shard block-scale-shaped params.

        Simulates FP4 block scale params by manually attaching them to a
        FusedMoE layer with EP_LOAD state, then verifies the transform
        produces the expected TP-sharded shapes.
        """
        config = _make_tiny_config(n_routed_experts=16)
        weights = _generate_checkpoint_weights(config)

        tp_size = self.world_size
        tp_rank = self.rank
        E = config.n_routed_experts  # 16 total routed experts
        I = config.moe_intermediate_size  # 256
        H = config.hidden_size  # 512
        block_size = 16  # NVFP4 group_size

        # Load model with EP_LOAD to set up internal state
        with patch.dict(os.environ, {"SGLANG_EP_LOAD_FOR_TP": "1"}):
            with torch.device("cuda"):
                from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM

                model = DeepseekV3ForCausalLM(config, quant_config=None)

            def weight_iter():
                for name, tensor in weights:
                    yield name, tensor.clone()

            model.load_weights(weight_iter())

        # After transform, MoE layers should have normal TP shapes.
        # Verify that if block scale params existed, they would have been
        # sharded correctly by checking the restored field values.
        for layer in model.model.layers:
            if not hasattr(layer.mlp, "experts") or not isinstance(
                layer.mlp.experts, FusedMoE
            ):
                continue
            moe = layer.mlp.experts

            # EP→TP should have restored these to normal TP values
            self.assertEqual(moe.moe_tp_size, tp_size)
            self.assertEqual(moe.moe_ep_size, 1)

            # The intermediate_size_per_partition should be I / tp_size
            self.assertEqual(
                moe.intermediate_size_per_partition,
                I // tp_size,
            )

        # Now verify the actual transform logic on block-scale-shaped tensors
        # by constructing a fresh model with EP_LOAD and manually adding
        # block scale params before calling ep_to_tp_transform.
        with patch.dict(os.environ, {"SGLANG_EP_LOAD_FOR_TP": "1"}):
            with torch.device("cuda"):
                model2 = DeepseekV3ForCausalLM(config, quant_config=None)

        # Find a MoE layer (still in EP_LOAD state before load_weights)
        moe = None
        for layer in model2.model.layers:
            if hasattr(layer.mlp, "experts") and isinstance(
                layer.mlp.experts, FusedMoE
            ):
                moe = layer.mlp.experts
                break
        self.assertIsNotNone(moe)
        self.assertTrue(moe._ep_load_for_tp)

        num_shared = moe.num_fused_shared_experts
        E_local = moe.num_local_experts - num_shared  # experts on this rank
        E_total = moe.num_experts - num_shared
        I_blocks = I // block_size  # ceil(I / block_size) for our exact divisor
        H_blocks = H // block_size

        # Register mock FP4 block scale params on the MoE layer
        # w13_weight_scale: [E_local + shared, 2*ceil(I/bn), ceil(H/bk)]
        w13_scale = torch.nn.Parameter(
            torch.randn(
                E_local + num_shared, 2 * I_blocks, H_blocks,
                device="cuda",
            ).to(torch.float8_e4m3fn)
        )
        moe.register_parameter("w13_weight_scale", w13_scale)

        # w2_weight_scale: [E_local + shared, ceil(H/bn), ceil(I/bk)]
        w2_scale = torch.nn.Parameter(
            torch.randn(
                E_local + num_shared, H_blocks, I_blocks,
                device="cuda",
            ).to(torch.float8_e4m3fn)
        )
        moe.register_parameter("w2_weight_scale", w2_scale)

        # Allocate _ep_to_tp_buf for each param (mimicking _ep_to_tp_transform_all_layers)
        for name, param in moe.named_parameters():
            shard_dim = moe._get_ep_to_tp_shard_dim(name)
            if not hasattr(param, "_ep_to_tp_buf") and name in (
                "w13_weight_scale",
                "w2_weight_scale",
            ):
                target_shape = list(param.data.shape)
                target_shape[0] = E_total + num_shared
                target_shape[shard_dim] = target_shape[shard_dim] // tp_size
                param._ep_to_tp_buf = torch.empty(
                    target_shape, dtype=param.data.dtype, device=param.data.device
                )

        # Also need _ep_to_tp_buf on existing weight params for transform to work
        for name, param in moe.named_parameters():
            if hasattr(param, "_ep_to_tp_buf"):
                continue
            shard_dim = moe._get_ep_to_tp_shard_dim(name)
            target_shape = list(param.data.shape)
            target_shape[0] = E_total + num_shared
            if shard_dim is not None:
                target_shape[shard_dim] = target_shape[shard_dim] // tp_size
            param._ep_to_tp_buf = torch.empty(
                target_shape, dtype=param.data.dtype, device=param.data.device
            )

        moe.ep_to_tp_transform()

        # Verify block scale shapes after transform
        w13_result = dict(moe.named_parameters())["w13_weight_scale"]
        w2_result = dict(moe.named_parameters())["w2_weight_scale"]

        # w13_weight_scale: [E_total + shared, 2*ceil(I/bn)/tp, ceil(H/bk)]
        self.assertEqual(w13_result.shape[0], E_total + num_shared)
        self.assertEqual(w13_result.shape[1], 2 * I_blocks // tp_size)
        self.assertEqual(w13_result.shape[2], H_blocks)

        # w2_weight_scale: [E_total + shared, ceil(H/bn), ceil(I/bk)/tp]
        self.assertEqual(w2_result.shape[0], E_total + num_shared)
        self.assertEqual(w2_result.shape[1], H_blocks)
        self.assertEqual(w2_result.shape[2], I_blocks // tp_size)

        del model, model2
        torch.cuda.empty_cache()

    def test_ep_to_tp_fused_shared_experts(self):
        """EP→TP transform must correctly handle fused shared experts.

        When shared experts are fused into FusedMoE, the weight tensors have
        extra rows for shared experts: [num_routed_local + num_shared, ...].
        The transform must shard the shared expert portion locally (no collective)
        while redistributing the routed portion via all_to_all.

        This exercises the `if num_shared > 0` branches in ep_to_tp_transform:
        - is_w13 + shared: split fused [gate|up], shard each half, re-fuse
        - w2 + shared: simple shard slice
        """
        from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM
        from sglang.srt.server_args import get_global_server_args

        config = _make_tiny_config(n_routed_experts=16)
        weights = _generate_checkpoint_weights(config)

        server_args = get_global_server_args()
        old_flag = server_args.disable_shared_experts_fusion

        # Force-enable shared expert fusion, bypassing the n_routed_experts==256 check
        def _force_fuse(self_model, architecture="DeepseekV3ForCausalLM"):
            self_model.num_fused_shared_experts = self_model.config.n_shared_experts

        try:
            server_args.disable_shared_experts_fusion = False

            with patch.object(
                DeepseekV3ForCausalLM,
                "determine_num_fused_shared_experts",
                _force_fuse,
            ):
                # Load normally (TP)
                tp_snapshot = _create_model_and_load(config, weights, ep_load=False)

                # Load as EP, then transform to TP
                ep_snapshot = _create_model_and_load(config, weights, ep_load=True)

                # Verify shared experts are actually fused by checking weight dim 0
                # With fusion: num_local_experts = n_routed + n_shared = 16 + 1 = 17
                n_shared = config.n_shared_experts  # 1
                found_fused = False
                for key in tp_snapshot:
                    if "w13_weight" in key and "scale" not in key and "bias" not in key:
                        self.assertEqual(
                            tp_snapshot[key].shape[0],
                            config.n_routed_experts + n_shared,
                            f"Shared experts not fused for {key}: "
                            f"expected dim 0 = {config.n_routed_experts + n_shared}, "
                            f"got {tp_snapshot[key].shape[0]}",
                        )
                        found_fused = True
                        break
                self.assertTrue(found_fused, "No w13_weight found in snapshot")

                # Compare EP-loaded vs TP-loaded weights
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
        finally:
            server_args.disable_shared_experts_fusion = old_flag


if __name__ == "__main__":
    unittest.main()
