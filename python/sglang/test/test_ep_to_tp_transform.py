"""Test EP→TP weight transformation for MoE layers.

Verifies that loading weights via load_tp_by_experts (EP-style loading
followed by all_to_all redistribution) produces identical MoE weights to
normal TP loading.

Launch with torchrun:
    torchrun --nproc_per_node=2 -m pytest python/sglang/test/test_ep_to_tp_transform.py -v -s
"""

from __future__ import annotations

import contextlib
import unittest
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
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
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
) -> list[tuple[str, torch.Tensor]]:
    """Generate random checkpoint weights matching DeepSeek V3 tensor names."""
    weights = []
    H = config.hidden_size
    V = config.vocab_size
    I_dense = config.intermediate_size
    I_moe = config.moe_intermediate_size
    kv_lora = config.kv_lora_rank
    qk_nope = config.qk_nope_head_dim
    qk_rope = config.qk_rope_head_dim
    v_head = config.v_head_dim
    n_heads = config.num_attention_heads

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

            # Shared expert (DSv3 always has exactly 1)
            assert config.n_shared_experts == 1
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


_MOCK_BLACKWELL = patch(
    "sglang.srt.layers.quantization.modelopt_quant.is_blackwell_supported",
    return_value=True,
)


class _MoEOnlyNvFp4Config(ModelOptFp4Config):
    """ModelOptFp4Config that only quantizes FusedMoE layers.

    Non-MoE LinearBase layers (attention, dense MLP, embeddings) stay
    unquantized so the test can use simple BF16 checkpoint weights for
    them while exercising the real NVFP4 param layout on MoE experts.
    """

    def get_quant_method(self, layer, prefix):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        if isinstance(layer, FusedMoE):
            return super().get_quant_method(layer, prefix)
        if isinstance(layer, LinearBase):
            # LinearBase requires a non-None quant_method when quant_config
            # is provided, so return the unquantized fallback explicitly.
            return UnquantizedLinearMethod()
        # RadixAttention and other layer types: None means no quantization.
        return None


def _make_nvfp4_config() -> _MoEOnlyNvFp4Config:
    """Create a real NVFP4 quant config that only quantizes MoE layers."""
    return _MoEOnlyNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        packed_modules_mapping={},
    )


def _generate_nvfp4_checkpoint_weights(
    config: PretrainedConfig,
) -> list[tuple[str, torch.Tensor]]:
    """Generate checkpoint weights matching a real NVFP4 DeepSeek V3 checkpoint.

    Non-MoE layers (attention, dense MLP, embeddings) use BF16 — these are
    excluded from quantization by _make_nvfp4_config().

    MoE expert layers use the real NVFP4 layout:
      - weight:        uint8 packed FP4, [I, H//2] for gate/up, [H, I//2] for down
      - weight_scale:  FP8_E4M3 block scale, [I, H//gs] / [H, I//gs]
      - weight_scale_2: F32 per-tensor scale (scalar per expert)
      - input_scale:    F32 per-tensor scale (scalar per expert)
    """
    weights = []
    H = config.hidden_size
    V = config.vocab_size
    I_dense = config.intermediate_size
    I_moe = config.moe_intermediate_size
    kv_lora = config.kv_lora_rank
    qk_nope = config.qk_nope_head_dim
    qk_rope = config.qk_rope_head_dim
    v_head = config.v_head_dim
    n_heads = config.num_attention_heads
    gs = 16  # NVFP4 group_size, must match _make_nvfp4_config

    gen = torch.Generator()
    gen.manual_seed(42)

    def rand_bf16(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.bfloat16)

    def rand_uint8(*shape):
        # NVFP4 packs two 4-bit floats (E2M1) per byte. Any uint8 value is a
        # valid pair of FP4 values, so random bytes are fine for testing the
        # weight redistribution path without needing a real quantizer.
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)

    def rand_fp8_scale(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(torch.float8_e4m3fn)

    def rand_scalar():
        return torch.randn(1, generator=gen, dtype=torch.float32).squeeze(0).abs()

    # Embeddings (BF16, excluded from quant)
    weights.append(("model.embed_tokens.weight", rand_bf16(V, H)))

    for i in range(config.num_hidden_layers):
        pfx = f"model.layers.{i}"

        # Layer norms
        weights.append((f"{pfx}.input_layernorm.weight", rand_bf16(H)))
        weights.append((f"{pfx}.post_attention_layernorm.weight", rand_bf16(H)))

        # Attention (BF16, excluded from quant)
        weights.append((f"{pfx}.self_attn.q_proj.weight", rand_bf16(n_heads * (qk_nope + qk_rope), H)))
        weights.append((f"{pfx}.self_attn.kv_a_proj_with_mqa.weight", rand_bf16(kv_lora + qk_rope, H)))
        weights.append((f"{pfx}.self_attn.kv_a_layernorm.weight", rand_bf16(kv_lora)))
        weights.append((f"{pfx}.self_attn.kv_b_proj.weight", rand_bf16(n_heads * (qk_nope + v_head), kv_lora)))
        weights.append((f"{pfx}.self_attn.o_proj.weight", rand_bf16(H, n_heads * v_head)))

        is_moe = (
            i >= config.first_k_dense_replace and i % config.moe_layer_freq == 0
        )

        if is_moe:
            # Gate / router (BF16)
            weights.append((f"{pfx}.mlp.gate.weight", rand_bf16(config.n_routed_experts, H)))
            weights.append(
                (f"{pfx}.mlp.gate.e_score_correction_bias", rand_bf16(config.n_routed_experts))
            )

            # Routed experts — NVFP4 layout
            for e in range(config.n_routed_experts):
                # gate/up: weight [I, H//2] uint8, scale [I, H//gs] fp8
                for proj in ("gate_proj", "up_proj"):
                    weights.append((f"{pfx}.mlp.experts.{e}.{proj}.weight", rand_uint8(I_moe, H // 2)))
                    weights.append((f"{pfx}.mlp.experts.{e}.{proj}.weight_scale", rand_fp8_scale(I_moe, H // gs)))
                    weights.append((f"{pfx}.mlp.experts.{e}.{proj}.weight_scale_2", rand_scalar()))
                    weights.append((f"{pfx}.mlp.experts.{e}.{proj}.input_scale", rand_scalar()))
                # down: weight [H, I//2] uint8, scale [H, I//gs] fp8
                weights.append((f"{pfx}.mlp.experts.{e}.down_proj.weight", rand_uint8(H, I_moe // 2)))
                weights.append((f"{pfx}.mlp.experts.{e}.down_proj.weight_scale", rand_fp8_scale(H, I_moe // gs)))
                weights.append((f"{pfx}.mlp.experts.{e}.down_proj.weight_scale_2", rand_scalar()))
                weights.append((f"{pfx}.mlp.experts.{e}.down_proj.input_scale", rand_scalar()))

            # Shared expert (DSv3 always has exactly 1)
            # When not fused into FusedMoE, shared experts are LinearBase layers
            # (unquantized by _MoEOnlyNvFp4Config), so only BF16 weights.
            # When fused, they become extra rows in FusedMoE and get NVFP4 layout
            # — but the checkpoint format is the same (unfused names), and the
            # weight_loader handles the mapping.
            assert config.n_shared_experts == 1
            weights.append((f"{pfx}.mlp.shared_experts.gate_proj.weight", rand_bf16(I_moe, H)))
            weights.append((f"{pfx}.mlp.shared_experts.up_proj.weight", rand_bf16(I_moe, H)))
            weights.append((f"{pfx}.mlp.shared_experts.down_proj.weight", rand_bf16(H, I_moe)))
        else:
            # Dense MLP (BF16, excluded from quant)
            weights.append((f"{pfx}.mlp.gate_proj.weight", rand_bf16(I_dense, H)))
            weights.append((f"{pfx}.mlp.up_proj.weight", rand_bf16(I_dense, H)))
            weights.append((f"{pfx}.mlp.down_proj.weight", rand_bf16(H, I_dense)))

    # Final norm
    weights.append(("model.norm.weight", rand_bf16(H)))
    # lm_head (BF16, excluded from quant)
    weights.append(("lm_head.weight", rand_bf16(V, H)))

    return weights


def _snapshot_moe_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
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


@contextlib.contextmanager
def _load_tp_by_experts_config(enabled: bool):
    """Temporarily set load_tp_by_experts in the global server args."""
    from sglang.srt.server_args import get_global_server_args

    server_args = get_global_server_args()
    old_value = server_args.model_loader_extra_config
    server_args.model_loader_extra_config = (
        '{"load_tp_by_experts": true}' if enabled else "{}"
    )
    try:
        yield
    finally:
        server_args.model_loader_extra_config = old_value


def _create_model_and_load(
    config: PretrainedConfig,
    weights: list[tuple[str, torch.Tensor]],
    ep_load: bool,
    quant_config: ModelOptFp4Config | None = None,
) -> dict[str, torch.Tensor]:
    """Create a DeepseekV3 model, load weights, and return MoE param snapshot."""
    from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM

    with _load_tp_by_experts_config(ep_load):
        with torch.device("cuda"):
            model = DeepseekV3ForCausalLM(config, quant_config=quant_config)

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

        with _load_tp_by_experts_config(True):
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

    def test_ep_to_tp_shard_dim_routing(self):
        """_get_ep_to_tp_shard_dim must route each param name to the correct dim.

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

        # Block scales (NVFP4 / FP8) — MUST return a shard dim (not None)
        self.assertEqual(moe._get_ep_to_tp_shard_dim("w13_weight_scale"), 1)
        self.assertEqual(moe._get_ep_to_tp_shard_dim("w2_weight_scale"), 2)
        self.assertEqual(moe._get_ep_to_tp_shard_dim("w13_weight_scale_inv"), 1)
        self.assertEqual(moe._get_ep_to_tp_shard_dim("w2_weight_scale_inv"), 2)
        self.assertEqual(moe._get_ep_to_tp_shard_dim("w13_weight_scale1"), 1)
        self.assertEqual(moe._get_ep_to_tp_shard_dim("w2_weight_scale1"), 2)

        # Per-tensor scales (NVFP4 weight_scale_2 / input_scale) — must return None
        self.assertIsNone(moe._get_ep_to_tp_shard_dim("w13_weight_scale_2"))
        self.assertIsNone(moe._get_ep_to_tp_shard_dim("w2_weight_scale_2"))
        self.assertIsNone(moe._get_ep_to_tp_shard_dim("w13_input_scale"))
        self.assertIsNone(moe._get_ep_to_tp_shard_dim("w2_input_scale"))

        # Weights (no scale/bias) — must return a shard dim
        self.assertIsNotNone(moe._get_ep_to_tp_shard_dim("w13_weight"))
        self.assertIsNotNone(moe._get_ep_to_tp_shard_dim("w2_weight"))

        del model
        torch.cuda.empty_cache()

    def test_ep_to_tp_nvfp4_scales(self):
        """EP-load + transform must match normal TP load for NVFP4 quantized weights.

        Exercises both EP→TP code paths via the real ModelOptFp4Config:
          - Block scales (w13/w2_weight_scale): shard_dim != None → all_to_all
          - Per-tensor scales (weight_scale_2, input_scale): shard_dim == None → all_gather
        """
        config = _make_tiny_config(n_routed_experts=16)
        weights = _generate_nvfp4_checkpoint_weights(config)
        qc = _make_nvfp4_config()

        with _MOCK_BLACKWELL:
            tp_snapshot = _create_model_and_load(config, weights, ep_load=False, quant_config=qc)
            ep_snapshot = _create_model_and_load(config, weights, ep_load=True, quant_config=qc)

        self.assertEqual(set(tp_snapshot.keys()), set(ep_snapshot.keys()))

        # Sanity: both block and per-tensor scale params must be present
        block_keys = [k for k in tp_snapshot if "weight_scale" in k and "scale_2" not in k]
        per_tensor_keys = [k for k in tp_snapshot if "scale_2" in k or "input_scale" in k]
        self.assertTrue(len(block_keys) > 0, "No block scale params found")
        self.assertTrue(len(per_tensor_keys) > 0, "No per-tensor scale params found")

        for key in sorted(tp_snapshot.keys()):
            torch.testing.assert_close(
                ep_snapshot[key],
                tp_snapshot[key],
                msg=f"Mismatch on rank {self.rank} for {key}",
            )

    def test_ep_to_tp_nvfp4_scales_with_shared_experts(self):
        """EP-load + transform must match normal TP load for NVFP4 scales with fused shared experts."""
        from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM
        from sglang.srt.server_args import get_global_server_args

        config = _make_tiny_config(n_routed_experts=16)
        weights = _generate_nvfp4_checkpoint_weights(config)
        qc = _make_nvfp4_config()

        server_args = get_global_server_args()
        old_flag = server_args.disable_shared_experts_fusion

        def _force_fuse(self_model, architecture="DeepseekV3ForCausalLM"):
            self_model.num_fused_shared_experts = self_model.config.n_shared_experts

        try:
            server_args.disable_shared_experts_fusion = False

            with patch.object(
                DeepseekV3ForCausalLM,
                "determine_num_fused_shared_experts",
                _force_fuse,
            ), _MOCK_BLACKWELL:
                tp_snapshot = _create_model_and_load(
                    config, weights, ep_load=False, quant_config=qc,
                )
                ep_snapshot = _create_model_and_load(
                    config, weights, ep_load=True, quant_config=qc,
                )
        finally:
            server_args.disable_shared_experts_fusion = old_flag

        self.assertEqual(set(tp_snapshot.keys()), set(ep_snapshot.keys()))

        # Verify per-tensor scale params exist and match
        scale_keys = [k for k in tp_snapshot if "scale" in k]
        self.assertTrue(len(scale_keys) > 0, "No scale params found in snapshot")
        for key in sorted(tp_snapshot.keys()):
            torch.testing.assert_close(
                ep_snapshot[key],
                tp_snapshot[key],
                msg=f"Mismatch on rank {self.rank} for {key}",
            )

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
