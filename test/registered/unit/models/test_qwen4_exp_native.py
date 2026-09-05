"""Unit coverage for the native Qwen4-Exp text entry class."""

import socket
import unittest

import torch
from transformers import AutoConfig

from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils.hf_transformers import common as _hf_common  # noqa: F401
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _single_rank_init_method() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _tiny_config():
    return Qwen4ExpConfig(
        text_config={
            "vocab_size": 13,
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_conv_kernel_dim": 2,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4,
            "num_experts": 3,
            "num_experts_per_tok": 1,
            "hc_count": 2,
            "hc_lowrank": 4,
            "layer_types": ["linear_attention"],
            "max_position_embeddings": 32,
            "eos_token_id": 12,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
            },
        },
        vision_config={
            "model_type": "qwen4_exp",
            "hidden_size": 8,
            "out_hidden_size": 8,
        },
        architectures=["Qwen4ExpForConditionalGeneration"],
    )


def _tiny_ple_config():
    return Qwen4ExpConfig(
        text_config={
            "vocab_size": 13,
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_conv_kernel_dim": 2,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4,
            "num_experts": 3,
            "num_experts_per_tok": 1,
            "hc_count": 2,
            "hc_lowrank": 4,
            "layer_types": ["linear_attention"],
            "max_position_embeddings": 32,
            "eos_token_id": 12,
            "ple_layer_ids": [1],
            "ple_embed_dim": 4,
            "heads_per_ngram": 1,
            "ngram_vocab_size_base": 17,
            "make_ngram_vocab_size_divisible_by": 2,
            "split_ngram_parts": 2,
        },
        vision_config={
            "model_type": "qwen4_exp",
            "hidden_size": 8,
            "out_hidden_size": 8,
        },
    )


class TestQwen4ExpNative(unittest.TestCase):
    def test_registry_resolves_qwen4_exp_native(self):
        from sglang.srt.models.registry import ModelRegistry

        model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
            "Qwen4ExpForConditionalGeneration"
        )
        self.assertEqual(resolved_arch, "Qwen4ExpForConditionalGeneration")
        self.assertEqual(model_cls.__module__, "sglang.srt.models.qwen4_exp")
        self.assertNotIn("Transformers", model_cls.__name__)

    def test_config_normalizes_official_layer_names(self):
        config = Qwen4ExpConfig(
            text_config={
                "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "hc_count": 2,
                "hc_lowrank": 4,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "eos_token_id": 7,
            },
            vision_config={"model_type": "qwen4_exp"},
        )
        self.assertEqual(config.model_type, "qwen4_exp")
        self.assertEqual(config.text_config.model_type, "qwen4_exp_text")
        self.assertEqual(config.vision_config.model_type, "qwen4_exp_vision")
        self.assertEqual(
            config.text_config.layer_types,
            ["linear_attention", "qwen_sparse_attention"],
        )
        auto_config = AutoConfig.for_model(
            "qwen4_exp",
            text_config={
                "num_hidden_layers": 1,
                "layer_types": ["linear_attention"],
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "hc_count": 2,
                "hc_lowrank": 4,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "eos_token_id": 7,
            },
            vision_config={"model_type": "qwen4_exp"},
        )
        self.assertIsInstance(auto_config, Qwen4ExpConfig)
        self.assertEqual(auto_config.text_config.model_type, "qwen4_exp_text")

    def test_config_mamba_cache_params_default_before_parallel_init(self):
        cache_shape = _tiny_config().text_config.mamba2_cache_params.shape

        self.assertEqual(cache_shape.conv, [(8, 1)])
        self.assertEqual(cache_shape.temporal, (2, 2, 2))

    def test_hybrid_gdn_registry_recognizes_qwen4_exp_text_config(self):
        from sglang.srt.configs.hybrid_arch import hybrid_gdn_config, mambaish_config

        config = _tiny_config()

        class StubModelConfig:
            hf_config = config
            linear_attn_registry_result = None

        self.assertIs(hybrid_gdn_config(StubModelConfig), config.text_config)
        self.assertIs(mambaish_config(StubModelConfig), config.text_config)

    def test_gdn_exposes_radix_attention_without_renaming_official_weights(self):
        from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
        from sglang.srt.models.qwen4_exp import Qwen4ExpTextGatedDeltaNet

        gdn = Qwen4ExpTextGatedDeltaNet(_tiny_config().text_config, 0)
        self.assertIsInstance(gdn.attn, RadixLinearAttention)

        parameter_names = dict(gdn.named_parameters())
        self.assertIn("in_proj_qkv.weight", parameter_names)
        self.assertIn("in_proj_z.weight", parameter_names)
        self.assertIn("in_proj_b.weight", parameter_names)
        self.assertIn("in_proj_a.weight", parameter_names)

    def test_gdn_tp2_shards_projection_conv_and_cache_shapes(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpTextGatedDeltaNet

        config = Qwen4ExpConfig(
            text_config={
                "vocab_size": 13,
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "linear_key_head_dim": 2,
                "linear_value_head_dim": 2,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_conv_kernel_dim": 2,
                "moe_intermediate_size": 4,
                "shared_expert_intermediate_size": 4,
                "num_experts": 3,
                "num_experts_per_tok": 1,
                "hc_count": 2,
                "hc_lowrank": 4,
                "layer_types": ["linear_attention"],
                "max_position_embeddings": 32,
                "eos_token_id": 12,
            },
            vision_config={"model_type": "qwen4_exp"},
        ).text_config

        with get_parallel().override(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            attn_dp_size=1,
        ):
            gdn = Qwen4ExpTextGatedDeltaNet(config, 0)
            cache_shape = config.mamba2_cache_params.shape

        self.assertEqual(gdn.local_num_k_heads, 1)
        self.assertEqual(gdn.local_num_v_heads, 2)
        self.assertEqual(tuple(gdn.in_proj_qkv.weight.shape), (8, 8))
        self.assertEqual(tuple(gdn.conv1d.weight.shape), (8, 1, 2))
        self.assertEqual(tuple(gdn.in_proj_z.weight.shape), (4, 8))
        self.assertEqual(tuple(gdn.in_proj_b.weight.shape), (2, 8))
        self.assertEqual(tuple(gdn.in_proj_a.weight.shape), (2, 8))
        self.assertEqual(tuple(gdn.A_log.shape), (2,))
        self.assertEqual(tuple(gdn.dt_bias.shape), (2,))
        self.assertEqual(gdn.attn.q_dim, 2)
        self.assertEqual(gdn.attn.k_dim, 2)
        self.assertEqual(gdn.attn.v_dim, 4)
        self.assertEqual(cache_shape.conv, [(8, 1)])
        self.assertEqual(cache_shape.temporal, (2, 2, 2))

    def test_gdn_flat_forward_dispatches_to_radix_backend(self):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.models.qwen4_exp import Qwen4ExpTextGatedDeltaNet

        config = _tiny_config().text_config
        gdn = Qwen4ExpTextGatedDeltaNet(config, 0)
        gdn.to(dtype=torch.float64)
        hidden_states = torch.randn(4, config.hidden_size, dtype=torch.float64)
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor([hidden_states.shape[0]], dtype=torch.int32),
            extend_seq_lens_cpu=[hidden_states.shape[0]],
        )

        class FakeRadixBackend:
            def __init__(self):
                self.calls = []

            def forward(self, layer, forward_batch, mixed_qkv, a, b):
                self.layer_conv_weight_dtype = layer.conv_weights.dtype
                self.layer_conv_weight_device = layer.conv_weights.device
                self.calls.append(
                    (layer, forward_batch, mixed_qkv.shape, a.shape, b.shape)
                )
                return mixed_qkv.new_zeros(
                    1, mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
                )

        backend = FakeRadixBackend()
        with forward_context(ForwardContext(attn_backend=backend)):
            output = gdn(hidden_states, forward_batch=metadata)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(len(backend.calls), 1)
        _, called_batch, mixed_qkv_shape, a_shape, b_shape = backend.calls[0]
        self.assertIs(called_batch, metadata)
        self.assertEqual(mixed_qkv_shape, (hidden_states.shape[0], 8))
        self.assertEqual(a_shape, (hidden_states.shape[0], 2))
        self.assertEqual(b_shape, (hidden_states.shape[0], 2))
        self.assertEqual(backend.layer_conv_weight_dtype, gdn.conv1d.weight.dtype)
        self.assertEqual(backend.layer_conv_weight_device, gdn.conv1d.weight.device)

    def test_moe_fused_path_preserves_official_weight_names(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpTextSparseMoeBlock

        moe = Qwen4ExpTextSparseMoeBlock(_tiny_config().text_config)
        parameter_names = dict(moe.named_parameters())

        self.assertIn("experts.gate_up_proj", parameter_names)
        self.assertIn("experts.down_proj", parameter_names)
        self.assertIn("shared_expert.gate_proj.weight", parameter_names)
        self.assertIn("shared_expert.up_proj.weight", parameter_names)
        self.assertIn("shared_expert.down_proj.weight", parameter_names)
        self.assertIn("shared_expert_gate.weight", parameter_names)
        self.assertNotIn("experts.w13_weight", parameter_names)
        self.assertFalse(moe.experts.moe_runner_config.gate_up_interleaved)
        self.assertEqual(moe.experts.moe_runner_config.top_k, 1)

    def test_moe_forward_matches_reference_loop_on_cpu(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpTextSparseMoeBlock

        torch.manual_seed(0)
        moe = Qwen4ExpTextSparseMoeBlock(_tiny_config().text_config)
        hidden_states = torch.randn(2, 3, moe.experts.hidden_dim)

        output = moe(hidden_states)
        moe.force_reference_moe = True
        reference = moe(hidden_states)

        torch.testing.assert_close(output, reference, rtol=1e-6, atol=1e-6)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for fused MoE")
    def test_fused_moe_matches_reference_loop_on_cuda(self):
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
            ensure_model_parallel_initialized,
            init_distributed_environment,
        )
        from sglang.srt.models.qwen4_exp import Qwen4ExpTextSparseMoeBlock

        override = get_context().override_server_args(
            enable_dp_lm_head=False, enable_fp32_lm_head=False
        )
        override.install()
        self.addCleanup(override.restore)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=_single_rank_init_method(),
            backend="nccl",
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="nccl",
        )
        self.addCleanup(destroy_distributed_environment)
        self.addCleanup(destroy_model_parallel)

        torch.manual_seed(0)
        with get_parallel().override(tp_size=1, attn_dp_size=1):
            config = _tiny_config().text_config
            config.hidden_size = 16
            config.moe_intermediate_size = 32
            config.shared_expert_intermediate_size = 32
            config.num_experts = 8
            config.num_experts_per_tok = 2
            moe = (
                Qwen4ExpTextSparseMoeBlock(config)
                .to(device="cuda", dtype=torch.bfloat16)
                .eval()
            )
            hidden_states = torch.randn(
                32, moe.experts.hidden_dim, device="cuda", dtype=torch.bfloat16
            )
            moe.force_reference_moe = True
            reference = moe(hidden_states)
            moe.force_reference_moe = False
            fused = moe(hidden_states)
            torch.cuda.synchronize()

        torch.testing.assert_close(fused, reference, rtol=5e-2, atol=5e-2)

    def test_ple_loader_preserves_official_shard_names(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

        override = get_context().override_server_args(
            enable_dp_lm_head=False, enable_fp32_lm_head=False
        )
        override.install()
        self.addCleanup(override.restore)

        with get_parallel().override(tp_size=1, attn_dp_size=1):
            target = Qwen4ExpForConditionalGeneration(_tiny_ple_config())
        embedding = target.model.layers[0].ple.ple_embedding.ngram_embedding
        shard_0 = torch.arange(
            embedding.shard_0.weight.numel(), dtype=torch.float32
        ).reshape_as(embedding.shard_0.weight)
        shard_1 = (
            torch.arange(embedding.shard_1.weight.numel(), dtype=torch.float32)
            .reshape_as(embedding.shard_1.weight)
            .add(1000)
        )

        loaded = target.load_weights(
            [
                (
                    "model.language_model.layers.0.ple.ple_embedding.layer_multipliers",
                    target.model.layers[0].ple.ple_embedding.layer_multipliers.clone(),
                ),
                (
                    "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight",
                    shard_0,
                ),
                (
                    "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight",
                    shard_1,
                ),
            ]
        )

        self.assertIn(
            "model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight", loaded
        )
        self.assertIn(
            "model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight", loaded
        )
        probe_ids = torch.tensor(
            [
                [
                    [0, embedding.shard_sizes[0] - 1],
                    [embedding.shard_sizes[0], embedding.num_embeddings - 1],
                ]
            ],
            dtype=torch.long,
        )
        actual = embedding(probe_ids)
        expected = torch.stack(
            [
                shard_0[0],
                shard_0[-1],
                shard_1[0],
                shard_1[-1],
            ]
        ).reshape_as(actual)
        torch.testing.assert_close(actual, expected)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_ple_shards_stay_on_cpu_when_constructed_under_cuda(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

        override = get_context().override_server_args(
            enable_dp_lm_head=False, enable_fp32_lm_head=False
        )
        override.install()
        self.addCleanup(override.restore)

        with get_parallel().override(tp_size=1, attn_dp_size=1), torch.device("cuda"):
            target = Qwen4ExpForConditionalGeneration(_tiny_ple_config())

        embedding = target.model.layers[0].ple.ple_embedding.ngram_embedding
        self.assertEqual(embedding.shard_0.weight.device.type, "cpu")
        self.assertEqual(embedding.shard_1.weight.device.type, "cpu")
        experts = target.model.layers[0].mlp.experts
        self.assertEqual(experts.gate_up_proj.device.type, "cpu")
        self.assertEqual(experts.down_proj.device.type, "cpu")
        hidden_states = torch.randn(
            2, experts.hidden_dim, device="cuda", dtype=target.lm_head.weight.dtype
        )
        output = target.model.layers[0].mlp(hidden_states)
        self.assertEqual(output.device.type, "cuda")
        self.assertEqual(output.shape, hidden_states.shape)

    def test_ple_flat_matches_sequence_reference_across_boundaries(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpTextPLELayer

        torch.manual_seed(0)
        config = _tiny_ple_config().text_config
        ple = Qwen4ExpTextPLELayer(config, layer_idx=0, ple_layer_index=0)
        ple.conv1d.weight.data.normal_(mean=0.0, std=0.01)
        seq_a = torch.tensor([1, 2, 12, 3, 4], dtype=torch.long)
        seq_b = torch.tensor([5, 12, 6], dtype=torch.long)
        input_ids = torch.cat([seq_a, seq_b])
        hidden_states = torch.randn(
            input_ids.numel(), config.hidden_size * config.hc_count
        )

        reference = torch.cat(
            [
                ple(
                    hidden_states[: seq_a.numel()].unsqueeze(0),
                    seq_a.unsqueeze(0),
                ).squeeze(0),
                ple(
                    hidden_states[seq_a.numel() :].unsqueeze(0),
                    seq_b.unsqueeze(0),
                ).squeeze(0),
            ],
            dim=0,
        )
        flat = ple.forward_flat(
            hidden_states, input_ids, [seq_a.numel(), seq_b.numel()]
        )

        torch.testing.assert_close(flat, reference, rtol=1e-6, atol=1e-6)

    def test_ple_flat_model_dispatches_to_radix_backend(self):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.models.qwen4_exp import Qwen4ExpTextModel

        torch.manual_seed(0)
        config = _tiny_ple_config().text_config
        model = Qwen4ExpTextModel(config).to(dtype=torch.float64).eval()
        input_ids = torch.tensor([1, 12, 3, 4, 5], dtype=torch.long)
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor([2, 3], dtype=torch.int32),
            extend_seq_lens_cpu=[2, 3],
        )

        class FakeRadixBackend:
            def __init__(self):
                self.calls = []

            def forward(self, layer, forward_batch, mixed_qkv, a, b):
                self.calls.append(
                    (layer, forward_batch, mixed_qkv.shape, a.shape, b.shape)
                )
                return mixed_qkv.new_zeros(
                    1, mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
                )

        backend = FakeRadixBackend()
        with forward_context(ForwardContext(attn_backend=backend)):
            output = model(input_ids, torch.arange(input_ids.numel()), metadata)

        self.assertEqual(output.shape, (input_ids.numel(), config.hidden_size))
        self.assertEqual(len(backend.calls), 1)
        _, called_batch, mixed_qkv_shape, a_shape, b_shape = backend.calls[0]
        self.assertIs(called_batch, metadata)
        self.assertEqual(mixed_qkv_shape, (input_ids.numel(), 8))
        self.assertEqual(a_shape, (input_ids.numel(), 2))
        self.assertEqual(b_shape, (input_ids.numel(), 2))

        model.force_sequence_ple = True
        backend.calls.clear()
        with forward_context(ForwardContext(attn_backend=backend)):
            fallback_output = model(
                input_ids, torch.arange(input_ids.numel()), metadata
            )
        self.assertEqual(fallback_output.shape, output.shape)
        self.assertEqual(len(backend.calls), 0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_grouped_cuda_ple_lookup_matches_cpu_reference(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpTextShardedNGramEmbedding

        embedding = Qwen4ExpTextShardedNGramEmbedding(
            num_embeddings=17, embedding_dim=3, num_shards=4
        )
        with torch.no_grad():
            row_offset = 0
            for shard_name in embedding.shard_names:
                shard = getattr(embedding, shard_name)
                values = torch.arange(
                    row_offset,
                    row_offset + shard.weight.numel(),
                    dtype=shard.weight.dtype,
                ).reshape_as(shard.weight)
                shard.weight.copy_(values)
                row_offset += shard.weight.numel()
        input_ids_cpu = torch.tensor(
            [[[0, 4, 5], [8, 16, 3]], [[12, 1, 9], [15, 6, 2]]],
            dtype=torch.long,
        )

        reference = embedding.forward_reference(input_ids_cpu).cuda()
        actual = embedding(input_ids_cpu.cuda())

        self.assertEqual(actual.device.type, "cuda")
        torch.testing.assert_close(actual, reference)

    def test_load_forward_logits_matches_zeroed_reference(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

        override = get_context().override_server_args(
            enable_dp_lm_head=False, enable_fp32_lm_head=False
        )
        override.install()
        self.addCleanup(override.restore)

        with get_parallel().override(tp_size=1, attn_dp_size=1):
            torch.manual_seed(0)
            source = Qwen4ExpForConditionalGeneration(_tiny_config())
            source.eval()
            with torch.no_grad():
                for param in source.parameters():
                    param.zero_()
                source.model.embed_tokens.weight.copy_(
                    torch.arange(13 * 8, dtype=torch.float32).reshape(13, 8) / 100
                )
                source.lm_head.weight.copy_(
                    torch.arange(13 * 8, dtype=torch.float32).reshape(13, 8) / 50
                )

            target = Qwen4ExpForConditionalGeneration(_tiny_config())
            weights = []
            for name, tensor in source.state_dict().items():
                if name.startswith("model."):
                    official_name = "model.language_model." + name[len("model.") :]
                else:
                    official_name = name
                weights.append((official_name, tensor.clone()))
            loaded = target.load_weights(weights)
            self.assertIn("model.embed_tokens.weight", loaded)
            self.assertIn("lm_head.weight", loaded)

            input_ids = torch.tensor([1, 4, 7, 2], dtype=torch.long)
            metadata = LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=torch.tensor([input_ids.numel()], dtype=torch.int32),
                extend_seq_lens_cpu=[input_ids.numel()],
            )
            output = target(
                input_ids=input_ids,
                positions=torch.arange(input_ids.numel()),
                forward_batch=metadata,
            )

            last_embed = source.model.embed_tokens(input_ids)[-1]
            expected_hidden = 0.5 * (
                last_embed
                * torch.rsqrt(last_embed.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            )
            expected_logits = (expected_hidden @ source.lm_head.weight.T).unsqueeze(0)
            torch.testing.assert_close(
                output.next_token_logits,
                expected_logits.float(),
                rtol=1e-5,
                atol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
