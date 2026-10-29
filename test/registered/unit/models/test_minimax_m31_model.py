"""M3.1 config resolution, DSpark draft wiring and the deferred MoE handoff."""

import copy
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.arg_groups.model_overrides.minimax_m3 import _minimax_m3_overrides
from sglang.srt.configs.minimax_vl import (
    MiniMaxM3VLConfig,
    configure_minimax_m3_dspark,
)
from sglang.srt.configs.model_config import (
    ModelConfig,
    is_dspark_draft,
)
from sglang.srt.layers.layer_boundary.fusions.moe_finalize import MoeDeferredFinalize
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.models import minimax_m3
from sglang.srt.models.dspark_minimax import DSparkMiniMaxDraftModel
from sglang.srt.models.minimax_m3 import (
    MiniMaxM3Model,
    get_spec_layer_idx_from_weight_name,
)
from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from sglang.srt.runtime_context import override_platform
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

ARCH = ["MiniMaxM3SparseForConditionalGeneration"]
NVFP4 = SimpleNamespace(
    architectures=ARCH, quantization_config={"quant_method": "minimax_m3_nvfp4"}
)


def draft_config():
    """A small DSpark draft in the layout of the released checkpoint's dspark/ folder."""
    config = MiniMaxM3VLConfig.from_dict(
        {
            "architectures": ["DSparkMiniMaxDraftModel"],
            "quantization_config": {"quant_method": "minimax_m3_nvfp4"},
            "text_config": {
                "model_type": "",
                "num_hidden_layers": 2,
                "dflash_config": {
                    "num_hidden_layers": 2,
                    "num_target_layers": 6,
                    "target_layer_ids": [1, 5],
                    "layer_types": ["sliding_attention"] * 2,
                },
                "dspark_use_dense_ffn": True,
                "dspark_ffn_hidden_size": 256,
                "dspark_block_size": 7,
                "dspark_markov_rank": 16,
                "dspark_noise_token_id": 1,
                "dspark_target_layer_ids": [1, 5],
            },
        }
    )
    # from_dict keeps text_config a dict; tests set draft fields as attributes.
    config.text_config = PretrainedConfig(**config.text_config)
    return config


def server_args(**kwargs):
    args = dict(
        quantization=None,
        _quantization_explicitly_unset=False,
        speculative_algorithm=None,
        moe_a2a_backend="none",
        moe_runner_backend="auto",
        page_size=None,
        kv_cache_dtype="auto",
        attention_backend=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        pp_size=1,
        speculative_draft_kv_cache_dtype=None,
        speculative_draft_attention_backend=None,
    )
    return SimpleNamespace(**(args | kwargs))


def packed_moe_hf(**quant_overrides):
    quant = {
        "quant_method": "mxfp8",
        "moe_quant_algo": "NVFP4",
        "moe_quant_format": "nvfp4-pack-quantized",
        "group_size": 16,
    }
    # None drops the field, matching checkpoints that do not carry it.
    quant = {k: v for k, v in (quant | quant_overrides).items() if v is not None}
    return SimpleNamespace(architectures=ARCH, quantization_config=quant)


class TestMiniMaxM31ConfigResolution(CustomTestCase):
    def test_nvfp4_defaults_on_blackwell_and_portable_fallback(self):
        base = dict(quantization="minimax_m3_nvfp4", page_size=128)
        blackwell = base | dict(
            moe_a2a_backend="megamoe",
            moe_runner_backend="deep_gemm",
            kv_cache_dtype="fp8_e4m3",
            attention_backend="trtllm_mha",
        )
        cases = [
            (True, None, blackwell),
            (
                True,
                "DSPARK",
                blackwell
                | dict(
                    speculative_draft_kv_cache_dtype="bf16",
                    speculative_draft_attention_backend="trtllm_mha",
                ),
            ),
            # MegaMoE and the packed NVFP4 KV cache are SM100-only.
            (
                False,
                None,
                base | dict(moe_runner_backend="marlin", attention_backend="fa3"),
            ),
        ]
        for is_blackwell, algorithm, expected in cases:
            with (
                self.subTest(blackwell=is_blackwell, algorithm=algorithm),
                override_platform(is_blackwell=is_blackwell),
                patch.dict("os.environ", SGLANG_RAGGED_VERIFY_MODE="static"),
            ):
                overrides = _minimax_m3_overrides(
                    server_args(speculative_algorithm=algorithm), NVFP4
                )
                self.assertEqual(overrides, expected)

    def test_packed_moe_mxfp8_resolves_like_nvfp4_and_plain_mxfp8_does_not(self):
        for blackwell in (True, False):
            with (
                self.subTest(blackwell=blackwell),
                override_platform(is_blackwell=blackwell),
            ):
                self.assertEqual(
                    _minimax_m3_overrides(server_args(), packed_moe_hf()),
                    _minimax_m3_overrides(server_args(), NVFP4),
                )
        for quant_overrides in (
            # The fully MXFP8 checkpoints of this family carry no MoE fields.
            dict(moe_quant_algo=None, moe_quant_format=None),
            dict(moe_quant_format="nvfp4-quantized"),
        ):
            with self.subTest(**quant_overrides), override_platform(is_blackwell=True):
                overrides = _minimax_m3_overrides(
                    server_args(), packed_moe_hf(**quant_overrides)
                )
                self.assertEqual(overrides["quantization"], "mxfp8")
                self.assertNotIn("moe_a2a_backend", overrides)
        with override_platform(is_blackwell=True):
            overrides = _minimax_m3_overrides(
                server_args(quantization="mxfp8"), packed_moe_hf()
            )
        self.assertNotIn("quantization", overrides)
        self.assertNotIn("moe_a2a_backend", overrides)

    def test_mxfp8_block_convert_flag_has_a_base_default(self):
        """Configs outside the Fp8 hierarchy also set use_mxfp8 and reach the Fp8
        methods, which read this flag without a defensive lookup."""
        from sglang.srt.layers.quantization.base_config import QuantizationConfig

        self.assertIsNone(QuantizationConfig.convert_mxfp8_to_block_fp8)

    def test_unsupported_combinations_are_rejected(self):
        def capture_ids(ids):
            config = draft_config()
            config.text_config.dflash_config["target_layer_ids"] = ids
            return lambda: configure_minimax_m3_dspark(config)

        def resolve(**kwargs):
            return lambda: _minimax_m3_overrides(server_args(**kwargs), NVFP4)

        cases = {
            "capture id past the target depth": ("static", capture_ids([1, 6])),
            "duplicate capture id": ("static", capture_ids([2, 2])),
            "compact verify": ("compact", resolve(speculative_algorithm="DSPARK")),
            "dspark pp": ("static", resolve(speculative_algorithm="DSPARK", pp_size=2)),
            "eagle": ("static", resolve(speculative_algorithm="EAGLE")),
            "text-only dspark": (
                "static",
                lambda: _minimax_m3_overrides(
                    server_args(speculative_algorithm="DSPARK"),
                    SimpleNamespace(
                        architectures=["MiniMaxM3SparseForCausalLM"],
                        quantization_config=NVFP4.quantization_config,
                    ),
                ),
            ),
        }
        for name, (verify_mode, call) in cases.items():
            with (
                self.subTest(name),
                override_platform(is_blackwell=True),
                patch.dict("os.environ", SGLANG_RAGGED_VERIFY_MODE=verify_mode),
                self.assertRaises(ValueError),
            ):
                call()


class TestMiniMaxM31DSparkDraft(CustomTestCase):
    def test_draft_target_ids_are_zero_based_and_the_input_is_kept(self):
        config = draft_config()
        before = copy.deepcopy(config.to_dict())
        obj = object.__new__(ModelConfig)
        obj.is_draft_model = True
        obj.hf_config = config
        obj.hf_text_config = config.text_config
        obj.speculative_algorithm = "DSPARK"
        obj.is_draft_quantization_explicit = False
        obj.quantization = None
        obj._config_draft_model()
        self.assertTrue(is_dspark_draft(obj.hf_config))
        self.assertEqual(obj.hf_config.num_hidden_layers, 2)
        parsed = parse_dflash_draft_config(draft_hf_config=obj.hf_config)
        self.assertEqual(parsed.resolve_target_layer_ids(target_num_layers=6), [1, 5])
        self.assertEqual(config.to_dict(), before)

    def test_verify_window_comes_from_the_draft_checkpoint(self):
        from sglang.srt.arg_groups.overrides import resolution_result
        from sglang.srt.arg_groups.speculative_hook import _handle_dspark
        from sglang.srt.server_args import ServerArgs

        config = draft_config()
        config.text_config.dspark_block_size = 5
        args = ServerArgs(
            model_path="dummy",
            speculative_algorithm="DSPARK",
            speculative_draft_model_path="dummy-draft",
            device="cuda",
        )
        args._model_config = SimpleNamespace(hf_config=config)
        with patch(
            "sglang.srt.utils.hf_transformers_utils.get_config", return_value=config
        ):
            _handle_dspark(args)
        self.assertEqual(
            resolution_result(args, "speculative_draft_model_path"), "dummy-draft"
        )
        self.assertEqual(resolution_result(args, "speculative_num_draft_tokens"), 6)

    def test_loader_requires_every_checkpoint_weight_except_runtime_scales(self):
        model = object.__new__(DSparkMiniMaxDraftModel)
        nn.Module.__init__(model)
        model.fc = nn.Linear(2, 2, bias=False)
        model.confidence_head = None
        model.k_scale = nn.Parameter(torch.tensor(-1.0))
        model.k_scale._skip_weight_check = True
        with self.assertRaisesRegex(ValueError, "Missing.*fc.weight"):
            model.load_weights([])
        with self.assertRaisesRegex(ValueError, "Unexpected DSpark"):
            model.load_weights([("dspark.unknown.weight", torch.ones(2))])
        model.attn = nn.Module()
        model.attn.qkv_proj = nn.Linear(2, 6, bias=False)
        model.attn.qkv_proj.weight.weight_loader = lambda param, weight, shard: None
        fc = [("language_model.model.dspark.fc.weight", torch.ones(2, 2))]
        shards = [(f"dspark.attn.{s}_proj.weight", torch.ones(2, 2)) for s in "qkv"]
        with self.assertRaisesRegex(ValueError, "Missing DSpark checkpoint shards"):
            model.load_weights(fc + shards[:2])
        self.assertEqual(
            model.load_weights(fc + shards), {"fc.weight", "attn.qkv_proj.weight"}
        )

    def test_target_loader_skips_bundled_draft_weights(self):
        self.assertEqual(
            get_spec_layer_idx_from_weight_name(
                SimpleNamespace(num_hidden_layers=60), "model.dspark.fc.weight"
            ),
            60,
        )


class CaptureLayer(nn.Module):
    """Adds ``1 + offset`` to the residual stream, writing it with nothing pending,
    so a capture at the layer's entry reads the full stream value."""

    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def forward(
        self, *, hidden_states, forward_batch, captured_last_layer_outputs, **kwargs
    ):
        if captured_last_layer_outputs is not None:
            captured_last_layer_outputs.capture(hidden_states)
        return residual_batch.set_written(
            hidden_states + 1 + self.offset, forward_batch
        )


class CaptureNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states + 100
        return hidden_states + residual + 100, residual


class TestMiniMaxM31Forward(CustomTestCase):
    def test_capture_includes_final_layer_before_output_norm(self):
        model = object.__new__(MiniMaxM3Model)
        nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        # The first layer's offset stands in for a residual already on the stream.
        model.layers = nn.ModuleList(
            [CaptureLayer(offset=10), CaptureLayer(), CaptureLayer()]
        )
        model.norm = CaptureNorm()
        model.finalize_norm = None
        model.start_layer, model.end_layer = 0, 3
        model.layers_to_capture = []
        wrapper = SimpleNamespace(pp_group=model.pp_group, model=model)
        MiniMaxM3SparseForConditionalGeneration.set_dspark_layers_to_capture(
            wrapper, [0, 2]
        )
        fb = SimpleNamespace(can_run_tbo=False)
        with patch(
            "sglang.srt.models.minimax_m3.check_cuda_graph_backend",
            return_value=True,
        ):
            output, captured = model(
                torch.tensor([0]), torch.tensor([0]), fb, input_embeds=torch.zeros(1, 2)
            )
        torch.testing.assert_close(captured[0], torch.full((1, 2), 11.0))
        torch.testing.assert_close(captured[1], torch.full((1, 2), 13.0))
        torch.testing.assert_close(output, torch.full((1, 2), 113.0))
        self.assertEqual(len(captured), 2)

    def test_deferred_moe_hands_unreduced_expert_output_to_the_boundary(self):
        x = torch.ones(2, 8, dtype=torch.bfloat16)
        shared = x * 2
        deferred = SimpleNamespace(
            gemm2_out=torch.ones(4, 8, dtype=torch.bfloat16),
            expert_weights=torch.ones(2, 2),
            expanded_idx_to_permuted_idx=torch.arange(4, dtype=torch.int32).view(2, 2),
            top_k=2,
        )
        # Not callable: a finalizing expert call would raise instead of passing.
        experts = SimpleNamespace(forward_deferred_finalize=lambda h, topk: deferred)
        moe = SimpleNamespace(
            alt_stream=None,
            shared_experts=object(),
            _forward_shared_experts=lambda h: shared,
            _compute_router_logits=lambda h: None,
            topk=lambda h, logits: None,
            experts=experts,
        )
        moe._forward_router_experts = partial(
            minimax_m3.MiniMaxM3MoE._forward_router_experts, moe
        )
        with patch.object(
            minimax_m3,
            "get_forward",
            return_value=SimpleNamespace(defer_moe_finalize=True),
        ):
            result = minimax_m3.MiniMaxM3MoE.forward_normal(moe, x)
        self.assertIsInstance(result, MoeDeferredFinalize)
        self.assertIs(result.gated_shared_output, shared)
        self.assertEqual(result.routed_output.data_ptr(), deferred.gemm2_out.data_ptr())


class TestMiniMaxM31Nvfp4Backend(CustomTestCase):
    def test_backend_uses_resolved_verify_window(self):
        from sglang.srt.layers.attention import minimax_sparse_nvfp4_backend as backend
        from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
        from sglang.srt.runtime_context import get_context

        pool = object.__new__(MiniMaxM3Nvfp4KVPool)
        pool.page_size, pool.head_num = 128, 1
        pool.sparse_layer_id_mapping = {0: 0}
        config = SimpleNamespace(
            num_attention_heads=64,
            num_key_value_heads=4,
            head_dim=128,
            sparse_attention_config={
                "sparse_index_dim": 128,
                "sparse_block_size": 128,
                "sparse_topk_blocks": 16,
                "sparse_num_index_heads": 4,
            },
        )
        runner = SimpleNamespace(
            token_to_kv_pool=pool,
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(1, 128)),
            model_config=SimpleNamespace(
                context_len=128, hf_config=config, hf_text_config=config
            ),
            server_args=SimpleNamespace(speculative_num_draft_tokens=None),
        )
        with (
            get_context().override_server_args(speculative_num_draft_tokens=8),
            patch.object(
                backend, "msa_v1_qmul4_decode_capability", return_value=(True, None)
            ),
        ):
            attention = backend.MiniMaxSparseNvfp4AttnBackend(runner)
        self.assertEqual(attention.speculative_num_draft_tokens, 8)

    def test_verify_replans_after_prefix_growth_without_mutating_input_lengths(self):
        from sglang.srt.layers.attention import minimax_sparse_nvfp4_backend as backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        class Plan:
            def __init__(self, **kwargs):
                self.query_length = kwargs["query_length"]
                self.fp8_attention = None
                self.page_table = self.final_seq_lens = None
                self.refreshes = 0

            def refresh(
                self,
                *,
                attention_plan,
                req_to_token,
                req_pool_indices,
                seq_lens,
                is_target_verify,
            ):
                self.refreshes += 1
                offset = self.query_length if is_target_verify else 0
                self.final_seq_lens = seq_lens.to(torch.int32) + offset

        obj = object.__new__(backend.MiniMaxSparseNvfp4AttnBackend)
        obj.speculative_num_draft_tokens = 8
        obj.page_size = obj.idx_head_dim = 128
        obj.req_to_token = torch.arange(1024).reshape(2, 512)
        obj._msa_v1_attention_workspace = None
        obj._msa_v1_qmul4_available = True
        obj._msa_v1_num_q_heads, obj._msa_v1_num_kv_heads = 32, 2
        obj._msa_v1_num_index_heads = 2
        obj._msa_v1_index_fp8_scratch = None
        obj._msa_v1_cuda_graph_plans = {}
        obj._msa_v1_cuda_graph_attention_plans = {}
        obj.forward_metadata = SimpleNamespace()
        fb = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            seq_lens=torch.tensor([126, 250]),
            req_pool_indices=torch.tensor([0, 1]),
        )
        with (
            patch.object(backend, "MSAV1DecodeIndexerPlan", Plan),
            patch.object(backend, "MSAV1DecodeAttentionPlan", Plan),
        ):
            first = obj._prepare_msa_v1_decode_plan(
                fb, use_cuda_graph=True, max_seq_len=512
            )
            self.assertEqual(first.query_length, 8)
            torch.testing.assert_close(
                first.final_seq_lens, torch.tensor([134, 258], dtype=torch.int32)
            )
            torch.testing.assert_close(fb.seq_lens, torch.tensor([126, 250]))
            fb.seq_lens += 3
            second = obj._prepare_msa_v1_decode_plan(
                fb, use_cuda_graph=True, max_seq_len=512
            )
            self.assertIs(first, second)
            torch.testing.assert_close(
                second.final_seq_lens, torch.tensor([137, 261], dtype=torch.int32)
            )
            fb.forward_mode = ForwardMode.DECODE
            decode = obj._prepare_msa_v1_decode_plan(
                fb, use_cuda_graph=True, max_seq_len=512
            )
            self.assertEqual(decode.query_length, 1)
            torch.testing.assert_close(
                decode.final_seq_lens, fb.seq_lens.to(torch.int32)
            )
        # One refresh per prepare, on the cached plan.
        self.assertEqual((first.refreshes, decode.refreshes), (2, 1))


if __name__ == "__main__":
    unittest.main()
