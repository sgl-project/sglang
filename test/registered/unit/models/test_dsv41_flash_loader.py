"""CPU tests for official DeepSeek-V4.1-Flash checkpoint names and v1 config.

No 476 GiB load. Index scan runs only when model.safetensors.index.json is
already on disk (download in flight is fine).
"""

from __future__ import annotations

import json
import os

from sglang.srt.configs.deepseek_v41 import (
    DeepseekV41Config,
    normalize_deepseek_v41_config,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.mem_cache.dsv41_v100_cpu_mock import (
    OFFICIAL_CONFIG,
    _REMAP_CASES,
    check_index_if_present,
    check_weight_remap,
    run_cpu_mock,
)
import torch
from torch import nn

from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    _dsv41_image_position_mask,
    _skip_dsv41_language_only_weight,
    resolve_dsv41_weight_name,
)
from sglang.srt.models.deepseek_v41_vit import CpuAligner, CpuViT
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestDsv41OfficialCheckpointNames(CustomTestCase):
    def test_inference_format_remap_matches_sglang_params(self):
        for src, want in _REMAP_CASES:
            self.assertEqual(
                DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(src),
                want,
                src,
            )

    def test_gate_bias_vl_is_not_mangled(self):
        name = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
            "layers.6.ffn.gate.bias_vl"
        )
        self.assertEqual(name, "model.layers.6.mlp.gate.bias_vl")
        self.assertTrue(_skip_dsv41_language_only_weight(name))

    def test_engram_embed_keeps_scale_wkv_becomes_weight_scale_inv(self):
        embed = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
            "layers.14.engram.embed.scale"
        )
        wkv = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
            "layers.14.engram.wkv.scale"
        )
        self.assertEqual(embed, "model.layers.14.engram.embed.scale")
        self.assertEqual(wkv, "model.layers.14.engram.wkv.weight_scale_inv")

    def test_language_only_skips_vision(self):
        self.assertTrue(
            _skip_dsv41_language_only_weight("vision.patch_embed.proj.weight")
        )
        self.assertFalse(
            _skip_dsv41_language_only_weight("model.layers.6.mlp.gate.weight")
        )

    def test_index_scan_when_downloaded(self):
        lines = []
        code = check_index_if_present(lines)
        self.assertIsNone(code, msg="\n".join(lines))


class TestDsv41OfficialConfig(CustomTestCase):
    def test_normalize_remaps_architecture(self):
        values = normalize_deepseek_v41_config(
            {
                "architectures": ["DeepseekV41ForCausalLM"],
                "text_config": {
                    "hidden_size": 5120,
                    "n_routed_experts": 384,
                    "num_hidden_layers": 40,
                },
            }
        )
        self.assertEqual(values["architectures"], ["DeepseekV4ForCausalLM"])
        self.assertEqual(values["hidden_size"], 5120)
        self.assertEqual(values["n_routed_experts"], 384)

    def test_fp8_config_official_quant_block(self):
        cfg = Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [32, 32],
                "scale_fmt": "ue8m0",
                "expert_dtype": "fp4",
            }
        )
        self.assertTrue(cfg.use_mxfp8)
        self.assertTrue(cfg.is_fp4_experts)
        self.assertEqual(list(cfg.weight_block_size), [32, 32])

    def test_on_disk_config_if_present(self):
        if not os.path.isfile(OFFICIAL_CONFIG):
            self.skipTest(f"official config not at {OFFICIAL_CONFIG}")
        raw = json.loads(open(OFFICIAL_CONFIG).read())
        hf = DeepseekV41Config(**raw)
        self.assertEqual(hf.hidden_size, 5120)
        self.assertEqual(hf.n_routed_experts, 384)
        self.assertEqual(hf.moe_intermediate_size, 2304)
        self.assertEqual(tuple(hf.engram_layer_ids), (1, 14))
        self.assertEqual(hf.num_experts_per_tok, 6)


class _TinyVision:
    vision_dim = 32
    vision_n_heads = 4
    vision_inter_dim = 48
    vision_patch_size = 2
    vision_n_layers = 1
    vision_rope_theta = 10000.0
    vision_downsample_ratio = 2
    hidden_size = 16


class _CpuTower(nn.Module):
    def __init__(self):
        super().__init__()
        args = _TinyVision()
        self.vision = CpuViT(args)
        self.aligner = CpuAligner(args)
        self.image_start = nn.Parameter(torch.zeros(args.hidden_size))
        gate = nn.Module()
        gate.e_score_correction_bias_vl = nn.Parameter(torch.zeros(4))
        mlp = nn.Module()
        mlp.gate = gate
        layer = nn.Module()
        layer.mlp = mlp
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])


def _fake_tower_state():
    args = _TinyVision()
    dim = args.vision_dim
    return {
        "vision.blocks.0.attn.wqkv.weight": torch.full((3 * dim, dim), 0.1),
        "vision.blocks.0.attn.wqkv.bias": torch.full((3 * dim,), 0.2),
        "vision.blocks.0.attn.wo.weight": torch.full((dim, dim), 0.3),
        "vision.blocks.0.mlp.w1.weight": torch.full((2 * args.vision_inter_dim, dim), 0.4),
        "aligner.w1.weight": torch.full(
            (args.hidden_size, dim * args.vision_downsample_ratio**2), 0.5
        ),
        "layers.0.ffn.gate.bias_vl": torch.arange(4, dtype=torch.float32),
    }


class TestDsv41CpuVisionLoad(CustomTestCase):
    def test_fake_state_dict_lands_on_cpu_parameters(self):
        module = _CpuTower()
        params = dict(module.named_parameters())
        for name, tensor in _fake_tower_state().items():
            mapped = resolve_dsv41_weight_name(name, tower_on=True)
            self.assertIsNotNone(mapped, name)
            self.assertIn(mapped, params, mapped)
            params[mapped].data.copy_(tensor)
        block = module.vision.blocks[0]
        self.assertTrue(torch.equal(block.attn.wqkv.weight, torch.full((96, 32), 0.1)))
        self.assertTrue(torch.equal(block.attn.wqkv.bias, torch.full((96,), 0.2)))
        self.assertTrue(torch.equal(block.attn.wo.weight, torch.full((32, 32), 0.3)))
        self.assertTrue(torch.equal(block.mlp.w1.weight, torch.full((96, 32), 0.4)))
        self.assertEqual(tuple(module.aligner.w1.weight.shape), (16, 128))
        self.assertTrue(
            torch.equal(
                module.model.layers[0].mlp.gate.e_score_correction_bias_vl,
                torch.arange(4, dtype=torch.float32),
            )
        )
        self.assertEqual(block.attn.wqkv.weight.dtype, torch.float32)
        self.assertEqual(block.attn.wqkv.weight.device.type, "cpu")

    def test_language_only_still_skips_tower_and_bias_vl(self):
        for name in _fake_tower_state():
            self.assertIsNone(resolve_dsv41_weight_name(name, tower_on=False), name)
        self.assertEqual(
            DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
                "layers.6.ffn.gate.bias_vl"
            ),
            "model.layers.6.mlp.gate.bias_vl",
        )
        self.assertEqual(
            DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
                "vision.blocks.0.attn.wqkv.weight"
            ),
            "vision.blocks.0.attn.wqkv.weight",
        )

    def test_streaming_gpu_matches_cpu(self):
        if not torch.cuda.is_available():
            self.skipTest("no CUDA")
        torch.manual_seed(0)
        args = _TinyVision()
        vision = CpuViT(args)
        aligner = CpuAligner(args)
        patches = torch.randn(4, 3, args.vision_patch_size, args.vision_patch_size)
        cpu = aligner(vision(patches, 2, 2), 2, 2)
        gpu = aligner.streaming_gpu(
            vision.streaming_gpu(patches, 2, 2, torch.device("cuda")), 2, 2
        )
        torch.cuda.synchronize()
        self.assertEqual(gpu.dtype, torch.float16)
        self.assertEqual(gpu.device.type, "cuda")
        self.assertTrue(
            torch.allclose(cpu, gpu.float().cpu(), atol=2e-2, rtol=2e-2),
            msg=f"max abs {(cpu - gpu.float().cpu()).abs().max().item():.4f}",
        )

    def test_cpu_forward_span_shape(self):
        args = _TinyVision()
        vision = CpuViT(args)
        aligner = CpuAligner(args)
        patches = torch.zeros(4, 3, args.vision_patch_size, args.vision_patch_size)
        features = aligner(vision(patches, 2, 2), 2, 2)
        self.assertEqual(tuple(features.shape), (1, args.hidden_size))
        self.assertEqual(features.dtype, torch.float32)

    def test_engram_image_mask_includes_hash_pads(self):
        self.assertEqual(1_000_000, MM_PAD_SHIFT_VALUE)
        ids = torch.tensor([1, 129264, 1_000_001, 42])
        mask = _dsv41_image_position_mask(ids, 129264)
        self.assertEqual(mask.tolist(), [False, True, True, False])

    def test_image_extend_sees_schedule_batch_fields(self):
        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            _extend_span_has_image_token,
        )

        class _Hit:
            def contains_image_inputs(self):
                return True

        class _Miss:
            def contains_image_inputs(self):
                return False

        class _HF:
            image_token_id = 7
            vision_n_layers = 1
            language_model_only = False

        class _Worker:
            class target_worker:
                class model_runner:
                    class model_config:
                        hf_config = _HF()

        hit = type("B", (), {})()
        hit.multimodal_inputs = [_Hit()]
        hit.input_ids = None
        self.assertTrue(_extend_span_has_image_token(_Worker(), hit))

        ids = type("B", (), {})()
        ids.multimodal_inputs = [_Miss()]
        ids.input_ids = None
        ids.prefill_input_ids_cpu = torch.tensor([1, 7, 2])
        self.assertTrue(_extend_span_has_image_token(_Worker(), ids))

        text = type("B", (), {})()
        text.multimodal_inputs = None
        text.input_ids = None
        text.prefill_input_ids_cpu = torch.tensor([1, 2, 3])
        self.assertFalse(_extend_span_has_image_token(_Worker(), text))


class TestDsv41CpuMock(CustomTestCase):
    def test_cpu_mock_v1_shape(self):
        report = run_cpu_mock()
        self.assertTrue(report.ok, msg="\n".join(report.lines))
        self.assertEqual(report.exit_code, 0)

    def test_weight_remap_helper(self):
        lines = []
        self.assertIsNone(check_weight_remap(lines), msg="\n".join(lines))
