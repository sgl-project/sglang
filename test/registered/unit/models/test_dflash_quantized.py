"""Pack-quantized (compressed-tensors W4A16) DFlash drafts must load `fc`.

The draft's target-context projection `fc` used to be a plain `nn.Linear`,
whose only parameter is `fc.weight`. A checkpoint that stores `fc`
pack-quantized (`fc.weight_packed` / `fc.weight_scale` / `fc.weight_shape`)
dropped all three keys silently, leaving `fc` at random init; the draft then
proposed near-random tokens and acceptance collapsed to the bonus token.

These tests pin `fc` to a quantization-aware linear so both dense
(`fc.weight`) and pack-quantized checkpoints load into it, and that a wrong
number of context features still fails loudly instead of loading silently.
"""

import unittest

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# Small stand-in for a real DFlash2 draft checkpoint (e.g. the 5-layer,
# hidden-5120, 5-context-feature Qwen DFlash2 export): 2 layers, hidden 128,
# 4 target-layer features, packed 4-bit group-128 symmetric weights.
DRAFT_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "is_causal": False,
    "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 1,
        "selector_rank": 16,
        "selector_top_k": 4,
        "target_layer_ids": [0, 1, 2, 3],
    },
    "dtype": "bfloat16",
    "eos_token_id": 2,
    "head_dim": 16,
    "hidden_act": "silu",
    "hidden_size": 128,
    "initializer_range": 0.02,
    "intermediate_size": 256,
    "layer_types": ["sliding_attention", "sliding_attention"],
    "max_position_embeddings": 262144,
    "max_window_layers": 2,
    "model_type": "qwen3",
    "num_attention_heads": 8,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "num_target_layers": 4,
    "pad_token_id": 2,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    "sliding_window": 2048,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": True,
    "vocab_size": 64,
}

# Mirrors the published pack-quantized W4A16 compressed-tensors config.
QUANT_CONFIG = {
    "config_groups": {
        "group_0": {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": 128,
                "num_bits": 4,
                "observer": "memoryless_minmax",
                "observer_kwargs": {},
                "scale_dtype": None,
                "strategy": "group",
                "symmetric": True,
                "type": "int",
                "zp_dtype": None,
            },
        }
    },
    "format": "pack-quantized",
    "global_compression_ratio": None,
    "ignore": [
        "re:.*kernel_projection$",
        "re:.*candidate_selector.*",
        "re:.*hidden_projection$",
    ],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "sparsity_config": {},
    "transform_config": {},
    "version": "0.17.0",
}

TP_OVERRIDE = dict(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0)


def _fc_in_out(model):
    """(out_features, in_features) of the draft fc, for either a plain
    nn.Linear (pre-quantization-support) or a quantization-aware linear."""
    fc = model.fc
    if isinstance(fc, torch.nn.Linear):
        return fc.out_features, fc.in_features
    return fc.output_size, fc.input_size


class TestDFlashQuantizedFc(CustomTestCase):
    def test_pack_quantized_fc_loads(self):
        """The W4A16 regression: fc.weight_packed/_scale/_shape must land in fc."""
        quant_config = CompressedTensorsConfig.from_config(dict(QUANT_CONFIG))
        with get_parallel().override(**TP_OVERRIDE):
            model = DFlash2DraftModel(
                PretrainedConfig(**DRAFT_CONFIG),
                quant_config=quant_config,
                prefix="",
            )

        # fc must be quantization-aware; a plain nn.Linear has no packed
        # parameters to receive the checkpoint's fc tensors.
        self.assertIsInstance(model.fc, ReplicatedLinear)
        self.assertIsInstance(model.fc.quant_method, CompressedTensorsLinearMethod)

        out, inp = _fc_in_out(model)
        packed = (torch.arange(out * (inp // 8), dtype=torch.int32) % 16).view(
            out, inp // 8
        )
        # Scales ship fp16 in real checkpoints; loading must cast them in.
        scale = (torch.rand(out, inp // 128, dtype=torch.float32) + 0.5).to(
            torch.float16
        )
        shape = torch.tensor([out, inp], dtype=torch.int64)

        model.load_weights(
            iter(
                [
                    ("fc.weight_packed", packed),
                    ("fc.weight_scale", scale),
                    ("fc.weight_shape", shape),
                ]
            )
        )

        self.assertTrue(torch.equal(model.fc.weight_packed.data, packed))
        self.assertTrue(torch.equal(model.fc.weight_scale.data.float(), scale.float()))
        self.assertTrue(torch.equal(model.fc.weight_shape.data, shape))

    def test_dense_fc_still_loads(self):
        """BF16 drafts keep working: fc.weight loads unchanged."""
        with get_parallel().override(**TP_OVERRIDE):
            model = DFlash2DraftModel(
                PretrainedConfig(**DRAFT_CONFIG),
                quant_config=None,
                prefix="",
            )
        out, inp = _fc_in_out(model)
        weight = torch.randn(out, inp)
        model.load_weights(iter([("fc.weight", weight)]))
        self.assertTrue(torch.equal(model.fc.weight, weight))

    def test_fc_shape_mismatch_raises(self):
        """A wrong number of context features must fail loudly, not load."""
        with get_parallel().override(**TP_OVERRIDE):
            model = DFlash2DraftModel(
                PretrainedConfig(**DRAFT_CONFIG),
                quant_config=None,
                prefix="",
            )
        out, inp = _fc_in_out(model)
        bad = torch.randn(out, 3 * inp)
        with self.assertRaisesRegex(ValueError, "fc.weight shape mismatch"):
            model.load_weights(iter([("fc.weight", bad)]))


if __name__ == "__main__":
    unittest.main()
