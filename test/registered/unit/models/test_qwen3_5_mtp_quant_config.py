import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.quantization.quark.utils import should_ignore_layer
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, _mtp_quant_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")

# The real `mtp.*` exclude entries of the two AMD Quark MXFP4 checkpoints.
# Quark names layers in the checkpoint namespace, where the draft is prefixed
# `mtp.`. The two lists are identical apart from the routed experts:
# Qwen3.5-397B excludes all 512x3 draft expert projections, so its whole MTP
# module is bf16; Qwen3.8-2.4T excludes none of them, so its draft experts stay
# MXFP4 while attention, the shared expert and fc are bf16.
_MIXED_EXCLUDES = [  # amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 (all 10 mtp.* entries)
    "mtp.fc",
    "mtp.layers.0.mlp.gate",
    "mtp.layers.0.mlp.shared_expert.down_proj",
    "mtp.layers.0.mlp.shared_expert.gate_proj",
    "mtp.layers.0.mlp.shared_expert.up_proj",
    "mtp.layers.0.mlp.shared_expert_gate",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj",
]
_ALL_BF16_EXCLUDES = _MIXED_EXCLUDES + [  # amd/Qwen3.5-397B-A17B-MXFP4
    f"mtp.layers.0.mlp.experts.{expert}.{proj}"
    for expert in range(512)
    for proj in ("gate_proj", "up_proj", "down_proj")
]


class _FakeQuantConfig:
    def __init__(self, name, exclude_layers):
        self._name = name
        self.exclude_layers = list(exclude_layers)

    def get_name(self):
        return self._name


class TestQwen3_5MTPQuantConfig(CustomTestCase):
    def test_mixed_quark_checkpoint_keeps_quantization(self):
        """Routed experts stay MXFP4, so the draft must stay quantized."""
        quant_config = _FakeQuantConfig("quark", _MIXED_EXCLUDES)

        self.assertIs(_mtp_quant_config(quant_config), quant_config)

    def test_fully_bf16_quark_checkpoint_skips_quantization(self):
        """Regression guard for #23146: a bf16 draft must stay unquantized."""
        quant_config = _FakeQuantConfig("quark", _ALL_BF16_EXCLUDES)

        self.assertIsNone(_mtp_quant_config(quant_config))

    def test_quark_checkpoint_without_mtp_excludes_keeps_quantization(self):
        quant_config = _FakeQuantConfig("quark", ["model.layers.0.self_attn.q_proj"])

        self.assertIs(_mtp_quant_config(quant_config), quant_config)

    def test_non_quark_quant_config_is_untouched(self):
        quant_config = _FakeQuantConfig("fp8", _MIXED_EXCLUDES)

        self.assertIs(_mtp_quant_config(quant_config), quant_config)

    def test_mtp_reuses_target_packed_modules_mapping(self):
        self.assertEqual(
            Qwen3_5ForCausalLMMTP.packed_modules_mapping,
            Qwen3_5ForCausalLM.packed_modules_mapping,
        )

    def test_excluded_fused_attention_projection_is_ignored(self):
        """The mapping has to expand qkv_proj before the exclude list matches.

        Without it `should_ignore_layer` compares the fused name against
        `q_proj`/`k_proj`/`v_proj` entries, finds nothing, and the layer is
        quantized against bf16 checkpoint shards.
        """
        ignored = should_ignore_layer(
            "mtp.layers.0.self_attn.qkv_proj",
            ignore=_MIXED_EXCLUDES,
            fused_mapping=Qwen3_5ForCausalLMMTP.packed_modules_mapping,
        )

        self.assertTrue(ignored)

    def test_unexcluded_fused_attention_projection_is_not_ignored(self):
        ignored = should_ignore_layer(
            "model.layers.0.self_attn.qkv_proj",
            ignore=_MIXED_EXCLUDES,
            fused_mapping=Qwen3_5ForCausalLMMTP.packed_modules_mapping,
        )

        self.assertFalse(ignored)


class _RowRecordingNorm(nn.Module):
    """Stands in for GemmaRMSNorm, whose CUDA kernel rejects a zero-row launch."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def forward(self, x):
        if x.shape[0] == 0:
            raise RuntimeError("CUDA launch failed: cudaErrorInvalidValue")
        self.rows.append(x.shape[0])
        return x


class _EmbedThenIdentity(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, hidden_size)

    def forward(self, input_ids, positions, forward_batch, hidden_states):
        return hidden_states


class TestQwen3_5MTPForwardRows(CustomTestCase):
    _HIDDEN = 4

    def _forward(self, num_rows):
        mtp = SimpleNamespace(
            quant_config=None,
            model=_EmbedThenIdentity(self._HIDDEN),
            pre_fc_norm_embedding=_RowRecordingNorm(),
            pre_fc_norm_hidden=_RowRecordingNorm(),
            fc=nn.Linear(2 * self._HIDDEN, self._HIDDEN, bias=False),
            lm_head=None,
            logits_processor=lambda input_ids, hidden_states, lm_head, fb: (
                hidden_states
            ),
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            mm_input_embeds=None,
            contains_mm_inputs=lambda: False,
            spec_info=SimpleNamespace(
                hidden_states=torch.zeros(num_rows, self._HIDDEN)
            ),
        )
        out = Qwen3_5ForCausalLMMTP.forward(
            mtp,
            input_ids=torch.zeros(num_rows, dtype=torch.long),
            positions=torch.zeros(num_rows, dtype=torch.long),
            forward_batch=forward_batch,
        )
        return out, mtp

    def test_zero_row_extend_skips_the_pre_fc_norms(self):
        """An idle attention-DP rank in a prefill round reaches the draft as a
        zero-row EXTEND (A2A MoE keeps local token counts); its forward must not
        launch the norms, while a rank with rows still normalizes them."""
        out, mtp = self._forward(num_rows=0)
        self.assertEqual(tuple(out.shape), (0, self._HIDDEN))
        self.assertEqual(mtp.pre_fc_norm_embedding.rows, [])

        _, mtp = self._forward(num_rows=3)
        self.assertEqual(mtp.pre_fc_norm_embedding.rows, [3])
        self.assertEqual(mtp.pre_fc_norm_hidden.rows, [3])


if __name__ == "__main__":
    unittest.main()
