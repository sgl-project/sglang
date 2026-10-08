"""Tied Falcon-H1 embeddings must stay in the model dtype.

`FalconH1ForCausalLM` used to alias `lm_head` to the embedding module and call
`Module.float()` on it. That rewrites the shared parameter in place, so the
token embedding becomes fp32 while RMSNorm weights stay bf16. The first
forward then dies in fused RMSNorm (`failed to dispatch data type Float`)
during server warmup.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.falcon_h1 import FalconH1Config
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models import falcon_h1
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase, published_topology


class _TinyDecoderLayer(nn.Module):
    """Stand-in for the hybrid block: the dtype crash is the first RMSNorm."""

    def __init__(self, config, layer_id, quant_config=None, prefix="", alt_stream=None):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def _config(*, tie_word_embeddings: bool) -> FalconH1Config:
    return FalconH1Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_d_ssm=32,
        mamba_n_heads=4,
        mamba_d_head=8,
        mamba_n_groups=1,
        mamba_d_state=8,
        tie_word_embeddings=tie_word_embeddings,
    )


class TestFalconH1TiedEmbeddings(CustomTestCase):
    def _build(self, *, tie_word_embeddings: bool):
        config = _config(tie_word_embeddings=tie_word_embeddings)
        group = SimpleNamespace(is_first_rank=True, is_last_rank=True, world_size=1)
        with (
            published_topology(tp_size=1, pp_size=1),
            get_parallel().override(pp_group=group, tp_group=group),
            patch.dict(
                falcon_h1.ALL_DECODER_LAYER_TYPES, {"falcon_h1": _TinyDecoderLayer}
            ),
            set_default_torch_dtype(torch.bfloat16),
        ):
            model = falcon_h1.FalconH1ForCausalLM(config)
            self._check(model, tie_word_embeddings=tie_word_embeddings)
        return model

    def _check(self, model, *, tie_word_embeddings: bool):
        embed = model.model.embed_tokens.weight
        layernorm = model.model.layers[0].input_layernorm
        self.assertEqual(embed.dtype, torch.bfloat16)
        self.assertEqual(layernorm.weight.dtype, torch.bfloat16)
        self.assertEqual(model.model.final_layernorm.weight.dtype, torch.bfloat16)

        hidden = model.model.embed_tokens(torch.tensor([1, 2, 3]))
        self.assertEqual(hidden.dtype, layernorm.weight.dtype)
        normalized = layernorm(hidden)
        self.assertEqual(normalized.dtype, torch.bfloat16)

        if tie_word_embeddings:
            self.assertEqual(model.lm_head.weight.data_ptr(), embed.data_ptr())
            self.assertTrue(model.logits_processor.use_fp32_lm_head)
        else:
            self.assertEqual(model.lm_head.weight.dtype, torch.float32)
            self.assertNotEqual(model.lm_head.weight.data_ptr(), embed.data_ptr())

        logits = model.logits_processor._compute_lm_head(hidden[:1], model.lm_head)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(embed.dtype, torch.bfloat16)

    def test_tied_embeddings_stay_in_model_dtype(self):
        config = _config(tie_word_embeddings=True)
        model = self._build(tie_word_embeddings=True)

        loaded = torch.arange(
            config.vocab_size * config.hidden_size, dtype=torch.bfloat16
        ).reshape(config.vocab_size, config.hidden_size)
        model.load_weights([("model.embed_tokens.weight", loaded)])
        embed = model.model.embed_tokens.weight
        self.assertEqual(embed.dtype, torch.bfloat16)
        torch.testing.assert_close(embed[: config.vocab_size].float(), loaded.float())

        # A separate lm_head tensor in the checkpoint must not retarget or
        # upcast the shared embedding.
        poison = torch.full(loaded.shape, 7, dtype=torch.float32)
        before = embed.detach().clone()
        model.load_weights([("lm_head.weight", poison)])
        self.assertEqual(embed.dtype, torch.bfloat16)
        torch.testing.assert_close(embed.float(), before.float())

        hidden = torch.randn(2, config.hidden_size, dtype=torch.bfloat16)
        logits = model.logits_processor._compute_lm_head(hidden, model.lm_head)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(model.model.embed_tokens.weight.dtype, torch.bfloat16)

    def test_untied_lm_head_is_fp32_without_touching_embeddings(self):
        self._build(tie_word_embeddings=False)


if __name__ == "__main__":
    unittest.main()
