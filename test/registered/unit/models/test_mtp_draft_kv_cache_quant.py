"""The MTP and Eagle3 drafts of ModelOpt checkpoints that keep the draft in BF16
are built unquantized, but their attention keeps the checkpoint's KV-cache
method, as the target's attention does."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8Config,
    ModelOptFp8KVCacheMethod,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.models import kimi_k25_eagle3, qwen3_5_mtp, qwen4_exp_mtp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _DraftStack(nn.Module):
    """Stands in for the draft's decoder stack (Qwen3_5ForCausalLM, Qwen4ExpModel,
    Eagle3MLAModel): its attention gets the quant config the stack is built
    with, as in the real attention modules."""

    def __init__(self, config, quant_config=None, prefix="", **kwargs):
        super().__init__()
        self.quant_config = quant_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].attn = RadixAttention(
            1,
            8,
            1.0,
            1,
            0,
            quant_config=quant_config,
            prefix=add_prefix("layers.0.attn", prefix),
        )


def _config(**kwargs):
    return SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        num_hidden_layers=1,
        **kwargs,
    )


def _nvfp4_config():
    return ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo="FP8",
        group_size=16,
        exclude_modules=["lm_head", "mtp*"],
    )


def _mixed_config():
    # MIXED_PRECISION lists no mtp.* layer: the MTP module is BF16.
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": "FP8",
            "quantized_layers": {
                "model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16}
            },
        }
    )


def _fp8_config():
    return ModelOptFp8Config(
        is_checkpoint_fp8_serialized=True,
        kv_cache_quant_method="FP8",
        exclude_modules=["lm_head"],
    )


def _parallel():
    return SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=True), enable_dp_lm_head=False
    )


_COMMON = dict(get_parallel=_parallel, LogitsProcessor=lambda config: None)


def _build_qwen3_5_mtp(quant_config):
    with patch.multiple(qwen3_5_mtp, Qwen3_5ForCausalLM=_DraftStack, **_COMMON):
        return qwen3_5_mtp.Qwen3_5ForCausalLMMTP(_config(), quant_config)


def _build_qwen4_exp_mtp(quant_config):
    with patch.multiple(
        qwen4_exp_mtp,
        Qwen4ExpModel=_DraftStack,
        ParallelLMHead=lambda num, dim, **kwargs: nn.Linear(dim, num, bias=False),
        **_COMMON,
    ):
        return qwen4_exp_mtp.Qwen4ExpForCausalLMMTP(_config(hc_count=1), quant_config)


def _build_kimi_k25_eagle3(quant_config):
    with patch.multiple(kimi_k25_eagle3, Eagle3MLAModel=_DraftStack, **_COMMON):
        return kimi_k25_eagle3.Eagle3DeepseekV2ForCausalLM(_config(), quant_config)


class TestMTPDraftKVCacheQuant(CustomTestCase):
    def _assert_bf16_draft_keeps_kv_cache_method(self, model):
        attn = model.model.layers[0].attn

        # The draft weights stay unquantized ...
        self.assertIsNone(model.quant_config)
        self.assertIsNone(model.model.quant_config)
        # ... but the attention keeps the checkpoint's KV-cache method.
        self.assertIsInstance(attn.quant_method, ModelOptFp8KVCacheMethod)
        params = dict(model.named_parameters())
        self.assertIs(params["model.layers.0.attn.k_scale"], attn.k_scale)
        self.assertIs(params["model.layers.0.attn.v_scale"], attn.v_scale)
        return attn

    def test_qwen3_5_mtp(self):
        for quant_config in (_nvfp4_config(), _mixed_config()):
            with self.subTest(quant=quant_config.get_name()):
                model = _build_qwen3_5_mtp(quant_config)
                attn = self._assert_bf16_draft_keeps_kv_cache_method(model)

                # The MTP layer's scales, named the way ModelOpt exports them,
                # load through the draft's loader.
                model.load_weights(
                    [
                        ("mtp.layers.0.self_attn.k_proj.k_scale", torch.tensor(0.0625)),
                        ("mtp.layers.0.self_attn.v_proj.v_scale", torch.tensor(0.125)),
                    ]
                )
                attn.quant_method.process_weights_after_loading(attn)
                self.assertEqual(attn.k_scale_float, 0.0625)
                self.assertEqual(attn.v_scale_float, 0.125)

    def test_qwen4_exp_mtp(self):
        self._assert_bf16_draft_keeps_kv_cache_method(
            _build_qwen4_exp_mtp(_nvfp4_config())
        )

    def test_kimi_k25_eagle3(self):
        attn = self._assert_bf16_draft_keeps_kv_cache_method(
            _build_kimi_k25_eagle3(_nvfp4_config())
        )

        # The MLA draft loads no checkpoint KV-cache scales, so post-loading
        # gives the 1.0 the target's MLA attention uses.
        attn.quant_method.process_weights_after_loading(attn)
        self.assertEqual(attn.k_scale_float, 1.0)
        self.assertEqual(attn.v_scale_float, 1.0)

    def test_quantized_draft_is_unchanged(self):
        model = _build_qwen3_5_mtp(_fp8_config())

        self.assertIsInstance(model.model.quant_config, ModelOptFp8Config)
        self.assertIsInstance(
            model.model.layers[0].attn.quant_method, ModelOptFp8KVCacheMethod
        )


if __name__ == "__main__":
    unittest.main()
