"""The NextN draft of a ModelOpt NVFP4 checkpoint is built unquantized, but its
attention keeps the checkpoint's KV-cache method, as the target's attention does."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp8KVCacheMethod,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.models import deepseek_nextn
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _DecoderLayer(nn.Module):
    """Stands in for DeepseekV2DecoderLayer: its attention gets the quant
    config the layer is built with, as in DeepseekV2AttentionMLA."""

    def __init__(self, config, layer_id, quant_config=None, prefix="", **kwargs):
        super().__init__()
        self.quant_config = quant_config
        self.self_attn = nn.Module()
        self.self_attn.attn_mqa = RadixAttention(
            1,
            8,
            1.0,
            1,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn.attn_mqa", prefix),
        )


def _build_nextn():
    quant_config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo="FP8",
        group_size=16,
        exclude_modules=["lm_head", "model.layers.61*"],
    )
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        rms_norm_eps=1e-6,
        qk_rope_head_dim=64,
        num_hidden_layers=61,
    )
    with patch.multiple(
        deepseek_nextn,
        DeepseekV2DecoderLayer=_DecoderLayer,
        VocabParallelEmbedding=lambda num, dim, **kwargs: nn.Embedding(num, dim),
        get_embedding_tp_kwargs=lambda: {},
        _is_cuda=False,
        _is_npu=False,
    ):
        return deepseek_nextn.DeepseekModelNextN(config, quant_config, prefix="model")


class TestDeepseekNextNKVCacheQuant(CustomTestCase):
    def test_nvfp4_nextn_keeps_kv_cache_method(self):
        model = _build_nextn()
        attn = model.decoder.self_attn.attn_mqa

        # The NextN weights stay unquantized ...
        self.assertIsNone(model.decoder.quant_config)
        # ... but the attention keeps the checkpoint's KV-cache method.
        self.assertIsInstance(attn.quant_method, ModelOptFp8KVCacheMethod)
        params = dict(model.named_parameters())
        self.assertIs(params["decoder.self_attn.attn_mqa.k_scale"], attn.k_scale)
        self.assertIs(params["decoder.self_attn.attn_mqa.v_scale"], attn.v_scale)

        # The MLA loader skips checkpoint KV-cache scales, so post-loading
        # gives the 1.0 the target's attn_mqa uses.
        attn.quant_method.process_weights_after_loading(attn)
        self.assertEqual(attn.k_scale_float, 1.0)
        self.assertEqual(attn.v_scale_float, 1.0)


if __name__ == "__main__":
    unittest.main()
