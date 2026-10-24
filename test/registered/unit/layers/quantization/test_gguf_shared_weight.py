import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.gguf import get_gguf_dequantized_weight
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _dense_module() -> torch.nn.Module:
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(4, 8))
    return module


def _quantized_module(weight_type: int) -> torch.nn.Module:
    """A module as GGUF quantization leaves it: packed qweight, no .weight."""
    module = torch.nn.Module()
    module.qweight = torch.zeros(4, 8, dtype=torch.uint8)
    module.qweight_type = SimpleNamespace(weight_type=weight_type)
    return module


class TestGgufSharedWeight(CustomTestCase):
    def test_dense_module_returns_weight_untouched(self):
        module = _dense_module()
        with patch(
            "sglang.srt.layers.quantization.gguf.dequantize_gguf_weight"
        ) as dequantize:
            result = get_gguf_dequantized_weight(module, torch.bfloat16)

        self.assertIs(result, module.weight)
        dequantize.assert_not_called()

    def test_quantized_module_is_dequantized(self):
        module = _quantized_module(weight_type=12)
        sentinel = torch.ones(4, 16, dtype=torch.bfloat16)
        with patch(
            "sglang.srt.layers.quantization.gguf.dequantize_gguf_weight",
            return_value=sentinel,
        ) as dequantize:
            result = get_gguf_dequantized_weight(module, torch.bfloat16)

        self.assertIs(result, sentinel)
        dequantize.assert_called_once_with(module.qweight, 12, torch.bfloat16)

    def test_dequantized_weight_is_cached_per_module(self):
        module = _quantized_module(weight_type=12)
        sentinel = torch.ones(4, 16, dtype=torch.bfloat16)
        with patch(
            "sglang.srt.layers.quantization.gguf.dequantize_gguf_weight",
            return_value=sentinel,
        ) as dequantize:
            first = get_gguf_dequantized_weight(module, torch.bfloat16)
            second = get_gguf_dequantized_weight(module, torch.bfloat16)

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        dequantize.assert_called_once()

    def test_target_getters_dequantize_shared_embed_and_head(self):
        """Regression for the AttributeError on `.weight` of a GGUF target.

        The draft model shares the target embedding/lm_head through
        `get_embed`/`get_embed_and_head`, which read `.weight`. GGUF keeps the
        packed `qweight` instead, so the getters must route through the
        dequantizing helper.
        """
        embed = _quantized_module(weight_type=12)
        head = _quantized_module(weight_type=14)
        embed_out = torch.ones(4, 16, dtype=torch.bfloat16)
        head_out = torch.full((4, 16), 2.0, dtype=torch.bfloat16)

        def _fake_dequantize(qweight, qweight_type, dtype):
            if qweight is embed.qweight:
                self.assertEqual(qweight_type, 12)
                return embed_out
            self.assertIs(qweight, head.qweight)
            self.assertEqual(qweight_type, 14)
            return head_out

        fake_target = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=embed),
            lm_head=head,
            config=SimpleNamespace(torch_dtype=torch.bfloat16),
        )
        with patch(
            "sglang.srt.layers.quantization.gguf.dequantize_gguf_weight",
            side_effect=_fake_dequantize,
        ) as dequantize:
            got_embed, got_head = Qwen3NextForCausalLM.get_embed_and_head(fake_target)
            got_embed_only = Qwen3NextForCausalLM.get_embed(fake_target)

        self.assertIs(got_embed, embed_out)
        self.assertIs(got_head, head_out)
        self.assertIs(got_embed_only, embed_out)
        # Two modules dequantized once each; get_embed reuses the cached tensor.
        self.assertEqual(dequantize.call_count, 2)
        for call in dequantize.call_args_list:
            self.assertEqual(call.args[2], torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
