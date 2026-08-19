# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.gguf import GGUFUninitializedParameter
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    _apply_gguf_grouped_wo_a,
    _fuse_deepseek_v4_wqkv_a_pair,
    _prepare_deepseek_v4_weights,
)
from sglang.srt.utils import set_weight_attrs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _QuantConfig:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name


class TestDeepseek4StreamingWeights(unittest.TestCase):
    def test_quantized_vocab_names_are_remapped(self) -> None:
        remap = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
        self.assertEqual(remap("embed.qweight_type"), "model.embed_tokens.qweight_type")
        self.assertEqual(remap("embed.qweight"), "model.embed_tokens.qweight")
        self.assertEqual(remap("head.qweight_type"), "lm_head.qweight_type")
        self.assertEqual(remap("head.qweight"), "lm_head.qweight")

    def test_expert_pack_does_not_materialize_iterator(self) -> None:
        consumed = []

        def weights():
            consumed.append(True)
            yield "model.norm.weight", torch.ones(1)

        source = weights()
        prepared = _prepare_deepseek_v4_weights(source, _QuantConfig("expert_pack"))
        self.assertIs(prepared, source)
        self.assertEqual(consumed, [])
        self.assertEqual(next(prepared)[0], "model.norm.weight")

    def test_other_formats_preserve_existing_preprocessing(self) -> None:
        consumed = []

        def weights():
            consumed.append(True)
            yield "model.norm.weight", torch.ones(1)

        prepared = _prepare_deepseek_v4_weights(weights(), _QuantConfig("gguf"))
        self.assertEqual(consumed, [])
        self.assertEqual(list(prepared)[0][0], "model.norm.weight")
        self.assertEqual(consumed, [True])

    def test_q8_wqkv_fusion_preserves_rows_and_type(self) -> None:
        fused = _fuse_deepseek_v4_wqkv_a_pair(
            "model.layers.0.self_attn.wqkv_a.qweight",
            {"q": torch.ones((2, 3)), "kv": torch.full((1, 3), 2)},
        )
        self.assertEqual(fused.tolist(), [[1, 1, 1], [1, 1, 1], [2, 2, 2]])
        qtype = _fuse_deepseek_v4_wqkv_a_pair(
            "model.layers.0.self_attn.wqkv_a.qweight_type",
            {"q": torch.tensor(8), "kv": torch.tensor(8)},
        )
        self.assertEqual(qtype.item(), 8)
        with self.assertRaisesRegex(ValueError, "different GGUF"):
            _fuse_deepseek_v4_wqkv_a_pair(
                "model.layers.0.self_attn.wqkv_a.qweight_type",
                {"q": torch.tensor(8), "kv": torch.tensor(9)},
            )

    def test_grouped_gguf_wo_a_makes_token_slices_contiguous(self) -> None:
        o = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
        qweight = torch.arange(6, dtype=torch.uint8).view(6, 1)
        seen_inputs = []

        def matmul_fn(x, weight, qweight_type):
            self.assertTrue(x.is_contiguous())
            self.assertEqual(qweight_type, 8)
            seen_inputs.append(x.clone())
            return x[:, : weight.shape[0]]

        result = _apply_gguf_grouped_wo_a(
            o, qweight, qweight_type=8, o_lora_rank=2, matmul_fn=matmul_fn
        )

        self.assertEqual(len(seen_inputs), 3)
        for group_id, seen in enumerate(seen_inputs):
            torch.testing.assert_close(seen, o[:, group_id, :])
        torch.testing.assert_close(result, o[:, :, :2])

    def test_replicated_linear_materializes_gguf_weight(self) -> None:
        qweight_type = torch.nn.Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(qweight_type, {"is_gguf_weight_type": True})
        ReplicatedLinear.weight_loader(None, qweight_type, torch.tensor(8))
        self.assertEqual(qweight_type.weight_type, 8)

        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(qweight, {"is_gguf_weight": True})
        loaded = torch.arange(12, dtype=torch.uint8).reshape(3, 4)
        ReplicatedLinear.weight_loader(None, qweight, loaded)
        self.assertEqual(tuple(qweight.shape), (3, 4))
        torch.testing.assert_close(qweight, loaded)


if __name__ == "__main__":
    unittest.main()
