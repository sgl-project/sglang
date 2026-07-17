import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.mimo_v2 import (
    _collect_mimo_qkv_adaptations,
    repack_mimo_v2_fused_qkv_block_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _block_quantize_fixture(weight, block_size):
    block_n, block_k = block_size
    n, k = weight.shape
    padded = torch.zeros(
        (
            (n + block_n - 1) // block_n * block_n,
            (k + block_k - 1) // block_k * block_k,
        ),
        dtype=torch.float32,
    )
    padded[:n, :k] = weight
    blocks = padded.view(
        padded.shape[0] // block_n,
        block_n,
        padded.shape[1] // block_k,
        block_k,
    )
    scale = (blocks.abs().amax(dim=(1, 3)) / 448.0).clamp(min=1e-12)
    quantized = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return quantized.view_as(padded)[:n, :k].contiguous(), scale.contiguous()


class TestMiMoV2CPWeightAdapter(CustomTestCase):
    def test_mtp_draft_input_embedding_clamps_sentinel_ids(self):
        from sglang.srt.model_executor.runner.eager_runner import (
            _get_cp_v2_input_embeds,
        )

        embedding = torch.nn.Embedding.from_pretrained(
            torch.arange(12, dtype=torch.float32).reshape(4, 3)
        )

        class DraftModel:
            config = SimpleNamespace(
                architectures=["MiMoV2MTP"],
                vocab_size=4,
            )

            def get_input_embedding(self, input_ids):
                raise AssertionError(
                    "MiMoV2MTP inherits an unavailable singular accessor"
                )

            def get_input_embeddings(self):
                return embedding

        input_ids = torch.tensor([-100, 1, 999])
        expected = embedding(torch.tensor([0, 1, 3]))

        self.assertTrue(
            torch.equal(_get_cp_v2_input_embeds(DraftModel(), input_ids), expected)
        )

    def test_mtp_draft_qkv_is_collected_for_cp_repacking(self):
        class FakeQKVParallelLinear:
            def __init__(self):
                self.q_proj_shard_size = 32
                self.kv_proj_shard_size = 24
                self.v_proj_shard_size = 16
                self.weight = torch.empty((72, 4), dtype=torch.float8_e4m3fn)
                self.weight_scale_inv = torch.empty((18, 2), dtype=torch.float32)
                self.quant_method = SimpleNamespace(
                    quant_config=SimpleNamespace(weight_block_size=[4, 2])
                )

        qkv_proj = FakeQKVParallelLinear()
        model = SimpleNamespace(
            config=SimpleNamespace(
                architectures=["MiMoV2MTP"],
                attention_projection_layout="fused_qkv",
                num_key_value_heads=4,
            ),
            named_modules=lambda: [("model.mtp_block.self_attn.qkv_proj", qkv_proj)],
        )

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            adaptations = _collect_mimo_qkv_adaptations(model)

        self.assertEqual(len(adaptations), 1)
        self.assertEqual(
            adaptations[0].module_name,
            "model.mtp_block.self_attn.qkv_proj",
        )

    def test_repack_fused_tp4_qkv_requantizes_across_block_boundaries(self):
        block_size = [4, 2]
        q_values = [1.0, 2.0, 4.0, 8.0]
        k_values = [16.0, 32.0, 64.0, 128.0]
        v_values = [-1.0, -2.0, -4.0, -8.0]

        checkpoint_groups = []
        for rank in range(4):
            checkpoint_groups.append(
                torch.cat(
                    [
                        torch.full((8, 4), q_values[rank]),
                        torch.full((6, 4), k_values[rank]),
                        torch.full((4, 4), v_values[rank]),
                    ]
                )
            )

        quantized_groups = [
            _block_quantize_fixture(group, block_size) for group in checkpoint_groups
        ]
        checkpoint_weight = torch.cat([item[0] for item in quantized_groups])
        checkpoint_scale = torch.cat([item[1] for item in quantized_groups])

        repacked_weight, repacked_scale = repack_mimo_v2_fused_qkv_block_fp8(
            checkpoint_weight,
            checkpoint_scale,
            q_rows=32,
            k_rows=24,
            v_rows=16,
            checkpoint_tp_size=4,
            block_size=block_size,
            output_dtype=torch.float32,
        )
        actual = block_quant_dequant(
            repacked_weight,
            repacked_scale,
            block_size,
            torch.float32,
        )
        expected = torch.cat(
            [
                *(torch.full((8, 4), value) for value in q_values),
                *(torch.full((6, 4), value) for value in k_values),
                *(torch.full((4, 4), value) for value in v_values),
            ]
        )

        self.assertEqual(tuple(checkpoint_weight.shape), (72, 4))
        self.assertEqual(tuple(checkpoint_scale.shape), (20, 2))
        self.assertEqual(tuple(repacked_weight.shape), (72, 4))
        self.assertEqual(tuple(repacked_scale.shape), (18, 2))
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
