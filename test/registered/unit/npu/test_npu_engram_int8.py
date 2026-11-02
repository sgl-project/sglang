"""CPU checks for the ModelSlim Engram table's INT8/E8M0 storage contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import engram
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _storage(rows=34, dim=64):
    return {
        "format": "int8_sym",
        "weight_dtype": "I8",
        "weight_shape": [rows, dim],
        "layout": "row_major",
        "group_size": 32,
        "axis": 1,
        "scale_suffix": "scale",
        "scale_dtype": "F8_E8M0",
        "scale_shape": [rows, dim // 32],
    }


def _table(storage, *, rank=0, size=1):
    parallel = SimpleNamespace(tp_rank=rank, tp_size=size)
    config = SimpleNamespace(placement="gpu", lookup="reduce_scatter")
    with (
        patch.object(engram, "get_parallel", return_value=parallel),
        patch.object(engram, "resolve_engram_config", return_value=config),
    ):
        return engram.EngramEmbedding(34, 64, 1, (0, 34), storage=storage)


class TestEngramInt8Storage(unittest.TestCase):
    def test_native_table_dtypes_are_unchanged(self):
        table = _table(None)
        self.assertEqual(table.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(table.scale.dtype, torch.float8_e8m0fnu)

    def test_int8_table_uses_no_fp8_tensor_storage(self):
        table = _table(_storage(), rank=15, size=16)
        self.assertEqual(table.weight.dtype, torch.int8)
        self.assertEqual(table.scale.dtype, torch.uint8)
        self.assertEqual(table.row_start, 31)
        self.assertEqual(tuple(table.weight.shape), (3, 64))
        self.assertEqual(tuple(table.scale.shape), (3, 2))

    def test_load_preserves_signed_weights_and_e8m0_bits(self):
        table = _table(_storage(), rank=1, size=2)
        weights = (torch.arange(34 * 64).view(34, 64) % 256 - 128).to(torch.int8)
        scales = (torch.arange(34 * 2).view(34, 2) + 100).to(torch.uint8)
        table._load_rows(table.weight, weights)
        table._load_rows(table.scale, scales.view(torch.float8_e8m0fnu))
        torch.testing.assert_close(table.weight, weights[17:])
        torch.testing.assert_close(table.scale, scales[17:])
        # Raw-byte scale loading is equivalent, never a numeric cast to UINT8.
        table._load_rows(table.scale, scales)
        torch.testing.assert_close(table.scale, scales[17:])

    def test_lookup_dequantizes_only_owned_rows(self):
        table = _table(_storage(), rank=1, size=2)
        weights = (torch.arange(34 * 64).view(34, 64) % 256 - 128).to(torch.int8)
        scales = torch.tensor([125, 128], dtype=torch.uint8).expand(34, 2)
        table._load_rows(table.weight, weights)
        table._load_rows(table.scale, scales)
        indices = torch.tensor([[17, 33, 16, -1], [18, 17, 34, 0]])
        with (
            patch.object(engram, "_cuda_kernels", return_value=False),
            patch.object(engram, "_needs_fp8_byte_view", side_effect=AssertionError),
        ):
            result = table._owned_rows(indices)
        expected = torch.zeros(2, 4, 64, dtype=torch.bfloat16)
        for i in range(2):
            for j in range(4):
                index = int(indices[i, j])
                if 17 <= index < 34:
                    expected[i, j] = (
                        (
                            weights[index].float().view(2, 32)
                            * torch.tensor([0.25, 2.0])[:, None]
                        )
                        .flatten()
                        .bfloat16()
                    )
        torch.testing.assert_close(result, expected, atol=0, rtol=0)

    def test_native_a5_byte_lookup_still_dequantizes_fp8(self):
        table = _table(None)
        weights = torch.full((34, 64), -1.5).to(torch.float8_e4m3fn)
        scales = torch.full((34, 2), 126, dtype=torch.uint8)
        table._load_rows(table.weight, weights)
        table._load_rows(table.scale, scales.view(torch.float8_e8m0fnu))
        with (
            patch.object(engram, "_cuda_kernels", return_value=False),
            patch.object(engram, "_needs_fp8_byte_view", return_value=True),
        ):
            result = table._owned_rows(torch.tensor([[0, 33, -1]]))
        expected = torch.full((1, 3, 64), -0.75, dtype=torch.bfloat16)
        expected[:, -1] = 0
        torch.testing.assert_close(result, expected, atol=0, rtol=0)

    def test_incompatible_storage_is_not_silently_reinterpreted(self):
        for key, value in (("axis", 0), ("group_size", 64), ("scale_dtype", "F32")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "storage"):
                _table({**_storage(), key: value})
        table = _table(_storage())
        with self.assertRaisesRegex(ValueError, "torch.int8"):
            table._load_rows(table.weight, torch.ones(34, 64))
        with self.assertRaisesRegex(ValueError, "E8M0 bytes"):
            table._load_rows(table.scale, torch.ones(34, 2))

    def test_metadata_reaches_only_npu_table_constructor(self):
        storage = _storage()
        quant = SimpleNamespace(
            quant_description={
                "optional": {
                    "embedding_storage": {"modules": {"layers.1.engram.embed": storage}}
                }
            }
        )
        layout = SimpleNamespace(
            layer_ids=[1],
            num_embeddings=[34],
            head_dim=64,
            max_ngram_size=2,
            n_heads=1,
            column_bounds=lambda index: (0, 34),
        )
        config = SimpleNamespace(rms_norm_eps=1e-6, hidden_size=64, hc_mult=4)
        for on_npu in (False, True):
            with (
                self.subTest(npu=on_npu),
                patch.object(engram, "is_npu", return_value=on_npu),
                patch.object(engram, "EngramEmbedding") as embedding,
                patch.object(engram, "ReplicatedLinear"),
            ):
                engram.Engram(config, 1, layout, quant_config=quant)
                self.assertEqual(
                    embedding.call_args.kwargs["storage"],
                    storage if on_npu else None,
                )


if __name__ == "__main__":
    unittest.main()
