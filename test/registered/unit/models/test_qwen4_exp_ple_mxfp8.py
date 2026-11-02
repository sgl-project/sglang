"""MXFP8 (fp8 e4m3 + E8M0 group-32 scales) PLE table of Qwen4-Exp.

Covers the table detection from the compressed-tensors target map, the
dequantizing lookup (single rank and a TP shard that masks foreign rows), and
the checkpoint loader copying `shard_N.weight` / `shard_N.weight_scale` rows
into the right local rows.
"""

import types
import unittest
from unittest import mock

import torch
import torch.nn as nn
from compressed_tensors.quantization import QuantizationType

from sglang.srt.layers import vocab_parallel_embedding as vpe
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.models.qwen4_exp import (
    MXFP8_GROUP_SIZE,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpMxfp8Embedding,
    Qwen4ExpNGramEmbedding,
    _ple_table_is_mxfp8,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PREFIX = "model.layers.1.ple.ple_embedding.ngram_embedding"
DIM = 64


def _ct_config(prefix, type_=QuantizationType.FLOAT, num_bits=8, group_size=32):
    weights = types.SimpleNamespace(
        type=type_, num_bits=num_bits, group_size=group_size
    )
    config = mock.Mock(spec=CompressedTensorsConfig)
    config.target_scheme_map = {prefix: {"weights": weights}}
    return config


def _tp_embedding(vocab, tp_rank=0, tp_size=1, dim=DIM):
    group = types.SimpleNamespace(tp_rank=tp_rank, tp_size=tp_size)
    with (
        mock.patch.object(vpe, "get_parallel", return_value=group),
        mock.patch.object(vpe, "_resolve_linear_group", return_value=None),
    ):
        return Qwen4ExpMxfp8Embedding(
            vocab,
            dim,
            params_dtype=torch.float8_e4m3fn,
            output_dtype=torch.bfloat16,
            padding_size=1,
        )


def _random_table(rows, dim=DIM):
    values = (torch.randn(rows, dim) * 4).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 135, (rows, dim // MXFP8_GROUP_SIZE), dtype=torch.uint8)
    return values, scales


def _dequant_reference(values, scales):
    grouped = values.float().reshape(values.shape[0], -1, MXFP8_GROUP_SIZE)
    factor = torch.pow(2.0, scales.float() - 127).unsqueeze(-1)
    return (grouped * factor).reshape(values.shape)


class TestPleTableIsMxfp8(CustomTestCase):
    def test_group32_fp8_target_is_mxfp8(self):
        self.assertTrue(_ple_table_is_mxfp8(_ct_config(PREFIX), PREFIX))

    def test_other_group_size_is_not(self):
        config = _ct_config(PREFIX, group_size=128)
        self.assertFalse(_ple_table_is_mxfp8(config, PREFIX))

    def test_other_bits_or_type_is_not(self):
        self.assertFalse(_ple_table_is_mxfp8(_ct_config(PREFIX, num_bits=4), PREFIX))
        self.assertFalse(
            _ple_table_is_mxfp8(_ct_config(PREFIX, type_=QuantizationType.INT), PREFIX)
        )

    def test_untargeted_prefix_is_not(self):
        self.assertFalse(_ple_table_is_mxfp8(_ct_config("other.prefix"), PREFIX))

    def test_no_or_foreign_quant_config_is_not(self):
        self.assertFalse(_ple_table_is_mxfp8(None, PREFIX))
        self.assertFalse(_ple_table_is_mxfp8(mock.Mock(), PREFIX))


class TestMxfp8Embedding(CustomTestCase):
    def test_scale_buffer_defaults_to_unit_scale(self):
        emb = _tp_embedding(8)
        self.assertEqual(emb.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(emb.weight_scale.shape), (8, DIM // MXFP8_GROUP_SIZE))
        self.assertTrue(bool((emb.weight_scale == 127).all()))

    def test_dim_must_be_a_multiple_of_the_group(self):
        with self.assertRaises(ValueError):
            _tp_embedding(8, dim=48)

    def test_lookup_matches_dequantized_reference(self):
        torch.manual_seed(0)
        emb = _tp_embedding(16)
        values, scales = _random_table(16)
        emb.weight.data.copy_(values)
        emb.weight_scale.copy_(scales)
        ids = torch.tensor([[0, 5, 15], [3, 3, 9]])
        out = emb._embed_local_shard(ids)
        self.assertEqual(out.dtype, torch.bfloat16)
        expected = _dequant_reference(values, scales)[ids].to(torch.bfloat16)
        self.assertTrue(torch.equal(out, expected))

    def test_tp_shard_zeroes_rows_owned_by_other_ranks(self):
        torch.manual_seed(1)
        emb = _tp_embedding(16, tp_rank=1, tp_size=2)
        self.assertEqual(emb.shard_indices.org_vocab_start_index, 8)
        values, scales = _random_table(8)
        emb.weight.data.copy_(values)
        emb.weight_scale.copy_(scales)
        ids = torch.tensor([1, 8, 15])
        out = emb._embed_local_shard(ids)
        reference = _dequant_reference(values, scales).to(torch.bfloat16)
        self.assertTrue(bool((out[0] == 0).all()))
        self.assertTrue(torch.equal(out[1], reference[0]))
        self.assertTrue(torch.equal(out[2], reference[7]))


class _LoaderHarness(nn.Module):
    """The slice of Qwen4ExpForConditionalGeneration that load_weights touches."""

    load_weights = Qwen4ExpForConditionalGeneration.load_weights

    def __init__(self, embedding, num_sync_shards):
        super().__init__()
        self.ple = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
        nn.Module.__init__(self.ple)
        self.ple.ngram_embedding = embedding
        self.config = types.SimpleNamespace(
            num_experts=None,
            tie_word_embeddings=False,
            encoder_only=False,
            split_ngram_parts=num_sync_shards,
        )
        self.language_model_only = False
        self.start_layer = 0
        self.end_layer = 1000
        self.pp_group = types.SimpleNamespace(is_last_rank=True)

    def named_modules(self, *args, **kwargs):
        return iter([("model.layers.1.ple.ple_embedding", self.ple)])

    def _load_qwen4_exp_ple_buffer(self, *args):
        return False


class TestMxfp8ShardLoader(CustomTestCase):
    def _load(self, harness, shard, values, scales):
        name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}"
        return harness.load_weights(
            [(f"{name}.weight", values), (f"{name}.weight_scale", scales)]
        )

    def test_shards_land_in_their_rows_with_their_scales(self):
        torch.manual_seed(2)
        vocab, num_shards = 16, 4
        values, scales = _random_table(vocab)
        harness = _LoaderHarness(_tp_embedding(vocab), num_shards)
        rows = vocab // num_shards
        loaded = set()
        # Out of order: row placement must come from the shard index.
        for shard in (2, 0, 3, 1):
            rows_slice = slice(shard * rows, (shard + 1) * rows)
            loaded |= self._load(harness, shard, values[rows_slice], scales[rows_slice])
        emb = harness.ple.ngram_embedding
        self.assertTrue(
            torch.equal(emb.weight.data.view(torch.uint8), values.view(torch.uint8))
        )
        self.assertTrue(torch.equal(emb.weight_scale, scales))
        self.assertEqual(
            loaded,
            {f"{PREFIX}.weight", f"{PREFIX}.weight_scale"},
        )

    def test_scale_shards_do_not_trigger_the_fp8_downcast_warning(self):
        values, scales = _random_table(8)
        harness = _LoaderHarness(_tp_embedding(8), 1)
        with self.assertNoLogs(level="WARNING"):
            self._load(harness, 0, values, scales)

    def test_tp_rank_keeps_only_its_rows(self):
        torch.manual_seed(3)
        vocab, num_shards = 16, 2
        values, scales = _random_table(vocab)
        harness = _LoaderHarness(_tp_embedding(vocab, tp_rank=1, tp_size=2), num_shards)
        for shard in range(num_shards):
            rows_slice = slice(shard * 8, (shard + 1) * 8)
            self._load(harness, shard, values[rows_slice], scales[rows_slice])
        emb = harness.ple.ngram_embedding
        self.assertTrue(
            torch.equal(emb.weight.data.view(torch.uint8), values[8:].view(torch.uint8))
        )
        self.assertTrue(torch.equal(emb.weight_scale, scales[8:]))

    def test_short_last_shard_is_loaded_partially(self):
        torch.manual_seed(4)
        vocab, num_shards = 10, 4  # shard rows are ceil(10 / 4) = 3, last has 1
        values, scales = _random_table(vocab)
        harness = _LoaderHarness(_tp_embedding(vocab), num_shards)
        for shard, (start, end) in enumerate([(0, 3), (3, 6), (6, 9), (9, 10)]):
            self._load(harness, shard, values[start:end], scales[start:end])
        emb = harness.ple.ngram_embedding
        self.assertTrue(
            torch.equal(
                emb.weight.data[:vocab].view(torch.uint8), values.view(torch.uint8)
            )
        )
        self.assertTrue(torch.equal(emb.weight_scale[:vocab], scales))


if __name__ == "__main__":
    unittest.main()
