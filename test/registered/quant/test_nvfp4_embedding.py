#!/usr/bin/env python3

import unittest

import torch

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4EmbeddingMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    find_embedding_module,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GROUP_SIZE = 16

# Written out independently of the implementation: the E2M1 code points in
# magnitude order, so index == the 3-bit magnitude code.
_REFERENCE_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def reference_dequant(
    packed: torch.Tensor, block_scale: torch.Tensor, global_scale: float
) -> torch.Tensor:
    """Comparison oracle. Kept as a plain per-element loop on purpose: a
    vectorized rewrite would mirror the code under test."""
    rows, half = packed.shape
    hidden = half * 2
    out = torch.zeros(rows, hidden, dtype=torch.float32)
    for r in range(rows):
        for c in range(hidden):
            byte = int(packed[r, c // 2])
            code = (byte & 0x0F) if c % 2 == 0 else (byte >> 4)
            magnitude = _REFERENCE_E2M1[code & 0x7]
            value = -magnitude if code & 0x8 else magnitude
            scale = float(block_scale[r, c // GROUP_SIZE]) * global_scale
            out[r, c] = value * scale
    return out


def build_layer(method, vocab_size: int, hidden_size: int) -> torch.nn.Module:
    """Materialize through create_weights, then fill as a checkpoint would."""
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=hidden_size,
        output_partition_sizes=[vocab_size],
        input_size=hidden_size,
        output_size=vocab_size,
        params_dtype=torch.bfloat16,
    )

    generator = torch.Generator().manual_seed(0)
    layer.weight.data.copy_(
        torch.randint(
            0,
            256,
            (vocab_size, hidden_size // 2),
            dtype=torch.uint8,
            generator=generator,
        )
    )
    # Keep the block scales in a range e4m3 represents exactly.
    layer.weight_scale.data.copy_(
        torch.randint(
            1,
            8,
            (vocab_size, hidden_size // GROUP_SIZE),
            dtype=torch.int32,
            generator=generator,
        ).to(torch.float8_e4m3fn)
    )
    layer.weight_scale_2.data.fill_(0.125)
    return layer


class TestNvFp4Embedding(CustomTestCase):
    def setUp(self):
        self.method = ModelOptNvFp4EmbeddingMethod(
            ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True, group_size=GROUP_SIZE
            )
        )

    def test_matches_reference_dequant(self):
        vocab_size, hidden_size = 24, 64
        layer = build_layer(self.method, vocab_size, hidden_size)
        self.assertEqual(tuple(layer.weight.shape), (vocab_size, hidden_size // 2))
        self.assertEqual(
            tuple(layer.weight_scale.shape), (vocab_size, hidden_size // GROUP_SIZE)
        )

        ids = torch.tensor([[0, 5, 5], [23, 11, 0]])
        got = self.method.embedding(layer, ids)
        expected = reference_dequant(
            layer.weight[ids.reshape(-1)],
            layer.weight_scale[ids.reshape(-1)].float(),
            float(layer.weight_scale_2),
        )

        self.assertEqual(tuple(got.shape), (2, 3, hidden_size))
        self.assertEqual(got.dtype, torch.bfloat16)
        torch.testing.assert_close(
            got.reshape(-1, hidden_size).float(),
            expected.to(torch.bfloat16).float(),
            rtol=0,
            atol=0,
        )

    def test_hidden_size_must_divide_group_size(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            self.method.create_weights(
                torch.nn.Module(),
                input_size_per_partition=40,
                output_partition_sizes=[8],
                input_size=40,
                output_size=8,
                params_dtype=torch.bfloat16,
            )


VOCAB_SIZE, HIDDEN_SIZE = 128, 64
EMBED_PREFIX = "model.embed_tokens"


def mixed_precision_config(embedding_quant_algo: str) -> ModelOptMixedPrecisionConfig:
    """A MIXED_PRECISION config shaped like the Tool-Suite recipes: the token
    embedding is NVFP4, and the draft branch is excluded from quantization."""
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "ignore": ["mtp*", "mtp.layers.0*"],
            "packed_modules_mapping": {},
            "quantized_layers": {
                EMBED_PREFIX: {
                    "quant_algo": embedding_quant_algo,
                    "group_size": GROUP_SIZE,
                },
                "lm_head": {"quant_algo": "FP8"},
            },
        }
    )


def build_embedding(quant_config, prefix: str) -> VocabParallelEmbedding:
    """A real VocabParallelEmbedding, off the TP path (no process group here)."""
    return VocabParallelEmbedding(
        VOCAB_SIZE,
        HIDDEN_SIZE,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix=prefix,
        enable_tp=False,
    )


def fill_nvfp4_table(layer: VocabParallelEmbedding) -> None:
    generator = torch.Generator().manual_seed(1)
    layer.weight.data.copy_(
        torch.randint(
            0,
            256,
            layer.weight.shape,
            dtype=torch.uint8,
            generator=generator,
        )
    )
    layer.weight_scale.data.copy_(
        torch.randint(
            1, 8, layer.weight_scale.shape, dtype=torch.int32, generator=generator
        ).to(torch.float8_e4m3fn)
    )
    layer.weight_scale_2.data.fill_(0.125)


def gather(layer: VocabParallelEmbedding, ids: torch.Tensor) -> torch.Tensor:
    """forward() needs the TP group; the gather itself does not."""
    return layer.quant_method.embedding(layer, ids.long())


class TestQuantizedEmbeddingSharing(CustomTestCase):
    """The speculative handover moves `weight`; a quantized table is a module."""

    def _target_and_draft(self):
        config = mixed_precision_config("NVFP4")
        target = build_embedding(config, EMBED_PREFIX)
        self.assertIsInstance(target.quant_method, ModelOptNvFp4EmbeddingMethod)
        fill_nvfp4_table(target)
        # The draft branch is excluded from quantization by the checkpoint, so
        # it builds a dense bf16 table of the full hidden size.
        draft = build_embedding(config, "mtp.embed_tokens")
        self.assertIsInstance(draft.quant_method, UnquantizedEmbeddingMethod)
        return target, draft

    def test_tensor_only_handover_returns_packed_rows(self):
        """The defect this fixes, stated as a test."""
        target, draft = self._target_and_draft()
        ids = torch.tensor([0, 5, 63, 127])

        del draft.weight
        draft.weight = target.weight  # what set_embed_and_head() does

        got = gather(draft, ids)
        self.assertEqual(got.shape[-1], HIDDEN_SIZE // 2)  # packed, half width
        self.assertEqual(got.dtype, torch.uint8)

    def test_share_table_from_matches_the_target_gather(self):
        target, draft = self._target_and_draft()
        ids = torch.tensor([[0, 5, 63], [127, 5, 0]])
        expected = gather(target, ids)

        del draft.weight
        draft.weight = target.weight
        draft.share_table_from(target)

        got = gather(draft, ids)
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(got, expected))

    def test_sharing_copies_nothing(self):
        target, draft = self._target_and_draft()
        del draft.weight
        draft.weight = target.weight
        draft.share_table_from(target)

        self.assertIs(draft.quant_method, target.quant_method)
        for name, param in target.named_parameters(recurse=False):
            self.assertIs(dict(draft.named_parameters(recurse=False))[name], param)
        for name, buffer in target.named_buffers(recurse=False):
            self.assertIs(dict(draft.named_buffers(recurse=False))[name], buffer)
        # No leftover bf16 table of its own.
        self.assertEqual(draft.weight.dtype, torch.uint8)

    def test_layout_mismatch_raises(self):
        target, _ = self._target_and_draft()
        other = VocabParallelEmbedding(
            VOCAB_SIZE * 2,
            HIDDEN_SIZE,
            params_dtype=torch.bfloat16,
            enable_tp=False,
        )
        with self.assertRaisesRegex(ValueError, "vocab-parallel"):
            other.share_table_from(target)

    def test_find_embedding_module_by_identity(self):
        target, draft = self._target_and_draft()
        model = torch.nn.Module()
        model.embed_tokens = target
        model.other = draft

        self.assertIs(find_embedding_module(model, target.weight), target)
        self.assertIsNone(
            find_embedding_module(model, torch.zeros(VOCAB_SIZE, HIDDEN_SIZE))
        )

    def test_unquantized_target_is_untouched(self):
        """The gate: an unquantized embedding keeps the plain tensor handover."""
        from sglang.srt.speculative.spec_utils import share_target_embedding

        target = build_embedding(None, EMBED_PREFIX)
        draft = build_embedding(None, "mtp.embed_tokens")
        self.assertIsInstance(target.quant_method, UnquantizedEmbeddingMethod)

        del draft.weight
        draft.weight = target.weight
        draft_method = draft.quant_method

        target_model, draft_model = torch.nn.Module(), torch.nn.Module()
        target_model.embed_tokens = target
        draft_model.embed_tokens = draft
        share_target_embedding(target_model, draft_model, target.weight)

        self.assertIs(draft.quant_method, draft_method)  # untouched


if __name__ == "__main__":
    unittest.main(verbosity=2)
