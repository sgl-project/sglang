"""Unit tests for weight loading, shard indexes, and GGUF tensor decoding."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    gguf_quant_weights_iterator,
    initialize_dummy_weights,
    maybe_add_mtp_safetensors,
)
from sglang.srt.utils import runai_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

INDEX_NAME = "model.safetensors.index.json"


def _write_index(folder, weight_map):
    with open(os.path.join(folder, INDEX_NAME), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def _touch(folder, name):
    path = os.path.join(folder, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    return path


class TestFilterDuplicateSafetensorsFiles(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_shard_raises(self):
        # Index lists two shards, only one on disk (interrupted download).
        _write_index(
            self.folder,
            {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
            },
        )
        present = _touch(self.folder, "model-00001-of-00002.safetensors")

        with self.assertRaises(RuntimeError) as cm:
            filter_duplicate_safetensors_files(
                hf_weights_files=[present],
                hf_folder=self.folder,
                index_file=INDEX_NAME,
            )
        self.assertIn("model-00002-of-00002.safetensors", str(cm.exception))

    def test_complete_checkpoint_filters_non_indexed(self):
        # All indexed shards present; a non-indexed duplicate is still filtered out.
        _write_index(
            self.folder,
            {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
            },
        )
        shard1 = _touch(self.folder, "model-00001-of-00002.safetensors")
        shard2 = _touch(self.folder, "model-00002-of-00002.safetensors")
        extra = _touch(self.folder, "consolidated.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[shard1, shard2, extra],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(sorted(result), sorted([shard1, shard2]))

    def test_missing_shard_outside_allow_patterns_is_ignored(self):
        # Cosmos3-style checkpoints use one root index for multiple subfolder
        # weight sources. Loading the transformer source should not require the
        # vision encoder shard to already be present; the secondary source
        # downloads and loads it separately.
        _write_index(
            self.folder,
            {
                "llm": "transformer/diffusion_pytorch_model.safetensors",
                "vit": "vision_encoder/model.safetensors",
            },
        )
        shard = "transformer/diffusion_pytorch_model.safetensors"
        _touch(self.folder, shard)

        for folder in (self.folder, "s3://bucket/model"):
            with self.subTest(folder=folder):
                transformer = os.path.join(folder, shard)
                result = filter_duplicate_safetensors_files(
                    hf_weights_files=[transformer],
                    hf_folder=folder,
                    index_file=os.path.join(self.folder, INDEX_NAME),
                    allow_patterns=["transformer/*.safetensors"],
                )
                self.assertEqual(result, [transformer])

    def test_missing_shard_inside_allow_patterns_raises(self):
        _write_index(
            self.folder,
            {
                "llm1": "transformer/model-00001-of-00002.safetensors",
                "llm2": "transformer/model-00002-of-00002.safetensors",
                "vit": "vision_encoder/model.safetensors",
            },
        )
        shard = "transformer/model-00001-of-00002.safetensors"
        _touch(self.folder, shard)

        for folder in (self.folder, "s3://bucket/model"):
            with (
                self.subTest(folder=folder),
                self.assertRaisesRegex(
                    RuntimeError, r"model-00002-of-00002\.safetensors"
                ),
            ):
                filter_duplicate_safetensors_files(
                    hf_weights_files=[os.path.join(folder, shard)],
                    hf_folder=folder,
                    index_file=os.path.join(self.folder, INDEX_NAME),
                    allow_patterns=["transformer/*.safetensors"],
                )

    def test_local_index_allows_subset_of_existing_shards(self):
        _write_index(
            self.folder,
            {"w1": "first.safetensors", "w2": "second.safetensors"},
        )
        first = _touch(self.folder, "first.safetensors")
        _touch(self.folder, "second.safetensors")

        result = filter_duplicate_safetensors_files([first], self.folder, INDEX_NAME)

        self.assertEqual(result, [first])

    def test_single_file_model_no_index_returns_unchanged(self):
        # No index on disk (single-file / dummy / object-storage): early return.
        single = _touch(self.folder, "model.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[single],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(result, [single])


class TestMaybeAddMtpSafetensors(CustomTestCase):
    def test_remote_mtp_guards(self):
        folder = "s3://bucket/model"
        model = f"{folder}/model.safetensors"
        mtp = f"{folder}/mtp.safetensors"
        cases = (
            ("Glm4MoeForCausalLM", 1, [model, mtp], [model, mtp], False),
            ("Glm4MoeForCausalLM", 1, [model], [model], True),
            ("Glm4MoeForCausalLM", 0, [model], [model, mtp], False),
            ("LlamaForCausalLM", 1, [model], [model, mtp], False),
        )
        for arch, nextn, selected, available, may_list in cases:
            with (
                self.subTest(arch=arch, nextn=nextn, selected=selected),
                patch.object(
                    runai_utils, "list_safetensors", return_value=available
                ) as listing,
            ):
                config = SimpleNamespace(
                    architectures=[arch], num_nextn_predict_layers=nextn
                )
                self.assertEqual(
                    maybe_add_mtp_safetensors(selected, folder, INDEX_NAME, config),
                    selected,
                )
                if not may_list:
                    listing.assert_not_called()

    def test_local_unindexed_mtp(self):
        with tempfile.TemporaryDirectory() as folder:
            model = _touch(folder, "model.safetensors")
            mtp = _touch(folder, "mtp.safetensors")
            config = SimpleNamespace(
                architectures=["Glm4MoeLiteForCausalLMNextN"],
                num_nextn_predict_layers=1,
            )
            with patch.object(runai_utils, "list_safetensors") as listing:
                self.assertEqual(
                    maybe_add_mtp_safetensors([model], folder, INDEX_NAME, config),
                    [model, mtp],
                )
                listing.assert_not_called()


class TestInitializeDummyWeights(CustomTestCase):
    def test_nvfp4_dummy_scales_are_neutral(self):
        model = torch.nn.Linear(8, 4, bias=False)
        for name, shape in (
            ("w13_weight_scale_2", (4, 2)),
            ("w2_weight_scale_2", (4,)),
        ):
            model.register_parameter(
                name, torch.nn.Parameter(torch.empty(shape), requires_grad=False)
            )

        initialize_dummy_weights(model, low=0.25, high=0.75)

        self.assertTrue(torch.equal(model.w13_weight_scale_2, torch.ones(4, 2)))
        self.assertTrue(torch.equal(model.w2_weight_scale_2, torch.ones(4)))
        self.assertTrue(torch.all((model.weight >= 0.25) & (model.weight <= 0.75)))


class TestGGUFTensors(CustomTestCase):
    def test_unquantized_gguf_tensors_preserve_values_and_logical_shape(self):
        import gguf

        for dtype, quant_type in (
            (torch.bfloat16, gguf.GGMLQuantizationType.BF16),
            (torch.float16, gguf.GGMLQuantizationType.F16),
            (torch.float32, gguf.GGMLQuantizationType.F32),
        ):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                expected = torch.tensor([[1.5, -2.0], [0.25, 4.0]], dtype=dtype)
                data = (
                    expected.view(torch.uint8).numpy()
                    if dtype == torch.bfloat16
                    else expected.numpy()
                )
                path = os.path.join(directory, "weights.gguf")
                writer = gguf.GGUFWriter(path, "llama")
                writer.add_tensor("blk.0.ffn_down.weight", data, raw_dtype=quant_type)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                weights = dict(
                    gguf_quant_weights_iterator(
                        path, {"blk.0.ffn_down.weight": "model.proj.weight"}
                    )
                )
                name = (
                    "model.proj.weight"
                    if dtype == torch.float32
                    else "model.proj.qweight"
                )
                torch.testing.assert_close(weights[name], expected, rtol=0, atol=0)

                from sglang.srt.layers.quantization.gguf import (
                    apply_gguf_embedding,
                    fused_mul_mat_gguf,
                )

                inputs = torch.tensor([[2.0, -1.0]], dtype=dtype)
                output = fused_mul_mat_gguf(inputs, weights[name], quant_type)
                torch.testing.assert_close(
                    output, torch.tensor([[5.0, -3.5]], dtype=dtype), rtol=0, atol=0
                )
                token_ids = torch.tensor([1, 0, 1])
                embeddings = apply_gguf_embedding(
                    token_ids, weights[name], quant_type, hidden_size=2
                )
                torch.testing.assert_close(
                    embeddings, expected[token_ids], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
