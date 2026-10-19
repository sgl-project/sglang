"""Unit tests for srt/model_loader/weight_utils.py shard-index consistency."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    initialize_dummy_weights,
    maybe_add_mtp_safetensors,
    maybe_remap_kv_scale_name,
    pad_loaded_weight,
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


class TestMaybeRemapKvScaleName(CustomTestCase):
    def test_already_canonical_scale_name_is_preserved(self):
        params_dict = {
            "model.layers.0.self_attn.attn.k_scale": torch.ones(1),
            "model.layers.0.self_attn.attn.v_scale": torch.ones(1),
        }
        self.assertEqual(
            maybe_remap_kv_scale_name(
                "model.layers.0.self_attn.attn.k_scale", params_dict
            ),
            "model.layers.0.self_attn.attn.k_scale",
        )
        self.assertEqual(
            maybe_remap_kv_scale_name(
                "model.layers.0.self_attn.attn.v_scale", params_dict
            ),
            "model.layers.0.self_attn.attn.v_scale",
        )

    def test_quark_prob_output_scale_and_missing_scale_handling(self):
        params_dict = {
            "model.layers.0.self_attn.attn.k_scale": torch.ones(1),
            "model.layers.0.self_attn.attn.v_scale": torch.ones(1),
            "model.layers.0.self_attn.attn.prob_scale": torch.ones(1),
        }
        self.assertEqual(
            maybe_remap_kv_scale_name(
                "model.layers.0.self_attn.prob_output_scale", params_dict
            ),
            "model.layers.0.self_attn.attn.prob_scale",
        )
        # When q_scale is not registered in params_dict, return None instead of
        # an unmapped name that would raise KeyError in load_weights.
        self.assertIsNone(
            maybe_remap_kv_scale_name(
                "model.layers.0.self_attn.q_proj.output_scale", params_dict
            )
        )


class TestPadLoadedWeight(CustomTestCase):
    def test_pad_loaded_weight_preserves_device_and_exact_integer_split(self):
        # 1 / 49 * 49 in IEEE-754 float64 is 0.9999999999999999, which int(...)
        # truncates to 0 unless integer division is used.
        output_sizes = [2, 96]  # sum = 98 = 2 * 49
        raw_weight = torch.arange(49 * 4, dtype=torch.bfloat16).reshape(49, 4)
        padded = pad_loaded_weight(raw_weight, output_dim=0, output_sizes=output_sizes)
        self.assertEqual(padded.shape, (98, 4))
        self.assertEqual(padded.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(padded[:1], raw_weight[:1]))
        self.assertTrue(torch.equal(padded[1:2], torch.zeros(1, 4, dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(padded[2:50], raw_weight[1:]))
        self.assertTrue(torch.equal(padded[50:], torch.zeros(48, 4, dtype=torch.bfloat16)))

        # Verify non-CPU device preservation using meta tensor.
        meta_weight = torch.empty((6, 4), dtype=torch.float16, device="meta")
        meta_padded = pad_loaded_weight(
            meta_weight, output_dim=0, output_sizes=[4, 4, 4]
        )
        self.assertEqual(meta_padded.device.type, "meta")
        self.assertEqual(meta_padded.dtype, torch.float16)
        self.assertEqual(meta_padded.shape, (12, 4))


if __name__ == "__main__":
    unittest.main()
