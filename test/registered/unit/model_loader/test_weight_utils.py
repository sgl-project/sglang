"""Unit tests for srt/model_loader/weight_utils.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from pydantic import ValidationError

from sglang.srt.model_loader.weight_utils import (
    KVCacheQuantSchema,
    QuantParamSchema,
    _check_index_files_exist,
    _shared_pointers,
    convert_pyslice_to_tensor,
    default_weight_loader,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    get_actual_shard_size,
    get_lock,
    maybe_remap_kv_scale_name,
    narrow_padded_param_and_loaded_weight,
    replace_prefix,
    replace_substrings,
    reset_param_data_if_needed,
)
from sglang.test.test_utils import CustomTestCase


# ---------------------------------------------------------------------------
# get_lock
# ---------------------------------------------------------------------------
class TestGetLock(CustomTestCase):
    def test_lock_created_in_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = get_lock("org/model-name", cache_dir=tmpdir)
            self.assertIn(tmpdir, lock.lock_file)
            self.assertTrue(lock.lock_file.endswith(".lock"))

    def test_lock_name_contains_hash(self):
        lock = get_lock("org/model-name")
        basename = os.path.basename(lock.lock_file)
        # The lock file should contain the hashed model name
        self.assertIn("org-model-name", basename)

    def test_lock_suffix_appended(self):
        lock = get_lock("org/model", suffix="-download")
        self.assertTrue(lock.lock_file.endswith("-download.lock"))

    def test_slash_replaced_with_dash(self):
        lock = get_lock("org/model/sub")
        basename = os.path.basename(lock.lock_file)
        self.assertIn("org-model-sub", basename)

    def test_lock_is_acquirable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = get_lock("test/model", cache_dir=tmpdir)
            with lock:
                self.assertTrue(lock.is_locked)


# ---------------------------------------------------------------------------
# _shared_pointers
# ---------------------------------------------------------------------------
class TestSharedPointers(CustomTestCase):
    def test_no_sharing(self):
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        result = _shared_pointers({"a": t1, "b": t2})
        self.assertEqual(result, [])

    def test_shared_storage(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        # Both names share the same storage
        tensors = {"a": t, "b": t}
        result = _shared_pointers(tensors)
        self.assertEqual(len(result), 1)
        self.assertIn("a", result[0])
        self.assertIn("b", result[0])

    def test_empty_dict(self):
        result = _shared_pointers({})
        self.assertEqual(result, [])

    def test_view_shares_storage(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # A view shares the same data pointer
        v = t.view(2, 2)
        result = _shared_pointers({"original": t, "view": v})
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# replace_prefix
# ---------------------------------------------------------------------------
class TestReplacePrefix(CustomTestCase):
    def test_simple_replacement(self):
        result = replace_prefix("encoder.layer.0.weight", {"encoder.": "model."})
        self.assertEqual(result, "model.layer.0.weight")

    def test_no_match(self):
        result = replace_prefix("decoder.layer.0.weight", {"encoder.": "model."})
        self.assertEqual(result, "decoder.layer.0.weight")

    def test_empty_mapping(self):
        result = replace_prefix("encoder.layer.0.weight", {})
        self.assertEqual(result, "encoder.layer.0.weight")

    def test_multiple_mappings_first_match(self):
        mapping = {"encoder.": "model.", "model.": "new_model."}
        result = replace_prefix("encoder.layer.0.weight", mapping)
        # Only replaces the first matching prefix once
        self.assertEqual(result, "model.layer.0.weight")

    def test_only_replaces_first_occurrence(self):
        result = replace_prefix(
            "encoder.encoder.weight", {"encoder.": "model."}
        )
        self.assertEqual(result, "model.encoder.weight")


# ---------------------------------------------------------------------------
# replace_substrings
# ---------------------------------------------------------------------------
class TestReplaceSubstrings(CustomTestCase):
    def test_simple_replacement(self):
        result = replace_substrings("layer.attn.weight", {"attn": "attention"})
        self.assertEqual(result, "layer.attention.weight")

    def test_no_match(self):
        result = replace_substrings("layer.mlp.weight", {"attn": "attention"})
        self.assertEqual(result, "layer.mlp.weight")

    def test_multiple_occurrences(self):
        result = replace_substrings("attn.attn.weight", {"attn": "x"})
        self.assertEqual(result, "x.x.weight")

    def test_empty_mapping(self):
        result = replace_substrings("layer.weight", {})
        self.assertEqual(result, "layer.weight")

    def test_chained_replacements(self):
        mapping = {"old": "new", "abc": "xyz"}
        result = replace_substrings("old.abc.weight", mapping)
        self.assertEqual(result, "new.xyz.weight")


# ---------------------------------------------------------------------------
# _check_index_files_exist
# ---------------------------------------------------------------------------
class TestCheckIndexFilesExist(CustomTestCase):
    def test_no_index_files_returns_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory with no index files
            open(os.path.join(tmpdir, "model.safetensors"), "w").close()
            ok, msg = _check_index_files_exist(tmpdir)
            self.assertTrue(ok)
            self.assertIsNone(msg)

    def test_complete_index_returns_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the weight files
            for fn in ["model-00001.safetensors", "model-00002.safetensors"]:
                open(os.path.join(tmpdir, fn), "w").close()
            # Create the index
            index = {
                "weight_map": {
                    "model.weight1": "model-00001.safetensors",
                    "model.weight2": "model-00002.safetensors",
                }
            }
            with open(
                os.path.join(tmpdir, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump(index, f)
            ok, msg = _check_index_files_exist(tmpdir)
            self.assertTrue(ok)
            self.assertIsNone(msg)

    def test_missing_file_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only create one of two expected files
            open(os.path.join(tmpdir, "model-00001.safetensors"), "w").close()
            index = {
                "weight_map": {
                    "model.weight1": "model-00001.safetensors",
                    "model.weight2": "model-00002.safetensors",
                }
            }
            with open(
                os.path.join(tmpdir, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump(index, f)
            ok, msg = _check_index_files_exist(tmpdir)
            self.assertFalse(ok)
            self.assertIn("Missing", msg)
            self.assertIn("model-00002.safetensors", msg)

    def test_empty_weight_map_returns_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = {"weight_map": {}}
            with open(
                os.path.join(tmpdir, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump(index, f)
            ok, msg = _check_index_files_exist(tmpdir)
            self.assertTrue(ok)

    def test_malformed_json_is_handled(self):
        """Malformed index JSON should not block loading.

        The implementation catches *all* exceptions (including JSONDecodeError),
        logs a warning, and continues — ultimately returning ``(True, None)``.
        This is intentional: the function only checks whether referenced weight
        files are missing; it does not validate index-file syntax.  If the
        index cannot be parsed, the safest behavior is to let loading proceed
        (the loader itself will surface a clear error if the file is truly
        unusable).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, "model.safetensors.index.json"), "w"
            ) as f:
                f.write("not valid json{{{")
            ok, msg = _check_index_files_exist(tmpdir)
            # Should not crash; logs warning and returns True
            self.assertTrue(ok)


# ---------------------------------------------------------------------------
# filter_duplicate_safetensors_files
# ---------------------------------------------------------------------------
class TestFilterDuplicateSafetensorsFiles(CustomTestCase):
    def test_no_index_file_returns_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                os.path.join(tmpdir, "shard1.safetensors"),
                os.path.join(tmpdir, "shard2.safetensors"),
            ]
            result = filter_duplicate_safetensors_files(
                files, tmpdir, "model.safetensors.index.json"
            )
            self.assertEqual(result, files)

    def test_consolidated_vs_model_safetensors(self):
        """Special case: consolidated.safetensors + model.safetensors => keep model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            consolidated = os.path.join(tmpdir, "consolidated.safetensors")
            model = os.path.join(tmpdir, "model.safetensors")
            for f in [consolidated, model]:
                open(f, "w").close()
            result = filter_duplicate_safetensors_files(
                [consolidated, model], tmpdir, "model.safetensors.index.json"
            )
            self.assertEqual(result, [model])

    def test_with_index_file_filters_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard1 = os.path.join(tmpdir, "shard1.safetensors")
            shard2 = os.path.join(tmpdir, "shard2.safetensors")
            extra = os.path.join(tmpdir, "extra.safetensors")
            for f in [shard1, shard2, extra]:
                open(f, "w").close()
            index = {
                "weight_map": {
                    "w1": "shard1.safetensors",
                    "w2": "shard2.safetensors",
                }
            }
            with open(
                os.path.join(tmpdir, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump(index, f)
            result = filter_duplicate_safetensors_files(
                [shard1, shard2, extra], tmpdir, "model.safetensors.index.json"
            )
            self.assertIn(shard1, result)
            self.assertIn(shard2, result)
            self.assertNotIn(extra, result)

    def test_three_files_no_index_returns_all(self):
        """Three files without index => no special casing, return all."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [os.path.join(tmpdir, f"shard{i}.safetensors") for i in range(3)]
            result = filter_duplicate_safetensors_files(
                files, tmpdir, "model.safetensors.index.json"
            )
            self.assertEqual(result, files)


# ---------------------------------------------------------------------------
# filter_files_not_needed_for_inference
# ---------------------------------------------------------------------------
class TestFilterFilesNotNeededForInference(CustomTestCase):
    def test_removes_training_artifacts(self):
        files = [
            "/path/model.safetensors",
            "/path/training_args.bin",
            "/path/optimizer.bin",
            "/path/optimizer.pt",
            "/path/scheduler.pt",
            "/path/scaler.pt",
        ]
        result = filter_files_not_needed_for_inference(files)
        self.assertEqual(result, ["/path/model.safetensors"])

    def test_keeps_model_files(self):
        files = ["/path/model-00001.safetensors", "/path/model-00002.safetensors"]
        result = filter_files_not_needed_for_inference(files)
        self.assertEqual(result, files)

    def test_empty_list(self):
        self.assertEqual(filter_files_not_needed_for_inference([]), [])


# ---------------------------------------------------------------------------
# convert_pyslice_to_tensor
# ---------------------------------------------------------------------------
class TestConvertPysliceToTensor(CustomTestCase):
    def test_tensor_passthrough(self):
        t = torch.tensor([1.0, 2.0])
        result = convert_pyslice_to_tensor(t)
        self.assertIs(result, t)

    def test_sliceable_object_converted(self):
        """Non-tensor objects with __getitem__ get sliced."""

        class FakeSlice:
            def __getitem__(self, key):
                return torch.tensor([3.0, 4.0])

        result = convert_pyslice_to_tensor(FakeSlice())
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.tolist(), [3.0, 4.0])


# ---------------------------------------------------------------------------
# default_weight_loader
# ---------------------------------------------------------------------------
class TestDefaultWeightLoader(CustomTestCase):
    def test_matching_shapes(self):
        param = torch.zeros(3, 4)
        loaded = torch.ones(3, 4)
        default_weight_loader(param, loaded)
        self.assertTrue(torch.equal(param, loaded))

    def test_scalar_broadcast(self):
        param = torch.zeros(1)
        loaded = torch.tensor([5.0])
        default_weight_loader(param, loaded)
        self.assertEqual(param.item(), 5.0)

    def test_mismatched_shapes_raises(self):
        param = torch.zeros(3, 4)
        loaded = torch.ones(2, 4)
        with self.assertRaises(AssertionError):
            default_weight_loader(param, loaded)

    def test_scalar_to_scalar(self):
        param = torch.tensor(0.0)
        loaded = torch.tensor(42.0)
        default_weight_loader(param, loaded)
        self.assertEqual(param.item(), 42.0)


# ---------------------------------------------------------------------------
# maybe_remap_kv_scale_name
# ---------------------------------------------------------------------------
class TestMaybeRemapKvScaleName(CustomTestCase):
    def test_kv_scale_remapped(self):
        params = {"model.layers.0.attn.k_scale": None}
        result = maybe_remap_kv_scale_name("model.layers.0.kv_scale", params)
        self.assertEqual(result, "model.layers.0.attn.k_scale")

    def test_kv_scale_not_in_params_returns_none(self):
        result = maybe_remap_kv_scale_name("model.layers.0.kv_scale", {})
        self.assertIsNone(result)

    def test_k_scale_remapped(self):
        params = {"model.layers.0.attn.k_scale": None}
        result = maybe_remap_kv_scale_name("model.layers.0.k_scale", params)
        self.assertEqual(result, "model.layers.0.attn.k_scale")

    def test_v_scale_remapped(self):
        params = {"model.layers.0.attn.v_scale": None}
        result = maybe_remap_kv_scale_name("model.layers.0.v_scale", params)
        self.assertEqual(result, "model.layers.0.attn.v_scale")

    def test_k_scale_not_in_params_returns_none(self):
        result = maybe_remap_kv_scale_name("model.layers.0.k_scale", {})
        self.assertIsNone(result)

    def test_modelopt_k_proj_scale_remapped(self):
        params = {"model.layers.0.self_attn.attn.k_scale": None}
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_proj.k_scale", params
        )
        self.assertEqual(result, "model.layers.0.self_attn.attn.k_scale")

    def test_modelopt_v_proj_scale_remapped(self):
        params = {"model.layers.0.self_attn.attn.v_scale": None}
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.v_proj.v_scale", params
        )
        self.assertEqual(result, "model.layers.0.self_attn.attn.v_scale")

    def test_mixer_prefix_scale_remapped(self):
        params = {"model.layers.0.mixer.attn.k_scale": None}
        result = maybe_remap_kv_scale_name(
            "model.layers.0.mixer.k_proj.k_scale", params
        )
        self.assertEqual(result, "model.layers.0.mixer.attn.k_scale")

    def test_quark_q_proj_output_scale(self):
        result = maybe_remap_kv_scale_name(
            "model.layers.0.q_proj.output_scale", {}
        )
        self.assertEqual(result, "model.layers.0.attn.q_scale")

    def test_quark_k_proj_output_scale(self):
        result = maybe_remap_kv_scale_name(
            "model.layers.0.k_proj.output_scale", {}
        )
        self.assertEqual(result, "model.layers.0.attn.k_scale")

    def test_quark_v_proj_output_scale(self):
        result = maybe_remap_kv_scale_name(
            "model.layers.0.v_proj.output_scale", {}
        )
        self.assertEqual(result, "model.layers.0.attn.v_scale")

    def test_unrelated_name_returned_as_is(self):
        result = maybe_remap_kv_scale_name("model.layers.0.weight", {})
        self.assertEqual(result, "model.layers.0.weight")

    def test_prob_output_scale_remapped(self):
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.prob_output_scale", {}
        )
        self.assertEqual(result, "model.layers.0.attn.prob_scale")


# ---------------------------------------------------------------------------
# get_actual_shard_size
# ---------------------------------------------------------------------------
class TestGetActualShardSize(CustomTestCase):
    def test_normal_shard(self):
        self.assertEqual(get_actual_shard_size(100, 0, 400), 100)

    def test_shard_exceeds_weight(self):
        self.assertEqual(get_actual_shard_size(100, 350, 400), 50)

    def test_weight_end_lt_start_returns_zero(self):
        self.assertEqual(get_actual_shard_size(100, 500, 400), 0)

    def test_exact_boundary(self):
        self.assertEqual(get_actual_shard_size(100, 300, 400), 100)

    def test_shard_size_zero(self):
        self.assertEqual(get_actual_shard_size(0, 0, 100), 0)


# ---------------------------------------------------------------------------
# reset_param_data_if_needed
# ---------------------------------------------------------------------------
class TestResetParamDataIfNeeded(CustomTestCase):
    def test_zeros_range(self):
        param = torch.ones(10)
        reset_param_data_if_needed(param, 0, 3, 4)
        self.assertTrue(torch.equal(param[3:7], torch.zeros(4)))
        self.assertTrue(torch.equal(param[0:3], torch.ones(3)))

    def test_length_zero_noop(self):
        param = torch.ones(10)
        reset_param_data_if_needed(param, 0, 5, 0)
        self.assertTrue(torch.equal(param, torch.ones(10)))

    def test_negative_length_raises(self):
        param = torch.ones(10)
        with self.assertRaises(AssertionError):
            reset_param_data_if_needed(param, 0, 5, -1)


# ---------------------------------------------------------------------------
# narrow_padded_param_and_loaded_weight
# ---------------------------------------------------------------------------
class TestNarrowPaddedParamAndLoadedWeight(CustomTestCase):
    def test_normal_narrow(self):
        param = torch.zeros(8)
        loaded = torch.arange(8, dtype=torch.float32)
        p, l = narrow_padded_param_and_loaded_weight(
            param, loaded, param_data_start=0, weight_start=0, dim=0, shard_size=4
        )
        self.assertEqual(p.shape[0], 4)
        self.assertEqual(l.shape[0], 4)

    def test_partial_shard_at_end(self):
        param = torch.zeros(8)
        loaded = torch.arange(6, dtype=torch.float32)
        p, l = narrow_padded_param_and_loaded_weight(
            param, loaded, param_data_start=0, weight_start=4, dim=0, shard_size=4
        )
        # Only 2 elements available from index 4 to 6
        self.assertEqual(p.shape[0], 2)
        self.assertEqual(l.shape[0], 2)

    def test_with_offset(self):
        param = torch.zeros(16)
        loaded = torch.arange(16, dtype=torch.float32)
        p, l = narrow_padded_param_and_loaded_weight(
            param, loaded, param_data_start=4, weight_start=4, dim=0, shard_size=4
        )
        self.assertEqual(p.shape[0], 4)
        self.assertEqual(l.shape[0], 4)


# ---------------------------------------------------------------------------
# KVCacheQuantSchema
# ---------------------------------------------------------------------------
class TestKVCacheQuantSchema(CustomTestCase):
    def test_valid_fp8_schema(self):
        schema = KVCacheQuantSchema(
            dtype="float8_e4m3fn",
            scaling_factor={0: {0: 1.0, 1: 1.0}},
        )
        self.assertEqual(schema.dtype, "float8_e4m3fn")

    def test_wrong_dtype_raises(self):
        with self.assertRaises(ValidationError):
            KVCacheQuantSchema(
                dtype="float16",
                scaling_factor={0: {0: 1.0}},
            )

    def test_with_context_validates_tp_size(self):
        context = {"tp_size": 2, "num_hidden_layers": 1, "tp_rank": 0}
        # Only 1 TP rank provided but tp_size=2
        with self.assertRaises(ValidationError):
            KVCacheQuantSchema.model_validate(
                {"dtype": "float8_e4m3fn", "scaling_factor": {0: {0: 1.0}}},
                context=context,
            )

    def test_with_context_valid(self):
        context = {"tp_size": 1, "num_hidden_layers": 2, "tp_rank": 0}
        schema = KVCacheQuantSchema.model_validate(
            {
                "dtype": "float8_e4m3fn",
                "scaling_factor": {0: {0: 1.0, 1: 1.5}},
            },
            context=context,
        )
        self.assertEqual(schema.scaling_factor[0][1], 1.5)


# ---------------------------------------------------------------------------
# QuantParamSchema
# ---------------------------------------------------------------------------
class TestQuantParamSchema(CustomTestCase):
    def test_valid_schema(self):
        schema = QuantParamSchema(
            model_type="llama",
            kv_cache=KVCacheQuantSchema(
                dtype="float8_e4m3fn",
                scaling_factor={0: {0: 1.0}},
            ),
        )
        self.assertEqual(schema.model_type, "llama")

    def test_model_type_mismatch_with_context(self):
        context = {"model_type": "gpt2"}
        with self.assertRaises(ValidationError):
            QuantParamSchema.model_validate(
                {
                    "model_type": "llama",
                    "kv_cache": {
                        "dtype": "float8_e4m3fn",
                        "scaling_factor": {0: {0: 1.0}},
                    },
                },
                context=context,
            )

    def test_none_model_type_accepted(self):
        schema = QuantParamSchema(
            model_type=None,
            kv_cache=KVCacheQuantSchema(
                dtype="float8_e4m3fn",
                scaling_factor={0: {0: 1.0}},
            ),
        )
        self.assertIsNone(schema.model_type)


if __name__ == "__main__":
    unittest.main()
