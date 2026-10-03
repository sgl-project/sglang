"""CPU admission tests for immutable local checkpoint headers."""

import ast
import fnmatch
import importlib.util
import json
import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_root = Path(__file__).resolve().parents[4] / "python/sglang/srt"
_spec = importlib.util.spec_from_file_location(
    "gpu_delta_checkpoint_under_test", _root / "weight_sync/gpu_delta_checkpoint.py"
)
checkpoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checkpoint)


class DefaultModelLoader:
    pass


class ModelOptModelLoader(DefaultModelLoader):
    pass


def selection_helpers():
    # Exercise the exact production selection rules without importing unrelated
    # quantization backends through weight_utils in this CPU-only test.
    names = {"filter_duplicate_safetensors_files", "maybe_add_mtp_safetensors"}
    tree = ast.parse((_root / "model_loader/weight_utils.py").read_text())
    module = types.ModuleType("sglang.srt.model_loader.weight_utils")
    module.__dict__.update(
        os=os,
        json=json,
        fnmatch=fnmatch,
        List=list,
        Optional=Optional,
        logger=logging.getLogger(__name__),
    )
    exec(  # noqa: S102 - exact local source, isolated from CUDA-only imports.
        compile(
            ast.Module(
                body=[node for node in tree.body if getattr(node, "name", "") in names],
                type_ignores=[],
            ),
            str(_root / "model_loader/weight_utils.py"),
            "exec",
        ),
        module.__dict__,
    )
    return module


class TestCanonicalCheckpointHeaders(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.runner = types.SimpleNamespace(
            model=types.SimpleNamespace(
                quant_config=types.SimpleNamespace(
                    is_checkpoint_nvfp4_serialized=True, is_nvfp4_online=False
                )
            ),
            model_config=types.SimpleNamespace(
                model_path=str(self.folder),
                hf_config=types.SimpleNamespace(architectures=["Glm4MoeForCausalLM"]),
                _is_already_quantized=lambda: True,
            ),
            load_config=types.SimpleNamespace(
                load_format="safetensors",
                draft_model_idx=None,
                decryption_key_file=None,
            ),
            loader=DefaultModelLoader(),
            is_draft_worker=False,
        )
        loader = types.ModuleType("sglang.srt.model_loader.loader")
        loader.DefaultModelLoader = DefaultModelLoader
        loader.ModelOptModelLoader = ModelOptModelLoader
        runai = types.ModuleType("sglang.srt.utils.runai_utils")
        runai.is_runai_obj_uri = lambda path: False
        runai.list_safetensors = lambda path: self.fail("unexpected remote source")
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    loader.__name__: loader,
                    "sglang.srt.model_loader.weight_utils": selection_helpers(),
                    runai.__name__: runai,
                },
            )
        )

    def save(self, name, tensors):
        path = self.folder / name
        save_file(tensors, str(path))
        return path

    def index(self, weight_map):
        (self.folder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )

    def read(self):
        return checkpoint.read_canonical_checkpoint_inventory(self.runner)

    def test_reads_source_metadata_and_preserves_alias_and_skipped_names(self):
        tensors = {
            "model.layers.0.self_attn.q_a_proj.weight": torch.zeros(
                2, 3, dtype=torch.bfloat16
            ),
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": torch.zeros(
                4, 3, dtype=torch.float16
            ),
            "model.layers.0.self_attn.indexer.k_norm.weight": torch.zeros(
                3, dtype=torch.bfloat16
            ),
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(
                2, 3, dtype=torch.uint8
            ),
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale": torch.zeros(
                2, 1, dtype=torch.float8_e4m3fn
            ),
            "model.layers.0.mlp.experts.0.gate_proj.input_scale": torch.zeros(
                (), dtype=torch.float32
            ),
            "model.layers.9.mtp.weight": torch.zeros(2, dtype=torch.bfloat16),
            "model.layers.0.self_attn.rotary_emb.inv_freq": torch.zeros(2),
        }
        self.save("model.safetensors", tensors)

        class MetadataOnly:
            def __init__(self, *args, **kwargs):
                self.source = safe_open(*args, **kwargs)

            def __enter__(self):
                self.source.__enter__()
                return self

            def __exit__(self, *args):
                return self.source.__exit__(*args)

            def keys(self):
                return self.source.keys()

            def get_slice(self, name):
                return self.source.get_slice(name)

            def get_tensor(self, name):
                raise AssertionError("metadata admission must not materialize tensors")

        with patch.object(checkpoint, "safe_open", MetadataOnly):
            inventory = self.read()
        self.assertEqual(set(inventory), set(tensors))
        for name, tensor in tensors.items():
            self.assertEqual(inventory[name]["shape"], list(tensor.shape))
        self.assertEqual(
            inventory["model.layers.0.self_attn.indexer.k_norm.weight"]["dtype"], "BF16"
        )
        self.assertEqual(
            inventory["model.layers.0.mlp.experts.0.gate_proj.weight_scale"]["dtype"],
            "F8_E4M3",
        )
        self.assertEqual(vars(self.runner.model).keys(), {"quant_config"})

    def test_index_selects_shards_and_excludes_duplicate_consolidated_file(self):
        self.save("shard.safetensors", {"weight": torch.zeros(2, 3)})
        self.save(
            "consolidated.safetensors",
            {"weight": torch.zeros(7), "unused": torch.zeros(1)},
        )
        self.index({"weight": "shard.safetensors"})
        self.assertEqual(self.read(), {"weight": {"shape": [2, 3], "dtype": "F32"}})

    def test_unindexed_bundled_mtp_follows_existing_loader_rule(self):
        self.save("shard.safetensors", {"target": torch.zeros(1)})
        self.save("mtp.safetensors", {"model.layers.9.mtp.weight": torch.zeros(2)})
        self.index({"target": "shard.safetensors"})
        self.assertEqual(set(self.read()), {"target"})
        self.runner.model_config.hf_config.num_nextn_predict_layers = 1
        self.assertEqual(set(self.read()), {"target", "model.layers.9.mtp.weight"})

    def test_missing_indexed_shard_and_ambiguous_sources_reject(self):
        self.index({"weight": "missing.safetensors"})
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.read()
        (self.folder / "model.safetensors.index.json").unlink()
        for name in ("first.safetensors", "second.safetensors"):
            self.save(name, {"weight": torch.zeros(1)})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.read()

    def test_empty_and_truncated_checkpoints_reject(self):
        with self.assertRaisesRegex(ValueError, "no selected"):
            self.read()
        path = self.save("model.safetensors", {})
        with self.assertRaisesRegex(ValueError, "empty"):
            self.read()
        path.write_bytes(b"truncated")
        with self.assertRaises(SafetensorError):
            self.read()

    def test_accepts_standard_formats_and_prequantized_modelopt_delegation(self):
        self.save("model.safetensors", {"weight": torch.zeros(1)})
        for load_format in ("auto", "safetensors", "fastsafetensors"):
            with self.subTest(load_format=load_format):
                self.runner.load_config.load_format = load_format
                self.assertEqual(set(self.read()), {"weight"})
        self.runner.loader = ModelOptModelLoader()
        self.assertEqual(set(self.read()), {"weight"})
        self.runner.model_config._is_already_quantized = lambda: False
        with self.assertRaisesRegex(ValueError, "conversion"):
            self.read()

    def test_rejects_custom_or_transformed_source_contracts_before_header_read(self):
        class CustomLoader(DefaultModelLoader):
            pass

        cases = [
            (self.runner, "loader", CustomLoader(), "standard local checkpoint loader"),
            (self.runner, "is_draft_worker", True, "target model"),
            (self.runner.load_config, "load_format", "pt", "safetensors load format"),
            (
                self.runner.load_config,
                "load_format",
                "presharded",
                "safetensors load format",
            ),
            (self.runner.load_config, "draft_model_idx", 0, "target model"),
            (self.runner.load_config, "decryption_key_file", "key", "encrypted"),
            (
                self.runner.model,
                "secondary_weights",
                [object()],
                "secondary or remapped",
            ),
            (
                self.runner.model,
                "allow_patterns_overrides",
                ["subdir/*.safetensors"],
                "secondary or remapped",
            ),
            (
                self.runner.model.quant_config,
                "is_checkpoint_nvfp4_serialized",
                False,
                "serialized NVFP4",
            ),
            (
                self.runner.model.quant_config,
                "is_nvfp4_online",
                True,
                "serialized NVFP4",
            ),
            (
                self.runner.model_config,
                "model_path",
                "remote/model",
                "immutable local checkpoint",
            ),
        ]
        for obj, key, value, message in cases:
            with self.subTest(key=key, value=value), patch.object(
                obj, key, value, create=True
            ), patch.object(
                checkpoint,
                "safe_open",
                side_effect=AssertionError("opened unsupported source"),
            ), self.assertRaisesRegex(
                ValueError, message
            ):
                self.read()


if __name__ == "__main__":
    unittest.main()
