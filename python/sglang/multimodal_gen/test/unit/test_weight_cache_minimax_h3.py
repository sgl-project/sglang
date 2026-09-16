# SPDX-License-Identifier: Apache-2.0
"""Native FL2VA admission, frozen partition metadata and finalized state schema."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.prepare import prepare_pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.weight_cache.adapters import common, dit_minimax_h3
from sglang.multimodal_gen.runtime.weight_cache.identity import checkpoint_identity
from sglang.weight_cache_common.mapping import validate_meta_schema
from sglang.weight_cache_common.traversal import snapshot_module


@pytest.fixture
def h3_args(tmp_path):
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

    # Root may also contain a modular Diffusers model; explicit native partition
    # selection must win without touching the checkpoint or bypassing validation.
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"})
    )
    root = tmp_path / "FL2VA"
    root.mkdir()
    index = {
        "_class_name": "MiniMaxH3Pipeline",
        "_diffusers_version": "0.32.2",
        "_minimax_h3": {
            "schema_version": 1,
            "partition": "fl2va",
            "tasks": ["t2va", "fl2va"],
            "task_aliases": {},
            "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
        },
        "scheduler": None,
    }
    for name in MiniMaxH3Pipeline._required_config_modules:
        (root / name).mkdir()
        index[name] = ["diffusers", "UnusedComponent"]
    index["transformer"] = ["diffusers", "MiniMaxH3DiTModel"]
    (root / "model_index.json").write_text(json.dumps(index))
    (root / "transformer/config.json").write_text(
        json.dumps(
            {"_class_name": "MiniMaxH3DiTModel", **dit_minimax_h3.EXPECTED_CONFIG}
        )
    )
    save_file({}, root / "transformer/model.safetensors")
    (root / "transformer/model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"unused": "model.safetensors"}})
    )
    ModelRegistry.resolve_model_cls("MiniMaxH3DiTModel")
    with patch.object(ServerArgs, "_adjust_network_ports"):
        args = ServerArgs(
            model_path=str(tmp_path),
            model_variant="fl2va",
            pipeline_config=MiniMaxH3PipelineConfig(),
            weight_cache_mode="client",
            performance_mode="manual",
        )
    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base.maybe_download_model",
            return_value=str(tmp_path),
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
            return_value=True,
        ),
    ):
        yield args


def test_h3_reuses_ordinary_loader_and_pure_frozen_preparation(h3_args):
    assert dit_minimax_h3.load_ordinary is common.load_ordinary
    with (
        patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("module")),
        patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA")),
        patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader.get_local_torch_device",
            side_effect=AssertionError("rank"),
        ),
    ):
        prepared = prepare_pipeline(MiniMaxH3Pipeline, h3_args, required=True)
    assert prepared.adapter is dit_minimax_h3
    assert prepared.model_path == str(Path(h3_args.model_path) / "FL2VA")
    assert h3_args.model_subfolder is None and h3_args.model_paths == {}
    off = prepare_pipeline(
        MiniMaxH3Pipeline, h3_args.resolve_variant(weight_cache_mode="off")
    )
    assert prepared.specs == off.specs
    assert prepared.adapter.fingerprint_fields(prepared.transformer) == (
        off.adapter.fingerprint_fields(off.transformer)
    )
    args = copy.deepcopy(h3_args)
    prepared.apply_config(args)
    assert args.model_subfolder == "FL2VA"
    # Both paths consume the same public metadata validator, including scales
    # and partition admission. Prepared construction must not reread model_index.
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.server_args = args
    pipeline.configure_model_index(json.loads(prepared.model_index_json))
    assert pipeline.release_metadata.partition == "fl2va"
    assert pipeline.release_metadata.sigma_shift_scales == {"video": 12, "audio": 3}
    with pytest.raises(ValueError, match="not served"):
        pipeline.release_metadata.canonical_task("ref2va")
    ordinary_args = copy.deepcopy(h3_args)
    root, index = MiniMaxH3Pipeline.resolve_model_config(
        h3_args.model_path, ordinary_args
    )
    assert root == prepared.model_path
    assert index == json.loads(prepared.model_index_json)


@pytest.mark.parametrize(
    "variant",
    [
        "class",
        "config",
        "resolved_config",
        "layout",
        "curve",
        "gates",
        "adaln",
        "quant",
        "dtype",
        "attention",
        "cpu",
        "tp",
        "compile",
    ],
)
def test_h3_rejects_unverified_representations(h3_args, variant):
    recipe = prepare_pipeline(
        MiniMaxH3Pipeline, h3_args, required=True
    ).transformer.thaw()
    attention = "fa"
    arch = recipe.init_params["config"].arch_config
    if variant == "class":
        recipe.model_cls = type("MiniMaxH3DiTModel", (), {})
    elif variant == "config":
        recipe.init_params["hf_config"]["extension"] = 1
    elif variant == "resolved_config":
        arch.num_layers = 1
    elif variant == "layout":
        arch.checkpoint_uses_diffusers_layout = True
    elif variant == "curve":
        arch.adaln_curve_grid = 256
    elif variant == "gates":
        arch.has_gate_compress = True
    elif variant == "adaln":
        recipe.server_args.minimax_h3_adaln_online = True
    elif variant == "quant":
        recipe.quant_spec.post_load_hooks.append(object())
    elif variant == "dtype":
        recipe.quant_spec = replace(recipe.quant_spec, param_dtype=torch.float16)
    elif variant == "attention":
        attention = "torch_sdpa"
    elif variant == "cpu":
        recipe.component_starts_on_cpu = True
    elif variant == "tp":
        recipe.server_args.tp_size = 2
    elif variant == "compile":
        recipe.server_args.enable_torch_compile = True
    with pytest.raises(ValueError):
        dit_minimax_h3.validate_supported(
            SimpleNamespace(thaw=lambda: recipe),
            pipeline_name="MiniMaxH3Pipeline",
            attention=attention,
        )


def test_partition_conflict_and_wrong_metadata_rejected(h3_args):
    args = copy.deepcopy(h3_args)
    args.model_subfolder = "Ref2VA"
    with pytest.raises(ValueError, match="different weight partitions"):
        prepare_pipeline(MiniMaxH3Pipeline, args, required=True)
    root = Path(h3_args.model_path) / "FL2VA"
    index = json.loads((root / "model_index.json").read_text())
    index["_minimax_h3"].update(partition="ref2va", tasks=["ref2va"])
    (root / "model_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        prepare_pipeline(MiniMaxH3Pipeline, h3_args, required=True)


def test_materialization_consumes_frozen_metadata_without_rediscovery(h3_args):
    prepared = prepare_pipeline(MiniMaxH3Pipeline, h3_args, required=True)
    # Changing the source must not change the already frozen construction plan.
    # Production identity checks reject this change before any IPC import.
    (Path(prepared.model_path) / "model_index.json").write_text("{}")
    with (
        patch.object(
            MiniMaxH3Pipeline,
            "resolve_model_config",
            side_effect=AssertionError("rediscovery"),
        ),
        patch.object(
            MiniMaxH3Pipeline, "_materialize_component_specs", return_value={}
        ),
        patch.object(MiniMaxH3Pipeline, "build_executor", return_value=Mock()),
        patch.object(MiniMaxH3Pipeline, "__post_init__"),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.get_local_torch_device",
            return_value=torch.device("cpu"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline.shutil.which",
            return_value="/usr/bin/media-tool",
        ),
    ):
        pipeline = prepared.materialize(copy.deepcopy(h3_args))
    assert pipeline.release_metadata.partition == "fl2va"
    assert pipeline.release_metadata.sigma_shift_scales == {"video": 12, "audio": 3}


def test_fast_h3_and_diffusers_layout_not_admitted(h3_args):
    from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
        FastH3Pipeline,
    )
    from sglang.multimodal_gen.runtime.weight_cache import adapters

    assert adapters.for_pipeline(FastH3Pipeline) is None
    config = {
        "_class_name": "MiniMaxH3Transformer3DModel",
        **dit_minimax_h3.EXPECTED_CONFIG,
    }
    assert not dit_minimax_h3.supports_config(config)


def test_finalized_rope_kind_matches_ordinary_assign_loader():
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3Rope

    def skeleton():
        model = torch.nn.Module()
        model.rope = MiniMaxH3Rope(16)
        return model.eval()

    ordinary = skeleton()
    ordinary.load_state_dict(
        {"rope.inv_freq": torch.nn.Parameter(torch.ones(16), requires_grad=False)},
        assign=True,
    )
    assert "inv_freq" in ordinary.rope._parameters
    with torch.device("meta"):
        meta = skeleton()
    with patch.object(common, "build_meta", return_value=meta) as build:
        assert dit_minimax_h3.build_meta("frozen") is meta
    build.assert_called_once_with("frozen")
    validate_meta_schema(meta, snapshot_module(ordinary).manifest)


@pytest.mark.parametrize(
    "field",
    [
        None,
        "_adaln_precomputed",
        "adaln_cache",
        "adaln_t_table",
        "adaln_basis",
        "adaln_mean",
    ],
)
def test_import_only_runs_read_only_native_precision_checks(field):
    model = SimpleNamespace(
        _adaln_precomputed=False,
        adaln_cache=None,
        adaln_t_table=None,
        adaln_basis=None,
        adaln_mean=None,
        post_load_weights=Mock(),
    )
    if field is not None:
        setattr(model, field, True)
    with patch.object(dit_minimax_h3, "finalize_loaded_model", return_value=model):
        if field is None:
            assert dit_minimax_h3.finalize_after_import(model) is model
            model.post_load_weights.assert_called_once_with()
        else:
            with pytest.raises(ValueError, match="AdaLN"):
                dit_minimax_h3.finalize_after_import(model)
            model.post_load_weights.assert_not_called()


def _hf_prepared(tmp_path, partition):
    root = tmp_path / "models--org--H3" / "snapshots" / ("a" * 40) / partition
    component = root / "transformer"
    component.mkdir(parents=True)
    for name in (
        "model_index.json",
        "transformer/config.json",
        "transformer/model.safetensors",
        "transformer/model.safetensors.index.json",
    ):
        (root / name).write_text("{}")
    args = SimpleNamespace(model_paths={"transformer": str(component)})
    recipe = SimpleNamespace(
        server_args=args, weight_files=[str(component / "model.safetensors")]
    )
    return SimpleNamespace(
        model_path=str(root), transformer=SimpleNamespace(thaw=lambda: recipe)
    )


def test_hf_partition_identity_and_native_index(tmp_path):
    args = SimpleNamespace(weight_cache_allow_weak_checkpoint_identity=False)
    prepared = _hf_prepared(tmp_path, "FL2VA")
    identity = checkpoint_identity(prepared, args)
    assert identity["kind"] == "hf_snapshot"
    assert identity["subfolder"] == "FL2VA"
    assert "transformer/model.safetensors.index.json" in identity["files"]
    other = checkpoint_identity(_hf_prepared(tmp_path, "Ref2VA"), args)
    assert identity != other and other["subfolder"] == "Ref2VA"
    index = Path(prepared.model_path) / "transformer/model.safetensors.index.json"
    index.write_text('{"changed": true}')
    assert identity != checkpoint_identity(prepared, args)
    external = tmp_path / "external.safetensors"
    external.write_text("{}")
    weight = Path(prepared.model_path) / "transformer/model.safetensors"
    weight.unlink()
    weight.symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        checkpoint_identity(prepared, args)


def test_hf_root_identity_stays_compatible_and_bad_revision_fails_closed(tmp_path):
    args = SimpleNamespace(weight_cache_allow_weak_checkpoint_identity=False)
    prepared = _hf_prepared(tmp_path, "")
    assert "subfolder" not in checkpoint_identity(prepared, args)
    original = Path(prepared.model_path)
    bad = original.with_name("not-a-pinned-revision")
    original.rename(bad)
    recipe = prepared.transformer.thaw()
    recipe.server_args.model_paths["transformer"] = str(bad / "transformer")
    recipe.weight_files = [str(bad / "transformer/model.safetensors")]
    prepared.model_path = str(bad)
    with pytest.raises(ValueError, match="content manifest"):
        checkpoint_identity(prepared, args)
