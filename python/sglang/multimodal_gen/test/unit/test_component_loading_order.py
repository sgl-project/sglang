import json
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.managers.memory_managers.component_loading_order import (
    ComponentLoadSpec,
    component_load_risk_rank,
    infer_component_weight_size_bytes,
    order_component_load_specs,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_weight_inventory import (
    ComponentWeightSource,
    estimate_component_weight_inventory,
    infer_safetensors_weight_stats_by_prefix,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)


def _spec(
    component_name: str, index: int, component_model_path: str = "/missing"
) -> ComponentLoadSpec:
    return ComponentLoadSpec(
        module_name=component_name,
        load_module_name=component_name,
        component_model_path=component_model_path,
        transformers_or_diffusers=(
            "transformers" if component_name.startswith("text_encoder") else "diffusers"
        ),
        architecture=None,
        index=index,
    )


def _write_safetensors(path, payload_size: int) -> None:
    header = {
        "weight": {
            "dtype": "F16",
            "shape": [payload_size // 2],
            "data_offsets": [0, payload_size],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(
        len(header_bytes).to_bytes(8, "little") + header_bytes + b"\0" * payload_size
    )


def test_component_load_order_prioritizes_weight_heavy_components():
    specs = [
        _spec("scheduler", 0),
        _spec("tokenizer", 1),
        _spec("text_encoder", 2),
        _spec("transformer", 3),
        _spec("vae", 4),
    ]

    ordered_names = [spec.module_name for spec in order_component_load_specs(specs)]

    assert ordered_names == [
        "transformer",
        "text_encoder",
        "vae",
        "scheduler",
        "tokenizer",
    ]


def test_component_load_order_prioritizes_larger_numbered_variants():
    specs = [
        _spec("transformer", 0),
        _spec("transformer_2", 1),
        _spec("text_encoder", 2),
        _spec("text_encoder_3", 3),
        _spec("text_encoder_2", 4),
    ]

    ordered_names = [spec.module_name for spec in order_component_load_specs(specs)]

    assert ordered_names == [
        "transformer_2",
        "transformer",
        "text_encoder_3",
        "text_encoder_2",
        "text_encoder",
    ]


def test_component_load_order_uses_load_module_name_for_extra_config_alias():
    specs = [
        ComponentLoadSpec(
            module_name="condition_image_encoder",
            load_module_name="condition_image_encoder",
            component_model_path="/missing",
            transformers_or_diffusers="diffusers",
            architecture=None,
            index=0,
        ),
        ComponentLoadSpec(
            module_name="encoder_alias",
            load_module_name="text_encoder_2",
            component_model_path="/missing",
            transformers_or_diffusers="transformers",
            architecture=None,
            index=1,
        ),
    ]

    ordered_names = [spec.module_name for spec in order_component_load_specs(specs)]

    assert ordered_names == ["encoder_alias", "condition_image_encoder"]


def test_component_load_risk_rank_keeps_small_helpers_last():
    assert component_load_risk_rank("transformer") < component_load_risk_rank(
        "scheduler"
    )
    assert component_load_risk_rank("text_encoder_2") < component_load_risk_rank(
        "processor"
    )
    assert component_load_risk_rank("vae") < component_load_risk_rank("tokenizer")


def test_component_load_order_prefers_inferred_safetensors_size(tmp_path):
    small_transformer_path = tmp_path / "small_transformer"
    large_encoder_path = tmp_path / "large_encoder"
    small_transformer_path.mkdir()
    large_encoder_path.mkdir()
    _write_safetensors(small_transformer_path / "model.safetensors", 16)
    _write_safetensors(large_encoder_path / "model.safetensors", 64)

    specs = [
        _spec("transformer", 0, str(small_transformer_path)),
        _spec("text_encoder", 1, str(large_encoder_path)),
        _spec("scheduler", 2),
    ]

    ordered_names = [spec.module_name for spec in order_component_load_specs(specs)]

    assert ordered_names == ["text_encoder", "transformer", "scheduler"]
    assert infer_component_weight_size_bytes(str(large_encoder_path)) == 64


def test_safetensors_inventory_splits_bundled_components_by_prefix(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    header = {
        "model.block.weight": {
            "dtype": "F16",
            "shape": [8],
            "data_offsets": [0, 16],
        },
        "vae.decoder.weight": {
            "dtype": "F16",
            "shape": [4],
            "data_offsets": [16, 24],
        },
        "model.proj.weight": {
            "dtype": "F16",
            "shape": [2],
            "data_offsets": [24, 28],
        },
    }
    header_bytes = json.dumps(header).encode("utf-8")
    checkpoint.write_bytes(
        len(header_bytes).to_bytes(8, "little") + header_bytes + b"\0" * 28
    )

    assert infer_safetensors_weight_stats_by_prefix(str(checkpoint)) == {
        "model": (20, 10),
        "vae": (8, 4),
    }


def test_component_weight_inventory_uses_explicit_bundled_size(tmp_path):
    component_dir = tmp_path / "transformer"
    component_dir.mkdir()
    _write_safetensors(component_dir / "model.safetensors", 64)

    inventory = estimate_component_weight_inventory(
        [
            ComponentWeightSource(
                "transformer",
                str(component_dir),
                supports_fsdp_loading=True,
            ),
            ComponentWeightSource(
                "bundled_vae",
                "/shared/model",
                checkpoint_bytes=8,
                parameter_count=2,
                target_element_size=4,
            ),
        ]
    )

    assert [
        (item.component_name, item.checkpoint_bytes, item.parameter_count)
        for item in inventory
    ] == [
        ("transformer", 64, 32),
        ("bundled_vae", 8, 2),
    ]
    assert inventory[0].materialized_bytes() == 64
    assert inventory[1].materialized_bytes() == 8
    assert inventory[0].supports_fsdp_loading
    assert not inventory[1].supports_fsdp_loading


def test_component_weight_inventory_distinguishes_config_only_and_unknown(tmp_path):
    config_only = tmp_path / "tokenizer"
    config_only.mkdir()
    (config_only / "tokenizer_config.json").write_text("{}")

    inventory = estimate_component_weight_inventory(
        [
            ComponentWeightSource("tokenizer", str(config_only)),
            ComponentWeightSource("transformer", "remote/model"),
        ]
    )

    assert [
        (item.component_name, item.checkpoint_bytes, item.parameter_count)
        for item in inventory
    ] == [
        ("tokenizer", 0, 0),
        ("transformer", None, None),
    ]


def test_preload_inventory_uses_the_actual_weight_override():
    server_args = SimpleNamespace(
        component_precisions={},
        component_weights_paths={"text_encoder": "/weights/text.safetensors"},
        transformer_weights_path="/weights/dit.gguf",
        pipeline_config=SimpleNamespace(
            dit_precision="bf16",
            vae_precision="fp32",
            image_encoder_precision="fp16",
            text_encoder_precisions=("bf16",),
        ),
    )

    text_source = ComposedPipelineBase._component_weight_source(
        _spec("text_encoder", 0, "/model/text_encoder"), server_args
    )
    transformer_source = ComposedPipelineBase._component_weight_source(
        _spec("transformer", 1, "/model/transformer"), server_args
    )
    secondary_source = ComposedPipelineBase._component_weight_source(
        _spec("transformer_2", 2, "/model/transformer_2"), server_args
    )

    assert text_source.component_model_path == "/weights/text.safetensors"
    assert transformer_source.component_model_path == "/weights/dit.gguf"
    assert secondary_source.component_model_path == "/model/transformer_2"
    assert transformer_source.target_element_size == 2
    assert secondary_source.target_element_size == 2
    assert not text_source.supports_fsdp_loading
    assert transformer_source.supports_fsdp_loading
    assert secondary_source.supports_fsdp_loading


def test_preload_inventory_uses_selected_transformer_safetensors(tmp_path):
    mixed_path = tmp_path / "flux2-dev-nvfp4-mixed.safetensors"
    _write_safetensors(mixed_path, 64)
    server_args = SimpleNamespace(
        component_precisions={},
        component_weights_paths={},
        transformer_weights_path="owner/flux2-nvfp4",
        revision=None,
        pipeline_config=SimpleNamespace(
            dit_precision="bf16",
            vae_precision="fp32",
            image_encoder_precision="fp16",
            text_encoder_precisions=("bf16",),
        ),
    )

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base."
        "resolve_transformer_checkpoint_files",
        return_value=SimpleNamespace(safetensors=(str(mixed_path),)),
    ):
        source = ComposedPipelineBase._component_weight_source(
            _spec("transformer", 0, "/model/transformer"), server_args
        )

    assert source.component_model_path == "owner/flux2-nvfp4"
    assert source.checkpoint_bytes == 64
    assert source.parameter_count == 32


def test_preload_inventory_resolves_group_precision_fallbacks():
    server_args = SimpleNamespace(
        component_precisions={},
        pipeline_config=SimpleNamespace(
            dit_precision="bf16",
            vae_precision="fp32",
            image_encoder_precision="fp16",
            text_encoder_precisions=("bf16",),
        ),
    )

    assert (
        ComposedPipelineBase._component_target_element_size(
            "unconditional_transformer", server_args
        )
        == 2
    )
    assert (
        ComposedPipelineBase._component_target_element_size(
            "condition_image_encoder", server_args
        )
        == 4
    )
