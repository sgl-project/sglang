from sglang.multimodal_gen.runtime.managers.memory_managers.component_loading_order import (
    ComponentLoadSpec,
    component_load_risk_rank,
    order_component_load_specs,
)


def _spec(component_name: str, index: int) -> ComponentLoadSpec:
    return ComponentLoadSpec(
        module_name=component_name,
        load_module_name=component_name,
        transformers_or_diffusers="diffusers",
        architecture=None,
        index=index,
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
            transformers_or_diffusers="diffusers",
            architecture=None,
            index=0,
        ),
        ComponentLoadSpec(
            module_name="encoder_alias",
            load_module_name="text_encoder_2",
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
