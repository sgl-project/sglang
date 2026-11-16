# SPDX-License-Identifier: Apache-2.0
"""Model routing in the registry, without Hub access."""

import json

import pytest

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan_Animate_2_14B_Config,
    Wan_Animate_2_14B_Distilled_Config,
)
from sglang.multimodal_gen.runtime.models import registry as model_registry


@pytest.fixture(autouse=True)
def clear_registry_cache():
    registry.get_model_info.cache_clear()
    registry._get_config_info.cache_clear()
    yield
    registry.get_model_info.cache_clear()
    registry._get_config_info.cache_clear()


def _checkpoint_dir(
    root,
    *,
    name: str = "ckpt-7f3a",
    index_class: str = "SomePipeline",
    modular_class: str | None = None,
) -> str:
    """Laid out like a published repo: model_index.json, the component folders the index
    verifier looks for, and modular_model_index.json when a class is given."""
    ckpt = root / name
    ckpt.mkdir(parents=True)
    (ckpt / "transformer").mkdir()
    (ckpt / "vae").mkdir()
    index = {"_class_name": index_class, "_diffusers_version": "0.40.0.dev0"}
    (ckpt / "model_index.json").write_text(json.dumps(index))
    if modular_class is not None:
        (ckpt / "modular_model_index.json").write_text(
            json.dumps({**index, "_class_name": modular_class})
        )
    return str(ckpt)


@pytest.mark.parametrize("modular_class", [None, "SomeModularPipeline"])
def test_detectors_are_probed_with_the_modular_pipeline_class(
    tmp_path, monkeypatch, modular_class
):
    # A checkpoint may ship modular_model_index.json next to model_index.json and name
    # a finer-grained pipeline there; without it the probes are the path and the
    # model_index class only.
    probes = []

    def detector(probe: str) -> bool:
        probes.append(probe)
        return False

    monkeypatch.setattr(registry, "_MODEL_NAME_DETECTORS", [("probe", detector)])
    path = _checkpoint_dir(tmp_path, modular_class=modular_class)
    assert registry._get_config_info(path) is None
    expected = [path.lower(), "somepipeline"]
    if modular_class is not None:
        expected.append(modular_class.lower())
    assert probes == expected


# Wan-Animate-2: both repos ship model_index.json with the same _class_name;
# modular_model_index.json names the variant, and a distill token in the directory's
# own name selects the distilled entry.

_WAN_ANIMATE_2 = "WanAnimate2Pipeline"


@pytest.mark.parametrize(
    "modular_class, config_cls",
    [
        ("WanAnimate2ModularPipeline", Wan_Animate_2_14B_Config),
        ("WanAnimate2DistilledModularPipeline", Wan_Animate_2_14B_Distilled_Config),
        # A mirror that skipped modular_model_index.json leaves only the shared class.
        (None, Wan_Animate_2_14B_Config),
    ],
)
def test_neutral_directory_routes_by_its_modular_pipeline_class(
    tmp_path, modular_class, config_cls
):
    path = _checkpoint_dir(
        tmp_path, index_class=_WAN_ANIMATE_2, modular_class=modular_class
    )
    assert registry._get_config_info(path).pipeline_config_cls is config_cls


def test_distill_token_in_the_directory_name_routes_to_the_distilled_config(tmp_path):
    # The token routes without the modular index; the shared model_index class matches
    # the base detector too, so the distilled entry is registered first.
    path = _checkpoint_dir(
        tmp_path, name="wan_animate_2_distilled", index_class=_WAN_ANIMATE_2
    )
    info = registry._get_config_info(path)
    assert info.pipeline_config_cls is Wan_Animate_2_14B_Distilled_Config


def test_distill_token_in_a_parent_directory_leaves_the_checkpoint_in_charge(tmp_path):
    # Only the directory's own name counts: a base checkpoint kept under a parent such
    # as distillation-runs/ routes by its modular class.
    path = _checkpoint_dir(
        tmp_path / "distillation-runs",
        name="wan_animate_2_base",
        index_class=_WAN_ANIMATE_2,
        modular_class="WanAnimate2ModularPipeline",
    )
    assert (
        registry._get_config_info(path).pipeline_config_cls is Wan_Animate_2_14B_Config
    )


def test_model_id_short_name_selects_the_variant_for_a_neutral_directory():
    info = registry._get_config_info(
        "/models/ckpt-7f3a", model_id="Wan2.2-Animate-2-14B-Distilled-Diffusers"
    )
    assert info.pipeline_config_cls is Wan_Animate_2_14B_Distilled_Config


@pytest.mark.parametrize("entry", ["Model", '[Model, "Model"]', '(Model, "Model")'])
def test_model_discovery_preserves_nested_declarations(tmp_path, monkeypatch, entry):
    """Skipping expression subtrees must retain declarations under statements."""
    component = tmp_path / "dits"
    component.mkdir()
    (component / "example.py").write_text(
        "import nonexistent_optional_backend\n"
        "if True:\n"
        "    class Model:\n"
        "        _aliases = ['external.Model']\n"
        "        def forward(self, x):\n"
        "            return (x + 1) * 2\n"
        f"    EntryClass = {entry}\n"
    )
    aliases = {}
    monkeypatch.setattr(model_registry, "MODELS_PATH", str(tmp_path))
    monkeypatch.setattr(model_registry, "COMPONENT_DIRS", ["dits"])
    monkeypatch.setattr(model_registry, "_IMAGE_ENCODER_MODELS", {})
    monkeypatch.setattr(model_registry, "_ALIAS_TO_MODEL", aliases)

    # bypass the cache without changing the registry used by other tests
    discovered = model_registry._discover_and_register_models.__wrapped__()
    assert discovered == {"Model": ("dits", "example", "Model")}
    assert aliases == {"external.Model": "Model"}
