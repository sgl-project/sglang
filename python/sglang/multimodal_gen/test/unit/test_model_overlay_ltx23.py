from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    load_overlay_manifest_if_present,
    maybe_load_overlay_model_index,
    resolve_model_overlay_target,
)


def test_ltx23_builtin_overlay_target_is_local():
    target = resolve_model_overlay_target("Lightricks/LTX-2.3")
    assert target is not None

    source_model_id, overlay_spec = target
    assert source_model_id == "Lightricks/LTX-2.3"
    assert str(overlay_spec["overlay_revision"]) == "local"

    overlay_dir = str(overlay_spec["overlay_repo_id"])
    manifest = load_overlay_manifest_if_present(overlay_dir)
    assert manifest is not None
    assert manifest["source_model_id"] == "Lightricks/LTX-2.3"


def test_ltx23_overlay_model_index_is_native():
    model_index = maybe_load_overlay_model_index(
        "Lightricks/LTX-2.3",
        snapshot_download_fn=lambda **kwargs: str(
            resolve_model_overlay_target("Lightricks/LTX-2.3")[1]["overlay_repo_id"]
        ),
        hf_hub_download_fn=lambda **kwargs: None,
    )
    assert model_index is not None
    assert model_index["_class_name"] == "LTX2Pipeline"


def test_ltx23_model_info_resolves_to_native_pipeline():
    model_info = get_model_info("Lightricks/LTX-2.3", backend="sglang")
    assert model_info is not None
    assert model_info.pipeline_cls.__name__ == "LTX2Pipeline"
