import os

from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    maybe_load_overlay_model_index,
    resolve_model_overlay_target,
)


def test_ltx23_builtin_overlay_target_is_hf_repo():
    target = resolve_model_overlay_target("Lightricks/LTX-2.3")
    assert target is not None

    source_model_id, overlay_spec = target
    assert source_model_id == "Lightricks/LTX-2.3"
    assert str(overlay_spec["overlay_revision"]) == "main"
    assert str(overlay_spec["overlay_repo_id"]) == "MickJ/LTX-2.3-overlay"


def test_ltx23_overlay_model_index_is_native():
    local_overlay_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "model_overlays",
            "ltx_2_3",
        )
    )
    model_index = maybe_load_overlay_model_index(
        "Lightricks/LTX-2.3",
        snapshot_download_fn=lambda **kwargs: local_overlay_dir,
        hf_hub_download_fn=lambda **kwargs: None,
    )
    assert model_index is not None
    assert model_index["_class_name"] == "LTX2Pipeline"


def test_ltx23_model_info_resolves_to_native_pipeline():
    model_info = get_model_info("Lightricks/LTX-2.3", backend="sglang")
    assert model_info is not None
    assert model_info.pipeline_cls.__name__ == "LTX2Pipeline"
