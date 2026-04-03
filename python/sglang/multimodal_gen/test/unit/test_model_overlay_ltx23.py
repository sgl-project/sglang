import os
from types import SimpleNamespace

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
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


def test_ltx23_model_info_uses_ltx23_sampling_params():
    model_info = get_model_info("Lightricks/LTX-2.3", backend="sglang")
    assert model_info is not None
    assert model_info.sampling_param_cls.__name__ == "LTX23SamplingParams"


def test_ltx23_prepare_request_sets_stage1_guider_defaults():
    sampling_params = SamplingParams.from_pretrained(
        "Lightricks/LTX-2.3",
        backend="sglang",
    )
    server_args = SimpleNamespace(
        attention_backend_config=SimpleNamespace(VSA_sparsity=0.0),
    )

    req = prepare_request(server_args, sampling_params)

    assert req.num_inference_steps == 30
    assert req.guidance_scale == 3.0
    assert req.extra["ltx2_stage1_guider_params"] == {
        "video_cfg_scale": 3.0,
        "video_stg_scale": 1.0,
        "video_rescale_scale": 0.7,
        "video_modality_scale": 3.0,
        "video_skip_step": 0,
        "video_stg_blocks": [28],
        "audio_cfg_scale": 7.0,
        "audio_stg_scale": 1.0,
        "audio_rescale_scale": 0.7,
        "audio_modality_scale": 3.0,
        "audio_skip_step": 0,
        "audio_stg_blocks": [28],
    }
