from __future__ import annotations

from sglang.multimodal_gen.test.server.testcase_configs import (
    ONE_GPU_CASES,
    TWO_GPU_CASES,
    DiffusionTestCase,
)


def _select_accuracy_cases(
    cases: list[DiffusionTestCase], enabled_ids: tuple[str, ...]
) -> list[DiffusionTestCase]:
    enabled = set(enabled_ids)
    return [case for case in cases if case.id in enabled]


ACCURACY_ONE_GPU_CASE_IDS = (
    "qwen_image_t2i",
    "qwen_image_t2i_cache_dit_enabled",
    "flux_image_t2i",
    "flux_2_image_t2i",
    "flux_2_klein_image_t2i",
    "layerwise_offload",
    "zimage_image_t2i",
    "zimage_image_t2i_fp8",
    "zimage_image_t2i_multi_lora",
    "qwen_image_edit_ti2i",
    "qwen_image_edit_2509_ti2i",
    "qwen_image_edit_2511_ti2i",
    "qwen_image_layered_i2i",
    "flux_2_image_t2i_upscaling_4x",
    "mova_360p_1gpu",
    "wan2_1_t2v_1.3b",
    "wan2_1_t2v_1.3b_text_encoder_cpu_offload",
    "wan2_1_t2v_1.3b_teacache_enabled",
    "wan2_1_t2v_1.3b_frame_interp_2x",
    "wan2_1_t2v_1.3b_upscaling_4x",
    "wan2_1_t2v_1.3b_frame_interp_2x_upscaling_4x",
    "wan2_1_t2v_1_3b_lora_1gpu",
    "flux_2_ti2i",
    "flux_2_t2i_customized_vae_path",
    "fast_hunyuan_video",
    "wan2_2_ti2v_5b",
    "fastwan2_2_ti2v_5b",
    "hunyuan3d_shape_gen",
    "turbo_wan2_1_t2v_1.3b",
    "flux_2_ti2i_multi_image_cache_dit",
)

ACCURACY_TWO_GPU_CASE_IDS = (
    "wan2_2_i2v_a14b_2gpu",
    "wan2_2_t2v_a14b_2gpu",
    "wan2_2_t2v_a14b_teacache_2gpu",
    "wan2_2_t2v_a14b_lora_2gpu",
    "wan2_1_t2v_14b_2gpu",
    "wan2_1_t2v_1.3b_cfg_parallel",
    "fsdp-inference",
    "mova_360p_tp2",
    "mova_360p_ring1_uly2",
    "mova_360p_ring2_uly1",
    "ltx_2_two_stage_t2v",
    "wan2_1_i2v_14b_480P_2gpu",
    "wan2_1_i2v_14b_lora_2gpu",
    "wan2_1_i2v_14b_720P_2gpu",
    "qwen_image_t2i_2_gpus",
    "zimage_image_t2i_2_gpus",
    "zimage_image_t2i_2_gpus_non_square",
    "flux_image_t2i_2_gpus",
    "flux_2_image_t2i_2_gpus",
    "flux_2_klein_ti2i_2_gpus",
)

ACCURACY_ONE_GPU_CASES = _select_accuracy_cases(
    ONE_GPU_CASES, ACCURACY_ONE_GPU_CASE_IDS
)
ACCURACY_TWO_GPU_CASES = _select_accuracy_cases(
    TWO_GPU_CASES, ACCURACY_TWO_GPU_CASE_IDS
)
