from __future__ import annotations

from sglang.multimodal_gen.test.server.testcase_configs import (
    T2V_PROMPT,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    T2I_sampling_params,
    TI2V_sampling_params,
)

ONE_GPU_MUSA_CASES: list[DiffusionTestCase] = [
    DiffusionTestCase(
        "qwen_image_t2i_musa",
        DiffusionServerArgs(
            # model_path="Qwen/Qwen-Image",
            model_path="/hf-cache/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6",
            modality="image",
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_musa",
        DiffusionServerArgs(
            # model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            model_path="/hf-cache/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b",
            modality="video",
            custom_validator="video",
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
        ),
        run_consistency_check=False,
    ),
]


TWO_GPU_MUSA_CASES: list[DiffusionTestCase] = [
    DiffusionTestCase(
        "wan2_1_i2v_14b_480P_2gpu_musa",
        DiffusionServerArgs(
            # model_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            model_path="/hf-cache/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers/snapshots/b184e23a8a16b20f108f727c902e769e873ffc73",
            modality="video",
            custom_validator="video",
            num_gpus=2,
        ),
        TI2V_sampling_params,
        run_consistency_check=False,
    ),
]
