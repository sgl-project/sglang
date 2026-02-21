from sglang.multimodal_gen.test.server.testcase_configs import (
    T2V_PROMPT,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    T2I_sampling_params,
)

ONE_NPU_CASES: list[DiffusionTestCase] = [
    # === Text to Image (T2I) ===
    DiffusionTestCase(
        "flux_image_t2i_npu",
        DiffusionServerArgs(
            model_path="/root/.cache/modelscope/hub/models/black-forest-labs/FLUX.1-dev",
            modality="image",
            warmup=True,
        ),
        T2I_sampling_params,
    ),
    # === Text to Video (T2V) ===
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_1_npu",
        DiffusionServerArgs(
            model_path="/root/.cache/modelscope/hub/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            modality="video",
            custom_validator="video",
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
        ),
    ),
]

TWO_NPU_CASES: list[DiffusionTestCase] = [
    # === Text to Image (T2I) ===
    DiffusionTestCase(
        "flux_2_image_t2i_2npu",
        DiffusionServerArgs(
            model_path="/root/.cache/modelscope/hub/models/black-forest-labs/FLUX.2-dev",
            modality="image",
            warmup=True,
            num_gpus=2,
            tp_size=2,
        ),
        T2I_sampling_params,
    ),
]
