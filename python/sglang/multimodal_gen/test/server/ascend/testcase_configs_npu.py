from sglang.multimodal_gen.test.server.testcase_configs import (
    T2V_PROMPT,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
)

ONE_NPU_CASES: list[DiffusionTestCase] = [
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
