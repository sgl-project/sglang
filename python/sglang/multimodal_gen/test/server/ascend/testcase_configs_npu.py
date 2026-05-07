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
        ),
        T2I_sampling_params,
    ),
    # === Text to Video (T2V) ===
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_1_npu",
        DiffusionServerArgs(
            model_path="/root/.cache/modelscope/hub/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
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
            num_gpus=2,
            tp_size=2,
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_t2i_2npu",
        DiffusionServerArgs(
            model_path="/root/.cache/modelscope/hub/models/Qwen/Qwen-Image",
            num_gpus=2,
            # test ring attn
            ulysses_degree=1,
            ring_degree=2,
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
]

EIGHT_NPU_CASES: list[DiffusionTestCase] = [
    # === Text to Video (T2V) ===
    DiffusionTestCase(
        "wan2_2_t2v_14b_w8a8_8npu",
        DiffusionServerArgs(
            model_path="/root/.cache/modelscope/hub/models/Eco-Tech/Wan2.2-T2V-A14B-Diffusers-w8a8",
            num_gpus=8,
            tp_size=4,
            ulysses_degree=2,
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
        ),
        run_consistency_check=False,
    ),
]

DEFAULT_EST_TIME_SECONDS = 300.0
STARTUP_OVERHEAD_SECONDS = 120.0
DEFAULT_STANDALONE_EST_TIME_SECONDS = 300.0

SUITES = {
    "1-npu": [
        "ascend/test_server_1_npu.py",
        # add new 1-npu test files here
    ],
    "2-npu": [
        "ascend/test_server_2_npu.py",
        # add new 2-npu test files here
    ],
    "8-npu": [
        "ascend/test_server_8_npu.py",
        # add new 8-npu test files here
    ],
}

PARAMETRIZED_CASE_GROUPS = {
    "1-npu": [
        ("ascend/test_server_1_npu.py", ONE_NPU_CASES),
    ],
    "2-npu": [
        ("ascend/test_server_2_npu.py", TWO_NPU_CASES),
    ],
}

FILE_SUITES = {}
STANDALONE_FILES = {}
COMPONENT_ACCURACY_SUITES = {}
