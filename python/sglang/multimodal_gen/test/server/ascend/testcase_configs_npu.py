import os

from sglang.multimodal_gen.test.server.testcase_configs import (
    T2V_PROMPT,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    T2I_sampling_params,
)

MODEL_WEIGHTS_DIR = "/root/.cache/modelscope/hub/models/"

FLUX_1_DEV_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "black-forest-labs/FLUX.1-dev"
)
FLUX_2_DEV_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "black-forest-labs/FLUX.2-dev"
)
QWEN_IMAGE_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen-Image")
WAN2_1_T2V_1_3B_DIFFUSERS_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
)
WAN2_2_T2V_A14B_DIFFUSERS_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Eco-Tech/Wan2.2-T2V-A14B-Diffusers-w8a8"
)

EXTRAS_DISABLE_WARMUP = ["--server-warmup", "false"]

ONE_NPU_CASES: list[DiffusionTestCase] = [
    # === Text to Image (T2I) ===
    DiffusionTestCase(
        "flux_image_t2i_npu",
        DiffusionServerArgs(
            model_path=FLUX_1_DEV_WEIGHTS_PATH,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    # === Text to Video (T2V) ===
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_1_npu",
        DiffusionServerArgs(
            model_path=WAN2_1_T2V_1_3B_DIFFUSERS_WEIGHTS_PATH,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
        ),
        run_consistency_check=False,
    ),
]

TWO_NPU_CASES: list[DiffusionTestCase] = [
    # === Text to Image (T2I) ===
    DiffusionTestCase(
        "flux_2_image_t2i_2npu",
        DiffusionServerArgs(
            model_path=FLUX_2_DEV_WEIGHTS_PATH,
            num_gpus=2,
            tp_size=2,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_t2i_2npu",
        DiffusionServerArgs(
            model_path=QWEN_IMAGE_WEIGHTS_PATH,
            num_gpus=2,
            # test ring attn
            ulysses_degree=1,
            ring_degree=2,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    # === Text to Video (T2V) ===
    DiffusionTestCase(
        "wan2_2_t2v_14b_w8a8_2npu",
        DiffusionServerArgs(
            model_path=WAN2_2_T2V_A14B_DIFFUSERS_W8A8_WEIGHTS_PATH,
            num_gpus=2,
            tp_size=1,
            ulysses_degree=2,
            extras=EXTRAS_DISABLE_WARMUP,
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
_UPDATE_WEIGHTS_FROM_DISK_TEST_FILE = None
