import os

from sglang.multimodal_gen.test.server.testcase_configs import (
    T2V_PROMPT,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    T2I_sampling_params,
    TI2V_sampling_params,
)

MODELSCOPE_MODEL_WEIGHTS_DIR = "/root/.cache/modelscope/hub/models/"


def use_modelscope(name: str):
    return os.path.join(MODELSCOPE_MODEL_WEIGHTS_DIR, name)


COSMOS3_NANO_WEIGHTS_PATH = use_modelscope("nv-community/Cosmos3-Nano")
ERNIE_IMAGE_WEIGHTS_PATH = use_modelscope("PaddlePaddle/ERNIE-Image")
FLUX_1_DEV_WEIGHTS_PATH = use_modelscope("black-forest-labs/FLUX.1-dev")
FLUX_2_DEV_WEIGHTS_PATH = use_modelscope("black-forest-labs/FLUX.2-dev")
FLUX_2_KLEIN_4B_WEIGHTS_PATH = use_modelscope("black-forest-labs/FLUX.2-klein-4B")
GLM_IMAGE_WEIGHTS_PATH = use_modelscope("ZhipuAI/GLM-Image")
JOYAI_IMAGE_EDIT_WEIGHTS_PATH = use_modelscope(
    "jd-opensource/JoyAI-Image-Edit-Diffusers"
)
LTX_2_WEIGHTS_PATH = use_modelscope("Lightricks/LTX-2")
MOVA_360_WEIGHTS_PATH = use_modelscope("openmoss/MOVA-360p")
QWEN_IMAGE_WEIGHTS_PATH = use_modelscope("Qwen/Qwen-Image")
WAN2_1_T2V_1_3B_DIFFUSERS_WEIGHTS_PATH = use_modelscope(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
)
WAN2_2_T2V_A14B_DIFFUSERS_W8A8_WEIGHTS_PATH = use_modelscope(
    "Eco-Tech/Wan2.2-T2V-A14B-Diffusers-w8a8"
)
Z_IMAGE_WEIGHTS_PATH = use_modelscope("Tongyi-MAI/Z-Image")

EXTRAS_DISABLE_WARMUP = ["--warmup-mode", "request"]

ONE_NPU_CASES: list[DiffusionTestCase] = [
    # === Text to Image (T2I) ===
    DiffusionTestCase(
        "cosmos3_nano_t2i_1npu",
        DiffusionServerArgs(
            model_path=COSMOS3_NANO_WEIGHTS_PATH,
            modality="image",
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        DiffusionSamplingParams(
            prompt="A red cube on a white table, product photo.",
            output_size="832x480",
            output_format="png",
            extras={
                "num_inference_steps": 35,
                "seed": 0,
                "max_sequence_length": 128,
                "flow_shift": 10.0,
                "extra_args": {
                    "guardrails": False,
                    "use_resolution_template": False,
                },
            },
        ),
        run_perf_check=False,
        run_consistency_check=True,
        run_component_accuracy_check=False,
    ),
    DiffusionTestCase(
        "ernie_image_t2i_1npu",
        DiffusionServerArgs(
            model_path=ERNIE_IMAGE_WEIGHTS_PATH,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "glm_image_t2i_1npu",
        DiffusionServerArgs(
            model_path=GLM_IMAGE_WEIGHTS_PATH,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "flux_image_t2i_npu",
        DiffusionServerArgs(
            model_path=FLUX_1_DEV_WEIGHTS_PATH,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "flux_2_klein_4b_t2i_1npu",
        DiffusionServerArgs(
            model_path=FLUX_2_KLEIN_4B_WEIGHTS_PATH,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "z_image_t2i_1npu",
        DiffusionServerArgs(
            model_path=Z_IMAGE_WEIGHTS_PATH,
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
    ),
    # === Text+Image to Image (TI2I)
    DiffusionTestCase(
        "joyai_image_edit_ti2i_1npu",
        DiffusionServerArgs(
            model_path=JOYAI_IMAGE_EDIT_WEIGHTS_PATH,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        run_consistency_check=False,
        run_component_accuracy_check=False,
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
    ),
    # === Text+Image to Video+Audio (TI2V)
    DiffusionTestCase(
        "ltx_2_ti2va_2npu",
        DiffusionServerArgs(
            model_path=LTX_2_WEIGHTS_PATH,
            num_gpus=2,
            ulysses_degree=2,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        TI2V_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "mova_360p_ti2va_2npu",
        DiffusionServerArgs(
            model_path=MOVA_360_WEIGHTS_PATH,
            num_gpus=2,
            tp_size=2,
            dit_layerwise_offload=True,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        run_perf_check=False,
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
