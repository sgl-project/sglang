import os
from dataclasses import replace

from sglang.multimodal_gen.test.server.testcase_configs import (
    T2V_PROMPT,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    T2I_sampling_params,
    TI2I_sampling_params,
    TI2V_sampling_params,
)
from sglang.srt.environ import envs

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
MINIMAX_H3_WEIGHTS_PATH = use_modelscope("MiniMax/MiniMax-H3")
QWEN_IMAGE_WEIGHTS_PATH = use_modelscope("Qwen/Qwen-Image")
QWEN_IMAGE_21_WEIGHTS_PATH = use_modelscope("Qwen/Qwen-Image-2.1")
WAN2_1_T2V_1_3B_DIFFUSERS_WEIGHTS_PATH = use_modelscope(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
)
WAN2_2_T2V_A14B_DIFFUSERS_W8A8_WEIGHTS_PATH = use_modelscope(
    "Eco-Tech/Wan2.2-T2V-A14B-Diffusers-w8a8"
)
Z_IMAGE_WEIGHTS_PATH = use_modelscope("Tongyi-MAI/Z-Image")

EXTRAS_DISABLE_WARMUP = ["--warmup-mode", "request"]

QWEN_IMAGE21_TRANSPARENT_PROMPT = (
    "A single fluffy orange cat sitting, full body, isolated on a transparent "
    "background. A clean cutout with an alpha channel, transparent outside the "
    "cat, no floor, no shadow, no background."
)
QWEN_IMAGE21_RECOLOR_PROMPT = (
    "Change the orange skirt and bow to deep blue, keeping her pose, face, hair, "
    "and the SGL logo unchanged."
)
QWEN_IMAGE21_MULTI_IMAGE_PROMPT = (
    "The girl from image 1 is standing on the left, and the white rabbit from "
    "image 2 is sitting on the right, in a park."
)


def _multi_image_inputs():
    """Both references are staged by prefetch_input_image.py in CI."""
    first = envs.SGLANG_TEST_TI2I_INPUT_IMAGE.get()
    second = envs.SGLANG_TEST_MULTI_IMAGE_INPUT_IMAGE.get()
    return [first, second] if first and second else None


ONE_NPU_CASES: list[DiffusionTestCase] = [
    # === Text to Image (T2I) ===
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
    DiffusionTestCase(
        "qwen_image21_t2i_1npu",
        DiffusionServerArgs(
            model_path=QWEN_IMAGE_21_WEIGHTS_PATH,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        # Pins the sampling contract the Ascend GT image is recorded with.
        replace(
            T2I_sampling_params,
            output_size="1024x1024",
            output_format="png",
            extras={
                "num_inference_steps": 40,
                "guidance_scale": 1,
                "seed": 42,
                "quality": "exact",
            },
        ),
        # Enabled together with the NPU GT pin once the GT-gen workflow has
        # published the image; keeping it off avoids blocking this PR on a GT
        # that does not exist yet.
        run_consistency_check=False,
        run_component_accuracy_check=False,
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
    DiffusionTestCase(
        "qwen_image21_edit_ti2i_1npu",
        DiffusionServerArgs(
            model_path=QWEN_IMAGE_21_WEIGHTS_PATH,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        # Reuses the CI-prefetched edit input, like the other TI2I NPU cases.
        # Names quality explicitly so a later consistency check would compare on
        # the same path the GT was recorded on.
        replace(
            TI2I_sampling_params,
            output_size="1024x1024",
            output_format="png",
            extras={
                "num_inference_steps": 40,
                "guidance_scale": 1,
                "seed": 42,
                "quality": "exact",
            },
        ),
        run_consistency_check=False,
        run_component_accuracy_check=False,
    ),
    # === Qwen-Image 2.1 feature coverage ===
    DiffusionTestCase(
        "qwen_image21_transparent_t2i_1npu",
        DiffusionServerArgs(
            model_path=QWEN_IMAGE_21_WEIGHTS_PATH,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        replace(
            T2I_sampling_params,
            prompt=QWEN_IMAGE21_TRANSPARENT_PROMPT,
            output_size="1024x1024",
            output_format="png",
            extras={
                "num_inference_steps": 40,
                "guidance_scale": 1,
                "seed": 42,
                "quality": "exact",
                "background": "transparent",
            },
        ),
        run_consistency_check=False,
        run_component_accuracy_check=False,
    ),
    DiffusionTestCase(
        "qwen_image21_edit_multi_image_ti2i_1npu",
        DiffusionServerArgs(
            model_path=QWEN_IMAGE_21_WEIGHTS_PATH,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        # Two references staged by prefetch_input_image.py; the second upload
        # exercises the <image2> slot.
        DiffusionSamplingParams(
            prompt=QWEN_IMAGE21_MULTI_IMAGE_PROMPT,
            image_path=_multi_image_inputs(),
            output_size="1024x1024",
            output_format="png",
            extras={
                "num_inference_steps": 40,
                "guidance_scale": 1,
                "seed": 42,
                "quality": "exact",
            },
        ),
        run_consistency_check=False,
        run_component_accuracy_check=False,
    ),
    DiffusionTestCase(
        "qwen_image21_edit_recolor_ti2i_1npu",
        DiffusionServerArgs(
            model_path=QWEN_IMAGE_21_WEIGHTS_PATH,
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        replace(
            TI2I_sampling_params,
            prompt=QWEN_IMAGE21_RECOLOR_PROMPT,
            output_size="1024x1024",
            output_format="png",
            extras={
                "num_inference_steps": 40,
                "guidance_scale": 1,
                "seed": 42,
                "quality": "exact",
            },
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
    # === Text to Video+Audio (T2VA)
    DiffusionTestCase(
        "minimax_h3_t2va_2npu",
        DiffusionServerArgs(
            model_path=MINIMAX_H3_WEIGHTS_PATH,
            modality="video",
            num_gpus=2,
            tp_size=2,
            extras=[
                "--model-variant",
                "fl2va",
                "--dit-cpu-offload",
                "false",
                "--sp-degree",
                "1",
                "--attention-backend",
                "laser_attn",
                "--component-residency",
                "text_encoder=layerwise-offload",
            ],
            env_vars={
                "SGLANG_CACHE_DIT_ENABLED": "true",
                "SGLANG_CACHE_DIT_FN": "2",
                "SGLANG_CACHE_DIT_BN": "1",
                "SGLANG_CACHE_DIT_WARMUP": "4",
                "SGLANG_CACHE_DIT_RDT": "0.4",
                "SGLANG_CACHE_DIT_MC": "4",
                "SGLANG_CACHE_DIT_TAYLORSEER": "true",
                "SGLANG_CACHE_DIT_TS_ORDER": "2",
                "HCCL_BUFFSIZE": "256",
                "SGLANG_DIFFUSION_TEST_FORCE_HOST_AVAILABLE_GIB": "64",
            },
        ),
        DiffusionSamplingParams(
            prompt=(
                "At night, while their owner sleeps in a bedroom, three cats "
                "march in loudly playing tiny brass instruments, then abruptly "
                "file out."
            ),
            output_size="1344x768",
            seconds=5,
            output_format="mp4",
            num_outputs_per_prompt=1,
            extras={
                "task": "t2va",
                "conditions": [],
                "target": {
                    "short_edge": 768,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 5.0,
                },
                "num_inference_steps": 50,
                "flow_shift": 12.0,
                "audio_flow_shift": 3.0,
                "seed": 1101,
            },
        ),
        pin_consistency_quality=False,
        run_perf_check=True,
        run_consistency_check=True,
        run_component_accuracy_check=False,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    ),
    # === Text+Image to Video+Audio (TI2VA)
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
            env_vars={
                "SGLANG_DIFFUSION_TEST_FORCE_HOST_AVAILABLE_GIB": "96",
            },
            extras=EXTRAS_DISABLE_WARMUP,
        ),
        run_consistency_check=False,
    ),
]

DEFAULT_EST_TIME_SECONDS = 300.0
STARTUP_OVERHEAD_SECONDS = 120.0
DEFAULT_STANDALONE_EST_TIME_SECONDS = 300.0

STANDALONE_FILES = {
    "2-npu": [
        "ascend/test_glm_image_distributed.py",
    ],
}

STANDALONE_FILE_EST_TIMES = {
    "2-npu": {
        "ascend/test_glm_image_distributed.py": 900.0,
    },
}

SUITES = {
    "1-npu": [
        "ascend/test_server_1_npu.py",
        *STANDALONE_FILES.get("1-npu", []),
    ],
    "2-npu": [
        "ascend/test_server_2_npu.py",
        *STANDALONE_FILES.get("2-npu", []),
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
COMPONENT_ACCURACY_SUITES = {}
_UPDATE_WEIGHTS_FROM_DISK_TEST_FILE = None
