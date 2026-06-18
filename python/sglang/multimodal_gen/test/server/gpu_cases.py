from dataclasses import replace
from pathlib import Path

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.test.server.testcase_configs import (
    MODELOPT_FLUX1_FP8_TRANSFORMER,
    MODELOPT_FLUX1_NVFP4_TRANSFORMER,
    MODELOPT_FLUX2_FP8_TRANSFORMER,
    MODELOPT_FLUX2_NVFP4_WEIGHTS,
    MODELOPT_HUNYUANVIDEO_FP8_TRANSFORMER,
    MODELOPT_NVFP4_B200_ENV_VARS,
    MODELOPT_QWEN_IMAGE_EDIT_FP8_TRANSFORMER,
    MODELOPT_QWEN_IMAGE_FP8_TRANSFORMER,
    MODELOPT_WAN22_FP8_MODEL,
    MODELOPT_WAN22_NVFP4_B200_ENV_VARS,
    MODELOPT_WAN22_NVFP4_MODEL,
    T2V_PROMPT,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    IDEOGRAM4_CI_sampling_params,
    LINGBOT_WORLD_REALTIME_sampling_params,
    MODELOPT_T2I_CI_sampling_params,
    MODELOPT_T2V_CI_sampling_params,
    MODELOPT_TI2I_CI_sampling_params,
    MULTI_FRAME_I2I_sampling_params,
    MULTI_IMAGE_TI2I_sampling_params,
    MULTI_IMAGE_TI2I_UPLOAD_sampling_params,
    SANA_WM_TI2V_CI_sampling_params,
    T2I_sampling_params,
    T2V_sampling_params,
    _make_modelopt_ci_case,
    _with_default_num_gpus,
)
from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_COSMOS3_NANO_MODEL_NAME_FOR_TEST,
    DEFAULT_FLUX_1_DEV_MODEL_NAME_FOR_TEST,
    DEFAULT_FLUX_2_DEV_MODEL_NAME_FOR_TEST,
    DEFAULT_FLUX_2_KLEIN_4B_MODEL_NAME_FOR_TEST,
    DEFAULT_FLUX_2_KLEIN_BASE_4B_MODEL_NAME_FOR_TEST,
    DEFAULT_JOYAI_IMAGE_EDIT_MODEL_NAME_FOR_TEST,
    DEFAULT_MOVA_360P_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_EDIT_2509_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_EDIT_2511_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_EDIT_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_LAYERED_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
    DEFAULT_SANA_WM_STREAMING_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_1_I2V_14B_480P_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_1_I2V_14B_720P_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_1_T2V_1_3B_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_1_T2V_14B_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_2_I2V_A14B_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_2_T2V_A14B_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_2_TI2V_5B_MODEL_NAME_FOR_TEST,
)

_CACHE_DIT_CONFIG_DIR = Path(__file__).parent / "configs"


# All test cases with clean default values
# To test different models, simply add more DiffusionCase entries
ONE_GPU_CASES: list[DiffusionTestCase] = [
    # === Text to Image (T2I) ===
    DiffusionTestCase(
        "qwen_image_t2i",
        DiffusionServerArgs(
            model_path=DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
        ),
    ),
    DiffusionTestCase(
        "qwen_image_t2i_cache_dit_enabled",
        DiffusionServerArgs(
            model_path=DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
            enable_cache_dit=True,
        ),
    ),
    DiffusionTestCase(
        "qwen_image_t2i_cache_dit_scm_config_diffusers_1gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
            extras=[
                "--backend",
                "diffusers",
                "--cache-dit-config",
                str(_CACHE_DIT_CONFIG_DIR / "cache_dit_scm_config.yaml"),
            ],
        ),
        replace(
            T2I_sampling_params,
            output_size="512x512",
            extras={"num_inference_steps": 8, "seed": 0},
        ),
        run_perf_check=False,
        run_consistency_check=False,
        run_component_accuracy_check=False,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    ),
    DiffusionTestCase(
        "flux_image_t2i",
        DiffusionServerArgs(model_path=DEFAULT_FLUX_1_DEV_MODEL_NAME_FOR_TEST),
    ),
    # TODO: modeling of flux different from official flux, so weights can't be loaded
    # consider opting for a different quantized hf-repo
    # DiffusionTestCase(
    #     "flux_image_t2i_override_transformer_weights_path_fp8",
    #     DiffusionServerArgs(
    #         model_path="black-forest-labs/FLUX.1-dev",
    #         extras=["--transformer-weights-path black-forest-labs/FLUX.1-dev-FP8"]
    #     ),
    #     T2I_sampling_params,
    # ),
    DiffusionTestCase(
        "flux_2_image_t2i",
        DiffusionServerArgs(model_path=DEFAULT_FLUX_2_DEV_MODEL_NAME_FOR_TEST),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "flux_2_klein_image_t2i",
        DiffusionServerArgs(
            model_path=DEFAULT_FLUX_2_KLEIN_4B_MODEL_NAME_FOR_TEST,
        ),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "flux_2_klein_base_image_t2i",
        DiffusionServerArgs(
            model_path=DEFAULT_FLUX_2_KLEIN_BASE_4B_MODEL_NAME_FOR_TEST,
        ),
        T2I_sampling_params,
        run_consistency_check=False,
        run_component_accuracy_check=False,
    ),
    # TODO: replace with a faster model to test the --dit-layerwise-offload
    # TODO: currently, we don't support sending more than one request in test, and setting `num_outputs_per_prompt` to 2 doesn't guarantee the denoising be executed twice,
    # so we do one warmup and send one request instead
    DiffusionTestCase(
        "layerwise_offload",
        DiffusionServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            dit_layerwise_offload=True,
            dit_offload_prefetch_size=2,
        ),
    ),
    DiffusionTestCase(
        "zimage_image_t2i",
        DiffusionServerArgs(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST),
    ),
    DiffusionTestCase(
        "zimage_image_t2i_fp8",
        DiffusionServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            extras=["--transformer-path MickJ/Z-Image-Turbo-fp8"],
        ),
    ),
    # Multi-LoRA test case for Z-Image-Turbo
    DiffusionTestCase(
        "zimage_image_t2i_multi_lora",
        DiffusionServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            lora_path="reverentelusarca/elusarca-anime-style-lora-z-image-turbo",
            second_lora_path="tarn59/pixel_art_style_lora_z_image_turbo",
        ),
        run_lora_basic_api_check=True,
        run_lora_dynamic_switch_check=True,
        run_multi_lora_api_check=True,
    ),
    DiffusionTestCase(
        "cosmos3_nano_t2i",
        DiffusionServerArgs(
            model_path=DEFAULT_COSMOS3_NANO_MODEL_NAME_FOR_TEST,
            modality="image",
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
    # === Text and Image to Image (TI2I) ===
    DiffusionTestCase(
        "qwen_image_edit_ti2i",
        DiffusionServerArgs(model_path=DEFAULT_QWEN_IMAGE_EDIT_MODEL_NAME_FOR_TEST),
    ),
    DiffusionTestCase(
        "qwen_image_edit_2509_ti2i",
        DiffusionServerArgs(
            model_path=DEFAULT_QWEN_IMAGE_EDIT_2509_MODEL_NAME_FOR_TEST,
        ),
        MULTI_IMAGE_TI2I_sampling_params,
    ),
    DiffusionTestCase(
        "qwen_image_edit_2511_ti2i",
        DiffusionServerArgs(
            model_path=DEFAULT_QWEN_IMAGE_EDIT_2511_MODEL_NAME_FOR_TEST,
        ),
    ),
    DiffusionTestCase(
        "qwen_image_layered_i2i",
        DiffusionServerArgs(
            model_path=DEFAULT_QWEN_IMAGE_LAYERED_MODEL_NAME_FOR_TEST,
        ),
        MULTI_FRAME_I2I_sampling_params,
    ),
    DiffusionTestCase(
        "joyai_image_edit_ti2i",
        DiffusionServerArgs(model_path=DEFAULT_JOYAI_IMAGE_EDIT_MODEL_NAME_FOR_TEST),
        run_consistency_check=False,
        run_component_accuracy_check=False,
    ),
    # Upscaling (Real-ESRGAN 4×) for T2I
    DiffusionTestCase(
        "flux_2_image_t2i_upscaling_4x",
        DiffusionServerArgs(
            model_path="black-forest-labs/FLUX.2-dev",
        ),
        DiffusionSamplingParams(
            prompt="Doraemon is eating dorayaki",
            output_size="1024x1024",
            extras={"enable_upscaling": True, "upscaling_scale": 4},
        ),
    ),
    # === Text to Video (T2V) ===
    DiffusionTestCase(
        "wan2_1_t2v_1.3b",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_1_T2V_1_3B_MODEL_NAME_FOR_TEST,
        ),
    ),
    DiffusionTestCase(
        "cosmos3_nano_t2v",
        DiffusionServerArgs(
            model_path=DEFAULT_COSMOS3_NANO_MODEL_NAME_FOR_TEST,
            modality="video",
            env_vars={"SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1"},
        ),
        DiffusionSamplingParams(
            prompt="A blue box slides across a clean warehouse floor.",
            output_size="832x480",
            seconds=1,
            num_frames=9,
            extras={
                "num_inference_steps": 4,
                "seed": 0,
                "max_sequence_length": 128,
                "flow_shift": 10.0,
                "use_guardrails": False,
                "use_duration_template": False,
                "use_resolution_template": False,
            },
        ),
        run_perf_check=False,
        run_consistency_check=True,
        run_component_accuracy_check=False,
    ),
    # TeaCache acceleration test for Wan video model
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_teacache_enabled",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_1_T2V_1_3B_MODEL_NAME_FOR_TEST,
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
            extras={"enable_teacache": True},
        ),
    ),
    # Frame interpolation (2× / exp=1)
    # Uses the same 1.3B model already in the suite;
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_frame_interp_2x",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
            extras={"enable_frame_interpolation": True, "frame_interpolation_exp": 1},
        ),
    ),
    # Upscaling (Real-ESRGAN 4×)
    # Uses the same 1.3B model already in the suite;
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_upscaling_4x",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
            extras={"enable_upscaling": True, "upscaling_scale": 4},
        ),
    ),
    # Combined: Frame interpolation (2×) + Upscaling (4×)
    # Verifies that both post-processing steps compose correctly.
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_frame_interp_2x_upscaling_4x",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
            extras={
                "enable_frame_interpolation": True,
                "frame_interpolation_exp": 1,
                "enable_upscaling": True,
                "upscaling_scale": 4,
            },
        ),
    ),
    # LoRA test case for single transformer + merge/unmerge API test
    # Note: Uses dynamic_lora_path instead of lora_path to test LayerwiseOffload + set_lora interaction
    # Server starts WITHOUT LoRA, then set_lora is called after startup (Wan models auto-enable layerwise offload)
    DiffusionTestCase(
        "wan2_1_t2v_1_3b_lora_1gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_1_T2V_1_3B_MODEL_NAME_FOR_TEST,
            num_gpus=1,
            dynamic_lora_path="Cseti/Wan-LoRA-Arcane-Jinx-v1",
        ),
        DiffusionSamplingParams(
            prompt="csetiarcane Nfj1nx with blue hair, a woman walking in a cyberpunk city at night",
        ),
        run_lora_basic_api_check=True,
        run_lora_dynamic_load_check=True,
    ),
    # NOTE(mick): flaky
    # DiffusionTestCase(
    #     "hunyuan_video",
    #     DiffusionServerArgs(
    #         model_path="hunyuanvideo-community/HunyuanVideo",
    #     ),
    #     DiffusionSamplingParams(
    #         prompt=T2V_PROMPT,
    #     ),
    # ),
    DiffusionTestCase(
        "flux_2_ti2i",
        DiffusionServerArgs(model_path=DEFAULT_FLUX_2_DEV_MODEL_NAME_FOR_TEST),
    ),
    DiffusionTestCase(
        "flux_2_t2i_customized_vae_path",
        DiffusionServerArgs(
            model_path=DEFAULT_FLUX_2_DEV_MODEL_NAME_FOR_TEST,
            extras=["--vae-path=fal/FLUX.2-Tiny-AutoEncoder"],
        ),
        T2I_sampling_params,
        run_perf_check=False,
    ),
    DiffusionTestCase(
        "fast_hunyuan_video",
        DiffusionServerArgs(
            model_path="FastVideo/FastHunyuan-diffusers",
        ),
    ),
    # === Text and Image to Video (TI2V) ===
    DiffusionTestCase(
        "wan2_2_ti2v_5b",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_2_TI2V_5B_MODEL_NAME_FOR_TEST,
        ),
    ),
    DiffusionTestCase(
        "fastwan2_2_ti2v_5b",
        DiffusionServerArgs(
            model_path="FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        ),
    ),
    DiffusionTestCase(
        "sana_wm_ti2v",
        DiffusionServerArgs(
            model_path=DEFAULT_SANA_WM_STREAMING_MODEL_NAME_FOR_TEST,
        ),
        SANA_WM_TI2V_CI_sampling_params,
        run_perf_check=False,
        run_component_accuracy_check=False,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    ),
    # flaky
    # === Helios T2V ===
    # DiffusionTestCase(
    #     "helios_base_t2v",
    #     DiffusionServerArgs(
    #         model_path="BestWishYsh/Helios-Base",
    #     ),
    #     DiffusionSamplingParams(
    #         prompt=T2V_PROMPT,
    #         output_size="640x384",
    #         num_frames=33,
    #     ),
    # ),
    # DiffusionTestCase(
    #     "helios_mid_t2v",
    #     DiffusionServerArgs(
    #         model_path="BestWishYsh/Helios-Mid",
    #     ),
    #     DiffusionSamplingParams(
    #         prompt=T2V_PROMPT,
    #         output_size="640x384",
    #         num_frames=33,
    #     ),
    # ),
    # DiffusionTestCase(
    #     "helios_distilled_t2v",
    #     DiffusionServerArgs(
    #         model_path="BestWishYsh/Helios-Distilled",
    #     ),
    #     DiffusionSamplingParams(
    #         prompt=T2V_PROMPT,
    #         output_size="640x384",
    #         num_frames=33,
    #     ),
    # ),
    DiffusionTestCase(
        "ltx_2_3_hq_pipeline",
        DiffusionServerArgs(
            model_path="Lightricks/LTX-2.3",
            extras=[
                "--pipeline-class-name LTX2TwoStageHQPipeline --ltx2-two-stage-device-mode original"
            ],
            env_vars={
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
        ),
        run_component_accuracy_check=False,
    ),
    DiffusionTestCase(
        "lingbot_world_realtime_plastic_beach",
        DiffusionServerArgs(
            model_path="robbyant/lingbot-world-fast-diffusers",
            modality="video",
            num_gpus=1,
            extras=[
                "--pipeline-class-name LingBotWorldCausalDMDPipeline --warmup false"
            ],
            text_encoder_cpu_offload=True,
        ),
        LINGBOT_WORLD_REALTIME_sampling_params,
        run_component_accuracy_check=False,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    ),
]

# Skip hunyuan3d on AMD: marching_cubes surface extraction produces invalid SDF on ROCm.
if not current_platform.is_hip():
    ONE_GPU_CASES.append(
        DiffusionTestCase(
            "hunyuan3d_shape_gen",
            DiffusionServerArgs(
                model_path="tencent/Hunyuan3D-2",
            ),
            run_consistency_check=False,
        ),
    )
# Skip turbowan on AMD: Triton requires 81920 shared memory, but AMD only has 65536.
if not current_platform.is_hip():
    ONE_GPU_CASES.append(
        DiffusionTestCase(
            "turbo_wan2_1_t2v_1.3b",
            DiffusionServerArgs(
                model_path="IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers",
            ),
        )
    )
# Skip all ModelOpt tests on AMD: FP8 requires torch._scaled_mm (HIPBLAS_STATUS_NOT_SUPPORTED
# on ROCm), NVFP4 requires flashinfer or sgl_kernel FP4 kernels (CUDA-only).
# Run FP8 cases on the regular H100 1-GPU CI shard and keep only B200-only
# quantization coverage in the B200 suite.
if current_platform.is_hip():
    ONE_GPU_MODELOPT_FP8_CASES = []
    ONE_GPU_MODELOPT_NVFP4_CASES = []
else:
    ONE_GPU_MODELOPT_FP8_CASES = [
        _make_modelopt_ci_case(
            "flux1_modelopt_fp8_t2i",
            model_path=DEFAULT_FLUX_1_DEV_MODEL_NAME_FOR_TEST,
            modality="image",
            sampling_params=MODELOPT_T2I_CI_sampling_params,
            extras=["--transformer-path", MODELOPT_FLUX1_FP8_TRANSFORMER],
            run_consistency_check=True,
        ),
        _make_modelopt_ci_case(
            "flux2_modelopt_fp8_t2i",
            model_path=DEFAULT_FLUX_2_DEV_MODEL_NAME_FOR_TEST,
            modality="image",
            sampling_params=MODELOPT_T2I_CI_sampling_params,
            extras=["--transformer-path", MODELOPT_FLUX2_FP8_TRANSFORMER],
            run_consistency_check=True,
        ),
        _make_modelopt_ci_case(
            "wan22_modelopt_fp8_t2v",
            model_path=MODELOPT_WAN22_FP8_MODEL,
            modality="video",
            sampling_params=MODELOPT_T2V_CI_sampling_params,
            extras=[],
            run_consistency_check=True,
        ),
        _make_modelopt_ci_case(
            "hunyuanvideo_modelopt_fp8_t2v",
            model_path="hunyuanvideo-community/HunyuanVideo",
            modality="video",
            sampling_params=MODELOPT_T2V_CI_sampling_params,
            extras=[
                "--transformer-path",
                MODELOPT_HUNYUANVIDEO_FP8_TRANSFORMER,
                "--text-encoder-cpu-offload",
                "--pin-cpu-memory",
            ],
        ),
        _make_modelopt_ci_case(
            "qwen_image_modelopt_fp8_t2i",
            model_path=DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
            modality="image",
            sampling_params=MODELOPT_T2I_CI_sampling_params,
            extras=["--transformer-path", MODELOPT_QWEN_IMAGE_FP8_TRANSFORMER],
        ),
        _make_modelopt_ci_case(
            "qwen_image_edit_modelopt_fp8_ti2i",
            model_path=DEFAULT_QWEN_IMAGE_EDIT_2511_MODEL_NAME_FOR_TEST,
            modality="image",
            sampling_params=MODELOPT_TI2I_CI_sampling_params,
            extras=["--transformer-path", MODELOPT_QWEN_IMAGE_EDIT_FP8_TRANSFORMER],
        ),
    ]
    ONE_GPU_MODELOPT_NVFP4_CASES = [
        _make_modelopt_ci_case(
            "flux1_modelopt_nvfp4_t2i",
            model_path=DEFAULT_FLUX_1_DEV_MODEL_NAME_FOR_TEST,
            modality="image",
            sampling_params=MODELOPT_T2I_CI_sampling_params,
            extras=["--transformer-path", MODELOPT_FLUX1_NVFP4_TRANSFORMER],
            env_vars=MODELOPT_NVFP4_B200_ENV_VARS,
            run_consistency_check=True,
        ),
        _make_modelopt_ci_case(
            "flux2_modelopt_nvfp4_t2i",
            model_path=DEFAULT_FLUX_2_DEV_MODEL_NAME_FOR_TEST,
            modality="image",
            sampling_params=MODELOPT_T2I_CI_sampling_params,
            extras=["--transformer-weights-path", MODELOPT_FLUX2_NVFP4_WEIGHTS],
            env_vars=MODELOPT_NVFP4_B200_ENV_VARS,
            run_consistency_check=True,
        ),
        _make_modelopt_ci_case(
            "ideogram4_nvfp4_t2i",
            model_path="Comfy-Org/Ideogram-4",
            modality="image",
            sampling_params=IDEOGRAM4_CI_sampling_params,
            extras=[],
            env_vars=MODELOPT_NVFP4_B200_ENV_VARS,
            run_consistency_check=True,
        ),
        _make_modelopt_ci_case(
            "wan22_modelopt_nvfp4_t2v",
            model_path=MODELOPT_WAN22_NVFP4_MODEL,
            modality="video",
            sampling_params=MODELOPT_T2V_CI_sampling_params,
            extras=[],
            env_vars=MODELOPT_WAN22_NVFP4_B200_ENV_VARS,
            run_consistency_check=True,
        ),
    ]

ONE_GPU_B200_CASES = ONE_GPU_MODELOPT_NVFP4_CASES

TWO_GPU_CASES = [
    DiffusionTestCase(
        "ideogram4_fp8_tp2_t2i",
        DiffusionServerArgs(
            model_path="ideogram-ai/ideogram-4-fp8",
            tp_size=2,
            extras=[
                "--attention-backend",
                "fa",
            ],
        ),
        IDEOGRAM4_CI_sampling_params,
        run_perf_check=False,
        run_consistency_check=False,
        run_component_accuracy_check=False,
    ),
    DiffusionTestCase(
        "wan2_2_i2v_a14b_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_2_I2V_A14B_MODEL_NAME_FOR_TEST,
        ),
    ),
    DiffusionTestCase(
        "wan2_2_t2v_a14b_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_2_T2V_A14B_MODEL_NAME_FOR_TEST,
            extras=["--ulysses-degree=2"],
        ),
    ),
    # TeaCache bring-up test for Wan2.2 T2V A14B — verifies enable_teacache=True
    # doesn't crash. Perf check disabled because Wan2.2-specific TeaCache
    # coefficients are not yet calibrated (teacache_params=None, so no speedup).
    DiffusionTestCase(
        "wan2_2_t2v_a14b_teacache_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_2_T2V_A14B_MODEL_NAME_FOR_TEST,
            extras=["--ulysses-degree=2"],
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
            extras={"enable_teacache": True},
        ),
        run_perf_check=False,
    ),
    # LoRA test case for transformer_2 support
    DiffusionTestCase(
        "wan2_2_t2v_a14b_lora_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_2_T2V_A14B_MODEL_NAME_FOR_TEST,
            lora_path="Cseti/wan2.2-14B-Arcane_Jinx-lora-v1",
            extras=[
                "--lora-weight-name",
                "985347-wan22_14B-low-Nfj1nx-e65.safetensors",
                "--lora-merge-mode",
                "dynamic",
            ],
        ),
        DiffusionSamplingParams(
            prompt="Nfj1nx with blue hair, a woman walking in a cyberpunk city at night",
        ),
        run_lora_basic_api_check=True,
    ),
    DiffusionTestCase(
        "wan2_1_t2v_14b_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_1_T2V_14B_MODEL_NAME_FOR_TEST,
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
            output_size="832x480",
        ),
    ),
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_cfg_parallel",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_1_T2V_1_3B_MODEL_NAME_FOR_TEST,
            cfg_parallel=True,
        ),
    ),
    DiffusionTestCase(
        "wan2_1_t2v_1_3b_cache_dit_sp_only_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_1_T2V_1_3B_MODEL_NAME_FOR_TEST,
            ulysses_degree=2,
            enable_cache_dit=True,
            env_vars={"SGLANG_CACHE_DIT_WARMUP": "2"},
        ),
        replace(
            T2V_sampling_params,
            output_size="832x480",
            num_frames=5,
            extras={"num_inference_steps": 8, "seed": 0},
        ),
        run_perf_check=False,
        run_consistency_check=False,
        run_component_accuracy_check=False,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    ),
    DiffusionTestCase(
        "fsdp-inference",
        DiffusionServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            extras=["--use-fsdp-inference"],
        ),
    ),
    DiffusionTestCase(
        "mova_360p_tp2",
        DiffusionServerArgs(
            model_path=DEFAULT_MOVA_360P_MODEL_NAME_FOR_TEST,
            tp_size=2,
            dit_layerwise_offload=True,
        ),
        run_perf_check=False,
    ),
    DiffusionTestCase(
        "mova_360p_ring1_uly2",
        DiffusionServerArgs(
            model_path=DEFAULT_MOVA_360P_MODEL_NAME_FOR_TEST,
            ring_degree=1,
            ulysses_degree=2,
            dit_layerwise_offload=True,
        ),
        run_perf_check=False,
    ),
    DiffusionTestCase(
        "ltx_2_two_stage_t2v",
        DiffusionServerArgs(
            model_path="Lightricks/LTX-2",
            cfg_parallel=True,
            extras=["--pipeline-class-name LTX2TwoStagePipeline"],
        ),
        T2V_sampling_params,
    ),
    DiffusionTestCase(
        "ltx_2_3_two_stage_ti2v_2gpus",
        DiffusionServerArgs(
            model_path="Lightricks/LTX-2.3",
            cfg_parallel=True,
            extras=[
                "--pipeline-class-name LTX2TwoStagePipeline --ltx2-two-stage-device-mode original",
            ],
        ),
        run_component_accuracy_check=False,
    ),
    DiffusionTestCase(
        "wan2_1_i2v_14b_480P_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_1_I2V_14B_480P_MODEL_NAME_FOR_TEST,
            extras=["--ulysses-degree=2"],
        ),
    ),
    DiffusionTestCase(
        "ltx_2.3_two_stage_t2v_2gpus",
        DiffusionServerArgs(
            model_path="Lightricks/LTX-2.3",
            cfg_parallel=True,
            extras=[
                "--pipeline-class-name LTX2TwoStagePipeline",
                "--component-attention-backends transformer=fa",
            ],
        ),
        DiffusionSamplingParams(prompt=T2V_PROMPT, extras={"seed": 42}),
        run_component_accuracy_check=False,
    ),
    # I2V LoRA test case
    DiffusionTestCase(
        "wan2_1_i2v_14b_lora_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_1_I2V_14B_720P_MODEL_NAME_FOR_TEST,
            lora_path="starsfriday/Wan2.1-Divine-Power-LoRA",
            extras=["--ulysses-degree=2"],
        ),
        run_lora_basic_api_check=True,
    ),
    DiffusionTestCase(
        "wan2_1_i2v_14b_720P_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_1_I2V_14B_720P_MODEL_NAME_FOR_TEST,
            extras=["--ulysses-degree=2"],
        ),
    ),
    DiffusionTestCase(
        "qwen_image_t2i_2_gpus",
        DiffusionServerArgs(
            model_path=DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
            # test ring attn
            ulysses_degree=1,
            ring_degree=2,
        ),
    ),
    DiffusionTestCase(
        "zimage_image_t2i_2_gpus",
        DiffusionServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            ulysses_degree=2,
        ),
    ),
    DiffusionTestCase(
        "zimage_image_t2i_2_gpus_non_square",
        DiffusionServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            ulysses_degree=2,
        ),
        DiffusionSamplingParams(
            prompt=T2I_sampling_params.prompt,
            output_size="1280x720",
        ),
        run_perf_check=False,
    ),
    DiffusionTestCase(
        "flux_image_t2i_2_gpus",
        DiffusionServerArgs(
            model_path=DEFAULT_FLUX_1_DEV_MODEL_NAME_FOR_TEST,
        ),
    ),
    DiffusionTestCase(
        "flux_2_image_t2i_2_gpus",
        DiffusionServerArgs(
            model_path=DEFAULT_FLUX_2_DEV_MODEL_NAME_FOR_TEST,
            tp_size=2,
        ),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "ltx_2.3_one_stage_ti2v",
        DiffusionServerArgs(
            model_path="Lightricks/LTX-2.3",
            cfg_parallel=True,
        ),
        run_component_accuracy_check=False,
    ),
]

if not current_platform.is_hip():
    # Flux2 multi-image edit with cache-dit, regression test
    ONE_GPU_CASES.append(
        DiffusionTestCase(
            "flux_2_ti2i_multi_image_cache_dit",
            DiffusionServerArgs(
                model_path="black-forest-labs/FLUX.2-dev",
                enable_cache_dit=True,
            ),
            MULTI_IMAGE_TI2I_UPLOAD_sampling_params,
        )
    )

ONE_GPU_CASES += ONE_GPU_MODELOPT_FP8_CASES
TWO_GPU_CASES = _with_default_num_gpus(TWO_GPU_CASES, 2)


def _discover_unit_tests() -> list[str]:
    unit_dir = Path(__file__).resolve().parent.parent / "unit"
    if not unit_dir.is_dir():
        return []
    return sorted(
        f"../unit/{f.name}" for f in unit_dir.glob("test_*.py") if f.is_file()
    )


FILE_SUITES = {
    "unit": _discover_unit_tests(),
    "component-accuracy": [
        "test_component_accuracy_1_gpu.py",
        "test_component_accuracy_2_gpu.py",
    ],
    "component-accuracy-1-gpu": [
        "test_component_accuracy_1_gpu.py",
    ],
    "component-accuracy-2-gpu": [
        "test_component_accuracy_2_gpu.py",
    ],
    "1-gpu-b200": [
        "test_server_b200.py",
    ],
}

PARAMETRIZED_CASE_GROUPS = {
    "1-gpu": [
        ("test_server_1_gpu.py", ONE_GPU_CASES),
    ],
    "2-gpu": [
        ("test_server_2_gpu.py", TWO_GPU_CASES),
    ],
}

STANDALONE_FILES = {
    "1-gpu": [
        "../cli/test_generate_t2i_perf.py",
        # Temporarily disabled: 24 timeout failures since 2026-04-09 across
        # multimodal-gen-test-1-gpu. Re-enable after the flakiness is fixed.
        # "test_update_weights_from_disk.py",
    ],
    "2-gpu": [
        "test_disagg_server.py",
    ],
}

# New standalone files may omit an estimate once to learn the real CI runtime.
# CI will use a fallback estimate for sharding, run the test, then print a
# measured value that must be copied into STANDALONE_FILE_EST_TIMES.
STANDALONE_FILE_EST_TIMES = {
    "1-gpu": {
        "../cli/test_generate_t2i_perf.py": 240.0,
        # See STANDALONE_FILES note above — temporarily disabled.
        # "test_update_weights_from_disk.py": 480.0,
    },
    "2-gpu": {
        # Two disagg clusters × (~3 min startup + ~1 min generate) ≈ 8 min.
        # Raise if CI reports a higher measured time.
        "test_disagg_server.py": 600.0,
    },
}

# Backward-compatible suite view for scripts that still operate on file lists.
SUITES = {
    **FILE_SUITES,
    **{
        suite: [filename for filename, _ in case_groups]
        + STANDALONE_FILES.get(suite, [])
        for suite, case_groups in PARAMETRIZED_CASE_GROUPS.items()
    },
}

STRICT_SUITES = {"unit"}
COMPONENT_ACCURACY_SUITES = {
    "component-accuracy",
    "component-accuracy-1-gpu",
    "component-accuracy-2-gpu",
}
COMPONENT_ACCURACY_FILE_NUM_GPUS = {
    "test_component_accuracy_1_gpu.py": 1,
    "test_component_accuracy_2_gpu.py": 2,
}

DEFAULT_EST_TIME_SECONDS = 300.0
STARTUP_OVERHEAD_SECONDS = 120.0
DEFAULT_STANDALONE_EST_TIME_SECONDS = 300.0

_UPDATE_WEIGHTS_FROM_DISK_TEST_FILE = "test_update_weights_from_disk.py"
_UPDATE_WEIGHTS_MODEL_PAIR_ENV = "SGLANG_MMGEN_UPDATE_WEIGHTS_PAIR"
_UPDATE_WEIGHTS_MODEL_PAIR_IDS = (
    "FLUX.2-klein-base-4B",
    "Qwen-Image",
)
