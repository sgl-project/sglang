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
    MODELOPT_WAN22_FP8_TRANSFORMER,
    MODELOPT_WAN22_NVFP4_TRANSFORMER,
    T2V_PROMPT,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    HUNYUAN3D_SHAPE_sampling_params,
    MODELOPT_T2I_CI_sampling_params,
    MODELOPT_T2V_CI_sampling_params,
    MODELOPT_TI2I_CI_sampling_params,
    MULTI_FRAME_I2I_sampling_params,
    MULTI_IMAGE_TI2I_sampling_params,
    MULTI_IMAGE_TI2I_UPLOAD_sampling_params,
    T2I_sampling_params,
    T2V_sampling_params,
    TI2I_sampling_params,
    TI2V_sampling_params,
    _make_modelopt_ci_case,
    _with_default_num_gpus,
)
from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_FLUX_1_DEV_MODEL_NAME_FOR_TEST,
    DEFAULT_FLUX_2_DEV_MODEL_NAME_FOR_TEST,
    DEFAULT_FLUX_2_KLEIN_4B_MODEL_NAME_FOR_TEST,
    DEFAULT_JOYAI_IMAGE_EDIT_MODEL_NAME_FOR_TEST,
    DEFAULT_MOVA_360P_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_EDIT_2509_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_EDIT_2511_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_EDIT_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_LAYERED_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_1_I2V_14B_480P_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_1_I2V_14B_720P_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_1_T2V_1_3B_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_1_T2V_14B_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_2_I2V_A14B_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_2_T2V_A14B_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_2_TI2V_5B_MODEL_NAME_FOR_TEST,
)

# All test cases with clean default values
# To test different models, simply add more DiffusionCase entries
ONE_GPU_CASES: list[DiffusionTestCase] = [
    # === Text to Image (T2I) ===
    DiffusionTestCase(
        "qwen_image_t2i",
        DiffusionServerArgs(
            model_path=DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
        ),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "qwen_image_t2i_cache_dit_enabled",
        DiffusionServerArgs(
            model_path=DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
            enable_cache_dit=True,
        ),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "flux_image_t2i",
        DiffusionServerArgs(model_path=DEFAULT_FLUX_1_DEV_MODEL_NAME_FOR_TEST),
        T2I_sampling_params,
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
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "zimage_image_t2i",
        DiffusionServerArgs(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "zimage_image_t2i_fp8",
        DiffusionServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            extras=["--transformer-path MickJ/Z-Image-Turbo-fp8"],
        ),
        T2I_sampling_params,
    ),
    # Multi-LoRA test case for Z-Image-Turbo
    DiffusionTestCase(
        "zimage_image_t2i_multi_lora",
        DiffusionServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            lora_path="reverentelusarca/elusarca-anime-style-lora-z-image-turbo",
            second_lora_path="tarn59/pixel_art_style_lora_z_image_turbo",
        ),
        T2I_sampling_params,
        run_lora_basic_api_check=True,
        run_lora_dynamic_switch_check=True,
        run_multi_lora_api_check=True,
    ),
    # === Text and Image to Image (TI2I) ===
    DiffusionTestCase(
        "qwen_image_edit_ti2i",
        DiffusionServerArgs(model_path=DEFAULT_QWEN_IMAGE_EDIT_MODEL_NAME_FOR_TEST),
        TI2I_sampling_params,
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
        TI2I_sampling_params,
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
        TI2I_sampling_params,
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
        T2V_sampling_params,
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
        TI2I_sampling_params,
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
        T2V_sampling_params,
    ),
    # === Text and Image to Video (TI2V) ===
    DiffusionTestCase(
        "wan2_2_ti2v_5b",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_2_TI2V_5B_MODEL_NAME_FOR_TEST,
        ),
        TI2V_sampling_params,
    ),
    DiffusionTestCase(
        "fastwan2_2_ti2v_5b",
        DiffusionServerArgs(
            model_path="FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        ),
        TI2V_sampling_params,
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
                "--pipeline-class-name LTX2TwoStageHQPipeline --ltx2-two-stage-device-mode snapshot"
            ],
            env_vars={
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "SGLANG_LTX2_SNAPSHOT_RELEASE_EMPTY_CACHE": "true",
            },
        ),
        T2I_sampling_params,
        run_component_accuracy_check=False,
    ),
]

# Skip hunyuan3d on AMD: marching_cubes surface extraction produces invalid SDF on ROCm.
if not current_platform.is_hip():
    ONE_GPU_CASES.append(
        DiffusionTestCase(
            "hunyuan3d_shape_gen",
            DiffusionServerArgs(
                model_path="tencent/Hunyuan3D-2",
                enable_warmup=False,
            ),
            HUNYUAN3D_SHAPE_sampling_params,
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
            T2V_sampling_params,
        )
    )
# Skip all ModelOpt tests on AMD: FP8 requires torch._scaled_mm (HIPBLAS_STATUS_NOT_SUPPORTED
# on ROCm), NVFP4 requires flashinfer or sgl_kernel FP4 kernels (CUDA-only)
if current_platform.is_hip():
    ONE_GPU_MODELOPT_CASES = []
else:
    ONE_GPU_MODELOPT_CASES = [
        _make_modelopt_ci_case(
            "flux1_modelopt_fp8_t2i",
            model_path=DEFAULT_FLUX_1_DEV_MODEL_NAME_FOR_TEST,
            modality="image",
            sampling_params=MODELOPT_T2I_CI_sampling_params,
            extras=["--transformer-path", MODELOPT_FLUX1_FP8_TRANSFORMER],
        ),
        _make_modelopt_ci_case(
            "flux2_modelopt_fp8_t2i",
            model_path=DEFAULT_FLUX_2_DEV_MODEL_NAME_FOR_TEST,
            modality="image",
            sampling_params=MODELOPT_T2I_CI_sampling_params,
            extras=["--transformer-path", MODELOPT_FLUX2_FP8_TRANSFORMER],
        ),
        _make_modelopt_ci_case(
            "wan22_modelopt_fp8_t2v",
            model_path=DEFAULT_WAN_2_2_T2V_A14B_MODEL_NAME_FOR_TEST,
            modality="video",
            sampling_params=MODELOPT_T2V_CI_sampling_params,
            extras=["--transformer-path", MODELOPT_WAN22_FP8_TRANSFORMER],
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
        _make_modelopt_ci_case(
            "flux1_modelopt_nvfp4_t2i",
            model_path=DEFAULT_FLUX_1_DEV_MODEL_NAME_FOR_TEST,
            modality="image",
            sampling_params=MODELOPT_T2I_CI_sampling_params,
            extras=["--transformer-path", MODELOPT_FLUX1_NVFP4_TRANSFORMER],
            env_vars=MODELOPT_NVFP4_B200_ENV_VARS,
        ),
        _make_modelopt_ci_case(
            "flux2_modelopt_nvfp4_t2i",
            model_path=DEFAULT_FLUX_2_DEV_MODEL_NAME_FOR_TEST,
            modality="image",
            sampling_params=MODELOPT_T2I_CI_sampling_params,
            extras=["--transformer-weights-path", MODELOPT_FLUX2_NVFP4_WEIGHTS],
            env_vars=MODELOPT_NVFP4_B200_ENV_VARS,
        ),
        _make_modelopt_ci_case(
            "wan22_modelopt_nvfp4_t2v",
            model_path=DEFAULT_WAN_2_2_T2V_A14B_MODEL_NAME_FOR_TEST,
            modality="video",
            sampling_params=MODELOPT_T2V_CI_sampling_params,
            extras=["--transformer-path", MODELOPT_WAN22_NVFP4_TRANSFORMER],
            env_vars=MODELOPT_NVFP4_B200_ENV_VARS,
        ),
    ]

TWO_GPU_CASES = [
    DiffusionTestCase(
        "wan2_2_i2v_a14b_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_2_I2V_A14B_MODEL_NAME_FOR_TEST,
        ),
        TI2V_sampling_params,
    ),
    DiffusionTestCase(
        "wan2_2_t2v_a14b_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_2_T2V_A14B_MODEL_NAME_FOR_TEST,
            extras=["--ulysses-degree=2"],
        ),
        T2V_sampling_params,
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
        T2V_sampling_params,
    ),
    DiffusionTestCase(
        "fsdp-inference",
        DiffusionServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            extras=["--use-fsdp-inference"],
        ),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "mova_360p_tp2",
        DiffusionServerArgs(
            model_path=DEFAULT_MOVA_360P_MODEL_NAME_FOR_TEST,
            tp_size=2,
            dit_layerwise_offload=True,
        ),
        TI2V_sampling_params,
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
        TI2V_sampling_params,
        run_perf_check=False,
    ),
    DiffusionTestCase(
        "ltx_2_two_stage_t2v",
        DiffusionServerArgs(
            model_path="Lightricks/LTX-2",
            ulysses_degree=2,
            dit_layerwise_offload=True,
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
                "--pipeline-class-name LTX2TwoStagePipeline --ltx2-two-stage-device-mode original"
            ],
        ),
        TI2V_sampling_params,
        run_component_accuracy_check=False,
    ),
    DiffusionTestCase(
        "wan2_1_i2v_14b_480P_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_1_I2V_14B_480P_MODEL_NAME_FOR_TEST,
            extras=["--ulysses-degree=2"],
        ),
        TI2V_sampling_params,
    ),
    DiffusionTestCase(
        "ltx_2.3_two_stage_t2v_2gpus",
        DiffusionServerArgs(
            model_path="Lightricks/LTX-2.3",
            cfg_parallel=True,
            extras=[
                "--pipeline-class-name LTX2TwoStagePipeline",
                "--ltx2-two-stage-device-mode original",
            ],
        ),
        T2V_sampling_params,
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
        TI2V_sampling_params,
        run_lora_basic_api_check=True,
    ),
    DiffusionTestCase(
        "wan2_1_i2v_14b_720P_2gpu",
        DiffusionServerArgs(
            model_path=DEFAULT_WAN_2_1_I2V_14B_720P_MODEL_NAME_FOR_TEST,
            extras=["--ulysses-degree=2"],
        ),
        TI2V_sampling_params,
    ),
    DiffusionTestCase(
        "qwen_image_t2i_2_gpus",
        DiffusionServerArgs(
            model_path=DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
            # test ring attn
            ulysses_degree=1,
            ring_degree=2,
        ),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "zimage_image_t2i_2_gpus",
        DiffusionServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            ulysses_degree=2,
        ),
        T2I_sampling_params,
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
        T2I_sampling_params,
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
            ulysses_degree=2,
        ),
        TI2V_sampling_params,
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

ONE_GPU_CASES += ONE_GPU_MODELOPT_CASES
TWO_GPU_CASES = _with_default_num_gpus(TWO_GPU_CASES, 2)
