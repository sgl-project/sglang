from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.test.server.testcase_configs import (
    HUNYUAN3D_SHAPE_sampling_params,
    DiffusionServerArgs,
    DiffusionTestCase,
    T2I_sampling_params,
    T2V_sampling_params,
    TI2I_sampling_params,
    TI2V_sampling_params,
    get_default_sampling_params_for_model_task,
)


def test_default_sampling_params_follow_model_task_type():
    assert (
        get_default_sampling_params_for_model_task(ModelTaskType.T2I)
        is T2I_sampling_params
    )
    assert (
        get_default_sampling_params_for_model_task(ModelTaskType.I2I)
        is TI2I_sampling_params
    )
    assert (
        get_default_sampling_params_for_model_task(ModelTaskType.TI2I)
        is TI2I_sampling_params
    )
    assert (
        get_default_sampling_params_for_model_task(ModelTaskType.T2V)
        is T2V_sampling_params
    )
    assert (
        get_default_sampling_params_for_model_task(ModelTaskType.I2V)
        is TI2V_sampling_params
    )
    assert (
        get_default_sampling_params_for_model_task(ModelTaskType.TI2V)
        is TI2V_sampling_params
    )
    assert (
        get_default_sampling_params_for_model_task(ModelTaskType.I2M)
        is HUNYUAN3D_SHAPE_sampling_params
    )


def test_test_case_infers_sampling_params_from_explicit_pipeline_class():
    case = DiffusionTestCase(
        "ltx_2_3_hq_pipeline",
        DiffusionServerArgs(
            model_path="Lightricks/LTX-2.3",
            modality="video",
            extras=[
                "--pipeline-class-name LTX2TwoStageHQPipeline "
                "--ltx2-two-stage-device-mode snapshot"
            ],
        ),
    )

    assert case.sampling_params is TI2V_sampling_params
