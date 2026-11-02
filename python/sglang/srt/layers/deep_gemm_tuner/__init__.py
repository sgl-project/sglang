"""SAIL DeepGEMM tuning entrypoint; importing it loads no vendor implementation and
installs no signal handler."""

from .utils import (
    get_deep_gemm_config,
    get_default_deep_gemm_dump_file,
    install_signal_handler,
)


def tuning_deepgemm_config_entrypoint(*args, **kwargs):
    from deep_gemm.deep_gemm_tuner.autotune_deepgemm import (
        tuning_deepgemm_config_entrypoint as tune,
    )

    return tune(*args, **kwargs)


def tuning_deepgemm_model_config(*args, **kwargs):
    from deep_gemm.deep_gemm_tuner.autotune_deepgemm import (
        tuning_deepgemm_model_config as tune,
    )

    return tune(*args, **kwargs)
