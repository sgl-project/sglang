import signal

from deep_gemm.deep_gemm_tuner.autotune_deepgemm import (
    tuning_deepgemm_config_entrypoint,
    tuning_deepgemm_model_config,
)

from .utils import (
    check_main_thread,
    get_deep_gemm_config,
    get_default_deep_gemm_dump_file,
    handle_deepgemm_signal,
)

if check_main_thread():
    # As we are in a child process, they will receive a SIGINT signal as long as its parent exit
    signal.signal(signal.SIGINT, handle_deepgemm_signal)
