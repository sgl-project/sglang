import os

from sglang_simulator.utils.logger import get_logger

logger = get_logger("sgl_simulator")


class Envs:
    @classmethod
    def config_path(cls) -> str:
        SGLANG_SIMULATOR_CONFIG_PATH = os.getenv("SGLANG_SIMULATOR_CONFIG_PATH")
        if not SGLANG_SIMULATOR_CONFIG_PATH or not os.path.exists(
            SGLANG_SIMULATOR_CONFIG_PATH
        ):
            raise RuntimeError(
                f"The mock configuration path is not set or does not exist({SGLANG_SIMULATOR_CONFIG_PATH}). Please set it using the system variable SGLANG_SIMULATOR_CONFIG_PATH"
            )
        return SGLANG_SIMULATOR_CONFIG_PATH

    @classmethod
    def output_dir(cls) -> str:
        SGLANG_SIMULATOR_OUTPUT_DIR = os.getenv(
            "SGLANG_SIMULATOR_OUTPUT_DIR", "/tmp/sgl_simulator/output/"
        )
        SGLANG_SIMULATOR_OUTPUT_DIR = os.path.realpath(SGLANG_SIMULATOR_OUTPUT_DIR)
        if os.path.exists(SGLANG_SIMULATOR_OUTPUT_DIR) and os.path.isfile(
            SGLANG_SIMULATOR_OUTPUT_DIR
        ):
            logger.error(
                f"The metrics output path, {SGLANG_SIMULATOR_OUTPUT_DIR}, exists and is a file."
            )
            raise RuntimeError(
                f"{SGLANG_SIMULATOR_OUTPUT_DIR} exists but is not a directory."
            )
        os.makedirs(os.path.dirname(SGLANG_SIMULATOR_OUTPUT_DIR), exist_ok=True)
        return SGLANG_SIMULATOR_OUTPUT_DIR

    @classmethod
    def simulation_mode(cls) -> str:
        SGLANG_SIMULATOR_OUTPUT_MODE = os.getenv(
            "SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE"
        ).upper()
        assert SGLANG_SIMULATOR_OUTPUT_MODE in ("BLOCKING", "OFFLINE")
        return SGLANG_SIMULATOR_OUTPUT_MODE

    @classmethod
    def num_warmup(cls) -> int:
        # The number of warmup requests.
        return int(os.getenv("SGLANG_SIMULATOR_NUM_WARMUP", "0"))
