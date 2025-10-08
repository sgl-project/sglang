"""Global configurations"""

import os


class GlobalConfig:
    """
    Store some global constants.

    See also python/sglang/srt/managers/schedule_batch.py::global_server_args_dict, which stores
    many global runtime arguments as well.
    """

    def __init__(self):
        # Verbosity level
        # 0: do not output anything
        # 2: output final text after every run
        self.verbosity = 0

        # Default backend of the language
        self.default_backend = None

        # Runtime constants: New generation token ratio estimation
        self.torch_empty_cache_interval = float(
            os.environ.get(
                "SGLANG_EMPTY_CACHE_INTERVAL", -1
            )  # in seconds. Set if you observe high memory accumulation over a long serving period.
        )

        # Runtime constants: others
        self.flashinfer_workspace_size = int(
            os.environ.get("FLASHINFER_WORKSPACE_SIZE", 384 * 1024 * 1024)
        )

        # Output tokenization configs
        self.skip_special_tokens_in_output = True
        self.spaces_between_special_tokens_in_out = True

        # Language frontend interpreter optimization configs
        self.enable_precache_with_tracing = True
        self.enable_parallel_encoding = True


global_config = GlobalConfig()
