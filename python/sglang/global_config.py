"""Global configurations"""

# FIXME: deprecate this file and move all usage to sglang.srt.environ or sglang.__init__.py


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

        # Output tokenization configs
        self.skip_special_tokens_in_output = True
        self.spaces_between_special_tokens_in_out = True

        # Language frontend interpreter optimization configs
        self.enable_precache_with_tracing = True
        self.enable_parallel_encoding = True


global_config = GlobalConfig()
