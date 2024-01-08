"""Global configurations"""


class GlobalConfig:
    def __init__(self):
        # Verbosity level
        # 0: do not output anything
        # 2: output final text after every run
        self.verbosity = 0

        self.default_backend = None

        # Output configs
        self.skip_special_tokens_in_output = True

        # Optimization configs
        self.eager_fill_image = False
        self.enable_prefix_sharing = True
        self.enable_parallel_encoding = True
        self.enable_parallel_decoding = True

        # Choices: ["no_adjust", "adjust_cache"]
        # no_adjust: Do not adjust the position embedding of KV cache.
        # adjust_cache: Adjust the position embedding of KV cache.
        self.concate_and_append_mode = "no_adjust"


global_config = GlobalConfig()
