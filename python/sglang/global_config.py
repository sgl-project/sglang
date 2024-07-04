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
        self.spaces_between_special_tokens_in_out = True

        # Optimization configs
        self.eager_fill_image = False
        self.enable_precache_with_tracing = True
        self.enable_parallel_encoding = True
        self.enable_parallel_decoding = True

        # Choices: ["no_adjust", "adjust_cache"]
        # no_adjust: Do not adjust the position embedding of KV cache.
        # adjust_cache: Adjust the position embedding of KV cache.
        self.concate_and_append_mode = "no_adjust"

        # Request dependency time due to network delay
        self.request_dependency_delay = 0.02
        self.wait_for_new_request_delay = 0.0006

        # New generation token ratio estimation
        self.base_new_token_ratio = 0.4
        self.base_min_new_token_ratio = 0.2
        self.new_token_ratio_decay = 0.0001
        self.new_token_ratio_recovery = 0.05

        # The threshold (number of tokens) to trigger layer-wise cuda sync.
        # This can improve the speed for large batch sizes during prefill.
        self.layer_sync_threshold = 8192

global_config = GlobalConfig()
