"""Global configurations"""

import os
from dataclasses import dataclass

@dataclass
class InfLLM2Switch:
    enable: bool = False
    backend: str = "triton"         # "triton" | "fa3" | "auto"
    stage1_impl: str = "triton"     # 预留："triton" | "cuda"
    stage2_impl: str = "triton"     # 预留："triton" | "fa3" | "cuda"
    topk: int = 8
    block_size: int = 64
    incremental: bool = True
    # 额外：滑窗/下沉配置（按 token 数）
    sw_span: int | None = 2048   # e.g. 2048 → 最后 2048 token 的窗口
    sink_len: int | None = 64      # e.g. 64   → 前 64 token 作为 sink

    @staticmethod
    def from_env():
        def _b(name, default=False):
            v = os.environ.get(name)
            return default if v is None else v.lower() in ("1","true","yes","on")
        def _s(name, default):
            return os.environ.get(name, default)
        def _i(name, default):
            try: return int(os.environ.get(name, default))
            except: return default
        def _opt_i(name):
            v = os.environ.get(name)
            if v is None or v == "":
                return None
            try:
                return int(v)
            except:
                return None
        return InfLLM2Switch(
            enable=_b("SGLANG_INFLLM_ENABLE", False),
            backend=_s("SGLANG_INFLLM_BACKEND", "triton"),
            stage1_impl=_s("SGLANG_INFLLM_STAGE1", "triton"),
            stage2_impl=_s("SGLANG_INFLLM_STAGE2", "triton"),
            topk=_i("SGLANG_INFLLM_TOPK", 8),
            block_size=_i("SGLANG_INFLLM_BLOCK", 64),
            incremental=_b("SGLANG_INFLLM_INCREMENTAL", True),
            sw_span=_opt_i("SGLANG_INFLLM_SW_SPAN"),
            sink_len=_opt_i("SGLANG_INFLLM_SINK_LEN"),
        )

INFLLM2_SWITCH = InfLLM2Switch.from_env()

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
        self.default_init_new_token_ratio = float(
            os.environ.get("SGLANG_INIT_NEW_TOKEN_RATIO", 0.7)
        )
        self.default_min_new_token_ratio_factor = float(
            os.environ.get("SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR", 0.14)
        )
        self.default_new_token_ratio_decay_steps = float(
            os.environ.get("SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS", 600)
        )
        self.torch_empty_cache_interval = float(
            os.environ.get(
                "SGLANG_EMPTY_CACHE_INTERVAL", -1
            )  # in seconds. Set if you observe high memory accumulation over a long serving period.
        )
        # Runtime constants: others
        self.retract_decode_steps = 20
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
