# temp NSA debugging environ
from sglang.srt.utils import get_bool_env_var

NSA_USE_REAL_INDEXER = get_bool_env_var("SGLANG_NSA_USE_REAL_INDEXER", "true")
NSA_DUAL_STREAM = get_bool_env_var("SGLANG_NSA_DUAL_STREAM", "true")
NSA_FUSE_TOPK = get_bool_env_var("SGLANG_NSA_FUSE_TOPK", "true")

NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8 = get_bool_env_var(
    "SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8", "true"
)
NSA_KV_CACHE_STORE_FP8 = get_bool_env_var("SGLANG_NSA_KV_CACHE_STORE_FP8", "false")
NSA_QUANT_K_CACHE_FAST = get_bool_env_var("SGLANG_NSA_QUANT_K_CACHE_FAST", "false")
NSA_DEQUANT_K_CACHE_FAST = get_bool_env_var("SGLANG_NSA_DEQUANT_K_CACHE_FAST", "false")


def _print_bool_env_vars():
    msg = ""
    for k, v in globals().items():
        if k.startswith("NSA_") and isinstance(v, bool):
            msg += f"{k}={v} "
    print(msg, flush=True)


_print_bool_env_vars()


if not NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8:
    assert not NSA_KV_CACHE_STORE_FP8


def compute_nsa_seqlens(original_seq_lens, nsa_index_topk: int):
    return original_seq_lens.clamp(max=nsa_index_topk)
